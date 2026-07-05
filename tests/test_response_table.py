#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical-parity tests for response_table.py against the structured-array logic.

These reproduce the trial windowing and the d'/FSI statistics exactly as written in
analysis_for_images.py, on synthetic data, and assert the xarray container gives identical
results. They also check the completeness invariant and the explicit-exclusion behaviour.

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_response_table.py -v
  or: ../.venv/bin/python marmanalysis/tests/test_response_table.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import f_oneway

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import response_table as rt


METRICS = ['FdFF', 'Fzsc', 'F0', 'Fraw']

# Category layout shared by the tests, mirroring the super-category logic in
# analysis_for_images.py:1915-1925 (face / non-face / non-face-object).
CATS = np.array([b'face_hum', b'face_mrm', b'obj', b'food', b'body_mrm', b'blank'])


def _category_bools(cats):
    is_face = np.array([b'face' in c and b'blank' not in c and b'scram' not in c
                        and b'ctn' not in c for c in cats])
    is_nonface = np.array([b'face' not in c and b'blank' not in c and b'scram' not in c
                           for c in cats])
    is_nonface_object = np.array([b'face' not in c and b'blank' not in c and b'scram' not in c
                                  and b'body' not in c for c in cats])
    return is_face, is_nonface, is_nonface_object


def _make_session(n_ROIs=12, n_reps=5, n_samp_isi=3, n_samp_stim=5,
                  onset0=5, spacing=20, seed=0):
    """Synthetic traces + stimlog with equal repeats and non-overlapping, in-bounds trials."""
    rng = np.random.default_rng(seed)
    n_conds = CATS.size
    n_trials = n_conds * n_reps

    cond_sequence = np.repeat(np.arange(n_conds), n_reps)
    cond_sequence = rng.permutation(cond_sequence)
    onsets = onset0 + spacing * np.arange(n_trials)
    stimlog = pd.DataFrame({'cond': cond_sequence, 'acqfr_stim_i': onsets})

    n_frames = int(onsets.max() + n_samp_stim + n_samp_isi + 5)
    traces = {m: rng.standard_normal((n_ROIs, n_frames)) + i
              for i, m in enumerate(METRICS)}
    return traces, stimlog, n_samp_isi, n_samp_stim


def _reference_table(traces, stimlog, n_samp_isi, n_samp_stim):
    """Window traces into (n_conds, n_ROIs, n_reps, n_samp_trial) exactly as the structured
    array does (analysis_for_images.py:1604-1657), normal path only."""
    n_ROIs = traces[METRICS[0]].shape[0]
    conditions = np.unique(stimlog['cond'].values)
    n_conds = conditions.size
    n_reps = int(stimlog['cond'].value_counts().iloc[0])
    n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

    ref = {m: np.full((n_conds, n_ROIs, n_reps, n_samp_trial), np.nan) for m in METRICS}
    for ci, c in enumerate(conditions):
        sub = stimlog[stimlog['cond'] == c]
        for t in range(n_reps):
            fr_start = int(sub.iloc[t]['acqfr_stim_i'] - n_samp_isi)
            fr_end = int(sub.iloc[t]['acqfr_stim_i'] + n_samp_stim + n_samp_isi)
            for m in METRICS:
                ref[m][ci, :, t, :] = traces[m][:, fr_start:fr_end]
    return ref, conditions


def _reference_stats(ref, n_samp_isi, n_samp_stim, cats):
    """resp_vect_cond, d', FSI exactly as analysis_for_images.py:2074, 2010, 2028-2035."""
    idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)
    is_face, is_nonface, is_nonface_object = _category_bools(cats)

    resp = {}
    dprime = {}
    fsi = {}
    for m in METRICS:
        resp[m] = np.nanmean(ref[m][:, :, :, idx_stim], axis=(2, 3)).T  # (n_ROIs, n_conds)

        muR_F = np.mean(resp[m][:, is_face], axis=1)
        muR_NF = np.mean(resp[m][:, is_nonface], axis=1)
        muR_O = np.mean(resp[m][:, is_nonface_object], axis=1)
        sigma_F = np.std(resp[m][:, is_face], axis=1)
        sigma_NF = np.std(resp[m][:, is_nonface], axis=1)

        dprime[m] = (muR_F - muR_NF) / np.sqrt((sigma_F ** 2 + sigma_NF ** 2) / 2)

        f = np.full(resp[m].shape[0], np.nan)
        same = np.sign(muR_F) == np.sign(muR_O)
        f[same] = (muR_F[same] - muR_O[same]) / (muR_F[same] + muR_O[same])
        f[np.sign(muR_F) > np.sign(muR_O)] = 1.0
        f[np.sign(muR_F) < np.sign(muR_O)] = -1.0
        fsi[m] = f
    return resp, dprime, fsi


def test_windowing_matches_structured_array():
    traces, stimlog, n_isi, n_stim = _make_session()
    ref, conditions = _reference_table(traces, stimlog, n_isi, n_stim)
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim,
                                 condition_coords={'cat': CATS})

    np.testing.assert_array_equal(ds['condition'].values, conditions)
    for m in METRICS:
        got = ds[m].transpose('condition', 'roi', 'repeat', 'time').values
        np.testing.assert_allclose(got, ref[m])


def test_dprime_and_fsi_match_structured_array():
    traces, stimlog, n_isi, n_stim = _make_session()
    ref, _ = _reference_table(traces, stimlog, n_isi, n_stim)
    resp_ref, dprime_ref, fsi_ref = _reference_stats(ref, n_isi, n_stim, CATS)

    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim,
                                 condition_coords={'cat': CATS})
    is_face, is_nonface, is_nonface_object = _category_bools(CATS)
    for m in METRICS:
        resp = rt.stim_window_response(ds, m)
        np.testing.assert_allclose(resp.transpose('roi', 'condition').values, resp_ref[m])
        np.testing.assert_allclose(
            rt.face_dprime(resp, is_face, is_nonface).values, dprime_ref[m])
        np.testing.assert_allclose(
            rt.face_selectivity_index(resp, is_face, is_nonface_object).values, fsi_ref[m])


def test_normalize_response_shape_and_bounds():
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, condition_coords={'cat': CATS})
    resp = rt.stim_window_response(ds, 'FdFF')
    for method in ['peak', 'l2', 'zscore', 'range']:
        norm = rt.normalize_response(resp, method=method)
        assert norm.dims == resp.dims and norm.sizes == resp.sizes
    assert float(np.abs(rt.normalize_response(resp, method='peak')).max()) <= 1.0 + 1e-9
    rng = rt.normalize_response(resp, method='range')
    assert float(rng.min()) >= -1e-9 and float(rng.max()) <= 1.0 + 1e-9
    with pytest.raises(ValueError):
        rt.normalize_response(resp, method='nonsense')


@pytest.mark.parametrize('method', ['peak', 'l2', 'zscore', 'range'])
def test_dprime_invariant_to_normalization(method):
    """face d' is affine-invariant, so any per-ROI positive-scale normalization leaves it unchanged."""
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, condition_coords={'cat': CATS})
    is_face, is_nonface, _ = _category_bools(CATS)
    for m in METRICS:
        resp = rt.stim_window_response(ds, m)
        d0 = rt.face_dprime(resp, is_face, is_nonface).values
        dn = rt.face_dprime(rt.normalize_response(resp, method=method), is_face, is_nonface).values
        np.testing.assert_allclose(dn, d0, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize('method', ['peak', 'l2'])
def test_fsi_invariant_to_multiplicative_normalization(method):
    """FSI is scale-invariant: a purely multiplicative normalization leaves it unchanged."""
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, condition_coords={'cat': CATS})
    is_face, _, is_nonface_object = _category_bools(CATS)
    for m in METRICS:
        resp = rt.stim_window_response(ds, m)
        f0 = rt.face_selectivity_index(resp, is_face, is_nonface_object).values
        fn = rt.face_selectivity_index(
            rt.normalize_response(resp, method=method), is_face, is_nonface_object).values
        np.testing.assert_allclose(fn, f0, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize('method', ['zscore', 'range'])
def test_fsi_changes_under_additive_normalization(method):
    """FSI is NOT shift-invariant: a zero-shifting normalization changes it -- which is exactly why
    FSI must come from a baseline-relative measure (FdFF, zero = F0 baseline), not Fzsc (zero =
    session mean)."""
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, condition_coords={'cat': CATS})
    is_face, _, is_nonface_object = _category_bools(CATS)
    resp = rt.stim_window_response(ds, 'FdFF')
    f0 = rt.face_selectivity_index(resp, is_face, is_nonface_object).values
    fn = rt.face_selectivity_index(
        rt.normalize_response(resp, method=method), is_face, is_nonface_object).values
    assert not np.allclose(fn, f0, equal_nan=True)


def test_first_trial_short_isi_pad_matches():
    """fr_start < 0 on the first trial: pad with the first frame (images:1622-1646)."""
    traces, stimlog, n_isi, n_stim = _make_session()
    # Force the first row of condition 0 to start before a full ISI is available.
    first0 = stimlog.index[stimlog['cond'] == 0][0]
    stimlog.loc[first0, 'acqfr_stim_i'] = 1  # fr_start = 1 - n_isi < 0
    n_samp_trial = n_isi + n_stim + n_isi

    conditions = np.unique(stimlog['cond'].values)
    ref = {m: np.full((conditions.size, traces[m].shape[0], 5, n_samp_trial), np.nan)
           for m in METRICS}
    for ci, c in enumerate(conditions):
        sub = stimlog[stimlog['cond'] == c]
        for t in range(5):
            fr_start = int(sub.iloc[t]['acqfr_stim_i'] - n_isi)
            fr_end = int(sub.iloc[t]['acqfr_stim_i'] + n_stim + n_isi)
            if fr_start < 0 and t == 0:
                miss = abs(fr_start)
                for m in METRICS:
                    ref[m][ci, :, t, 0:miss] = np.repeat(traces[m][:, 0:1], miss, axis=1)
                    ref[m][ci, :, t, miss:n_samp_trial] = traces[m][:, 0:fr_end]
                continue
            for m in METRICS:
                ref[m][ci, :, t, :] = traces[m][:, fr_start:fr_end]

    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim)
    for m in METRICS:
        got = ds[m].transpose('condition', 'roi', 'repeat', 'time').values
        np.testing.assert_allclose(got, ref[m])


def test_completeness_invariant_catches_unfilled_cell():
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim)
    # A freshly built table is complete...
    rt.assert_complete(ds)
    # ...but a stray NaN (an unfilled cell) must be caught.
    ds['FdFF'][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        rt.assert_complete(ds)


def test_ragged_repeats_rejected():
    traces, stimlog, n_isi, n_stim = _make_session()
    stimlog = stimlog.drop(stimlog.index[stimlog['cond'] == 0][0])  # one cond now short a rep
    with pytest.raises(ValueError):
        rt.build_response_table(traces, stimlog, n_isi, n_stim)


def test_exclusion_is_a_mask_not_a_nan():
    """Excluded trials are recorded in the mask and dropped from reductions, while the
    underlying values are preserved (no silent NaN overwrite)."""
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim)

    ds_excl = rt.exclude_trials(ds, [(0, 0), (2, 3)])
    # Underlying data is untouched; only the mask changed.
    assert not bool(ds_excl['FdFF'].isnull().any())
    assert bool(ds_excl['excluded'].sel(condition=0, repeat=0))
    assert bool(ds_excl['excluded'].sel(condition=2, repeat=3))

    # The stim-window response drops the excluded repeat: condition 0 reduces over repeats 1..4.
    idx_stim = range(n_isi, n_isi + n_stim)
    kept = ds['FdFF'].sel(condition=0).isel(repeat=[1, 2, 3, 4], time=list(idx_stim))
    manual = kept.mean(dim=('repeat', 'time')).values
    got = rt.stim_window_response(ds_excl, 'FdFF').sel(condition=0).values
    np.testing.assert_allclose(got, manual)


def _ds_from_trial_values(vals, metric='Fzsc'):
    """Minimal response table with a single stim frame whose value is the desired per-trial response.

    ``vals`` has shape (roi, condition, repeat); the returned Dataset has ``trial_response`` equal to
    ``vals`` exactly, so trial-level statistics can be tested without the windowing machinery."""
    n_roi, n_cond, n_rep = vals.shape
    return xr.Dataset(
        {metric: (('roi', 'condition', 'repeat', 'time'), vals[..., None].astype(float))},
        coords={'roi': np.arange(n_roi), 'condition': np.arange(n_cond),
                'repeat': np.arange(n_rep), 'time': [0],
                'epoch': ('time', np.array([rt.EPOCH_STIM])),
                'excluded': (('condition', 'repeat'), np.zeros((n_cond, n_rep), bool))})


def test_trial_response_keeps_repeat_axis():
    traces, stimlog, n_isi, n_stim = _make_session()
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, condition_coords={'cat': CATS})
    tr = rt.trial_response(ds, 'FdFF')
    assert set(tr.dims) == {'roi', 'condition', 'repeat'}
    manual = ds['FdFF'].isel(time=list(range(n_isi, n_isi + n_stim))).mean('time')
    np.testing.assert_allclose(tr.transpose('roi', 'condition', 'repeat').values,
                               manual.transpose('roi', 'condition', 'repeat').values)


def test_trial_scalar_anova_detects_structure_and_parity():
    rng = np.random.default_rng(0)
    n_cond, n_rep = 6, 8
    roi0 = (np.arange(n_cond) * 3.0)[:, None] + rng.normal(scale=0.3, size=(n_cond, n_rep))
    roi1 = rng.normal(scale=1.0, size=(n_cond, n_rep))
    ds = _ds_from_trial_values(np.stack([roi0, roi1]))
    p = rt.trial_scalar_anova(ds, 'Fzsc')
    assert p[0] < 1e-6 and p[1] > 0.05
    tr = rt.trial_response(ds, 'Fzsc').transpose('roi', 'condition', 'repeat').values
    for r in range(2):
        np.testing.assert_allclose(p[r], f_oneway(*[tr[r, c] for c in range(n_cond)]).pvalue)


def test_split_half_reliability_reliable_vs_noise():
    rng = np.random.default_rng(0)
    n_cond, n_rep = 100, 10  # many conditions so the pure-noise reliability estimate collapses to ~0
    sig = rng.normal(size=n_cond)
    roi0 = np.repeat(sig[:, None], n_rep, axis=1)          # identical across repeats -> reliable
    roi1 = rng.normal(size=(n_cond, n_rep))                # pure noise -> unreliable
    ds = _ds_from_trial_values(np.stack([roi0, roi1]))
    rel = rt.split_half_reliability(ds, 'Fzsc', n_splits=50, seed=1)
    assert rel[0] > 0.95 and rel[1] < 0.3 and (rel[0] - rel[1]) > 0.6


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
