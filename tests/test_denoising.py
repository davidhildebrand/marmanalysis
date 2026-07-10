#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for denoising.denoise_psn on a synthetic response table (no real session / no GPU)."""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import response_table as rt
import denoising as dn


def _low_rank_signal(nu=20, nc=30, rank=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(nu, rank)) @ rng.normal(size=(rank, nc))


def _ds(signal, noise_sd=0.5, n_isi=2, n_stim=3, n_rep=6, seed=0):
    """Response table whose stim-window per-trial response = signal + iid noise (ISI frames = 0 baseline)."""
    rng = np.random.default_rng(seed)
    nu, nc = signal.shape
    trials = signal[:, :, None] + rng.normal(0, noise_sd, (nu, nc, n_rep))     # (units, cond, rep)
    T = 2 * n_isi + n_stim
    arr = np.zeros((nu, nc, n_rep, T))
    arr[..., n_isi:n_isi + n_stim] = trials[..., None]                          # stim frames = trial value
    ep = np.array([rt.EPOCH_ISI_PRE] * n_isi + [rt.EPOCH_STIM] * n_stim + [rt.EPOCH_ISI_POST] * n_isi)
    ds = xr.Dataset(
        {'Fzsc': (('roi', 'condition', 'repeat', 'time'), arr)},
        coords={'roi': np.arange(nu), 'condition': np.arange(nc), 'repeat': np.arange(n_rep),
                'time': np.arange(T), 'epoch': ('time', ep),
                'excluded': (('condition', 'repeat'), np.zeros((nc, n_rep), bool))})
    return ds, trials


def test_denoise_psn_returns_matching_ds_and_keeps_isi():
    ds, _ = _ds(_low_rank_signal())
    dsd, info = dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=False)
    assert dsd['Fzsc'].dims == ds['Fzsc'].dims and dsd['Fzsc'].sizes == ds['Fzsc'].sizes
    isi = ds['epoch'].values != rt.EPOCH_STIM
    np.testing.assert_allclose(dsd['Fzsc'].values[..., isi], ds['Fzsc'].values[..., isi])   # ISI untouched
    assert not np.allclose(dsd['Fzsc'].values[..., ~isi], ds['Fzsc'].values[..., ~isi])     # stim changed
    assert 0 < int(np.ravel(info['n_signal_dims'])[0]) <= ds.sizes['roi']


def test_denoise_psn_trial_average_matches_psn_denoiseddata():
    """Per-trial denoiser applied then averaged == PSN's trial-averaged denoiseddata (linearity)."""
    ds, _ = _ds(_low_rank_signal())
    dsd, info = dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=False)
    got = rt.trial_response(dsd, 'Fzsc').transpose('roi', 'condition', 'repeat').mean('repeat').values
    np.testing.assert_allclose(got, info['psn']['denoiseddata'], rtol=1e-4, atol=1e-6)


def test_denoise_psn_reduces_trial_noise():
    """Low-rank signal + heavy noise: denoised within-condition (across-trial) variance < raw."""
    ds, _ = _ds(_low_rank_signal(), noise_sd=1.0)
    dsd, _ = dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=False)
    raw = rt.trial_response(ds, 'Fzsc').transpose('roi', 'condition', 'repeat').values
    den = rt.trial_response(dsd, 'Fzsc').transpose('roi', 'condition', 'repeat').values
    assert np.nanmean(den.var(axis=2)) < np.nanmean(raw.var(axis=2))


def test_denoise_psn_xval_preserves_error_term():
    """The whole point of CV: within-condition scatter (the ANOVA error term) is NOT self-collapsed.

    in-sample denoising drives within-condition variance toward zero (that is the artifact); cross-validated
    denoising reduces it but keeps it well above the in-sample floor and below raw."""
    ds, _ = _ds(_low_rank_signal(), noise_sd=1.0)
    def within_var(d):
        v = rt.trial_response(d, 'Fzsc').transpose('roi', 'condition', 'repeat').values
        return float(np.nanmean(v.var(axis=2)))
    v_raw = within_var(ds)
    v_ins = within_var(dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=False)[0])
    dscv, info = dn.denoise_psn_xval(ds, 'Fzsc', 'conservative', n_folds=3)
    v_cv = within_var(dscv)
    assert v_ins < v_cv < v_raw
    assert np.isfinite(rt.trial_response(dscv, 'Fzsc').values).all()   # every trial denoised out-of-fold
    assert len(info['fold_signal_dims']) == 3


def test_denoise_psn_diagnostic_flag(tmp_path):
    ds, _ = _ds(_low_rank_signal())
    _, off = dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=False, outdir=str(tmp_path))
    assert off['figure_path'] is None and not any(p.suffix == '.png' for p in tmp_path.iterdir())
    _, on = dn.denoise_psn(ds, 'Fzsc', 'conservative', diagnostic=True, outdir=str(tmp_path))
    assert on['figure_path'] and os.path.exists(on['figure_path'])


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
