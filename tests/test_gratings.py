#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the gratings driver (gratings.py): synthetic checks that orientation_selectivity recovers
known orientation- and direction-tuned ROIs with the recommended (Mazurek, Kager and Van Hooser 2014)
measures, and an end-to-end run of process_session on a real multimodal-gratings session.

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_gratings.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gratings
import paths
import response_table as rt

BASE = paths.suite2p()
GRATINGS_SESSION = (BASE + '/Louwho/20240616d/172850tUTC_SP_depth250um_fov0636x1500um_'
                    'res2p00x2p00umpx_fr10p334Hz_pow028p4mW_stimMultimodalGratings')


def test_orientation_selectivity_recovers_tuned_roi():
    n_dir, n_reps, n_isi, n_stim, n_roi = 16, 4, 2, 2, 4
    directions = np.arange(0.0, 360.0, 22.5)
    # per-(roi, direction) target response: ROI 0 orientation-tuned at 45 deg, the rest flat.
    target = np.ones((n_roi, n_dir))
    target[0] = np.cos(np.radians(directions - 45.0)) ** 2

    cond_seq = np.tile(np.arange(n_dir), n_reps)
    onsets = 5 + 6 * np.arange(len(cond_seq))
    stimlog = pd.DataFrame({
        'cond': cond_seq, 'acqfr_stim_i': onsets,
        'stim_class': 'grating', 'stim_subclass': 'drifting',
        'grating_ori': directions[cond_seq], 'grating_sf': 1.0, 'grating_tf': 4.0,
    })
    n_frames = int(onsets.max() + n_stim + n_isi + 3)
    traces = {'Fzsc': np.zeros((n_roi, n_frames))}
    for c, o in zip(cond_seq, onsets):
        traces['Fzsc'][:, o:o + n_stim] = target[:, c][:, None]

    gr = gratings.select_drifting_gratings(stimlog)
    assert len(gr) == n_dir * n_reps
    ds = rt.build_response_table(traces, gr, n_isi, n_stim)
    ds = gratings.attach_condition_metadata(ds, gratings.build_condition_metadata(gr))

    tun = gratings.orientation_selectivity(ds, 'Fzsc', compute_dsi=True)
    assert tun['osi'][0] > tun['osi'][1:].max()              # tuned ROI is the most selective
    assert abs(tun['ori_pref'][0] - 45.0) < 5.0              # preferred orientation ~45 deg
    assert np.all((tun['osi'] >= 0) & (tun['osi'] <= 1.0001))
    assert abs(tun['osi'][0] - 0.5) < 1e-6                   # 1 - CirVar of cos^2 tuning is exactly 0.5
    assert tun['dsi'][0] < 1e-6                              # oriented, not directional
    assert {'ori_p', 'dir_p', 'oi_peak', 'di_peak', 'osi_peak', 'dsi_peak'} <= set(tun)

    legacy = gratings.orientation_selectivity(ds, 'Fzsc', compute_dsi=False, method='legacy')
    assert set(legacy) == {'osi', 'ori_pref', 'dsi', 'dir_pref'}
    assert abs(legacy['ori_pref'][0] - 45.0) < 5.0 and np.all(np.isnan(legacy['dsi']))


def _direction_tuned_table(noise=0.3, n_reps=8, seed=0):
    """ROI 0 direction tuned (pref 90, null 270), ROI 1 oriented only, ROI 2 untuned, ROI 3 suppressed below baseline.
    Every ROI sits on a baseline of 5, which the default baseline subtraction has to remove."""
    import tuning
    rng = np.random.default_rng(seed)
    n_dir, n_isi, n_stim, n_roi = 16, 4, 3, 4
    directions = np.arange(0.0, 360.0, 22.5)
    target = np.zeros((n_roi, n_dir))
    target[0] = tuning.gaussian_tuning(directions, 0.2, 4.0, 90.0, 25.0, 1.0)
    target[1] = tuning.gaussian_tuning(directions, 0.2, 3.0, 90.0, 25.0, 3.0)
    target[2] = 1.0
    target[3] = -1.0
    cond_seq = np.concatenate([rng.permutation(n_dir) for _ in range(n_reps)])
    onsets = 6 + (n_stim + 2 * n_isi) * np.arange(len(cond_seq))
    stimlog = pd.DataFrame({'cond': cond_seq, 'acqfr_stim_i': onsets, 'stim_class': 'grating',
                            'stim_subclass': 'drifting', 'grating_ori': directions[cond_seq],
                            'grating_sf': 1.0, 'grating_tf': 4.0})
    n_frames = int(onsets.max() + n_stim + n_isi + 3)
    traces = {'Fzsc': 5.0 + noise * rng.standard_normal((n_roi, n_frames))}
    for c, o in zip(cond_seq, onsets):
        traces['Fzsc'][:, o:o + n_stim] += target[:, c][:, None]
    ds = rt.build_response_table(traces, stimlog, n_isi, n_stim, framerate=6.0)
    return gratings.attach_condition_metadata(ds, gratings.build_condition_metadata(stimlog))


def test_recommended_measures_on_a_noisy_baseline():
    ds = _direction_tuned_table()
    tun = gratings.orientation_selectivity(ds, fit=True, n_boot=20)
    assert tun['ori_p'][0] < 0.01 and tun['ori_p'][1] < 0.01          # both tuned ROIs pass Hotelling's test
    assert tun['dir_p'][0] < 0.01 and tun['dir_p'][1] > 0.01          # only ROI 0 is directional
    assert tun['dsi'][0] > 3 * tun['dsi'][1] and tun['osi'][1] > tun['osi'][2]
    assert abs((tun['dir_pref'][0] - 90 + 180) % 360 - 180) < 10
    assert abs((tun['dir_pref_axis'][0] - 90 + 180) % 360 - 180) < 10
    assert np.isnan(tun['osi'][3]) and np.isnan(tun['dsi'][3])        # nothing above baseline: undefined, not 0 or 1
    assert tun['fit_reportable'][0] and abs((tun['fit_theta'][0] - 90 + 180) % 360 - 180) < 6
    assert abs(tun['fit_sigma'][0] - 25.0) < 8 and tun['fit_theta_ci'][0] < 15
    assert tun['fit_sigma_lo'][0] <= tun['fit_sigma'][0] <= tun['fit_sigma_hi'][0]
    # without baseline subtraction the offset of 5 swamps the tuning: selectivity collapses
    raw = gratings.orientation_selectivity(ds, baseline=None)
    assert raw['osi'][1] < 0.5 * tun['osi'][1]


def test_excluded_presentations_do_not_break_the_measures():
    ds = _direction_tuned_table()
    full = gratings.orientation_selectivity(ds)
    ds = rt.exclude_trials(ds, [(3, 0), (9, 5)])
    part = gratings.orientation_selectivity(ds)
    assert np.isfinite(part['osi'][:3]).all() and np.isfinite(part['ori_p'][:3]).all()
    assert np.abs(part['osi'][:3] - full['osi'][:3]).max() < 0.1


def test_bayesian_tuning_agrees_with_the_recommended_tests():
    ds = _direction_tuned_table()
    bay = gratings.bayesian_tuning(ds, n_burn=2000, n_samples=4000, thin=20)
    assert bay['log_bf_tuned'][0] > 5 and bay['log_bf_tuned'][1] > 5      # both tuned ROIs
    assert bay['log_bf_tuned'][2] < 3 and bay['log_bf_tuned_bic'][2] < 0  # the untuned ROI
    assert bay['log_bf_direction'][0] > 5 and bay['log_bf_direction'][1] < 3
    assert abs((bay['dir_pref'][0] - 90 + 180) % 360 - 180) < 8
    assert abs((bay['ori_pref'][1] - 90 + 90) % 180 - 90) < 8
    assert bay['ori_sigma_lo'][1] <= 25.0 <= bay['ori_sigma_hi'][1]       # the true width is inside the interval
    some = gratings.bayesian_tuning(ds, roi_mask=[True, False, False, False], models=('constant', 'circ180'),
                                    n_burn=500, n_samples=1000, thin=10)
    assert np.isfinite(some['log_bf_tuned'][0]) and np.all(np.isnan(some['log_bf_tuned'][1:]))
    assert 'log_bf_direction' not in some and some['results']['circ180']['samples'].shape[0] == 1


def test_direction_comes_from_grating_dir_when_the_log_has_it():
    log = pd.DataFrame({'cond': [0, 1, 2, 3], 'grating_ori': [0.0, 90.0, 0.0, 90.0],
                        'grating_dir': [0.0, 90.0, 180.0, 270.0], 'grating_sf': 1.0, 'grating_tf': 4.0})
    meta = gratings.build_condition_metadata(log)
    assert meta['direction'].tolist() == [0.0, 90.0, 180.0, 270.0]
    assert meta['orientation'].tolist() == [0.0, 90.0, 0.0, 90.0]
    old = gratings.build_condition_metadata(log.drop(columns='grating_dir').assign(grating_ori=[0.0, 90.0, 180.0, 270.0]))
    assert old['direction'].tolist() == [0.0, 90.0, 180.0, 270.0]        # older logs: the one angle column


def test_process_session_real_multimodal_gratings():
    if not os.path.isdir(GRATINGS_SESSION):
        pytest.skip('session data not present')
    res = gratings.process_session(GRATINGS_SESSION, compute_dsi=False)   # osi only for speed
    ds, tun = res['dataset'], res['tuning']
    assert {'direction', 'orientation', 'sf', 'tf'}.issubset(ds.coords)
    assert ds.sizes['condition'] == 16                       # 16 drift directions selected
    assert np.allclose(np.sort(ds['direction'].values), np.arange(0.0, 360.0, 22.5))   # directions, not dir % 180
    n_roi = ds.sizes['roi']
    assert tun['osi'].shape == (n_roi,)
    ok = np.isfinite(tun['osi'])                             # NaN = no response above baseline at any direction
    assert ok.mean() > 0.8
    assert np.all((tun['osi'][ok] >= 0) & (tun['osi'][ok] <= 1.0001))
    assert np.all((tun['ori_pref'][ok] >= 0) & (tun['ori_pref'][ok] < 180))
    assert np.nanmax(tun['osi']) > 0.2                       # at least some orientation-selective cells
    assert np.all(np.isnan(tun['dsi']))                      # compute_dsi=False skips the direction measures
    p = tun['ori_p'][np.isfinite(tun['ori_p'])]
    assert p.size > 0.8 * n_roi and np.mean(p < 0.05) > 0.05 # more ROIs pass Hotelling's test than chance allows


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
