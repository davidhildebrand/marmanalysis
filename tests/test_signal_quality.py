#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for signal_quality: standardized noise (Rupprecht nu), sigma estimators, and the active gate."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal_quality as sq


def test_standardized_noise_matches_rupprecht_formula_and_percent_scale():
    rng = np.random.default_rng(0)
    dff = rng.normal(0, 0.02, (3, 5000))
    fr = 6.0
    nu = sq.standardized_noise(dff, fr, percent=True)
    expect = np.median(np.abs(np.diff(dff * 100, axis=1)), axis=1) / np.sqrt(fr)
    np.testing.assert_allclose(nu, expect)
    np.testing.assert_allclose(nu, sq.standardized_noise(dff, fr, percent=False) * 100)  # x100 units fix


def test_diff_mad_std_recover_gaussian_sigma():
    rng = np.random.default_rng(1)
    sigma_true = 0.03
    dff = rng.normal(0, sigma_true, (4, 20000))
    for method in ('diff', 'mad', 'std'):
        np.testing.assert_allclose(sq.noise_sigma(dff, method=method), sigma_true, rtol=0.06)


def test_active_mask_separates_dynamic_from_noise():
    rng = np.random.default_rng(2)
    n = 4000
    dff = rng.normal(0, 0.02, (2, n))
    dff[0, rng.choice(n, 120, replace=False)] += 0.6        # ROI0: real transients (3% of frames)
    mask, snr, peak, sigma = sq.active_mask(dff, framerate=6.0, method='diff', k=5.0)
    assert mask[0] and not mask[1]
    assert snr[0] > snr[1]


def test_mad_isi_excludes_stimulus_and_above_baseline():
    rng = np.random.default_rng(3)
    n = 2000
    dff = rng.normal(0, 0.02, (1, n))
    isi = np.ones(n, bool)
    isi[500:1500] = False                                   # middle = stimulus frames
    dff[0, 700] = 5.0                                       # huge positive transient inside the stim block
    s_isi = sq.noise_sigma(dff, method='mad_isi', isi_mask=isi)
    assert np.isfinite(s_isi[0]) and s_isi[0] < 0.05        # stim-block positive spike must not inflate it


def test_active_mask_warns_at_slow_framerate():
    import warnings
    dff = np.random.default_rng(0).normal(0, 0.02, (2, 500))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        sq.active_mask(dff, framerate=2.5, method='nu')     # slow -> should warn
        sq.active_mask(dff, framerate=6.4, method='nu')     # healthy -> should not
    msgs = [str(w.message) for w in rec]
    assert any('undersampled' in m and '2.50 Hz' in m for m in msgs)
    assert not any('6.40 Hz' in m for m in msgs)


def test_n_transients_detects_events_not_noise():
    rng = np.random.default_rng(0)
    n = 4000
    dff = rng.normal(0, 0.02, (2, n))
    for t in rng.choice(np.arange(0, n - 6, 12), 30, replace=False):
        dff[0, t:t + 5] += 0.5                              # ROI0: 30 clear 5-frame transients
    counts = sq.n_transients(dff, framerate=6.0, onset=3.0, offset=1.0, min_duration_sec=0.3)
    assert counts[0] >= 25                                  # detects most injected transients
    assert counts[1] <= 5                                   # noise ROI: essentially none
