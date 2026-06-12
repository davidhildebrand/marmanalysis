#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Value tests for filters.calculate_baselines -- the F0 baseline used by
sessionio.compute_fluorescence_metrics. These check the SEMANTICS of the baseline (it tracks the
slow/low signal and ignores sparse positive events) rather than re-deriving the filter math.

Note: calculate_baselines squeezes single-ROI output to 1D, so the multi-ROI tests use
np.atleast_2d to stay shape-robust (in real use it always sees many ROIs).

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_filters.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filters


def test_constant_signal_returns_that_constant():
    """A flat trace -> F0 equals that constant everywhere (median is constant, low-pass passes DC)."""
    frois = np.full((3, 600), 500.0)
    f0 = filters.calculate_baselines(frois, framerate=6.0, window=60, method='medianbw')
    np.testing.assert_allclose(f0, 500.0, atol=1e-4)


@pytest.mark.parametrize('method', ['median', 'medianbw'])
def test_baseline_ignores_sparse_transients(method):
    """baseline + sparse positive transients -> F0 stays near the baseline (so dF/F is large at the
    transients), which is the whole point of F0."""
    n_frames, baseline = 1200, 500.0
    frois = np.full((1, n_frames), baseline)
    spikes = np.array([100, 300, 700, 1000])
    frois[0, spikes] += 400.0
    f0 = np.atleast_2d(filters.calculate_baselines(frois, framerate=6.0, window=60, method=method))
    assert np.max(np.abs(f0[0] - baseline)) < 50.0              # F0 hugs the baseline
    assert np.all(frois[0, spikes] - f0[0, spikes] > 300.0)     # transients tower over F0


def test_baseline_tracks_slow_drift():
    """A slow linear drift is low-frequency -> F0 follows it closely (median of a monotonic window
    is its centre value)."""
    n_frames = 1500
    drift = np.linspace(400.0, 600.0, n_frames)
    f0 = np.atleast_2d(filters.calculate_baselines(drift[None, :], framerate=6.0, window=60,
                                                   method='median'))
    mid = slice(400, n_frames - 400)                            # away from reflect-padded edges
    np.testing.assert_allclose(f0[0, mid], drift[mid], atol=5.0)


def test_single_roi_is_squeezed_to_1d():
    """A single ROI (1D input, or a 1-row 2D input) returns a 1D baseline."""
    f0 = filters.calculate_baselines(np.full(600, 500.0), framerate=6.0, window=60, method='median')
    assert f0.ndim == 1 and f0.shape == (600,)
    np.testing.assert_allclose(f0, 500.0, atol=1e-6)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
