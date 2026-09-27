#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for activity_geometry: the exact Currier & Clandinin D_phys recipe and the row-z-scored distance."""
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import activity_geometry as ag


def test_dphys_lossless_when_rank_below_n_pc():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    D = ag.dphys_currier_clandinin(X, n_pc=100)
    Xc = X - X.mean(0)
    assert np.allclose(D, squareform(pdist(Xc)), atol=1e-9)


def test_dphys_normalize_max():
    rng = np.random.default_rng(1)
    D = ag.dphys_currier_clandinin(rng.standard_normal((30, 10)), normalize_max=True)
    assert np.isclose(D.max(), 1.0) and D.min() >= 0


def test_zscored_distance_is_correlation_distance():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((25, 60)) * rng.uniform(0.1, 10, (25, 1))   # per-row gain
    D = ag.dact_zscored(X)
    k = X.shape[1]
    r = np.corrcoef(X)
    expected = np.sqrt(np.clip(2 * k * (1 - r), 0, None))      # population std: ||z||^2 = k
    np.fill_diagonal(expected, 0)
    assert np.allclose(D, expected, atol=1e-6)


def test_gain_invariance_only_for_zscored():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((10, 30))
    X = np.vstack([base, 5.0 * base])          # same tuning shapes, 5x gain
    Dz = ag.dact_zscored(X)
    assert np.allclose(Dz[:10, 10:].diagonal(), 0, atol=1e-6)        # identical shape -> zero distance
    Dc = ag.dphys_currier_clandinin(X)
    assert Dc[:10, 10:].diagonal().min() > 1.0                        # raw recipe sees the gain difference


def test_compare_to_physical_sign():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 500, (80, 2))
    dist = squareform(pdist(xy))
    out = ag.compare_to_physical(dist, xy, min_dist=0, max_dist=1e9)   # activity distance == physical distance
    assert out['rho'] > 0.99 and out['n_pairs'] == 80 * 79 // 2
