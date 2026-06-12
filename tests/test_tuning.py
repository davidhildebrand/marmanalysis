#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for tuning.calculate_osi (orientation-selectivity index) on known inputs.

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_tuning.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tuning

ORIS8 = np.array([0., 22.5, 45., 67.5, 90., 112.5, 135., 157.5])     # 8 orientations
DIRS16 = np.arange(0.0, 360.0, 22.5)                                  # 16 drift directions


def test_osi_delta_response_is_one():
    R = np.zeros(8); R[2] = 1.0                                       # peak at 45 deg
    osi, pref = tuning.calculate_osi(ORIS8, R)
    assert osi > 0.99
    assert abs(pref - 45.0) < 1.0


def test_osi_uniform_response_is_zero():
    osi, _ = tuning.calculate_osi(ORIS8, np.ones(8))
    assert osi < 1e-9


def test_osi_preferred_orientation_recovers_90():
    R = np.zeros(8); R[4] = 1.0                                       # peak at 90 deg
    _, pref = tuning.calculate_osi(ORIS8, R)
    assert abs(pref - 90.0) < 1.0


def test_osi_treats_opposite_directions_as_same_orientation():
    # equal response at 0 and 180 deg (same orientation) -> high OSI, preferred orientation ~0
    R = np.zeros(16); R[0] = 1.0; R[8] = 1.0
    osi, pref = tuning.calculate_osi(DIRS16, R)
    assert osi > 0.99
    assert min(pref % 180, 180 - (pref % 180)) < 1.0


def test_osi_in_range_and_orientation_bounds():
    rng = np.random.default_rng(0)
    osi, pref = tuning.calculate_osi(ORIS8, np.abs(rng.standard_normal(8)))
    assert 0.0 <= osi <= 1.0
    assert 0.0 <= pref < 180.0


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
