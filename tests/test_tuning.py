#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for tuning: the legacy calculate_osi, and the Mazurek, Kager and Van Hooser 2014 measures
(vector indices, peak indices, Hotelling's T^2, direction dot product test, constrained Gaussian fits, bootstrap).

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


# ---- Mazurek, Kager and Van Hooser 2014 ----------------------------------------------------------------------------
def _direction_cell(rng, n_trials=8, Rp=4.0, Rn=1.0, theta=90.0, sigma=25.0, C=0.5, noise=0.4):
    curve = tuning.gaussian_tuning(DIRS16, C, Rp, theta, sigma, Rn)
    return curve, curve[None, :] + noise * rng.standard_normal((n_trials, DIRS16.size))


def test_vector_indices_on_known_curves():
    one = np.zeros(16); one[4] = 1.0                                   # a single direction, 90 deg
    assert tuning.vector_selectivity(DIRS16, one, 'direction')[0] > 0.999
    assert abs(tuning.vector_selectivity(DIRS16, one, 'direction')[1] - 90.0) < 1e-6
    both = np.zeros(16); both[4] = 1.0; both[12] = 1.0                 # 90 and 270: oriented, not directional
    assert tuning.vector_selectivity(DIRS16, both, 'orientation')[0] > 0.999
    assert tuning.vector_selectivity(DIRS16, both, 'direction')[0] < 1e-9
    assert tuning.vector_selectivity(DIRS16, np.ones(16), 'orientation')[0] < 1e-9
    cos2 = np.cos(np.radians(DIRS16 - 45.0)) ** 2                      # 1 - CirVar of cos^2 is exactly 0.5
    index, pref = tuning.vector_selectivity(DIRS16, cos2, 'orientation')
    assert abs(index - 0.5) < 1e-9 and abs(pref - 45.0) < 1e-6


def test_vector_index_handling_of_negative_responses():
    r = np.array([2.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0])         # suppressed below background at 4 angles
    rect = tuning.vector_selectivity(ORIS8, r, 'orientation', 'rectify')[0]
    shift = tuning.vector_selectivity(ORIS8, r, 'orientation', 'shift')[0]
    keep = tuning.vector_selectivity(ORIS8, r, 'orientation', 'keep')[0]
    assert abs(rect - 1.0) < 1e-9                                      # only one angle is above background
    assert 0.0 < shift < rect                                          # the shift counts suppression as response
    assert keep < 0 or keep > 1 or not np.isfinite(keep)               # unbounded: the sum of responses is <= 0
    assert np.isnan(tuning.vector_selectivity(ORIS8, -np.ones(8))[0])  # nothing above background: undefined


def test_peak_indices_match_their_definitions():
    r = np.full(16, 1.0); r[4] = 5.0; r[12] = 3.0                      # pref 90 (5), null 270 (3), orth 1
    idx = tuning.peak_selectivity(DIRS16, r, 'direction')
    assert idx['theta_pref'] == 90.0
    assert abs(idx['DI'] - (5 - 3) / 5) < 1e-12
    assert abs(idx['DSI'] - (5 - 3) / (5 + 3)) < 1e-12
    assert abs(idx['OI'] - (5 + 3 - 1 - 1) / (5 + 3)) < 1e-12
    assert abs(idx['OSI'] - (4 - 1) / (4 + 1)) < 1e-12
    assert abs(idx['O/P'] - (1 - idx['OI'])) < 1e-12
    ori = tuning.peak_selectivity(ORIS8, np.array([1., 1, 1, 1, 4, 1, 1, 1]), 'orientation')
    assert abs(ori['OI'] - (4 - 1) / 4) < 1e-12 and 'DI' not in ori


def test_hotelling_false_positive_rate_and_power():
    rng = np.random.default_rng(1)
    null = [tuning.orientation_test(DIRS16, 1.0 + rng.standard_normal((8, 16)))['p'] for _ in range(400)]
    assert 0.02 < np.mean(np.array(null) < 0.05) < 0.09                # untuned cells: about 5% pass
    tuned = [tuning.orientation_test(DIRS16, _direction_cell(rng)[1])['p'] for _ in range(50)]
    assert np.mean(np.array(tuned) < 0.05) > 0.95
    assert np.isnan(tuning.orientation_test(DIRS16, np.ones(16))['p']) # one trial: no test


def test_direction_dot_product_test():
    rng = np.random.default_rng(2)
    directional = [tuning.direction_dot_product_test(DIRS16, _direction_cell(rng)[1]) for _ in range(50)]
    assert np.mean([d['p'] < 0.05 for d in directional]) > 0.9
    assert np.median([abs((d['preferred_direction'] - 90 + 180) % 360 - 180) for d in directional]) < 15
    # oriented but not directional (equal peaks): the test stays near its nominal rate
    oriented = [tuning.direction_dot_product_test(DIRS16, _direction_cell(rng, Rp=3.0, Rn=3.0)[1])['p'] for _ in range(300)]
    assert np.mean(np.array(oriented) < 0.05) < 0.1


def test_two_sample_tests():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((40, 2)); b = rng.standard_normal((40, 2)) + [1.5, 0.0]
    assert tuning.hotelling_t2_two_sample(a, b)['p'] < 1e-4
    assert tuning.hotelling_t2_two_sample(a, rng.standard_normal((40, 2)))['p'] > 0.01
    assert tuning.hotelling_t2_two_sample(a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1])['p'] < 1e-4   # complex in
    assert tuning.compare_selectivity(rng.uniform(0.0, 0.3, 50), rng.uniform(0.4, 0.8, 50))['p'] < 1e-6


def test_double_gaussian_fit_recovers_parameters_and_respects_constraints():
    rng = np.random.default_rng(4)
    curve, trials = _direction_cell(rng, n_trials=12, noise=0.2)
    fit = tuning.fit_gaussian(DIRS16, trials, 'direction')
    assert abs((fit['theta_pref'] - 90 + 180) % 360 - 180) < 5
    assert abs(fit['sigma'] - 25.0) < 5 and abs(fit['Rp'] - 4.0) < 0.6 and abs(fit['Rn'] - 1.0) < 0.6
    assert abs(fit['hwhh'] - fit['sigma'] * np.sqrt(2 * np.log(2))) < 1e-12 and fit['r2'] > 0.95
    assert fit['Rp'] >= fit['Rn']                                      # the larger peak is the preferred direction
    # pure noise: the constraints keep the parameters physical
    M = None
    for _ in range(20):
        noise = rng.standard_normal(16)
        f = tuning.fit_gaussian(DIRS16, noise, 'direction')
        M = max(noise.max(), 1e-12)
        assert f['sigma'] >= 22.5 / 2 - 1e-6 and abs(f['C']) <= M + 1e-6
        assert -1e-9 <= f['Rn'] <= f['Rp'] <= 3 * M + 1e-6


def test_single_gaussian_fit_in_orientation_space():
    oris = np.arange(0.0, 180.0, 22.5)
    curve = tuning.gaussian_tuning(oris, 0.2, 3.0, 170.0, 20.0)        # a peak next to the 0 / 180 wrap
    fit = tuning.fit_gaussian(oris, curve, 'orientation')
    assert abs((fit['theta_pref'] - 170 + 90) % 180 - 90) < 2 and abs(fit['sigma'] - 20) < 2
    assert np.isnan(fit['Rn']) and np.isnan(fit['DI'])


def test_bootstrap_gives_tighter_intervals_for_a_better_measured_cell():
    rng = np.random.default_rng(5)
    quiet = tuning.bootstrap_fit(DIRS16, _direction_cell(rng, n_trials=10, noise=0.2)[1], n_boot=40, seed=0)
    noisy = tuning.bootstrap_fit(DIRS16, _direction_cell(rng, n_trials=10, noise=2.0)[1], n_boot=40, seed=0)
    assert quiet['theta_ci'] < noisy['theta_ci']
    assert abs((quiet['theta_pref'] - 90 + 180) % 360 - 180) < 6
    lo, hi = quiet['ci']['sigma']
    assert lo <= 25.0 + 6 and hi >= 25.0 - 6 and quiet['samples']['Rp'].shape == (40,)


def test_selectivity_bundles_the_recommended_measures():
    rng = np.random.default_rng(6)
    out = tuning.selectivity(DIRS16, _direction_cell(rng)[1], fit=True)
    assert {'ori_index', 'ori_pref', 'dir_index', 'dir_pref', 'ori_p', 'dir_p', 'OI', 'DI', 'OSI', 'DSI', 'fit'} <= set(out)
    assert out['ori_p'] < 0.05 and out['dir_p'] < 0.05 and out['fit_reportable']
    assert 0 < out['dir_index'] < 1 and 0 < out['ori_index'] < 1
    flat = tuning.selectivity(DIRS16, 1.0 + 0.3 * rng.standard_normal((8, 16)), fit=True)
    assert flat['ori_index'] < out['ori_index']
    single = tuning.selectivity(DIRS16, _direction_cell(rng)[0])       # a mean curve only: indices, no tests
    assert np.isnan(single['ori_p']) and np.isnan(single['dir_p']) and single['ori_index'] > 0


def test_missing_presentations_are_used_for_the_curve_but_not_the_tests():
    rng = np.random.default_rng(7)
    trials = _direction_cell(rng, n_trials=8, noise=0.1)[1]
    holed = trials.copy(); holed[0, 3] = np.nan; holed[5, 9] = np.nan
    full = tuning.vector_selectivity(DIRS16, trials, 'direction')[0]
    part = tuning.vector_selectivity(DIRS16, holed, 'direction')[0]
    assert np.isfinite(part) and abs(part - full) < 0.02
    assert tuning.orientation_test(DIRS16, holed)['n'] == 6            # the two incomplete trials are left out


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
