#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for tuning_bayes (Cronin et al. 2010 tuning-curve sampling) on the paper's simulated cells: parameter
recovery with credibility intervals, model selection by Bayes factor, and the vectorized many-cell path.

Run:  .venv/bin/python -m pytest marmanalysis/tests/test_tuning_bayes.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tuning_bayes as tb

ORIS = np.arange(0.0, 180.0, 22.5)
DIRS = np.arange(0.0, 360.0, 22.5)
FAST = dict(n_burn=2000, n_samples=4000, thin=20)                    # 200 kept samples: enough for the checks here


def _circ_dist(a, b, period):
    return np.abs((np.asarray(a) - b + period / 2.0) % period - period / 2.0)


def test_tuning_curve_functions():
    grid = np.arange(0.0, 360.0, 1.0)
    c180 = tb.tuning_curve('circ180', (1.0, 4.0, 90.0, 20.0), grid)
    assert np.allclose(c180[:180], c180[180:])                       # the same for opposite directions
    assert abs(c180[90] - 5.0) < 1e-3 and abs(c180[0] - 1.0) < 1e-3  # b + a at the peak, b far from it
    c360 = tb.tuning_curve('circ360', (1.0, 4.0, 90.0, 20.0), grid)
    assert abs(c360[90] - 5.0) < 1e-3 and abs(c360[270] - 1.0) < 1e-3
    d = tb.tuning_curve('direction', (1.0, 4.0, 2.0, 90.0, 20.0), grid)
    assert abs(d[90] - 5.0) < 1e-3 and abs(d[270] - 3.0) < 1e-3
    wrap = tb.tuning_curve('circ180', (0.0, 1.0, 178.0, 10.0), np.array([2.0, 174.0]))
    assert abs(wrap[0] - wrap[1]) < 1e-9                             # 4 deg either side of the peak, across the wrap
    assert tb.tuning_curve('constant', (2.5,), grid).shape == grid.shape
    assert tb.tuning_curve('circ180', np.ones((3, 7, 4)) * [1, 4, 90, 20], ORIS).shape == (3, 7, 8)
    with pytest.raises(ValueError):
        tb.tuning_curve('nope', (1.0,), grid)


def test_recovers_the_papers_orientation_cell_with_poisson_noise():
    truth = (1.0, 4.0, 90.0, 20.0)
    trials = tb.simulate('circ180', truth, ORIS, 25, 'poisson', seed=3)
    res = tb.sample(ORIS, trials, 'circ180', 'poisson', **FAST)
    assert res['names'] == ('b', 'a', 'mu', 'sigma') and res['samples'].shape == (200, 4)
    for j, value in enumerate(truth):
        lo, hi = res['ci'][j]
        assert lo <= value <= hi, (res['names'][j], lo, hi)
    assert _circ_dist(res['median'][2], 90.0, 180.0) < 6 and abs(res['median'][3] - 20.0) < 6
    assert np.all((res['accept'] > 0.15) & (res['accept'] < 0.8))    # proposals adapted to a workable width


def test_gaussian_noise_many_cells_at_once_and_missing_presentations():
    truths = [(0.5, 3.0, 45.0, 25.0), (0.0, 2.0, 135.0, 15.0), (1.0, 5.0, 100.0, 30.0)]
    trials = np.stack([tb.simulate('circ180', t, ORIS, 15, 'gaussian', 0.5, seed=10 + i) for i, t in enumerate(truths)])
    trials[0, 2, 3] = np.nan
    trials[2, 7, 0] = np.nan
    res = tb.sample(ORIS, trials, 'circ180', **FAST)
    assert res['names'][-1] == 'noise' and res['samples'].shape == (3, 200, 5) and res['log_evidence'].shape == (3,)
    for c, t in enumerate(truths):
        assert _circ_dist(res['median'][c, 2], t[2], 180.0) < 6
        assert abs(res['median'][c, 1] - t[1]) < 0.8 and abs(res['median'][c, 4] - 0.5) < 0.12
    one = tb.sample(ORIS, trials[1], 'circ180', **FAST)             # one cell alone: no leading axis
    assert one['samples'].shape == (200, 5) and _circ_dist(one['median'][2], 135.0, 180.0) < 6


def test_preferred_angle_next_to_the_wrap_has_a_circular_interval():
    trials = tb.simulate('circ180', (0.0, 3.0, 178.0, 20.0), ORIS, 15, 'gaussian', 0.5, seed=4)
    res = tb.sample(ORIS, trials, 'circ180', **FAST)
    assert _circ_dist(res['median'][2], 178.0, 180.0) < 6
    lo, hi = res['ci'][2]
    assert _circ_dist(lo, 178.0, 180.0) < 15 and _circ_dist(hi, 178.0, 180.0) < 15


def test_more_trials_tighten_the_credibility_interval():
    width = []
    for n_trials in (5, 40):
        trials = tb.simulate('circ180', (1.0, 4.0, 90.0, 20.0), ORIS, n_trials, 'gaussian', 1.0, seed=5)
        res = tb.sample(ORIS, trials, 'circ180', **FAST)
        width.append(res['ci'][1, 1] - res['ci'][1, 0])              # amplitude
    assert width[1] < 0.6 * width[0]


def test_model_selection_tuned_untuned_and_directional():
    cells = np.stack([tb.simulate('constant', (1.0,), DIRS, 10, 'gaussian', 1.0, seed=20),
                      tb.simulate('circ180', (1.0, 3.0, 90.0, 20.0), DIRS, 10, 'gaussian', 1.0, seed=21),
                      tb.simulate('direction', (1.0, 4.0, 0.5, 90.0, 20.0), DIRS, 10, 'gaussian', 1.0, seed=22)])
    cmp = tb.compare_models(DIRS, cells, **FAST)
    assert cmp['log_bf_tuned'][0] < 3 and cmp['log_bf_tuned_bic'][0] < 0          # untuned: no strong evidence
    assert cmp['log_bf_tuned'][1] > 5 and cmp['log_bf_tuned_bic'][1] > 5          # oriented: very strong
    assert cmp['log_bf_direction'][1] < 3                                         # ... but not directional
    assert cmp['log_bf_tuned'][2] > 5 and cmp['log_bf_direction'][2] > 5          # directional on both counts
    assert cmp['log_bf_direction_bic'][2] > 5
    assert cmp['best'][2] in ('direction', 'circ360') and cmp['best_bic'][1] == 'circ180'
    res = cmp['results']['direction']
    names = res['names']
    a1, a2 = res['samples'][2][:, names.index('a1')], res['samples'][2][:, names.index('a2')]
    assert np.all(a1 >= a2)                                                       # relabelled: a1 is the preferred side
    assert _circ_dist(res['median'][2, names.index('mu')], 90.0, 360.0) < 8


def test_posterior_curves_bracket_the_true_curve():
    truth = (1.0, 4.0, 90.0, 20.0)
    trials = tb.simulate('circ180', truth, ORIS, 30, 'gaussian', 0.7, seed=6)
    res = tb.sample(ORIS, trials, 'circ180', **FAST)
    grid, curves, median, band = tb.posterior_curves(res)
    assert curves.shape == (200, grid.size) and band.shape == (grid.size, 2)
    true = tb.tuning_curve('circ180', truth, grid)
    assert np.mean((true >= band[:, 0]) & (true <= band[:, 1])) > 0.85
    assert np.max(np.abs(median - true)) < 1.0


def test_priors_can_be_given_and_are_respected():
    trials = tb.simulate('circ180', (1.0, 4.0, 90.0, 20.0), ORIS, 10, 'gaussian', 1.0, seed=7)
    names, prior = tb.default_priors(ORIS, trials, 'circ180')
    assert prior.shape == (1, 5, 2) and prior[0, names.index('sigma'), 0] == 22.5 / 2
    prior[0, names.index('sigma')] = (40.0, 60.0)                    # force a wide tuning curve
    res = tb.sample(ORIS, trials, 'circ180', priors=prior, **FAST)
    sigma = res['samples'][:, names.index('sigma')]
    assert sigma.min() >= 40.0 and sigma.max() <= 60.0
    with pytest.raises(ValueError):
        tb.sample(ORIS, trials, 'circ180', noise='lognormal')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
