#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bayesian tuning-curve analysis by Markov chain Monte Carlo sampling, after Cronin, Stevenson, Sur and Kording 2010,
J Neurophysiol 103:591, "Hierarchical Bayesian modeling and Markov chain Monte Carlo sampling for tuning-curve
analysis", https://doi.org/10.1152/jn.00379.2009

The generative model, as in the paper: a stimulus angle goes through a parametric TUNING-CURVE function, whose output
is the mean of a NOISE model that generates each trial's response. Every parameter has a flat prior over a stated
range. Metropolis sampling (one parameter at a time, normally distributed proposals) draws from the joint posterior of
the parameters given the trials. The samples then answer the questions the paper poses:

  what are the parameters, and how well are they known    posterior median and percentile credibility intervals
  is the cell tuned at all                                Bayes factor, circular Gaussian against a constant
  orientation or direction selective                      Bayes factor, 180 deg against 360 deg periodicity, or the
                                                          two-amplitude direction model against the 180 deg model
  did tuning change between two conditions                compare the credibility intervals of the two posteriors

Tuning-curve functions (angles in degrees; "circular Gaussian" = a Gaussian wrapped around the circle):
  'constant'    b                                                                    an untuned cell
  'circ180'     b + a  G(x - mu; 180, sigma)                                          orientation tuned
  'circ360'     b + a  G(x - mu; 360, sigma)                                          one direction only
  'direction'   b + a1 G(x - mu; 360, sigma) + a2 G(x - mu - 180; 360, sigma)         two directions, unequal
  with G(d; P, sigma) = sum_k exp(-(d + k P)^2 / (2 sigma^2)).
Noise models: 'gaussian' (default: additive, one more parameter ``noise`` = its SD, suited to fluorescence responses)
and 'poisson' (spike counts, the paper's examples).

Differences from the paper, all deliberate:
  * Sampler settings default to the paper's (10,000 burn-in sweeps, 20,000 sampled sweeps, every 50th kept = 400
    samples). Proposal widths adapt during burn-in only, toward about 44% acceptance, then stay fixed.
  * Many cells are sampled at once (one independent chain per cell, vectorized), which is what makes a whole field of
    view practical. Trials at the same angle enter through their count, sum, and sum of squares, which is exact.
  * The evidence for a model is estimated as in the paper by the harmonic mean of the sampled likelihoods. That
    estimator is known to be noisy and to under-penalize extra parameters (it barely feels the prior), so
    ``log_evidence_bic`` (= maximum log likelihood - k/2 ln n) is returned beside it as a conservative cross-check.
    Trust a Bayes factor when both agree.

Run the built-in demonstration on the paper's simulated cells:  .venv/bin/python marmanalysis/tuning_bayes.py
"""

import numpy as np
from scipy.special import gammaln

MODELS = {
    'constant': ('b',),
    'circ180': ('b', 'a', 'mu', 'sigma'),
    'circ360': ('b', 'a', 'mu', 'sigma'),
    'direction': ('b', 'a1', 'a2', 'mu', 'sigma'),
}
PERIOD = {'constant': None, 'circ180': 180.0, 'circ360': 360.0, 'direction': 360.0}
N_WRAP = 3                                    # wraps kept on each side: exact to 1e-5 for sigma up to half a period


def _wrapped_gaussian(d, period, sigma):
    """sum_k exp(-(d + k P)^2 / (2 sigma^2)) for d (..., K) and sigma (..., 1)."""
    d = (d + period / 2.0) % period - period / 2.0
    out = np.zeros(np.broadcast(d, sigma).shape)
    for k in range(-N_WRAP, N_WRAP + 1):
        out += np.exp(-(d + k * period) ** 2 / (2.0 * sigma ** 2))
    return out


def tuning_curve(model, params, angles):
    """The tuning-curve function. ``params`` (..., n_params) in the order of MODELS[model] (a trailing noise column is
    ignored); ``angles`` (K,). Returns (..., K)."""
    p = np.asarray(params, dtype=float)
    x = np.asarray(angles, dtype=float)
    b = p[..., 0:1]
    if model == 'constant':
        return b + np.zeros(x.shape)
    if model in ('circ180', 'circ360'):
        a, mu, sigma = p[..., 1:2], p[..., 2:3], p[..., 3:4]
        return b + a * _wrapped_gaussian(x - mu, PERIOD[model], sigma)
    if model == 'direction':
        a1, a2, mu, sigma = p[..., 1:2], p[..., 2:3], p[..., 3:4], p[..., 4:5]
        return b + a1 * _wrapped_gaussian(x - mu, 360.0, sigma) + a2 * _wrapped_gaussian(x - mu - 180.0, 360.0, sigma)
    raise ValueError('unknown model %r; expected one of %s' % (model, ', '.join(MODELS)))


def _statistics(angles, responses):
    """Per cell and angle: number of trials, their sum, and their sum of squares (all a Gaussian or Poisson likelihood
    needs). ``responses`` is (n_trials, n_angles) or (n_cells, n_trials, n_angles); NaN = missing presentation."""
    x = np.asarray(angles, dtype=float).ravel()
    y = np.asarray(responses, dtype=float)
    single = y.ndim == 2
    if single:
        y = y[None]
    if y.ndim != 3 or y.shape[2] != x.size:
        raise ValueError('responses must be (n_trials, n_angles) or (n_cells, n_trials, n_angles); got %s for %d angles'
                         % (np.shape(responses), x.size))
    ok = np.isfinite(y)
    y0 = np.where(ok, y, 0.0)
    return x, y, single, ok.sum(1).astype(float), y0.sum(1), (y0 ** 2).sum(1)


def default_priors(angles, responses, model, noise='gaussian'):
    """The flat prior ranges, (n_cells, n_params, 2), set from each cell's own data so that they are wide but proper:
      b        the range of the single-trial responses (Poisson: 0 to the largest count)
      a, a1, a2   0 to twice that range
      mu       the whole circle (one period)
      sigma    half the angle step (finer is not resolvable) to half a period (wider is indistinguishable from flat)
      noise    just above 0 to the range of the single-trial responses
    """
    x, y, _, n, s1, s2 = _statistics(angles, responses)
    lo_y, hi_y = np.nanmin(y, axis=(1, 2)), np.nanmax(y, axis=(1, 2))
    span = np.maximum(hi_y - lo_y, 1e-9)
    names = MODELS[model] + (('noise',) if noise == 'gaussian' else ())
    period = PERIOD[model]
    ux = np.unique(x % period) if period else x
    step = float(np.min(np.diff(ux))) if ux.size > 1 else (period or 360.0)
    prior = np.zeros((y.shape[0], len(names), 2))
    for j, name in enumerate(names):
        if name == 'b':
            prior[:, j, 0] = 1e-6 if noise == 'poisson' else lo_y
            prior[:, j, 1] = hi_y if noise == 'gaussian' else np.maximum(hi_y, 1.0)
        elif name in ('a', 'a1', 'a2'):
            prior[:, j, 1] = 2.0 * span
        elif name == 'mu':
            prior[:, j, 1] = period
        elif name == 'sigma':
            prior[:, j] = (step / 2.0, period / 2.0)
        elif name == 'noise':
            prior[:, j, 0], prior[:, j, 1] = 1e-9 * span, span
    return names, prior


def _log_likelihood(model, noise, params, x, n, s1, s2, const):
    f = tuning_curve(model, params, x)                                 # (C, K)
    if noise == 'gaussian':
        sd = params[:, -1]
        sse = np.sum(s2 - 2.0 * f * s1 + n * f * f, axis=1)
        return -n.sum(1) * np.log(sd) - 0.5 * sse / sd ** 2 + const
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = np.sum(np.where(n > 0, s1 * np.log(f) - n * f, 0.0), axis=1) + const
    return np.where(np.all(f > 0, axis=1), ll, -np.inf)


def _initial(model, noise, x, n, s1, s2, prior):
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
    lo, hi = np.nanmin(mean, axis=1), np.nanmax(mean, axis=1)
    best = x[np.nanargmax(mean, axis=1)]
    names = MODELS[model] + (('noise',) if noise == 'gaussian' else ())
    p = np.zeros((mean.shape[0], len(names)))
    for j, name in enumerate(names):
        if name == 'b':
            p[:, j] = np.nanmean(mean, axis=1) if model == 'constant' else lo
        elif name in ('a', 'a1'):
            p[:, j] = hi - lo
        elif name == 'a2':
            p[:, j] = 0.5 * (hi - lo)
        elif name == 'mu':
            p[:, j] = best % PERIOD[model]
        elif name == 'sigma':
            p[:, j] = 30.0
        elif name == 'noise':
            with np.errstate(invalid='ignore', divide='ignore'):
                within = np.sum(s2 - np.where(n > 0, s1 ** 2 / np.maximum(n, 1), 0.0), axis=1) / np.maximum(n.sum(1) - (n > 0).sum(1), 1)
            p[:, j] = np.sqrt(np.maximum(within, 1e-12))
    eps = 1e-6 * (prior[:, :, 1] - prior[:, :, 0])
    return np.clip(p, prior[:, :, 0] + eps, prior[:, :, 1] - eps)


def sample(angles, responses, model='circ180', noise='gaussian', n_burn=10000, n_samples=20000, thin=50, seed=0,
           priors=None):
    """Draw from the posterior of one tuning-curve model, for one cell or many.

    angles       (n_angles,) stimulus angles in degrees
    responses    (n_trials, n_angles) for one cell or (n_cells, n_trials, n_angles); NaN = missing presentation
    model, noise see the module docstring
    n_burn, n_samples, thin   sweeps discarded, sweeps sampled, and the spacing of the kept ones (a sweep updates every
                 parameter once)
    priors       optional (n_cells, n_params, 2) flat prior ranges in the order of the returned ``names`` (default:
                 default_priors)

    Returns a dict (arrays lose the leading cell axis when one cell was given):
      names            parameter names; for the 'direction' model the samples are relabelled so that a1 >= a2 and mu is
                       the preferred direction
      samples          (n_cells, n_kept, n_params)
      median, ci       posterior median and 95% credibility interval per parameter, (n_cells, n_params) and
                       (n_cells, n_params, 2); circular for mu (the interval can run through the wrap, lo > hi)
      loglik           (n_cells, n_kept) log likelihood of the kept samples
      log_evidence     harmonic-mean estimate of ln Pr(data | model) over all sampled sweeps
      log_evidence_bic maximum log likelihood - k/2 ln n, the conservative cross-check
      accept           (n_cells, n_params) acceptance rate after burn-in
    """
    if model not in MODELS:
        raise ValueError('unknown model %r; expected one of %s' % (model, ', '.join(MODELS)))
    if noise not in ('gaussian', 'poisson'):
        raise ValueError("noise must be 'gaussian' or 'poisson'")
    x, y, single, n, s1, s2 = _statistics(angles, responses)
    names, default = default_priors(x, y, model, noise)
    prior = default if priors is None else np.broadcast_to(np.asarray(priors, float), default.shape).copy()
    lo, hi = prior[:, :, 0], prior[:, :, 1]
    n_cells, n_par = lo.shape
    if noise == 'gaussian':
        const = -0.5 * n.sum(1) * np.log(2.0 * np.pi)
    else:
        const = -np.nansum(gammaln(y + 1.0), axis=(1, 2))
    circular = [j for j, name in enumerate(names) if name == 'mu']
    rng = np.random.default_rng(seed)
    p = _initial(model, noise, x, n, s1, s2, prior)
    ll = _log_likelihood(model, noise, p, x, n, s1, s2, const)
    step = 0.1 * (hi - lo)
    n_keep = n_samples // thin
    samples = np.empty((n_cells, n_keep, n_par))
    kept_ll = np.empty((n_cells, n_keep))
    neg_lse = np.full(n_cells, -np.inf)                                # running log sum exp(-loglik), harmonic mean
    ll_max = ll.copy()
    window = np.zeros((n_cells, n_par))
    accepted = np.zeros((n_cells, n_par))
    for it in range(n_burn + n_samples):
        for j in range(n_par):
            q = p.copy()
            q[:, j] += step[:, j] * rng.standard_normal(n_cells)
            if j in circular:
                q[:, j] = lo[:, j] + (q[:, j] - lo[:, j]) % (hi[:, j] - lo[:, j])
            inside = (q[:, j] >= lo[:, j]) & (q[:, j] <= hi[:, j])     # flat prior: 0 outside its range
            q[~inside, j] = p[~inside, j]
            ll_q = np.where(inside, _log_likelihood(model, noise, q, x, n, s1, s2, const), -np.inf)
            take = np.log(rng.random(n_cells)) < ll_q - ll
            p[take, j] = q[take, j]
            ll[take] = ll_q[take]
            window[:, j] += take
            if it >= n_burn:
                accepted[:, j] += take
        if it < n_burn:
            if (it + 1) % 50 == 0:                                     # adapt the proposal widths, burn-in only
                rate = window / 50.0
                step *= np.where(rate > 0.5, 1.3, np.where(rate < 0.35, 1.0 / 1.3, 1.0))
                step = np.minimum(step, hi - lo)
                window[:] = 0.0
        else:
            neg_lse = np.logaddexp(neg_lse, -ll)
            ll_max = np.maximum(ll_max, ll)
            k = it - n_burn
            if (k + 1) % thin == 0:
                samples[:, (k + 1) // thin - 1] = p
                kept_ll[:, (k + 1) // thin - 1] = ll
    if model == 'direction':                                           # (a1, a2, mu) = (a2, a1, mu + 180): relabel
        i1, i2, im = names.index('a1'), names.index('a2'), names.index('mu')
        swap = samples[:, :, i2] > samples[:, :, i1]
        a1 = np.where(swap, samples[:, :, i2], samples[:, :, i1])
        a2 = np.where(swap, samples[:, :, i1], samples[:, :, i2])
        samples[:, :, im] = np.where(swap, (samples[:, :, im] + 180.0) % 360.0, samples[:, :, im])
        samples[:, :, i1], samples[:, :, i2] = a1, a2
    median = np.median(samples, axis=1)
    ci = np.stack(np.percentile(samples, [2.5, 97.5], axis=1), axis=-1)
    for j in circular:
        period = PERIOD[model]
        centre = np.angle(np.mean(np.exp(2j * np.pi * samples[:, :, j] / period), axis=1)) * period / (2 * np.pi)
        dev = (samples[:, :, j] - centre[:, None] + period / 2.0) % period - period / 2.0
        median[:, j] = (centre + np.median(dev, axis=1)) % period
        ci[:, j] = (centre[:, None] + np.stack(np.percentile(dev, [2.5, 97.5], axis=1), axis=-1)) % period
    out = {'model': model, 'noise': noise, 'names': names, 'angles': x, 'samples': samples, 'median': median, 'ci': ci,
           'loglik': kept_ll, 'log_evidence': np.log(n_samples) - neg_lse,
           'log_evidence_bic': ll_max - 0.5 * n_par * np.log(np.maximum(n.sum(1), 1.0)),
           'accept': accepted / float(max(n_samples, 1)), 'prior': prior}
    if single:
        for key in ('samples', 'median', 'ci', 'loglik', 'log_evidence', 'log_evidence_bic', 'accept', 'prior'):
            out[key] = out[key][0]
    return out


def posterior_curves(result, grid=None):
    """The sampled tuning curves (the light lines in the paper's figures): each kept parameter sample pushed through
    the tuning-curve function. Returns (grid, curves (..., n_kept, n_grid), median curve, 95% band (..., n_grid, 2))."""
    period = PERIOD[result['model']] or 360.0
    grid = np.arange(0.0, max(period, float(np.ptp(result['angles'])) + 1.0), 2.0) if grid is None else np.asarray(grid, float)
    curves = tuning_curve(result['model'], result['samples'], grid)
    band = np.stack(np.percentile(curves, [2.5, 97.5], axis=-2), axis=-1)
    return grid, curves, np.median(curves, axis=-2), band


def compare_models(angles, responses, models=('constant', 'circ180', 'circ360', 'direction'), noise='gaussian',
                   evidence='harmonic', **sampler):
    """Bayesian model selection: sample every model, then compare their evidence.

    Returns a dict:
      results            {model: sample(...)}
      log_evidence       {model: (n_cells,)} by the chosen ``evidence`` ('harmonic', the paper's, or 'bic')
      best               (n_cells,) the model with the most evidence
      log_bf_tuned       ln Bayes factor of the best tuned model against 'constant'
      log_bf_direction   ln Bayes factor of the better of 'circ360' / 'direction' against 'circ180': does the response
                         differ between opposite directions. Only meaningful for a cell that is tuned at all: for an
                         untuned cell it compares two models that fit equally badly.
      log_bf_tuned_bic, log_bf_direction_bic, best_bic   the same from the BIC evidence, always returned as the
                         conservative cross-check

    Reading the numbers. Following Kass and Raftery 1995, ln Bayes factors above about 1, 3, and 5 are positive,
    strong, and very strong evidence. Decide at ln BF > 3, not > 0. Simulated calibration (200 cells, 16 directions,
    10 trials each, Gaussian noise, tuning amplitude = 1 noise SD, tmp/bayes_evidence_calibration.py, 2026-11-18):

                                         harmonic mean          BIC
      untuned cells called tuned         57% at > 0, 3.5% at > 3     0% at > 0, 0% at > 3
      oriented cells called tuned        98% at > 0, 84% at > 3      78% at > 0, 53% at > 3

    The harmonic mean barely penalizes the extra parameters, so at ln BF > 0 it calls an untuned cell tuned about half
    the time. At ln BF > 3 it is well behaved and more sensitive than BIC."""
    if evidence not in ('harmonic', 'bic'):
        raise ValueError("evidence must be 'harmonic' or 'bic'")
    results = {m: sample(angles, responses, m, noise, **sampler) for m in models}
    out = {'results': results, 'evidence': evidence}
    tuned = [m for m in models if m != 'constant']
    directional = [m for m in ('circ360', 'direction') if m in models]
    for kind, key in (('harmonic', 'log_evidence'), ('bic', 'log_evidence_bic')):
        suffix = '' if kind == evidence else '_' + kind
        log_ev = {m: np.atleast_1d(results[m][key]) for m in models}
        if kind == evidence:
            out['log_evidence'] = log_ev
        out['best' + suffix] = np.array(models)[np.argmax(np.stack([log_ev[m] for m in models]), axis=0)]
        if 'constant' in models and tuned:
            out['log_bf_tuned' + suffix] = np.max([log_ev[m] for m in tuned], axis=0) - log_ev['constant']
        if 'circ180' in models and directional:
            out['log_bf_direction' + suffix] = np.max([log_ev[m] for m in directional], axis=0) - log_ev['circ180']
    return out


def simulate(model, params, angles, n_trials, noise='poisson', noise_sd=1.0, seed=0):
    """Trials from the generative model, (n_trials, n_angles): the paper's simulated cells are
    simulate('circ180', (1, 4, 90, 20), ...) and simulate('direction', (1, 4, 2, 90, 20), ...)."""
    rng = np.random.default_rng(seed)
    f = np.tile(tuning_curve(model, np.asarray(params, float), angles), (n_trials, 1))
    return rng.poisson(f).astype(float) if noise == 'poisson' else f + noise_sd * rng.standard_normal(f.shape)


def _demo():
    """The paper's simulated cells: recover the parameters and pick the right model."""
    oris, dirs = np.arange(0.0, 180.0, 22.5), np.arange(0.0, 360.0, 22.5)
    cases = [('orientation cell, 8 orientations', 'circ180', (1, 4, 90, 20), oris, ('constant', 'circ180')),
             ('direction cell, 16 directions', 'direction', (1, 4, 2, 90, 20), dirs, ('constant', 'circ180', 'circ360', 'direction')),
             ('untuned cell, 16 directions', 'constant', (1,), dirs, ('constant', 'circ180', 'circ360', 'direction'))]
    for title, model, truth, angles, models in cases:
        trials = simulate(model, truth, angles, 10, 'poisson', seed=1)
        cmp = compare_models(angles, trials, models, noise='poisson')
        bic = {m: float(cmp['results'][m]['log_evidence_bic']) for m in models}
        other = 'best_bic' if cmp['evidence'] == 'harmonic' else 'best_harmonic'
        print('\n%s (truth: %s %s, 10 trials per angle, Poisson noise)' % (title, model, truth))
        for m in models:
            res = cmp['results'][m]
            fields = ', '.join('%s %.2f [%.2f, %.2f]' % (name, res['median'][j], res['ci'][j, 0], res['ci'][j, 1])
                               for j, name in enumerate(res['names']))
            print('  %-9s ln evidence %8.2f (harmonic) %8.2f (BIC)   %s' % (m, float(res['log_evidence']), bic[m], fields))
        print('  best model: %s (%s), %s (%s); ln BF tuned %.1f'
              % (cmp['best'][0], cmp['evidence'], cmp[other][0], other[5:], cmp['log_bf_tuned'][0]))


if __name__ == '__main__':
    _demo()
