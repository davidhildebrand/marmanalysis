#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize as scipy_minimize


def von_mises(params, theta):
    # Based on Pattadkal et al Priebe 2024 Neuron https://doi.org/10.1016/j.neuron.2023.11.005
    #   Methods section 'Quantification and Statistical Analysis'  
    
    theta_pref = params[0]  # rad, preferred direction
    beta = params[1]  # 'tuning width factor'
    a0 = params[2]  # baseline
    a1 = params[3]  # peak 1 maximum amplitude
    a2 = params[4]  # peak 2 maximum amplitude
    r = (a0 + 
        (a1 * np.exp(beta * np.cos(theta - theta_pref))) + 
        (a2 * np.exp(beta * np.cos(np.pi + theta - theta_pref))))
    return r


def gf(params, theta):
    # Based on Fahey et al Tolias 2019 bioRxiv
    #   https://doi.org/10.1101/745323
    
    theta_pref = params[0]  # rad, preferred direction
    w = params[1]  # peak concentration or 'tuning width factor'
    g = np.exp(-w * (1 - np.cos(theta - theta_pref)))
    return g


def von_mises_twopeakscaled(params, theta):
    # Based on Fahey et al Tolias 2019 bioRxiv
    #   https://doi.org/10.1101/745323
    
    # theta_pref = params[0]  # rad, preferred direction
    # w = params[1]  # peak concentration or 'tuning width factor'
    a0 = params[2]  # baseline
    a1 = params[3]  # peak 1 maximum amplitude
    a2 = params[4]  # peak 2 maximum amplitude
    v = a0 + (a1 * gf(params, theta)) + (a2 * gf(params, theta - np.pi))
    # v = (a0 + 
    #     (a1 * np.exp(-w * (1 - np.cos(theta - theta_pref)))) + 
    #     (a2 * np.exp(-w * (1 - np.cos(theta - np.pi - theta_pref)))))
    return v


def dsi_model(params, theta):
    resp = von_mises(params, theta)  # model used by Pattadkal et al Priebe 2024 Neuron
    # r = von_mises_twopeakscaled(params, theta)  # model used by Fahey et al Tolias 2019 bioRxiv
    return resp


def dsi_objective(params, theta, responses_measured):
    responses_predicted = dsi_model(params, theta)
    mse = np.square(np.subtract(responses_predicted, responses_measured)).mean()
    return mse


def calculate_dsi(xs, ys, unit='deg', plotting=False, debugging=False):
    if unit == 'deg':
        thetas = np.radians(xs)
    elif unit == 'rad':
        thetas = xs
    responses_measured = ys

    # params = [theta_pref, w/beta, a0, a1, a2]
    guess_baseline = np.nanmean(responses_measured)
    guess_amplitude = np.nanpercentile(responses_measured, 90)
    guess = [(np.pi / 2),
             np.radians(360 / 8),
             guess_baseline,
             guess_amplitude,
             guess_amplitude]

    result = scipy_minimize(dsi_objective, guess, args=(thetas, responses_measured), method='L-BFGS-B')
    fit = result['x']
    theta_pref_f = fit[0]
    w_f = fit[1]
    a0_f = fit[2]
    a1_f = fit[3]
    a2_f = fit[4]

    # Based on Pattadkal et al Priebe 2022 bioRxiv
    #   "The location of the peak of this fitted tuning curve is used as the preferred direction of each cell."
    fit_curve = dsi_model(fit, np.radians(np.arange(0, 360)))
    max_peak_arg = fit_curve.argmax()
    # peak_locs = find_peaks(fit_curve, height=0)[0]
    # peaks = fit_curve[peak_locs]
    # max_peak_loc = peak_locs[peaks.argmax()]
    theta_pref = np.radians(max_peak_arg)

    if plotting:
        import matplotlib.pyplot as plt
        deg_symbol = u'\N{DEGREE SIGN}'

        plt.figure()
        plt.scatter(np.degrees(thetas), responses_measured, s=4, facecolors='none', edgecolors='k')
        plt.plot(dsi_model(fit, np.radians(np.arange(0, 360))))
        plt.axvline(np.degrees(theta_pref), color='m')
        ax = plt.gca()
        ax.set_xlabel('Direction (' + deg_symbol + ')', fontsize=8)
        ax.set_ylabel('dF/F', fontsize=8)
        # ax.set_xlim((0,360))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([d for d in range(0, 360, np.diff(np.degrees(thetas)).max().astype('int'))])
        # ax.set_xticklabels(['', 0, '', 2, ''])
        # plt.scatter(thetas, measRs, s=4, facecolors='none', edgecolors='k')
        # plt.plot(dsi_model(fit, np.arange(0, 2*np.pi))) #np.radians(np.arange(0, 360))))
        # plt.axvline(theta_pref, color='m')
        plt.show()

    dT = thetas
    Rn = responses_measured + np.abs(np.min(responses_measured))
    dR = Rn
    # DSI based on Pattadkal et al Priebe 2022 bioRxiv
    dsi = np.sqrt(np.sum(dR * np.sin(dT))**2 + np.sum(dR * np.cos(dT))**2) / np.sum(dR)
    if unit == 'deg':
        theta_pref = np.degrees(theta_pref)
        theta_pref_f = np.degrees(theta_pref_f)

    if debugging:
        print('calculate_dsi dsi={:.2f} theta_pref={} '.format(dsi, theta_pref) +
              'theta_pref_f={:.2f} w={:.2f} a0={:.2f} '.format(theta_pref_f, w_f, a0_f) +
              'a1={:.2f} a2={:.2f}'.format(a1_f, a2_f))

    return dsi, theta_pref


def calculate_osi(xs, ys, unit='deg'):
    """Orientation-selectivity index (OSI): the orientation vector strength (1 - circular variance
    at 2*theta), with the preferred orientation from the resultant-vector angle.

    Based on Mazurek, Kager and Van Hooser 2014 Front Neural Circuits
      https://doi.org/10.3389/fncir.2014.00092
    "Robust quantification of orientation selectivity and direction selectivity."
    OSI = |sum_k R_k * exp(2i*theta_k)| / sum_k R_k, the 2*theta analogue of the DSI vector strength
    in calculate_dsi (so orientations 180 deg apart are treated as identical).

    xs: stimulus angles (orientations or drift directions), ys: responses (one per angle).
    Returns (osi, ori_pref) with ori_pref the preferred ORIENTATION in [0, 180) deg (or [0, pi) rad).
    Responses are shifted to be non-negative first, matching calculate_dsi.
    """
    thetas = np.radians(xs) if unit == 'deg' else np.asarray(xs, dtype=float)
    R = np.asarray(ys, dtype=float)
    R = R + np.abs(np.min(R))
    sin2 = np.sum(R * np.sin(2 * thetas))
    cos2 = np.sum(R * np.cos(2 * thetas))
    osi = np.sqrt(sin2**2 + cos2**2) / np.sum(R)
    ori_pref = 0.5 * np.arctan2(sin2, cos2)
    if unit == 'deg':
        ori_pref = np.degrees(ori_pref) % 180
    else:
        ori_pref = ori_pref % np.pi
    return osi, ori_pref


# ---------------------------------------------------------------------------------------------------------------------
# Orientation and direction selectivity after Mazurek, Kager and Van Hooser 2014, Front Neural Circuits 8:92,
# "Robust quantification of orientation selectivity and direction selectivity", https://doi.org/10.3389/fncir.2014.00092
#
# Their Table 2, which `selectivity` follows by default:
#   how selective (orientation)            1 - CirVar                               vector_selectivity(space='orientation')
#   how selective (direction)              1 - DirCirVar                            vector_selectivity(space='direction')
#   is it orientation selective            Hotelling's T^2 on orientation vectors   orientation_test
#   is it direction selective              direction dot product test               direction_dot_product_test
#   two populations, selectivity           2-sample t-test on the vector indices    compare_selectivity
#   two populations, any tuning parameter  2-sample Hotelling's T^2 on vectors      hotelling_t2_two_sample
#   tuning angle, tuning width             Gaussian / double Gaussian fit           fit_gaussian
#   uncertainty of those                   bootstrap over trials, refit             bootstrap_fit
# The classic peak-based indices (OI, DI, and the OSI / DSI variants) are in peak_selectivity. The paper shows they
# are biased upward by noise for weakly tuned cells, so they are provided for comparison, not as the default.
#
# Conventions: angles in degrees. "Direction space" = stimulus directions over 360 deg; "orientation space" = angles
# over 180 deg. ``responses`` is (n_angles,) or (n_trials, n_angles); a trial is one measurement at every angle, and
# the tests need trials. Responses are expected background-subtracted, as in the paper.
# ---------------------------------------------------------------------------------------------------------------------
def _angles_trials(angles, responses):
    a = np.asarray(angles, dtype=float).ravel()
    r = np.asarray(responses, dtype=float)
    trials = r if r.ndim == 2 else r[None, :]
    if trials.shape[1] != a.size:
        raise ValueError('responses must be (n_angles,) or (n_trials, n_angles); got %s for %d angles' % (r.shape, a.size))
    return a, trials[np.all(np.isfinite(trials), axis=1)]           # a trial needs every angle


def _mean_curve(angles, responses):
    """The trial-averaged tuning curve. Unlike the tests, it uses every measurement, including trials in which some
    angles are missing (excluded presentations are NaN)."""
    a = np.asarray(angles, dtype=float).ravel()
    r = np.asarray(responses, dtype=float)
    if r.ndim == 1:
        return a, r
    with np.errstate(invalid='ignore'):
        counts = np.sum(np.isfinite(r), axis=0)
        return a, np.where(counts > 0, np.nansum(r, axis=0) / np.maximum(counts, 1), np.nan)


def _ang_ori(x):
    """Angular difference wrapped onto 0..90 deg (orientation space)."""
    return np.abs((np.asarray(x, dtype=float) + 90.0) % 180.0 - 90.0)


def _ang_dir(x):
    """Angular difference wrapped onto 0..180 deg (direction space)."""
    return np.abs((np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0)


def response_vectors(angles, responses, space='orientation'):
    """One complex vector per trial: sum_k R(theta_k) * exp(i * m * theta_k), with m = 2 in orientation space (angles
    180 deg apart count as the same) and m = 1 in direction space. These are the paper's "orientation vectors" and
    "direction vectors"."""
    a, trials = _angles_trials(angles, responses)
    m = 2.0 if space == 'orientation' else 1.0
    return trials @ np.exp(1j * m * np.radians(a))


def vector_selectivity(angles, responses, space='orientation', negative='rectify'):
    """The recommended index of HOW selective a cell is: 1 - CirVar (orientation) or 1 - DirCirVar (direction),

        L = | sum_k R(theta_k) exp(i m theta_k) | / sum_k R(theta_k),   m = 2 (orientation) or 1 (direction)

    on the trial-averaged tuning curve. 0 = no selectivity, 1 = response to a single angle. Also returns the preferred
    angle from the vector (orientation in [0, 180), direction in [0, 360)).

    The index assumes non-negative responses. ``negative``: 'rectify' (default) sets negative mean responses to 0,
    'shift' subtracts the minimum when it is negative, 'keep' uses them as they are (the index can then exceed 1).
    Returns (index, preferred_angle); NaN when the summed response is not positive."""
    a, r = _mean_curve(angles, responses)
    if not np.all(np.isfinite(r)):
        return np.nan, np.nan
    if negative == 'rectify':
        r = np.clip(r, 0.0, None)
    elif negative == 'shift':
        r = r - min(r.min(), 0.0)
    elif negative != 'keep':
        raise ValueError("negative must be 'rectify', 'shift' or 'keep'")
    m = 2.0 if space == 'orientation' else 1.0
    total = r.sum()
    if not total > 0:
        return np.nan, np.nan
    v = np.sum(r * np.exp(1j * m * np.radians(a)))
    period = 180.0 if space == 'orientation' else 360.0
    return float(np.abs(v) / total), float((np.degrees(np.angle(v)) / m) % period)


def _curve_at(angles, curve, theta, period):
    """The tuning curve at ``theta``: the measured value if that angle was shown, else linear interpolation around the
    circle."""
    a = np.asarray(angles, float) % period
    order = np.argsort(a)
    return float(np.interp(theta % period, a[order], np.asarray(curve, float)[order], period=period))


def peak_selectivity(angles, responses, space='direction'):
    """The classic peak-based indices, from the best measured angle (the paper's OI and DI, and the OSI / DSI variants
    other papers use). Biased upward by noise for weakly tuned cells: use vector_selectivity to quantify selectivity.

    Direction space (angles over 360 deg):
        R_pref = largest response, R_null = response 180 deg away, R_orth+- = responses 90 deg to either side
        OI  = (R_pref + R_null - R_orth+ - R_orth-) / (R_pref + R_null)        DI  = (R_pref - R_null) / R_pref
        OSI = (R_pref_ori - R_orth) / (R_pref_ori + R_orth)                    DSI = (R_pref - R_null) / (R_pref + R_null)
        with R_pref_ori = (R_pref + R_null) / 2 and R_orth = (R_orth+ + R_orth-) / 2
    Orientation space (angles over 180 deg): OI = (R_pref_ori - R_orth) / R_pref_ori and OSI as above; no DI / DSI.
    O/P = R_orth / R_pref_ori = 1 - OI. Returns a dict."""
    a, r = _mean_curve(angles, responses)
    if not np.all(np.isfinite(r)):
        return {key: np.nan for key in ('theta_pref', 'R_pref', 'R_null', 'R_pref_ori', 'R_orth', 'OI', 'DI', 'OSI', 'DSI', 'O/P')}
    k = int(np.argmax(r))
    out = {'theta_pref': float(a[k]), 'R_pref': float(r[k])}
    if space == 'direction':
        null = _curve_at(a, r, a[k] + 180.0, 360.0)
        orth = 0.5 * (_curve_at(a, r, a[k] + 90.0, 360.0) + _curve_at(a, r, a[k] - 90.0, 360.0))
        pref_ori = 0.5 * (r[k] + null)
        out.update(R_null=null, DI=(r[k] - null) / r[k] if r[k] else np.nan,
                   DSI=(r[k] - null) / (r[k] + null) if (r[k] + null) else np.nan)
    else:
        orth = _curve_at(a, r, a[k] + 90.0, 180.0)
        pref_ori = float(r[k])
    out.update(R_pref_ori=float(pref_ori), R_orth=float(orth),
               OI=(pref_ori - orth) / pref_ori if pref_ori else np.nan,
               OSI=(pref_ori - orth) / (pref_ori + orth) if (pref_ori + orth) else np.nan)
    out['O/P'] = 1.0 - out['OI'] if np.isfinite(out['OI']) else np.nan
    return {key: (float(v) if v is not None else np.nan) for key, v in out.items()}


def hotelling_t2(vectors):
    """One-sample Hotelling's T^2 test that the mean of 2-D vectors is [0, 0] (the multivariate t-test).
    ``vectors``: complex (n,) or real (n, 2). Returns dict(t2, f, p, n); p is NaN with fewer than 3 vectors."""
    from scipy import stats
    v = np.asarray(vectors)
    x = np.column_stack([v.real, v.imag]) if np.iscomplexobj(v) else np.asarray(v, float)
    n, p = x.shape
    if n <= p:
        return {'t2': np.nan, 'f': np.nan, 'p': np.nan, 'n': int(n)}
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False)
    t2 = float(n * mean @ np.linalg.pinv(cov) @ mean)
    f = (n - p) / (p * (n - 1.0)) * t2
    return {'t2': t2, 'f': float(f), 'p': float(stats.f.sf(f, p, n - p)), 'n': int(n)}


def hotelling_t2_two_sample(vectors_a, vectors_b):
    """Two-sample Hotelling's T^2 test that two sets of 2-D vectors share a mean: the paper's screen for ANY
    difference in tuning between two populations (one vector per cell). Returns dict(t2, f, p)."""
    from scipy import stats

    def as_xy(v):
        v = np.asarray(v)
        return np.column_stack([v.real, v.imag]) if np.iscomplexobj(v) else np.asarray(v, float)
    a, b = as_xy(vectors_a), as_xy(vectors_b)
    n1, n2, p = len(a), len(b), a.shape[1]
    if n1 + n2 - p - 1 <= 0:
        return {'t2': np.nan, 'f': np.nan, 'p': np.nan}
    d = a.mean(0) - b.mean(0)
    pooled = ((n1 - 1) * np.cov(a, rowvar=False) + (n2 - 1) * np.cov(b, rowvar=False)) / (n1 + n2 - 2.0)
    t2 = float(n1 * n2 / (n1 + n2) * d @ np.linalg.pinv(pooled) @ d)
    f = (n1 + n2 - p - 1.0) / (p * (n1 + n2 - 2.0)) * t2
    return {'t2': t2, 'f': float(f), 'p': float(stats.f.sf(f, p, n1 + n2 - p - 1))}


def orientation_test(angles, responses):
    """The recommended test for orientation selectivity: Hotelling's T^2 on the per-trial orientation vectors against
    a mean of [0, 0]. ``responses`` must be (n_trials, n_angles). Returns dict(t2, f, p, n)."""
    return hotelling_t2(response_vectors(angles, responses, 'orientation'))


def direction_dot_product_test(angles, responses):
    """The recommended test for direction selectivity (needs directions over 360 deg and trials).

    The orientation axis is the angle of the mean orientation vector. Each trial's direction vector is projected onto
    that axis (its "direction dot product"), and a t-test asks whether the projections have mean 0. Hotelling's test
    on the direction vectors themselves is insensitive because direction space is sampled coarsely.
    Returns dict(t, p, n, axis (deg, in [0, 180)), preferred_direction (deg, the end of the axis the responses favor))."""
    from scipy import stats
    ori = response_vectors(angles, responses, 'orientation')
    axis = (np.degrees(np.angle(ori.mean())) / 2.0) % 180.0
    unit = np.exp(1j * np.radians(axis))
    dots = np.real(response_vectors(angles, responses, 'direction') * np.conj(unit))
    if dots.size < 2:
        return {'t': np.nan, 'p': np.nan, 'n': int(dots.size), 'axis': float(axis), 'preferred_direction': np.nan}
    t, p = stats.ttest_1samp(dots, 0.0)
    return {'t': float(t), 'p': float(p), 'n': int(dots.size), 'axis': float(axis),
            'preferred_direction': float(axis if dots.mean() >= 0 else (axis + 180.0) % 360.0)}


def compare_selectivity(index_a, index_b):
    """The recommended comparison of selectivity between two populations: a 2-sample t-test on their vector indices
    (1 - CirVar or 1 - DirCirVar, one value per cell). Returns dict(t, p)."""
    from scipy import stats
    a, b = np.asarray(index_a, float), np.asarray(index_b, float)
    t, p = stats.ttest_ind(a[np.isfinite(a)], b[np.isfinite(b)])
    return {'t': float(t), 'p': float(p)}


def gaussian_tuning(theta, C, Rp, theta_pref, sigma, Rn=None):
    """The paper's fit functions. Orientation space (Rn None): C + Rp exp(-ang_ori(theta - theta_pref)^2 / 2 sigma^2).
    Direction space: C + Rp exp(-ang_dir(theta - theta_pref)^2 / 2 sigma^2) + Rn exp(-ang_dir(theta - theta_pref + 180)^2 / 2 sigma^2)."""
    theta = np.asarray(theta, float)
    if Rn is None:
        return C + Rp * np.exp(-_ang_ori(theta - theta_pref) ** 2 / (2.0 * sigma ** 2))
    return (C + Rp * np.exp(-_ang_dir(theta - theta_pref) ** 2 / (2.0 * sigma ** 2))
            + Rn * np.exp(-_ang_dir(theta - theta_pref + 180.0) ** 2 / (2.0 * sigma ** 2)))


def fit_gaussian(angles, responses, space='direction'):
    """Gaussian (orientation space) or double Gaussian (direction space) fit of the trial-averaged tuning curve, with
    the paper's two safeguards against poor fits:
      constraints  sigma >= half the angle step; C in [-M, M]; Rp and Rn in [0, 3 M], M = the largest response
      starts       theta_pref at the best angle, Rp = Rn = M, C = 0, and sigma from {step/2, step, 40, 60, 90} deg;
                   the fit with the lowest squared error is kept.
    Returns dict(C, Rp, Rn, theta_pref, sigma, hwhh (half-width at half-height = sigma * sqrt(2 ln 2)), sse, r2, and
    the indices read from the fitted curve: OI, DI). Per the paper, report the angle and width only for cells that
    pass orientation_test."""
    from scipy.optimize import least_squares
    nan_fit = {key: np.nan for key in ('C', 'Rp', 'Rn', 'theta_pref', 'sigma', 'hwhh', 'sse', 'r2', 'OI', 'DI')}
    a, r = _mean_curve(angles, responses)
    if not np.all(np.isfinite(r)):
        return nan_fit
    direction = space == 'direction'
    period = 360.0 if direction else 180.0
    step = float(np.min(np.diff(np.sort(np.unique(a % period))))) if np.unique(a % period).size > 1 else period
    M = float(np.max(r))
    if not M > 0:
        M = float(np.max(np.abs(r))) or 1.0
    k = int(np.argmax(r))
    lo = [-M, 0.0, -np.inf, step / 2.0] + ([0.0] if direction else [])
    hi = [M, 3.0 * M, np.inf, max(period, step)] + ([3.0 * M] if direction else [])

    def model(p):
        return gaussian_tuning(a, p[0], p[1], p[2], p[3], p[4] if direction else None)
    best = None
    for s0 in (step / 2.0, step, 40.0, 60.0, 90.0):
        x0 = [0.0, M, a[k], float(np.clip(s0, lo[3] * 1.0001, hi[3]))] + ([M] if direction else [])
        try:
            res = least_squares(lambda p: model(p) - r, x0, bounds=(lo, hi))
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return nan_fit
    p = best.x
    C, Rp, theta, sigma = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    Rn = float(p[4]) if direction else np.nan
    if direction and Rn > Rp:                                      # the larger peak is the preferred direction
        Rp, Rn, theta = Rn, Rp, theta + 180.0
    theta %= period
    sse = float(2.0 * best.cost)
    tot = float(np.sum((r - r.mean()) ** 2))
    fine = np.arange(0.0, period, 1.0)
    curve = gaussian_tuning(fine, C, Rp, theta, sigma, Rn if direction else None)
    idx = peak_selectivity(fine, curve, space)
    return {'C': C, 'Rp': Rp, 'Rn': Rn, 'theta_pref': float(theta), 'sigma': sigma,
            'hwhh': float(sigma * np.sqrt(2.0 * np.log(2.0))), 'sse': sse, 'r2': float(1.0 - sse / tot) if tot > 0 else np.nan,
            'OI': idx['OI'], 'DI': idx.get('DI', np.nan)}


def bootstrap_fit(angles, responses, space='direction', n_boot=100, seed=0):
    """The recommended measure of uncertainty for fitted tuning parameters: resample the trials with replacement,
    refit, repeat. Returns dict(samples={param: (n_boot,)}, theta_pref (circular mean), theta_ci (95% circular
    interval half-width, deg), and ci={param: (2.5th, 97.5th percentile)} for the other parameters)."""
    a, trials = _angles_trials(angles, responses)
    if trials.shape[0] < 2:
        raise ValueError('the bootstrap needs trials: responses must be (n_trials, n_angles) with n_trials >= 2')
    rng = np.random.default_rng(seed)
    keys = ('C', 'Rp', 'Rn', 'theta_pref', 'sigma', 'hwhh', 'OI', 'DI')
    samples = {key: np.empty(n_boot) for key in keys}
    for b in range(n_boot):
        fit = fit_gaussian(a, trials[rng.integers(0, trials.shape[0], trials.shape[0])], space)
        for key in keys:
            samples[key][b] = fit[key]
    period = 360.0 if space == 'direction' else 180.0
    z = np.exp(2j * np.pi * samples['theta_pref'] / period)
    centre = (np.angle(np.nanmean(z)) * period / (2 * np.pi)) % period
    dev = np.abs((samples['theta_pref'] - centre + period / 2.0) % period - period / 2.0)
    return {'samples': samples, 'theta_pref': float(centre), 'theta_ci': float(np.nanpercentile(dev, 95)),
            'ci': {key: (float(np.nanpercentile(samples[key], 2.5)), float(np.nanpercentile(samples[key], 97.5)))
                   for key in keys if key != 'theta_pref'}}


def selectivity(angles, responses, space='direction', negative='rectify', fit=False, alpha=0.05):
    """Everything the paper recommends for one cell, in one call.

    Always: ``ori_index`` = 1 - CirVar and its preferred orientation; in direction space also ``dir_index`` =
    1 - DirCirVar and its preferred direction. With trials: ``ori_p`` (Hotelling's T^2) and, in direction space,
    ``dir_p`` (direction dot product test). The peak-based OI / DI / OSI / DSI are included under their own names for
    comparison. With ``fit=True`` the Gaussian fit is added under 'fit'; its angle and width are meant to be used only
    when ``ori_p < alpha`` (flag ``fit_reportable``)."""
    a, trials = _angles_trials(angles, responses)
    out = {}
    out['ori_index'], out['ori_pref'] = vector_selectivity(a, responses, 'orientation', negative)
    if space == 'direction':
        out['dir_index'], out['dir_pref'] = vector_selectivity(a, responses, 'direction', negative)
    has_trials = trials.shape[0] >= 3
    out['ori_p'] = orientation_test(a, trials)['p'] if has_trials else np.nan
    if space == 'direction':
        dot = direction_dot_product_test(a, trials) if has_trials else {'p': np.nan, 'preferred_direction': np.nan}
        out['dir_p'], out['dir_pref_axis'] = dot['p'], dot['preferred_direction']
    out.update({key: v for key, v in peak_selectivity(a, responses, space).items() if key in ('OI', 'DI', 'OSI', 'DSI', 'O/P')})
    if fit:
        out['fit'] = fit_gaussian(a, responses, space)
        out['fit_reportable'] = bool(np.isfinite(out['ori_p']) and out['ori_p'] < alpha)
    return out
