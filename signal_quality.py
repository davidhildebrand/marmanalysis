#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Signal-quality / active-cell metrics on full-session ΔF/F traces.

Ports the SNR/noise block out of the monolithic ``analysis_for_images.py`` into the modular library
(that block was NOT part of the earlier sessionio/response_table/images refactor). An **active** /
dynamic cell shows real calcium transients above its own noise, independent of any stimulus -- the
stimulus-agnostic gate that sits above **responsive** (responds to a stimulus) and **selective**
(differentiates stimuli):

    active  <=>  peak_response / sigma_noise  >  k

The peak is a robust high percentile of ΔF/F; ``sigma_noise`` is a robust NOISE estimate (built from the
high-frequency or below-baseline structure of the trace, so it measures noise rather than signal). Several
estimators are provided so they can be compared:

  'nu'      Rupprecht et al. 2021 standardized noise = median|diff(ΔF/F%)| / sqrt(framerate). Frame-rate
            NORMALISED, so comparable across acquisition rates as a noise LEVEL -- but note nu =
            sigma_perframe * sqrt(2/fr) * 0.6745*..., so peak/nu is itself frame-rate dependent.
  'diff'    robust per-frame noise from consecutive differences = median|diff(ΔF/F)| / (sqrt(2)*0.6745).
  'mad'     MAD -> std over the whole trace = median|x - median(x)| / 0.6745.
  'mad_isi' v9-style: MAD of the BELOW-F0 deviations during the inter-stimulus periods / 0.6745 -- the
            most signal-excluding estimate (needs a baseline-relative trace + a boolean isi_mask).
  'std'     plain std of ΔF/F (NOT recommended: includes signal, so it is inflated for active cells).

Rupprecht et al. 2021 Nat Neurosci https://doi.org/10.1038/s41593-021-00895-5
"""
import numpy as np
from warnings import warn

# median(|x|) = 0.6745 * sigma for zero-mean Gaussian noise, so sigma = median(|x - med|) / 0.6745.
_MAD_TO_STD = 0.6745

# 'nu' is the recommended default: peak/nu is frame-rate INDEPENDENT (Rupprecht's nu divides out the
# sqrt(framerate) scaling of per-frame shot noise), so a threshold transfers across sessions imaged at
# different rates. 'mad'/'diff'/'mad_isi' give a Gaussian-equivalent per-frame sigma whose SNR is
# frame-rate DEPENDENT -- within-session alternates.
#
# nu k=7.0: a PERMISSIVE "is this a real cell" gate, not a "clearly-driven" bar. Goal: drop only debris and
# dead/silent ROIs, keeping the neurons -- cellpose already enforces morphology and over a long session most
# neurons fire, so we expect to remove very few. On the reference session (Cadbury PD) k=7 keeps ~773/791
# (98%) and, crucially, CONTAINS all responsive+selective cells (their min nu-SNR ~7.2 > 7), so the
# active/responsive/selective nesting holds. nu is frame-rate INDEPENDENT (per-frame sigma ~ sqrt(framerate),
# which nu divides out), so the threshold transfers across acquisition rates; it is also simple/structured --
# unlike raw-dF/F transient counting, which is param-sensitive. Very slow rates (2-3 Hz) can undersample the
# transient PEAK (nu fixes the noise term, not the peak) -- active_mask warns below warn_slow_hz (default
# 4 Hz). The biologically-principled alternative is an event/transient count (n_transients), best done via
# deconvolution. mad/diff/mad_isi k=3 = the plain within-session 3-sigma bar.
_DEFAULT_K = {'nu': 7.0, 'diff': 3.0, 'mad': 3.0, 'mad_isi': 3.0, 'std': 3.0}


def standardized_noise(dff, framerate, percent=True):
    """Rupprecht standardized noise nu per ROI: median|diff(ΔF/F)| / sqrt(framerate).

    ``dff`` is (n_roi, n_frames) ΔF/F as a FRACTION. With ``percent=True`` it is taken to percent (x100)
    so the result is on Rupprecht's %·Hz^-1/2 scale (nu ~ 1 low ... 8 high). This is the fix for
    analysis_for_images.py:1011, which computed on the fraction and so ran ~100x low.
    """
    x = np.asarray(dff, float) * (100.0 if percent else 1.0)
    return np.nanmedian(np.abs(np.diff(x, axis=1)), axis=1) / np.sqrt(framerate)


def noise_sigma(dff, framerate=None, method='diff', isi_mask=None):
    """Per-ROI noise sigma by the chosen estimator (see module docstring). Same ΔF/F units as ``dff``,
    except 'nu' which returns Rupprecht's frame-rate-standardized value (percent scale)."""
    dff = np.asarray(dff, float)
    if method == 'nu':
        if framerate is None:
            raise ValueError("method='nu' needs framerate")
        return standardized_noise(dff, framerate, percent=True)
    if method == 'diff':
        return np.nanmedian(np.abs(np.diff(dff, axis=1)), axis=1) / (np.sqrt(2.0) * _MAD_TO_STD)
    if method == 'mad':
        med = np.nanmedian(dff, axis=1, keepdims=True)
        return np.nanmedian(np.abs(dff - med), axis=1) / _MAD_TO_STD
    if method == 'std':
        return np.nanstd(dff, axis=1)
    if method == 'mad_isi':
        if isi_mask is None:
            raise ValueError("method='mad_isi' needs a boolean isi_mask over frames")
        dev = np.where(np.asarray(isi_mask, bool)[None, :], dff, np.nan)  # ISI frames only
        dev = np.where(dev > 0, np.nan, dev)                              # below-baseline (noise) only
        return np.nanmedian(np.abs(dev), axis=1) / _MAD_TO_STD
    raise ValueError('unknown noise method %r' % method)


def peak_response(dff, pct=99.0):
    """Robust peak ΔF/F per ROI (the ``pct``-th percentile of the trace)."""
    return np.nanpercentile(np.asarray(dff, float), pct, axis=1)


def active_mask(dff, framerate=None, method='nu', k=None, pct=99.0, isi_mask=None, warn_slow_hz=4.0):
    """Boolean active/dynamic mask + the SNR used. active <=> peak_response / sigma_noise > k.

    ``method`` defaults to 'nu' (Rupprecht standardized noise) -- the frame-rate-INDEPENDENT choice,
    recommended when comparing across sessions at different acquisition rates; 'mad'/'diff'
    (Gaussian-equivalent per-frame sigma) and 'mad_isi' (v9-style) are within-session alternates. ``k``
    defaults per-method (nu ~7, others ~3; see _DEFAULT_K) -- choose it from the SNR distribution for
    your data. Returns ``(mask, snr, peak, sigma)`` per ROI. For 'nu', peak is taken to the same percent
    scale as sigma so the ratio is consistent.

    When ``framerate`` is below ``warn_slow_hz`` (default 4 Hz) a warning is emitted: at slow acquisition
    rates the transient peak can be undersampled, biasing the active SNR (and cell count) low, because nu
    standardizes the noise term but NOT the peak -- consider lowering k or inspecting the traces.
    """
    if k is None:
        k = _DEFAULT_K.get(method, 3.0)
    if framerate is not None and framerate < warn_slow_hz:
        warn('active_mask: framerate %.2f Hz < %.1f Hz -- the transient peak may be undersampled, '
             'biasing the active SNR/count low (nu standardizes noise, not peak); consider lowering k.'
             % (framerate, warn_slow_hz), stacklevel=2)
    dff = np.asarray(dff, float)
    sigma = noise_sigma(dff, framerate, method, isi_mask)
    peak = peak_response(dff * (100.0 if method == 'nu' else 1.0), pct)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = peak / sigma
    return snr > k, snr, peak, sigma


def _count_transient_runs(x, onset_val, offset_val, min_frames):
    """Count contiguous above-``offset_val`` runs of x that peak >= ``onset_val`` and span >= ``min_frames``
    (hysteresis: a transient is bounded by the offset crossing but must reach the onset threshold)."""
    above = np.flatnonzero(x >= offset_val)
    if above.size == 0:
        return 0
    brk = np.flatnonzero(np.diff(above) > 1)
    starts = np.concatenate(([above[0]], above[brk + 1]))
    ends = np.concatenate((above[brk], [above[-1]]))
    n = 0
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_frames and x[s:e + 1].max() >= onset_val:
            n += 1
    return n


def n_transients(dff, framerate, sigma=None, sigma_method='diff', isi_mask=None,
                    onset=3.0, offset=1.0, min_duration_sec=0.5):
    """Per-ROI count of significant calcium transients over the whole session.

    A transient is a contiguous ΔF/F excursion that PEAKS above ``onset``*sigma and, with hysteresis, stays
    above ``offset``*sigma for at least ``min_duration_sec`` (converted to frames via ``framerate``). This is
    a biologically grounded, PERMISSIVE activity measure -- real neurons fire calcium transients; debris and
    silent/dead cells do not. Using ``n_transients >= N`` as the 'active' gate keeps functioning neurons
    (including any stimulus-responsive cell, which by definition fires) and removes only non-cells -- unlike
    the peak-SNR gate, which also drops small-but-reliable responders. ``sigma`` is the per-ROI noise
    (default from ``noise_sigma(method=sigma_method)``); ΔF/F units.

    CAVEAT: raw-ΔF/F thresholding is param-sensitive -- too-loose ``onset``/``min_duration_sec`` count noise
    excursions as transients (every ROI gets hundreds), so the defaults are matched to real calcium kinetics
    (amplitude a few sigma, sustained >= the decay time). A robust, threshold-free event count should come
    from deconvolution (OASIS/Cascade), which infers events from the indicator kinetics; treat this as the
    interim measure.
    """
    dff = np.asarray(dff, float)
    if sigma is None:
        sigma = noise_sigma(dff, framerate, sigma_method, isi_mask)
    min_frames = max(1, int(round(min_duration_sec * framerate)))
    counts = np.zeros(dff.shape[0], dtype=int)
    for r in range(dff.shape[0]):
        counts[r] = _count_transient_runs(dff[r], onset * sigma[r], offset * sigma[r], min_frames)
    return counts


def calculate_indicator_decay_tau(dff, framerate, isi_mask=None, method='half_decay', min_amp_sigma=4.0,
                                  sigma_method='diff', isolation_sec=1.0, decay_sec=2.0, tau_bounds=(0.05, 2.0),
                                  max_peaks_per_roi=30, p0_tau=0.29, min_r2=0.8, max_rois=None):
    """Empirical indicator decay time constant tau (seconds) from ISOLATED calcium transients in the dF/F
    traces -- a per-session, data-driven CHECK on the published jGCaMP8s value (tau ~0.29 s, from the paper's
    ~0.20 s single-AP HALF-decay / ln2; cf. process_with_suite2p.py and eyetracking.INDICATOR_DECAY_TAU_SEC).

    Per ROI: estimate the noise sigma; find peaks >= ``min_amp_sigma`` * sigma separated by >=
    ``isolation_sec`` (isolated events, not shoulders of sustained firing; at most ``max_peaks_per_roi``, the
    largest). For each, measure the HALF-decay time t_half -- the interpolated time from the peak until dF/F
    falls to 50% of (peak - pre-peak baseline) -- and convert to a decay time constant tau = t_half / ln2.
    The 50%-crossing (``method='half_decay'``, default) is preferred over an exponential fit (available as
    ``method='curve_fit'``, which fits A*exp(-t/tau)+b per transient, slower, keeping R^2 >= ``min_r2``)
    because calcium transients are not clean single
    exponentials (rounded peak + slow tail), which biases exp-fit tau by fit window; t_half is baseline-robust
    AND is exactly the quantity the jGCaMP8 paper reports, so the empirical value compares directly (its
    t_half ~0.20 s). Transients that do not reach 50% within ``decay_sec`` are skipped. Pass ``isi_mask``
    (per-frame bool, True on NON-stimulus frames) to measure only spontaneous decays -- stimulus-driven
    SUSTAINED firing inflates the apparent decay, so this is an UPPER bound on the pure-indicator value:
    compare it against the published value, do not blindly replace. (On the Cadbury PD session the median came
    to ~0.30 s tau / ~0.21 s t_half, matching the published 0.29 / 0.20 s.) Returns {'tau_median', 'tau_iqr'
    (q25,q75), 't_half_median', 'n_events', 'per_event_tau', 'per_event_roi'}.
    """
    from scipy.signal import find_peaks
    if method not in ('half_decay', 'curve_fit'):
        raise ValueError("method must be 'half_decay' or 'curve_fit', got %r" % method)
    _curve_fit = None
    if method == 'curve_fit':
        from scipy.optimize import curve_fit as _curve_fit

    def _exp_decay(tt, amp, tau, base):
        return amp * np.exp(-tt / tau) + base

    dff = np.atleast_2d(np.asarray(dff, float))
    n_roi, n_t = dff.shape
    sig = noise_sigma(dff, framerate, sigma_method, isi_mask)
    iso = max(1, int(round(isolation_sec * framerate)))
    win = max(2, int(round(decay_sec * framerate)))
    good = None if isi_mask is None else np.asarray(isi_mask, bool)
    ln2 = np.log(2.0)

    taus, tau_rois = [], []
    for r in (range(n_roi) if max_rois is None else range(min(n_roi, max_rois))):
        s = sig[r]
        if not np.isfinite(s) or s <= 0:
            continue
        x = dff[r]
        peaks, _ = find_peaks(x, height=min_amp_sigma * s, distance=iso)
        if peaks.size == 0:
            continue
        if max_peaks_per_roi and peaks.size > max_peaks_per_roi:
            peaks = peaks[np.argsort(x[peaks])[::-1][:max_peaks_per_roi]]   # keep the largest-amplitude events
        for p in peaks:
            if p < iso:
                continue
            end = min(p + win, n_t)
            if good is not None:                                  # cut the window at the first non-spontaneous frame
                bad = np.flatnonzero(~good[p:end])
                if bad.size:
                    end = p + bad[0]
            if end - p < 2:
                continue
            base = float(np.percentile(x[max(0, p - 3 * win):p + 1], 20))   # pre-peak resting baseline
            amp = x[p] - base
            if amp <= min_amp_sigma * s:
                continue
            if method == 'half_decay':
                d = x[p:end] - base                                        # first 50%-of-amplitude crossing
                below = np.flatnonzero(d <= 0.5 * amp)
                if below.size == 0 or below[0] == 0:
                    continue
                j = below[0]
                y0, y1 = d[j - 1], d[j]                                     # interpolate the crossing time
                frac = (y0 - 0.5 * amp) / (y0 - y1) if y0 != y1 else 0.0
                tau = ((j - 1 + frac) / framerate) / ln2                    # t_half -> tau (single-exponential)
            else:                                                          # 'curve_fit': A*exp(-t/tau)+b
                tt = np.arange(end - p) / framerate
                seg = x[p:end]
                try:
                    popt, _ = _curve_fit(_exp_decay, tt, seg, p0=[amp, p0_tau, base],
                                         bounds=([0.0, tau_bounds[0], -np.inf], [np.inf, tau_bounds[1], np.inf]),
                                         maxfev=3000)
                except Exception:
                    continue
                sst = float(np.sum((seg - seg.mean()) ** 2))
                r2 = 1.0 - float(np.sum((seg - _exp_decay(tt, *popt)) ** 2)) / sst if sst > 0 else 0.0
                if r2 < min_r2:
                    continue
                tau = float(popt[1])
            if tau_bounds[0] < tau < tau_bounds[1]:
                taus.append(float(tau)); tau_rois.append(r)
    taus = np.array(taus)
    if taus.size == 0:
        return {'tau_median': float('nan'), 'tau_iqr': (float('nan'), float('nan')),
                't_half_median': float('nan'), 'n_events': 0,
                'per_event_tau': taus, 'per_event_roi': np.array(tau_rois, int)}
    med = float(np.median(taus))
    return {'tau_median': med, 'tau_iqr': (float(np.percentile(taus, 25)), float(np.percentile(taus, 75))),
            't_half_median': med * ln2, 'n_events': int(taus.size),
            'per_event_tau': taus, 'per_event_roi': np.array(tau_rois, int)}
