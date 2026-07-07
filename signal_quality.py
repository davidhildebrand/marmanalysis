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
# nu k=9.0: an inclusive-but-clean active bar. On the nu scale it is frame-rate INDEPENDENT, so it
# transfers across sessions at different rates as a consistent DETECTABILITY criterion -- a fixed
# sigma-multiple would NOT (per-frame sigma ~ sqrt(framerate), so "3 sigma" is trivially easy when imaging
# slow and too strict when fast). In sigma terms k=9 is ~2.7 sigma at the 6.4 Hz reference session
# (Cadbury PD: ~412/791 active, more inclusive than suite2p's 310) and exactly 3 sigma at ~5.7 Hz -- a
# central choice for a ~5-6 Hz range -- while staying above the pure-noise floor (~2.33 sigma, the 99th-pct
# peak of Gaussian noise). The 3-sigma-equivalent k scales as ~3.8*sqrt(framerate) (e.g. ~8.4 at 5 Hz,
# ~11.9 at 10 Hz). Very slow rates (2-3 Hz) can undersample the transient PEAK (nu fixes the noise term,
# not the peak), biasing the active count low -- active_mask emits a runtime warning below warn_slow_hz
# (default 4 Hz); consider lowering k there. mad/diff/mad_isi k=3 = the plain within-session 3-sigma bar.
_DEFAULT_K = {'nu': 9.0, 'diff': 3.0, 'mad': 3.0, 'mad_isi': 3.0, 'std': 3.0}


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
    defaults per-method (nu ~9, others ~3; see _DEFAULT_K) -- choose it from the SNR distribution for
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
