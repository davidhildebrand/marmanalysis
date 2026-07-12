#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Response-level denoising of the trial response table.

``denoise_psn`` -- Partitioning Signal and Noise (Kay et al. 2025, GSN-based): a linear, covariance-based
subspace denoiser of the (roi x condition x repeat) response tensor. It estimates signal-vs-noise
covariance across ROIs from the trial repeats, projects onto the signal subspace, and returns a denoised
response. This is a RESPONSE-level denoiser (post-extraction), complementary to movie-level denoisers
(DeepCAD / PMD / SUPPORT -> future ``denoise_*`` siblings) and to spike deconvolution. It reduces
trial-to-trial noise while preserving the across-condition signal, so downstream tuning statistics
(anova_responsive / anova_selective, d'/FSI, RSA) see a cleaner response matrix. Landscape + validation
guidance: literature/reports/topics/cluster_200_denoising_methods.md.

The learned denoiser is applied PER TRIAL and only the stimulus-window response is replaced; the ISI
baseline frames are kept raw, so a stim-vs-baseline responsiveness test compares the denoised response
against the measured baseline. PSN denoises the trial-scalar response, not the intra-trial time course, so
the returned stim-window frames are flat (the per-trial denoised value) -- correct for the scalar tuning
statistics, not a denoised movie.

CAVEAT (from that survey): a denoiser is NOT neutral for dF/F-derived scalars -- always compare
responsive / selective / OSI / DSI / d' before vs after (see compare_denoising.py) and confirm amplitude
fidelity rather than assuming it.

Kay et al. 2025 PLoS Comput Biol https://doi.org/10.1371/journal.pcbi.1012092
"""
import os
from datetime import datetime, timezone
from warnings import warn

import numpy as np
import xarray as xr

from response_table import trial_response, EPOCH_STIM

# PSN named modes (see the PSN README / psn() docstring). 'conservative' = 99% signal-variance retention.
PSN_MODES = ('conservative', 'standard', 'aggressive', 'compare', 'wiener')


def _utc_stamp():
    """David's session-style timestamp: yyyymmdd'd'HHMMSS'tUTC' (e.g. 20260709d224430tUTC)."""
    return datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')


def _apply_denoiser_per_trial(trials, denoiser, unit_means):
    """Apply the PSN denoiser to every trial: ``denoised[:, c, t] = denoiser.T @ (x - mu) + mu``.

    ``trials`` is (n_units, n_cond, n_rep); ``denoiser`` (n_units, n_units); ``unit_means`` (n_units,).
    Linear, so the trial mean of the result equals PSN's ``denoiseddata`` (asserted in the tests). Assumes
    complete trials -- a NaN in any unit at (c, t) propagates across units for that trial (see warning in
    ``denoise_psn``)."""
    mu = np.asarray(unit_means, float).reshape(-1, 1, 1)
    denoised = np.einsum('ji,jcr->icr', denoiser, trials - mu) + mu   # denoiser.T @ centered, per (cond, rep)
    return denoised


def _inject_denoised(ds, metric, denoised_trials):
    """Return a copy of ``ds`` whose ``metric`` STIM-window frames hold ``denoised_trials`` (roi, condition,
    repeat), broadcast across the stim frames, with ISI frames unchanged and the original dim order kept."""
    ds2 = ds.copy()
    ds2[metric] = xr.where(ds['epoch'] == EPOCH_STIM, denoised_trials, ds[metric]).transpose(*ds[metric].dims)
    return ds2


def _ncsnr(svnv):
    """Per-ROI noise-ceiling SNR = sqrt(signal_var / noise_var) from PSN's (n_units, 2) [signal, noise]."""
    svnv = np.asarray(svnv, float)
    return np.sqrt(np.clip(svnv[:, 0], 0, None) / np.clip(svnv[:, 1], 1e-12, None))


def denoise_psn(ds, metric='Fzsc', mode='conservative', diagnostic=True, outdir='output', tag=None):
    """Response-level PSN denoise of ``ds``'s (roi x condition x repeat) trial responses for ``metric``.

    Runs PSN on the per-trial stim-window response tensor, applies the learned denoiser to every trial,
    and returns a COPY of ``ds`` whose ``metric`` stimulus-window frames hold the denoised per-trial
    response (ISI frames unchanged) -- so the existing gates (``anova_responsive`` / ``anova_selective``)
    and ``stim_window_response`` run on it unchanged -- plus a diagnostics dict.

    Parameters
    ----------
    ds : xarray.Dataset
        Response table (see ``response_table.build_response_table``).
    metric : str
        Metric variable to denoise (default 'Fzsc').
    mode : {'conservative', 'standard', 'aggressive', 'compare', 'wiener'}
        PSN operating point; 'conservative' (99% signal-variance retention) is the safe default.
    diagnostic : bool, default True
        If True, generate PSN's diagnostic figure and save it (Agg, UTC-stamped) to ``outdir``. If False,
        pass ``wantfig=False`` so PSN skips figure generation entirely.
    outdir : str
        Directory for the diagnostic figure (created if missing; keep it gitignored, e.g. ``output/``).
    tag : str, optional
        Label prepended to the figure filename (e.g. a session id).

    Returns
    -------
    (ds_denoised, info) : (xarray.Dataset, dict)
        ``info`` keys: ``mode``, ``metric``, ``n_signal_dims`` (retained), ``ncsnr_before`` /
        ``ncsnr_after`` (per-ROI) and their medians, ``signalvar`` / ``noisevar`` (per-dim),
        ``figure_path`` (or None), and the raw PSN result under ``psn``.
    """
    if mode not in PSN_MODES:
        raise ValueError('mode %r not in %s' % (mode, PSN_MODES))
    if metric not in ds.data_vars:
        raise ValueError('metric %r not a data variable of ds' % metric)

    from psn import psn
    import matplotlib.pyplot as plt

    X = trial_response(ds, metric).transpose('roi', 'condition', 'repeat')
    Xv = np.asarray(X.values, float)                       # (n_units, n_cond, n_rep)
    if not np.isfinite(Xv).all():
        warn('denoise_psn: non-finite trials present (excluded/uneven); per-trial denoiser reconstruction '
             'assumes complete trials, so affected (condition, repeat) columns will be NaN.', stacklevel=2)

    opt = {'wantfig': bool(diagnostic), 'wantverbose': False}
    was_interactive = plt.isinteractive()
    plt.ioff()
    _real_show, plt.show = plt.show, (lambda *a, **k: None)  # PSN calls plt.show(); neutralize (no window/block)
    fignums_before = set(plt.get_fignums())
    try:
        result = psn(Xv, mode, opt)
    finally:
        plt.show = _real_show
        if was_interactive:
            plt.ion()

    figure_path = None
    new_figs = [n for n in plt.get_fignums() if n not in fignums_before]
    if diagnostic and new_figs:
        os.makedirs(outdir, exist_ok=True)
        figure_path = os.path.join(outdir, 'denoise_psn_%s%s_%s.png'
                                   % ((tag + '_') if tag else '', mode, _utc_stamp()))
        plt.figure(new_figs[-1]).savefig(figure_path, dpi=150, bbox_inches='tight')
    for n in new_figs:
        plt.close(n)

    Xd = _apply_denoiser_per_trial(Xv, result['denoiser'], result['unit_means'])
    Xd = xr.DataArray(Xd, dims=('roi', 'condition', 'repeat'), coords=X.coords)

    ds_denoised = _inject_denoised(ds, metric, Xd)

    info = {
        'mode': mode, 'metric': metric,
        'n_signal_dims': result['best_threshold'],
        'ncsnr_before': _ncsnr(result['svnv_before']),
        'ncsnr_after': _ncsnr(result['svnv_after']),
        'ncsnr_before_median': float(np.median(_ncsnr(result['svnv_before']))),
        'ncsnr_after_median': float(np.median(_ncsnr(result['svnv_after']))),
        'signalvar': result['signalvar'], 'noisevar': result['noisevar'],
        'figure_path': figure_path, 'psn': result,
    }
    return ds_denoised, info


def denoise_psn_xval(ds, metric='Fzsc', mode='conservative', n_folds=5):
    """Cross-validated PSN denoise -- the HONEST version for downstream significance gates.

    In-sample ``denoise_psn`` drives within-condition trial scatter toward zero, and that scatter IS the
    ANOVA error term, so every significance test inflates (the ``selective -> all cells`` artifact). Here the
    denoiser for each trial is learned on OTHER trials: the repeats are split into ``n_folds`` folds; for
    each fold PSN is fit on the remaining folds and applied to the held-out repeats. Every trial is denoised
    OUT-OF-FOLD, so within-condition scatter is reduced but not self-collapsed, and gate counts computed on
    the result are trustworthy. Returns ``(ds_denoised, info)`` with ``info['fold_signal_dims']`` (dims
    retained per fold). Only stim-window frames are replaced (ISI kept raw), as in ``denoise_psn``.
    """
    if mode not in PSN_MODES:
        raise ValueError('mode %r not in %s' % (mode, PSN_MODES))
    if metric not in ds.data_vars:
        raise ValueError('metric %r not a data variable of ds' % metric)
    from psn import psn

    X = trial_response(ds, metric).transpose('roi', 'condition', 'repeat')
    Xv = np.asarray(X.values, float)
    n_rep = Xv.shape[2]
    n_folds = int(min(n_folds, n_rep))
    if n_folds < 2:
        raise ValueError('cross-validated denoising needs >=2 repeats')
    Xd = np.full_like(Xv, np.nan)
    fold_signal_dims = []
    for test_idx in np.array_split(np.arange(n_rep), n_folds):
        train_idx = np.setdiff1d(np.arange(n_rep), test_idx)
        if train_idx.size < 2:
            raise ValueError('n_folds=%d leaves <2 training repeats (n_rep=%d)' % (n_folds, n_rep))
        res = psn(Xv[:, :, train_idx], mode, {'wantfig': False, 'wantverbose': False})
        Xd[:, :, test_idx] = _apply_denoiser_per_trial(Xv[:, :, test_idx], res['denoiser'], res['unit_means'])
        fold_signal_dims.append(res['best_threshold'])
    Xd = xr.DataArray(Xd, dims=('roi', 'condition', 'repeat'), coords=X.coords)
    ds_denoised = _inject_denoised(ds, metric, Xd)
    info = {'mode': mode, 'metric': metric, 'n_folds': n_folds, 'fold_signal_dims': fold_signal_dims}
    return ds_denoised, info


def denoise_psn_xval(ds, metric='Fzsc', mode='conservative', n_folds=5, seed=0):
    """Cross-validated PSN denoise: each trial is denoised by a denoiser fit on the OTHER trials.

    Splits the repeats into ``n_folds`` folds; for each fold, fits PSN on the out-of-fold (train) trials
    and applies that denoiser to the held-out (test) trials. Because no trial is denoised by a denoiser
    that saw it, the within-condition trial scatter is NOT self-collapsed -- so recomputing the trial-level
    ANOVA gates (anova_responsive / anova_selective) on the result is statistically honest, unlike the
    in-sample ``denoise_psn`` which shrinks the ANOVA error term and spuriously inflates selectivity toward
    ~100%. Returns a COPY of ``ds`` with the ``metric`` stim-window frames replaced by the out-of-fold
    denoised per-trial response (ISI kept raw) and an info dict. No diagnostic figure (one per fold is noise).

    Parameters are as ``denoise_psn`` plus ``n_folds`` (2..n_repeats) and ``seed`` (fold assignment).
    """
    if mode not in PSN_MODES:
        raise ValueError('mode %r not in %s' % (mode, PSN_MODES))
    if metric not in ds.data_vars:
        raise ValueError('metric %r not a data variable of ds' % metric)
    from psn import psn

    X = trial_response(ds, metric).transpose('roi', 'condition', 'repeat')
    Xv = np.asarray(X.values, float)                       # (n_units, n_cond, n_rep)
    n_rep = Xv.shape[2]
    if not 2 <= n_folds <= n_rep:
        raise ValueError('n_folds must be in [2, n_repeats=%d], got %d' % (n_rep, n_folds))

    fold = np.random.default_rng(seed).permutation(n_rep) % n_folds
    Xd = np.full_like(Xv, np.nan)
    fold_dims = []
    for k in range(n_folds):
        train = np.where(fold != k)[0]
        test = np.where(fold == k)[0]
        if train.size < 2 or test.size == 0:
            continue
        res = psn(Xv[:, :, train], mode, {'wantfig': False, 'wantverbose': False})
        Xd[:, :, test] = _apply_denoiser_per_trial(Xv[:, :, test], res['denoiser'], res['unit_means'])
        fold_dims.append(int(np.ravel(res['best_threshold'])[0]))

    Xd = xr.DataArray(Xd, dims=('roi', 'condition', 'repeat'), coords=X.coords)
    ds_denoised = ds.copy()
    ds_denoised[metric] = xr.where(ds['epoch'] == EPOCH_STIM, Xd, ds[metric]).transpose(*ds[metric].dims)
    info = {'mode': mode, 'metric': metric, 'n_folds': n_folds, 'fold_signal_dims': fold_dims}
    return ds_denoised, info
