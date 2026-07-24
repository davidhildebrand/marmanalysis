#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cortical topography: does tuning similarity fall off with cortical distance, and does that spatial structure
match the SOM's? (SOM plan main-track step 4; artifact controls = roadmap item 12.)

Core object is the tuning-similarity-vs-distance relationship over ROI pairs: similarity = correlation of two
ROIs' condition-response profiles, distance = microns between their FOV centroids. A topographic map has
NEARBY cells MORE similar. Summarised two ways: a scalar ``topography_score`` (Spearman of similarity vs
NEGATIVE distance; positive = topographic) and an exponential length constant lambda (the distance over which
similarity decays).

Two tuning inputs are analysed in parallel (DH 2026-07-23): the FULL response, and the CATEGORY-RESIDUAL
response (per-category mean removed) which isolates the finer-than-category structure that step 3b showed is
the only part not already explained by gross category.

NULLS / CONTROLS (a topographic-looking result can be an artifact):
  * ``position_shuffle_null`` -- permute which tuning profile sits at which ROI position. Standard but WEAK:
    short-range contamination (neuropil bleed) survives it, so it only rules out "no spatial structure at all".
  * ``nuisance_topography`` -- run the identical score on a per-cell NUISANCE scalar (nu-SNR, F0). If the
    nuisance is itself spatially clustered and scores like the tuning, the tuning topography may be nuisance.
  * variogram-matched surrogates (roadmap 12c) are the rigorous autocorrelation-preserving null -- TODO, added
    as a separate function once this pass is validated.
"""
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def pairwise_tuning_similarity(response):
    """(n_roi x n_roi) tuning similarity = Pearson correlation between ROIs' condition-response profiles.
    ``response`` is (n_roi x n_condition)."""
    return np.corrcoef(np.asarray(response, float))


def pairwise_distance(xy_um):
    """(n x n) Euclidean cortical distance (microns) from (n x 2) ROI centroids."""
    return squareform(pdist(np.asarray(xy_um, float)))


def residualize_category(response, categories):
    """Per-category-mean-removed response: subtract, within each category, that category's mean response vector
    from every ROI's profile -> the FINER-than-category tuning. ``categories`` is one label per CONDITION."""
    R = np.asarray(response, float).copy()
    cats = np.asarray([str(c) for c in categories])
    for c in np.unique(cats):
        m = cats == c
        R[:, m] = R[:, m] - R[:, m].mean(axis=1, keepdims=True)
    return R


def _pairs(sim, dist, max_dist=None):
    iu = np.triu_indices_from(sim, k=1)
    s, d = sim[iu], dist[iu]
    ok = np.isfinite(s) & np.isfinite(d)
    if max_dist is not None:
        ok &= d <= max_dist
    return s[ok], d[ok]


def topography_score(sim, dist, max_dist=None):
    """Scalar topography index: Spearman correlation between pairwise tuning similarity and NEGATIVE distance
    over ROI pairs. Positive => nearby ROIs more similar (topographic); ~0 => no spatial organisation."""
    s, d = _pairs(sim, dist, max_dist)
    return float(spearmanr(s, -d).statistic) if s.size > 2 else np.nan


def similarity_vs_distance(sim, dist, n_bins=15, max_dist=None):
    """Binned tuning similarity vs distance. Returns (centers, mean, sem, count) per distance bin."""
    s, d = _pairs(sim, dist, max_dist)
    edges = np.linspace(d.min(), d.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan); sem = np.full(n_bins, np.nan); cnt = np.zeros(n_bins, int)
    for i in range(n_bins):
        b = (d >= edges[i]) & (d < edges[i + 1] if i < n_bins - 1 else d <= edges[i + 1])
        if b.any():
            mean[i] = s[b].mean(); sem[i] = s[b].std() / np.sqrt(b.sum()); cnt[i] = int(b.sum())
    return centers, mean, sem, cnt


def fit_length_constant(sim, dist, max_dist=None):
    """Fit similarity(d) = a*exp(-d/lambda) + b; return {'lambda_um', 'a', 'b', 'ok'}. lambda = the distance
    over which similarity decays by 1/e. Falls back to ok=False if the fit does not converge."""
    from scipy.optimize import curve_fit
    s, d = _pairs(sim, dist, max_dist)
    try:
        popt, _ = curve_fit(lambda x, a, lam, b: a * np.exp(-x / lam) + b, d, s,
                            p0=[max(s.max() - s.min(), 0.05), max(np.median(d), 50.0), np.median(s)],
                            bounds=([0, 1.0, -1.0], [3.0, 1e4, 1.0]), maxfev=8000)
        return {'lambda_um': float(popt[1]), 'a': float(popt[0]), 'b': float(popt[2]), 'ok': True}
    except Exception:
        return {'lambda_um': np.nan, 'a': np.nan, 'b': np.nan, 'ok': False}


def position_shuffle_null(sim, dist, score=None, n_perm=1000, seed=0, max_dist=None):
    """Null for ``topography_score`` by permuting ROI POSITION labels (destroys tuning<->position pairing while
    preserving both the tuning-similarity matrix and the spatial point pattern). One-sided p (H1: positive).
    NB weak: short-range artifact beats it -- see the module docstring. Returns {'score','p','null_mean'}."""
    n = sim.shape[0]
    obs = topography_score(sim, dist, max_dist) if score is None else score
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = topography_score(sim, dist[np.ix_(p, p)], max_dist)
    return {'score': float(obs), 'p': (1 + int(np.sum(null >= obs))) / (1 + n_perm),
            'null_mean': float(np.nanmean(null)), 'n_perm': int(n_perm)}


def nuisance_similarity(scalar):
    """(n x n) 'similarity' of a per-cell scalar nuisance = negative absolute difference (closer values ->
    higher). Feed to ``topography_score`` to ask whether the nuisance is itself spatially clustered."""
    v = np.asarray(scalar, float)
    return -np.abs(v[:, None] - v[None, :])


def plot_topography(curves, outdir='output', tag=None, max_dist=None):
    """Plot tuning-similarity-vs-distance curves. ``curves`` = list of (label, centers, mean, sem). Returns the
    saved figure path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for lab, c, m, e in curves:
        ax.plot(c, m, marker='o', ms=3, lw=1.6, label=lab)
        ax.fill_between(c, m - e, m + e, alpha=0.15)
    ax.axhline(0, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('cortical distance (µm)'); ax.set_ylabel('tuning similarity (Pearson r)')
    if max_dist:
        ax.set_xlim(0, max_dist)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('tuning similarity vs cortical distance — %s' % (tag or 'session'))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'topography_%s_%s.png'
                     % (tag or 'session', datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p
