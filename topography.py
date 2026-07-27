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
    short-range contamination survives it, so it only rules out "no spatial structure at all".
  * ``nuisance_similarity`` -- build a per-cell NUISANCE-scalar similarity (nu-SNR, F0) and score it; if the
    nuisance is itself spatially clustered and scores like the tuning, the tuning topography may be nuisance.
  * ROI-OVERLAP controls (chunk 1) -- ``min_dist`` drops the shortest pairs, and ``roi_shared_pixel_pairs`` +
    ``exclude`` drops pairs whose masks actually touch/overlap: rules out touching-ROI / segmentation bleed.
  * ``subtract_local_mean`` (chunk 1) -- remove each ROI's local-neighbourhood mean response before scoring:
    rules out a spatially-shared ADDITIVE signal (residual neuropil / hemodynamic / vascular).
  * variogram-matched surrogates (roadmap 12c) = the rigorous autocorrelation-preserving null; the
    spatial-frequency / template detector (roadmap 12b) = the frequency-resolved alternative -- chunks 2 and 3.
NB the indicator is soma-enriched ribo-L1-jGCaMP8s, so the NEUROPIL prior is a priori weak; these controls
mainly guard the OTHER short-range artifacts (mask overlap, vascular, motion). See analysis_roadmap.md item 12.
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


def _pairs(sim, dist, max_dist=None, min_dist=None, exclude=None):
    """Upper-triangle (similarity, distance) pairs. ``min_dist`` drops the SHORTEST pairs (an ROI-overlap /
    short-range-artifact control -- if topography survives with the nearest pairs gone, touching ROIs are not
    driving it); ``exclude`` is an (n x n) boolean (True = drop a specific pair, e.g. overlapping ROI masks)."""
    iu = np.triu_indices_from(sim, k=1)
    s, d = sim[iu], dist[iu]
    ok = np.isfinite(s) & np.isfinite(d)
    if max_dist is not None:
        ok &= d <= max_dist
    if min_dist is not None:
        ok &= d >= min_dist
    if exclude is not None:
        ok &= ~exclude[iu].astype(bool)
    return s[ok], d[ok]


def topography_score(sim, dist, max_dist=None, min_dist=None, exclude=None):
    """Scalar topography index: Spearman correlation between pairwise tuning similarity and NEGATIVE distance
    over ROI pairs. Positive => nearby ROIs more similar (topographic); ~0 => no spatial organisation.
    ``min_dist``/``exclude`` support the ROI-overlap controls (see ``_pairs``)."""
    s, d = _pairs(sim, dist, max_dist, min_dist, exclude)
    return float(spearmanr(s, -d).statistic) if s.size > 2 else np.nan


def similarity_vs_distance(sim, dist, n_bins=15, max_dist=None, min_dist=None, exclude=None):
    """Binned tuning similarity vs distance. Returns (centers, mean, sem, count) per distance bin."""
    s, d = _pairs(sim, dist, max_dist, min_dist, exclude)
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


def position_shuffle_null(sim, dist, score=None, n_perm=1000, seed=0, max_dist=None, min_dist=None, exclude=None):
    """Null for ``topography_score`` by permuting ROI POSITION labels (destroys tuning<->position pairing while
    preserving both the tuning-similarity matrix and the spatial point pattern). One-sided p (H1: positive).
    NB weak: short-range artifact beats it -- see the module docstring. Returns {'score','p','null_mean'}."""
    n = sim.shape[0]
    obs = topography_score(sim, dist, max_dist, min_dist, exclude) if score is None else score
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        ex = None if exclude is None else exclude[np.ix_(p, p)]     # overlap is a position property -> permute it
        null[k] = topography_score(sim, dist[np.ix_(p, p)], max_dist, min_dist, ex)
    return {'score': float(obs), 'p': (1 + int(np.sum(null >= obs))) / (1 + n_perm),
            'null_mean': float(np.nanmean(null)), 'n_perm': int(n_perm)}


def _matched_correlation(sim, dist, max_dist=None):
    """Fit ``corr(d) ~ a*exp(-d/lam)`` to the observed tuning-similarity-vs-distance and return (a, lam): ``a`` in
    [0,1] = the spatially-STRUCTURED fraction (the amplitude of the decaying part), ``lam`` = the range, capped to
    [1 um, half the 95th-pct pairwise distance] (a range beyond ~half the map extent is unidentifiable).

    Matching the AMPLITUDE, not just the range, is what makes the surrogate null honest. Real tuning similarity is
    mostly per-cell noise (small ``a`` ~ 0.05-0.2), so a fixed large structured fraction produces a surrogate map
    far cleaner than the data -- its topography sits well above the real value and forces p=1, and its spectrum is
    too smooth so even random data looks 'peaky'. Fitting ``a`` from the data calibrates both nulls (random -> a~0
    -> C~I -> surrogate ~ the data)."""
    n = dist.shape[0]
    du = dist[np.triu_indices(n, 1)]
    fit = fit_length_constant(sim, dist, max_dist=max_dist)
    lam = fit['lambda_um'] if (fit['ok'] and fit['lambda_um'] > 0) else float(np.nanmedian(du))
    lam = float(np.clip(lam, 1.0, 0.5 * float(np.nanpercentile(du, 95))))
    a = float(np.clip(fit['a'] if fit['ok'] else 0.0, 0.0, 1.0))
    return a, lam


def _matched_field_operator(dist, a, lam):
    """Factor L (via ``eigh``) of the amplitude-matched correlation matrix ``C = a*exp(-D/lam) + (1-a)*I``
    (diagonal 1, off-diagonal ``a*exp(-D/lam)``), so a surrogate field is ``L @ z`` for standard-normal z. The
    ``(1-a)*I`` term is the nugget = the per-cell-noise fraction estimated from the data, so the surrogate carries
    the SAME autocorrelation strength as the observed map rather than an arbitrarily clean one."""
    C = a * np.exp(-dist / lam) + (1.0 - a) * np.eye(dist.shape[0])
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None))


def variogram_matched_null(response, xy_um, score=None, n_perm=1000, seed=0, lam=None,
                           max_dist=None, min_dist=None, exclude=None):
    """Autocorrelation-PRESERVING null for ``topography_score`` (roadmap 12c) -- the rigorous upgrade of
    ``position_shuffle_null``.

    ``position_shuffle_null`` destroys ALL spatial structure, so any smoothness at all (real tuning OR a smooth
    nuisance) beats it -- it only rejects "no spatial organisation whatsoever". This null instead generates
    surrogate response maps that carry the SAME spatial autocorrelation as the data but NO coordinated tuning:
    each condition's map is an independent Gaussian random field with a matched-autocorrelation covariogram
    ``C_ij = a*exp(-D_ij/lam) + (1-a)*delta_ij`` where BOTH the structured amplitude ``a`` and the range ``lam``
    are fit from the data's similarity-vs-distance (``_matched_correlation``) -- so the surrogate carries the
    data's REAL (typically weak) autocorrelation, not an arbitrarily clean one -- rescaled to that condition's
    spatial variance. Because the conditions are drawn INDEPENDENTLY, nearby ROIs are still
    similar WITHIN each condition (smoothness preserved) but the across-condition tuning PROFILE similarity is
    only what that per-condition smoothness produces by chance -- there is no cell-specific co-tuning.

    So the test is: does the observed topography exceed what a map with matched per-condition smoothness gives?
      * p < 0.05  -> the tuning map is MORE spatially organised than matched-smoothness fields: evidence for
        genuine coordinated (cell-specific) tuning topography beyond generic smoothness.
      * p ~ 0.5   -> the topography is consistent with generic spatial smoothness of the per-condition responses
        (a common, appropriately-humbling outcome for a weak residual) -- NOT evidence of coordinated tuning.
    A smooth SHARED nuisance (e.g. residual neuropil, vascular) inflates each condition's autocorrelation, so it
    is absorbed into ``lam`` and the null rather than mistaken for tuning -- which is the point.

    ``lam`` defaults to ``fit_length_constant`` on the observed similarity-vs-distance (falls back to the median
    pairwise distance if that fit fails). ``max_dist``/``min_dist``/``exclude`` are applied to the SCORE exactly
    as in ``topography_score`` (the fields themselves are generated over all ROIs). Returns
    {'score','p','null_mean','null_std','lambda_um','n_perm'}."""
    resp = np.asarray(response, float)
    n, ncond = resp.shape
    dist = pairwise_distance(xy_um)
    sim = pairwise_tuning_similarity(resp)
    obs = topography_score(sim, dist, max_dist, min_dist, exclude) if score is None else float(score)
    a, lam_fit = _matched_correlation(sim, dist, max_dist=max_dist)
    lam = lam_fit if lam is None else float(lam)
    L = _matched_field_operator(dist, a, lam)                 # amplitude+range-matched; cheap draws = L @ z
    col_std = np.nanstd(resp, axis=0, keepdims=True)          # match each condition's spatial variance
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        S = L @ rng.standard_normal((n, ncond))
        S = (S - S.mean(0)) / (S.std(0) + 1e-12) * col_std
        null[k] = topography_score(pairwise_tuning_similarity(S), dist, max_dist, min_dist, exclude)
    null = null[np.isfinite(null)]
    return {'score': obs, 'p': ((1 + int(np.sum(null >= obs))) / (1 + null.size)) if null.size else np.nan,
            'null_mean': float(np.nanmean(null)) if null.size else np.nan,
            'null_std': float(np.nanstd(null)) if null.size else np.nan,
            'amplitude': float(a), 'lambda_um': float(lam), 'n_perm': int(null.size)}


def roi_shared_pixel_pairs(rois):
    """(n x n) boolean: True where two ROI masks share >= 1 pixel -- adjacent/overlapping segmentation, a
    short-range contamination source that a distance cut alone may miss. ``rois`` is the suite2p stat array
    (each entry has integer 'xpix'/'ypix'). Also returns the count via ``.sum()//2``."""
    from collections import defaultdict
    owners = defaultdict(list)
    for i, r in enumerate(rois):
        for px in zip(np.asarray(r['ypix']).tolist(), np.asarray(r['xpix']).tolist()):
            owners[px].append(i)
    n = len(rois)
    ov = np.zeros((n, n), bool)
    for ids in owners.values():
        if len(ids) > 1:
            u = np.unique(ids)
            ov[np.ix_(u, u)] = True
    np.fill_diagonal(ov, False)
    return ov


def subtract_local_mean(response, xy_um, radius_um):
    """Remove each ROI's LOCAL-NEIGHBOURHOOD mean response (mean over ROIs within ``radius_um``, self excluded).

    CAVEAT (verified empirically 2026-07-24): this control is CONFOUNDED for topography detection and does NOT
    discriminate real from artifact. Topography = nearby ROIs being similar, so the local mean CARRIES the
    topographic signal; subtracting it removes real functional structure and shared-neuropil contamination
    ALIKE (score 0.11 -> -0.01 here, whether the signal is real or not). For a non-circular neuropil test use
    the Fneu control (run this topography on the neuropil signal; roadmap #13) + ``roi_shared_pixel_pairs`` +
    ``min_dist``. Kept for the record.
    ``response`` is (n_roi x n_condition), ``xy_um`` (n_roi x 2)."""
    R = np.asarray(response, float)
    D = squareform(pdist(np.asarray(xy_um, float)))
    idx = np.arange(R.shape[0])
    out = R.copy()
    for i in range(R.shape[0]):
        nb = (D[i] <= radius_um) & (idx != i)
        if nb.any():
            out[i] = R[i] - R[nb].mean(axis=0)
    return out


def nuisance_similarity(scalar):
    """(n x n) 'similarity' of a per-cell scalar nuisance = negative absolute difference (closer values ->
    higher). Feed to ``topography_score`` to ask whether the nuisance is itself spatially clustered."""
    v = np.asarray(scalar, float)
    return -np.abs(v[:, None] - v[None, :])


def rasterize_tuning(response, xy_um, n_grid=32, n_pc=3):
    """Rasterise the top-``n_pc`` principal components of the (n_roi x n_condition) tuning onto an
    ``n_grid`` x ``n_grid`` FOV grid (per-bin mean of the PC score; empty bins = NaN). Returns
    (maps [n_pc, n_grid, n_grid], explained_var [n_pc], bin_um). The PCs compress the multivariate tuning to the
    few scalar spatial fields whose spatial-frequency content the template detector then examines. Assumes a
    roughly square FOV (bin_um is the mean of the x/y extents over the grid)."""
    R = np.nan_to_num(np.asarray(response, float) - np.nanmean(response, axis=0, keepdims=True))
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    K = int(min(n_pc, s.size))
    scores = U[:, :K] * s[:K]                                # (n_roi, K) PC scores
    ev = (s[:K] ** 2) / (s ** 2).sum()
    xy = np.asarray(xy_um, float)

    def binidx(a):
        lo, hi = a.min(), a.max()
        return np.clip(((a - lo) / ((hi - lo) or 1.0) * (n_grid - 1)).round().astype(int), 0, n_grid - 1)

    xi, yi = binidx(xy[:, 0]), binidx(xy[:, 1])
    maps = np.full((K, n_grid, n_grid), np.nan)
    for k in range(K):
        acc = np.zeros((n_grid, n_grid)); cnt = np.zeros((n_grid, n_grid))
        np.add.at(acc, (yi, xi), scores[:, k]); np.add.at(cnt, (yi, xi), 1.0)
        maps[k] = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
    extent = 0.5 * ((xy[:, 0].max() - xy[:, 0].min()) + (xy[:, 1].max() - xy[:, 1].min()))
    return maps, ev, float(extent / (n_grid - 1) if n_grid > 1 else 1.0)


def radial_power_spectrum(map2d, bin_um, n_rbins=None):
    """Radially-averaged 2D power spectrum of a rasterised map. Empty (NaN) bins are mean-filled to 0 (do the
    SAME to the null maps so the comparison is fair). Returns (freq_cyc_per_um, power) with DC dropped. A map
    with a characteristic scale (patches/domains) shows a PEAK at nonzero frequency; a monotonic-decay map does
    not."""
    m = np.array(map2d, float)
    ok = np.isfinite(m)
    m = m - (m[ok].mean() if ok.any() else 0.0)
    m[~ok] = 0.0
    P = np.abs(np.fft.fftshift(np.fft.fft2(m))) ** 2
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    n_rbins = int(n_rbins or rmax)
    edges = np.linspace(0, rmax, n_rbins + 1)
    freq, power = [], []
    for i in range(n_rbins):
        b = (r >= edges[i]) & ((r < edges[i + 1]) if i < n_rbins - 1 else (r <= edges[i + 1]))
        if b.any():
            rc = 0.5 * (edges[i] + edges[i + 1])
            freq.append(rc / nx / bin_um)                   # (cycles per pixel) / (um per pixel) = cycles/um
            power.append(float(P[b].mean()))
    freq, power = np.asarray(freq), np.asarray(power)
    keep = freq > 0                                          # drop DC
    return freq[keep], power[keep]


def spatial_frequency_detector(response, xy_um, n_grid=32, n_pc=3, n_perm=500, seed=0, lam=None,
                               min_wavelength_um=None):
    """Template / spatial-frequency detector (roadmap 12b): is the tuning map organised at a CHARACTERISTIC
    SCALE (patchy/columnar/domain structure), which the monotonic ``topography_score`` cannot see?

    Rasterise each of the top ``n_pc`` tuning PCs (``rasterize_tuning``), take its radial power spectrum, and
    test for EXCESS power at a nonzero spatial frequency versus variogram-matched surrogates -- Gaussian random
    fields on the SAME ROI positions with the matched autocorrelation ``lam`` (from the tuning-similarity length
    constant), rasterised identically. Matched-autocorrelation fields have monotonically-decaying spectra with no
    intrinsic scale, so a genuine domain periodicity shows up as a spectral bump above them. Each spectrum is
    power-normalised (shape only, amplitude removed) and the statistic is the MAX over frequency of the observed
    excess z (vs the surrogate mean/SD per frequency); the p-value compares it to the surrogates' own max-excess
    (a max-statistic that controls for scanning many frequency bins). ``min_wavelength_um`` ignores the very
    finest bins (below the ROI sampling, dominated by binning noise).

    Returns a dict per PC index: {'peak_wavelength_um','peak_freq','stat','p','explained_var'} plus 'lambda_um'.
    p >= ~0.05 => no scale beyond matched smoothness (the expected outcome unless real domains exist)."""
    resp = np.asarray(response, float)
    n, ncond = resp.shape
    maps, ev, bin_um = rasterize_tuning(resp, xy_um, n_grid=n_grid, n_pc=n_pc)
    K = maps.shape[0]
    dist = pairwise_distance(xy_um)
    a, lam_fit = _matched_correlation(pairwise_tuning_similarity(resp), dist, max_dist=None)
    lam = lam_fit if lam is None else float(lam)
    L = _matched_field_operator(dist, a, lam)
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy_um, float)

    freq_ref = None
    fmax = (1.0 / min_wavelength_um) if min_wavelength_um else np.inf
    out = {'lambda_um': float(lam)}
    for k in range(K):
        f_obs, p_obs = radial_power_spectrum(maps[k], bin_um)
        if freq_ref is None:
            freq_ref = f_obs
        band = f_obs <= fmax
        obs_norm = p_obs / (p_obs[band].sum() + 1e-12)
        # surrogate spectra: scalar GRF fields matched to lam, this PC's spatial variance, rasterised the same
        null_specs = np.empty((n_perm, f_obs.size))
        sd = float(np.nanstd(maps[k][np.isfinite(maps[k])]) or 1.0)
        for b in range(n_perm):
            g = L @ rng.standard_normal(n)
            g = (g - g.mean()) / (g.std() + 1e-12) * sd
            gm, _, _ = rasterize_tuning(g[:, None], xy, n_grid=n_grid, n_pc=1)
            _, pb = radial_power_spectrum(gm[0], bin_um)
            null_specs[b] = pb / (pb[band].sum() + 1e-12) if pb.size == f_obs.size else np.nan
        mu, sg = np.nanmean(null_specs, 0), np.nanstd(null_specs, 0) + 1e-12
        excess = (obs_norm - mu) / sg
        null_excess_max = np.nanmax(((null_specs - mu) / sg)[:, band], axis=1)
        stat = float(np.nanmax(excess[band]))
        peak = int(np.nanargmax(np.where(band, excess, -np.inf)))
        out[k] = {'peak_wavelength_um': float(1.0 / f_obs[peak]) if f_obs[peak] > 0 else np.inf,
                  'peak_freq': float(f_obs[peak]), 'stat': stat,
                  'p': float((1 + np.sum(null_excess_max >= stat)) / (1 + n_perm)),
                  'explained_var': float(ev[k])}
    return out


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
