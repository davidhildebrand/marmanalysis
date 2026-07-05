#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Labeled response-data container for trial-sorted two-photon fluorescence.

This is a prototype replacement for the numpy structured-array ``data`` table used in
``analysis_for_images.py`` / ``analysis_for_dots.py``. It keeps the two properties of the
structured array that are worth keeping:

  1. Logical, labeled access -- here via xarray dimensions (roi, condition, repeat, time)
     and coordinates, instead of positional axes that must be tracked by counting commas.
  2. A completeness invariant -- every cell is expected to hold a real value before any
     trial exclusion (cf. analysis_for_images.py:1663-1667). ``assert_complete`` enforces
     this, so an accidental un-filled cell is caught rather than silently averaged over.

It improves on both the structured array and the dict-of-arrays approach by recording trial
exclusions in an explicit boolean ``excluded`` mask coordinate instead of overwriting the
excluded values with NaN. A NaN in a metric variable therefore never means "excluded" and
never means "forgot to fill" -- both of those are represented explicitly -- which removes the
class of bug where un-overwritten NaNs are mistaken for excluded trials.
"""

import numpy as np
import xarray as xr
from scipy.stats import f_oneway


# Trial structure: a trial is [ISI_pre | stimulus | ISI_post], each ISI being n_samp_isi
# frames and the stimulus being n_samp_stim frames. These labels go on the ``time`` axis.
EPOCH_ISI_PRE = 'isi_pre'
EPOCH_STIM = 'stim'
EPOCH_ISI_POST = 'isi_post'


def trial_epochs(n_samp_isi, n_samp_stim):
    """Per-frame epoch labels for one trial of length n_samp_isi + n_samp_stim + n_samp_isi."""
    return np.array([EPOCH_ISI_PRE] * n_samp_isi
                    + [EPOCH_STIM] * n_samp_stim
                    + [EPOCH_ISI_POST] * n_samp_isi)


def build_response_table(traces, stimlog, n_samp_isi, n_samp_stim,
                         condition_coords=None, framerate=None,
                         acqfr_col='acqfr_stim_i', cond_col='cond'):
    """Sort full-length fluorescence traces into a trial-aligned xarray Dataset.

    Mirrors the windowing in analysis_for_images.py:1604-1667 so results are directly
    comparable to the structured-array table.

    Parameters
    ----------
    traces : dict[str, np.ndarray]
        Metric name -> full-session trace array of shape (n_ROIs, n_frames). Typical keys
        are 'FdFF', 'Fzsc', 'F0', 'Fraw'. Every metric becomes a data variable of the
        returned Dataset with dims (roi, condition, repeat, time).
    stimlog : pandas.DataFrame
        One row per presented trial. Must contain an integer condition column (``cond_col``,
        values 0..n_conds-1) and a stimulus-onset acquisition-frame column (``acqfr_col``).
        Conditions are assumed to have an equal number of repeats (as in the existing code).
    n_samp_isi, n_samp_stim : int
        Frame counts for one ISI and for the stimulus.
    condition_coords : dict[str, array_like], optional
        Per-condition metadata (length n_conds, indexed by condition id), e.g.
        {'cond': [...], 'cat': [...], 'pitch': [...], 'imagename': [...]}. Each becomes a
        coordinate on the ``condition`` dimension so stimulus metadata travels with the data.
    framerate : float, optional
        If given, a ``time_s`` coordinate (seconds relative to stimulus onset) is added.

    Returns
    -------
    xarray.Dataset
        dims (roi, condition, repeat, time); data variables = metrics; coordinates include
        ``epoch`` (per-frame trial-epoch label) and a boolean ``excluded`` (condition, repeat)
        mask initialised to False. The metric variables are guaranteed complete (no NaN) on
        return -- see ``assert_complete``.
    """
    metric_names = list(traces)
    n_ROIs, n_frames = traces[metric_names[0]].shape
    for m in metric_names:
        if traces[m].shape != (n_ROIs, n_frames):
            raise ValueError('All traces must share shape (n_ROIs, n_frames); '
                             '{} has {}.'.format(m, traces[m].shape))

    conditions = np.unique(stimlog[cond_col].values)
    n_conds = conditions.size
    n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi
    counts = stimlog[cond_col].value_counts()
    if counts.nunique() != 1:
        raise ValueError('Conditions have unequal repeat counts {}; ragged repeats are not '
                         'yet supported by this prototype.'.format(dict(counts)))
    n_reps = int(counts.iloc[0])

    # Initialise to NaN so that any cell left unfilled is detectable (the completeness
    # invariant), exactly as np.zeros(...) then data[m] = np.nan does for the structured array.
    arrays = {m: np.full((n_ROIs, n_conds, n_reps, n_samp_trial), np.nan) for m in metric_names}

    for ci, c in enumerate(conditions):
        sub = stimlog[stimlog[cond_col] == c]
        for t in range(n_reps):
            fr_start = sub.iloc[t][acqfr_col] - n_samp_isi
            fr_end = sub.iloc[t][acqfr_col] + n_samp_stim + n_samp_isi
            fr_start = _as_int_frame(fr_start)
            fr_end = _as_int_frame(fr_end)

            if fr_start < 0 and t == 0:
                # Pre-first-trial period shorter than one ISI: pad with the first frame value
                # rather than erroring (matches analysis_for_images.py:1622-1646).
                n_samp_miss = abs(fr_start)
                for m in metric_names:
                    arrays[m][:, ci, t, 0:n_samp_miss] = np.repeat(
                        traces[m][:, 0:1], n_samp_miss, axis=1)
                    arrays[m][:, ci, t, n_samp_miss:n_samp_trial] = traces[m][:, 0:fr_end]
                continue
            if fr_end > n_frames:
                raise RuntimeError('Imaging stopped before stimulus (cond {}, rep {}); '
                                   'discarding trailing trials is not yet implemented.'.format(c, t))
            for m in metric_names:
                arrays[m][:, ci, t, :] = traces[m][:, fr_start:fr_end]

    coords = {
        'roi': np.arange(n_ROIs),
        'condition': conditions,
        'repeat': np.arange(n_reps),
        'time': np.arange(n_samp_trial),
        'epoch': ('time', trial_epochs(n_samp_isi, n_samp_stim)),
    }
    if framerate is not None:
        # Seconds relative to stimulus onset (negative during the pre-stimulus ISI).
        coords['time_s'] = ('time', (np.arange(n_samp_trial) - n_samp_isi) / framerate)
    if condition_coords is not None:
        for name, values in condition_coords.items():
            values = np.asarray(values)
            if values.shape[0] != n_conds:
                raise ValueError('condition_coords[{!r}] has length {}, expected n_conds={}.'
                                 .format(name, values.shape[0], n_conds))
            coords[name] = ('condition', values)

    data_vars = {m: (('roi', 'condition', 'repeat', 'time'), arrays[m]) for m in metric_names}
    ds = xr.Dataset(data_vars=data_vars, coords=coords,
                    attrs={'n_samp_isi': n_samp_isi, 'n_samp_stim': n_samp_stim})
    # Explicit exclusion mask: excluded trials are recorded here, never NaN-ed in place.
    ds = ds.assign_coords(
        excluded=(('condition', 'repeat'), np.zeros((n_conds, n_reps), dtype=bool)))

    assert_complete(ds)
    return ds


def assert_complete(ds):
    """Raise if any metric cell is NaN. Run before applying exclusions.

    This is the structured-array completeness check (analysis_for_images.py:1663-1667):
    after sorting, every (roi, condition, repeat, time) cell must hold a real value.
    """
    for m in ds.data_vars:
        if bool(ds[m].isnull().any()):
            raise ValueError('Found NaNs in {!r} after building the response table; a trial '
                             'window was not fully filled.'.format(m))


def exclude_trials(ds, trials):
    """Mark (condition, repeat) pairs as excluded. Returns a new Dataset.

    ``trials`` is an iterable of (condition, repeat) pairs. Values are NOT overwritten; only
    the ``excluded`` mask is set, so the underlying data and the un-filled-vs-excluded
    distinction are preserved. Reductions below honour this mask.
    """
    excluded = ds['excluded'].copy()
    for c, t in trials:
        excluded.loc[dict(condition=c, repeat=t)] = True
    return ds.assign_coords(excluded=excluded)


def _valid(ds, metric):
    """Metric values with excluded trials masked to NaN (without mutating ds)."""
    return ds[metric].where(~ds['excluded'])


def stim_window_response(ds, metric):
    """Per-(roi, condition) mean response over the stimulus window and non-excluded repeats.

    Equivalent to resp_vect_cond in analysis_for_images.py:2074
    (np.nanmean(data[m][:, :, :, idx_stim], axis=(2, 3))), but excluded repeats are dropped
    via the mask rather than relying on them having been NaN-ed.
    """
    stim = _valid(ds, metric).where(ds['epoch'] == EPOCH_STIM)
    return stim.mean(dim=('repeat', 'time'), skipna=True)


def normalize_response(resp_vect_cond, method='peak', dim='condition'):
    """Per-ROI normalization of a response matrix across the stimulus set.

    Rescales each ROI's response vector over ``dim`` (default the stimulus ``condition`` axis) so
    ROIs can be pooled or compared on a common footing -- the per-neuron normalization reported in
    the Freiwald/Tsao face-patch studies. The reduction is over ``dim`` only, independently per ROI;
    other axes broadcast. Intended for the ``(roi, condition)`` output of ``stim_window_response``.

    Parameters
    ----------
    resp_vect_cond : xarray.DataArray
        Response array with a ``dim`` axis (e.g. the ``(roi, condition)`` matrix).
    method : {'peak', 'l2', 'zscore', 'range'}
        ``'peak'``   -- divide by the per-ROI maximum absolute value over ``dim`` (sign kept, peak
        magnitude 1). Purely multiplicative, so ``face_dprime`` and ``face_selectivity_index`` are
        unchanged. ``'l2'`` -- divide by the per-ROI L2 norm over ``dim``; multiplicative, d'/FSI
        unchanged. ``'zscore'`` -- ``(r - mean) / SD`` over ``dim``; affine (moves the zero point),
        so d' is unchanged but FSI changes (FSI is not shift-invariant). ``'range'`` --
        ``(r - min) / (max - min)`` over ``dim`` -> [0, 1]; affine, d' unchanged, FSI changes.
    dim : str
        Axis to normalize over (the stimulus axis). Default ``'condition'``.

    Returns
    -------
    xarray.DataArray
        Same shape/coords as the input. An ROI whose normalizer is zero (a flat/silent response
        vector) is returned as zeros.

    Notes
    -----
    Because face d' is affine-invariant and FSI is only scale-invariant, this helper does not alter
    the selectivity indices except where a method shifts the zero point (``'zscore'``/``'range'``
    change FSI). It is meant for population-level comparison/visualization, not for changing d'/FSI.
    """
    r = resp_vect_cond
    if method == 'peak':
        num, denom = r, np.abs(r).max(dim)
    elif method == 'l2':
        num, denom = r, np.sqrt((r ** 2).sum(dim))
    elif method == 'zscore':
        num, denom = r - r.mean(dim), r.std(dim)
    elif method == 'range':
        lo = r.min(dim)
        num, denom = r - lo, r.max(dim) - lo
    else:
        raise ValueError("Unknown normalization method {!r}; expected one of "
                         "'peak', 'l2', 'zscore', 'range'.".format(method))
    return xr.where(denom > 0, num / xr.where(denom > 0, denom, 1.0), 0.0)


def category_mean_std(resp_vect_cond, in_category):
    """Across-stimulus mean and SD of the per-condition responses for one category.

    ``resp_vect_cond`` is the (roi, condition) DataArray from ``stim_window_response``;
    ``in_category`` is a boolean array over the condition axis. Matches the ordering of
    operations in analysis_for_images.py:1976-1989 (mean/SD taken across conditions).
    """
    sel = resp_vect_cond.isel(condition=np.where(np.asarray(in_category))[0])
    return sel.mean('condition', skipna=True), sel.std('condition', skipna=True)


def face_dprime(resp_vect_cond, is_face, is_nonface):
    """Face discriminability index d'.

    Vinken et al. Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
    d' = (mu_F - mu_NF) / sqrt((sigma_F**2 + sigma_NF**2) / 2)
    where mu/sigma are the across-stimulus mean/SD of the trial-averaged responses.
    """
    mu_F, sigma_F = category_mean_std(resp_vect_cond, is_face)
    mu_NF, sigma_NF = category_mean_std(resp_vect_cond, is_nonface)
    return (mu_F - mu_NF) / np.sqrt((sigma_F ** 2 + sigma_NF ** 2) / 2)


def face_selectivity_index(resp_vect_cond, is_face, is_nonface_object):
    """Face selectivity index (FSI).

    Freiwald and Tsao 2010 Science https://doi.org/10.1126/science.1194908
    FSI = (R_face - R_nonfaceobject) / (R_face + R_nonfaceobject), using mean responses.
    When the two responses have opposite sign, FSI is clamped to +1 (face>0, object<0) or
    -1 (face<0, object>0), matching analysis_for_images.py:2028-2035.
    """
    mu_F, _ = category_mean_std(resp_vect_cond, is_face)
    mu_O, _ = category_mean_std(resp_vect_cond, is_nonface_object)
    fsi = xr.full_like(mu_F, np.nan)

    same_sign = np.sign(mu_F) == np.sign(mu_O)
    fsi = xr.where(same_sign, (mu_F - mu_O) / (mu_F + mu_O), fsi)
    fsi = xr.where(np.sign(mu_F) > np.sign(mu_O), 1.0, fsi)
    fsi = xr.where(np.sign(mu_F) < np.sign(mu_O), -1.0, fsi)
    return fsi


def trial_response(ds, metric):
    """Per-(roi, condition, repeat) mean response over the stimulus window.

    Like ``stim_window_response`` but keeps the ``repeat`` axis, giving one scalar per trial. Excluded
    trials become NaN via the exclusion mask. This is the per-trial response used by trial-level
    statistics -- a repeats-as-samples ANOVA or split-half reliability -- which treat the trial (not
    the frame) as the unit of observation.
    """
    stim = _valid(ds, metric).where(ds['epoch'] == EPOCH_STIM)
    return stim.mean(dim='time', skipna=True)


def trial_scalar_anova(ds, metric='Fzsc'):
    """Per-ROI one-way ANOVA across conditions using one scalar per trial (repeats as samples).

    The statistically sound counterpart to the frame-pooled responsiveness ANOVA in
    ``analysis_for_images.py`` (mirrored by ``images.responsive_anova``): each (condition, repeat) is
    first reduced to a single stim-window mean, so strongly autocorrelated within-trial frames are NOT
    treated as independent observations. Pooling frames inflates the degrees of freedom and makes the
    test anti-conservative; reducing to one scalar per trial removes that pseudoreplication. Returns a
    per-ROI p-value array (NaN where a cell has too few valid trials).
    """
    tr = trial_response(ds, metric).transpose('roi', 'condition', 'repeat').values
    n_roi, n_cond = tr.shape[0], tr.shape[1]
    pvals = np.full(n_roi, np.nan)
    for r in range(n_roi):
        groups = [g[~np.isnan(g)] for g in tr[r]]
        groups = [g for g in groups if g.size > 0]
        if len(groups) >= 2 and sum(g.size for g in groups) > len(groups):
            pvals[r] = f_oneway(*groups).pvalue
    return pvals


def split_half_reliability(ds, metric, n_splits=100, seed=0, spearman_brown=True):
    """Per-ROI response reliability by split-half correlation (Vinken et al. Livingstone 2023).

    For each ROI the repeats are randomly split in half ``n_splits`` times; each half is trial-averaged
    to a condition-response vector, the two vectors are Pearson-correlated across conditions, and the
    correlations are averaged to r. With ``spearman_brown`` the full-set reliability is
    rho = 2r / (1 + r) (the correction noted at analysis_for_images.py:2038-2047). This is a
    trial-level signal-consistency measure -- a principled complement to, or replacement for, the
    responsiveness ANOVA. Returns a per-ROI array (NaN where fewer than two repeats are available).

    Parameters
    ----------
    ds : xarray.Dataset
        A response table (see ``build_response_table``).
    metric : str
        Metric variable to use (e.g. 'Fzsc' or 'FdFF').
    n_splits : int
        Number of random half-splits to average over.
    seed : int
        Seed for the split permutations (drawn once and shared across ROIs, for reproducibility).
    spearman_brown : bool
        Apply the Spearman-Brown correction rho = 2r / (1 + r).
    """
    tr = trial_response(ds, metric).transpose('roi', 'condition', 'repeat').values
    n_roi, _, n_rep = tr.shape
    half = n_rep // 2
    out = np.full(n_roi, np.nan)
    if half < 1:
        return out
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n_rep) for _ in range(n_splits)]
    for r in range(n_roi):
        corrs = []
        for perm in perms:
            a = np.nanmean(tr[r][:, perm[:half]], axis=1)
            b = np.nanmean(tr[r][:, perm[half:2 * half]], axis=1)
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])
        if corrs:
            r_mean = float(np.mean(corrs))
            out[r] = (2 * r_mean / (1 + r_mean)) if (spearman_brown and (1 + r_mean) != 0) else r_mean
    return out


def _as_int_frame(value):
    """Coerce a frame index to int, raising on a non-integer float (cf. images:1606-1616)."""
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError('Non-integer acquisition frame index: {}.'.format(value))
    return int(value)
