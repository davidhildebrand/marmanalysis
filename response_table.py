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

import warnings

import numpy as np
import xarray as xr
from scipy.stats import f_oneway, ttest_rel, wilcoxon, false_discovery_control
from warnings import warn


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


def exclude_trials_by_stimlog(ds, stimlog, keep_mask, cond_col='cond'):
    """Exclude the response-table cells whose stimlog row is False in ``keep_mask``.

    ``keep_mask`` is a boolean aligned to stimlog rows (True = keep). Each dropped row is mapped to its
    (condition, repeat) cell using the SAME grouping ``build_response_table`` uses -- group by condition
    value, repeat = positional order within that group -- so the pairs index exactly the cells the trial's
    data occupies. Defers to ``exclude_trials`` (mask only; values are never overwritten). Returns
    (dataset, dropped_pairs). This is the join point for an external per-trial gate (e.g. the eye-position
    gate in eyetracking.calculate_eyepos_stimlog_keep)."""
    keep_mask = np.asarray(keep_mask, bool)
    if keep_mask.shape[0] != len(stimlog):
        raise ValueError('keep_mask length %d != stimlog rows %d' % (keep_mask.shape[0], len(stimlog)))
    conditions = np.unique(stimlog[cond_col].values)
    dropped = []
    for c in conditions:
        rows = np.where((stimlog[cond_col] == c).to_numpy(dtype=bool, na_value=False))[0]
        for t, ridx in enumerate(rows):                     # repeat index = position within the condition
            if not keep_mask[int(ridx)]:
                dropped.append((c, t))
    return exclude_trials(ds, dropped), dropped


def map_stimlog_values_to_cells(stimlog, values, cond_col='cond'):
    """Map a per-stimlog-ROW value array onto the (condition, repeat) grid, using the SAME grouping as
    ``build_response_table`` (group by condition value; repeat = positional order within that group).

    Returns an (n_condition, n_repeat) float array, NaN where no row maps to a cell. Use it to carry a
    per-trial quantity into the response table's coordinate system -- e.g. the eye gate's ``landing_sec``, so a
    per-trial response window can be shifted to when the animal actually looked (see ``trial_response_shifted``).
    """
    values = np.asarray(values, float)
    if values.shape[0] != len(stimlog):
        raise ValueError('values length %d != stimlog rows %d' % (values.shape[0], len(stimlog)))
    conditions = np.unique(stimlog[cond_col].values)
    n_rep = max(int((stimlog[cond_col] == c).to_numpy(dtype=bool, na_value=False).sum()) for c in conditions)
    out = np.full((len(conditions), n_rep), np.nan)
    for ci, c in enumerate(conditions):
        rows = np.where((stimlog[cond_col] == c).to_numpy(dtype=bool, na_value=False))[0]
        for t, ridx in enumerate(rows):
            out[ci, t] = values[int(ridx)]
    return out


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

    NOTE: this ratio index is NOT shift-invariant, so ``resp_vect_cond`` must be a baseline-relative
    "response above baseline" measure (FdFF, whose zero is the F0 baseline). Do NOT pass z-scored
    responses (Fzsc): they are mean-zero over the session, so R_face + R_nonfaceobject crosses zero and
    the index (and its +/-1 clamp) becomes meaningless. Contrast d' -- a scale/shift-invariant
    standardized difference that is valid on any affine-transformed metric. (See the images.py wrapper,
    which defaults to FdFF, and test_fsi_changes_under_additive_normalization.)
    """
    mu_F, _ = category_mean_std(resp_vect_cond, is_face)
    mu_O, _ = category_mean_std(resp_vect_cond, is_nonface_object)
    fsi = xr.full_like(mu_F, np.nan)

    same_sign = np.sign(mu_F) == np.sign(mu_O)
    fsi = xr.where(same_sign, (mu_F - mu_O) / (mu_F + mu_O), fsi)
    fsi = xr.where(np.sign(mu_F) > np.sign(mu_O), 1.0, fsi)
    fsi = xr.where(np.sign(mu_F) < np.sign(mu_O), -1.0, fsi)
    return fsi


def trial_response(ds, metric, epoch=EPOCH_STIM, reduce='mean', framerate=None,
                   offset_response_window_sec=0.0, response_window_sec=None):
    """Per-(roi, condition, repeat) response scalar over a per-trial response window.

    Keeps the ``repeat`` axis, giving one scalar per trial -- the unit of observation for trial-level
    statistics (repeats-as-samples ANOVA, split-half reliability, stim-vs-baseline responsiveness).
    Excluded trials become NaN via the exclusion mask.

    By DEFAULT (``reduce='mean'``, no offset/window) it is the mean over the ``epoch`` frames -- the original
    behaviour; pass ``epoch=EPOCH_ISI_PRE`` for the pre-stimulus baseline. The window can be shifted and
    resized to track calcium kinetics / response latency, and reduced differently, so the scalar can be made
    robust to brief transients or to response-timing jitter (e.g. from uncontrolled eye position):

      offset_response_window_sec  shift the window later by this many seconds from epoch onset (indicator rise +
                        neural latency); it may extend past the epoch into the following ISI (the decay
                        tail). Requires ``framerate``.
      response_window_sec        window length in seconds from the (shifted) onset; default = the epoch's length.
      reduce            'mean' (default) | 'peak' (max over the window) | 'auc' (time integral, per second).

    All three reductions are compared across conditions by a scale-invariant F-test, so the ANOVA gates
    accept any of them (note 'auc' scales with window length, so keep the window fixed when using it).
    """
    da = _valid(ds, metric)
    epoch_idx = np.where(ds['epoch'].values == epoch)[0]
    if epoch_idx.size == 0:
        raise ValueError('no %r frames in the response table' % epoch)
    if offset_response_window_sec == 0.0 and response_window_sec is None:
        idx = epoch_idx
    else:
        if framerate is None:
            raise ValueError('offset_response_window_sec / response_window_sec require framerate')
        start = int(epoch_idx[0] + round(offset_response_window_sec * framerate))
        n_win = epoch_idx.size if response_window_sec is None else max(1, int(round(response_window_sec * framerate)))
        idx = np.arange(max(start, 0), min(start + n_win, ds.sizes['time']))
        if idx.size == 0:
            raise ValueError('response window falls outside the trial (offset_response_window_sec/response_window_sec too large)')
    win = da.isel(time=idx)
    if reduce == 'mean':
        return win.mean('time', skipna=True)
    if reduce == 'peak':
        return win.max('time', skipna=True)
    if reduce == 'auc':
        return win.sum('time', skipna=True) / float(framerate or 1.0)
    raise ValueError("unknown reduce %r; expected 'mean', 'peak', or 'auc'" % reduce)


def trial_response_shifted(ds, metric, offset_sec, framerate, response_window_sec=None,
                           epoch=EPOCH_STIM, reduce='mean'):
    """Per-(roi, condition, repeat) response over a PER-TRIAL window whose start is shifted by
    ``offset_sec[condition, repeat]`` seconds from the epoch onset -- the counterpart to ``trial_response``,
    whose ``offset_response_window_sec`` is a single value shared by every trial.

    Motivation: the eye gate's ``landing_window`` mode keeps a trial whose look arrives LATE and reports when
    it arrived (``landing_sec``). Under a fixed full-stim-window mean, such a trial is measured over a window
    that largely PRECEDES its response, so the response is diluted. Shifting the window to start at the landing
    measures the response where the calcium actually is -- and because jGCaMP8s decays with tau ~0.29 s, the
    window may legitimately run past stimulus offset into the early ISI.

    ``offset_sec`` is (n_condition, n_repeat) -- e.g. from ``map_stimlog_values_to_cells`` applied to the gate's
    per-trial ``landing_sec``. A NaN offset (no landing) yields NaN for that trial. The window is clipped to the
    trial and is ``response_window_sec`` long (default: the epoch's own length). Excluded trials are NaN via the
    exclusion mask, as in ``trial_response``.
    """
    if framerate is None:
        raise ValueError('trial_response_shifted requires framerate')
    epoch_idx = np.where(ds['epoch'].values == epoch)[0]
    if epoch_idx.size == 0:
        raise ValueError('no %r frames in the response table' % epoch)
    n_win = epoch_idx.size if response_window_sec is None else max(1, int(round(response_window_sec * framerate)))
    vals = _valid(ds, metric).transpose('roi', 'condition', 'repeat', 'time').values
    off = np.asarray(offset_sec, float)
    n_roi, n_cond, n_rep, n_time = vals.shape
    if off.shape != (n_cond, n_rep):
        raise ValueError('offset_sec must be (n_condition, n_repeat) = %s, got %s'
                         % ((n_cond, n_rep), off.shape))
    out = np.full((n_roi, n_cond, n_rep), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)      # all-NaN (excluded) trials are expected
        for c in range(n_cond):
            for t in range(n_rep):
                if not np.isfinite(off[c, t]):
                    continue
                start = int(epoch_idx[0] + round(off[c, t] * framerate))
                i0, i1 = max(start, 0), min(start + n_win, n_time)
                if i1 <= i0:
                    continue
                win = vals[:, c, t, i0:i1]
                out[:, c, t] = np.nanmax(win, axis=1) if reduce == 'peak' else np.nanmean(win, axis=1)
    return xr.DataArray(out, dims=('roi', 'condition', 'repeat'),
                        coords={'roi': ds['roi'], 'condition': ds['condition'], 'repeat': ds['repeat']})


def anova_selective(ds, metric='Fzsc', exclude_blank=True, reduce='mean', framerate=None,
                    offset_response_window_sec=0.0, response_window_sec=None):
    """Per-ROI one-way ANOVA across conditions using one scalar per trial (repeats as samples).

    The statistically sound counterpart to the frame-pooled responsiveness ANOVA in
    ``analysis_for_images.py`` (mirrored by ``images.responsive_anova``): each (condition, repeat) is
    first reduced to a single stim-window mean, so strongly autocorrelated within-trial frames are NOT
    treated as independent observations. Pooling frames inflates the degrees of freedom and makes the
    test anti-conservative; reducing to one scalar per trial removes that pseudoreplication. Returns a
    per-ROI p-value array (NaN where a cell has too few valid trials).

    SELECTIVITY: with ``exclude_blank`` the no-stimulus 'blank' condition is dropped so the test isolates
    stimulus-vs-stimulus differentiation and is not fired by mere responsiveness (a uniform responder that
    differs only from blank would otherwise be mislabeled selective). Scrambles are stimuli and are kept.

    This is the DIFFERENTIATION test in isolation; report the selective POPULATION as its intersection with
    responsiveness (``classify_responses``) so ``selective ⊆ responsive`` holds -- a cell must respond to
    differentiate, so an independent selective count can otherwise flag borderline non-responders.
    """
    tr = trial_response(ds, metric, reduce=reduce, framerate=framerate,
                        offset_response_window_sec=offset_response_window_sec, response_window_sec=response_window_sec
                        ).transpose('roi', 'condition', 'repeat').values
    if exclude_blank:
        tr = tr[:, _stimulus_conditions(ds), :]
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


def _rowwise_corr(a, b):
    """Per-row Pearson correlation between two (n_roi, n_cond) matrices, NaN-safe. Returns (n_roi,)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i]) & np.isfinite(b[i])
        if ok.sum() >= 3 and np.std(a[i][ok]) > 0 and np.std(b[i][ok]) > 0:
            out[i] = np.corrcoef(a[i][ok], b[i][ok])[0, 1]
    return out


def calculate_temporal_split_stability(ds, metric='Fzsc', n_random=100, seed=0, roi_mask=None):
    """Per-ROI stability of the condition-tuning profile between the EARLY and LATE halves of the session --
    the drift/bleaching control (roadmap item 12f).

    Conditions are interleaved, so the ``repeat`` index is a proxy for session time: repeats 0..h are early,
    the last h are late. A slow nuisance that evolves through the session (z-drift, bleaching, arousal/state
    drift) makes the tuning estimated early differ from the tuning estimated late; genuine tuning does not.

    The diagnostic is the CONTRAST with a random split, NOT the temporal correlation alone: a random partition
    of the same size has identical trial-count noise but no temporal structure, so
    ``median(random_r) - median(temporal_r)`` isolates the time-varying component. A near-zero gap means the
    tuning is time-stable and any spatial structure it produces is not drift-driven.

    ``roi_mask`` (boolean, one per ROI) restricts the SUMMARY medians to a subset. In practice this matters a
    lot: over ALL ROIs both medians sit near zero because most cells carry no reliable tuning, which washes the
    contrast out — pass the responsive set so the comparison is made where tuning actually exists.

    Returns {'temporal_r', 'random_r' (per-ROI arrays), 'temporal_median', 'random_median', 'gap', 'n_repeat',
    'n_roi_used'}.
    """
    tr = trial_response(ds, metric).transpose('roi', 'condition', 'repeat').values
    n_rep = tr.shape[2]
    half = n_rep // 2
    if half < 1:
        raise ValueError('need >= 2 repeats for a temporal split (have %d)' % n_rep)
    with warnings.catch_warnings():                      # all-NaN condition cells are expected (exclusions)
        warnings.simplefilter('ignore', category=RuntimeWarning)
        temporal_r = _rowwise_corr(np.nanmean(tr[:, :, :half], axis=2),
                                   np.nanmean(tr[:, :, n_rep - half:], axis=2))
        rng = np.random.default_rng(seed)
        acc = []
        for _ in range(n_random):
            perm = rng.permutation(n_rep)
            acc.append(_rowwise_corr(np.nanmean(tr[:, :, perm[:half]], axis=2),
                                     np.nanmean(tr[:, :, perm[half:2 * half]], axis=2)))
        random_r = np.nanmean(np.array(acc), axis=0)
    m = np.ones(temporal_r.shape[0], bool) if roi_mask is None else np.asarray(roi_mask, bool)
    tm, rm = float(np.nanmedian(temporal_r[m])), float(np.nanmedian(random_r[m]))
    return {'temporal_r': temporal_r, 'random_r': random_r, 'temporal_median': tm,
            'random_median': rm, 'gap': rm - tm, 'n_repeat': int(n_rep), 'n_roi_used': int(m.sum())}


def shuffle_condition_labels(ds, seed=0):
    """Return a copy of ``ds`` with trials permuted across the (condition, repeat) cells -- i.e. the stimulus
    labels shuffled, DESTROYING tuning while leaving every trial's data intact (roadmap item 12g).

    The null for "this spatial map reflects stimulus tuning": recompute the map on the shuffled table and ask
    whether the spatial structure survives. If it does, the structure comes from something other than
    condition-tuning (a cell-intrinsic property such as SNR/depth, or a condition-independent nuisance), not
    from what the cells are tuned to. Each trial keeps its own samples, so per-cell noise and amplitude
    statistics are preserved exactly; only the condition assignment is randomised.
    """
    rng = np.random.default_rng(seed)
    n_cond, n_rep = ds.sizes['condition'], ds.sizes['repeat']
    perm = rng.permutation(n_cond * n_rep)
    out = ds.copy()
    for m in ds.data_vars:
        arr = ds[m].transpose('roi', 'condition', 'repeat', 'time').values
        n_roi, _, _, n_t = arr.shape
        flat = arr.reshape(n_roi, n_cond * n_rep, n_t)[:, perm, :]
        out[m] = (('roi', 'condition', 'repeat', 'time'), flat.reshape(n_roi, n_cond, n_rep, n_t))
    return out


def _late_isi_baseline(ds, metric, framerate, baseline_sec, min_baseline_frames, reduce='mean', n_frames=None):
    """Per-(roi, condition, repeat) mean over the LATE pre-stimulus ISI window.

    Uses the last ``baseline_sec`` seconds of the isi_pre epoch (nearest to stimulus onset), NOT the whole
    ISI: the early ISI is the decay tail of the previous trial's response and would bias the baseline. The
    window is framerate-aware and capped at the last 50% of the isi_pre frames, so a short ISI
    (< ~2x baseline_sec) falls back to its last half rather than reaching into the decay-tail region.
    """
    if framerate is None:
        raise ValueError('framerate (frames/s) is required to size the late-baseline window')
    epoch = ds['epoch'].values
    pre_idx = np.where(epoch == EPOCH_ISI_PRE)[0]
    if pre_idx.size == 0:
        raise ValueError('no isi_pre frames in the response table')
    # last baseline_sec, but never more than the last 50% of the ISI (leave the early decay-tail half out);
    # a short ISI (< ~2x baseline_sec) falls back to its last half. ``n_frames`` overrides the length -- used
    # to LENGTH-MATCH the baseline to the stim window for a fair peak comparison (peak grows with #frames).
    want = round(baseline_sec * framerate) if n_frames is None else n_frames
    n_base = int(np.clip(min(want, pre_idx.size // 2), 1, pre_idx.size))
    if n_base < min_baseline_frames:
        warn('anova_responsive: baseline window is %d frame(s) (< %d) at %.2f Hz -- stim-vs-baseline '
             'may be unreliable; use a longer baseline_sec or a faster session.'
             % (n_base, min_baseline_frames, framerate), stacklevel=3)
    win = _valid(ds, metric).isel(time=pre_idx[-n_base:])
    if reduce == 'peak':
        return win.max('time', skipna=True)
    if reduce == 'auc':
        return win.sum('time', skipna=True) / float(framerate)
    return win.mean('time', skipna=True)


def _stimulus_conditions(ds):
    """Boolean mask over the condition axis: True for real-stimulus conditions (everything except the
    no-stimulus 'blank'). Scrambles ARE stimuli and stay True. Uses the 'cat' coord if present, else all."""
    if 'cat' not in ds.coords:
        return np.ones(ds.sizes['condition'], dtype=bool)
    return np.array([c != b'blank' for c in ds['cat'].values])


def anova_responsive(ds, metric='Fzsc', framerate=None, baseline_sec=1.0, min_baseline_frames=3,
                     reduce='mean', offset_response_window_sec=0.0, response_window_sec=None, match_baseline_len=False):
    """Per-ROI visual RESPONSIVENESS gate: is the mean response modulated across stimuli AND baseline?

    A one-way ANOVA over [real-stimulus conditions] + [a no-stimulus baseline group], where the baseline is
    the BLANK condition's trials when the session has one, else the LATE pre-stimulus ISI window
    (_late_isi_baseline). Because it pools evidence exactly like the selectivity ANOVA (anova_selective)
    but adds the baseline group, it is a true SUPERSET of selectivity -- "the cell responds to the stimulus
    stream, including vs baseline". Scrambles count as stimuli; only blank is the baseline. For a stricter
    "responds to a SPECIFIC stimulus" label see fdr_responsive. Returns a per-ROI p-value.
    """
    tr = trial_response(ds, metric, EPOCH_STIM, reduce=reduce, framerate=framerate,
                        offset_response_window_sec=offset_response_window_sec, response_window_sec=response_window_sec
                        ).transpose('roi', 'condition', 'repeat').values
    is_stim = _stimulus_conditions(ds)
    cats = ds['cat'].values if 'cat' in ds.coords else None
    blank = None if cats is None else np.array([c == b'blank' for c in cats])
    if blank is not None and blank.any():
        base = tr[:, blank, :]                              # the measured no-stimulus (blank) condition
    else:
        n_win = None
        if match_baseline_len and framerate is not None:    # length-match baseline to the stim window (fair peak)
            n_stim = int((ds['epoch'].values == EPOCH_STIM).sum())
            n_win = max(1, int(round(response_window_sec * framerate))) if response_window_sec else n_stim
        base = (_late_isi_baseline(ds, metric, framerate, baseline_sec, min_baseline_frames,
                                   reduce=reduce, n_frames=n_win)
                .transpose('roi', 'condition', 'repeat').values)
    stim = tr[:, is_stim, :]
    n_roi, n_stimcond = stim.shape[0], stim.shape[1]
    pvals = np.full(n_roi, np.nan)
    for r in range(n_roi):
        groups = [stim[r, c][~np.isnan(stim[r, c])] for c in range(n_stimcond)]
        groups.append(base[r].ravel()[~np.isnan(base[r].ravel())])   # pooled no-stimulus baseline group
        groups = [g for g in groups if g.size > 0]
        if len(groups) >= 2 and sum(g.size for g in groups) > len(groups):
            pvals[r] = f_oneway(*groups).pvalue
    return pvals


def fdr_responsive(ds, metric='Fzsc', framerate=None, baseline_sec=1.0, test='ttest',
                                alternative='two-sided', min_baseline_frames=3):
    """Per-ROI FDR-controlled per-stimulus responsiveness: does the cell respond to a SPECIFIC stimulus?

    For each ROI and each condition the per-trial stim-window response is compared to the SAME trial's
    LATE pre-stimulus baseline (last ``baseline_sec`` of isi_pre -- see _late_isi_baseline) with a paired
    t-test ('ttest', default) or Wilcoxon signed-rank ('wilcoxon'). The per-condition p-values are
    Benjamini-Hochberg FDR-corrected across conditions and the ROI's score is the MINIMUM adjusted p (an
    FDR q-value) -- i.e. "responsive if ANY single stimulus beats baseline". ``alternative='two-sided'``
    counts both driven and suppressed cells. This asks a DIFFERENT question from stimulus selectivity
    (anova_selective) and, unlike the ANOVA gate (anova_responsive), is NOT a guaranteed superset of
    it: a distributed-selective cell (tuning spread across conditions, no single dominant stimulus) can be
    selective yet fail BH here. Kept as a distinct, stricter "specific-stimulus" label because the
    selectivity/responsiveness definitions are not yet settled and its misses are informative.

    NOTE with ~10 reps/condition a per-condition test has limited power (Wilcoxon especially, whose p has a
    ~2/2^n floor), so single-stimulus responders can fail BH here; the max-statistic permutation variant is
    better powered for that (anova_responsive_perm, staged separately). Returns per-ROI min BH-adjusted
    p (NaN where no condition had enough valid trials).
    """
    stim = trial_response(ds, metric, EPOCH_STIM).transpose('roi', 'condition', 'repeat').values
    base = (_late_isi_baseline(ds, metric, framerate, baseline_sec, min_baseline_frames)
            .transpose('roi', 'condition', 'repeat').values)
    n_roi, n_cond = stim.shape[0], stim.shape[1]
    out = np.full(n_roi, np.nan)
    for r in range(n_roi):
        pc = np.full(n_cond, np.nan)
        for c in range(n_cond):
            s, b = stim[r, c], base[r, c]
            ok = ~np.isnan(s) & ~np.isnan(b)
            s, b = s[ok], b[ok]
            if s.size < 2 or np.allclose(s, b):
                continue
            try:
                pc[c] = (ttest_rel(s, b, alternative=alternative).pvalue if test == 'ttest'
                         else wilcoxon(s, b, alternative=alternative).pvalue)
            except ValueError:
                pass
        valid = ~np.isnan(pc)
        if valid.any():
            out[r] = float(np.min(false_discovery_control(pc[valid], method='bh')))
    return out


def classify_responses(ds, metric='Fzsc', framerate=None, alpha=0.05, baseline_sec=1.0,
                       reduce='mean', offset_response_window_sec=0.0, response_window_sec=None, match_baseline_len=False):
    """Nested response classes with ``selective`` a strict subset of ``responsive``, by construction.

    A cell is RESPONSIVE if it shows ANY stimulus-driven modulation -- either it differs from baseline
    (``anova_responsive``, the omnibus conditions+baseline ANOVA) OR it differentiates among stimuli
    (``anova_selective``). It is SELECTIVE if it differentiates among stimuli. Taking responsive as the
    UNION guarantees ``selective ⊆ responsive`` (a cell must respond to differentiate) while keeping
    responsiveness permissive -- a differentiating cell is never dropped for failing the baseline contrast,
    which is what produced the 11 borderline "selective non-responders" under two independent ANOVAs. Mirrors
    v9, whose single ANOVA over [conditions + blank] is the gate and whose OSI/DSI are within-responsive
    DESCRIPTORS, so no selective non-responder can arise. Suppression counts: both ANOVAs are two-sided.

    Returns a dict of per-ROI boolean masks ``responsive`` and ``selective`` plus the p-value arrays
    ``p_responsive`` (omnibus) and ``p_selective`` (differentiation).
    """
    p_omni = anova_responsive(ds, metric, framerate=framerate, baseline_sec=baseline_sec,
                              reduce=reduce, offset_response_window_sec=offset_response_window_sec, response_window_sec=response_window_sec,
                              match_baseline_len=match_baseline_len)
    p_diff = anova_selective(ds, metric, reduce=reduce, framerate=framerate,
                             offset_response_window_sec=offset_response_window_sec, response_window_sec=response_window_sec)
    differentiates = (p_diff < alpha) & np.isfinite(p_diff)
    responds_vs_base = (p_omni < alpha) & np.isfinite(p_omni)
    return {'responsive': responds_vs_base | differentiates, 'selective': differentiates,
            'p_responsive': p_omni, 'p_selective': p_diff}


def _as_int_frame(value):
    """Coerce a frame index to int, raising on a non-integer float (cf. images:1606-1616)."""
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError('Non-integer acquisition frame index: {}.'.format(value))
    return int(value)
