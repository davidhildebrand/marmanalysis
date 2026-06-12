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


def _as_int_frame(value):
    """Coerce a frame index to int, raising on a non-integer float (cf. images:1606-1616)."""
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError('Non-integer acquisition frame index: {}.'.format(value))
    return int(value)
