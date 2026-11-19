#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gratings-paradigm driver on the shared loaders (sessionio) + response container
(response_table), the gratings analogue of images.py.

Multimodal sessions interleave gratings with other stimuli (tones/flashes), so reps are ragged
across modalities and sessionio.build_session_response_table can't be used directly. This driver
therefore filters the stimlog to the drifting-grating trials BEFORE building the container.

Per-ROI tuning follows Mazurek, Kager and Van Hooser 2014 (https://doi.org/10.3389/fncir.2014.00092)
by default: 1 - CirVar and 1 - DirCirVar for how selective a cell is, Hotelling's T^2 and the direction
dot product test for whether it is selective, on responses measured relative to the pre-stimulus
baseline (see orientation_selectivity). The earlier measures (tuning.calculate_osi, and the von Mises
fit in tuning.calculate_dsi) remain available with method='legacy'. The original
analysis_for_gratings.py mistakenly computed DSI under an `threshold_osi` name and never loaded its
data; this is the rebuild on the shared library.

Column note: the drift DIRECTION (0-360) is the normalized `grating_dir` column, and ORIENTATION =
direction % 180. Until 2026-11-18 the schema had no grating_dir and the direction was read from
`grating_ori`. That was wrong for logs that record both columns (the Multimodal stimulus code writes
grating_ori = dir % 180): the 16 directions arrived as 8 angles, each twice. Stimulus logs normalized
before that date have no grating_dir, so `grating_ori` remains the fallback.
"""

from warnings import warn

import numpy as np
import pandas as pd

import response_table
import sessionio
import tuning

# Normalized columns that can hold the drift direction, in order of preference (see module note).
GRATING_DIR_COLS = ('grating_dir', 'grating_ori')


def direction_column(stimlog):
    """The stimlog column holding the drift direction: grating_dir when it has values, else grating_ori."""
    for col in GRATING_DIR_COLS:
        if col in stimlog.columns and stimlog[col].notna().any():
            return col
    raise KeyError('no grating direction column ({}) with values in the stimulus log'.format(', '.join(GRATING_DIR_COLS)))


def select_drifting_gratings(stimlog):
    """Subset a stimlog to drifting-grating trials (stim_class 'grating', subclass 'drifting')."""
    n = len(stimlog)
    is_grating = (stimlog['stim_class'].astype(str) == 'grating'
                  if 'stim_class' in stimlog.columns else np.ones(n, dtype=bool))
    is_drifting = (stimlog['stim_subclass'].astype(str) == 'drifting'
                   if 'stim_subclass' in stimlog.columns else np.ones(n, dtype=bool))
    return stimlog[np.asarray(is_grating) & np.asarray(is_drifting)].reset_index(drop=True)


def identical_conditions(stimlog, cond_col='cond'):
    """Groups of conditions whose LOGGED grating parameters are all the same: [[cond, cond, ...], ...].

    Two conditions the log cannot tell apart are either the same stimulus shown under two condition numbers, or a
    parameter that was dropped on the way in. Both have happened. Until 2026-11-18 the drift direction was lost for
    logs that record grating_dir and grating_ori, so each direction looked identical to its opposite. And the
    Multimodal stimulus code up to 2024 built every "static grating" condition with the orientation of the last
    moving grating (a stale loop variable), so its six conditions are one stimulus: 48 sessions on disk, confirmed in
    each session's PsychoPy log (tmp/audit_static_gratings.py). Compared: every scalar grating_* column."""
    cols = [c for c in stimlog.columns if c.startswith('grating_')]
    keys = {}
    for c in np.unique(stimlog[cond_col].dropna().to_numpy()):
        sub = stimlog[stimlog[cond_col] == c]
        key = []
        for col in cols:
            present = [v for v in sub[col].to_list() if isinstance(v, (list, tuple, np.ndarray)) or not pd.isna(v)]
            value = present[0] if present else None                   # array-valued cells (size, pos) are not hashable
            if isinstance(value, (list, tuple, np.ndarray)):
                value = tuple(np.ravel(value).tolist())
            key.append(value)
        keys.setdefault(tuple(map(repr, key)), []).append(c)
    return [group for group in keys.values() if len(group) > 1]


def build_condition_metadata(stimlog, cond_col='cond'):
    """Per-condition grating metadata (direction, orientation = dir % 180, sf, tf). Warns when the log cannot tell
    two conditions apart (identical_conditions): their responses are then repeats of one stimulus, not a tuning
    curve."""
    conditions = np.unique(stimlog[cond_col].dropna().to_numpy())
    dir_col = direction_column(stimlog)
    same = identical_conditions(stimlog, cond_col)
    if same:
        warn('conditions with identical logged grating parameters (read from `%s`): %s. Check the session stimulus '
             'code and PsychoPy log for what was shown.' % (dir_col, '; '.join(str([_plain(c) for c in g]) for g in same)))
    rows = []
    for c in conditions:
        sub = stimlog[stimlog[cond_col] == c]
        direction = _unique(sub, dir_col)
        orientation = (float(direction) % 180
                       if direction is not None and not pd.isna(direction) else np.nan)
        rows.append({'condition': c, 'direction': direction, 'orientation': orientation,
                     'sf': _unique(sub, 'grating_sf'), 'tf': _unique(sub, 'grating_tf')})
    return pd.DataFrame(rows).set_index('condition')


def attach_condition_metadata(ds, meta):
    coords = {col: ('condition', meta[col].to_numpy())
              for col in ['direction', 'orientation', 'sf', 'tf']}
    return ds.assign_coords(coords)


def trial_responses(ds, metric='Fzsc', baseline='isi', baseline_sec=1.0):
    """Per-trial stimulus-window responses, (roi, repeat, condition), NaN for excluded presentations.

    ``baseline='isi'`` (default) subtracts each presentation's own late pre-stimulus mean (the last
    ``baseline_sec`` seconds of the ISI, at most its last half), so a response is the change the grating
    produced, as the selectivity indices assume. ``baseline=None`` uses the metric as it is. A repeat
    stands in for the paper's "trial" (one measurement at every angle)."""
    resp = response_table.trial_response(ds, metric)
    if baseline == 'isi':
        pre = np.where(ds['epoch'].values == response_table.EPOCH_ISI_PRE)[0]
        n_base = max(1, pre.size // 2)
        if 'time_s' in ds.coords and ds.sizes['time'] > 1:
            framerate = 1.0 / float(ds['time_s'].values[1] - ds['time_s'].values[0])
            n_base = int(np.clip(round(baseline_sec * framerate), 1, n_base))
        resp = resp - response_table._valid(ds, metric).isel(time=pre[-n_base:]).mean('time', skipna=True)
    elif baseline is not None:
        raise ValueError("baseline must be 'isi' or None")
    return resp.transpose('roi', 'repeat', 'condition').values


def orientation_selectivity(ds, metric='Fzsc', compute_dsi=True, method='vanhooser', baseline='isi',
                            negative='rectify', fit=False, n_boot=0, alpha=0.05, seed=0):
    """Per-ROI orientation and direction selectivity over the grating drift directions on ds. Returns a
    dict of (n_roi,) arrays.

    method='vanhooser' (default), the measures Mazurek, Kager and Van Hooser 2014 recommend:
        osi, ori_pref      1 - CirVar and the preferred orientation of the orientation vector
        dsi, dir_pref      1 - DirCirVar and the preferred direction of the direction vector
        ori_p              Hotelling's T^2 test on the per-trial orientation vectors
        dir_p              direction dot product test
        dir_pref_axis      the preferred direction constrained to the orientation axis (from that test)
        oi_peak, di_peak, osi_peak, dsi_peak   the classic peak-based indices, for comparison only
      Responses are per-presentation and relative to the pre-stimulus baseline (``baseline``, see
      trial_responses); mean responses below baseline are set to 0 in the vector indices (``negative``,
      see tuning.vector_selectivity). An ROI with no response above baseline at any direction gets NaN.
      ``fit=True`` adds the constrained double-Gaussian fit (fit_theta, fit_sigma, fit_hwhh, fit_Rp,
      fit_Rn, fit_C, fit_r2, fit_OI, fit_DI) and ``fit_reportable`` = ori_p < alpha, the paper's rule for
      when the fitted angle and width mean anything. ``n_boot`` > 0 adds bootstrap uncertainty for the
      reportable ROIs (fit_theta_ci, the 95% interval half-width in degrees, and fit_sigma_lo / _hi).
      Fits are slow (about 5 optimizations per ROI, times n_boot), which is why they are opt-in.
      ``compute_dsi=False`` skips the direction measures.

    method='legacy': the earlier behaviour, tuning.calculate_osi and tuning.calculate_dsi on the mean
    stimulus-window response with no baseline subtraction (returns osi, ori_pref, dsi, dir_pref).
    """
    directions = np.asarray(ds['direction'].values, dtype=float)
    if method == 'legacy':
        resp = response_table.stim_window_response(ds, metric).transpose('roi', 'condition').values
        n_roi = resp.shape[0]
        out = {key: np.full(n_roi, np.nan) for key in ('osi', 'ori_pref', 'dsi', 'dir_pref')}
        for r in range(n_roi):
            out['osi'][r], out['ori_pref'][r] = tuning.calculate_osi(directions, resp[r])
            if compute_dsi:
                out['dsi'][r], out['dir_pref'][r] = tuning.calculate_dsi(directions, resp[r])
        return out
    if method != 'vanhooser':
        raise ValueError("method must be 'vanhooser' or 'legacy'")

    trials = trial_responses(ds, metric, baseline)                       # (roi, repeat, direction)
    n_roi = trials.shape[0]
    full_circle = np.ptp(directions) >= 180.0                            # directions, not orientations only
    direction_space = bool(compute_dsi and full_circle)
    space = 'direction' if full_circle else 'orientation'
    keys = ['osi', 'ori_pref', 'dsi', 'dir_pref', 'ori_p', 'dir_p', 'dir_pref_axis',
            'oi_peak', 'di_peak', 'osi_peak', 'dsi_peak']
    fit_keys = ['fit_theta', 'fit_sigma', 'fit_hwhh', 'fit_Rp', 'fit_Rn', 'fit_C', 'fit_r2', 'fit_OI', 'fit_DI']
    boot_keys = ['fit_theta_ci', 'fit_sigma_lo', 'fit_sigma_hi']
    out = {key: np.full(n_roi, np.nan) for key in keys + (fit_keys if fit else []) + (boot_keys if fit and n_boot else [])}
    if fit:
        out['fit_reportable'] = np.zeros(n_roi, dtype=bool)
    for r in range(n_roi):
        complete = trials[r][np.all(np.isfinite(trials[r]), axis=1)]
        out['osi'][r], out['ori_pref'][r] = tuning.vector_selectivity(directions, trials[r], 'orientation', negative)
        if complete.shape[0] >= 3:
            out['ori_p'][r] = tuning.orientation_test(directions, complete)['p']
        peak = tuning.peak_selectivity(directions, trials[r], space)
        out['oi_peak'][r], out['osi_peak'][r] = peak['OI'], peak['OSI']
        if direction_space:
            out['dsi'][r], out['dir_pref'][r] = tuning.vector_selectivity(directions, trials[r], 'direction', negative)
            out['di_peak'][r], out['dsi_peak'][r] = peak['DI'], peak['DSI']
            if complete.shape[0] >= 3:
                dot = tuning.direction_dot_product_test(directions, complete)
                out['dir_p'][r], out['dir_pref_axis'][r] = dot['p'], dot['preferred_direction']
        if fit:
            f = tuning.fit_gaussian(directions, trials[r], space)
            for key, name in zip(fit_keys, ('theta_pref', 'sigma', 'hwhh', 'Rp', 'Rn', 'C', 'r2', 'OI', 'DI')):
                out[key][r] = f[name]
            out['fit_reportable'][r] = bool(np.isfinite(out['ori_p'][r]) and out['ori_p'][r] < alpha)
            if n_boot and out['fit_reportable'][r] and complete.shape[0] >= 2:
                boot = tuning.bootstrap_fit(directions, complete, space, n_boot=n_boot, seed=seed + r)
                out['fit_theta_ci'][r] = boot['theta_ci']
                out['fit_sigma_lo'][r], out['fit_sigma_hi'][r] = boot['ci']['sigma']
    return out


def bayesian_tuning(ds, metric='Fzsc', baseline='isi', models=None, noise='gaussian', roi_mask=None,
                    **sampler):
    """Per-ROI Bayesian tuning-curve analysis (Cronin et al. 2010, see tuning_bayes): posterior medians and 95%
    credibility intervals of the tuning parameters, and Bayes factors for "tuned at all" and "direction selective".

    Uses the same per-presentation, baseline-relative responses as orientation_selectivity. ``models`` defaults to
    all four when the directions cover the circle, else to 'constant' and 'circ180'. ``roi_mask`` restricts the
    sampling to some ROIs (the others get NaN); every ROI is sampled in one vectorized run. ``sampler`` goes to
    tuning_bayes.sample (n_burn, n_samples, thin, seed). The paper's settings take about half an hour for 1,300
    ROIs and four models.

    Returns the tuning_bayes.compare_models dict (results per model for the sampled ROIs, under 'results', with
    'roi' = their indices) plus (n_roi,) arrays: log_bf_tuned, log_bf_direction (and the _bic cross-checks),
    ori_pref, ori_pref_lo, ori_pref_hi, ori_sigma, ori_sigma_lo, ori_sigma_hi from the 180 deg model, and
    dir_pref, dir_pref_lo, dir_pref_hi from the two-amplitude direction model."""
    import tuning_bayes
    directions = np.asarray(ds['direction'].values, dtype=float)
    trials = trial_responses(ds, metric, baseline)
    n_roi = trials.shape[0]
    roi = np.arange(n_roi) if roi_mask is None else np.flatnonzero(np.asarray(roi_mask, bool))
    if models is None:
        models = (('constant', 'circ180', 'circ360', 'direction') if np.ptp(directions) >= 180.0
                  else ('constant', 'circ180'))
    cmp = tuning_bayes.compare_models(directions, trials[roi], models, noise, **sampler)
    out = {'results': cmp['results'], 'roi': roi, 'models': tuple(models), 'evidence': cmp['evidence']}

    def spread(values):
        full = np.full(n_roi, np.nan)
        full[roi] = values
        return full
    for key in ('log_bf_tuned', 'log_bf_direction', 'log_bf_tuned_bic', 'log_bf_direction_bic'):
        if key in cmp:
            out[key] = spread(cmp[key])
    for model, prefix, wanted in (('circ180', 'ori', ('mu', 'sigma')), ('direction', 'dir', ('mu',))):
        if model not in cmp['results']:
            continue
        res = cmp['results'][model]
        for name in wanted:
            j = res['names'].index(name)
            label = prefix + ('_pref' if name == 'mu' else '_' + name)
            out[label] = spread(res['median'][:, j])
            out[label + '_lo'], out[label + '_hi'] = spread(res['ci'][:, j, 0]), spread(res['ci'][:, j, 1])
    return out


def process_session(session_path, metric='Fzsc', variant=None, baseline_method='medianbw',
                    compute_dsi=True, **selectivity_options):
    """Thin gratings driver: a session path -> per-ROI orientation/direction tuning, via the shared
    loaders + container, filtering to drifting-grating trials. Returns a results dict with the
    response Dataset, the load context, and the tuning arrays. ``selectivity_options`` go to
    orientation_selectivity (method, baseline, negative, fit, n_boot, alpha)."""
    md = sessionio.load_metadata(session_path)
    s2p = sessionio.load_suite2p(session_path, variant=variant)
    traces = sessionio.compute_fluorescence_metrics(s2p['Frois'], md['framerate'], method=baseline_method)

    stimlog, prov = sessionio.load_stimlog(session_path)
    stimlog = sessionio.correct_acqfr_index(stimlog)
    gratings_log = select_drifting_gratings(stimlog)
    if gratings_log.empty:
        raise RuntimeError('No drifting-grating trials found in {}.'.format(session_path))

    n_samp_isi, n_samp_stim = sessionio.derive_trial_timing(gratings_log,
                                                            md.get('stim_locked_to_acqfr', True))
    gratings_log = sessionio.trim_to_imaged_trials(gratings_log, traces['Fraw'].shape[1],
                                                   n_samp_isi, n_samp_stim)
    ds = response_table.build_response_table(traces, gratings_log, n_samp_isi, n_samp_stim,
                                             framerate=md['framerate'])
    ds = attach_condition_metadata(ds, build_condition_metadata(gratings_log))
    return {
        'dataset': ds,
        'context': {'md': md, 's2p': s2p, 'traces': traces, 'stimlog': gratings_log,
                    'n_samp_isi': n_samp_isi, 'n_samp_stim': n_samp_stim, 'stim_provenance': prov},
        'tuning': orientation_selectivity(ds, metric, compute_dsi=compute_dsi, **selectivity_options),
    }


def _plain(value):
    return value.item() if hasattr(value, 'item') else value


def _unique(sub, col):
    if col not in sub.columns:
        return np.nan
    vals = pd.unique(sub[col].dropna().values)
    return vals[0] if len(vals) else np.nan
