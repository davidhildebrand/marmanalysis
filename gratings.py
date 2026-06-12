#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gratings-paradigm driver on the shared loaders (sessionio) + response container
(response_table), the gratings analogue of images.py.

Multimodal sessions interleave gratings with other stimuli (tones/flashes), so reps are ragged
across modalities and sessionio.build_session_response_table can't be used directly. This driver
therefore filters the stimlog to the drifting-grating trials BEFORE building the container.

Per-ROI tuning: OSI via tuning.calculate_osi (orientation), DSI via tuning.calculate_dsi (drift
direction). The original analysis_for_gratings.py mistakenly computed DSI under an `threshold_osi`
name and never loaded its data; this is the rebuild on the shared library.

Column note: convert_stimulus_record maps the source grating drift DIRECTION into the normalized
`grating_ori` column (the schema has no grating_dir), so here the per-condition angle read from
`grating_ori` is the DIRECTION, and ORIENTATION = direction % 180. A schema cleanup adding a real
`grating_dir` column is future work (flagged with task #17).
"""

import numpy as np
import pandas as pd

import response_table
import sessionio
import tuning

# Normalized column that actually holds the drift direction (see module note).
GRATING_DIR_COL = 'grating_ori'


def select_drifting_gratings(stimlog):
    """Subset a stimlog to drifting-grating trials (stim_class 'grating', subclass 'drifting')."""
    n = len(stimlog)
    is_grating = (stimlog['stim_class'].astype(str) == 'grating'
                  if 'stim_class' in stimlog.columns else np.ones(n, dtype=bool))
    is_drifting = (stimlog['stim_subclass'].astype(str) == 'drifting'
                   if 'stim_subclass' in stimlog.columns else np.ones(n, dtype=bool))
    return stimlog[np.asarray(is_grating) & np.asarray(is_drifting)].reset_index(drop=True)


def build_condition_metadata(stimlog, cond_col='cond'):
    """Per-condition grating metadata (direction, orientation = dir % 180, sf, tf)."""
    conditions = np.unique(stimlog[cond_col].dropna().to_numpy())
    rows = []
    for c in conditions:
        sub = stimlog[stimlog[cond_col] == c]
        direction = _unique(sub, GRATING_DIR_COL)
        orientation = (float(direction) % 180
                       if direction is not None and not pd.isna(direction) else np.nan)
        rows.append({'condition': c, 'direction': direction, 'orientation': orientation,
                     'sf': _unique(sub, 'grating_sf'), 'tf': _unique(sub, 'grating_tf')})
    return pd.DataFrame(rows).set_index('condition')


def attach_condition_metadata(ds, meta):
    coords = {col: ('condition', meta[col].to_numpy())
              for col in ['direction', 'orientation', 'sf', 'tf']}
    return ds.assign_coords(coords)


def orientation_selectivity(ds, metric='Fzsc', compute_dsi=True):
    """Per-ROI OSI (+ preferred orientation) and optionally DSI (+ preferred direction), computed
    over the grating drift directions on ds. Returns a dict of (n_roi,) arrays.

    OSI is fast (vector strength); DSI fits a von Mises per ROI and is much slower, so it can be
    disabled with compute_dsi=False.
    """
    resp = response_table.stim_window_response(ds, metric).transpose('roi', 'condition').values
    directions = np.asarray(ds['direction'].values, dtype=float)
    n_roi = resp.shape[0]
    osi = np.full(n_roi, np.nan)
    ori_pref = np.full(n_roi, np.nan)
    dsi = np.full(n_roi, np.nan)
    dir_pref = np.full(n_roi, np.nan)
    for r in range(n_roi):
        osi[r], ori_pref[r] = tuning.calculate_osi(directions, resp[r])
        if compute_dsi:
            dsi[r], dir_pref[r] = tuning.calculate_dsi(directions, resp[r])
    return {'osi': osi, 'ori_pref': ori_pref, 'dsi': dsi, 'dir_pref': dir_pref}


def process_session(session_path, metric='Fzsc', variant=None, baseline_method='medianbw',
                    compute_dsi=True):
    """Thin gratings driver: a session path -> per-ROI orientation/direction tuning, via the shared
    loaders + container, filtering to drifting-grating trials. Returns a results dict with the
    response Dataset, the load context, and the tuning arrays."""
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
        'tuning': orientation_selectivity(ds, metric, compute_dsi=compute_dsi),
    }


def _unique(sub, col):
    if col not in sub.columns:
        return np.nan
    vals = pd.unique(sub[col].dropna().values)
    return vals[0] if len(vals) else np.nan
