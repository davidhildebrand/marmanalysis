#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-level loaders for the marmoset 2p pipeline (shared library, task #2).

Step 2a: stimlog loading with a layered source merge. Different recording eras saved the stimulus
log differently (text .log only -> structured pickle/csv -> multimodal pickle), and the sources
overlap with the text sometimes carrying fields omitted from the structured export. So rather than
pick one source, prefer the richest STRUCTURED source (pickle > h5 > csv) as the base and then
backfill missing values from the text .log (which is also the only source for the oldest sessions).
Everything is normalized to the parsers.create_stimulus_record schema.
"""

import glob
import os
import pickle
import re
import socket
from warnings import warn

import numpy as np
import pandas as pd

import eyetracking
import filters
import metadata
import parsers
import response_table


# Text-log parser per stimulus paradigm (the text format is paradigm-specific).
_PARADIGM_TEXT_PARSER = {
    'image': parsers.parse_log_stim_image,
    'dots': parsers.parse_log_stim_dots,
    'gratings': parsers.parse_log_stim_gratings,
}

# stimlog columns whose disagreement between sources actually matters for trial windowing.
_CRITICAL_COLS = ('cond', 'acqfr_stim_i', 'acqfr_stim_f', 'acqfr_isi_i', 'acqfr_isi_f')


def _paradigm_from_name(name):
    """Map a single directory/file name to a paradigm token (or 'unknown')."""
    n = name.lower()
    if 'multimodal' in n:
        return 'multimodal'
    if 'image' in n:
        return 'image'
    if 'grating' in n:
        return 'gratings'
    if 'dot' in n:
        return 'dots'
    if 'auditory' in n or 'tone' in n or 'vocal' in n:
        return 'auditory'
    if 'flash' in n:
        return 'flash'
    if 'dummy' in n:
        return 'dummy'
    return 'unknown'


def infer_paradigm(session_path, log_path=None):
    """Guess the stimulus paradigm. The stimulus LOG filename (``..._Stimulus_<Type>.log``) is a
    more reliable signal than the session directory's ``stim<...>`` token, which is occasionally
    mislabeled (e.g. a DriftingGratings session foldered as 'MovingDots'), so prefer it when a text
    log is available and fall back to the directory name otherwise."""
    candidates = ([os.path.basename(log_path)] if log_path else [])
    candidates.append(os.path.basename(os.path.normpath(session_path)))
    for name in candidates:
        paradigm = _paradigm_from_name(name)
        if paradigm != 'unknown':
            return paradigm
    return 'unknown'


def find_stimlog_sources(session_path):
    """Locate available stimlog sources in a session directory.

    Returns a dict with keys 'pickle', 'h5', 'csv', 'text' (value = path or None).
    """
    def first(patterns):
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(session_path, p)))
            if hits:
                return hits[0]
        return None

    text = [f for f in sorted(glob.glob(os.path.join(session_path, '*Stimulus*.log')))
            if 'disptimes' not in os.path.basename(f)]
    return {
        'pickle': first(['*_stimlog.p', '*_stimlog.pickle', '*_stimlog.pkl']),
        'h5': first(['*_stimlog.h5', '*_stimlog.hdf5']),
        'csv': first(['*_stimlog.csv']),
        'text': text[0] if text else None,
    }


def _load_structured(sources):
    """Load + normalize the best available structured source (pickle > h5 > csv)."""
    if sources['pickle']:
        return parsers.convert_stimulus_record(pd.read_pickle(sources['pickle'])), 'pickle', sources['pickle']
    if sources['h5']:
        return parsers.convert_stimulus_record(pd.read_hdf(sources['h5'])), 'h5', sources['h5']
    if sources['csv']:
        return parsers.convert_stimulus_record(pd.read_csv(sources['csv'])), 'csv', sources['csv']
    return None, None, None


def _parse_text(path, paradigm, required):
    """Parse a text .log with the paradigm's parser; warn and return None on failure.

    Parser failures are non-fatal when a structured base already exists (text is only backfill),
    but fatal when the text is the only source.
    """
    parser = _PARADIGM_TEXT_PARSER.get(paradigm)
    if parser is None:
        if required:
            raise RuntimeError('No text-log parser for paradigm {!r}.'.format(paradigm))
        warn('No text-log parser for paradigm {!r}; skipping text backfill.'.format(paradigm))
        return None
    try:
        with open(path) as f:
            return parsers.normalize_stimlog_dtypes(parser(f.read()))
    except Exception as e:
        if required:
            raise
        warn('Text-log parse failed ({}); skipping text backfill.'.format(e))
        return None


def load_stimlog(session_path, paradigm='auto', backfill_from_text=True):
    """Load a normalized stimlog by merging the best structured source with the text log.

    The structured source (pickle > h5 > csv) is the base; the text .log fills missing/NaN values
    (overwrite=False) and becomes the base when no structured source exists. All sources are
    normalized to the parsers.create_stimulus_record schema.

    Parameters
    ----------
    session_path : str
        Path to a single session directory.
    paradigm : str
        Stimulus paradigm selecting the text-log parser; 'auto' infers it from the text-log
        filename (preferred) or the session directory name.
    backfill_from_text : bool
        Whether to backfill missing values from the text log when a structured base exists.

    Returns
    -------
    (stimlog, provenance) : (pandas.DataFrame, dict)
        ``provenance`` records base_source, the source files, whether/how much the text backfilled,
        and any base/text disagreements on windowing-critical columns.
    """
    sources = find_stimlog_sources(session_path)
    if paradigm == 'auto':
        paradigm = infer_paradigm(session_path, sources['text'])

    base, base_source, base_file = _load_structured(sources)

    text_df = None
    if sources['text'] is not None and (base is None or backfill_from_text):
        text_df = _parse_text(sources['text'], paradigm, required=(base is None))

    if base is None:
        if text_df is None:
            raise RuntimeError('No loadable stimlog source found in {}.'.format(session_path))
        stimlog = text_df
        base_source, base_file = 'text', sources['text']
        n_backfilled, conflicts = 0, {}
    else:
        stimlog = base.copy()
        conflicts = _critical_conflicts(base, text_df) if text_df is not None else {}
        before = int(stimlog.isnull().sum().sum())
        if text_df is not None:
            stimlog.update(text_df, overwrite=False)          # backfill NaNs only
        n_backfilled = before - int(stimlog.isnull().sum().sum())

    provenance = {
        'paradigm': paradigm,
        'base_source': base_source,
        'base_file': base_file,
        'text_file': sources['text'],
        'text_backfilled': bool(text_df is not None and base is not None),
        'n_cells_backfilled': int(n_backfilled),
        'conflicts': conflicts,
    }
    stimlog = parsers.normalize_stimlog_dtypes(stimlog)
    return stimlog.reset_index(drop=True), provenance


def _critical_conflicts(base, text_df):
    """Count base/text disagreements on windowing-critical columns (warns; base wins)."""
    conflicts = {}
    for c in _CRITICAL_COLS:
        if c in base.columns and c in text_df.columns:
            both = base[c].notnull() & text_df[c].notnull()
            n_diff = int((both & (base[c] != text_df[c])).sum())
            if n_diff:
                conflicts[c] = n_diff
    if conflicts:
        warn('stimlog base/text disagree on {}; keeping base values.'.format(conflicts))
    return conflicts


# --- session loaders (2b) ------------------------------------------------------------------

# Default suite2p variant when a session has more than one suite2p_* folder
# (matches dirstr_suite2p_pref in the analysis_for_*.py scripts).
SUITE2P_VARIANT_DEFAULT = r'suite2p_cellpose3_d[0-9]+px_pt-3p5_ft1p5'


def resolve_paths(hostname=None):
    """Map the host to (base_path, stim_path), as in the analysis_for_*.py host block."""
    h = (hostname or socket.gethostname()).lower()
    if 'galactica' in h:
        return (r'/Users/davidh/Data/Freiwald/suite2p_results',
                r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets')
    if 'obsidian' in h:
        return (r'F:\Data', r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets')
    if 'dobbin' in h:
        return (r'D:\Data', r'C:\Users\DavidH\Sync\Freiwald\MarmoScope\Stimulus\Sets')
    return (None, None)


def load_metadata(session_path):
    """Load session metadata: the *_metadata.pickle if present, else parsed from the *_00001.tif,
    merged onto metadata.default_metadata() with derived fov width/height in um.
    Replicates analysis_for_images.py:790-813.
    """
    md_files = [f for f in glob.glob(os.path.join(session_path, '*_metadata.pickle')) if os.path.isfile(f)]
    img_files = [f for f in glob.glob(os.path.join(session_path, '*_00001.tif')) if os.path.isfile(f)]
    if md_files:
        with open(md_files[0], 'rb') as f:
            md = pickle.load(f)
    elif img_files:
        warn('Could not find metadata file, loading from image data file.')
        md = metadata.extract_useful_metadata(metadata.get_metadata(img_files[0]))
    else:
        raise RuntimeError('Could not find metadata or image data file in {}.'.format(session_path))
    md = {**metadata.default_metadata(), **md}
    fov = md.get('fov')
    if fov and 'resolution_umpx' in fov:
        if 'w_um' not in fov and 'w_px' in fov:
            fov['w_um'] = fov['resolution_umpx'][0] * fov['w_px']
        if 'h_um' not in fov and 'h_px' in fov:
            fov['h_um'] = fov['resolution_umpx'][1] * fov['h_px']
    return md


def find_suite2p_dir(session_path, variant=None):
    """Pick a session's suite2p_* output directory. With variant=None and several present, prefer
    the SUITE2P_VARIANT_DEFAULT pattern; pass a regex/substring to select a specific extraction
    (e.g. cellpose-anatomical vs functional). Matches analysis_for_images.py:938-947.
    """
    dirs = [d for d in sorted(glob.glob(os.path.join(session_path, 'suite2p*'))) if os.path.isdir(d)]
    if not dirs:
        raise RuntimeError('Could not find a suite2p folder in {}.'.format(session_path))
    if variant is not None:
        matches = [d for d in dirs if re.search(variant, os.path.basename(d))]
        if not matches:
            raise RuntimeError('No suite2p variant matching {!r} in {} (have {}).'.format(
                variant, session_path, [os.path.basename(d) for d in dirs]))
        if len(matches) > 1:
            warn('Multiple suite2p variants match {!r}, using {}.'.format(
                variant, os.path.basename(matches[0])))
        return matches[0]
    idx = 0
    if len(dirs) > 1:
        preferred = [i for i, d in enumerate(dirs)
                     if re.search(SUITE2P_VARIANT_DEFAULT, os.path.basename(d))]
        idx = preferred[0] if preferred else 0
        warn('Found multiple suite2p folders, using {}.'.format(os.path.basename(dirs[idx])))
    return dirs[idx]


def load_suite2p(session_path, variant=None, threshold_cellprob=0.0):
    """Load a session's suite2p plane0 outputs and select accepted, active ROIs.
    Replicates analysis_for_images.py:954-975. Returns a dict with Frois, Fneu (the per-ROI neuropil
    trace, aligned to Frois, or None if the extraction has no Fneu.npy), ROIs (stat), ops, badframes,
    cellinds, iscell, fov_image, fov_size, path.
    """
    s2p_dir = find_suite2p_dir(session_path, variant=variant)
    plane = os.path.join(s2p_dir, 'plane0')
    if not os.path.isdir(plane):
        raise RuntimeError('Could not find suite2p plane0 in {}.'.format(s2p_dir))
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
    F = np.load(os.path.join(plane, 'F.npy'))
    fneu_path = os.path.join(plane, 'Fneu.npy')                # suite2p neuropil trace (for correction)
    Fneu = np.load(fneu_path) if os.path.isfile(fneu_path) else None
    stat = np.load(os.path.join(plane, 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()

    cellinds = np.where(iscell[:, 1] >= threshold_cellprob)[0]
    inactives = np.where(np.std(F, axis=1) == 0)[0]
    if len(inactives) > 0:
        warn('Excluded {} inactive ROIs.'.format(len(inactives)))
    cellinds = np.setdiff1d(cellinds, inactives)
    return {
        'path': s2p_dir,
        'Frois': F[cellinds],
        'Fneu': (Fneu[cellinds] if Fneu is not None else None),
        'ROIs': stat[cellinds],
        'ops': ops,
        'badframes': np.where(ops['badframes'])[0],
        'cellinds': cellinds,
        'iscell': iscell,
        'fov_image': ops['meanImg'],
        'fov_size': (ops['Ly'], ops['Lx']),
    }


def compute_fluorescence_metrics(frois, framerate, window=60, method='medianbw',
                                 fneu=None, neucoeff=0.7, neuropil_subtract=True):
    """Baseline F0 and the dF/F + z-scored traces from raw ROI fluorescence.
    Replicates analysis_for_images.py:985-994 (images use method='medianbw', dots use 'meanbw').
    Returns {'FdFF', 'Fzsc', 'F0', 'Fraw'}.

    Neuropil correction (opt-out). When ``neuropil_subtract`` is True AND a neuropil trace ``fneu`` is
    provided, the ROI trace is corrected as ``Fc = F - neucoeff * (Fneu - median(Fneu))`` BEFORE the
    baseline/dF-F, matching v9 (def_load_data.py:1717) and suite2p's convention (``neucoeff`` default 0.7).
    Subtracting the neuropil FLUCTUATIONS (with the per-ROI median added back) is baseline-preserving:
    the DC level of Fc stays ~F, so F0 stays positive and dF/F is unaffected by the additive shift -- unlike
    naive ``F - r*Fneu``, which pulls the baseline down and can go negative. All downstream metrics (dF/F,
    z-score, and thus every responsiveness/RSA/topography result) then run on the corrected trace, since it
    is the single trace the baseline is computed from. ``Fraw`` in the returned dict is this corrected trace
    (== the input ``frois`` object when correction is off), so ``Fraw == F0 * (1 + FdFF)`` always holds.

    With ``fneu is None`` (e.g. a direct call, or a suite2p extraction lacking Fneu.npy) NO correction is
    applied regardless of ``neuropil_subtract``, and ``Fraw is frois``. Set ``neuropil_subtract=False`` to
    force the uncorrected traces even when a neuropil trace is available (the side-by-side comparison).
    """
    if neuropil_subtract and fneu is not None:
        fc = frois - neucoeff * (fneu - np.median(fneu, axis=1, keepdims=True))
    else:
        fc = frois                                            # identity: Fraw stays the input object
    f0 = filters.calculate_baselines(fc, framerate=framerate, window=window, method=method)
    fd = fc - f0
    fdff = fd / f0
    fzsc = (fd - np.mean(fd, axis=1)[:, np.newaxis]) / np.std(fd, axis=1)[:, np.newaxis]
    return {'FdFF': fdff, 'Fzsc': fzsc, 'F0': f0, 'Fraw': fc}


def correct_acqfr_index(stimlog):
    """Subtract 1 from all acqfr_* columns (the frame counter starts at 1, not 0).
    Matches analysis_for_images.py:1267-1271.
    """
    stimlog = stimlog.copy()
    for c in [c for c in stimlog.columns if 'acqfr' in c]:
        stimlog[c] = stimlog[c] - 1
    return stimlog


def derive_trial_timing(stimlog, stim_locked_to_acqfr=True):
    """Derive (n_samp_isi, n_samp_stim) from the acqfr spans (analysis_for_images.py:1314-1325)."""
    stim_span = (stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).dropna().astype(int).to_numpy()
    isi_span = (stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).dropna().astype(int).to_numpy()
    if stim_locked_to_acqfr:
        n_samp_stim = int(np.bincount(stim_span).argmax())
    else:
        nz = np.bincount(stim_span).nonzero()[0]
        n_samp_stim = int(nz[0] if nz[0] != 0 else nz[1])
    nz = np.bincount(isi_span).nonzero()[0]
    n_samp_isi = int(nz[0] if nz[0] != 0 else nz[1])
    return n_samp_isi, n_samp_stim


def trim_to_imaged_trials(stimlog, n_frames, n_samp_isi, n_samp_stim):
    """Drop trials whose [pre-ISI | stim | post-ISI] window falls outside the recorded frames
    (null or out-of-range onsets). A dtype-safe stand-in for the abort handling in
    analysis_for_images.py:1274-1303.
    """
    onset = stimlog['acqfr_stim_i']
    fr_end = onset + n_samp_stim + n_samp_isi
    keep = (onset.notnull() & (fr_end <= n_frames) & (onset >= 0)).fillna(False)
    return stimlog[keep].reset_index(drop=True)


def _apply_eyepos_gate(ds, session_path, stimlog, framerate, mode, gate_kw):
    """Apply the eye-position gate as response-table exclusions (used when ``eye_gate_mode != 'none'``).

    Runs eyetracking.gate_trials_by_eyepos in the requested ``mode``, joins the gated eye trials to stimlog
    rows by stimulus-onset frame (eyetracking.calculate_eyepos_stimlog_keep), and marks the failed
    (condition, repeat) cells excluded via response_table.exclude_trials_by_stimlog. NO-OPS with a warning
    when the session has no usable eye recording, so requesting a gate never blocks a session that simply
    lacks eye data (modality heterogeneity is expected). Returns (dataset, summary)."""
    try:
        oc = eyetracking.load_eye_ai(session_path)
    except (FileNotFoundError, ValueError) as e:
        warn('eye_gate_mode={!r} requested but no usable eye data ({}); no eye exclusions applied.'
             .format(mode, e))
        return ds, {'mode': mode, 'applied': False, 'reason': str(e)}
    gate = eyetracking.gate_trials_by_eyepos(oc, mode=mode, framerate=framerate, **gate_kw)
    keep_rows, info = eyetracking.calculate_eyepos_stimlog_keep(oc, gate, stimlog)
    ds, dropped = response_table.exclude_trials_by_stimlog(ds, stimlog, keep_rows)
    info.update({'applied': True, 'n_excluded_cells': len(dropped)})
    return ds, info


def build_session_response_table(session_path, variant=None, baseline_method='medianbw',
                                 paradigm='auto', threshold_cellprob=0.0,
                                 eye_gate_mode='none', eye_gate_kw=None,
                                 neuropil_subtract=True, neucoeff=0.7,
                                 roi_subsample=None, roi_subsample_seed=0, equalize_repeats=False):
    """Load a session end-to-end into a response_table xarray Dataset.

    Orchestrates load_metadata -> load_suite2p -> compute_fluorescence_metrics -> load_stimlog ->
    acqfr correction / timing / trim -> response_table.build_response_table. Returns
    (dataset, context), where context holds the intermediates (md, s2p, traces, stimlog,
    n_samp_isi, n_samp_stim, stim_provenance). Paradigm-specific condition metadata (category,
    image name, etc.) is layered on by the paradigm driver; this orchestrator stays general.

    Eye-position gating (opt-in). With ``eye_gate_mode != 'none'`` the eye-position gate
    (eyetracking.gate_trials_by_eyepos) is run and its rejected trials are recorded in the table's
    ``excluded`` mask -- values are never NaN-ed, so exclusions stay reversible and visible. ``eye_gate_kw``
    passes mode parameters through to the gate (e.g. ``near_radius_v``, ``min_fraction``,
    ``min_eye_near_stim_sec``); a per-session summary lands in ``context['eye_gate']``. The DEFAULT ``'none'``
    changes nothing (no eye data is even loaded), keeping eye exclusion a deliberate choice; and a session
    with no eye recording is a warned no-op rather than an error. See eyetracking.GATE_MODES for the modes.

    Neuropil correction (on by default). ``neuropil_subtract=True`` applies ``Fc = F - neucoeff*(Fneu -
    median(Fneu))`` in compute_fluorescence_metrics before the baseline (v9 form; ``neucoeff`` default 0.7),
    so every downstream metric runs on the corrected trace. It no-ops when the extraction has no Fneu.npy.
    Pass ``neuropil_subtract=False`` for the uncorrected traces (the with/without side-by-side).
    ``context['neuropil']`` records what was applied.
    """
    md = load_metadata(session_path)
    s2p = load_suite2p(session_path, variant=variant, threshold_cellprob=threshold_cellprob)
    if roi_subsample is not None and s2p['Frois'].shape[0] > roi_subsample:
        # Random ROI subsample BEFORE building the trial tensor -- large-FOV sessions carry 5-16k ROIs whose full
        # (roi, cond, repeat, time) table is multi-GB, and the pairwise topography controls are O(n^2)-O(n^3).
        # Subsampling here (seeded, sorted to keep order) keeps the spatial point pattern representative.
        rng = np.random.default_rng(roi_subsample_seed)
        sel = np.sort(rng.choice(s2p['Frois'].shape[0], int(roi_subsample), replace=False))
        s2p = dict(s2p)
        s2p['Frois'] = s2p['Frois'][sel]
        s2p['Fneu'] = None if s2p['Fneu'] is None else s2p['Fneu'][sel]
        s2p['ROIs'] = s2p['ROIs'][sel]
        s2p['cellinds'] = s2p['cellinds'][sel]
    traces = compute_fluorescence_metrics(s2p['Frois'], md['framerate'], method=baseline_method,
                                          fneu=s2p['Fneu'], neucoeff=neucoeff,
                                          neuropil_subtract=neuropil_subtract)
    n_frames = s2p['Frois'].shape[1]

    stimlog, stim_prov = load_stimlog(session_path, paradigm=paradigm)
    stimlog = correct_acqfr_index(stimlog)
    n_samp_isi, n_samp_stim = derive_trial_timing(stimlog, md.get('stim_locked_to_acqfr', True))
    stimlog = trim_to_imaged_trials(stimlog, n_frames, n_samp_isi, n_samp_stim)

    ds = response_table.build_response_table(
        traces, stimlog, n_samp_isi, n_samp_stim, framerate=md['framerate'],
        equalize_repeats=equalize_repeats)
    context = {'md': md, 's2p': s2p, 'traces': traces, 'stimlog': stimlog,
               'n_samp_isi': n_samp_isi, 'n_samp_stim': n_samp_stim, 'stim_provenance': stim_prov,
               'neuropil': {'subtracted': bool(neuropil_subtract and s2p['Fneu'] is not None),
                            'neucoeff': neucoeff}}
    if eye_gate_mode != 'none':
        ds, context['eye_gate'] = _apply_eyepos_gate(
            ds, session_path, stimlog, md['framerate'], eye_gate_mode, eye_gate_kw or {})
    return ds, context
