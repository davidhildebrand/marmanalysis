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
from warnings import warn

import pandas as pd

import parsers


# Text-log parser per stimulus paradigm (the text format is paradigm-specific).
_PARADIGM_TEXT_PARSER = {
    'image': parsers.parse_log_stim_image,
    'dots': parsers.parse_log_stim_dots,
    'gratings': parsers.parse_log_stim_gratings,
}

# stimlog columns whose disagreement between sources actually matters for trial windowing.
_CRITICAL_COLS = ('cond', 'acqfr_stim_i', 'acqfr_stim_f', 'acqfr_isi_i', 'acqfr_isi_f')


def infer_paradigm(session_path):
    """Guess the stimulus paradigm from the session directory's 'stim<...>' token."""
    name = os.path.basename(os.path.normpath(session_path)).lower()
    if 'multimodal' in name:
        return 'multimodal'
    if 'image' in name:
        return 'image'
    if 'grating' in name:
        return 'gratings'
    if 'dot' in name:
        return 'dots'
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
            return parser(f.read())
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
        Stimulus paradigm selecting the text-log parser; 'auto' infers it from the directory name.
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
        paradigm = infer_paradigm(session_path)

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
