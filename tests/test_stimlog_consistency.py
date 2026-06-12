#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistency check: the ORIGINAL analysis_for_images.py stimlog loading vs sessionio.load_stimlog.

Reproduces the original in-script stimlog loading (analysis_for_images.py:829-880) -- choose the
structured source (pickle>h5>csv) -> convert_stimulus_record, else parse the text log; fill the
dur_* columns; conditionally backfill from the text log -- and compares its result column-by-column
against the new layered-merge loader. The windowing-critical columns must match exactly; other
differences (e.g. the new loader's always-on text backfill filling extra cells) are reported.

Run:  ../.venv/bin/python marmanalysis/tests/test_stimlog_consistency.py
  or: ../.venv/bin/python -m pytest marmanalysis/tests/test_stimlog_consistency.py -v
"""

import glob
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parsers
import sessionio

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'

SESSIONS = [
    BASE + '/Cadbury/20231007d/153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany',
    BASE + '/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly',
    BASE + '/Dali/20230810d/150108tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesSongFOBonly',
]

# Columns that drive trial windowing + condition identity -- these MUST match exactly.
CRITICAL_COLS = ['cond', 'acqfr_isi_i', 'acqfr_isi_f', 'acqfr_stim_i', 'acqfr_stim_f',
                 'stim_mode', 'stim_class', 'image', 'image_path']
# Scalar columns compared for the informational report (skip tuple/array columns like pos/size).
REPORT_COLS = CRITICAL_COLS + ['dur_stim', 'dur_isi_pre']


def load_stimlog_original(session_path):
    """Replicate the stimlog loading of analysis_for_images.py:829-880 (image paradigm)."""
    session_logs = [f for f in glob.glob(os.path.join(session_path, '*.log'))
                    if re.search(r'.*^((?!disptimes).)*$', f) and os.path.isfile(f)]
    session_log = open(session_logs[0]).read() if session_logs else None

    sfiles = [f for f in glob.glob(os.path.join(session_path, '*_stimlog.*')) if os.path.isfile(f)]
    pkls = [f for f in sfiles if f.endswith(('.pickle', '.pkl', '.p'))]
    hdf5s = [f for f in sfiles if f.endswith(('.h5', '.hdf5'))]
    csvs = [f for f in sfiles if f.endswith('.csv')]
    if pkls:
        stimlog = parsers.convert_stimulus_record(pd.read_pickle(pkls[0]))
    elif hdf5s:
        stimlog = parsers.convert_stimulus_record(pd.read_hdf(hdf5s[0]))
    elif csvs:
        stimlog = parsers.convert_stimulus_record(pd.read_csv(csvs[0]))
    elif session_log is not None:
        stimlog = parsers.parse_log_stim_image(session_log)
    else:
        raise RuntimeError('no stimlog source')

    if stimlog['dur_isi_pre'].isnull().values.any():
        if not stimlog['t_isi_i'].isnull().values.any() and not stimlog['t_isi_f'].isnull().values.any():
            stimlog['dur_isi_pre'] = stimlog['t_isi_f'] - stimlog['t_isi_i']
    if stimlog['dur_stim'].isnull().values.any():
        if not stimlog['t_stim_i'].isnull().values.any() and not stimlog['t_stim_f'].isnull().values.any():
            stimlog['dur_stim'] = stimlog['t_stim_f'] - stimlog['t_stim_i']
    if stimlog['dur_isi_post'].isnull().values.any():
        if not stimlog['t_isi_i'].isnull().values.any() and not stimlog['t_isi_f'].isnull().values.any():
            for t in range(len(stimlog['dur_isi_post']) - 1):
                stimlog.at[t, 'dur_isi_post'] = stimlog['t_isi_f'].loc[t + 1] - stimlog['t_isi_i'].loc[t + 1]
    if stimlog.isnull().values.any() and session_log is not None:
        stimlog.update(parsers.parse_log_stim_image(session_log), overwrite=False)
    return stimlog.reset_index(drop=True)


def column_diffs(a, b, cols):
    """Per-column count of cells that differ between two stimlogs (both-null counts as equal)."""
    diffs = {}
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        neq = ~((x.isna() & y.isna()) | (x.astype(object) == y.astype(object)))
        diffs[c] = int(pd.Series(neq).fillna(True).sum())
    return diffs


@pytest.mark.parametrize('path', SESSIONS, ids=[p.split('/')[-2] for p in SESSIONS])
def test_stimlog_matches_original(path):
    if not os.path.isdir(path):
        pytest.skip('session data not present')
    original = load_stimlog_original(path)
    new, _ = sessionio.load_stimlog(path)
    assert len(original) == len(new), 'row count differs: {} vs {}'.format(len(original), len(new))
    diffs = column_diffs(original, new, REPORT_COLS)
    critical = {c: n for c in CRITICAL_COLS for n in [diffs.get(c, 0)] if n}
    assert not critical, 'critical-column mismatches vs original: {}'.format(critical)


if __name__ == '__main__':
    for path in SESSIONS:
        print('\n' + '=' * 78)
        print('/'.join(path.split('/')[-2:])[:60])
        if not os.path.isdir(path):
            print('  (data not present, skipped)')
            continue
        original = load_stimlog_original(path)
        new, prov = sessionio.load_stimlog(path)
        print('  rows: original={}  new={}  | new base={}'.format(len(original), len(new), prov['base_source']))
        diffs = column_diffs(original, new, REPORT_COLS)
        for c in REPORT_COLS:
            if c in diffs:
                flag = '  <-- CRITICAL' if (c in CRITICAL_COLS and diffs[c]) else ''
                mark = 'OK ' if diffs[c] == 0 else 'DIFF'
                print('    {:14s} {:4s} ({} cells differ){}'.format(c, mark, diffs[c], flag))
        crit = {c: diffs.get(c, 0) for c in CRITICAL_COLS if diffs.get(c, 0)}
        print('  CRITICAL COLUMNS:', 'ALL MATCH' if not crit else 'MISMATCH {}'.format(crit))
