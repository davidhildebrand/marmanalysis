#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the parsers.py schema/dtype machinery added/fixed during the #2 extraction:
the canonical create_stimulus_record schema, normalize_stimlog_dtypes (nullable Int64 / float64 /
object), and the convert_stimulus_record dur_stim-from-times fallback.

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_parsers.py -v
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parsers

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'


def test_create_stimulus_record_schema_dtypes():
    log = parsers.create_stimulus_record(trials=4)
    assert len(log) == 4
    assert str(log['acqfr_stim_i'].dtype) == 'Int64'     # frame indices -> nullable Int64
    assert str(log['cond'].dtype) == 'Int64'
    assert str(log['dur_stim'].dtype) == 'float64'        # durations -> float64
    assert log['stim_mode'].dtype == object               # labels -> object
    assert log['image'].dtype == object


def test_normalize_stimlog_dtypes_casts_and_preserves():
    df = pd.DataFrame({
        'cond': [0.0, 1.0],
        'acqfr_stim_i': [10.0, 20.0],
        'dur_stim': ['1.0', '1.5'],          # stringy numeric -> float64
        'stim_class': ['image', 'image'],    # label -> stays object
        'dots_dir': [45.0, 90.0],            # not in the int/float schema -> left untouched
    })
    out = parsers.normalize_stimlog_dtypes(df)
    assert str(out['cond'].dtype) == 'Int64'
    assert str(out['acqfr_stim_i'].dtype) == 'Int64'
    assert str(out['dur_stim'].dtype) == 'float64'
    assert out['stim_class'].dtype == object
    assert str(out['dots_dir'].dtype) == 'float64'        # untouched (object_fill=False default)


def test_normalize_keeps_missing_as_na_in_int_column():
    out = parsers.normalize_stimlog_dtypes(pd.DataFrame({'acqfr_stim_i': [10.0, np.nan, 30.0]}))
    assert str(out['acqfr_stim_i'].dtype) == 'Int64'
    assert out['acqfr_stim_i'].isna().tolist() == [False, True, False]
    assert out['acqfr_stim_i'].iloc[0] == 10


def test_convert_derives_dur_stim_from_stim_times():
    """A CSV-like record with stim on/off TIMES but no dur_stim/stim_dur column: convert must
    derive dur_stim = t_stim_f - t_stim_i (the previously-unreachable fallback)."""
    df = pd.DataFrame({
        'trial_n': [0, 1], 'cond': [0, 1], 'image': ['a.png', 'b.png'],
        'acqfr_isi_i': [5, 15], 'acqfr_isi_f': [9, 19],
        'acqfr_stim_i': [10, 20], 'acqfr_stim_f': [15, 25],
        't_stim_i': [1.0, 2.0], 't_stim_f': [1.5, 2.6],
    })
    out = parsers.convert_stimulus_record(df)
    np.testing.assert_allclose(out['dur_stim'].astype(float).values, [0.5, 0.6])
    assert (out['stim_class'] == 'image').all()           # inferred for the legacy variant


def test_parse_log_stim_gratings_real():
    """The fixed parse_log_stim_gratings loads a real DriftingGratings text log into the canonical
    schema (acqfr_isi/stim from the AI-less ISI lines; float grating_ori, no int() crash)."""
    logs = [f for f in glob.glob(BASE + '/Larry/20241101d/*stimDriftingGratings8dirFF/*Grating*.log')
            if 'disptimes' not in f]
    if not logs:
        pytest.skip('grating log not present')
    sl = parsers.parse_log_stim_gratings(open(logs[0]).read())
    for c in ['cond', 'acqfr_isi_i', 'acqfr_isi_f', 'acqfr_stim_i', 'acqfr_stim_f',
              'grating_ori', 'grating_sf', 'grating_tf']:
        assert c in sl.columns and not sl[c].isnull().all(), 'missing/empty column ' + c
    assert (sl['stim_class'].dropna() == 'grating').all()
    assert (sl['stim_subclass'].dropna() == 'drifting').all()
    assert sl['grating_ori'].dropna().nunique() == 8            # 8 drift directions (8dirFF)
    assert sl['grating_ori'].dropna().between(0, 360).all()     # float ori parsed (no int() crash)


def test_assert_parser_progress_distinguishes_failures():
    """The shared conclusion-check helper raises a 'wrong parser' message when the paradigm's stim
    function never appeared, but a distinct 'aborted session' message when the right parser ran yet
    no trials were recorded (the Cadbury/20231018d case: ImageStim present, no stim-start lines)."""
    parsers._assert_parser_progress(True, True, 'image', 'ImageStim')        # both seen -> no raise
    with pytest.raises(Exception, match='Incorrect log parser chosen'):
        parsers._assert_parser_progress(False, False, 'image', 'ImageStim')  # stim func missing
    with pytest.raises(Exception, match='aborted'):
        parsers._assert_parser_progress(False, True, 'image', 'ImageStim')   # found func, no trials


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
