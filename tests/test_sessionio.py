#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for sessionio.load_stimlog (the layered-merge stimlog loader) and the
convert_stimulus_record backward-compat fix for pre-multimodal stimlogs.

The real-session cases are skipped automatically when the suite2p_results data isn't present, so
the convert_stimulus_record unit test still runs anywhere.

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_sessionio.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parsers
import sessionio

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'

# (expected base_source, expected paradigm, session path) spanning the stimlog format eras.
CASES = [
    ('csv', 'image',
     BASE + '/Cadbury/20231007d/153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'),
    ('text', 'image',
     BASE + '/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'),
    ('csv', 'image',
     BASE + '/Dali/20230810d/150108tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesSongFOBonly'),
    ('pickle', 'multimodal',
     BASE + '/Louwho/20240616d/172850tUTC_SP_depth250um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow028p4mW_stimMultimodalGratings'),
]

REQUIRED_COLS = ['cond', 'acqfr_stim_i', 'acqfr_stim_f', 'acqfr_isi_i', 'acqfr_isi_f',
                 'stim_mode', 'stim_class']


@pytest.mark.parametrize('expected_base, expected_paradigm, path', CASES,
                         ids=[c[2].split('/')[-2] + ':' + c[0] for c in CASES])
def test_load_stimlog_real_sessions(expected_base, expected_paradigm, path):
    if not os.path.isdir(path):
        pytest.skip('session data not present')
    stimlog, prov = sessionio.load_stimlog(path)
    assert prov['base_source'] == expected_base
    assert prov['paradigm'] == expected_paradigm
    assert len(stimlog) > 0
    for c in REQUIRED_COLS:
        assert c in stimlog.columns, 'missing column ' + c
        assert not stimlog[c].isnull().all(), 'all-null column ' + c


def test_convert_stimulus_record_infers_missing_stim_columns():
    """The image-only CSV variant lacks stim_mode/stim_class/stim_subclass; convert must infer
    them (default visual/image) instead of raising KeyError."""
    df = pd.DataFrame({
        'trial_n': [0, 1], 'cond': [0, 1],
        'image': ['a.png', 'b.png'], 'image_path': ['/x/a.png', '/x/b.png'],
        'acqfr_isi_i': [5, 15], 'acqfr_isi_f': [9, 19],
        'acqfr_stim_i': [10, 20], 'acqfr_stim_f': [15, 25],
        't_stim_i': [1.0, 2.0], 't_stim_f': [1.5, 2.5],
    })
    out = parsers.convert_stimulus_record(df)
    assert (out['stim_mode'] == 'visual').all()
    assert (out['stim_class'] == 'image').all()
    assert 'stim_subclass' in out.columns
    assert list(out['cond']) == [0, 1]
    assert list(out['acqfr_stim_i']) == [10, 20]


def test_convert_stimulus_record_preserves_explicit_columns():
    """When stim_mode/stim_class are already present, they must be left untouched."""
    df = pd.DataFrame({
        'trial_n': [0], 'cond': [0],
        'stim_mode': ['audio'], 'stim_class': ['tone'], 'stim_subclass': [None],
        'acqfr_isi_i': [5], 'acqfr_isi_f': [9], 'acqfr_stim_i': [10], 'acqfr_stim_f': [15],
        'f': [4000], 'lev': [60],
    })
    out = parsers.convert_stimulus_record(df)
    assert out.loc[0, 'stim_mode'] == 'auditory'   # audio -> normalized to 'auditory'
    assert out.loc[0, 'stim_class'] == 'tone'


def test_compute_f0_dff_shapes_and_keys():
    rng = np.random.default_rng(0)
    frois = (rng.standard_normal((8, 2000)).astype('float32') * 40 + 500)
    traces = sessionio.compute_f0_dff(frois, framerate=6.0, window=60)
    assert set(traces) == {'FdFF', 'Fzsc', 'F0', 'Fraw'}
    for k in traces:
        assert traces[k].shape == frois.shape
    assert traces['Fraw'] is frois
    assert np.isfinite(traces['FdFF']).all()
    assert np.isfinite(traces['Fzsc']).all()


def test_load_metadata_and_suite2p():
    path = CASES[1][2]  # Cadbury/20221016d (small FOV)
    if not os.path.isdir(path):
        pytest.skip('session data not present')
    md = sessionio.load_metadata(path)
    assert md['framerate'] > 0 and 'fov' in md

    s2p = sessionio.load_suite2p(path)
    n_rois, n_frames = s2p['Frois'].shape
    assert n_rois > 0 and n_frames > 0
    assert len(s2p['ROIs']) == n_rois
    assert tuple(s2p['fov_image'].shape) == tuple(s2p['fov_size'])
    assert s2p['badframes'].ndim == 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
