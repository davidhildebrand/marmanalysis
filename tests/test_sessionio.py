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


CONTINUATION = (BASE + '/Dali/20230910d/180205tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_'
                'pow060p4mW_stimImagesFOBmany_Continuation')


def test_shift_continuation_rows_synthetic():
    """Rows whose onset lies beyond the previous recording are kept and every acqfr column is re-based."""
    log = pd.DataFrame({'cond': [0, 1, 2, 3],
                        'acqfr_isi_i': [90, 95, 101, 110], 'acqfr_isi_f': [94, 99, 104, 114],
                        'acqfr_stim_i': [95, 100, 105, 115], 'acqfr_stim_f': [98, 103, 108, 118]})
    out = sessionio.shift_continuation_rows(log, 100)
    assert out['cond'].tolist() == [2, 3]
    assert out['acqfr_stim_i'].tolist() == [5, 15]
    assert out['acqfr_isi_i'].tolist() == [1, 10]
    assert out['acqfr_stim_f'].tolist() == [8, 18]
    assert len(sessionio.shift_continuation_rows(log, 200)) == 0


def test_check_continuation_rows_synthetic():
    """A trial shown during the acquisition gap (logged at the resumed counter but off the frame clock) and a trial
    whose ISI was not imaged are dropped; the rest are kept."""
    fr = 4.0
    onset = np.array([5, 17, 29, 41, 53], float)                 # local frames
    t = 100.0 + onset / fr
    t[0] -= 40.0                                                   # shown 40 s earlier, during the gap
    log = pd.DataFrame({'cond': [0, 1, 2, 3, 4], 'acqfr_isi_i': [0, 9, 21, 33, 45], 'acqfr_isi_f': [4, 16, 28, 40, 52],
                        'acqfr_stim_i': onset.astype(int), 'acqfr_stim_f': onset.astype(int) + 4, 't_stim_i': t})
    kept, chk = sessionio.check_continuation_rows(log, fr)
    assert kept['cond'].tolist() == [1, 2, 3, 4]
    assert chk['n_off_clock'] == 1 and chk['n_isi_not_imaged'] == 1 and chk['n_dropped'] == 1


def test_continuation_session_real():
    """Dali 20230910d: the log-less _Continuation folder is served from 170833tUTC's log (14,200-frame offset)."""
    if not os.path.isdir(CONTINUATION):
        pytest.skip('session data not present')
    assert sessionio.is_continuation_session(CONTINUATION)
    parent = sessionio.find_continuation_parent(CONTINUATION)
    assert parent is not None and os.path.basename(parent).startswith('170833tUTC')
    stimlog, prov = sessionio.load_stimlog(CONTINUATION)
    assert prov['continuation_of'] == parent
    assert prov['frame_offset'] == 14200
    assert prov['n_rows_source'] == 1248
    # 60 trials lie beyond the first recording; the first ran its ISI during the 44 s gap (baseline not imaged) and is
    # dropped; every other trial sits on the continuation's frame clock and none was shown during the gap
    chk = prov['continuation_checks']
    assert chk['n_isi_not_imaged'] == 1 and chk['n_off_clock'] == 0 and chk['n_duplicate_onsets'] == 0
    assert chk['max_clock_residual_frames'] < 1.0
    assert prov['n_rows_kept'] == len(stimlog) == 59
    assert int(stimlog['acqfr_isi_i'].min()) >= 1 and int(stimlog['acqfr_stim_f'].max()) <= 1000
    # the parent's own loading is untouched
    full, prov_full = sessionio.load_stimlog(parent)
    assert len(full) == 1248 and 'continuation_of' not in prov_full
    # end-to-end: the continuation builds a (small) response table from its own 1,000 frames
    ds, ctx = sessionio.build_session_response_table(CONTINUATION, equalize_repeats=True)
    assert ctx['stim_provenance']['frame_offset'] == 14200
    assert 0 < ds.sizes['condition'] <= 60
    assert ctx['s2p']['Frois'].shape[1] == 1000


def test_infer_paradigm_tokens_and_log_precedence():
    """infer_paradigm recognizes each stimulus token and prefers the (more reliable) log filename
    over the session directory name, which is occasionally mislabeled."""
    assert sessionio._paradigm_from_name('x_Stimulus_Images.log') == 'image'
    assert sessionio._paradigm_from_name('x_Stimulus_DriftingGratingsFullField.log') == 'gratings'
    assert sessionio._paradigm_from_name('x_Stimulus_MovingDots.log') == 'dots'
    assert sessionio._paradigm_from_name('x_Stimulus_MultimodalGratings.log') == 'multimodal'
    assert sessionio._paradigm_from_name('x_Stimulus_Auditory_max15.log') == 'auditory'
    assert sessionio._paradigm_from_name('x_Stimulus_Dummy.log') == 'dummy'
    assert sessionio._paradigm_from_name('x_Stimulus_Mystery.log') == 'unknown'
    # log filename wins over a mislabeled directory; directory is the fallback when no log given.
    assert sessionio.infer_paradigm('/d/Larry/20241101d/161123tUTC_stimMovingDots',
                                    '161123tUTC_Stimulus_DriftingGratingsFullField.log') == 'gratings'
    assert sessionio.infer_paradigm('/d/Larry/20241101d/161123tUTC_stimDriftingGratings8dirFF') == 'gratings'


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


def test_compute_fluorescence_metrics():
    rng = np.random.default_rng(0)
    frois = (rng.standard_normal((8, 2000)).astype('float32') * 40 + 500)
    traces = sessionio.compute_fluorescence_metrics(frois, framerate=6.0, window=60)
    assert set(traces) == {'FdFF', 'Fzsc', 'F0', 'Fraw'}
    for k in traces:
        assert traces[k].shape == frois.shape
    assert traces['Fraw'] is frois
    assert np.isfinite(traces['FdFF']).all() and np.isfinite(traces['Fzsc']).all()

    # value invariants (non-circular): FdFF = (Fraw - F0)/F0  <=>  Fraw == F0 * (1 + FdFF)
    np.testing.assert_allclose(traces['Fraw'], traces['F0'] * (1 + traces['FdFF']), rtol=1e-4)
    # Fzsc is z-scored per ROI: each ROI trace has ~0 mean and unit std (ddof=0)
    np.testing.assert_allclose(traces['Fzsc'].mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(traces['Fzsc'].std(axis=1), 1, atol=1e-4)


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
