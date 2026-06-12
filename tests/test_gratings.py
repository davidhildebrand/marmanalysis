#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the gratings driver (gratings.py): a synthetic check that orientation_selectivity
recovers a known orientation-tuned ROI, and an end-to-end run of process_session on a real
multimodal-gratings session (osi only, for speed).

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_gratings.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gratings
import response_table as rt

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'
GRATINGS_SESSION = (BASE + '/Louwho/20240616d/172850tUTC_SP_depth250um_fov0636x1500um_'
                    'res2p00x2p00umpx_fr10p334Hz_pow028p4mW_stimMultimodalGratings')


def test_orientation_selectivity_recovers_tuned_roi():
    n_dir, n_reps, n_isi, n_stim, n_roi = 16, 4, 2, 2, 4
    directions = np.arange(0.0, 360.0, 22.5)
    # per-(roi, direction) target response: ROI 0 orientation-tuned at 45 deg, the rest flat.
    target = np.ones((n_roi, n_dir))
    target[0] = np.cos(np.radians(directions - 45.0)) ** 2

    cond_seq = np.tile(np.arange(n_dir), n_reps)
    onsets = 5 + 6 * np.arange(len(cond_seq))
    stimlog = pd.DataFrame({
        'cond': cond_seq, 'acqfr_stim_i': onsets,
        'stim_class': 'grating', 'stim_subclass': 'drifting',
        'grating_ori': directions[cond_seq], 'grating_sf': 1.0, 'grating_tf': 4.0,
    })
    n_frames = int(onsets.max() + n_stim + n_isi + 3)
    traces = {'Fzsc': np.zeros((n_roi, n_frames))}
    for c, o in zip(cond_seq, onsets):
        traces['Fzsc'][:, o:o + n_stim] = target[:, c][:, None]

    gr = gratings.select_drifting_gratings(stimlog)
    assert len(gr) == n_dir * n_reps
    ds = rt.build_response_table(traces, gr, n_isi, n_stim)
    ds = gratings.attach_condition_metadata(ds, gratings.build_condition_metadata(gr))

    tun = gratings.orientation_selectivity(ds, 'Fzsc', compute_dsi=True)
    assert tun['osi'][0] > tun['osi'][1:].max()              # tuned ROI is the most selective
    assert abs(tun['ori_pref'][0] - 45.0) < 5.0              # preferred orientation ~45 deg
    assert np.all((tun['osi'] >= 0) & (tun['osi'] <= 1.0001))


def test_process_session_real_multimodal_gratings():
    if not os.path.isdir(GRATINGS_SESSION):
        pytest.skip('session data not present')
    res = gratings.process_session(GRATINGS_SESSION, compute_dsi=False)   # osi only for speed
    ds, tun = res['dataset'], res['tuning']
    assert {'direction', 'orientation', 'sf', 'tf'}.issubset(ds.coords)
    assert ds.sizes['condition'] == 16                       # 16 drift directions selected
    n_roi = ds.sizes['roi']
    assert tun['osi'].shape == (n_roi,)
    assert np.all((tun['osi'] >= 0) & (tun['osi'] <= 1.0001))
    assert np.all((tun['ori_pref'] >= 0) & (tun['ori_pref'] < 180))
    assert np.nanmax(tun['osi']) > 0.2                       # at least some orientation-selective cells


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
