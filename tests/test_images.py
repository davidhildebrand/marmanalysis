#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the image-paradigm driver (images.py): classify_image on real filenames, and
end-to-end d'/FSI/ANOVA parity against an INDEPENDENT inline reference on real sessions.

The classifier is checked directly on representative filenames; the stats are checked against a
from-scratch reproduction (independent per-condition response vectors + the muR/sigma/d'/FSI math
of analysis_for_images.py:1976-2036, and a per-ROI f_oneway computed straight from the traces).

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_images.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import f_oneway

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import images
import sessionio

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'

SESSIONS = [
    ('Cadbury', '20231007d',
     '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'),
    ('Cadbury', '20221016d',
     '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'),
    ('Dali', '20230810d',
     '150108tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesSongFOBonly'),
]

# (basename, expected cat, expected cond) -- real Freiwald-FOB and Song filenames.
CLASSIFY_CASES = [
    ('FreiwaldFOB2012_Human_Head_13_erode3px', b'face_hum', b'fh13e'),
    ('FreiwaldFOB2012_MacaqueRhesus_Head_11_erode3px', b'face_rhe', b'fr11e'),
    ('FreiwaldFOB2018_Marm_Head_Hunter_1', b'face_mrm', b'fmHun01'),
    ('FreiwaldFOB2018_Marm_Body_Fordham_8_erode3px', b'body_mrm', b'bmFor08e'),
    ('FreiwaldFOB2018_Objects_Manmade2_6_erode3px', b'obj', b'om2006e'),
    ('FreiwaldFOB2018_Objects_FruitVeg2_7_erode3px', b'food', b'vf2007e'),
    ('m20', b'face_mrm', b'Sm20'),
    ('u18', b'obj', b'Su18'),
    ('b9', b'body_mrm', b'Sb09'),
    ('blank', b'blank', b'blank'),
]


@pytest.mark.parametrize('imn, cat, cond', CLASSIFY_CASES, ids=[c[0][:24] for c in CLASSIFY_CASES])
def test_classify_image(imn, cat, cond):
    got_cond, got_cat, _, _, _, _ = images.classify_image(imn)
    assert got_cat == cat
    assert got_cond == cond


def test_marm_head_pose():
    # inverted marmoset head -> view 9 -> roll 180; a yaw view -> nonzero yaw.
    _, cat, _, p, y, r = images.classify_image('FreiwaldFOB2018_Marm_Head_Hunter_5')
    assert cat == b'face_mrm' and (p, y, r) == (0, -90, 0)


def _resp_vect(traces, stimlog, n_samp_stim, metric):
    """Independent per-(roi, condition) stim-window mean from the traces."""
    conds = np.unique(stimlog['cond'].values)
    out = np.full((traces[metric].shape[0], conds.size), np.nan)
    for ci, c in enumerate(conds):
        onsets = stimlog.loc[stimlog['cond'] == c, 'acqfr_stim_i'].astype(int).to_numpy()
        out[:, ci] = np.mean([traces[metric][:, o:o + n_samp_stim].mean(axis=1) for o in onsets], axis=0)
    return out


def _ref_dprime_fsi(resp, is_face, is_nonface, is_nonface_object):
    """Inline d'/FSI per analysis_for_images.py:2010, 2028-2035."""
    mu_f = resp[:, is_face].mean(1)
    mu_nf = resp[:, is_nonface].mean(1)
    mu_o = resp[:, is_nonface_object].mean(1)
    s_f = resp[:, is_face].std(1)
    s_nf = resp[:, is_nonface].std(1)
    dprime = (mu_f - mu_nf) / np.sqrt((s_f ** 2 + s_nf ** 2) / 2)
    fsi = np.full(resp.shape[0], np.nan)
    same = np.sign(mu_f) == np.sign(mu_o)
    fsi[same] = (mu_f[same] - mu_o[same]) / (mu_f[same] + mu_o[same])
    fsi[np.sign(mu_f) > np.sign(mu_o)] = 1.0
    fsi[np.sign(mu_f) < np.sign(mu_o)] = -1.0
    return dprime, fsi


def _ref_anova(traces, stimlog, n_samp_stim, metric='Fzsc'):
    conds = np.unique(stimlog['cond'].values)
    per_cond = []
    for c in conds:
        onsets = stimlog.loc[stimlog['cond'] == c, 'acqfr_stim_i'].astype(int).to_numpy()
        per_cond.append(np.concatenate([traces[metric][:, o:o + n_samp_stim] for o in onsets], axis=1))
    n_roi = traces[metric].shape[0]
    p = np.full(n_roi, np.nan)
    for r in range(n_roi):
        _, p[r] = f_oneway(*[pc[r] for pc in per_cond])
    return p


@pytest.mark.parametrize('a, d, s', SESSIONS, ids=[x[1] for x in SESSIONS])
def test_dprime_fsi_anova_parity(a, d, s):
    path = os.path.join(BASE, a, d, s)
    if not os.path.isdir(path):
        pytest.skip('session data not present')
    ds, ctx = sessionio.build_session_response_table(path)
    ds = images.attach_condition_metadata(ds, images.build_condition_metadata(ctx['stimlog']))

    dprime_new = images.face_dprime(ds, 'Fzsc').values
    fsi_new = images.face_selectivity_index(ds, 'Fzsc').values
    p_new = images.responsive_anova(ds, 'Fzsc')

    is_face, is_nonface, is_nfo = images.supercategory_bools(ds['cat'].values)
    resp = _resp_vect(ctx['traces'], ctx['stimlog'], ctx['n_samp_stim'], 'Fzsc')
    dprime_ref, fsi_ref = _ref_dprime_fsi(resp, is_face, is_nonface, is_nfo)
    p_ref = _ref_anova(ctx['traces'], ctx['stimlog'], ctx['n_samp_stim'], 'Fzsc')

    assert np.allclose(dprime_new, dprime_ref, rtol=1e-4, atol=1e-5, equal_nan=True)
    assert np.allclose(fsi_new, fsi_ref, rtol=1e-4, atol=1e-5, equal_nan=True)
    assert np.allclose(p_new, p_ref, rtol=1e-4, atol=1e-6, equal_nan=True)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
