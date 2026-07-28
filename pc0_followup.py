#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOM step-4 follow-up (#1): firm up the one topography lead — the large-scale spatial power in the dominant
tuning PC — on the NEUROPIL-CORRECTED (0.7) PD session (ZETA subset + all ROIs).

  1. IDENTIFY PC0/1/2 — per-condition loadings grouped by category, and per-ROI PC score vs a face-preference
     contrast — to see what the dominant tuning axis encodes.
  2. PER-PC-CALIBRATED template detector (each PC's null matched to its OWN spatial autocorrelation): says
     whether PC0's large-scale structure is just a smooth GRADIENT (p~0.5, reproduced by its null) or a genuine
     periodic/domain SCALE (p<0.05, a bump beyond the exponential).
  3. PC0 spatial autocorrelation (a0, lam0) directly.
  4. DRIFT check — temporal split-half stability (responsive set) + early-vs-late PC0-map correlation.
Run:  .venv/bin/python marmanalysis/pc0_followup.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import response_table as rt
import topography as tg

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
ZETA_CSV = 'tmp/responsiveness_pd.csv'


def _svd_scores(response):
    """(scores [n_roi, K], loadings Vt [K, n_cond], explained_var) of the mean-centred response."""
    R = np.nan_to_num(np.asarray(response, float) - np.nanmean(response, axis=0, keepdims=True))
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    return U * s, Vt, (s ** 2) / (s ** 2).sum()


def _safecorr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.nan


def _cat_masks(cats):
    c = np.array([str(x).lower() for x in cats])
    is_face = np.array(['face' in x for x in c])
    is_blank = np.array(['blank' in x for x in c])
    return is_face, (~is_face & ~is_blank), c


def identify(data, tag):
    resp, xy = data['response'], data['roi_xy_um']
    cats = data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()
    is_face, is_non, c = _cat_masks(cats)
    scores, Vt, ev = _svd_scores(resp)
    dist = tg.pairwise_distance(xy)
    fp = (np.nanmean(resp[:, is_face], 1) - np.nanmean(resp[:, is_non], 1)
          if is_face.any() and is_non.any() else np.full(resp.shape[0], np.nan))
    print('\n=== %s (n=%d) — PC identity ===' % (tag, resp.shape[0]))
    print('  categories:', {u: int((c == u).sum()) for u in pd.unique(c)})
    for k in range(3):
        by_cat = {u: round(float(np.nanmean(Vt[k][c == u])), 2) for u in pd.unique(c)}
        a_k, lam_k = tg._scalar_autocorrelation(scores[:, k], dist)
        print('  PC%d ev=%.2f | corr(score, face_pref)=%+.2f | autocorr a=%.2f lam=%.0fum | loading-by-cat=%s'
              % (k, ev[k], _safecorr(scores[:, k], fp), a_k, lam_k, by_cat))


def detector(data, tag):
    sfd = tg.spatial_frequency_detector(data['response'], data['roi_xy_um'],
                                        n_grid=24, n_pc=3, n_perm=500, min_wavelength_um=30)
    print('\n=== %s — per-PC template detector (null = each PC own autocorrelation) ===' % tag)
    for k in range(3):
        s = sfd[k]
        print('  PC%d ev=%.2f a=%.2f lam=%3.0fum: peak wl=%5.0fum  p=%.3f  %s'
              % (k, s['explained_var'], s['amplitude'], s['lambda_um'], s['peak_wavelength_um'], s['p'],
                 'periodic/domain scale' if s['p'] < 0.05 else 'gradient (no characteristic scale)'))


def drift(data, tag):
    ds, fr = data['ds'], data['framerate']
    cls = rt.classify_responses(ds, metric='Fzsc', framerate=fr, alpha=0.05)
    st = rt.calculate_temporal_split_stability(ds, metric='Fzsc', roi_mask=cls['responsive'])
    tr = rt.trial_response(ds, 'Fzsc').transpose('roi', 'condition', 'repeat').values
    nrep = tr.shape[2]; h = nrep // 2

    def pc0(m):
        M = np.nan_to_num(m - np.nanmean(m, 0, keepdims=True))
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        return U[:, 0] * s[0]

    with np.errstate(all='ignore'):
        r_el = _safecorr(pc0(np.nanmean(tr[:, :, :h], 2)), pc0(np.nanmean(tr[:, :, nrep - h:], 2)))
    print('\n=== %s — drift check ===' % tag)
    print('  temporal split-half (responsive n=%d): temporal=%.3f random=%.3f gap=%+.3f -> %s'
          % (st['n_roi_used'], st['temporal_median'], st['random_median'], st['gap'],
             'time-stable' if abs(st['gap']) < 0.05 else 'possible drift'))
    print('  PC0 map early-vs-late |corr| = %.3f (high -> stable, not drift/bleaching)' % abs(r_el))


def main():
    corr = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5,
                                    neuropil_subtract=True, neucoeff=0.7)
    n = corr['response'].shape[0]
    subsets = [('all', np.ones(n, bool))]
    if os.path.exists(ZETA_CSV):
        z = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if z.size == n:
            subsets.append(('ZETA', z))
    for sname, smask in subsets:
        d = rm.apply_roi_mask(corr, smask)
        identify(d, 'CORRECTED/' + sname)
        detector(d, 'CORRECTED/' + sname)
    drift(corr, 'CORRECTED/all (responsive)')


if __name__ == '__main__':
    main()
