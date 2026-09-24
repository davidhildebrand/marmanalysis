#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-SPECIFIC topography vs generic smoothness -- follow-up to feature_topography_v2 (2026-09-21).

v2 found most tuning directions 'topographic' against a position-shuffle null: 15-18/20 PCs of every image-feature
basis, all 60 top SOM units, 16/20 neural PCs. A position-shuffle null cannot tell a FEATURE-SPECIFIC map from ONE
shared short-range smooth component that every linear projection of the activity inherits (and the earlier
variogram result -- full-tuning topography = generic smoothness -- points to the latter). Two controls:

  (1) RANDOM-PROJECTION null: T (near<75um minus far>150um z-product) for random unit directions in condition
      space = what a GENERIC tuning direction looks like spatially. A feature is SPECIFICALLY mapped only if its
      T exceeds this distribution (report counts above its 95th / 99th percentiles per basis).
  (2) NEUCOEFF sensitivity: T for the face axis (neural PC0), the median relu7 feature map, and the median random
      projection at neuropil coefficient none / 0.7 / 0.85 / 1.0. If the shared smoothness is residual neuropil,
      T shrinks toward 0 with stronger correction; genuine local functional clustering survives.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/feature_specificity.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import feature_topography_v2 as v2
import response_matrix as rm
import som
import topography as tg

R_PROJ = 500


def build(neucoeff):
    if neucoeff is None:
        return rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3, neuropil_subtract=False)
    return rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3,
                                    neuropil_subtract=True, neucoeff=neucoeff)


def prep(data, zeta):
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    dist = tg.pairwise_distance(xy)
    iu = np.triu_indices(resp.shape[0], 1)
    dd = dist[iu]
    return resp, dist, iu, dd < v2.NEAR, dd > v2.FAR


def zcols(M):
    M = np.nan_to_num(np.asarray(M, float))
    return (M - M.mean(0)) / (M.std(0) + 1e-12)


def T_of(maps, iu, near, far):
    return v2.topo_stat(zcols(maps), iu, near, far)


def random_projection_T(resp, iu, near, far, n=R_PROJ, seed=0):
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((R.shape[1], n))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    return T_of(R @ U, iu, near, far)


def main():
    zeta = pd.read_csv(v2.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    data = build(0.7)
    resp, dist, iu, near, far = prep(data, zeta)
    paths = som.condition_image_paths(data, v2.STIM_DIR)
    images = dnn_som.load_images_rgb(paths)
    model, _ = dnn_som.prep_dnn_model('cpu')
    feats = {nm: dnn_som.extract_dnn_features(model, layer, images, 'cpu').numpy() for nm, layer in v2.LAYERS.items()}
    sca = dnn_som.som_sca(dnn_som.load_som(som.DEFAULT_SOM), feats['relu7']).numpy()

    # ---- (1) random-projection null ----
    Trand = random_projection_T(resp, iu, near, far)
    q95, q99 = np.percentile(Trand, [95, 99])
    print('RANDOM-PROJECTION null (corrected 0.7, ZETA n=%d, %d random tuning directions):' % (resp.shape[0], R_PROJ))
    print('  T mean=%+.4f  median=%+.4f  p95=%+.4f  p99=%+.4f  max=%+.4f' % (Trand.mean(), np.median(Trand), q95, q99, Trand.max()))
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Un, Sn, _ = np.linalg.svd(R, full_matrices=False)
    bases = {'neural-PC': Un[:, :20] * Sn[:20]}
    for nm in v2.LAYERS:
        fp, _ = v2.pcs(feats[nm])
        bases[nm + '-PC'] = np.column_stack([v2.rowcorr(resp, fp[:, k]) for k in range(fp.shape[1])])
    pref_som = np.column_stack([v2.rowcorr(resp, sca[:, u]) for u in range(sca.shape[1])])
    bases['SOM-unit top60'] = pref_som[:, np.argsort(-np.nanmedian(np.abs(pref_som), 0))[:60]]
    print('\n  %-16s %3s  %9s %9s | %-14s %-14s  (feature-SPECIFIC = beyond a generic direction)'
          % ('basis', 'K', 'T median', 'T max', '> random p95', '> random p99'))
    for nm, maps in bases.items():
        T = T_of(maps, iu, near, far)
        print('  %-16s %3d  %+9.4f %+9.4f |   %2d / %-3d      %2d / %-3d    best k=%d'
              % (nm, len(T), np.median(T), T.max(), int((T > q95).sum()), len(T), int((T > q99).sum()), len(T),
                 int(np.argmax(T))))

    # ---- (2) neuropil-coefficient sensitivity ----
    print('\nNEUCOEFF sensitivity of short-range topography (ZETA; T = near-minus-far z-product):')
    print('  %-6s  %-20s  %-24s  %-22s  %s' % ('coeff', 'T face axis (nPC0)', 'median T relu7 feat maps',
                                              'median T random proj', 'D_activity vs D_phys'))
    fp7, _ = v2.pcs(feats['relu7'])
    for cf in (None, 0.7, 0.85, 1.0):
        dd_ = data if cf == 0.7 else build(cf)
        r_, dist_, iu_, near_, far_ = prep(dd_, zeta)
        Rc = np.nan_to_num(r_ - np.nanmean(r_, 0, keepdims=True))
        Uc, Sc, _ = np.linalg.svd(Rc, full_matrices=False)
        t_face = T_of(Uc[:, :1] * Sc[:1], iu_, near_, far_)[0]
        t_feat = float(np.median(T_of(np.column_stack([v2.rowcorr(r_, fp7[:, k]) for k in range(fp7.shape[1])]),
                                      iu_, near_, far_)))
        t_rand = float(np.median(random_projection_T(r_, iu_, near_, far_, n=200)))
        dact = pdist(v2.znorm_rows(r_))
        ok = (dist_[iu_] >= v2.MIND) & (dist_[iu_] <= v2.MAXD)
        r_sv = spearmanr(dact[ok], dist_[iu_][ok]).statistic
        print('  %-6s  %+20.4f  %+24.4f  %+22.4f  %+.3f'
              % ('none' if cf is None else '%.2f' % cf, t_face, t_feat, t_rand, r_sv))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
