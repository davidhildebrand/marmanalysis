#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-space TOPOGRAPHY, v2 -- DH reframe (2026-09-21): the question is "are image FEATURES represented
TOPOGRAPHICALLY?" (smoothly organised in cortex), not "periodically". On the corrected (0.7) PD session, ZETA set.

FEATURE BASES (each yields per-image feature coordinates):
   neural-PC  the neural-tuning PCs themselves (reference)
   relu7-PC   AlexNet classifier.5 (Doshi & Konkle's layer), top-20 PCs of the 60 images' features
   relu6-PC   AlexNet classifier.2, top-20 PCs
   pool5-PC   AlexNet features.12 (conv5 max-pool, 9216-D; mid-level features), top-20 PCs
   SOM-unit   the 400 shipped SOM units' simulated cortical activations (each unit = a learned feature detector)
PER-ROI FEATURE MAPS: preference_k = corr(ROI tuning over images, feature-k coordinates)  [univariate], and for the
   PC bases also a RIDGE encoding model  tuning ~ features @ w  (multivariate weight maps).
TOPOGRAPHY TEST per map (fast, vectorised, no curve fit):
   T = mean over NEAR pairs (d < 75 um) of z_i*z_j  -  mean over FAR pairs (d > 150 um)   -- "nearby ROIs share
   this feature preference"; null = position shuffle (300 perms); BH-FDR across the basis. a/lambda (exponential
   autocorrelation fit) reported for the observed best map. This replaces the periodicity detector as the primary.
GEOMETRY (Cell-2025 style, all on z-scored vectors so Euclidean ~ correlation distance): D_featpref vs D_phys
   (is the feature embedding spatially organised), and D_featpref vs D_activity (how much of ROI-ROI activity
   similarity this basis explains).
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/feature_topography_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import false_discovery_control, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import response_matrix as rm
import som
import topography as tg

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
STIM_DIR = 'stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly'
ZETA_CSV = 'tmp/responsiveness_pd.csv'
LAYERS = {'relu7': 'classifier.5', 'relu6': 'classifier.2', 'pool5': 'features.12'}
K, N_PERM, NEAR, FAR, MIND, MAXD = 20, 300, 75.0, 150.0, 15.0, 600.0


def znorm_rows(M):
    M = np.nan_to_num(np.asarray(M, float))
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def rowcorr(A, v):
    A, v = np.asarray(A, float), np.asarray(v, float)
    Az, vz = A - np.nanmean(A, 1, keepdims=True), v - v.mean()
    with np.errstate(all='ignore'):
        return np.nansum(Az * vz, 1) / np.sqrt(np.nansum(Az ** 2, 1) * (vz ** 2).sum())


def fdr(p):
    p = np.asarray(p, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if ok.sum():
        out[ok] = false_discovery_control(p[ok], method='bh')
    return out


def pcs(F, k=K):
    Fc = F - F.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Fc, full_matrices=False)
    kk = min(k, S.size)
    return U[:, :kk] * S[:kk], (S[:kk] ** 2) / (S ** 2).sum()


def ridge_weights(resp, feat, lam=1.0):
    """Encoding model per ROI: tuning (n_img) ~ z(features) (n_img x K) @ w ; returns W (n_roi x K)."""
    Fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    R = np.nan_to_num(resp - np.nanmean(resp, 1, keepdims=True))
    return np.linalg.solve(Fz.T @ Fz + lam * np.eye(Fz.shape[1]), Fz.T @ R.T).T


def topo_stat(Z, iu, near, far):
    prod = Z[iu[0]] * Z[iu[1]]                      # (n_pairs, K)
    return prod[near].mean(0) - prod[far].mean(0)


def topo_test(maps, dist, n_perm=N_PERM, seed=0):
    """Per-map near-minus-far z-product topography statistic vs a position-shuffle null. Returns (T, p, null_mean)."""
    M = np.nan_to_num(np.asarray(maps, float))
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    n = Z.shape[0]
    iu = np.triu_indices(n, 1); d = dist[iu]
    near, far = d < NEAR, d > FAR
    T = topo_stat(Z, iu, near, far)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, Z.shape[1]))
    for b in range(n_perm):
        null[b] = topo_stat(Z[rng.permutation(n)], iu, near, far)
    return T, (1 + (null >= T).sum(0)) / (1 + n_perm), null.mean(0)


def report_basis(name, maps, ev, dist, ref):
    T, p, null = topo_test(maps, dist)
    q = fdr(p)
    kb = int(np.nanargmin(p))
    a_b, lam_b = tg._scalar_autocorrelation(np.nan_to_num(maps[:, kb]), dist)
    print('  %-16s K=%3d | mapped q<.05: %2d (p<.05: %2d) | best %2d: T=%+.4f (null %+.4f) p=%.3f q=%.3f a=%.2f lam=%3.0fum%s'
          % (name, maps.shape[1], int(np.nansum(q < 0.05)), int(np.nansum(p < 0.05)), kb, T[kb], null[kb], p[kb], q[kb],
             a_b, lam_b, ('  ev=%.2f' % ev[kb]) if ev is not None else ''))
    dfe = pdist(znorm_rows(maps)); ok = ref['ok']
    print('  %-16s      D_featpref vs D_phys r=%+.3f | D_featpref vs D_activity r=%+.3f'
          % ('', spearmanr(dfe[ok], ref['dphys'][ok]).statistic, spearmanr(dfe[ok], ref['dact'][ok]).statistic))
    return T, p, q


def main():
    data = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    dist = tg.pairwise_distance(xy)
    iu = np.triu_indices(resp.shape[0], 1); dd = dist[iu]
    ref = {'dphys': dd, 'dact': pdist(znorm_rows(resp)), 'ok': (dd >= MIND) & (dd <= MAXD)}
    print('FEATURE TOPOGRAPHY v2 | corrected 0.7 | ZETA n=%d | %d images | T = near(<%.0fum) - far(>%.0fum) z-product, '
          '%d-perm position shuffle, BH-FDR per basis' % (resp.shape[0], resp.shape[1], NEAR, FAR, N_PERM))
    print('  reference: D_activity vs D_phys r=%+.3f' % spearmanr(ref['dact'][ref['ok']], dd[ref['ok']]).statistic)

    paths = som.condition_image_paths(data, STIM_DIR)
    images = dnn_som.load_images_rgb(paths)
    model, _ = dnn_som.prep_dnn_model('cpu')
    feats = {nm: dnn_som.extract_dnn_features(model, layer, images, 'cpu').numpy() for nm, layer in LAYERS.items()}
    sca = dnn_som.som_sca(dnn_som.load_som(som.DEFAULT_SOM), feats['relu7']).numpy()      # (n_img, 400 units)
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Un, Sn, Vtn = np.linalg.svd(R, full_matrices=False)

    print('\n-- per-basis topography of feature-preference maps --')
    report_basis('neural-PC', Un[:, :K] * Sn[:K], (Sn[:K] ** 2) / (Sn ** 2).sum(), dist, ref)
    for nm in LAYERS:
        fp, fev = pcs(feats[nm])
        pref = np.column_stack([rowcorr(resp, fp[:, k]) for k in range(fp.shape[1])])
        report_basis(nm + '-PC corr', pref, fev, dist, ref)
        report_basis(nm + '-PC ridge', ridge_weights(resp, fp), fev, dist, ref)
        al = np.array([abs(np.corrcoef(Vtn[0], fp[:, k])[0, 1]) for k in range(fp.shape[1])])
        print('  %-16s      neural PC0 best-aligned with %s-PC %d (|r|=%.2f, fev=%.3f)'
              % ('', nm, int(np.argmax(al)), al.max(), fev[int(np.argmax(al))]))
    pref_som = np.column_stack([rowcorr(resp, sca[:, u]) for u in range(sca.shape[1])])   # (n_roi, 400)
    order = np.argsort(-np.nanmedian(np.abs(pref_som), 0))[:60]
    report_basis('SOM-unit top60', pref_som[:, order], None, dist, ref)
    dfe = pdist(znorm_rows(pref_som)); ok = ref['ok']
    print('  %-16s      [all 400 units] D_SOMpref vs D_phys r=%+.3f | vs D_activity r=%+.3f'
          % ('', spearmanr(dfe[ok], dd[ok]).statistic, spearmanr(dfe[ok], ref['dact'][ok]).statistic))


if __name__ == '__main__':
    main()
