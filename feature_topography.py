#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature-space topography (DH question, 2026-09-20): are IMAGE FEATURES represented in a spatially organised
-- smooth or periodic -- manner? The earlier template detector looked only at the top-3 NEURAL-tuning PCs, which
on this stimulus set are dominated by the coarse face/object axis. This extends it three ways, on the corrected
(neucoeff 0.7) PD session, ZETA subset:

  (A) MORE NEURAL PCs -- the per-PC-calibrated template detector on the top 20 neural-tuning PCs, BH-FDR across
      PCs (higher PCs = finer tuning axes, untested before).
  (B) PER-IMAGE maps -- each image's own response map tested for periodicity, FDR across images (low prior; the
      literal per-image scan, kept as a baseline).
  (C) PER-FEATURE maps -- the real question. Image features = top-K PCs of the AlexNet relu7 features of the
      stimulus images. Each ROI's PREFERENCE for feature k = correlation of its image-tuning with the images'
      feature-k coordinates -> a per-ROI feature-preference scalar map. Each map is tested for (i) SMOOTH
      topography (similarity-vs-distance vs position-shuffle; plus its autocorrelation a/lambda) and (ii)
      PERIODICITY (calibrated detector), FDR across features. Also: which feature-PC the dominant neural PC
      aligns with, and the Cell-2025-style geometry tests -- ROI-ROI distance in NEURAL-PC space, and in
      FEATURE-PREFERENCE space, each against physical distance.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/feature_topography.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import false_discovery_control, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import som
import topography as tg

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
STIM_DIR = 'stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly'
ZETA_CSV = 'tmp/responsiveness_pd.csv'
MAXD, MIND = 600.0, 15.0
K_FEAT, K_NPC = 20, 20


def rowcorr(A, v):
    """Pearson correlation of each row of A (n x m) with vector v (m,), NaN-safe."""
    A, v = np.asarray(A, float), np.asarray(v, float)
    Az, vz = A - np.nanmean(A, 1, keepdims=True), v - v.mean()
    with np.errstate(all='ignore'):
        return np.nansum(Az * vz, 1) / np.sqrt(np.nansum(Az ** 2, 1) * (vz ** 2).sum())


def fdr(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = false_discovery_control(p[ok], method='bh')
    return out


def geometry_vs_physical(coords, xy, label, normalize=True):
    """Cell-2025-style: ROI-ROI Euclidean distance in an embedding space vs physical distance (Spearman over
    pairs within [MIND, MAXD]). Positive => ROIs close in the embedding are close in cortex. ``normalize``
    z-scores each ROI's vector first (mean 0, unit norm) so Euclidean distance ~ CORRELATION distance -- i.e. it
    compares tuning SHAPE. Without it, Euclidean distance is dominated by per-cell GAIN (a strongly-driven cell is
    'far' from everything regardless of shape), which is why the unnormalised version came out ~0/negative while
    the correlation-based topography score is positive."""
    C = np.nan_to_num(np.asarray(coords, float))
    if normalize:
        C = C - C.mean(1, keepdims=True)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    de, dp = pdist(C), pdist(np.asarray(xy, float))
    ok = np.isfinite(de) & (dp >= MIND) & (dp <= MAXD)
    r = spearmanr(de[ok], dp[ok]).statistic
    print('  D_%-18s vs D_phys : Spearman r=%+.3f  (n_pairs=%d)' % (label, r, int(ok.sum())))


def main():
    data = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5,
                                    neuropil_subtract=True, neucoeff=0.7)
    n = data['response'].shape[0]
    zeta = (pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05) if os.path.exists(ZETA_CSV) else np.ones(n, bool)
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    dist = tg.pairwise_distance(xy)
    cats = np.array([str(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
    n_img = resp.shape[1]
    print('FEATURE-SPACE TOPOGRAPHY | corrected 0.7 | ZETA n=%d ROIs | %d images' % (resp.shape[0], n_img))

    # ---- (A) more neural PCs ----
    sfd = tg.spatial_frequency_detector(resp, xy, n_grid=24, n_pc=K_NPC, n_perm=400, min_wavelength_um=30)
    pA = np.array([sfd[k]['p'] for k in range(K_NPC)]); qA = fdr(pA)
    print('\n(A) NEURAL-TUNING PCs, top %d (per-PC-calibrated template detector; BH-FDR across PCs)' % K_NPC)
    print('    PC   ev     a   lam(um)  peak_wl(um)   p      q')
    for k in range(K_NPC):
        s = sfd[k]
        print('    %2d  %.3f  %.2f  %5.0f    %7.0f    %.3f  %.3f%s'
              % (k, s['explained_var'], s['amplitude'], s['lambda_um'], s['peak_wavelength_um'], s['p'], qA[k],
                 '  *' if qA[k] < 0.05 else ''))
    print('    -> periodic neural PCs at q<0.05: %d / %d' % (int(np.nansum(qA < 0.05)), K_NPC))

    # ---- (B) per-image maps ----
    per_img = tg.scalar_map_periodicity(resp, xy, n_grid=24, n_perm=300, min_wavelength_um=30)
    pB = np.array([per_img[k]['p'] for k in range(n_img)]); qB = fdr(pB)
    kb = int(np.nanargmin(pB))
    print('\n(B) PER-IMAGE response maps: %d images | periodic at q<0.05: %d | best: image %d (%s) p=%.3f q=%.3f '
          'peak_wl=%.0fum' % (n_img, int(np.nansum(qB < 0.05)), kb, cats[kb], pB[kb], qB[kb],
                              per_img[kb]['peak_wavelength_um']))

    # ---- (C) per-feature maps ----
    model = som.build_model_matrices(som.condition_image_paths(data, STIM_DIR))
    F = model['relu7'] - model['relu7'].mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    K = int(min(K_FEAT, S.size))
    feat, fev = U[:, :K] * S[:K], (S[:K] ** 2) / (S ** 2).sum()      # (n_img, K) image coords on feature-PCs
    pref = np.column_stack([rowcorr(resp, feat[:, k]) for k in range(K)])   # (n_roi, K) feature preference
    per_feat = tg.scalar_map_periodicity(pref, xy, n_grid=24, n_perm=400, min_wavelength_um=30)
    topo = []
    for k in range(K):
        sim = tg.nuisance_similarity(np.nan_to_num(pref[:, k]))
        ps = tg.position_shuffle_null(sim, dist, n_perm=400, max_dist=MAXD, min_dist=MIND)
        topo.append((ps['score'], ps['p']))
    p_topo = np.array([t[1] for t in topo]); q_topo = fdr(p_topo)
    p_per = np.array([per_feat[k]['p'] for k in range(K)]); q_per = fdr(p_per)
    print('\n(C) IMAGE-FEATURE (relu7-PC) preference maps, top %d features (BH-FDR across features)' % K)
    print('    feat  fev   med|pref|   a   lam(um) | SMOOTH: topo   p_topo  q_topo | PERIODIC: peak_wl(um)  p     q')
    for k in range(K):
        s = per_feat[k]
        print('    %2d   %.3f   %.2f     %.2f  %5.0f  |         %+.3f  %.3f  %.3f%s |          %7.0f   %.3f %.3f%s'
              % (k, fev[k], float(np.nanmedian(np.abs(pref[:, k]))), s['amplitude'], s['lambda_um'],
                 topo[k][0], p_topo[k], q_topo[k], ' *' if q_topo[k] < 0.05 else '  ',
                 s['peak_wavelength_um'], p_per[k], q_per[k], ' *' if q_per[k] < 0.05 else ''))
    print('    -> smoothly-mapped features (q_topo<0.05): %d/%d | periodic features (q<0.05): %d/%d'
          % (int(np.nansum(q_topo < 0.05)), K, int(np.nansum(q_per < 0.05)), K))

    # which feature does the dominant neural PC align with?
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Un, Sn, Vtn = np.linalg.svd(R, full_matrices=False)
    align = np.array([abs(np.corrcoef(Vtn[0], feat[:, k])[0, 1]) for k in range(K)])
    print('\n    neural PC0 (face axis) aligns best with feature-PC %d: |corr of condition loadings|=%.2f (fev=%.3f); '
          'next: PC %d (%.2f)' % (int(np.argmax(align)), align.max(), fev[int(np.argmax(align))],
                                 int(np.argsort(align)[-2]), np.sort(align)[-2]))

    # ---- Cell-2025-style geometry vs physical distance ----
    print('\n(D) EMBEDDING-DISTANCE vs PHYSICAL-DISTANCE (Spearman over ROI pairs, %.0f-%.0f um)' % (MIND, MAXD))
    geometry_vs_physical(Un[:, :K_NPC] * Sn[:K_NPC], xy, 'neuralPC%d' % K_NPC)
    geometry_vs_physical(Un[:, :3] * Sn[:3], xy, 'neuralPC3')
    geometry_vs_physical(pref, xy, 'featurePref%d' % K)
    geometry_vs_physical(resp, xy, 'fullTuning(%d)' % n_img)


if __name__ == '__main__':
    main()
