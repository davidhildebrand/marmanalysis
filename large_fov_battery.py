#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#2 -- neuropil-corrected topography battery on LARGE-FOV sessions (Curly first, then both Dali), to test
whether the smooth face-selectivity gradient found on the 730 um PD FOV extends across a much larger patch, or
whether finer structure appears at spatial scales the small FOV cannot resolve (roadmap 12e).

Image-free analyses (topography controls + PC identity / per-PC template detector / drift) so they run for any
image stimulus set. Corrected (neucoeff=0.7). ROIs are gated to `responsive` and the response table is subsampled
to `SUBSAMPLE` up front (large-FOV sessions carry 5-16k ROIs; the full tensor is multi-GB and the pairwise
controls are O(n^2)-O(n^3)). Distances are capped per session at ~0.8x the FOV diagonal so the large FOV actually
resolves the low spatial frequencies. eye_gate_mode='none' for robustness across sessions.
Run:  .venv/bin/python marmanalysis/large_fov_battery.py
"""
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import response_table as rt
import topography as tg

SUBSAMPLE = 2000
MIND = 15.0
TARGETS = [
    ('Curly', '20231103d', 'ImagesFOBmin'),
    ('Dali', '20230810d', 'ImagesSongFOBonly'),
    ('Dali', '20230910d', 'ImagesFOBmany'),
]


def resolve(animal, date, token):
    hits = [h for h in sorted(glob.glob('suite2p_results/%s/%s/*%s*' % (animal, date, token)))
            if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
    return hits[0] if hits else None


def cat_masks(cats):
    c = np.array([str(x).lower() for x in cats])
    face = np.array(['face' in x for x in c])
    blank = np.array(['blank' in x for x in c])
    return face, (~face & ~blank)


def run_session(path):
    corr = rm.build_response_matrix(path, denoise=False, reliability_splits=3, eye_gate_mode='none',
                                    neuropil_subtract=True, neucoeff=0.7, roi_subsample=SUBSAMPLE,
                                    equalize_repeats=True)
    unc = rm.build_response_matrix(path, denoise=False, reliability_splits=3, eye_gate_mode='none',
                                   neuropil_subtract=False, roi_subsample=SUBSAMPLE, equalize_repeats=True)
    fr = corr['framerate']
    resp_mask = rt.classify_responses(corr['ds'], metric='Fzsc', framerate=fr, alpha=0.05)['responsive']
    fov = corr['roi_xy_um'].max(0) - corr['roi_xy_um'].min(0)
    maxd = 0.8 * float(np.hypot(*fov))
    cats = corr['conditions'].reindex(corr['condition_ids'])['cat'].to_numpy()
    print('  n_sub=%d  responsive=%d  |  FOV~%.0fx%.0f um  maxd=%.0f  |  cats=%s'
          % (corr['response'].shape[0], int(resp_mask.sum()), fov[0], fov[1], maxd,
             list(pd.unique([str(x) for x in cats]))))
    # Topography on the FULL subsampled set (like PD's 'all'): dense spatial sampling, and not starved when the
    # responsive fraction is low (subsampling happens BEFORE gating). Responsive count reported for context.
    for lab, dd in (('corrected', corr), ('uncorrected', unc)):
        dist = tg.pairwise_distance(dd['roi_xy_um'])
        sim = tg.pairwise_tuning_similarity(dd['response'])
        s0 = tg.topography_score(sim, dist, max_dist=maxd, min_dist=MIND)
        ps = tg.position_shuffle_null(sim, dist, n_perm=400, max_dist=maxd, min_dist=MIND)
        vn = tg.variogram_matched_null(dd['response'], dd['roi_xy_um'], n_perm=400, max_dist=maxd, min_dist=MIND)
        print('    %-11s topo=%+.3f | pos-shuffle p=%.3f | variogram a=%.2f lam=%.0fum null=%+.3f p=%.3f'
              % (lab, s0, ps['p'], vn['amplitude'], vn['lambda_um'], vn['null_mean'], vn['p']))

    # PC identity + per-PC template detector (corrected, full subsampled set)
    resp, xy = corr['response'], corr['roi_xy_um']
    dist = tg.pairwise_distance(xy)
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    scores, ev = U * s, (s ** 2) / (s ** 2).sum()
    face, non = cat_masks(cats)
    fp = (np.nanmean(resp[:, face], 1) - np.nanmean(resp[:, non], 1)
          if face.any() and non.any() else np.full(resp.shape[0], np.nan))
    sfd = tg.spatial_frequency_detector(resp, xy, n_grid=28, n_pc=3, n_perm=300, min_wavelength_um=30)
    for k in range(3):
        ok = np.isfinite(scores[:, k]) & np.isfinite(fp)
        rfp = (np.corrcoef(scores[ok, k], fp[ok])[0, 1]
               if ok.sum() > 2 and np.std(fp[ok]) > 0 else np.nan)
        a_k, lam_k = tg._scalar_autocorrelation(scores[:, k], dist)
        print('    PC%d ev=%.2f corr(face_pref)=%+.2f a=%.2f lam=%3.0fum | detector wl=%5.0fum p=%.3f  %s'
              % (k, ev[k], rfp, a_k, lam_k, sfd[k]['peak_wavelength_um'], sfd[k]['p'],
                 'periodic/domain' if sfd[k]['p'] < 0.05 else 'gradient'))
    st = rt.calculate_temporal_split_stability(corr['ds'], metric='Fzsc', roi_mask=resp_mask)
    print('    drift: temporal split-half gap=%+.3f -> %s'
          % (st['gap'], 'time-stable' if abs(st['gap']) < 0.05 else 'POSSIBLE DRIFT'))


def main():
    print('#2 LARGE-FOV topography battery (corrected 0.7; responsive gate + subsample %d; eye_gate=none)'
          % SUBSAMPLE)
    for animal, date, token in TARGETS:
        print('\n' + '=' * 78)
        print('%s / %s / %s' % (animal, date, token))
        path = resolve(animal, date, token)
        if path is None:
            print('  (no session dir found)'); sys.stdout.flush(); continue
        print('  ' + os.path.basename(path)[:72])
        try:
            run_session(path)
        except Exception as e:
            traceback.print_exc()
            print('  FAILED:', type(e).__name__, str(e)[:200])
        sys.stdout.flush()


if __name__ == '__main__':
    main()
