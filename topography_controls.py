#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topography artifact controls on the PD session (SOM plan step 4 / roadmap item 12), run on NEUROPIL-CORRECTED
traces (neucoeff=0.7) with the uncorrected traces alongside for reference. Bundles all four controls:

  * chunk 1 -- min_dist sweep + ROI-shared-pixel exclusion: does the score PERSIST as the nearest pairs are
    dropped (distributed structure) or COLLAPSE (touching-ROI / segmentation short-range artifact)?
  * chunk 2 -- variogram_matched_null: autocorrelation-PRESERVING null (stronger than position-shuffle). p~0.5
    => topography no more than matched per-condition smoothness; p<0.05 => coordinated tuning beyond smoothness.
  * chunk 3 -- spatial_frequency_detector: is there a CHARACTERISTIC SCALE (patch/domain periodicity) beyond a
    matched-autocorrelation null, which the monotonic decay score cannot see?

A self-check on synthetic smooth vs random maps confirms the two new nulls behave before the real numbers.
Run:  .venv/bin/python marmanalysis/topography_controls.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import topography as tg

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
ZETA_CSV = 'tmp/responsiveness_pd.csv'
MAXD = 600.0
MIND = 15.0


def selfcheck():
    """Calibration: RANDOM must be non-significant on BOTH nulls (the previous fixed-nugget version wrongly gave
    the template detector p=0.005 on noise); a PERIODIC map should trip the template detector; a GRADED/monotonic
    map is reproduced by the amplitude-matched variogram null (p~0.5) and has no periodic peak."""
    rng = np.random.default_rng(0)
    n, nc = 220, 30
    xy = rng.uniform(0, 700, (n, 2))
    rand = rng.standard_normal((n, nc))
    basis = np.column_stack([xy[:, 0], xy[:, 1], xy[:, 0] * xy[:, 1] / 700.0])
    basis = (basis - basis.mean(0)) / basis.std(0)
    graded = basis @ rng.standard_normal((3, nc)) + 0.5 * rng.standard_normal((n, nc))     # smooth, monotonic
    periodic = (np.column_stack([np.sin(xy[:, 0] / 60.0 + j) + np.cos(xy[:, 1] / 60.0 - j) for j in range(nc)])
                + 0.4 * rng.standard_normal((n, nc)))
    print('--- self-check (RANDOM must be non-sig on both; PERIODIC should trip the template detector) ---')
    for lab, r in (('random', rand), ('graded', graded), ('periodic', periodic)):
        v = tg.variogram_matched_null(r, xy, n_perm=300)
        s = tg.spatial_frequency_detector(r, xy, n_grid=20, n_pc=1, n_perm=200)
        print('  %-8s varnull: score=%+.3f null_mean=%+.3f a=%.2f p=%.3f | sfd PC0: wl=%4.0fum p=%.3f'
              % (lab, v['score'], v['null_mean'], v['amplitude'], v['p'],
                 s[0]['peak_wavelength_um'], s[0]['p']))


def controls(resp, xy, rois, cats, tag):
    dist = tg.pairwise_distance(xy)
    sim = tg.pairwise_tuning_similarity(resp)
    ov = tg.roi_shared_pixel_pairs(rois)
    print('\n=== %s (n=%d ROIs) ===' % (tag, resp.shape[0]))

    print(' [chunk1] min_dist sweep (FULL tuning; score / position-shuffle p):')
    for md in (0, 15, 30, 50):
        ps = tg.position_shuffle_null(sim, dist, n_perm=500, max_dist=MAXD, min_dist=md)
        print('    d>=%2dum   s=%+.3f  p=%.3f' % (md, ps['score'], ps['p']))
    s_ov = tg.topography_score(sim, dist, max_dist=MAXD, min_dist=MIND, exclude=ov)
    print('    ROI-shared-pixel pairs excluded (d>=15): s=%+.3f  (%d overlapping pairs dropped)'
          % (s_ov, int(ov.sum() // 2)))

    vf = tg.variogram_matched_null(resp, xy, n_perm=500, max_dist=MAXD, min_dist=MIND)
    vr = tg.variogram_matched_null(tg.residualize_category(resp, cats), xy, n_perm=500, max_dist=MAXD, min_dist=MIND)
    print(' [chunk2] variogram-matched null (d>=15, lam=%.0fum, a=%.2f):' % (vf['lambda_um'], vf['amplitude']))
    print('    FULL      score=%+.3f  null_mean=%+.3f  p=%.3f' % (vf['score'], vf['null_mean'], vf['p']))
    print('    RESIDUAL  score=%+.3f  null_mean=%+.3f  p=%.3f' % (vr['score'], vr['null_mean'], vr['p']))

    sfd = tg.spatial_frequency_detector(resp, xy, n_grid=24, n_pc=3, n_perm=400, min_wavelength_um=30)
    print(' [chunk3] spatial-frequency / template detector (peak nonzero-freq power vs matched null):')
    for k in range(3):
        if k in sfd:
            print('    PC%d (ev=%.2f): peak wavelength=%5.0fum  p=%.3f'
                  % (k, sfd[k]['explained_var'], sfd[k]['peak_wavelength_um'], sfd[k]['p']))


def main():
    selfcheck()
    corr = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5,
                                    neuropil_subtract=True, neucoeff=0.7)
    unc = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5, neuropil_subtract=False)
    cats = corr['conditions'].reindex(corr['condition_ids'])['cat'].to_numpy()
    print('\nsession n_roi=%d | correction(corr)=%s neucoeff=%.2f'
          % (corr['response'].shape[0], corr['neuropil_subtract'], corr['neucoeff']))

    zeta = None
    if os.path.exists(ZETA_CSV):
        z = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if z.size == corr['response'].shape[0]:
            zeta = z

    # CORRECTED is the primary (all ROIs + fixed ZETA); uncorrected ZETA for reference.
    controls(corr['response'], corr['roi_xy_um'], corr['ctx']['s2p']['ROIs'], cats, 'CORRECTED 0.7 / all')
    if zeta is not None:
        d = rm.apply_roi_mask(corr, zeta)
        controls(d['response'], d['roi_xy_um'], corr['ctx']['s2p']['ROIs'][zeta], cats, 'CORRECTED 0.7 / ZETA')
        du = rm.apply_roi_mask(unc, zeta)
        controls(du['response'], du['roi_xy_um'], unc['ctx']['s2p']['ROIs'][zeta], cats, 'uncorrected / ZETA (ref)')


if __name__ == '__main__':
    main()
