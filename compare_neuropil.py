#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-by-side: does the 0.7 neuropil correction change the headline SOM-track numbers?

Recomputes, WITH neuropil correction (``Fc = F - 0.7*(Fneu - median(Fneu))``) and WITHOUT, on the PD
session (Cadbury/20221016d):
  1. RESPONSIVENESS / SELECTIVITY gate counts (ANOVA gate) -- recomputed per variant, so this shows how the
     correction shifts gate MEMBERSHIP.
  2. RSA cortex~SOM(SCA), cortex~relu7, and cortex~SOM controlling for the category model -- on FIXED ROI
     subsets (all ROIs, and the fixed ZETA set), so the only thing that changes between columns is the
     correction, not which cells are in the subset (isolates the correction's effect on the geometry).
  3. TOPOGRAPHY score (Spearman[similarity, -distance]) on FULL and CATEGORY-RESIDUAL tuning, with a
     position-shuffle null and the min_dist ROI-overlap control -- also on fixed subsets.
Plus the SOM-vs-relu7 difference test (does SOM beat relu7 after category?) under both variants.

neucoeff=0.7 is the working value (roadmap #13; FISSA estimate parked pending the raw-movie transfer).
Run:  .venv/bin/python marmanalysis/compare_neuropil.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import response_table as rt
import som
import topography as tg

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
STIM_DIR = 'stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly'
ZETA_CSV = 'tmp/responsiveness_pd.csv'
MAXD = 600.0          # cap pairwise distance near the FOV extent (sparse far tail)
MIND = 15.0           # drop the shortest pairs: ROI-overlap / short-range-artifact control
N_PERM_RSA = 2000
N_PERM_TOPO = 1000

VARIANTS = [('without', dict(neuropil_subtract=False)),
            ('with 0.7', dict(neuropil_subtract=True, neucoeff=0.7))]


def main():
    # Both cortical response matrices (default gate=landing_window, metric Fzsc). Correction does not change
    # cellinds (based on raw-F std), so the ROI set is identical + same order across variants and the CSV.
    data = {name: rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5, **kw)
            for name, kw in VARIANTS}
    n_roi = data['without']['response'].shape[0]
    print('session: %s' % os.path.basename(SESSION)[:60])
    print('n_roi=%d | neucoeff(with)=%.2f | correction applied(with)=%s'
          % (n_roi, data['with 0.7']['neucoeff'], data['with 0.7']['neuropil_subtract']))

    cats = data['without']['conditions'].reindex(data['without']['condition_ids'])['cat'].to_numpy()

    # model side is independent of the cortical correction -> build once
    paths = som.condition_image_paths(data['without'], STIM_DIR)
    missing = [p for p in paths if not os.path.exists(p)]
    print('conditions=%d | images present=%d/%d' % (len(paths), len(paths) - len(missing), len(paths)))
    model = som.build_model_matrices(paths)
    rdm_sca, rdm_relu7 = som.build_rdm(model['sca']), som.build_rdm(model['relu7'])
    rdm_cat = som.category_model_rdm(cats)

    # fixed subsets: all ROIs, and the fixed ZETA set (from uncorrected traces; same cells both columns)
    subsets = [('all', np.ones(n_roi, bool))]
    if os.path.exists(ZETA_CSV):
        zeta = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if zeta.size == n_roi:
            subsets.append(('ZETA(fix)', zeta))
        else:
            print('  (ZETA csv has %d rows != %d ROIs; skipping fixed-ZETA subset)' % (zeta.size, n_roi))

    # ---- 1) gates recomputed per variant ----
    print('\n=== RESPONSIVENESS / SELECTIVITY (recomputed per variant; ANOVA gate, Fzsc) ===')
    print('%-9s %6s %11s %13s %13s' % ('variant', 'n_roi', 'responsive', 'selective.05', 'selective.01'))
    for name, _ in VARIANTS:
        d = data[name]
        c05 = rt.classify_responses(d['ds'], metric='Fzsc', framerate=d['framerate'], alpha=0.05)
        c01 = rt.classify_responses(d['ds'], metric='Fzsc', framerate=d['framerate'], alpha=0.01)
        print('%-9s %6d %11d %13d %13d' % (name, d['response'].shape[0], int(c05['responsive'].sum()),
                                           int(c05['selective'].sum()), int(c01['selective'].sum())))

    # ---- 2) RSA on fixed subsets ----
    print('\n=== RSA (correlation RDM, %d perm) — fixed subset isolates the correction ===' % N_PERM_RSA)
    print('%-10s %-9s %6s  %-19s %-19s %-19s' % ('subset', 'variant', 'n_roi',
                                                 'cortex~SOM', 'cortex~relu7', 'cortex~SOM|cat'))
    for sname, smask in subsets:
        for name, _ in VARIANTS:
            d = rm.apply_roi_mask(data[name], smask)
            rdm_cx = som.cortical_rdm(d['response'])
            rs = som.rsa(rdm_cx, rdm_sca, n_perm=N_PERM_RSA)
            r7 = som.rsa(rdm_cx, rdm_relu7, n_perm=N_PERM_RSA)
            rc = som.rsa(rdm_cx, rdm_sca, control=rdm_cat, n_perm=N_PERM_RSA)
            print('%-10s %-9s %6d  r=%+.3f p=%.3f  r=%+.3f p=%.3f  r=%+.3f p=%.3f'
                  % (sname if name == 'without' else '', name, d['n_roi'],
                     rs['r'], rs['p'], r7['r'], r7['p'], rc['r'], rc['p']))

    # ---- 3) topography on fixed subsets ----
    print('\n=== TOPOGRAPHY score=Spearman[sim,-dist] (%d-perm pos-shuffle null; min_dist=%.0f max_dist=%.0f) ==='
          % (N_PERM_TOPO, MIND, MAXD))
    print('%-10s %-9s %6s  %-22s %-22s' % ('subset', 'variant', 'n_roi', 'FULL', 'CATEGORY-RESIDUAL'))
    for sname, smask in subsets:
        for name, _ in VARIANTS:
            d = rm.apply_roi_mask(data[name], smask)
            dist = tg.pairwise_distance(d['roi_xy_um'])
            sim_f = tg.pairwise_tuning_similarity(d['response'])
            sim_r = tg.pairwise_tuning_similarity(tg.residualize_category(d['response'], cats))
            nf = tg.position_shuffle_null(sim_f, dist, n_perm=N_PERM_TOPO, max_dist=MAXD, min_dist=MIND)
            nr = tg.position_shuffle_null(sim_r, dist, n_perm=N_PERM_TOPO, max_dist=MAXD, min_dist=MIND)
            print('%-10s %-9s %6d  s=%+.3f p=%.3f%s  s=%+.3f p=%.3f%s'
                  % (sname if name == 'without' else '', name, d['n_roi'],
                     nf['score'], nf['p'], '*' if nf['p'] < 0.05 else ' ',
                     nr['score'], nr['p'], '*' if nr['p'] < 0.05 else ' '))

    # ---- SOM-vs-relu7 difference (does the SOM's topographic geometry beat raw relu7?) ----
    print('\n=== SOM vs relu7 difference test (stimulus bootstrap; controlling category) ===')
    sname, smask = subsets[-1]
    for name, _ in VARIANTS:
        d = rm.apply_roi_mask(data[name], smask)
        rdm_cx = som.cortical_rdm(d['response'])
        diff = som.rsa_difference_bootstrap(rdm_cx, rdm_sca, rdm_relu7, control=rdm_cat, n_boot=N_PERM_RSA)
        print('  %-9s [%s] r(SOM)-r(relu7)|cat = %+.3f  CI[%+.3f,%+.3f]  p(SOM not better)=%.3f'
              % (name, sname, diff['diff_mean'], diff['ci'][0], diff['ci'][1], diff['p_le0']))


if __name__ == '__main__':
    main()
