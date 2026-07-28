#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOM step 5: unit -> map alignment. Does the DNN-SOM's SPATIAL organisation predict the cortical spatial layout?

Each cortical ROI is assigned the SOM unit whose simulated-cortical-activation (SCA) profile over our stimuli best
matches the ROI's response profile (BMU in stimulus space). Then a Mantel-style test asks whether ROIs NEARBY IN
CORTEX map to units NEARBY IN THE SOM GRID -- with a position-shuffle null AND a within-category-preference null
(so we can separate coarse-face-gradient alignment from anything finer). On the neuropil-CORRECTED (0.7) PD
session, all ROIs + ZETA + the well-matched ZETA subset.

CAVEAT (DH 2026-07-27): this is the Song FOB set (20 face / 20 body / 20 object) -- coarse CATEGORY sampling, so a
positive alignment most likely reflects the smooth face-selectivity gradient (step-4 finding), and the
within-category null is the honest test of whether the SOM predicts anything beyond that.
Run:  .venv/bin/python marmanalysis/som_alignment.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import response_matrix as rm
import som

SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
           'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
STIM_DIR = 'stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly'
ZETA_CSV = 'tmp/responsiveness_pd.csv'


def roi_category(response, cats):
    """Per-ROI preferred category = argmax of the mean response over each category's conditions."""
    c = np.array([str(x) for x in cats])
    uc = pd.unique(c)
    means = np.column_stack([np.nanmean(response[:, c == u], 1) for u in uc])
    return np.array([uc[i] for i in np.nanargmax(means, axis=1)])


def run(data, som_xy, sca, cats, tag):
    resp, xy = data['response'], data['roi_xy_um']
    bmu, match = som.assign_bmu_units(resp, sca)
    assigned = som_xy[bmu]
    grp = roi_category(resp, cats)
    ap = som.map_alignment(xy, assigned, n_perm=1000, groups=None)
    ag = som.map_alignment(xy, assigned, n_perm=1000, groups=grp)
    print('  %-26s n=%4d  match med=%+.2f  |  align r=%+.3f  p(pos-shuffle)=%.3f  p(within-cat)=%.3f  %s'
          % (tag, resp.shape[0], float(np.median(match)), ap['r'], ap['p'], ag['p'],
             'BEYOND category' if ag['p'] < 0.05 else '(category-level only)'))


def main():
    corr = rm.build_response_matrix(SESSION, denoise=False, reliability_splits=5,
                                    neuropil_subtract=True, neucoeff=0.7)
    cats = corr['conditions'].reindex(corr['condition_ids'])['cat'].to_numpy()
    paths = som.condition_image_paths(corr, STIM_DIR)
    model = som.build_model_matrices(paths)
    sca = model['sca']
    som_xy = dnn_som.load_som(som.DEFAULT_SOM).locations.detach().numpy()
    n = corr['response'].shape[0]
    print('SOM step 5 — unit->map alignment (Mantel: cortical dist vs assigned-SOM-unit dist; corrected 0.7)')
    print('  sca %s | som units %d | n_roi %d | categories %s'
          % (str(sca.shape), som_xy.shape[0], n, list(pd.unique([str(x) for x in cats]))))
    run(corr, som_xy, sca, cats, 'CORRECTED / all')
    if os.path.exists(ZETA_CSV):
        z = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if z.size == n:
            d = rm.apply_roi_mask(corr, z)
            run(d, som_xy, sca, cats, 'CORRECTED / ZETA')
            _, match = som.assign_bmu_units(d['response'], sca)
            run(rm.apply_roi_mask(d, match > np.median(match)), som_xy, sca, cats,
                'CORRECTED / ZETA well-matched')


if __name__ == '__main__':
    main()
