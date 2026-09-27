#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Should the PROBE side apply ImageNet mean/std normalisation? (DH 2026-09-27)

Doshi's TRAINING notebook feeds AlexNet ``Resize -> CenterCrop -> ToTensor -> Normalize(ImageNet mean/std)``; his
PROBE notebooks (and our ``dnn_som``, for parity) omit the Normalize step, so the probe images enter the network on
a different input scale ([0, 1] instead of the standardised ~[-2, 2] the network and the shipped SOM codebook were
built on). Quantify what that does on the 60 PD images with the shipped 20x20 SOM:
  * relu7 activation magnitude / sparsity under each transform;
  * relu7 RDM agreement between the two transforms (Spearman of upper triangles);
  * where the images land on the map (distinct BMUs, mean pairwise BMU distance, within- vs between-category BMU
    distance) and the SCA RDM agreement;
  * cortex~SOM and cortex~relu7 RSA (PD corrected 0.7, ZETA ROIs) under each transform.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/probe_normalization_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import feature_topography_v2 as v2
import response_matrix as rm
import som

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    data = rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(v2.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    cats = data['conditions'].reindex(data['condition_ids'])['cat'].astype(str).to_numpy()
    paths = som.condition_image_paths(data, v2.STIM_DIR)
    x_plain = dnn_som.load_images_rgb(paths)
    x_norm = (x_plain - MEAN) / STD
    model, layer = dnn_som.prep_dnn_model('cpu')
    sm = dnn_som.load_som(som.DEFAULT_SOM)
    rdm_cx = som.cortical_rdm(d['response'])
    out = {}
    for nm, x in (('plain [0,1] (probe side, current)', x_plain), ('ImageNet-normalised (training side)', x_norm)):
        f = dnn_som.extract_dnn_features(model, layer, x, 'cpu').numpy()
        sca = dnn_som.som_sca(sm, f).numpy()
        bmu = dnn_som.som_bmu(sm, f).numpy()
        rdm_f, rdm_s = som.build_rdm(f), som.build_rdm(sca)
        db = pdist(bmu)
        same = (cats[:, None] == cats[None, :])[np.triu_indices(len(cats), 1)]
        rs = som.rsa(rdm_cx, rdm_s, n_perm=1000)
        r7 = som.rsa(rdm_cx, rdm_f, n_perm=1000)
        out[nm] = dict(f=f, rdm_f=rdm_f, rdm_s=rdm_s, bmu=bmu)
        print('%s\n  relu7: mean activation %.3f | fraction of zero units %.3f | mean L2 norm %.1f'
              % (nm, f.mean(), (f == 0).mean(), np.linalg.norm(f, axis=1).mean()))
        print('  map: %d distinct BMUs of %d images | mean pairwise BMU distance %.2f units | within-category %.2f vs between %.2f'
              % (len(np.unique(bmu, axis=0)), len(bmu), db.mean(), db[same].mean(), db[~same].mean()))
        print('  RSA cortex~SOM r=%+.3f (p=%.3f) | cortex~relu7 r=%+.3f (p=%.3f)' % (rs['r'], rs['p'], r7['r'], r7['p']))
    a, b = list(out.values())
    iu = np.triu_indices(len(paths), 1)
    print('\nagreement between the two transforms: relu7 RDM Spearman %.3f | SCA RDM Spearman %.3f | mean BMU displacement %.2f units'
          % (spearmanr(a['rdm_f'][iu], b['rdm_f'][iu]).statistic, spearmanr(a['rdm_s'][iu], b['rdm_s'][iu]).statistic,
             np.linalg.norm(a['bmu'] - b['bmu'], axis=1).mean()))


if __name__ == '__main__':
    main()
