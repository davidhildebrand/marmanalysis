#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOM-vs-cortex representational comparison (SOM plan, main-track step 3): RSA between the cortical response
geometry, the DNN-SOM's simulated-cortical-activation geometry, and raw relu7 (the control).

Pipeline: our presented images -> ``dnn_som`` relu7 features -> the shipped object-rec SOM's simulated cortical
activations (SCA). Each representation (cortex, SOM, relu7) is reduced to a condition x condition
representational dissimilarity matrix (RDM); RSA is the Spearman correlation of two RDMs' upper triangles, with
a condition-label permutation null. The question this answers: does the SOM's representational geometry over the
stimuli match the cortex's -- and does it match BETTER than the raw relu7 features it was built from (i.e. does
the topographic organisation add explanatory power)? Topography proper (nearby-units <-> nearby-ROIs) is the
NEXT step; this is the representational-geometry layer.
"""
import os

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

import dnn_som

DEFAULT_SOM = ('Doshi_and_Konkle_2023_SciAdv/Code/models_from_paper/'
               'som_weights/objectrec_imagenet_trained_som.pth')


def condition_image_paths(data, stimulus_dir):
    """Ordered image-file paths aligned to ``data['condition_ids']`` via the condition->imagename map (from
    ``response_matrix.build_response_matrix``). So model rows line up with cortical response columns."""
    conds = data['conditions'].reindex(data['condition_ids'])
    return [os.path.join(stimulus_dir, str(name)) for name in conds['imagename'].values]


def build_model_matrices(image_paths, som_path=DEFAULT_SOM, device='cpu'):
    """For a list of image files (in order): relu7 features (n x 4096) and the SOM SCA (n x n_units), plus the
    per-image BMU. Returns {'relu7', 'sca', 'bmu', 'n_units'}."""
    images = dnn_som.load_images_rgb(image_paths)
    model, layer = dnn_som.prep_dnn_model(device)
    relu7 = dnn_som.extract_dnn_features(model, layer, images, device).numpy()
    som = dnn_som.load_som(som_path, device)
    return {'relu7': relu7,
            'sca': dnn_som.som_sca(som, relu7).numpy(),
            'bmu': dnn_som.som_bmu(som, relu7).numpy(),
            'n_units': int(som.weight.shape[1])}


def build_rdm(matrix, metric='correlation'):
    """Representational dissimilarity matrix over the ROWS of ``matrix`` (n_items x n_features). Returns the
    (n x n) square RDM. Default metric = correlation distance (1 - Pearson), the RSA standard."""
    return squareform(pdist(np.asarray(matrix, float), metric=metric))


def rsa(rdm_a, rdm_b, n_perm=5000, seed=0):
    """Second-order RSA: Spearman correlation between the upper triangles of two RDMs, with a condition-label
    permutation p-value (relabel rdm_b's items and recompute). One-sided (H1: positive association), since a
    representational MATCH is a positive RDM correlation. Returns {'r', 'p', 'n_perm', 'null_mean'}."""
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    ra = rankdata(rdm_a[iu]); ra = ra - ra.mean()
    da = np.sqrt((ra ** 2).sum())

    def spearman(bt):
        rb = rankdata(bt); rb = rb - rb.mean()
        db = np.sqrt((rb ** 2).sum())
        return float((ra * rb).sum() / (da * db)) if da > 0 and db > 0 else np.nan

    r = spearman(rdm_b[iu])
    rng = np.random.default_rng(seed)
    null = np.array([spearman(rdm_b[np.ix_(p, p)][iu]) for p in (rng.permutation(n) for _ in range(n_perm))])
    p = (1 + int(np.sum(null >= r))) / (1 + n_perm)
    return {'r': r, 'p': p, 'n_perm': n_perm, 'null_mean': float(np.nanmean(null))}


def cortical_rdm(response, metric='correlation'):
    """Condition x condition RDM from a (n_roi x n_condition) response matrix (dissimilarity between the
    across-ROI response patterns of each pair of conditions)."""
    return build_rdm(np.asarray(response, float).T, metric=metric)


def plot_rdms(rdms, labels, rsa_to=None, outdir='output', tag=None):
    """Side-by-side RDM heatmaps (conditions x conditions). ``rsa_to`` optionally annotates each panel's RSA r
    against a named reference (e.g. the cortex RDM). Returns the saved figure path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    n = len(rdms)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False)
    vmax = max(np.nanpercentile(r, 99) for r in rdms)
    for ax, rdm, lab in zip(axes[0], rdms, labels):
        im = ax.imshow(rdm, cmap='viridis', vmin=0, vmax=vmax, interpolation='none')
        sub = '' if not rsa_to or lab not in rsa_to else '\nRSA r=%.3f (p=%.3g)' % (rsa_to[lab]['r'], rsa_to[lab]['p'])
        ax.set_title('%s%s' % (lab, sub), fontsize=10)
        ax.set_xlabel('condition'); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label='1 - r')
    fig.suptitle('representational dissimilarity — %s' % (tag or 'session'))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'som_rdm_%s_%s.png'
                     % (tag or 'session', datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p
