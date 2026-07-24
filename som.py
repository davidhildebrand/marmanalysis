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


def _rc(v):
    """Mean-centred ranks of a vector (Spearman = Pearson on these)."""
    r = rankdata(v).astype(float)
    return r - r.mean()


def _spearman(a, b):
    """Spearman correlation between two vectors."""
    ra, rb = _rc(a), _rc(b)
    da, db = np.sqrt((ra ** 2).sum()), np.sqrt((rb ** 2).sum())
    return float((ra * rb).sum() / (da * db)) if da > 0 and db > 0 else np.nan


def _partial_spearman(a, b, c):
    """Partial Spearman correlation of a and b, controlling for c (partial Pearson on ranks)."""
    ra, rb, rc = _rc(a), _rc(b), _rc(c)

    def cor(x, y):
        dx, dy = np.sqrt((x ** 2).sum()), np.sqrt((y ** 2).sum())
        return (x * y).sum() / (dx * dy) if dx > 0 and dy > 0 else np.nan

    rab, rac, rbc = cor(ra, rb), cor(ra, rc), cor(rb, rc)
    denom = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    return float((rab - rac * rbc) / denom) if denom > 0 else np.nan


def rsa(rdm_a, rdm_b, control=None, n_perm=5000, seed=0):
    """Second-order RSA: Spearman correlation between the upper triangles of two RDMs (``control`` given -> the
    PARTIAL Spearman controlling for a third RDM, e.g. the category model). One-sided permutation p-value: the
    items of ``rdm_b`` are relabelled and the (partial) correlation recomputed, with ``rdm_a`` and ``control``
    held fixed. H1 = positive association. Returns {'r', 'p', 'n_perm', 'null_mean'}."""
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = rdm_a[iu]
    c = None if control is None else control[iu]

    def stat(bt):
        return _spearman(a, bt) if c is None else _partial_spearman(a, bt, c)

    r = stat(rdm_b[iu])
    rng = np.random.default_rng(seed)
    null = np.array([stat(rdm_b[np.ix_(p, p)][iu]) for p in (rng.permutation(n) for _ in range(n_perm))])
    p = (1 + int(np.sum(null >= r))) / (1 + n_perm)
    return {'r': r, 'p': p, 'n_perm': n_perm, 'null_mean': float(np.nanmean(null))}


def category_model_rdm(categories):
    """Binary category-model RDM: 0 if two conditions share a category, 1 otherwise. Partialling this out of an
    RSA removes the coarse between-category block structure, isolating any FINER (within/cross-category) match."""
    c = np.asarray([str(x) for x in categories])
    return (c[:, None] != c[None, :]).astype(float)


def cortical_rdm(response, metric='correlation'):
    """Condition x condition RDM from a (n_roi x n_condition) response matrix (dissimilarity between the
    across-ROI response patterns of each pair of conditions)."""
    return build_rdm(np.asarray(response, float).T, metric=metric)


def rsa_difference_bootstrap(rdm_target, rdm_a, rdm_b, control=None, n_boot=5000, seed=0):
    """Does model A match ``rdm_target`` BETTER than model B? Stimulus bootstrap on the RSA difference
    r(target, A) - r(target, B) (PARTIAL, controlling for ``control``, if given). Resamples conditions with
    replacement, excluding duplicate-condition pairs (their dissimilarity is a spurious 0). Returns the mean
    difference, a 95% percentile CI, and a one-sided p = fraction of bootstraps with difference <= 0
    (i.e. A NOT better than B)."""
    n = rdm_target.shape[0]
    iu = np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        I, J = idx[iu[0]], idx[iu[1]]
        m = I != J
        t, a, b = rdm_target[I, J][m], rdm_a[I, J][m], rdm_b[I, J][m]
        if control is None:
            diffs[k] = _spearman(t, a) - _spearman(t, b)
        else:
            cc = control[I, J][m]
            diffs[k] = _partial_spearman(t, a, cc) - _partial_spearman(t, b, cc)
    return {'diff_mean': float(np.nanmean(diffs)), 'ci': [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])],
            'p_le0': float(np.nanmean(diffs <= 0)), 'n_boot': int(n_boot)}


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
