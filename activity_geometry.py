#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Activity-space distance between neurons (to compare with physical distance) -- two definitions, both kept.

(1) ``dphys_currier_clandinin``: the EXACT recipe of Currier & Clandinin 2025 Cell, "Interneuron diversity and
    normalization specificity in a visual system" (https://doi.org/10.1016/j.cell.2025.05.007), STAR Methods
    "Principal components analysis and Dphys" (quoted from the full text in our library):
        "we first projected each neuron into the space defined by the first 100 PCs. We then found the length of
         the vector connecting a pair of cells ... in this space. For the clustering analysis ... we normalized
         Dphys by dividing all type-to-type Dphys by the largest observed type-to-type Dphys."
    Their PCA samples are one vector per neuron: the flattened and concatenated centre-40 deg / last-1 s blue and
    UV STRFs plus the dF/F responses to each flicker stimulus (Fig. S4A). The PCA is fit ACROSS neurons (samples =
    neurons, features = the response vector), each neuron is projected onto the first 100 PCs, and D_phys is the
    Euclidean distance between two neurons' projections. No per-neuron normalisation is described (their STRFs are
    in z-units by construction, so between-neuron amplitude differences are comparatively small). Here ``X`` is any
    per-ROI response vector: the roi x condition tuning matrix, or a flattened roi x (condition x time) tensor to
    mirror their time-resolved samples.

(2) ``dact_zscored``: each ROI's response vector is z-scored ACROSS its own entries first, then the Euclidean
    distance is taken (optionally in a PC space). WHY THIS IS THE BETTER DEFAULT FOR OUR DATA: calcium tuning
    vectors carry a per-ROI GAIN (peak dF/F varies more than 10x between ROIs with expression level, depth, neuropil
    fraction and health, and the Fzsc metric only removes the session-wide scale, not the stimulus-driven amplitude),
    so a raw Euclidean distance mostly measures "how bright / how strongly modulated" rather than "what the cell
    prefers": two ROIs with identical preferences but different amplitudes come out far apart, and one strongly
    modulated ROI is far from everybody. Z-scoring removes that gain: ||z_i - z_j||^2 = 2 k (1 - r_ij) (k =
    number of conditions, population std), a monotone function of the tuning correlation, i.e. the distance then
    reflects tuning SHAPE. On PD this is the
    difference between a raw D_phys-vs-cortical-distance correlation near zero / negative and a positive one. Their
    STRF-based samples do not have this problem to the same degree, which is why the two recipes differ here.

WHY (1) REDUCES TO RAW EUCLIDEAN ON OUR TUNING MATRICES: with 60 (or 208) conditions the centred response matrix
has rank <= n_conditions < 100, so the 100-PC projection is lossless and D_phys equals the Euclidean distance on
the centred tuning vectors (``n_pc`` only matters for time-resolved samples with more features than neurons).

Run (demo on PD):  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/activity_geometry.py
"""
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dphys_currier_clandinin(X, n_pc=100, normalize_max=False):
    """Currier & Clandinin 2025 D_phys (see module docstring). ``X``: (n_neurons, n_features) response vectors.
    PCA across neurons -> first ``n_pc`` PCs (their 100; capped at the rank) -> Euclidean distance between neurons.
    ``normalize_max`` divides by the largest distance (their type-level normalisation). Returns (n, n)."""
    X = np.nan_to_num(np.asarray(X, float))
    Xc = X - X.mean(0, keepdims=True)                    # PCA centring across samples (neurons)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(n_pc, S.size))
    P = U[:, :k] * S[:k]                                 # projection onto the first k PCs
    D = squareform(pdist(P))
    if normalize_max:
        D = D / (D.max() + 1e-12)
    return D


def dact_zscored(X, n_pc=None):
    """Row-z-scored activity distance (see module docstring): each ROI's vector is standardised across its own
    entries, then Euclidean distance (optionally after projecting onto the first ``n_pc`` PCs). Returns (n, n)."""
    X = np.nan_to_num(np.asarray(X, float))
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    if n_pc is not None:
        Zc = Z - Z.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Zc, full_matrices=False)
        k = int(min(n_pc, S.size))
        Z = U[:, :k] * S[:k]
    return squareform(pdist(Z))


def compare_to_physical(D_act, xy_um, min_dist=15.0, max_dist=600.0):
    """Spearman correlation between activity distance and physical distance over ROI pairs within
    [min_dist, max_dist] um (positive = nearby cells are more similar). Returns {'rho', 'p', 'n_pairs'}."""
    D_act = np.asarray(D_act, float)
    xy = np.asarray(xy_um, float)
    dist = squareform(pdist(xy))
    iu = np.triu_indices(len(xy), 1)
    a, d = D_act[iu], dist[iu]
    ok = (d >= min_dist) & (d <= max_dist) & np.isfinite(a)
    r = spearmanr(a[ok], d[ok])
    return {'rho': float(r.statistic), 'p': float(r.pvalue), 'n_pairs': int(ok.sum())}


def _demo():
    import pandas as pd
    import feature_topography_v2 as v2
    import response_matrix as rm
    data = rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(v2.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    X, xy = d['response'], d['roi_xy_um']
    gain = np.nanmax(X, 1) - np.nanmin(X, 1)
    iu = np.triu_indices(X.shape[0], 1)
    print('PD corrected, %d ZETA ROIs x %d conditions (rank <= %d < 100 -> the 100-PC projection is lossless)' % (X.shape + (X.shape[1],)))
    D1 = dphys_currier_clandinin(X)
    D1raw = squareform(pdist(np.nan_to_num(X)))
    print('  exact D_phys == raw Euclidean on centred vectors: max |diff| = %.2e' % np.abs(D1 - D1raw).max())
    D2 = dact_zscored(X)
    for nm, D in (('Currier & Clandinin D_phys (100 PCs)', D1), ('row-z-scored distance', D2)):
        c = compare_to_physical(D, xy)
        r_gain = spearmanr(D[iu], np.abs(gain[iu[0]] - gain[iu[1]])).statistic
        print('  %-38s rho(D_act, cortical distance) = %+.3f (p=%.2g, %d pairs) | rho(D_act, |gain_i - gain_j|) = %+.3f'
              % (nm, c['rho'], c['p'], c['n_pairs'], r_gain))


if __name__ == '__main__':
    _demo()
