#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What are the MID-LEVEL features aligned with the face axis? (roadmap item 12, secondary result ii: the neural
face axis = neural PC0 aligns best with pool5-PC0, |r|=0.86, better than relu6 / relu7.) A partial description
in four parts, on the PD session (Cadbury 20221016d, 60 FOB images, ZETA-responsive ROIs, corrected 0.7):

  (1) ALIGNMENT: corr between the neural-PC0 condition loading (60-vector) and each layer's top-20 PC scores;
  (2) MONTAGE: the 60 images ranked by pool5-PC0 score and by the neural-PC0 loading (category-coloured frames)
      -- where do non-face images with face-like mid-level statistics land?
  (3) IMAGE STATISTICS: correlate pool5-PC0 and neural-PC0 with simple statistics (mean luminance, RMS contrast,
      spectral slope, high-SF energy fraction, mirror symmetry, colour saturation, edge density, foreground
      fraction) OVERALL and WITHIN category (both residualised on face / body / object dummies) -- which
      statistics carry the axis beyond the face-vs-rest split;
  (4) SALIENCY: input-gradient of the pool5-PC0 projection for the top-4 and bottom-4 images -- which image
      regions drive the axis (face parts vs texture / outline).
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/midlevel_features.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import feature_topography_v2 as v2
import response_matrix as rm
import som

OUTDIR = 'output'
CAT_COLORS = {'face': 'tab:red', 'body': 'tab:green', 'object': 'tab:blue'}


def cat_key(c):
    c = str(c).lower()
    for k in CAT_COLORS:
        if k in c:
            return k
    return 'object'


def pca_scores(F, k=20):
    F = np.asarray(F, float)
    Fc = F - F.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    return U[:, :k] * S[:k], Vt[:k], ev[:k]


def image_stats(x):
    """x: (3, H, W) tensor in [0,1] (Resize->CenterCrop->ToTensor, no normalisation)."""
    rgb = x.numpy().transpose(1, 2, 0)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum = g.mean()
    rms = g.std() / (g.mean() + 1e-9)
    P = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    Y, X = np.mgrid[:h, :w]
    R = np.hypot(Y - h / 2, X - w / 2)
    rb = np.round(R).astype(int)
    prof = np.bincount(rb.ravel(), P.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
    f = np.arange(len(prof))
    sel = (f >= 4) & (f <= 100)
    slope = np.polyfit(np.log(f[sel]), np.log(prof[sel] + 1e-12), 1)[0]
    hi = P[(R > 28)].sum() / (P[R > 0.5].sum() + 1e-12)
    sym = np.corrcoef(g.ravel(), g[:, ::-1].ravel())[0, 1]
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.mean((mx - mn) / (mx + 1e-9))
    edge = np.mean(np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1)))
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    fg = np.mean(np.abs(g - np.median(border)) > 0.06)
    return dict(luminance=lum, rms_contrast=rms, spectral_slope=slope, highSF_fraction=hi, mirror_symmetry=sym,
                saturation=sat, edge_density=edge, foreground_fraction=fg)


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def saliency(model, w, x):
    """|d(w . pool5(x)) / dx| summed over channels, smoothed. ``w``: (9216,) unit direction in pool5 space."""
    x = x.clone().unsqueeze(0).requires_grad_(True)
    feat = model.features(x).flatten(1)
    proj = (feat * torch.as_tensor(w, dtype=feat.dtype)).sum()
    proj.backward()
    g = x.grad[0].abs().sum(0).numpy()
    return ndimage.gaussian_filter(g, 3)


def montage(ax_grid, order, images, cats, scores, title, fig):
    for k, ax in enumerate(ax_grid.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(order):
            ax.axis('off'); continue
        j = order[k]
        ax.imshow(images[j].numpy().transpose(1, 2, 0))
        for sp in ax.spines.values():
            sp.set_edgecolor(CAT_COLORS[cat_key(cats[j])]); sp.set_linewidth(3)
        ax.set_title('%+.2f' % scores[j], fontsize=6.5, pad=1.5)
    fig.suptitle(title, fontsize=9)


def main():
    data = rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(v2.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp = d['response']
    cats = data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()
    paths = som.condition_image_paths(data, v2.STIM_DIR)
    images = dnn_som.load_images_rgb(paths)
    model, _ = dnn_som.prep_dnn_model('cpu')
    feats = {nm: dnn_som.extract_dnn_features(model, layer, images, 'cpu').numpy() for nm, layer in v2.LAYERS.items()}
    ck = np.array([cat_key(c) for c in cats])
    isf = ck == 'face'

    # neural PC0 condition loading (oriented so faces load positive)
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    npc0 = Vt[0] * np.sign(np.corrcoef(Vt[0], isf.astype(float))[0, 1])
    print('PD: %d ZETA ROIs x %d images | categories %s | neural PC0 explains %.1f%% of ROI-centred variance'
          % (resp.shape[0], resp.shape[1], dict(zip(*np.unique(ck, return_counts=True))), 100 * S[0] ** 2 / (S ** 2).sum()))

    # (1) alignment
    print('\n[1] ALIGNMENT of the neural face axis (PC0 condition loading) with each layer\'s top-20 feature PCs')
    layer_pcs = {}
    for nm in v2.LAYERS:
        sc, comps, ev = pca_scores(feats[nm])
        layer_pcs[nm] = (sc, comps, ev)
        r = np.array([np.corrcoef(npc0, sc[:, k])[0, 1] for k in range(sc.shape[1])])
        k = int(np.argmax(np.abs(r)))
        r_face = np.corrcoef(sc[:, 0], isf.astype(float))[0, 1]
        print('    %-6s best PC %2d |r|=%.2f (PC0: r=%+.2f, ev=%.2f, r(PC0, face-indicator)=%+.2f)'
              % (nm, k, abs(r[k]), r[0], ev[0], r_face))
    print('    r(neural PC0 loading, face-indicator) = %+.2f' % np.corrcoef(npc0, isf.astype(float))[0, 1])

    # pool5-PC0, oriented like the face axis
    sc5, comps5, ev5 = layer_pcs['pool5']
    sgn = np.sign(np.corrcoef(sc5[:, 0], npc0)[0, 1])
    p5 = sc5[:, 0] * sgn
    w5 = comps5[0] * sgn

    # (2) montages
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axg = plt.subplots(6, 10, figsize=(15, 9.6))
    montage(axg, np.argsort(p5), images, cats, p5, 'PD images ranked by pool5-PC0 (low -> high, row-major); frame = category (red face, green body, blue object); '
            'number = PC0 score', fig)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'midlevel_montage_pool5pc0_pd.png'), dpi=110); plt.close(fig)
    fig, axg = plt.subplots(6, 10, figsize=(15, 9.6))
    montage(axg, np.argsort(npc0), images, cats, npc0, 'PD images ranked by the NEURAL PC0 loading (the face axis; low -> high); frame = category; number = loading', fig)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'midlevel_montage_neuralpc0_pd.png'), dpi=110); plt.close(fig)

    # (3) image statistics
    st = pd.DataFrame([image_stats(images[j]) for j in range(len(images))])
    Xcat = np.column_stack([(ck == 'face').astype(float), (ck == 'body').astype(float)])
    print('\n[3] IMAGE STATISTICS vs the two axes: r overall | r WITHIN category (both residualised on face/body dummies)')
    print('    %-20s %14s %14s | %14s %14s' % ('statistic', 'r pool5-PC0', 'r neural-PC0', 'within: pool5', 'within: neural'))
    p5w, npw = residualize(p5, Xcat), residualize(npc0, Xcat)
    for col in st.columns:
        v = st[col].to_numpy(float)
        vw = residualize(v, Xcat)
        print('    %-20s %+14.2f %+14.2f | %+14.2f %+14.2f' % (col, np.corrcoef(v, p5)[0, 1], np.corrcoef(v, npc0)[0, 1],
                                                              np.corrcoef(vw, p5w)[0, 1], np.corrcoef(vw, npw)[0, 1]))
    print('    within-category corr(pool5-PC0, neural-PC0) = %+.2f  (overall %+.2f)' % (np.corrcoef(p5w, npw)[0, 1], np.corrcoef(p5, npc0)[0, 1]))
    for k in ('face', 'body', 'object'):
        m = ck == k
        if m.sum() > 3:
            print('    within %-6s (n=%2d): corr(pool5-PC0, neural-PC0) = %+.2f' % (k, m.sum(), np.corrcoef(p5[m], npc0[m])[0, 1]))

    # (4) saliency
    order = np.argsort(p5)
    pick = list(order[-4:][::-1]) + list(order[:4])
    fig, axg = plt.subplots(2, 8, figsize=(16, 4.6))
    for c, j in enumerate(pick):
        sal = saliency(model, w5, images[j])
        axg[0, c].imshow(images[j].numpy().transpose(1, 2, 0)); axg[0, c].set_title('%s %+.2f' % (cat_key(cats[j]), p5[j]), fontsize=8)
        axg[1, c].imshow(images[j].numpy().transpose(1, 2, 0)); axg[1, c].imshow(sal, cmap='inferno', alpha=0.6)
        for a in axg[:, c]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle('input-gradient saliency of the pool5-PC0 projection: 4 highest (left) and 4 lowest (right) images; bright = pixels that move the score', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, 'midlevel_saliency_pool5pc0_pd.png'), dpi=110); plt.close(fig)
    print('\nfigures: output/midlevel_montage_pool5pc0_pd.png, output/midlevel_montage_neuralpc0_pd.png, output/midlevel_saliency_pool5pc0_pd.png')


if __name__ == '__main__':
    main()
