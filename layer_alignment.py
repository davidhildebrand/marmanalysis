#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Face-axis alignment across EVERY AlexNet layer, plus TDANN's VTC-like layer when its checkpoint is present -- a
sanity check (DH 2026-10-18: "We should have code ready to produce a comparison across all layers").

For one session the neural axes are PCs of the ROI x condition response matrix (ROI-centered), as condition loadings.
First a table of what the top-5 neural PCs ARE: share of variance, |corr| with each category indicator and with the
images' mean luminance and RMS contrast. In the PD session PC0 is the face axis (|r| 0.90 with the face indicator);
in the 3-mm Curly fields it is not (0.11-0.16). The axes compared with the models are nPC0 and, when another top-5 PC
tracks faces (|r| >= spatial_structure.FACE_MIN_R), that face-aligned PC (oriented so faces load positive). For each
model layer, PCA of the layer's activations over the same images (conditions with an image; blanks and non-image
conditions excluded), then:
  r_pc0        |corr| between the neural axis and the layer's PC0 (the number quoted so far for relu7/relu6/pool5);
  best_pc      which of the layer's top-20 PCs matches the neural axis best, and its |corr| (selected, so optimistic);
  r2cv         leave-one-image-out R^2 of the neural axis from the layer's top-N_FIT PCs (not selected; about 0 or
               below for an unrelated layer);
  r2cv_within  the same beyond the category means: LOO error with category dummies + the layer's PCs, relative to
               the LOO error with the dummies alone -- how much of the axis a layer explains WITHIN categories, the
               part a category split carried by low-level image statistics cannot explain;
  r2cv_face    leave-one-face-out R^2 of the axis across the FACE images from the layer's top-3 PCs computed over
               the faces alone (which faces the axis prefers). Ceiling: in PD the axis's within-face split-half
               reliability is 0.61 (within category 0.50; tmp/face_axis_within_reliability.py);
  face_pc0     |corr| of the layer's PC0 with the face indicator (how much of PC0 is simply face vs rest).
  decode40     the layer comparison of Bao et al. 2020 (Nature, Extended Data Figs. 4g and 11): decode each image
               from the WHOLE population -- leave one image out, ridge regression from the population responses to
               the layer's top-50 PC scores (its "object space"), predict the left-out image's feature vector and
               identify it among 40 candidates by Euclidean distance (chance 0.025). Unlike the alignments above it
               uses every dimension of the layer and of the population, not just one axis. (Bao et al. found fc6 best
               with 51 objects x 24 views; with a set whose categories differ in pixel statistics, such as the PD
               face/body/object set, the one-axis alignments are flat across layers by construction.)
Both input transforms are run: plain [0,1] (the parity probe used so far) and ImageNet-normalized (the networks'
training input), and the input PIXELS are included as a 'layer 0' reference (if pixels align as well as pool5, the
alignment is a property of the image set, not of a processing stage). Images are the session's effective stimuli
(som.condition_image_paths).
Outputs: output/layer_alignment_<slug>_gate-<gate>_<UTC>.csv and .png
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/layer_alignment.py --session pd
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/layer_alignment.py --session Curly20240130d172813tUTC --gate responsive
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import som
import spatial_structure as ss

ALEXNET_LAYERS = [('conv1', 'features.0'), ('relu1', 'features.1'), ('pool1', 'features.2'),
                  ('conv2', 'features.3'), ('relu2', 'features.4'), ('pool2', 'features.5'),
                  ('conv3', 'features.6'), ('relu3', 'features.7'), ('conv4', 'features.8'), ('relu4', 'features.9'),
                  ('conv5', 'features.10'), ('relu5', 'features.11'), ('pool5', 'features.12'),
                  ('fc6', 'classifier.1'), ('relu6', 'classifier.2'), ('fc7', 'classifier.4'), ('relu7', 'classifier.5'),
                  ('fc8', 'classifier.6')]
N_PC = 20
N_FIT = 10
N_FIT_FACE = 3              # within-face fit: the layer's top-3 PCs over the face images only (20-25 faces per session)
N_NEURAL = 5
N_DEC_PC = 50               # Bao et al. 2020: a 50-D object space (fc6 PCs, 85% of the variance)
N_CAND = 40                 # 40-way identification, as in Bao et al.
N_DRAW = 50                 # random candidate sets per left-out image
RIDGE_GRID = np.logspace(-3, 3, 13)   # ridge penalties relative to the mean eigenvalue of the training Gram matrix
CATS = ('face', 'body', 'animal', 'object', 'scramble')
Y_LO = -0.5                 # figure floor; lower LOO R^2 values are printed at the floor
OUTDIR = 'output'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')


def layer_activations(model, images, layer_names):
    """All requested layers in one forward pass: {name: (n_images, n_units)}."""
    mods = dict(model.named_modules())
    out, hooks = {}, []
    for name in layer_names:
        hooks.append(mods[name].register_forward_hook(
            lambda _m, _i, o, name=name: out.__setitem__(name, o.detach().flatten(1).float().cpu().clone())))
    try:
        with torch.no_grad():
            model(images)
    finally:
        for h in hooks:
            h.remove()
    return {k: v.numpy() for k, v in out.items()}


def pc_scores(F, k=N_PC):
    F = np.asarray(F, float)
    Fc = F - F.mean(0)
    U, S, _ = np.linalg.svd(Fc, full_matrices=False)
    k = min(k, int((S > 1e-9 * S[0]).sum()) if S.size and S[0] > 0 else 0)
    return U[:, :k] * S[:k]


def loo_sse(y, X):
    """Leave-one-out squared error of an OLS fit of y on X (hat-matrix shortcut)."""
    H = X @ np.linalg.pinv(X)
    h = np.clip(np.diag(H), None, 1 - 1e-9)
    return float(np.sum(((y - H @ y) / (1 - h)) ** 2))


def alignment_row(axis, P, face, D, Pf=None):
    """``P``: the layer's PC scores; ``D``: category dummies (one category dropped); ``Pf``: the layer's PC scores
    over the face images only (for r2cv_face)."""
    if P.shape[1] == 0:
        return dict(r_pc0=np.nan, best_pc=-1, r_best=np.nan, r2cv=np.nan, r2cv_within=np.nan, r2cv_face=np.nan,
                    face_pc0=np.nan)
    r = np.array([abs(np.corrcoef(axis, P[:, j])[0, 1]) for j in range(P.shape[1])])
    one = np.ones((len(axis), 1))
    Pk = P[:, :N_FIT]
    r2_face = np.nan
    if Pf is not None and Pf.shape[1]:
        y = axis[face > 0]
        one_f = np.ones((len(y), 1))
        r2_face = 1 - loo_sse(y, np.hstack([one_f, Pf[:, :N_FIT_FACE]])) / loo_sse(y, one_f)
    return dict(r_pc0=float(r[0]), best_pc=int(np.argmax(r)), r_best=float(r.max()),
                r2cv=1 - loo_sse(axis, np.hstack([one, Pk])) / loo_sse(axis, one),
                r2cv_within=1 - loo_sse(axis, np.hstack([one, D, Pk])) / loo_sse(axis, np.hstack([one, D])),
                r2cv_face=r2_face, face_pc0=float(abs(np.corrcoef(P[:, 0], face)[0, 1])))


def loo_ridge_folds(X):
    """Per left-out image, the pieces for kernel-ridge prediction from the population responses ``X`` (n_images x
    n_cells), centered on the training images: the eigen-decomposition of the training Gram matrix and the left-out
    image's kernel row. Computed once and reused for every layer."""
    folds = []
    for i in range(len(X)):
        tr = np.delete(np.arange(len(X)), i)
        mu = X[tr].mean(0)
        Xt = X[tr] - mu
        lam, U = np.linalg.eigh(Xt @ Xt.T)
        folds.append((tr, np.clip(lam, 0, None), U, Xt @ (X[i] - mu)))
    return folds


def decode_accuracy(folds, Y, rng):
    """Bao et al. 2020-style decoding: for each left-out image, kernel ridge regression from the population responses
    to the layer's PC scores ``Y`` (n_images x k) fitted on the other images (penalty chosen by closed-form inner
    leave-one-out), the left-out image's feature vector predicted and identified among N_CAND candidates (it and
    random others) by Euclidean distance. Returns the fraction correct (chance 1 / N_CAND)."""
    n = len(Y)
    Yhat = np.zeros_like(Y)
    for i, (tr, lam, U, k) in enumerate(folds):
        ym = Y[tr].mean(0)
        UtY = U.T @ (Y[tr] - ym)
        best = (np.inf, None)
        for a in RIDGE_GRID * max(lam.mean(), 1e-12):
            f = lam / (lam + a)
            resid = (Y[tr] - ym) - U @ (f[:, None] * UtY)
            err = np.sum((resid / (1 - (U ** 2) @ f)[:, None]) ** 2)
            if err < best[0]:
                best = (err, a)
        Yhat[i] = ym + (k @ U) @ (UtY / (lam + best[1])[:, None])
    nc = min(N_CAND, n)
    hits = []
    for i in range(n):
        others = np.delete(np.arange(n), i)
        for _ in range(N_DRAW if nc < n else 1):
            cand = np.r_[i, rng.choice(others, nc - 1, replace=False)]
            hits.append(np.argmin(np.sum((Y[cand] - Yhat[i]) ** 2, 1)) == 0)
    return float(np.mean(hits))


def image_stats(x):
    """Mean luminance and RMS contrast of the plain [0,1] image tensors (n, 3, H, W)."""
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).numpy()
    lum = g.mean((1, 2))
    return dict(luminance=lum, rms_contrast=g.std((1, 2)) / (lum + 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default='pd', choices=sorted(ss.SESSIONS) + sorted(ss.ALIASES))
    ap.add_argument('--gate', default=None, choices=['zeta', 'responsive', 'all'])
    a = ap.parse_args()
    cfg = ss.resolve_session(a.session)
    slug = cfg['slug']
    gate = a.gate or cfg['gate']
    data, resp, xy, rel, mask, n_total = ss.load_field(cfg, gate, cfg.get('chunk_um', 0))
    ck = np.array([ss.cat_key(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
    paths = som.condition_image_paths(data)
    use = np.array([p is not None and os.path.exists(p) for p in paths]) & ~np.isin(ck, ('blank', 'other'))
    cku = ck[use]
    face = (cku == 'face').astype(float)
    present = [c for c in CATS if (cku == c).any()]
    D = np.column_stack([(cku == c).astype(float) for c in present[1:]]) if len(present) > 1 else np.zeros((use.sum(), 0))
    img_paths = [p for p, k in zip(paths, use) if k]
    x_plain = dnn_som.load_images_rgb(img_paths, normalize=False)
    stats = image_stats(x_plain)

    R = np.nan_to_num(resp[:, use] - np.nanmean(resp[:, use], 0, keepdims=True))
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    print('%s | gate %s, %d of %d ROIs | %d images (%s)' % (slug, gate, resp.shape[0], n_total, use.sum(),
                                                           dict(zip(*np.unique(cku, return_counts=True)))))
    print('what the top-%d neural PCs are: |corr| of each condition loading with each category indicator and image statistic'
          % N_NEURAL)
    print('  PC  %%var  ' + ' '.join('%8s' % c for c in present) + '  luminance  rms_contrast')
    face_r = []
    for j in range(min(N_NEURAL, len(Vt))):
        rc = [abs(np.corrcoef(Vt[j], (cku == c).astype(float))[0, 1]) for c in present]
        rs = [abs(np.corrcoef(Vt[j], stats[s])[0, 1]) for s in ('luminance', 'rms_contrast')]
        face_r.append(abs(np.corrcoef(Vt[j], face)[0, 1]))
        print('  %2d  %4.1f  ' % (j, 100 * ev[j]) + ' '.join('%8.2f' % v for v in rc) + '  %9.2f  %12.2f' % tuple(rs))
    axes = {'nPC0': Vt[0] * np.sign(np.corrcoef(Vt[0], face)[0, 1])}
    jf = int(np.argmax(face_r))
    if jf != 0 and face_r[jf] >= ss.FACE_MIN_R:
        axes['nPC%d' % jf] = Vt[jf] * np.sign(np.corrcoef(Vt[jf], face)[0, 1])
    print('  -> neural PC0 %s the face axis here (|r| %.2f)%s' % (
        'IS' if face_r[0] >= ss.FACE_MIN_R else 'is NOT', face_r[0],
        '; the face-aligned PC is nPC%d (|r| %.2f), compared too' % (jf, face_r[jf]) if len(axes) > 1 else
        ('' if face_r[0] >= ss.FACE_MIN_R else '; no top-%d PC tracks faces (max |r| %.2f)' % (N_NEURAL, max(face_r)))))
    folds = loo_ridge_folds(np.nan_to_num(resp[:, use]).T)   # population responses, images x cells
    k_dec = min(N_DEC_PC, int(use.sum()) - 2)
    drng = np.random.default_rng(0)

    model, _ = dnn_som.prep_dnn_model('cpu')
    rows = []
    for transform in ('plain', 'normalized'):
        x = x_plain if transform == 'plain' else dnn_som.load_images_rgb(img_paths, normalize=True)
        acts = layer_activations(model, x, [m for _, m in ALEXNET_LAYERS])
        acts['pixels'] = x.flatten(1).numpy().astype(float)          # the input itself: a 'layer 0' reference
        for name, mod in [('pixels', 'pixels')] + ALEXNET_LAYERS:
            P, Pf = pc_scores(acts[mod]), pc_scores(acts[mod][face > 0], k=N_FIT_FACE)
            dec = decode_accuracy(folds, pc_scores(acts[mod], k=k_dec), drng)
            for an, axis in axes.items():
                rows.append(dict(neural_axis=an, model='AlexNet', layer=name, transform=transform,
                                 n_units=acts[mod].shape[1], decode40=dec, **alignment_row(axis, P, face, D, Pf)))
        del acts
    try:
        import tdann_probe as tp
        if os.path.exists(tp.path_of('checkpoint')):
            tmodel = tp.load_tdann(tp.path_of('checkpoint'))
            F = tp.layer_responses(tmodel, dnn_som.load_images_rgb(img_paths, normalize=True))
            P, Pf = pc_scores(F), pc_scores(F[face > 0], k=N_FIT_FACE)
            dec = decode_accuracy(folds, pc_scores(F, k=k_dec), drng)
            for an, axis in axes.items():
                rows.append(dict(neural_axis=an, model='TDANN', layer='layer4.1', transform='normalized',
                                 n_units=F.shape[1], decode40=dec, **alignment_row(axis, P, face, D, Pf)))
    except Exception as e:                                   # TDANN is optional here
        print('  (TDANN row skipped: %s: %s)' % (type(e).__name__, str(e)[:120]))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(df.to_string(index=False, float_format=lambda v: '%.3f' % v))
    for an in axes:
        for transform in ('plain', 'normalized'):
            d = df[(df['neural_axis'] == an) & (df['model'] == 'AlexNet') & (df['transform'] == transform)]
            b, w, f = d.loc[d['r2cv'].idxmax()], d.loc[d['r2cv_within'].idxmax()], d.loc[d['r2cv_face'].idxmax()]
            print('  %s, %s input: best LOO R^2 %s %.2f | best within-category LOO R^2 %s %.2f | best within-face '
                  'LOO R^2 %s %.2f' % (an, transform, b['layer'], b['r2cv'], w['layer'], w['r2cv_within'],
                                       f['layer'], f['r2cv_face']))
    for transform in ('plain', 'normalized'):
        d = df[(df['neural_axis'] == 'nPC0') & (df['model'] == 'AlexNet') & (df['transform'] == transform)]
        print('  Bao-style %d-way decoding (top-%d layer PCs), %s input: %s | best %s %.3f (chance %.3f)'
              % (min(N_CAND, use.sum()), k_dec, transform, ' '.join('%s %.2f' % (r_['layer'], r_['decode40']) for _, r_ in d.iterrows()),
                 d.loc[d['decode40'].idxmax(), 'layer'], d['decode40'].max(), 1.0 / min(N_CAND, use.sum())))
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, 'layer_alignment_%s_gate-%s_%s' % (slug, gate, STAMP))
    df.to_csv(base + '.csv', index=False)

    names = ['pixels'] + [n for n, _ in ALEXNET_LAYERS]
    xs = np.arange(len(names))
    fig, ax = plt.subplots(len(axes) + 1, 2, figsize=(15, 6.2 * (len(axes) + 1)), squeeze=False)
    for row, (an, axis) in enumerate(axes.items()):
        for a_, transform in zip(ax[row], ('plain', 'normalized')):
            d = df[(df['neural_axis'] == an) & (df['model'] == 'AlexNet') & (df['transform'] == transform)].set_index('layer').loc[names]
            for col, style, color, lab in (('r_pc0', '-o', 'tab:blue', '|r| with layer PC0'),
                                           ('r_best', '-s', 'tab:orange', '|r| with best of top-%d PCs (selected)' % N_PC),
                                           ('r2cv', '-^', 'tab:green', 'LOO R^2 from top-%d PCs' % N_FIT),
                                           ('r2cv_within', '-v', 'tab:brown', 'LOO R^2 within categories'),
                                           ('r2cv_face', '-d', 'tab:pink', 'LOO R^2 within faces (top-%d face PCs)' % N_FIT_FACE),
                                           ('face_pc0', ':', '0.4', 'layer PC0 vs face indicator')):
                v = d[col].to_numpy(float)
                a_.plot(xs, np.maximum(v, Y_LO), style, color=color, alpha=0.5 if col == 'r_best' else 1, label=lab)
                for x_, val in zip(xs, v):
                    if val < Y_LO:                           # off-scale (leverage of single-image PCs): print the value
                        a_.annotate('%.1f' % val, (x_, Y_LO), xytext=(0, 5), textcoords='offset points', ha='center',
                                    fontsize=6, color=color)
            t = df[(df['model'] == 'TDANN') & (df['neural_axis'] == an)]
            if transform == 'normalized' and len(t):
                a_.axhline(float(t['r_pc0'].iloc[0]), color='tab:purple', ls='--', lw=1, label='TDANN layer4.1 PC0 |r|')
                a_.axhline(max(float(t['r2cv_within'].iloc[0]), Y_LO), color='tab:purple', ls=':', lw=1,
                           label='TDANN within-category LOO R^2')
            a_.axhline(abs(np.corrcoef(axis, face)[0, 1]), color='tab:red', ls='--', lw=1, label='neural axis vs face indicator')
            a_.axhline(0, color='k', lw=0.5)
            a_.set_title('neural %s, %s input' % (an, transform), fontsize=10)
            a_.set_xticks(xs); a_.set_xticklabels(names, rotation=60)
            a_.set_ylim(Y_LO - 0.02, 1.02)
            a_.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax[row, 0].set_ylabel('alignment with neural %s' % an)
    chance = 1.0 / min(N_CAND, use.sum())
    for a_, transform in zip(ax[-1], ('plain', 'normalized')):
        d = df[(df['neural_axis'] == 'nPC0') & (df['model'] == 'AlexNet') & (df['transform'] == transform)].set_index('layer').loc[names]
        a_.plot(xs, d['decode40'], '-o', color='k', label='%d-way decoding from the population, top-%d layer PCs (Bao et al. 2020)'
                % (min(N_CAND, use.sum()), k_dec))
        a_.axhline(chance, color='0.5', ls=':', label='chance %.3f' % chance)
        t = df[(df['model'] == 'TDANN') & (df['neural_axis'] == 'nPC0')]
        if transform == 'normalized' and len(t):
            a_.axhline(float(t['decode40'].iloc[0]), color='tab:purple', ls='--', lw=1, label='TDANN layer4.1')
        a_.set_title('image identification from the whole population, %s input' % transform, fontsize=10)
        a_.set_xticks(xs); a_.set_xticklabels(names, rotation=60)
        a_.set_ylim(0, max(0.2, float(np.nanmax(df['decode40'])) * 1.15))
        a_.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[-1, 0].set_ylabel('fraction correct')
    fig.suptitle('%s: neural axes (%d ROIs, gate %s) vs every AlexNet layer, %d effective images; neural PC0 vs face '
                 'indicator |r| %.2f' % (slug, resp.shape[0], gate, use.sum(), face_r[0]), fontsize=10)
    fig.tight_layout()
    fig.savefig(base + '.png', dpi=110, bbox_inches='tight')
    print('table: %s.csv | figure: %s.png' % (base, base))


if __name__ == '__main__':
    main()
