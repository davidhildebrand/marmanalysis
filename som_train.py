#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train our OWN self-organizing maps with Doshi & Konkle's recipe (roadmap item 15) -- two ADDITIONAL models kept
separate from the shipped 20x20 object-rec SOM:

  (a) a FINER map (e.g. 100x100 units) on a large natural-image relu7 manifold: the shipped 400-unit map tiles all
      of ImageNet object space at fMRI granularity, so a 0.7-3 mm cellular-resolution patch corresponds to a
      handful of its units and it can only predict a coarse smooth gradient there;
  (b) a map trained on OUR stimulus manifold (relu7 of the FOB / FOBmin / FOBmany images), so the layout reflects
      what the neurons were actually probed with.

The recipe reproduces `Doshi_and_Konkle_2023_SciAdv/Code/1_notebooks_train_SOM/train_som.ipynb` +
`python_scripts/{py_som_model,py_pca_init}.py`:
  * features: AlexNet (ImageNet object recognition) relu7 = ``classifier.5`` (any layer name works, e.g. pool5
    ``features.12``); NB the TRAINING notebook applies ImageNet mean/std normalisation to the images while the
    PROBE notebooks (and our ``dnn_som``) do not -- ``--normalize`` reproduces the training-side choice;
  * map shape: ``base_neurons`` units, aspect ratio = int(eigval0 / eigval1) of a 3-PC PCA on a 400-image
    "SRS" subset (``rows = int(sqrt(base_neurons / ar))``, ``cols = int(rows * ar)``) -- or a fixed ``--grid``;
  * PCA initialisation: codebook = mean + col_pos * sqrt(eigval0) * unit(eigvec0) + row_pos * sqrt(eigval1) *
    unit(eigvec1), positions min-max normalised to [-1, 1] (``get_init_codebook``);
  * training: 100 epochs over the training images in shuffled batches of 32; per batch, BMU by Euclidean
    distance, Gaussian neighbourhood ``exp(-d^2 / sigma^2)`` on the grid, update ``w += lr * h * (x - w)``
    averaged over the batch; ``lr = 0.3 * (1 - epoch / n_epochs)``; ``sigma = (max(rows, cols) / 2) *
    (1 - epoch / n_epochs)`` ('reducing' neighbourhood) -- both decay LINEARLY per EPOCH (not per batch).
    (His literal ``forward`` returns a feature index under torch>=2 -- see ``dnn_som.som_bmu``; we use cdist.)
  * output: a plain dict checkpoint {'weight' (input_size x n_units), 'locations' (n_units x 2), 'out_size',
    'meta'} that ``dnn_som.load_som`` wraps into the same module interface, so ``som.build_model_matrices``,
    ``som_alignment`` and the feature battery run unchanged with ``som_path=<new checkpoint>``.

Inputs: an image FOLDER (features extracted here, all images loaded to RAM -- fine for <= ~10k images) or a
pre-extracted features ``.npy`` (n_images x D). For the finer natural-image map a large image set is needed
(ImageNet val is NOT on this machine; downloading COCO / THINGS is a decision for DH).

DO NOT RUN ON THE DATA YET (DH 2026-09-24): implement now, test after the preprocessing / denoising details
(roadmap items 1, 6, 13) are settled. A synthetic self-test is provided (``--selftest``).

Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/som_train.py --selftest
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/som_train.py --images stimuli/StimSpace/FOBmany230728d \\
            --grid 12 12 --epochs 100 --out output/som/fobmany_relu7_12x12.pt
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/som_train.py --features tmp/imagenet_relu7.npy \\
            --base-neurons 10000 --out output/som/natural_relu7_100x100.pt
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som

IMG_EXT = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')


# ------------------------------------------------------------------------------------------------- features ----
def extract_features(image_dir, layer=dnn_som.RELU7_LAYER, normalize=False, batch=64, device='cpu', img_dim=224):
    """Features for every image in ``image_dir`` (sorted; alpha dropped). ``normalize`` applies the ImageNet
    mean/std transform used by Doshi's TRAINING notebook (the probe side omits it)."""
    from PIL import Image
    from torchvision import transforms
    paths = sorted(p for ext in IMG_EXT for p in glob.glob(os.path.join(image_dir, ext)))
    if not paths:
        raise FileNotFoundError('no images in %s' % image_dir)
    tf = [transforms.Resize(img_dim), transforms.CenterCrop((img_dim, img_dim)), transforms.ToTensor()]
    if normalize:
        tf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tf = transforms.Compose(tf)
    model, _ = dnn_som.prep_dnn_model(device)
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + batch]])
        out.append(dnn_som.extract_dnn_features(model, layer, x, device).numpy())
    return np.concatenate(out), paths


# ------------------------------------------------------------------------------------------- map shape + init ----
def _top_pcs(data, k=3):
    """Top-k eigenvectors / eigenvalues of the (mean-removed) data covariance, as sklearn PCA(svd_solver='full')."""
    X = np.asarray(data, np.float64)
    X = X - X.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigval = S[:k] ** 2 / (X.shape[0] - 1)
    return Vt[:k], eigval


def doshi_map_shape(data, base_neurons, set_ar=None):
    """``set_parameters_neurons``: aspect ratio = int(eigval0 / eigval1); rows = int(sqrt(n / ar)); cols = int(rows * ar)."""
    _, eigval = _top_pcs(data)
    ar = int(eigval[0] / eigval[1]) if set_ar is None else int(set_ar)
    ar = max(ar, 1)
    rows = int(math.sqrt(base_neurons / ar))
    cols = int(rows * ar)
    return rows, cols


def grid_locations(rows, cols):
    """(n_units, 2) grid coordinates in Doshi's order: unit i -> (i // cols, i % cols)."""
    return np.array([(i // cols, i % cols) for i in range(rows * cols)], float)


def pca_init_codebook(data, rows, cols):
    """``get_init_codebook``: mean + col_pos * sqrt(ev0) * unit(ev_vec0) + row_pos * sqrt(ev1) * unit(ev_vec1),
    grid positions min-max normalised to [-1, 1]. Returns (input_size, n_units)."""
    X = np.asarray(data, np.float64)
    mean = X.mean(0)
    eigvec, eigval = _top_pcs(X - mean)
    unit = eigvec / np.linalg.norm(eigvec, axis=1, keepdims=True)
    scaled = unit * np.sqrt(eigval)[:, None]
    pos = grid_locations(rows, cols)
    mx, mn = pos.max(0), pos.min(0)
    pos_n = (pos - mn) / np.where(mx > mn, mx - mn, 1.0)
    pos_n = (pos_n - 0.5) * 2
    codebook = mean[None, :] + pos_n[:, [0]] * scaled[1][None, :] + pos_n[:, [1]] * scaled[0][None, :]
    return codebook.T.astype(np.float32)


# ------------------------------------------------------------------------------------------------- training ----
def train_som(features, rows, cols, epochs=100, batch_size=32, lr0=0.3, sigma0=None, neighborhood='reducing',
              init='pca', seed=0, device='cpu', verbose=True, log_every=10):
    """Kohonen training exactly as ``SOM.self_organizing`` driven by Doshi's ``train_som`` loop. Returns
    (weight (input_size x n_units) float32 numpy, locations (n_units x 2), epoch_loss list)."""
    X = torch.as_tensor(np.asarray(features, np.float32), device=device)
    n, D = X.shape
    n_units = rows * cols
    loc = torch.as_tensor(grid_locations(rows, cols), dtype=torch.float32, device=device)
    g = torch.Generator(device='cpu').manual_seed(seed)
    if init == 'pca':
        W = torch.as_tensor(pca_init_codebook(features, rows, cols), device=device)
    else:
        W = torch.randn(D, n_units, generator=g).to(device)
    if sigma0 is None:
        sigma0 = max(rows, cols) / 2.0 if neighborhood == 'reducing' else 1.0
    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        corr = 1.0 - epoch / epochs
        lr = lr0 * corr
        sigma = sigma0 * corr if neighborhood == 'reducing' else sigma0
        perm = torch.randperm(n, generator=g)
        run, nb = 0.0, 0
        for i in range(0, n, batch_size):
            x = X[perm[i:i + batch_size]]                              # (b, D)
            d = torch.cdist(x, W.t())                                  # (b, n_units) Euclidean
            dmin, bmu = d.min(dim=1)
            run += float(dmin.sum()) / x.shape[0]; nb += 1
            d2 = ((loc[None, :, :] - loc[bmu][:, None, :]) ** 2).sum(-1)   # (b, n_units) grid distance^2
            h = torch.exp(-d2 / (sigma ** 2)) * lr                     # (b, n_units)
            # sum_b h[b,u] * (x[b,:] - W[:,u]) / B  ==  (x^T h - W * sum_b h[b,u]) / B  -- identical to Doshi's
            # broadcast form but without the (b, D, n_units) intermediate (5 GB at 100x100 units x 4096-D)
            delta = (x.t() @ h - W * h.sum(0)[None, :]) / x.shape[0]
            W = W + delta
        losses.append(run / max(nb, 1))
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print('  epoch %3d/%d  lr %.4f  sigma %.3f  BMU loss %.4f  (%.0f s)' % (epoch + 1, epochs, lr, sigma, losses[-1], time.time() - t0))
            sys.stdout.flush()
    return W.cpu().numpy().astype(np.float32), loc.cpu().numpy(), losses


def save_checkpoint(path, weight, locations, rows, cols, meta):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'weight': torch.as_tensor(weight), 'locations': torch.as_tensor(locations), 'out_size': (rows, cols), 'meta': meta}, path)


# -------------------------------------------------------------------------------------------------- self-test ----
def selftest():
    """Synthetic: 2000 points on a curved 2-D manifold embedded in 50-D + noise; a 10x10 map should (i) run,
    (ii) drop its BMU loss, (iii) come out topographically ordered (neighbouring units have nearby codebooks),
    and (iv) round-trip through ``dnn_som.load_som`` / ``som_bmu`` / ``som_sca``."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-1, 1, (2000, 2))
    basis = rng.standard_normal((4, 50))
    X = np.column_stack([t[:, 0], t[:, 1], t[:, 0] ** 2, t[:, 0] * t[:, 1]]) @ basis + 0.05 * rng.standard_normal((2000, 50))
    rows, cols = doshi_map_shape(X, 100)
    print('selftest: data %s -> Doshi map shape for 100 units = %dx%d' % (X.shape, rows, cols))
    rows, cols = 10, 10
    W, loc, losses = train_som(X, rows, cols, epochs=30, log_every=10)
    assert losses[-1] < losses[0], 'BMU loss did not decrease'
    # topographic order: codebook distance between grid neighbours << between random unit pairs
    Wt = W.T
    nb = [(i, i + 1) for i in range(rows * cols) if (i + 1) % cols]
    d_nb = np.mean([np.linalg.norm(Wt[i] - Wt[j]) for i, j in nb])
    pairs = rng.integers(0, rows * cols, (500, 2))
    d_rand = np.mean([np.linalg.norm(Wt[i] - Wt[j]) for i, j in pairs if i != j])
    print('  codebook distance grid-neighbours %.3f vs random unit pairs %.3f (ratio %.2f; ordered map << 1)' % (d_nb, d_rand, d_nb / d_rand))
    assert d_nb < 0.5 * d_rand, 'map not topographically ordered'
    path = 'tmp/som_selftest.pt'
    save_checkpoint(path, W, loc, rows, cols, {'selftest': True})
    som = dnn_som.load_som(path)
    bmu = dnn_som.som_bmu(som, X[:5]).numpy(); sca = dnn_som.som_sca(som, X[:5]).numpy()
    print('  round-trip via dnn_som.load_som: weight %s locations %s out_size %s | BMU %s | SCA %s'
          % (tuple(som.weight.shape), tuple(som.locations.shape), som.out_size, bmu.tolist()[:3], sca.shape))
    # the BMU of a point should be the unit whose codebook is nearest -- and points nearby in t should map nearby
    order = np.argsort(t[:, 0])[:200]
    b = dnn_som.som_bmu(som, X[order]).numpy()
    print('  200 points along manifold axis 1 -> BMU grid coords span rows %s cols %s (should mostly follow one axis)'
          % (np.ptp(b[:, 0]), np.ptp(b[:, 1])))
    os.remove(path)
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', help='image folder (features extracted here)')
    ap.add_argument('--features', help='pre-extracted features .npy (n_images x D) instead of --images')
    ap.add_argument('--layer', default=dnn_som.RELU7_LAYER, help='AlexNet hook layer (default relu7 classifier.5; pool5 = features.12)')
    ap.add_argument('--normalize', action='store_true', help='ImageNet mean/std normalisation (Doshi TRAINING-notebook side)')
    ap.add_argument('--grid', type=int, nargs=2, metavar=('ROWS', 'COLS'), help='fixed map shape (else Doshi aspect-ratio rule)')
    ap.add_argument('--base-neurons', type=int, default=400, help='total units for the aspect-ratio rule (Doshi: 400)')
    ap.add_argument('--srs-subset', type=int, default=400, help='size of the random image subset used for shape + PCA init (Doshi: 400)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.3)
    ap.add_argument('--sigma', type=float, default=None, help='initial neighbourhood sd (default max(rows, cols)/2)')
    ap.add_argument('--neighborhood', default='reducing', choices=['reducing', 'constant'])
    ap.add_argument('--init', default='pca', choices=['pca', 'random'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', help='checkpoint path (.pt)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.images or a.features) or not a.out:
        ap.error('need --images or --features, and --out (or --selftest)')
    if a.features:
        feats, paths = np.load(a.features), None
    else:
        feats, paths = extract_features(a.images, a.layer, a.normalize, device=a.device)
    print('features: %s from %s' % (feats.shape, a.features or a.images))
    rng = np.random.default_rng(a.seed)
    srs = feats[rng.choice(feats.shape[0], min(a.srs_subset, feats.shape[0]), replace=False)]
    rows, cols = tuple(a.grid) if a.grid else doshi_map_shape(srs, a.base_neurons)
    print('map shape %dx%d (%d units) | init %s | %s neighbourhood | %d epochs, batch %d, lr0 %.2f'
          % (rows, cols, rows * cols, a.init, a.neighborhood, a.epochs, a.batch_size, a.lr))
    init_feats = srs if a.init == 'pca' else feats
    W0 = pca_init_codebook(init_feats, rows, cols) if a.init == 'pca' else None
    W, loc, losses = train_som(feats if W0 is None else feats, rows, cols, epochs=a.epochs, batch_size=a.batch_size, lr0=a.lr,
                               sigma0=a.sigma, neighborhood=a.neighborhood, init=a.init, seed=a.seed, device=a.device)
    meta = {'source': a.features or a.images, 'n_images': int(feats.shape[0]), 'layer': a.layer, 'normalize': a.normalize,
            'rows': rows, 'cols': cols, 'epochs': a.epochs, 'batch_size': a.batch_size, 'lr0': a.lr, 'sigma0': a.sigma,
            'neighborhood': a.neighborhood, 'init': a.init, 'seed': a.seed, 'epoch_loss': losses,
            'image_paths': paths, 'recipe': 'Doshi & Konkle 2023 train_som.ipynb / py_som_model.py / py_pca_init.py'}
    save_checkpoint(a.out, W, loc, rows, cols, meta)
    with open(os.path.splitext(a.out)[0] + '_meta.json', 'w') as f:
        json.dump({k: v for k, v in meta.items() if k != 'image_paths'}, f, indent=1)
    print('saved %s (BMU loss %.4f -> %.4f)' % (a.out, losses[0], losses[-1]))


if __name__ == '__main__':
    main()
