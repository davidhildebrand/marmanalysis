#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DNN feature extraction + self-organizing-map probing -- the MODEL side of the SOM-vs-cortex topography
comparison (Doshi & Konkle 2023, https://doi.org/10.1126/sciadv.ade8187).

Reimplements Fenil Doshi's model pipeline in our style, to be numerically PARITY-TESTED against his originals
(`Doshi_and_Konkle_2023_SciAdv/Code/python_scripts/{new_py_modelprep_extractfeatures,py_som_model,
py_pytorch_functions}.py`). Two stages:

  1. IMAGE -> DNN FEATURES: AlexNet (ImageNet object-recognition trained) relu7 = ``classifier.5``, 4096-D.
  2. FEATURES -> SOM: the shipped 20x20 = 400-unit object-rec SOM assigns each image a best-matching unit
     (BMU = argmin L2 over the 400 codebook vectors) and a simulated cortical activation
     (SCA = dot product of the feature vector with each unit's codebook).

Parity-critical details matched to the originals:
  * images: ``Resize -> CenterCrop(square) -> ToTensor`` with **NO ImageNet normalization** (the probe
    notebooks omit ``transforms.Normalize``); PIL images are converted to RGB first (drops the alpha channel);
  * inplace ReLUs are converted to non-inplace so the ``classifier.5`` forward hook reads the true relu7 output;
  * BMU uses Euclidean distance; SCA is the raw dot product (both over the 4096-D feature axis).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision

RELU7_LAYER = 'classifier.5'    # AlexNet relu7 (4096-D) -- the hook layer in Fenil's get_input_features
IMG_DIM = 224                   # Resize + CenterCrop square size (AlexNet input)


def convert_relu_inplace_false(module):
    """Recursively replace every ``nn.ReLU`` with a non-inplace one (Fenil's ``convert_relu``). A forward hook
    on an inplace-ReLU output can observe the tensor after a later inplace op mutates it; non-inplace makes the
    ``classifier.5`` (relu7) activation the hook captures well-defined."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif list(child.children()):
            convert_relu_inplace_false(child)


def prep_dnn_model(device='cpu'):
    """AlexNet trained on ImageNet object recognition, ReLUs made non-inplace, eval mode -- the object-rec DNN
    of Doshi & Konkle. Returns ``(model, hook_layer_name)``. Mirrors ``prep_dnn_model`` with
    ``training_mode='trained_default', task='objectrec', dataset='imagenet'`` (torchvision pretrained AlexNet)."""
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    convert_relu_inplace_false(model)
    return model.eval().to(device), RELU7_LAYER


def dnn_transform(img_dim=IMG_DIM):
    """PIL-image -> tensor transform matching Fenil's probe pipeline: ``Resize -> CenterCrop(square) ->
    ToTensor``, with NO ImageNet normalization. Apply to RGB images (see ``load_images_rgb``)."""
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(img_dim),
                               transforms.CenterCrop((img_dim, img_dim)),
                               transforms.ToTensor()])


def load_images_rgb(paths, img_dim=IMG_DIM):
    """Load image files -> a preprocessed ``(n, 3, img_dim, img_dim)`` float tensor. Each image is opened and
    ``.convert('RGB')``-ed (dropping alpha, as in Fenil's ImageFolder loader) then passed through
    ``dnn_transform``."""
    from PIL import Image
    tf = dnn_transform(img_dim)
    return torch.stack([tf(Image.open(p).convert('RGB')) for p in paths])


def extract_dnn_features(model, hook_layer_name, images, device='cpu'):
    """relu7 (4096-D) features for a batch of preprocessed image tensors, via a forward hook on
    ``hook_layer_name`` (``classifier.5``). ``images`` is ``(n, 3, H, W)``; returns ``(n, 4096)`` float32 on CPU.

    Reimplements the hook path of ``get_input_features``, but processes ALL ``n`` images in one pass -- Fenil's
    ``next(iter(dataloader))`` consumed only the first batch, so for a like-for-like parity comparison the
    reference must be run with ``batch_size >= n`` (or all images in one folder/batch)."""
    captured = {}

    def hook(_mod, _inp, out):
        a = out.detach()
        if a.dim() == 4:                                # flatten conv activations (N,C,H,W)->(N,C*H*W)
            a = a.reshape(a.shape[0], -1)
        captured['features'] = a

    module = dict(model.named_modules())[hook_layer_name]
    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(images.to(device))
    finally:
        handle.remove()
    return captured['features'].to('cpu').float()


def _register_som_stub():
    """Register a lightweight ``python_scripts.py_som_model.SOM`` (nn.Module subclass) in ``sys.modules`` so the
    shipped SOM checkpoint unpickles WITHOUT importing Fenil's heavy module (seaborn/networkx/etc.). Unpickling a
    ``torch.save``-d module restores its ``__dict__`` (weight, locations, ...) via ``__new__`` + state, so an
    empty subclass suffices -- we reimplement the SOM math (``som_bmu``/``som_sca``) rather than call its methods."""
    import types

    class SOM(nn.Module):
        pass

    pkg = sys.modules.setdefault('python_scripts', types.ModuleType('python_scripts'))
    pkg.__path__ = []                                  # mark as a package so the submodule import resolves
    for modname in ('python_scripts.py_som_model', 'py_som_model'):
        mod = sys.modules.setdefault(modname, types.ModuleType(modname))
        mod.SOM = SOM
    return SOM


def load_som(som_path, device='cpu'):
    """Load a shipped SOM checkpoint (e.g. ``.../models_from_paper/som_weights/objectrec_imagenet_trained_som.pth``).
    The checkpoint is a pickled SOM instance; a lightweight stub class is registered so it unpickles without
    Fenil's heavy imports. Returns the SOM module, whose ``.weight`` is ``(input_size, n_units)`` and
    ``.locations`` is ``(n_units, 2)``."""
    _register_som_stub()
    som = torch.load(som_path, map_location=device, weights_only=False)
    return som.to(device)


def som_bmu(som, features):
    """Best-matching-unit map location per feature vector: ``argmin`` L2 distance over the SOM's codebook units,
    via ``torch.cdist``. ``features`` is ``(n, input_size)``; returns ``(n, 2)`` map coordinates (same units as
    ``som.locations``).

    NB this is the INTENDED distance. Fenil's ``SOM.get_bmu``/``forward`` uses ``nn.PairwiseDistance`` on
    ``(n, D, 1)`` vs ``(n, D, U)``, which under torch >= 2.x reduces the LAST axis and returns ``(n, D)`` instead
    of ``(n, U)`` -- so his literal ``get_bmu`` yields a feature index, not a unit. ``cdist`` is the
    version-robust equivalent; verified in ``parity_dnn_som.py`` (his ``get_all_som_corr_act`` SCA is unaffected
    -- it reduces with an explicit ``sum``)."""
    features = torch.as_tensor(features, dtype=torch.float32)
    w = som.weight.detach().to(features)               # (input_size, n_units)
    dists = torch.cdist(features, w.t())               # (n, n_units) Euclidean
    idx = dists.argmin(dim=1)
    return som.locations.detach().to(features)[idx]


def som_sca(som, features):
    """Simulated cortical activation per SOM unit: dot product of each feature vector with each unit's codebook
    (matches ``SOM.get_all_som_corr_act``). ``features`` is ``(n, input_size)``; returns ``(n, n_units)``."""
    features = torch.as_tensor(features, dtype=torch.float32)
    w = som.weight.detach().to(features)               # (input_size, n_units)
    return features @ w                                # (n, n_units)


def _demo():
    """Smoke test: extract relu7 features for a few presented stimulus images and probe the shipped SOM."""
    import glob
    doshi = 'Doshi_and_Konkle_2023_SciAdv/Code'
    som_path = os.path.join(doshi, 'models_from_paper/som_weights/objectrec_imagenet_trained_som.pth')
    imgs = sorted(glob.glob('stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly/*.png'))[:5]
    print('images:', [os.path.basename(p) for p in imgs])

    model, layer = prep_dnn_model()
    x = load_images_rgb(imgs)
    feats = extract_dnn_features(model, layer, x)
    print('input tensor:', tuple(x.shape), '| relu7 features:', tuple(feats.shape),
          '| feature range [%.3f, %.3f]' % (float(feats.min()), float(feats.max())))

    if os.path.exists(som_path):
        som = load_som(som_path)
        print('SOM weight:', tuple(som.weight.shape), '| locations:', tuple(som.locations.shape))
        bmu = som_bmu(som, feats)
        sca = som_sca(som, feats)
        print('BMU locations:\n', bmu.numpy())
        print('SCA shape:', tuple(sca.shape), '| per-image argmax unit:', sca.argmax(1).numpy())
    else:
        print('SOM checkpoint not found at', som_path)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _demo()
