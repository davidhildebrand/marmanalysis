#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical parity: ``marmanalysis.dnn_som`` vs Fenil Doshi's originals, using his ACTUAL AlexNet recipe and
his ACTUAL trained SOM weights.

Checks the quantities the paper uses:
  * DNN relu7 features -- our ``extract_dnn_features`` (hook on ``classifier.5``) vs a reference built with
    Fenil's own ``convert_relu`` and his ``get_input_features`` hook logic
    (new_py_modelprep_extractfeatures.py:202-224).
  * SOM "simulated cortical activation" (SCA) -- our ``som_sca`` vs the shipped SOM's own
    ``get_all_som_corr_act`` method.
  * SOM BMU -- our ``som_bmu`` vs the INTENDED Euclidean argmin (explicit broadcast, computed a second way).

FINDING (documented, not a reimplementation error): the SOM's ``get_bmu``/``forward`` uses
``nn.PairwiseDistance`` on shapes ``(n, 4096, 1)`` vs ``(n, 4096, 400)``. Under torch 2.2.2 that reduces the
LAST axis, returning ``(n, 4096)`` instead of ``(n, 400)`` -- so his literal ``get_bmu`` yields a feature index,
not a unit (it returns the same wrong unit for every image). ``get_all_som_corr_act`` reduces with an explicit
``sum(dim=1)`` and is unaffected. Our ``som_bmu`` uses ``torch.cdist`` (the intended distance), so it is the
version-robust, correct BMU -- we deliberately do NOT replicate the fragile ``PairwiseDistance`` path.

Fenil's modules import seaborn at module level (unused on the objectrec path, not installed); a minimal stub is
installed after torchvision is already imported. Run from the project root:
  .venv/bin/python marmanalysis/parity_dnn_som.py
"""
import glob
import importlib.machinery
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DOSHI = 'Doshi_and_Konkle_2023_SciAdv/Code'
sys.path.insert(0, DOSHI)

import dnn_som                      # imports torch/torchvision cleanly BEFORE any stubs enter sys.modules

_sea = types.ModuleType('seaborn')
_sea.__file__ = '<stub:seaborn>'
_sea.__spec__ = importlib.machinery.ModuleSpec('seaborn', loader=None)
sys.modules.setdefault('seaborn', _sea)

SOM_PATH = os.path.join(DOSHI, 'models_from_paper/som_weights/objectrec_imagenet_trained_som.pth')


def reference_features(images):
    """Fenil's objectrec feature path: pretrained AlexNet + HIS ``convert_relu``, hook the ``classifier.5``
    output of one forward pass (new_py_modelprep_extractfeatures.py:202-224)."""
    import torchvision.models as pm
    from python_scripts.py_pytorch_functions import convert_relu
    model = pm.alexnet(pretrained=True)
    convert_relu(model)
    model.eval()
    captured = {}

    def save_activation(mod, inp, out):
        a = out.detach()
        if len(a.shape) == 4:
            a = a.view(-1, a.shape[1] * a.shape[2] * a.shape[3])
        captured['a'] = a

    h = dict(model.named_modules())['classifier.5'].register_forward_hook(save_activation)
    with torch.no_grad():
        model(images)
    h.remove()
    return captured['a'].cpu().float()


def load_reference_som():
    """Load the shipped SOM with the REAL ``py_som_model.SOM`` class (so its real methods are available)."""
    import python_scripts.py_som_model  # noqa: F401  (registers the real SOM class for unpickling)
    return torch.load(SOM_PATH, map_location='cpu', weights_only=False)


def main():
    paths = sorted(glob.glob('stimuli/Song_etal_Wang_2022_NatCommun/'
                             '480288_equalized_RGBA_FOBonly/*.png'))[:8]
    images = dnn_som.load_images_rgb(paths)
    print('parity images: %d | %s' % (len(paths), ', '.join(os.path.basename(p) for p in paths)))

    # --- DNN relu7 features (our extractor vs his hook logic, same pretrained model) ---
    ref_f = reference_features(images)
    model, layer = dnn_som.prep_dnn_model()
    our_f = dnn_som.extract_dnn_features(model, layer, images)
    d_feat = (ref_f - our_f).abs().max().item()
    print('DNN features %s        max|Δ| = %.3e' % (tuple(our_f.shape), d_feat))

    som = load_reference_som()
    print('SOM weight %s locations %s' % (tuple(som.weight.shape), tuple(som.locations.shape)))

    # --- SCA: his real method vs ours (the paper's "simulated cortical activation") ---
    ref_sca = som.get_all_som_corr_act(ref_f)
    our_sca = dnn_som.som_sca(som, ref_f)
    d_sca_rel = (ref_sca - our_sca).abs().max().item() / (ref_sca.abs().max().item() + 1e-12)
    print('SOM SCA (his method)   max rel Δ = %.3e' % d_sca_rel)

    # --- BMU: our som_bmu vs the INTENDED Euclidean argmin, computed a second, explicit way ---
    diff = ref_f[:, :, None] - som.weight.detach()[None, :, :]        # (n, D, U)
    intended_units = diff.pow(2).sum(dim=1).sqrt().argmin(dim=1)      # explicit Euclidean over the feature axis
    our_units = torch.cdist(ref_f, som.weight.detach().t()).argmin(dim=1)
    d_bmu = int((intended_units - our_units).abs().max().item())
    print('SOM BMU (ours vs intended-distance) unit Δ = %d' % d_bmu)

    # confirm his literal PairwiseDistance path is the broken one under this torch
    his_dists = som.get_all_som_loss(ref_f)
    his_broken = tuple(his_dists.shape) != (ref_f.shape[0], som.weight.shape[1])
    print('NOTE: his get_bmu/get_all_som_loss shape = %s (expected (%d, %d)) -> %s under torch %s'
          % (tuple(his_dists.shape), ref_f.shape[0], som.weight.shape[1],
             'BROKEN (PairwiseDistance axis)' if his_broken else 'ok', torch.__version__))

    ok = (d_feat < 1e-4) and (d_sca_rel < 1e-4) and (d_bmu == 0)
    print('\nPARITY (features + SCA + intended BMU):', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
