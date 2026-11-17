"""Unit tests for the detection-only sweep driver's pure helpers (suite2p_sweep.py): the scipy trace extraction that
replaces suite2p 1.x's torch version, the variant naming, and the settings merge. No binary, no suite2p run."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suite2p_sweep as sw  # noqa: E402


def test_extract_traces_numpy_matches_dense():
    rng = np.random.default_rng(0)
    T, Ly, Lx = 37, 20, 24
    mov = rng.integers(0, 2000, size=(T, Ly, Lx)).astype(np.int16)
    cell_masks, neuropil_masks = [], []
    for k in range(5):
        pix = rng.choice(Ly * Lx, 9, replace=False)
        lam = rng.uniform(0.1, 1.0, 9).astype(np.float32)
        lam /= lam.sum()
        cell_masks.append((pix, lam))
        neuropil_masks.append(rng.choice(Ly * Lx, 30, replace=False))
    F, Fneu = sw.extract_traces_numpy(mov, cell_masks, neuropil_masks, batch_size=10)
    flat = mov.reshape(T, -1).astype(np.float32)
    for k, (pix, lam) in enumerate(cell_masks):
        np.testing.assert_allclose(F[k], flat[:, pix] @ lam, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(Fneu[k], flat[:, neuropil_masks[k]].mean(1), rtol=1e-5, atol=1e-3)
    assert F.shape == (5, T) and F.dtype == np.float32


def test_extract_traces_numpy_without_neuropil():
    mov = np.ones((5, 4, 4), np.int16)
    F, Fneu = sw.extract_traces_numpy(mov, [(np.array([0, 1]), np.array([0.5, 0.5], np.float32))], None, batch_size=2)
    assert np.allclose(F, 1.0) and np.allclose(Fneu, 0.0)


BASE = {'anatomical_only': 3, 'diameter': 14.0, 'cellprob_threshold': -6.0, 'flow_threshold': 0.0, 'spatial_hp_cp': 0.5,
        'pretrained_model': 'cyto', 'max_overlap': 0.75, 'threshold_scaling': 0.2, 'sparse_mode': True,
        'min_neuropil_pixels': 350, 'inner_neuropil_radius': 2, 'allow_overlap': False, 'lam_percentile': 50.0,
        'neucoeff': 0.7, 'tau': 0.29}


def test_variant_name_encodes_detection_and_changed_extraction_keys():
    name = sw.variant_name(BASE, BASE)
    assert name.startswith('cp3_cyto_d14_pt-6_ft0_mo0p75_hp0p5') and '_np' not in name and name.endswith('s2p' + sw.S2P_VERSION.replace('.', ''))
    eff = dict(BASE, min_neuropil_pixels=175, inner_neuropil_radius=4, cellprob_threshold=-3.5)
    name = sw.variant_name(eff, BASE)
    assert 'pt-3p5' in name and '_np175' in name and '_in4' in name
    func = dict(BASE, anatomical_only=0)
    assert sw.variant_name(func, BASE).startswith('func_ts0p2_mo0p75')


def test_effective_settings_merges_and_rejects_unknown_keys():
    eff = sw.effective_settings(BASE, {'flow_threshold': 1.0, 'diameter': None})
    assert eff['flow_threshold'] == 1.0 and eff['diameter'] == 14.0 and eff['neucoeff'] == 0.7
    with pytest.raises(KeyError):
        sw.effective_settings(BASE, {'cellprob': -2})


def test_effective_settings_cellpose4_only_ships_cpsam():
    eff = sw.effective_settings(BASE, {'pretrained_model': 'cyto'})
    assert eff['pretrained_model'] == ('cpsam' if sw.V1 else 'cyto')
