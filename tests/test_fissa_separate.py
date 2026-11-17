"""Tests for the FISSA-free parts of fissa_separate: region averaging matrix, binary trace extraction, and the
effective-coefficient bookkeeping. (The separation itself needs the isolated FISSA venv and is smoke-tested there.)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import fissa_separate as fs  # noqa: E402


def test_region_matrix_and_extraction(tmp_path):
    Ly, Lx, T = 6, 5, 40
    rng = np.random.default_rng(0)
    movie = rng.integers(0, 500, size=(T, Ly, Lx)).astype(np.int16)
    reg_file = str(tmp_path / 'data.bin')
    movie.tofile(reg_file)
    # two ROIs x three regions (ROI + 2 sectors), given as flat pixel indices; one sector is empty
    regions = [[np.array([0, 1, 5]), np.array([2, 3]), np.array([10, 11, 12, 13])],
               [np.array([20, 21]), np.array([], int), np.array([25, 26, 27])]]
    M = fs.build_region_matrix(regions, Ly * Lx)
    assert M.shape == (6, Ly * Lx)
    sums = np.asarray(M.sum(1)).ravel()
    assert np.allclose(sums[[0, 1, 2, 3, 5]], 1.0) and sums[4] == 0          # rows average; the empty sector is all zero
    tr = fs.extract_region_traces(reg_file, T, Ly, Lx, M, batch=7, progress=False)   # batch not dividing T
    flat = movie.reshape(T, -1).astype(np.float64)
    for i, regs in enumerate(regions):
        for j, idx in enumerate(regs):
            want = flat[:, idx].mean(1) if idx.size else np.zeros(T)
            assert np.allclose(tr[i * 3 + j], want, atol=1e-3)


def test_effective_coefficients_recover_known_mixture():
    rng = np.random.default_rng(1)
    T, n = 4000, 3
    cell = rng.gamma(2.0, 20.0, size=(n, T))
    fneu = 300 + 30 * rng.standard_normal((n, T))
    true = np.array([0.0, 0.6, 1.0])
    raw = cell + true[:, None] * fneu                       # ROI trace = cell + c * neuropil
    out = fs.effective_coefficients(raw, cell, raw, fneu, neucoeff=0.7)
    assert np.allclose(out['coef_eff'][1:], true[1:], atol=1e-6)
    assert abs(out['coef_eff'][0]) < 1e-9
    assert np.allclose(out['r_removed_vs_fneu'][1:], 1.0)
    assert np.all(np.abs(out['r_fissa_vs_fneu']) < 0.1)     # the recovered cell trace is independent of the neuropil
    assert out['r_raw_vs_fneu'][2] > out['r_raw_vs_fneu'][1] > 0.1
    # F - 0.7 Fneu is closest to the true cell where the true coefficient is nearest 0.7, and worst where there was
    # no contamination to remove (over-correction injects -0.7 * neuropil noise).
    r = out['r_fissa_vs_conventional']
    assert r[1] > 0.99 and r[1] > r[2] > r[0] > 0.7
