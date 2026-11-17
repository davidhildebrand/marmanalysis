#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FISSA neuropil separation on a registered suite2p binary (roadmap item 13).

Runs in the ISOLATED FISSA venv (tmp/fissa_venv: python 3.10, fissa 1.0.0, scikit-learn 1.1.3, numpy < 2), because
FISSA 1.0 needs the old scikit-learn NMF API. FISSA itself reads TIFF stacks into memory; a 17 GB session does not
fit, so this script performs FISSA's two steps directly on the memory-mapped registered binary:
  1. regions   for every suite2p ROI, FISSA's own surround masks (fissa.roitools.getmasks_npil: `--n-regions` angular
               sectors around the ROI, each `--expansion` x the ROI area; neighbouring cells are NOT excluded, by
               design: the separation is what removes them), then the mean trace of the ROI and of each sector
               (chunked sparse products over the frames);
  2. separate  fissa.core.separate_trials (NMF, sparsity `--alpha`) per ROI on its (1 + n_regions) x T traces. The
               source contributing most to the ROI trace, scaled by that contribution, is the decontaminated cell
               trace (FISSA's "matched" signal 0).
OUTPUT = a pseudo extraction variant under <session>/sweep_suite2p/<name>/plane0/ that the pipeline loads like any
other variant (sessionio.find_suite2p_dir accepts the directory):
  stat.npy / iscell.npy / ops.npy   copied from the source variant (same ROIs, same order)
  F.npy      FISSA cell trace, shifted so its median equals the raw ROI trace's median (baseline-preserving, like the
             pipeline's own F - 0.7 (Fneu - median) correction, so F0 and dF/F stay meaningful)
  Fneu.npy   zeros -> the pipeline's neuropil subtraction is a no-op on this variant
  fissa_raw.npy (roi, 1 + n_regions, T)  fissa_cell.npy (roi, T; unshifted)  fissa_mixmat.npy  fissa_converged.npy
  fissa_coef.csv   per ROI: effective neuropil coefficient = slope of (raw ROI trace - FISSA cell trace) on the
                   source variant's suite2p Fneu, its correlation, and the trace correlation with F - 0.7 Fneu
  fissa_info.json / variant.json
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/tmp/fissa_venv/bin/python marmanalysis/fissa_separate.py \\
          <source variant dir> --reg-file <registered data.bin> [--out <dir>] [--limit N]
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse


def region_pixels(ypix, xpix, Ly, Lx, n_regions, expansion):
    """Flat pixel indices of the ROI and of FISSA's surround sectors (list of 1 + n_regions index arrays)."""
    from fissa import roitools
    mask = np.zeros((Ly, Lx), bool)
    mask[np.asarray(ypix), np.asarray(xpix)] = True
    # getmasks_npil returns ONLY the surround sectors (its docstring says "soma and neuropil masks", but the soma mask
    # is not in the list; fissa.core.extract prepends it the same way).
    sectors = roitools.getmasks_npil(mask, nNpil=n_regions, expansion=expansion)
    return [np.flatnonzero(mask)] + [np.flatnonzero(np.asarray(m, bool)) for m in sectors]


def build_region_matrix(regions, n_pix):
    """Row-normalised sparse matrix (n_roi * n_reg, n_pix): each row averages one region's pixels."""
    n_reg = len(regions[0])
    rows, cols, vals = [], [], []
    for i, regs in enumerate(regions):
        for j, idx in enumerate(regs):
            if idx.size == 0:
                continue
            rows.append(np.full(idx.size, i * n_reg + j, np.int64))
            cols.append(idx.astype(np.int64))
            vals.append(np.full(idx.size, 1.0 / idx.size, np.float32))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(regions) * n_reg, n_pix))


def extract_region_traces(reg_file, n_frames, Ly, Lx, M, batch=500, progress=True):
    """Mean trace of every region from the registered binary (int16, frames x Ly x Lx), in chunks."""
    mm = np.memmap(reg_file, dtype=np.int16, mode='r', shape=(n_frames, Ly, Lx))
    out = np.empty((M.shape[0], n_frames), np.float32)
    t_start = time.time()
    for t0 in range(0, n_frames, batch):
        chunk = np.asarray(mm[t0:t0 + batch], dtype=np.float32).reshape(-1, Ly * Lx)
        out[:, t0:t0 + chunk.shape[0]] = M @ np.ascontiguousarray(chunk.T)
        if progress and (t0 // batch) % 5 == 0:
            print('   traces: %d / %d frames (%.0f s)' % (t0 + chunk.shape[0], n_frames, time.time() - t_start))
            sys.stdout.flush()
    del mm
    return out


def separate_roi(X, alpha, max_iter, tol, max_tries):
    """FISSA separation of one ROI's (1 + n_regions, T) traces -> (cell trace, mixing matrix, converged, iterations)."""
    import warnings
    from fissa import core
    X = np.asarray(X, np.float64)
    if not np.isfinite(X).all() or np.ptp(X[0]) == 0:
        return X[0].astype(np.float32), np.full((X.shape[0], X.shape[0]), np.nan), False, 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, Xmatch, mixmat, conv = core.separate_trials([X], alpha=alpha, max_iter=max_iter, tol=tol,
                                                       max_tries=max_tries, method='nmf', verbosity=0)
    return (np.asarray(Xmatch[0][0], np.float32), np.asarray(mixmat, float), bool(conv.get('converged', False)),
            int(conv.get('iterations', 0) or 0))


def effective_coefficients(raw_roi, cell, F_s2p, Fneu_s2p, neucoeff=0.7):
    """Per ROI: slope / r of the part FISSA removed (raw ROI - cell) on suite2p's Fneu, and the correlation of the
    FISSA cell trace with the conventional correction F - neucoeff * Fneu."""
    removed = raw_roi - cell
    rc = removed - removed.mean(1, keepdims=True)
    nc = Fneu_s2p - Fneu_s2p.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = (rc * nc).sum(1) / (nc ** 2).sum(1)
        r_removed = (rc * nc).sum(1) / np.sqrt((rc ** 2).sum(1) * (nc ** 2).sum(1))
        conv = F_s2p - neucoeff * Fneu_s2p
        cc = conv - conv.mean(1, keepdims=True)
        fc = cell - cell.mean(1, keepdims=True)
        r_conv = (cc * fc).sum(1) / np.sqrt((cc ** 2).sum(1) * (fc ** 2).sum(1))
        rr = raw_roi - raw_roi.mean(1, keepdims=True)
        r_raw_neu = (rr * nc).sum(1) / np.sqrt((rr ** 2).sum(1) * (nc ** 2).sum(1))
        r_cell_neu = (fc * nc).sum(1) / np.sqrt((fc ** 2).sum(1) * (nc ** 2).sum(1))
        frac_removed_var = (rc ** 2).sum(1) / (rr ** 2).sum(1)
    return {'coef_eff': slope, 'r_removed_vs_fneu': r_removed, 'r_fissa_vs_conventional': r_conv,
            'r_raw_vs_fneu': r_raw_neu, 'r_fissa_vs_fneu': r_cell_neu, 'removed_var_frac': frac_removed_var}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='source variant directory (contains plane0/ with stat.npy, F.npy, Fneu.npy, ops.npy)')
    ap.add_argument('--reg-file', default=None, help='registered binary [default: <source>/plane0/data.bin]')
    ap.add_argument('--out', default=None, help='output variant dir [default: <session>/sweep_suite2p/fissa_<source name>]')
    ap.add_argument('--n-regions', type=int, default=4)
    ap.add_argument('--expansion', type=float, default=1.0, help='area of each surround sector relative to the ROI')
    ap.add_argument('--alpha', type=float, default=0.1, help='NMF sparsity weight (FISSA default 0.1)')
    ap.add_argument('--max-iter', type=int, default=20000)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--max-tries', type=int, default=1)
    ap.add_argument('--n-jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--batch', type=int, default=500, help='frames per extraction chunk')
    ap.add_argument('--limit', type=int, default=None, help='only the first N ROIs (smoke test)')
    a = ap.parse_args()

    import fissa
    import sklearn
    src = os.path.normpath(a.source)
    plane = os.path.join(src, 'plane0')
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(plane, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
    F_s2p = np.load(os.path.join(plane, 'F.npy'))
    Fneu_s2p = np.load(os.path.join(plane, 'Fneu.npy'))
    if a.limit:
        stat, iscell, F_s2p, Fneu_s2p = stat[:a.limit], iscell[:a.limit], F_s2p[:a.limit], Fneu_s2p[:a.limit]
    Ly, Lx, n_frames = int(ops['Ly']), int(ops['Lx']), int(ops['nframes'])
    reg_file = a.reg_file or os.path.join(plane, 'data.bin')
    if not os.path.isfile(reg_file):
        sys.exit('registered binary not found: %s' % reg_file)
    expect = n_frames * Ly * Lx * 2
    if os.path.getsize(reg_file) != expect:
        sys.exit('binary size %d does not match %d frames x %d x %d int16 (%d)' % (os.path.getsize(reg_file), n_frames, Ly, Lx, expect))
    session = os.path.dirname(src)
    out = os.path.normpath(a.out or os.path.join(session, 'sweep_suite2p', 'fissa_' + os.path.basename(src)))
    out_plane = os.path.join(out, 'plane0')
    os.makedirs(out_plane, exist_ok=True)
    n_roi, n_reg = len(stat), 1 + a.n_regions
    print('fissa %s, scikit-learn %s, numpy %s | source %s | %d ROIs, %d frames, %d x %d | binary %s'
          % (fissa.__version__, sklearn.__version__, np.__version__, src, n_roi, n_frames, Ly, Lx, reg_file))
    print('out %s | regions %d x expansion %g | NMF alpha %g | %d jobs' % (out, a.n_regions, a.expansion, a.alpha, a.n_jobs))
    sys.stdout.flush()
    t_all = time.time()

    t0 = time.time()
    regions = Parallel(n_jobs=a.n_jobs)(delayed(region_pixels)(s['ypix'], s['xpix'], Ly, Lx, a.n_regions, a.expansion)
                                         for s in stat)
    bad_regions = np.array([len(r) != n_reg or any(x.size == 0 for x in r) for r in regions])
    for i in np.flatnonzero(bad_regions):                        # keep the matrix rectangular; flagged below
        regions[i] = (list(regions[i]) + [np.zeros(0, np.int64)] * n_reg)[:n_reg]
    surround_px = np.array([[x.size for x in r] for r in regions])
    print('regions: %.0f s | ROI px median %d | surround sector px median %d | ROIs with a missing sector: %d'
          % (time.time() - t0, np.median(surround_px[:, 0]), np.median(surround_px[:, 1:]), int(bad_regions.sum())))
    sys.stdout.flush()

    t0 = time.time()
    M = build_region_matrix(regions, Ly * Lx)
    raw = extract_region_traces(reg_file, n_frames, Ly, Lx, M, batch=a.batch).reshape(n_roi, n_reg, n_frames)
    print('traces: %.0f s' % (time.time() - t0))
    sys.stdout.flush()

    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(separate_roi)(raw[i], a.alpha, a.max_iter, a.tol, a.max_tries)
                                                for i in range(n_roi))
    cell = np.stack([r[0] for r in res]).astype(np.float32)
    mixmat = np.stack([r[1] for r in res])
    converged = np.array([r[2] for r in res]) & ~bad_regions
    iterations = np.array([r[3] for r in res])
    cell[bad_regions] = raw[bad_regions, 0]                      # no separation possible -> raw ROI trace, flagged
    print('separation: %.0f s | converged %d / %d | iterations median %d' % (time.time() - t0, int(converged.sum()), n_roi,
                                                                             int(np.median(iterations))))
    sys.stdout.flush()

    raw_roi = raw[:, 0]
    F_out = cell - np.median(cell, 1, keepdims=True) + np.median(raw_roi, 1, keepdims=True)
    coef = effective_coefficients(raw_roi.astype(np.float64), cell.astype(np.float64), F_s2p.astype(np.float64),
                                  Fneu_s2p.astype(np.float64))
    coef['r_fissa_raw_vs_s2p_F'] = np.array([np.corrcoef(raw_roi[i], F_s2p[i])[0, 1] for i in range(n_roi)])

    np.save(os.path.join(out_plane, 'F.npy'), F_out.astype(np.float32))
    np.save(os.path.join(out_plane, 'Fneu.npy'), np.zeros_like(F_out, dtype=np.float32))
    np.save(os.path.join(out_plane, 'stat.npy'), stat)
    np.save(os.path.join(out_plane, 'iscell.npy'), iscell)
    ops_out = dict(ops)
    ops_out['fissa'] = {'version': fissa.__version__, 'n_regions': a.n_regions, 'expansion': a.expansion, 'alpha': a.alpha,
                        'source_variant': src, 'reg_file': reg_file}
    np.save(os.path.join(out_plane, 'ops.npy'), ops_out)
    np.save(os.path.join(out_plane, 'fissa_raw.npy'), raw)
    np.save(os.path.join(out_plane, 'fissa_cell.npy'), cell)
    np.save(os.path.join(out_plane, 'fissa_mixmat.npy'), mixmat)
    np.save(os.path.join(out_plane, 'fissa_converged.npy'), converged)
    cols = ['roi', 'accepted', 'converged', 'iterations', 'roi_px', 'sector_px_med'] + list(coef)
    table = np.column_stack([np.arange(n_roi), iscell[:, 0], converged, iterations, surround_px[:, 0],
                             np.median(surround_px[:, 1:], 1)] + [coef[k] for k in coef])
    np.savetxt(os.path.join(out_plane, 'fissa_coef.csv'), table, delimiter=',', header=','.join(cols), comments='', fmt='%.6g')

    def med(x, m=None):
        x = np.asarray(x, float)
        x = x[m] if m is not None else x
        return [float(v) for v in np.nanpercentile(x, [25, 50, 75])] if x.size else [np.nan] * 3

    acc = iscell[:, 0] > 0.5
    summary = {'n_roi': int(n_roi), 'n_converged': int(converged.sum()), 'n_missing_sector': int(bad_regions.sum()),
               'coef_eff_p25_p50_p75_all': med(coef['coef_eff']), 'coef_eff_p25_p50_p75_accepted': med(coef['coef_eff'], acc),
               'r_removed_vs_fneu_p25_p50_p75': med(coef['r_removed_vs_fneu']),
               'r_fissa_vs_conventional_p25_p50_p75': med(coef['r_fissa_vs_conventional']),
               'r_raw_vs_fneu_p25_p50_p75': med(coef['r_raw_vs_fneu']), 'r_fissa_vs_fneu_p25_p50_p75': med(coef['r_fissa_vs_fneu']),
               'removed_var_frac_p25_p50_p75': med(coef['removed_var_frac']),
               'r_fissa_raw_vs_s2p_F_p50': med(coef['r_fissa_raw_vs_s2p_F'])[1]}
    info = {'fissa_version': fissa.__version__, 'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'source_variant': src, 'reg_file': reg_file, 'n_regions': a.n_regions, 'expansion': a.expansion,
            'alpha': a.alpha, 'max_iter': a.max_iter, 'tol': a.tol, 'max_tries': a.max_tries, 'limit': a.limit,
            'runtime_s': round(time.time() - t_all, 1),
            'finished_utc': datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC'), 'summary': summary}
    with open(os.path.join(out_plane, 'fissa_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
    with open(os.path.join(out, 'variant.json'), 'w') as f:
        json.dump({'name': os.path.basename(out), 'suite2p_version': str(ops.get('suite2p_version', '')),
                   'settings_changed_vs_anchor': {'neuropil': 'FISSA %s NMF alpha %g, %d sectors x %g'
                                                  % (fissa.__version__, a.alpha, a.n_regions, a.expansion),
                                                  'rois_from': os.path.basename(src)},
                   'n_rois': int(n_roi), 'n_accepted': int(acc.sum()), 'runtime_s': info['runtime_s'],
                   'finished_utc': info['finished_utc']}, f, indent=1)
    for extra in ('spks.npy',):                                  # not used by the pipeline; copied for GUI completeness
        p = os.path.join(plane, extra)
        if os.path.isfile(p) and not a.limit:
            shutil.copyfile(p, os.path.join(out_plane, extra))
    print('\nsummary (p25, p50, p75):')
    for k, v in summary.items():
        print('   %-40s %s' % (k, ('[%.3f, %.3f, %.3f]' % tuple(v)) if isinstance(v, list) else v))
    print('done in %.0f s -> %s' % (time.time() - t_all, out))


if __name__ == '__main__':
    main()
