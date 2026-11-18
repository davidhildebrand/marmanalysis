#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI matching between extraction variants of one session (roadmap item 1: detection completeness).

A variant's ROI COUNT says nothing about whether its extra ROIs are cells. This script matches the ROIs of a reference
variant to those of each comparison variant by pixel overlap (IoU of the suite2p masks; a pair is SHARED when the two
ROIs are each other's best match and IoU >= --iou) and reports, for the SHARED / LOST (reference only) / NEW
(comparison only) ROIs:
  classifier-accepted fraction, responsive and selective (alpha 0.01) fractions from the trial-level ANOVA gates,
  median split-half tuning reliability, median ROI size (pixels)
and, for the shared pairs, the correlation between the two tuning vectors of the same cell (how much the extraction
variant changes a cell's measured tuning; also restricted to cells responsive in the reference).
Response-side numbers use the pipeline defaults (Fzsc, neuropil correction 0.7, eye gate 'landing_window'), exactly
as suite2p_sweep_readouts.py does, so the two tables are consistent.
Writes tmp/suite2p_sweep/<slug>_match_<UTC>.csv and prints a comparison table.
Run:  .venv/bin/python marmanalysis/suite2p_sweep_match.py \\
          --refs <variant> [<variant> ...] [--variants <variant> ...] [--iou 0.5]
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_matrix as rm
import response_table as rt
import sessionio
import suite2p_sweep_readouts as sr

GROUP_COLS = ('accepted', 'responsive', 'selective01')


def variant_props(session, variant, reliability_splits=20):
    """Per-ROI masks and properties of one variant, rows aligned with the response table (sessionio 'cellinds')."""
    data = rm.build_response_matrix(session, variant=variant, denoise=False, reliability_splits=reliability_splits)
    s2p = data['ctx']['s2p']
    rois = s2p['ROIs']
    Ly, Lx = (int(v) for v in s2p['fov_size'])
    cls = rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)
    p_sel = np.asarray(cls['p_selective'], float)
    n = len(rois)
    if data['response'].shape[0] != n:
        raise RuntimeError('response table rows (%d) do not match the ROI list (%d)' % (data['response'].shape[0], n))
    rows = np.concatenate([np.full(len(r['xpix']), i, np.int64) for i, r in enumerate(rois)])
    cols = np.concatenate([np.asarray(r['ypix'], np.int64) * Lx + np.asarray(r['xpix'], np.int64) for r in rois])
    mask = sparse.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, Ly * Lx))
    mask.data[:] = 1.0                                           # duplicate pixel entries (if any) count once
    return {'name': os.path.basename(os.path.normpath(variant)) if os.path.isdir(variant) else variant,
            'mask': mask, 'npix': np.asarray(mask.sum(1)).ravel(),
            'accepted': np.asarray(s2p['iscell'])[s2p['cellinds'], 0] > 0.5,
            'responsive': np.asarray(cls['responsive'], bool),
            'selective01': (p_sel < 0.01) & np.isfinite(p_sel),
            'reliability': np.asarray(data['reliability'], float),
            'response': np.asarray(data['response'], float), 'condition_ids': np.asarray(data['condition_ids'])}


def match_rois(A, B, thr=0.5):
    """Mutual-best-match pairs (i in A, j in B) with IoU >= thr, plus the dense IoU matrix."""
    inter = np.asarray((A['mask'] @ B['mask'].T).todense(), float)
    union = A['npix'][:, None] + B['npix'][None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    if iou.size == 0:
        return np.zeros((0, 2), int), iou
    best_b, best_a = iou.argmax(1), iou.argmax(0)
    ia = np.arange(iou.shape[0])
    mutual = (best_a[best_b] == ia) & (iou[ia, best_b] >= thr)
    return np.column_stack([ia[mutual], best_b[mutual]]), iou


def group_stats(P, idx, prefix):
    out = {prefix + '_n': int(len(idx))}
    if len(idx) == 0:
        return out
    for k in GROUP_COLS:
        out['%s_%s' % (prefix, k)] = float(P[k][idx].mean())
    out[prefix + '_reliability_med'] = float(np.nanmedian(P['reliability'][idx]))
    out[prefix + '_npix_med'] = float(np.median(P['npix'][idx]))
    return out


def pair_tuning_r(A, B, pairs):
    """Pearson r between the tuning vectors of each matched pair, over the conditions both variants have."""
    common, ia, ib = np.intersect1d(A['condition_ids'], B['condition_ids'], return_indices=True)
    if len(pairs) == 0 or common.size < 3:
        return np.zeros(0)
    X, Y = A['response'][pairs[:, 0]][:, ia], B['response'][pairs[:, 1]][:, ib]
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = np.where(ok, X, np.nan), np.where(ok, Y, np.nan)
    X = X - np.nanmean(X, 1, keepdims=True)
    Y = Y - np.nanmean(Y, 1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nansum(X * Y, 1) / np.sqrt(np.nansum(X ** 2, 1) * np.nansum(Y ** 2, 1))


def compare(A, B, thr=0.5):
    pairs, iou = match_rois(A, B, thr)
    na, nb = len(A['npix']), len(B['npix'])
    lost = np.setdiff1d(np.arange(na), pairs[:, 0])
    new = np.setdiff1d(np.arange(nb), pairs[:, 1])
    best_a = iou.max(1) if iou.size else np.zeros(na)
    row = {'ref': A['name'], 'cmp': B['name'], 'n_ref': na, 'n_cmp': nb, 'iou_thr': thr,
           'ref_recovered_frac': len(pairs) / max(na, 1),
           'ref_partial_frac': float(np.mean((best_a >= 0.2) & (best_a < thr))) if na else np.nan,   # split / merged / shifted
           'ref_no_overlap_frac': float(np.mean(best_a < 0.2)) if na else np.nan,
           'iou_med_shared': float(np.median(iou[pairs[:, 0], pairs[:, 1]])) if len(pairs) else np.nan,
           'n_responsive_ref': int(A['responsive'].sum()), 'n_responsive_cmp': int(B['responsive'].sum()),
           'n_selective01_ref': int(A['selective01'].sum()), 'n_selective01_cmp': int(B['selective01'].sum())}
    row.update(group_stats(A, pairs[:, 0], 'shared_ref'))
    row.update(group_stats(B, pairs[:, 1], 'shared_cmp'))
    row.update(group_stats(A, lost, 'lost'))
    row.update(group_stats(B, new, 'new'))
    # Unmatched ROIs split by whether they overlap ANY ROI of the other variant: 'partial' (best IoU >= 0.2) is a
    # reshaped / split / merged version of a cell the other variant also has; 'isolated' is territory it lacks.
    best_b = iou.max(0) if iou.size else np.zeros(nb)
    row.update(group_stats(A, lost[best_a[lost] >= 0.2], 'lost_partial'))
    row.update(group_stats(A, lost[best_a[lost] < 0.2], 'lost_isolated'))
    row.update(group_stats(B, new[best_b[new] >= 0.2], 'new_partial'))
    row.update(group_stats(B, new[best_b[new] < 0.2], 'new_isolated'))
    row['new_responsive_n'] = int(B['responsive'][new].sum())
    row['lost_responsive_n'] = int(A['responsive'][lost].sum())
    r = pair_tuning_r(A, B, pairs)
    if r.size:
        resp = A['responsive'][pairs[:, 0]]
        row.update(tuning_r_med=float(np.nanmedian(r)),
                   tuning_r_med_ref_responsive=float(np.nanmedian(r[resp])) if resp.any() else np.nan,
                   tuning_r_frac_gt_0p8_ref_responsive=float(np.nanmean(r[resp] > 0.8)) if resp.any() else np.nan,
                   responsive_agree_frac=float(np.mean(A['responsive'][pairs[:, 0]] == B['responsive'][pairs[:, 1]])))
    return row


SHOW = ['n_ref', 'n_cmp', 'ref_recovered_frac', 'ref_partial_frac', 'iou_med_shared', 'lost_n', 'lost_responsive',
        'new_n', 'new_accepted', 'new_responsive', 'new_reliability_med', 'new_partial_n', 'new_partial_responsive',
        'new_isolated_n', 'new_isolated_accepted', 'new_isolated_responsive', 'new_isolated_selective01',
        'new_isolated_reliability_med', 'new_isolated_npix_med', 'shared_ref_responsive', 'shared_cmp_responsive',
        'shared_cmp_reliability_med', 'tuning_r_med_ref_responsive', 'n_responsive_ref', 'n_responsive_cmp']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default=sr.PD)
    ap.add_argument('--refs', nargs='+', required=True, help='reference variant dirs / names')
    ap.add_argument('--variants', nargs='*', default=None,
                    help='comparison variants [default: all suite2p_* and sweep_suite2p/* of the session]')
    ap.add_argument('--iou', type=float, default=0.5, help='IoU threshold for a shared ROI')
    ap.add_argument('--reliability-splits', type=int, default=20)
    ap.add_argument('--outdir', default=sr.OUTDIR)
    a = ap.parse_args()
    session = a.session.rstrip('/')
    refs = [sr.resolve_variant(session, v) for v in a.refs]
    variants = [sr.resolve_variant(session, v) for v in a.variants] if a.variants else sr.discover_variants(session)
    slug = sessionio.session_slug(session) if hasattr(sessionio, 'session_slug') else os.path.basename(session)
    os.makedirs(a.outdir, exist_ok=True)
    out_csv = os.path.join(a.outdir, '%s_match_%s.csv' % (slug, sr.STAMP))
    cache, rows = {}, []

    def props(v):
        if v not in cache:
            t0 = time.time()
            cache[v] = variant_props(session, v, a.reliability_splits)
            print('   loaded %s: %d ROIs, %d responsive (%.0f s)' % (cache[v]['name'], len(cache[v]['npix']),
                                                                    int(cache[v]['responsive'].sum()), time.time() - t0))
            sys.stdout.flush()
        return cache[v]

    for ref in refs:
        for v in variants:
            if os.path.normpath(v) == os.path.normpath(ref):
                continue
            print('\n== %s  vs  %s' % (os.path.basename(ref), os.path.basename(v)))
            try:
                rows.append(compare(props(ref), props(v), a.iou))
                r = rows[-1]
                print('   ' + ' | '.join('%s=%s' % (k, ('%.3g' % r[k]) if isinstance(r[k], float) else r[k]) for k in SHOW if k in r))
                pd.DataFrame(rows).to_csv(out_csv, index=False)             # incremental
            except Exception as exc:
                traceback.print_exc()
                print('   FAILED: %s: %s' % (type(exc).__name__, str(exc)[:200]))
            sys.stdout.flush()
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 60, 'display.float_format', lambda x: '%.3g' % x):
        print('\n' + df[['ref', 'cmp'] + [c for c in SHOW if c in df.columns]].to_string(index=False))
    print('\nwrote %s' % out_csv)


if __name__ == '__main__':
    main()
