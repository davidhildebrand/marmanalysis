"""Tests for suite2p_sweep_match: IoU matching and the shared / lost / new bookkeeping on hand-built masks."""
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import suite2p_sweep_match as sm  # noqa: E402

LY = LX = 40


def make_props(name, boxes, responsive, response, accepted=None):
    """boxes: list of (y0, y1, x0, x1) pixel rectangles, one ROI each."""
    rows, cols = [], []
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rows.append(np.full(yy.size, i))
        cols.append(yy.ravel() * LX + xx.ravel())
    n = len(boxes)
    mask = sparse.csr_matrix((np.ones(sum(r.size for r in rows), np.float32),
                              (np.concatenate(rows), np.concatenate(cols))), shape=(n, LY * LX))
    responsive = np.asarray(responsive, bool)
    return {'name': name, 'mask': mask, 'npix': np.asarray(mask.sum(1)).ravel(),
            'accepted': np.ones(n, bool) if accepted is None else np.asarray(accepted, bool),
            'responsive': responsive, 'selective01': responsive.copy(), 'reliability': np.where(responsive, 0.8, 0.0),
            'response': np.asarray(response, float), 'condition_ids': np.arange(np.asarray(response).shape[1])}


def test_match_shared_lost_new():
    tune = np.array([[0., 1., 2., 3.], [3., 2., 1., 0.], [1., 0., 1., 0.]])
    # A: three ROIs. B: ROI 0 identical, ROI 1 shifted by one column (IoU 0.8), A's ROI 2 missing, plus one new ROI.
    A = make_props('A', [(0, 10, 0, 10), (20, 30, 0, 10), (0, 10, 25, 35)], [True, True, False], tune)
    B = make_props('B', [(0, 10, 0, 10), (20, 30, 1, 11), (25, 35, 25, 35)], [True, False, True],
                   np.array([[0., 2., 4., 6.], [0., 1., 2., 3.], [5., 5., 5., 6.]]))
    pairs, iou = sm.match_rois(A, B, thr=0.5)
    assert pairs.tolist() == [[0, 0], [1, 1]]
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[1, 1], 90 / 110)
    row = sm.compare(A, B, thr=0.5)
    assert row['n_ref'] == 3 and row['n_cmp'] == 3
    assert np.isclose(row['ref_recovered_frac'], 2 / 3)
    assert row['lost_n'] == 1 and row['new_n'] == 1
    assert row['lost_responsive_n'] == 0 and row['new_responsive_n'] == 1
    assert row['new_isolated_n'] == 1 and row['new_partial_n'] == 0      # the new ROI overlaps nothing in A
    assert row['lost_isolated_n'] == 1 and row['lost_partial_n'] == 0
    assert np.isclose(row['new_isolated_responsive'], 1.0)
    assert np.isclose(row['shared_ref_responsive'], 1.0) and np.isclose(row['shared_cmp_responsive'], 0.5)
    # pair 0: perfectly correlated tuning; pair 1: perfectly anti-correlated -> median over ref-responsive pairs is 0
    r = sm.pair_tuning_r(A, B, pairs)
    assert np.allclose(r, [1.0, -1.0])
    assert np.isclose(row['tuning_r_med_ref_responsive'], 0.0)
    assert np.isclose(row['responsive_agree_frac'], 0.5)


def test_iou_threshold_and_mutual_best():
    # One big A ROI split into two B ROIs: each half has IoU 0.5 with A; only one can be A's mutual best match.
    A = make_props('A', [(0, 10, 0, 20)], [True], np.zeros((1, 3)))
    B = make_props('B', [(0, 10, 0, 10), (0, 10, 10, 20)], [True, True], np.zeros((2, 3)))
    pairs, iou = sm.match_rois(A, B, thr=0.5)
    assert np.allclose(iou, [[0.5, 0.5]])
    assert len(pairs) == 1                       # mutual best, IoU exactly at threshold
    assert len(sm.match_rois(A, B, thr=0.6)[0]) == 0
    row = sm.compare(A, B, thr=0.6)
    assert row['lost_n'] == 1 and row['new_n'] == 2 and np.isclose(row['ref_partial_frac'], 1.0)
    assert row['new_partial_n'] == 2 and row['new_isolated_n'] == 0      # both halves are fragments of A's ROI
    assert row['lost_partial_n'] == 1 and row['lost_isolated_n'] == 0


def test_empty_overlap():
    A = make_props('A', [(0, 5, 0, 5)], [False], np.zeros((1, 3)))
    B = make_props('B', [(20, 25, 20, 25)], [False], np.zeros((1, 3)))
    row = sm.compare(A, B)
    assert row['ref_recovered_frac'] == 0 and row['lost_n'] == 1 and row['new_n'] == 1
    assert np.isclose(row['ref_no_overlap_frac'], 1.0)
