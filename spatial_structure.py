#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-SOM spatial-structure tests (roadmap #12; DH 2026-09-24/27, 2026-10-18): is there ANY structure beyond a smooth
gradient in a map -- column-like clusters, patches, or segregation between preferences -- without assuming
periodicity?

Maps (one scalar per ROI, z-scored across ROIs): the neural PC0 score, and face / body / object preference (d' of the
category vs the rest, blanks excluded). PC0 is called the FACE AXIS only where its condition loading tracks the face
indicator (|r| >= FACE_MIN_R; PD 0.90). In the 3-mm Curly fields it does not (0.11-0.16), so there the map is
labeled 'neural PC0' and the figure's face panels (red clusters on the mean image, Gi*) use the face d' map.
Statistics:
  1. MARK-CORRELATION FUNCTION k(d) = mean(z_i z_j | d_ij in bin): the tuning correlation of two cells as a function
     of their distance; T = k(near) - k(far), and for standardized marks k(d) = 1 - gamma(d) (the semivariogram).
     Monotone decay = gradient; plateau then drop = patch of that size; a dip below the null = hole / periodicity.
  2. LOCAL HOTSPOTS (Getis-Ord Gi*): per cell the standardized sum of z over neighbors within R_GI (self included);
     a hotspot is |Gi*| > 2.58. CLUSTERS = connected (within R_GI), same-sign groups of hotspot cells; every cluster
     of >= MIN_CLUSTER cells is listed and numbered, and the NUMBER of clusters is tested against the nulls (how many
     column-like clusters, and are there more than a generic tuning direction would produce?).
  3. SCAN STATISTIC: over circular windows centered on every cell with radii RADII, the standardized difference of
     the mean z inside vs outside; the maximum over windows is the scan statistic (the single most anomalous patch).
  4. RIPLEY-K PAIR COUNTS of the top-TOPQ cells of a map at radii KR vs a RANDOM-LABELING null.
  5. BIVARIATE CROSS-K between the top cells of two maps (face-body, face-object, body-object): attraction (ratio > 1)
     vs segregation (ratio < 1).
Nulls for 1-3 (n_perm maps each): (a) POSITION SHUFFLE; (b) RANDOM PROJECTION in condition space (the generic
smoothness ANY tuning direction inherits); (c) RANDOM FEATURE DIRECTION in relu7-PC space (Chang & Tsao 2017 STA-style).

Sessions are named by their slug (sessionio.session_slug, AnimalDateTime, e.g. Curly20240130d172813t; aliases 'pd'
and 'curly'), and every figure filename and title carries it (filenames also carry the gate and a non-default
--rmax-um). Multimodal sessions are restricted to their image
conditions. Geometry is SPARSE: only neighbor pairs within RMAX = 300 um (the largest distance any LOCAL statistic
uses; ``--rmax-um`` keeps more pairs to extend the mark-correlation function to long range), so a whole
3-mm field (~15k ROIs) is analyzed at once at full ROI density -- no tiles, no clusters cut at tile edges. Large
sessions are LOADED in spatial chunks (``chunk_um``) only to bound memory; the analysis runs on the whole field.
``--tile-um`` still gives separate per-tile analyses. The figure shows the suite2p mean image next to the maps
(all spatial panels in image orientation: y increases downward), the numbered face and object clusters and the
best scan window.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/spatial_structure.py --session pd
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/spatial_structure.py --session curly --n-perm 500
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import feature_specificity as fs
import feature_topography_v2 as v2
import neuropil_checks as nc
import response_matrix as rm
import response_table as rt
import sessionio
import som
import topography as tg

BASE_BINS = [0, 25, 50, 75, 100, 150, 200, 300]
LONG_BINS = [400, 600, 800, 1000, 1500, 2000, 3000]
BINS = np.array(BASE_BINS, float)
RMAX = float(BINS[-1])        # every LOCAL statistic needs <= 200 um; --rmax-um extends only the mark-correlation range
RADII = (40.0, 60.0, 80.0, 100.0)
R_GI = 50.0
GI_THR = 2.58
MIN_CLUSTER = 5
FACE_MIN_R = 0.5          # neural PC0 is called the face axis only if |corr(its condition loading, face indicator)| >= this
TOPQ = 0.2
KR = np.array([25, 50, 75, 100, 150, 200], float)
DENSE_MAX = 6000          # keep a dense distance matrix for fields this small (cluster_profile.py uses G.dist)
OUTDIR = 'output'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')   # project convention: every output file is UTC-stamped

# Sessions are keyed by their slug (sessionio.session_slug: AnimalDateTime). Stimulus images are located from each
# session's own stimulus log (images.local_stimulus_path), not configured here. Multimodal sessions are restricted to
# their IMAGE conditions (and blanks) -- moving dots etc. carry no image category.
_LARGE = dict(zeta=None, kw=dict(equalize_repeats=True, eye_gate_mode='none'), gate='all', chunk_um=1500)
SESSIONS = {
    'Cadbury20221016d152643t': dict(path=nc.PD, zeta=nc.ZETA_CSV, kw={}, gate='zeta', chunk_um=0),   # area PD, 730 um
    'Curly20231103d162355t': dict(glob=nc.CURLY_GLOB, **_LARGE),                                       # FOBmin, 3 mm
    'Curly20240130d172813t': dict(glob='suite2p_results/Curly/20240130d/172813tUTC*', **_LARGE),         # multimodal, 3 mm
}
ALIASES = {'pd': 'Cadbury20221016d152643t', 'curly': 'Curly20231103d162355t'}
MAP_NAMES = ('face axis (nPC0)', "face d'", "body d'", "object d'")
INTACT = ('face', 'body', 'object', 'animal')   # intact image categories; category d' contrasts within these


def zs(v):
    v = np.nan_to_num(np.asarray(v, float))
    return (v - v.mean()) / (v.std() + 1e-12)


def bins_for(rmax_um):
    """Mark-correlation distance bins: the local bins to 300 um plus long-range bins up to ``rmax_um``."""
    return np.array(BASE_BINS + [b for b in LONG_BINS if b <= rmax_um], float)


class Geometry:
    """Neighbor structure of a field of cells, sparse: pairs within ``rmax_um`` (default RMAX = 300 um, which covers
    every local statistic; a larger value only adds long-range mark-correlation bins)."""

    def __init__(self, xy, rmax_um=RMAX):
        self.xy = np.asarray(xy, float)
        self.n = len(self.xy)
        self.bins = bins_for(max(float(rmax_um), RMAX))
        pairs = cKDTree(self.xy).query_pairs(r=float(self.bins[-1]), output_type='ndarray').astype(np.int32)
        self.pi, self.pj = pairs[:, 0], pairs[:, 1]
        del pairs
        self.pd = np.hypot(self.xy[self.pi, 0] - self.xy[self.pj, 0], self.xy[self.pi, 1] - self.xy[self.pj, 1])
        self.nb = len(self.bins) - 1
        b = np.digitize(self.pd, self.bins) - 1
        self.mc_ok = (b >= 0) & (b < self.nb)
        self.mc_bin = b[self.mc_ok]
        self.mc_count = np.bincount(self.mc_bin, minlength=self.nb).astype(float)
        loc = self.pd <= KR.max()                       # Ripley / cross-K only need pairs within 200 um
        self.li, self.lj, self.ld = self.pi[loc], self.pj[loc], self.pd[loc]
        self.M = {R: self._adjacency(R) for R in RADII}
        self.n_in = {R: np.asarray(self.M[R].sum(1)).ravel() for R in RADII}
        self.W_gi = self._adjacency(R_GI)
        self.n_gi = np.asarray(self.W_gi.sum(1)).ravel()
        self.area = float(ConvexHull(self.xy).volume) if self.n >= 4 else np.nan
        self.dist = tg.pairwise_distance(self.xy) if self.n <= DENSE_MAX else None

    def _adjacency(self, R):
        m = self.ld <= R
        i, j = self.li[m], self.lj[m]
        A = coo_matrix((np.ones(2 * i.size), (np.r_[i, j], np.r_[j, i])), shape=(self.n, self.n)).tocsr()
        return (A + identity(self.n, format='csr')).tocsr()

    def mark_corr(self, z):
        s = np.bincount(self.mc_bin, weights=(z[self.pi] * z[self.pj])[self.mc_ok], minlength=self.nb)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.mc_count > 0, s / self.mc_count, np.nan)

    def gistar(self, z):
        return (self.W_gi @ z) / np.sqrt(self.n_gi * (self.n - self.n_gi) / (self.n - 1.0))

    def clusters(self, g, sign=1, min_size=1):
        """Connected (within R_GI) groups of same-sign hotspot cells (|Gi*| > GI_THR), largest first."""
        idx = np.flatnonzero((np.abs(g) > GI_THR) & (np.sign(g) == sign))
        if idx.size == 0:
            return []
        _, lab = connected_components(self.W_gi[idx][:, idx], directed=False)
        groups = [idx[lab == k] for k in np.unique(lab)]
        return sorted([c for c in groups if c.size >= min_size], key=lambda c: -c.size)

    def largest_cluster(self, g, sign=1):
        c = self.clusters(g, sign)
        return c[0] if c else np.array([], int)

    def hotspots(self, z):
        g = self.gistar(z)
        best = max(len(self.largest_cluster(g, 1)), len(self.largest_cluster(g, -1)))
        return int((np.abs(g) > GI_THR).sum()), int(best), g

    def cluster_counts(self, g):
        return len(self.clusters(g, 1, MIN_CLUSTER)), len(self.clusters(g, -1, MIN_CLUSTER))

    def scan(self, z):
        tot = z.sum()
        best = (-np.inf, -1, np.nan)
        for R in RADII:
            s_in = self.M[R] @ z
            n_in = self.n_in[R]
            n_out = self.n - n_in
            ok = (n_in >= 5) & (n_out >= 5)
            stat = np.full(self.n, -np.inf)
            stat[ok] = ((s_in[ok] / n_in[ok]) - (tot - s_in[ok]) / n_out[ok]) / np.sqrt(1.0 / n_in[ok] + 1.0 / n_out[ok])
            i = int(np.argmax(stat))
            if stat[i] > best[0]:
                best = (float(stat[i]), i, R)
        return best

    def pair_counts(self, idx):
        m = np.zeros(self.n, bool)
        m[idx] = True
        d = self.ld[m[self.li] & m[self.lj]]
        return np.array([(d <= r).sum() for r in KR], float)

    def cross_counts(self, ia, ib):
        A = np.zeros(self.n, bool); A[ia] = True
        B = np.zeros(self.n, bool); B[ib] = True
        d = self.ld[(A[self.li] & B[self.lj]) | (B[self.li] & A[self.lj])]
        return np.array([(d <= r).sum() for r in KR], float)


def null_stats(G, maps):
    out = {k: [] for k in ('mark_corr', 'n_hot', 'largest', 'scan', 'n_pos', 'n_neg')}
    for z in maps:
        z = zs(z)
        out['mark_corr'].append(G.mark_corr(z))
        h, c, g = G.hotspots(z)
        npos, nneg = G.cluster_counts(g)
        out['n_hot'].append(h); out['largest'].append(c); out['n_pos'].append(npos); out['n_neg'].append(nneg)
        out['scan'].append(G.scan(z)[0])
    return {k: np.array(v) for k, v in out.items()}


def pval(null, obs):
    null = np.asarray(null, float)
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def category_dprime(resp, ck, key):
    """d' of category ``key`` vs the REST of the intact image categories (INTACT: face, body, object, animal); blanks,
    scrambles and non-image conditions are excluded from both sides."""
    m = ck == key
    rest = np.isin(ck, INTACT) & ~m
    a, b = resp[:, m], resp[:, rest]
    return (np.nanmean(a, 1) - np.nanmean(b, 1)) / np.sqrt(0.5 * (np.nanvar(a, 1) + np.nanvar(b, 1)) + 1e-12)


def cat_key(c):
    c = str(c).lower()
    if 'blank' in c:
        return 'blank'
    if c in ('none', 'nan', '', "b''"):
        return 'other'                                   # a non-image condition (e.g. moving dots in multimodal sessions)
    if 'scram' in c:
        return 'scramble'
    if 'face' in c:
        return 'face'
    if 'body' in c:
        return 'body'
    if 'animal' in c:
        return 'animal'
    return 'object'                                      # obj, food, ...


def face_axis_map(resp, ck):
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    m = U[:, 0] * S[0]
    return m * np.sign(np.corrcoef(m, category_dprime(resp, ck, 'face'))[0, 1])


def pc0_face_r(resp, ck):
    """|corr| between the neural PC0 condition loading and the face indicator, and PC0's share of the variance:
    is PC0 the face axis in this session? (PD 0.90 yes; Curly 3-mm fields 0.11-0.16 no.)"""
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    return float(abs(np.corrcoef(Vt[0], (ck == 'face').astype(float))[0, 1])), float(S[0] ** 2 / (S ** 2).sum())


def resolve_session(key):
    """Session config by slug or alias; adds 'path' and 'slug'."""
    slug = ALIASES.get(key, key)
    cfg = dict(SESSIONS[slug])
    if 'path' not in cfg:
        hits = [h for h in sorted(glob.glob(cfg['glob'])) if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
        cfg['path'] = hits[0]
    cfg['slug'] = sessionio.session_slug(cfg['path'])
    return cfg


def feature_pcs(data, cond_mask=None):
    """relu7 PC scores of the presented images (for the random-feature null), plus the mask of conditions used: every
    presented condition, blanks included, whose image resolves to a local file (images.build_condition_metadata;
    a blank without its own file uses the set's common blank image). ``cond_mask`` restricts to the analyzed
    conditions (e.g. a multimodal session's image conditions). None if fewer than 10 resolve."""
    paths = som.condition_image_paths(data)
    if cond_mask is not None:
        paths = [p for p, k in zip(paths, cond_mask) if k]
    keep = np.array([p is not None and os.path.exists(p) for p in paths])
    folders = sorted({os.path.dirname(p) for p in paths if p is not None})
    if keep.sum() < 10:
        print('  (only %d/%d stimulus images resolved locally in %s -> random-feature null skipped)'
              % (keep.sum(), keep.size, folders))
        return None
    print('  random-feature null: %d/%d presented conditions have a local image in %s' % (keep.sum(), keep.size, folders))
    if not keep.all():
        print('  no local image for: %s' % [p for p, k in zip(paths, keep) if not k])
    feats = dnn_som.extract_dnn_features(*dnn_som.prep_dnn_model('cpu'),
                                         dnn_som.load_images_rgb([p for p, k in zip(paths, keep) if k]), 'cpu').numpy()
    return v2.pcs(feats)[0], keep


def mean_image(path):
    """The suite2p mean image and its um-per-pixel (x, y), for display under the maps."""
    s2p = sessionio.load_suite2p(path)
    res = np.asarray(sessionio.load_metadata(path)['fov']['resolution_umpx'], float).ravel()
    return np.asarray(s2p['ops']['meanImg'], float), float(res[0]), float(res[-1])


def _frame(ax, xy, pad=20.0):
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)          # image orientation: y increases downward
    ax.set_aspect('equal')


def _draw_clusters(ax, xy, groups, color, marker='o', label=None):
    for k, c in enumerate(groups, 1):
        ax.scatter(xy[c, 0], xy[c, 1], marker=marker, facecolors='none', edgecolors=color, s=45, linewidths=1.1,
                   label=label if k == 1 else None)
        cx, cy = xy[c].mean(0)
        ax.annotate(str(k), (cx, cy), color=color, fontsize=9, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.8))


def cross_k(G, ia, ib, n_perm, rng):
    both = np.intersect1d(ia, ib)
    ia, ib = np.setdiff1d(ia, both), np.setdiff1d(ib, both)
    cc = G.cross_counts(ia, ib)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(G.n)
        null.append(G.cross_counts(p[:len(ia)], p[len(ia):len(ia) + len(ib)]))
    null = np.array(null)
    p_lo = np.array([(np.sum(null[:, k] <= cc[k]) + 1) / (n_perm + 1) for k in range(len(KR))])
    p_hi = np.array([(np.sum(null[:, k] >= cc[k]) + 1) / (n_perm + 1) for k in range(len(KR))])
    return cc / null.mean(0), p_lo, p_hi, np.percentile(null, [5, 95], axis=0) / null.mean(0), len(ia), len(ib)


def analyse(resp, xy, ck, label, n_perm, Fpc, rng, fig_path, bg=None, rmax_um=RMAX):
    n = resp.shape[0]
    G = Geometry(xy, rmax_um)
    print('\n##### %s: %d cells, hull area %.2f mm2, density %.0f / mm2, %d pairs within %g um'
          % (label, n, G.area / 1e6, n / (G.area / 1e6), G.pd.size, G.bins[-1]))
    face_r, pc0_var = pc0_face_r(resp, ck)
    k0 = 'face axis (nPC0)' if face_r >= FACE_MIN_R else 'neural PC0'
    kf = k0 if face_r >= FACE_MIN_R else "face d'"          # the map behind the figure's face panels
    print('  neural PC0 (%.1f%% of the variance): |corr| of its condition loading with the face indicator %.2f%s'
          % (100 * pc0_var, face_r, '' if face_r >= FACE_MIN_R else
             " -> NOT a face axis here: its map is labeled 'neural PC0' and the face panels use the face d' map"))
    maps = {k0: face_axis_map(resp, ck)}
    for k in ('face', 'body', 'object'):
        maps["%s d'" % k] = category_dprime(resp, ck, k)

    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Uc = rng.standard_normal((resp.shape[1], n_perm)); Uc /= np.linalg.norm(Uc, axis=0, keepdims=True)
    NRP = null_stats(G, (R @ Uc).T)
    if Fpc is not None:
        F, keep = Fpc
        Uf = rng.standard_normal((F.shape[1], n_perm)); Uf /= np.linalg.norm(Uf, axis=0, keepdims=True)
        NRF = null_stats(G, np.column_stack([v2.rowcorr(resp[:, keep], F @ Uf[:, k]) for k in range(n_perm)]).T)
    else:
        NRF = NRP
    results = {}
    print('=== mark-correlation k(d) [bins %s um] ===' % ' '.join('%g' % b for b in G.bins))
    print('  null p95 per bin: random projection %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRP['mark_corr'], 95, axis=0)))
    print('                    random feature    %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRF['mark_corr'], 95, axis=0)))
    for nm, m in maps.items():
        z = zs(m)
        mc = G.mark_corr(z)
        nh, lc, g = G.hotspots(z)
        s, ci, sr = G.scan(z)
        shuf = null_stats(G, np.array([z[rng.permutation(n)] for _ in range(n_perm)]))
        results[nm] = dict(z=z, mc=mc, n_hot=nh, largest=lc, g=g, scan=s, scan_i=ci, scan_r=sr, shuf=shuf,
                           pos=G.clusters(g, 1, MIN_CLUSTER), neg=G.clusters(g, -1, MIN_CLUSTER))
        print('  %-18s k(d): %s' % (nm, ' '.join('%+.3f' % v for v in mc)))
    print('=== hotspots (Gi*, R=%g um, |Gi*|>%.2f) and scan statistic (radii %s um) ===' % (R_GI, GI_THR, RADII))
    print('  %-18s %7s %7s | %6s %5s %5s | %6s %7s %7s | null medians n_hot/largest/scan (shuf | rproj | rfeat)'
          % ('map', 'n_hot', 'largest', 'scan', 'n_in', 'R', 'p_shuf', 'p_rproj', 'p_rfeat'))
    for nm, r in results.items():
        n_in = int(G.n_in[r['scan_r']][r['scan_i']]) if r['scan_i'] >= 0 else 0
        print('  %-18s %7d %7d | %6.2f %5d %5g | %6.3f %7.3f %7.3f | %d/%d/%.2f | %d/%d/%.2f | %d/%d/%.2f'
              % (nm, r['n_hot'], r['largest'], r['scan'], n_in, r['scan_r'],
                 pval(r['shuf']['scan'], r['scan']), pval(NRP['scan'], r['scan']), pval(NRF['scan'], r['scan']),
                 np.median(r['shuf']['n_hot']), np.median(r['shuf']['largest']), np.median(r['shuf']['scan']),
                 np.median(NRP['n_hot']), np.median(NRP['largest']), np.median(NRP['scan']),
                 np.median(NRF['n_hot']), np.median(NRF['largest']), np.median(NRF['scan'])))
        print('  %-18s hotspot-count p: shuf %.3f rproj %.3f rfeat %.3f | largest-cluster p: shuf %.3f rproj %.3f rfeat %.3f'
              % ('', pval(r['shuf']['n_hot'], r['n_hot']), pval(NRP['n_hot'], r['n_hot']), pval(NRF['n_hot'], r['n_hot']),
                 pval(r['shuf']['largest'], r['largest']), pval(NRP['largest'], r['largest']), pval(NRF['largest'], r['largest'])))
        if r['scan_i'] >= 0:
            print('  %-18s best scan window centered at (%.0f, %.0f) um' % ('', xy[r['scan_i'], 0], xy[r['scan_i'], 1]))
    print('=== CLUSTERS: connected same-sign hotspot groups of >= %d cells; count vs the nulls (p = fraction of null maps with as many) ===' % MIN_CLUSTER)
    for nm, r in results.items():
        for sign, key, lab in ((1, 'pos', 'preferring'), (-1, 'neg', 'anti')):
            groups = r[key]
            nk = 'n_pos' if sign == 1 else 'n_neg'
            print('  %-18s %-10s %2d clusters | null median shuf %.0f rproj %.0f rfeat %.0f | p shuf %.3f rproj %.3f rfeat %.3f'
                  % (nm, lab, len(groups), np.median(r['shuf'][nk]), np.median(NRP[nk]), np.median(NRF[nk]),
                     pval(r['shuf'][nk], len(groups)), pval(NRP[nk], len(groups)), pval(NRF[nk], len(groups))))
            if groups:
                print('  %-29s %s' % ('', '; '.join('#%d: %d cells at (%.0f, %.0f) um, mean z %+.2f'
                                                    % (k, c.size, xy[c, 0].mean(), xy[c, 1].mean(), r['z'][c].mean())
                                                    for k, c in enumerate(groups[:12], 1))))

    m_top = int(round(TOPQ * n))
    print('=== Ripley pair-count ratio of the top-%d%% cells vs random labeling (radii %s um): ratio (p) ===' % (100 * TOPQ, ' '.join('%g' % r for r in KR)))
    null_pc = np.array([G.pair_counts(rng.choice(n, m_top, replace=False)) for _ in range(n_perm)])
    tops = {}
    for nm in results:
        idx = np.argsort(-results[nm]['z'])[:m_top]
        tops[nm] = idx
        pc = G.pair_counts(idx)
        ratio = pc / null_pc.mean(0)
        print('  %-18s %s' % (nm, '  '.join('%.2f (%.3f)' % (ratio[k], pval(null_pc[:, k], pc[k])) for k in range(len(KR)))))
        results[nm]['kratio'] = ratio; results[nm]['kband'] = np.percentile(null_pc, [5, 95], axis=0) / null_pc.mean(0)
    xk = {}
    print('=== cross-K between the top-%d%% cells of two maps: ratio to random labeling; <1 segregation (p_low), >1 attraction (p_high) ===' % (100 * TOPQ))
    for a_, b_ in (("face d'", "body d'"), ("face d'", "object d'"), ("body d'", "object d'")):
        ratio, p_lo, p_hi, band, na, nb_ = cross_k(G, tops[a_], tops[b_], n_perm, rng)
        xk[(a_, b_)] = (ratio, band)
        print('  %s vs %s (n=%d / %d): %s' % (a_.split()[0], b_.split()[0], na, nb_,
              '  '.join('r<=%g: %.2f (%.3f/%.3f)' % (KR[k], ratio[k], p_lo[k], p_hi[k]) for k in range(len(KR)))))

    # ---- figure ----
    fig, ax = plt.subplots(3, 3, figsize=(19, 18))
    r0, rf, ro = results[k0], results[kf], results["object d'"]
    if bg is not None:
        img, rx, ry = bg
        lo, hi = np.nanpercentile(img, [1, 99.5])
        ax[0, 0].imshow(img, cmap='gray', vmin=lo, vmax=hi, extent=(0, img.shape[1] * rx, img.shape[0] * ry, 0))
        _draw_clusters(ax[0, 0], xy, rf['pos'], 'red', label='%s clusters' % kf)
        _draw_clusters(ax[0, 0], xy, ro['pos'], 'cyan', marker='s', label='object-preferring clusters')
        ax[0, 0].legend(fontsize=7, loc='lower left')
        ax[0, 0].set_title('suite2p mean image with the numbered clusters', fontsize=9)
    _frame(ax[0, 0], xy)
    sc = ax[0, 1].scatter(xy[:, 0], xy[:, 1], c=r0['z'], cmap='coolwarm', vmin=-2, vmax=2, s=6 if n > 3000 else 14, rasterized=True)
    _draw_clusters(ax[0, 1], xy, r0['pos'], 'darkred', label='%s clusters (>= %d cells)' % (k0, MIN_CLUSTER))
    if r0['scan_i'] >= 0:
        ax[0, 1].add_patch(Circle(xy[r0['scan_i']], r0['scan_r'], fill=False, ec='k', lw=2, ls='--',
                                  label='best scan window (stat %.2f, R=%g)' % (r0['scan'], r0['scan_r'])))
    ax[0, 1].legend(fontsize=7, loc='lower left'); _frame(ax[0, 1], xy)
    ax[0, 1].set_title('%s z per cell (|r| of PC0 with the face indicator %.2f): %d clusters' % (k0, face_r, len(r0['pos'])), fontsize=9)
    plt.colorbar(sc, ax=ax[0, 1], fraction=0.04)
    sc2 = ax[0, 2].scatter(xy[:, 0], xy[:, 1], c=ro['z'], cmap='coolwarm', vmin=-2, vmax=2, s=6 if n > 3000 else 14, rasterized=True)
    _draw_clusters(ax[0, 2], xy, ro['pos'], 'navy', marker='s', label='object-preferring clusters')
    _draw_clusters(ax[0, 2], xy, ro['neg'], 'darkorange', marker='D', label='anti-object clusters')
    ax[0, 2].legend(fontsize=7, loc='lower left'); _frame(ax[0, 2], xy)
    ax[0, 2].set_title("object d' z per cell: %d object-preferring, %d anti-object clusters" % (len(ro['pos']), len(ro['neg'])), fontsize=9)
    plt.colorbar(sc2, ax=ax[0, 2], fraction=0.04)
    ax[1, 0].scatter(xy[:, 0], xy[:, 1], c=rf['g'], cmap='coolwarm', vmin=-4, vmax=4, s=6 if n > 3000 else 14, rasterized=True)
    if rf['scan_i'] >= 0:
        ax[1, 0].add_patch(Circle(xy[rf['scan_i']], rf['scan_r'], fill=False, ec='k', lw=2, ls='--',
                                  label='best %s scan window (stat %.2f, R=%g)' % (kf, rf['scan'], rf['scan_r'])))
        ax[1, 0].legend(fontsize=7, loc='lower left')
    _frame(ax[1, 0], xy); ax[1, 0].set_title('Getis-Ord Gi* (R=%g um) of the %s map' % (R_GI, kf), fontsize=9)
    ctr = 0.5 * (G.bins[1:] + G.bins[:-1])
    if G.bins[-1] > RMAX:
        ax[1, 1].set_xscale('log')
    for nm, col in ((k0, 'tab:red'), ("face d'", 'tab:orange'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[1, 1].plot(ctr, results[nm]['mc'], '-o', color=col, label=nm)
    for N_, col, lab in ((NRP, '0.3', 'random projection p5-p95'), (NRF, '0.6', 'random feature p5-p95')):
        lo_, hi_ = np.nanpercentile(N_['mark_corr'], [5, 95], axis=0)
        ax[1, 1].fill_between(ctr, lo_, hi_, color=col, alpha=0.3, label=lab)
    ax[1, 1].axhline(0, color='k', lw=0.5); ax[1, 1].set_xlabel('pair distance (um)'); ax[1, 1].set_ylabel('k(d) = mean z_i z_j')
    ax[1, 1].legend(fontsize=7); ax[1, 1].set_title('mark-correlation function', fontsize=9)
    for nm, col in ((k0, 'tab:red'), ("face d'", 'tab:orange'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[1, 2].plot(KR, results[nm]['kratio'], '-o', color=col, label=nm)
    band = results["face d'"]['kband']
    ax[1, 2].fill_between(KR, band[0], band[1], color='0.5', alpha=0.3, label='random-labeling p5-p95')
    ax[1, 2].axhline(1, color='k', lw=0.5); ax[1, 2].set_xlabel('r (um)'); ax[1, 2].set_ylabel('pair count / null mean')
    ax[1, 2].legend(fontsize=7); ax[1, 2].set_title('Ripley pair-count ratio, top-%d%% cells of each map' % (100 * TOPQ), fontsize=9)
    for (a_, b_), col in zip(xk, ('purple', 'tab:brown', 'teal')):
        ratio, band_ = xk[(a_, b_)]
        ax[2, 0].plot(KR, ratio, '-o', color=col, label='%s vs %s' % (a_.split()[0], b_.split()[0]))
        ax[2, 0].fill_between(KR, band_[0], band_[1], color=col, alpha=0.08)
    ax[2, 0].axhline(1, color='k', lw=0.5); ax[2, 0].set_xlabel('r (um)'); ax[2, 0].set_ylabel('cross pair count / null mean')
    ax[2, 0].legend(fontsize=7); ax[2, 0].set_title('cross-K: <1 = segregated, >1 = intermixed / attracted (bands: random labeling p5-p95)', fontsize=9)
    ax[2, 1].hist(NRP['scan'], bins=30, alpha=0.6, color='0.3', label='random projection')
    ax[2, 1].hist(NRF['scan'], bins=30, alpha=0.6, color='0.7', label='random feature')
    ax[2, 1].hist(rf['shuf']['scan'], bins=30, alpha=0.6, color='tab:green', label='position shuffle')
    for nm, col in ((k0, 'tab:red'), ("face d'", 'tab:orange'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[2, 1].axvline(results[nm]['scan'], color=col, ls='--', lw=1.5, label='%s = %.2f' % (nm, results[nm]['scan']))
    ax[2, 1].set_xlabel('scan statistic (max over windows)'); ax[2, 1].legend(fontsize=7); ax[2, 1].set_title('scan statistic vs the three nulls', fontsize=9)
    edges = np.arange(-0.5, max([NRP['n_pos'].max()] + [len(results[nm]['pos']) for nm in results]) + 1.5)
    ax[2, 2].hist(NRP['n_pos'], bins=edges, alpha=0.5, color='0.3', label='random projection (any direction)')
    ax[2, 2].hist(NRF['n_pos'], bins=edges, alpha=0.5, color='0.7', label='random feature')
    for nm, col in ((k0, 'tab:red'), ("face d'", 'tab:orange'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[2, 2].axvline(len(results[nm]['pos']), color=col, ls='--', lw=1.5, label='%s: %d' % (nm, len(results[nm]['pos'])))
    ax[2, 2].set_xlabel('number of positive clusters (>= %d cells)' % MIN_CLUSTER); ax[2, 2].legend(fontsize=7)
    ax[2, 2].set_title('how many clusters vs a generic tuning direction', fontsize=9)
    fig.suptitle('%s: hotspot clusters, scan window, mark-correlation, Ripley K, cross-K (spatial panels in image orientation)' % label, fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print('figure: %s' % fig_path)
    sys.stdout.flush()
    return results


def gate_mask(data, gate, cfg):
    n_all = data['response'].shape[0]
    if gate == 'zeta' and cfg.get('zeta') and os.path.exists(cfg['zeta']):
        return pd.read_csv(cfg['zeta'])['p_zeta'].to_numpy() < 0.05
    if gate == 'responsive':
        return rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)['responsive']
    return np.ones(n_all, bool)


def spatial_chunks(path, chunk_um, min_rois=1):
    md = sessionio.load_metadata(path)
    s2p = sessionio.load_suite2p(path)
    xy_all = rm.calculate_roi_centroids_um(s2p['ROIs'], md['fov']['resolution_umpx'])
    x0, y0 = xy_all.min(0)
    nx = int(np.ceil((xy_all[:, 0].max() - x0 + 1e-6) / chunk_um)); ny = int(np.ceil((xy_all[:, 1].max() - y0 + 1e-6) / chunk_um))
    out = []
    for iy in range(ny):
        for ix in range(nx):
            sel = np.flatnonzero((xy_all[:, 0] >= x0 + ix * chunk_um) & (xy_all[:, 0] < x0 + (ix + 1) * chunk_um + (ix == nx - 1)) &
                                 (xy_all[:, 1] >= y0 + iy * chunk_um) & (xy_all[:, 1] < y0 + (iy + 1) * chunk_um + (iy == ny - 1)))
            if sel.size >= min_rois:
                out.append(('tile%d%d' % (iy, ix), sel))
    return out, len(xy_all)


def load_field(cfg, gate, chunk_um):
    """Whole-field response matrix: built in spatial chunks (to bound memory) and concatenated."""
    kw = dict(denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7, **cfg['kw'])
    if not chunk_um:
        data = rm.build_response_matrix(cfg['path'], **kw)
        mask = gate_mask(data, gate, cfg)
        d = rm.apply_roi_mask(data, mask)
        return data, d['response'], d['roi_xy_um'], d['reliability'], mask, data['response'].shape[0]
    chunks, n_total = spatial_chunks(cfg['path'], chunk_um)
    first, parts = None, []
    for name, sel in chunks:
        data = rm.build_response_matrix(cfg['path'], roi_select=sel, **kw)
        if first is None:
            first = data
        elif not np.array_equal(np.asarray(data['condition_ids']), np.asarray(first['condition_ids'])):
            raise RuntimeError('condition order differs between chunks')
        mask = gate_mask(data, gate, cfg)
        d = rm.apply_roi_mask(data, mask)
        parts.append((d['response'], d['roi_xy_um'], d['reliability'], mask))
        print('  loaded %s: %d ROIs (%d pass gate %s)' % (name, sel.size, int(mask.sum()), gate)); sys.stdout.flush()
    resp = np.vstack([p[0] for p in parts]); xy = np.vstack([p[1] for p in parts])
    rel = np.concatenate([p[2] for p in parts]); mask = np.concatenate([p[3] for p in parts])
    return first, resp, xy, rel, mask, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default='pd', choices=sorted(SESSIONS) + sorted(ALIASES),
                    help='session slug (AnimalDateTime, sessionio.session_slug) or alias: %s' % ALIASES)
    ap.add_argument('--gate', default=None, choices=['zeta', 'responsive', 'all'])
    ap.add_argument('--tile-um', type=float, default=None, help='separate per-tile analyses with this tile edge (0 = whole field, default)')
    ap.add_argument('--min-tile-rois', type=int, default=300)
    ap.add_argument('--only-tile', default=None, help='with --tile-um: run a single tile by name (e.g. tile01)')
    ap.add_argument('--rmax-um', type=float, default=RMAX,
                    help='longest pair distance kept (default 300 um, enough for every local statistic); larger values '
                         'add long-range mark-correlation bins up to this distance (memory and time grow with it)')
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--no-sensitivity', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cfg = resolve_session(a.session)
    slug = cfg['slug']
    is_pd = slug == ALIASES['pd']
    gate = a.gate or cfg['gate']
    tile_um = cfg.get('tile_um', 0) if a.tile_um is None else a.tile_um
    tag = 'gate-%s%s' % (gate, '' if a.rmax_um == RMAX else '_rmax%gum' % a.rmax_um)   # analysis choices in the filename
    os.makedirs(OUTDIR, exist_ok=True)
    bg = mean_image(cfg['path'])
    print('session %s: %s' % (slug, cfg['path']))

    def image_conditions(data):
        ck_all = np.array([cat_key(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
        keep = ck_all != 'other'
        if not keep.all():
            print('  analyzing the %d image / blank conditions; %d non-image conditions (e.g. moving dots) left out'
                  % (keep.sum(), (~keep).sum()))
        return ck_all[keep], keep

    if tile_um and tile_um > 0:
        tiles, n_total = spatial_chunks(cfg['path'], tile_um, a.min_tile_rois)
        print('%s: %d ROIs total -> %d tiles of %g um: %s' % (slug, n_total, len(tiles), tile_um,
                                                              ', '.join('%s (%d)' % (t, s.size) for t, s in tiles)))
        if a.only_tile:
            tiles = [t for t in tiles if t[0] == a.only_tile]
        Fpc = None
        for tname, sel in tiles:
            data = rm.build_response_matrix(cfg['path'], denoise=False, reliability_splits=3, neuropil_subtract=True,
                                            neucoeff=0.7, roi_select=sel, **cfg['kw'])
            ck, keep = image_conditions(data)
            mask = gate_mask(data, gate, cfg)
            d = rm.apply_roi_mask(data, mask)
            Fpc = feature_pcs(data, keep) if Fpc is None else Fpc
            label = '%s %s (corrected 0.7, gate=%s, %d/%d cells)' % (slug, tname, gate, d['response'].shape[0], sel.size)
            analyse(d['response'][:, keep], d['roi_xy_um'], ck, label, a.n_perm, Fpc, rng,
                    os.path.join(OUTDIR, 'spatial_structure_%s_%s_%s_%s.png' % (slug, tname, tag, STAMP)), bg=bg, rmax_um=a.rmax_um)
        return

    data, resp, xy, rel, mask, n_total = load_field(cfg, gate, cfg.get('chunk_um', 0))
    ck, keep = image_conditions(data)
    resp = resp[:, keep]
    Fpc = feature_pcs(data, keep)
    label = '%s whole field (corrected 0.7, gate=%s, %d/%d cells)' % (slug, gate, resp.shape[0], n_total)
    if is_pd:
        cls = rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)
        sel_ = cls['selective'][mask]
        q = np.nanpercentile(rel, [25, 50, 75, 90])
        print('split-half tuning reliability (3 splits): gated quartiles %.3f / %.3f / %.3f, p90 %.3f, frac above 0.2 = %.2f | '
              'ANOVA-selective subset (n=%d) median %.3f, frac above 0.2 = %.2f'
              % (q[0], q[1], q[2], q[3], np.nanmean(rel > 0.2), sel_.sum(), np.nanmedian(rel[sel_]), np.nanmean(rel[sel_] > 0.2)))
        dist = tg.pairwise_distance(xy)
        iu = np.triu_indices(len(xy), 1)
        near, far = dist[iu] < v2.NEAR, dist[iu] > v2.FAR
        for nd in (500, 5000):
            T = fs.random_projection_T(resp, iu, near, far, n=nd, seed=1)
            print('random-projection T with %4d directions: median %+.4f  p95 %+.4f  p99 %+.4f' % (nd, np.median(T), np.percentile(T, 95), np.percentile(T, 99)))
    analyse(resp, xy, ck, label, a.n_perm, Fpc, rng, os.path.join(OUTDIR, 'spatial_structure_%s_%s_%s.png' % (slug, tag, STAMP)),
            bg=bg, rmax_um=a.rmax_um)

    if is_pd and not a.no_sensitivity:
        print('\n=== face-axis scan / hotspots at other neuropil coefficients (random-projection null per build) ===')
        for cf in (None, 1.0):
            dd_ = rm.apply_roi_mask(nc.build(cfg['path'], cf), mask)
            r_ = dd_['response']
            G_ = Geometry(dd_['roi_xy_um'])
            z_ = zs(face_axis_map(r_, ck))
            R_ = np.nan_to_num(r_ - np.nanmean(r_, 0, keepdims=True))
            Uc = rng.standard_normal((r_.shape[1], 300)); Uc /= np.linalg.norm(Uc, axis=0, keepdims=True)
            N_ = null_stats(G_, (R_ @ Uc).T)
            s_, i_, R_scan = G_.scan(z_)
            nh_, lc_, g_ = G_.hotspots(z_)
            npos_, _ = G_.cluster_counts(g_)
            print('  neucoeff %-4s scan %.2f (R=%g, p_rproj %.3f) | hotspots %d (p %.3f) | largest cluster %d (p %.3f) | face-axis clusters %d (p %.3f)'
                  % ('none' if cf is None else cf, s_, R_scan, pval(N_['scan'], s_), nh_, pval(N_['n_hot'], nh_), lc_,
                     pval(N_['largest'], lc_), npos_, pval(N_['n_pos'], npos_)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
