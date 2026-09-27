#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-SOM spatial-structure tests (roadmap #12; DH 2026-09-24/27): is there ANY structure beyond a smooth gradient in
a map -- a single column-like cluster, patches, or segregation between preferences -- without assuming periodicity?

Maps (one scalar per ROI, z-scored across ROIs): the face axis (neural PC0 score), and face / body / object
preference (d' of the category vs the rest). Statistics:
  1. MARK-CORRELATION FUNCTION k(d) = mean(z_i z_j | d_ij in bin): the near-minus-far T as a full function of
     distance (for standardised marks k(d) = 1 - gamma(d), the semivariogram, and T = k(near) - k(far)).
     Monotone decay = gradient; plateau then drop = patch of that size; a dip below the null = hole / periodicity.
  2. LOCAL HOTSPOTS (Getis-Ord Gi*): per cell the standardised sum of z over neighbours within R_GI (self included);
     a hotspot is |Gi*| > 2.58; we report how many cells are hotspots and the size of the largest connected (within
     R_GI, same-sign) hotspot cluster -- the "column-like cluster" candidate.
  3. SCAN STATISTIC: over circular windows centred on every cell with radii RADII, the standardised difference of
     the mean z inside vs outside; the maximum over windows is the scan statistic (the single most anomalous
     patch) and its p-value is the fraction of null maps whose own maximum over the same windows is as large.
  4. RIPLEY-K PAIR COUNTS of the top-TOPQ cells of a map at radii KR, vs a RANDOM-LABELLING null (the same number of
     cells drawn at random from the same positions): clustering of e.g. face-preferring cells at ANY scale.
  5. BIVARIATE CROSS-K between two subpopulations (face-top vs body-top): attraction (ratio > 1) vs segregation.
Nulls for 1-3 (N_PERM maps each): (a) POSITION SHUFFLE (any structure); (b) RANDOM PROJECTION in condition space
(the generic smoothness ANY tuning direction inherits); (c) RANDOM FEATURE DIRECTION in relu7-PC space (a random
but image-plausible feature -- the Chang & Tsao 2017 STA-style axis null).

Sessions: ``--session pd`` (730 um, ZETA gate) or ``--session curly`` (3 mm FOBmin field). A large field is
analysed at FULL ROI DENSITY tile by tile (``--tile-um``, default 1500 um for curly; ``roi_select`` in the loader),
because the local statistics need the real neighbour density (a 2000-ROI subsample of 14.6k ROIs would leave ~1
cell per 40-um window). Gate: ``--gate zeta|responsive|all`` (default zeta for pd, all for curly -- only ~10% of
Curly ROIs pass the ANOVA gate, too sparse for local windows; the maps of non-driven cells are noise around 0).
The figure highlights the best scan window, the Gi* hotspot cells, the largest connected face-axis hotspot
cluster and the largest connected OBJECT-preference cluster (DH 2026-09-27).
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/spatial_structure.py --session pd
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/spatial_structure.py --session curly --n-perm 500
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

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

BINS = np.array([0, 25, 50, 75, 100, 150, 200, 300], float)
RADII = (40.0, 60.0, 80.0, 100.0)
R_GI = 50.0
GI_THR = 2.58
TOPQ = 0.2
KR = np.array([25, 50, 75, 100, 150, 200], float)
OUTDIR = 'output'

SESSIONS = {
    'pd': dict(path=nc.PD, zeta=nc.ZETA_CSV, stim_dir=v2.STIM_DIR, kw={}, gate='zeta', tile_um=0),
    'curly': dict(glob=nc.CURLY_GLOB, zeta=None, stim_dir='stimuli/StimSpace/FOBmin230728d',
                  kw=dict(equalize_repeats=True, eye_gate_mode='none'), gate='all', tile_um=1500),
}


def zs(v):
    v = np.nan_to_num(np.asarray(v, float))
    return (v - v.mean()) / (v.std() + 1e-12)


class Geometry:
    def __init__(self, xy):
        self.xy = np.asarray(xy, float)
        self.n = len(self.xy)
        self.dist = tg.pairwise_distance(self.xy)
        self.iu = np.triu_indices(self.n, 1)
        self.dd = self.dist[self.iu]
        b = np.digitize(self.dd, BINS) - 1
        self.bin_masks = [b == k for k in range(len(BINS) - 1)]
        self.M = {R: (self.dist <= R).astype(np.float32) for R in RADII}
        self.n_in = {R: self.M[R].sum(1).astype(float) for R in RADII}
        self.W_gi = (self.dist <= R_GI)
        self.n_gi = self.W_gi.sum(1).astype(float)
        self.area = float(ConvexHull(self.xy).volume) if self.n >= 4 else np.nan

    def mark_corr(self, z):
        p = z[self.iu[0]] * z[self.iu[1]]
        return np.array([p[m].mean() if m.any() else np.nan for m in self.bin_masks])

    def gistar(self, z):
        return (self.W_gi @ z) / np.sqrt(self.n_gi * (self.n - self.n_gi) / (self.n - 1.0))

    def largest_cluster(self, g, sign=1):
        """Members of the largest connected (within R_GI) component of same-sign hotspot cells."""
        idx = np.flatnonzero((np.abs(g) > GI_THR) & (np.sign(g) == sign))
        if idx.size == 0:
            return idx
        _, lab = connected_components(csr_matrix(self.W_gi[np.ix_(idx, idx)]), directed=False)
        return idx[lab == np.argmax(np.bincount(lab))]

    def hotspots(self, z):
        g = self.gistar(z)
        hot = np.abs(g) > GI_THR
        best = max(len(self.largest_cluster(g, 1)), len(self.largest_cluster(g, -1)))
        return int(hot.sum()), int(best), g

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
        d = self.dist[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        return np.array([(d <= r).sum() for r in KR], float)

    def cross_counts(self, ia, ib):
        d = self.dist[np.ix_(ia, ib)].ravel()
        return np.array([(d <= r).sum() for r in KR], float)


def null_stats(G, maps):
    mc, nh, lc, sc = [], [], [], []
    for z in maps:
        z = zs(z)
        mc.append(G.mark_corr(z))
        h, c, _ = G.hotspots(z)
        nh.append(h); lc.append(c)
        sc.append(G.scan(z)[0])
    return {'mark_corr': np.array(mc), 'n_hot': np.array(nh), 'largest': np.array(lc), 'scan': np.array(sc)}


def pval(null, obs):
    null = np.asarray(null, float)
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def category_dprime(resp, ck, key):
    m = ck == key
    a, b = resp[:, m], resp[:, ~m]
    return (np.nanmean(a, 1) - np.nanmean(b, 1)) / np.sqrt(0.5 * (np.nanvar(a, 1) + np.nanvar(b, 1)) + 1e-12)


def cat_key(c):
    c = str(c).lower()
    return 'face' if 'face' in c else ('body' if 'body' in c else 'object')


def face_axis_map(resp, ck):
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    m = U[:, 0] * S[0]
    return m * np.sign(np.corrcoef(m, category_dprime(resp, ck, 'face'))[0, 1])


def resolve_session(key):
    cfg = dict(SESSIONS[key])
    if 'path' not in cfg:
        hits = [h for h in sorted(glob.glob(cfg['glob'])) if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
        cfg['path'] = hits[0]
    return cfg


def feature_pcs(data, stim_dir):
    """relu7 PC scores of the presented images (for the random-feature null); None if images are missing."""
    paths = som.condition_image_paths(data, stim_dir)
    if not all(os.path.exists(p) for p in paths):
        print('  (stimulus images not all found under %s -> random-feature null skipped)' % stim_dir)
        return None
    feats = dnn_som.extract_dnn_features(*dnn_som.prep_dnn_model('cpu'), dnn_som.load_images_rgb(paths), 'cpu').numpy()
    return v2.pcs(feats)[0]


def analyse(resp, xy, ck, label, n_perm, Fpc, rng, fig_path):
    n = resp.shape[0]
    G = Geometry(xy)
    print('\n##### %s: %d cells, hull area %.2f mm2, density %.0f / mm2, %d pairs' % (label, n, G.area / 1e6, n / (G.area / 1e6), G.dd.size))
    maps = {'face axis (nPC0)': face_axis_map(resp, ck)}
    for k in ('face', 'body', 'object'):
        maps["%s d'" % k] = category_dprime(resp, ck, k)

    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Uc = rng.standard_normal((resp.shape[1], n_perm)); Uc /= np.linalg.norm(Uc, axis=0, keepdims=True)
    NRP = null_stats(G, (R @ Uc).T)
    if Fpc is not None:
        Uf = rng.standard_normal((Fpc.shape[1], n_perm)); Uf /= np.linalg.norm(Uf, axis=0, keepdims=True)
        NRF = null_stats(G, np.column_stack([v2.rowcorr(resp, Fpc @ Uf[:, k]) for k in range(n_perm)]).T)
    else:
        NRF = NRP
    results = {}
    print('=== mark-correlation k(d) [bins %s um] ===' % ' '.join('%g' % b for b in BINS))
    print('  null p95 per bin: random projection %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRP['mark_corr'], 95, axis=0)))
    print('                    random feature    %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRF['mark_corr'], 95, axis=0)))
    for nm, m in maps.items():
        z = zs(m)
        mc = G.mark_corr(z)
        nh, lc, g = G.hotspots(z)
        s, ci, sr = G.scan(z)
        shuf = null_stats(G, np.array([z[rng.permutation(n)] for _ in range(n_perm)]))
        results[nm] = dict(z=z, mc=mc, n_hot=nh, largest=lc, g=g, scan=s, scan_i=ci, scan_r=sr, shuf=shuf)
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

    m_top = int(round(TOPQ * n))
    print('=== Ripley pair-count ratio of the top-%d%% cells vs random labelling (radii %s um): ratio (p) ===' % (100 * TOPQ, ' '.join('%g' % r for r in KR)))
    null_pc = np.array([G.pair_counts(rng.choice(n, m_top, replace=False)) for _ in range(n_perm)])
    tops = {}
    for nm in results:
        idx = np.argsort(-results[nm]['z'])[:m_top]
        tops[nm] = idx
        pc = G.pair_counts(idx)
        ratio = pc / null_pc.mean(0)
        print('  %-18s %s' % (nm, '  '.join('%.2f (%.3f)' % (ratio[k], pval(null_pc[:, k], pc[k])) for k in range(len(KR)))))
        results[nm]['kratio'] = ratio; results[nm]['kband'] = np.percentile(null_pc, [5, 95], axis=0) / null_pc.mean(0)
    fa, bo = tops["face d'"], tops["body d'"]
    both = np.intersect1d(fa, bo)
    fa, bo = np.setdiff1d(fa, both), np.setdiff1d(bo, both)
    cc = G.cross_counts(fa, bo)
    null_cc = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        null_cc.append(G.cross_counts(perm[:len(fa)], perm[len(fa):len(fa) + len(bo)]))
    null_cc = np.array(null_cc)
    ratio_cc = cc / null_cc.mean(0)
    p_lo = [(np.sum(null_cc[:, k] <= cc[k]) + 1) / (n_perm + 1) for k in range(len(KR))]
    p_hi = [(np.sum(null_cc[:, k] >= cc[k]) + 1) / (n_perm + 1) for k in range(len(KR))]
    print("=== cross-K face-top (n=%d) vs body-top (n=%d): ratio to random labelling, p_low (segregation) / p_high (attraction) ===" % (len(fa), len(bo)))
    print('  ' + '  '.join('r<=%g: %.2f (%.3f/%.3f)' % (KR[k], ratio_cc[k], p_lo[k], p_hi[k]) for k in range(len(KR))))

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    r = results['face axis (nPC0)']
    sc = ax[0, 0].scatter(xy[:, 0], xy[:, 1], c=r['z'], cmap='coolwarm', vmin=-2, vmax=2, s=14)
    hot = np.abs(r['g']) > GI_THR
    ax[0, 0].scatter(xy[hot, 0], xy[hot, 1], facecolors='none', edgecolors='k', s=50, linewidths=0.8, label='Gi* hotspot cells (n=%d)' % hot.sum())
    fc = G.largest_cluster(r['g'], 1)
    if fc.size:
        ax[0, 0].scatter(xy[fc, 0], xy[fc, 1], facecolors='none', edgecolors='darkred', s=110, linewidths=1.4, label='largest FACE hotspot cluster (n=%d)' % fc.size)
    oc = G.largest_cluster(results["object d'"]['g'], 1)
    if oc.size:
        ax[0, 0].scatter(xy[oc, 0], xy[oc, 1], marker='s', facecolors='none', edgecolors='navy', s=110, linewidths=1.4, label='largest OBJECT hotspot cluster (n=%d)' % oc.size)
    if r['scan_i'] >= 0:
        ax[0, 0].add_patch(Circle(xy[r['scan_i']], r['scan_r'], fill=False, ec='k', lw=2, ls='--', label='best scan window (stat %.2f, R=%g)' % (r['scan'], r['scan_r'])))
    ax[0, 0].set_aspect('equal'); ax[0, 0].legend(fontsize=6.5, loc='lower left'); ax[0, 0].set_title('face axis z per cell; hotspots, clusters, best scan window', fontsize=9)
    plt.colorbar(sc, ax=ax[0, 0], fraction=0.04)
    ax[0, 1].scatter(xy[:, 0], xy[:, 1], c=r['g'], cmap='coolwarm', vmin=-4, vmax=4, s=14)
    ax[0, 1].set_aspect('equal'); ax[0, 1].set_title('Getis-Ord Gi* (R=%g um) of the face axis' % R_GI, fontsize=9)
    ctr = 0.5 * (BINS[1:] + BINS[:-1])
    for nm, col in (('face axis (nPC0)', 'tab:red'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[0, 2].plot(ctr, results[nm]['mc'], '-o', color=col, label=nm)
    for N_, col, lab in ((NRP, '0.3', 'random projection p5-p95'), (NRF, '0.6', 'random feature p5-p95')):
        lo, hi = np.nanpercentile(N_['mark_corr'], [5, 95], axis=0)
        ax[0, 2].fill_between(ctr, lo, hi, color=col, alpha=0.3, label=lab)
    ax[0, 2].axhline(0, color='k', lw=0.5); ax[0, 2].set_xlabel('pair distance (um)'); ax[0, 2].set_ylabel('k(d) = mean z_i z_j'); ax[0, 2].legend(fontsize=7)
    ax[0, 2].set_title('mark-correlation function (T as a function of distance)', fontsize=9)
    for nm, col in (('face axis (nPC0)', 'tab:red'), ("face d'", 'tab:orange'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[1, 0].plot(KR, results[nm]['kratio'], '-o', color=col, label=nm)
    band = results["face d'"]['kband']
    ax[1, 0].fill_between(KR, band[0], band[1], color='0.5', alpha=0.3, label='random-labelling p5-p95')
    ax[1, 0].axhline(1, color='k', lw=0.5); ax[1, 0].set_xlabel('r (um)'); ax[1, 0].set_ylabel('pair count / null mean'); ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title('Ripley pair-count ratio, top-%d%% cells of each map' % (100 * TOPQ), fontsize=9)
    lo, hi = np.percentile(null_cc, [5, 95], axis=0) / null_cc.mean(0)
    ax[1, 1].fill_between(KR, lo, hi, color='0.5', alpha=0.3, label='random-labelling p5-p95')
    ax[1, 1].plot(KR, ratio_cc, '-o', color='purple', label='face-top vs body-top')
    ax[1, 1].axhline(1, color='k', lw=0.5); ax[1, 1].set_xlabel('r (um)'); ax[1, 1].set_ylabel('cross pair count / null mean'); ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title('bivariate cross-K: face-preferring vs body-preferring cells', fontsize=9)
    ax[1, 2].hist(NRP['scan'], bins=30, alpha=0.6, color='0.3', label='random projection')
    ax[1, 2].hist(NRF['scan'], bins=30, alpha=0.6, color='0.7', label='random feature')
    ax[1, 2].hist(r['shuf']['scan'], bins=30, alpha=0.6, color='tab:green', label='position shuffle')
    for nm, col in (('face axis (nPC0)', 'tab:red'), ("body d'", 'tab:green'), ("object d'", 'tab:blue')):
        ax[1, 2].axvline(results[nm]['scan'], color=col, ls='--', lw=1.5, label='%s = %.2f' % (nm, results[nm]['scan']))
    ax[1, 2].set_xlabel('scan statistic (max over windows)'); ax[1, 2].legend(fontsize=7); ax[1, 2].set_title('scan statistic vs the three nulls', fontsize=9)
    fig.suptitle('%s: non-gradient structure tests -- hotspots, scan window, mark-correlation, Ripley K, cross-K' % label, fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print('figure: %s' % fig_path)
    sys.stdout.flush()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default='pd', choices=sorted(SESSIONS))
    ap.add_argument('--gate', default=None, choices=['zeta', 'responsive', 'all'])
    ap.add_argument('--tile-um', type=float, default=None, help='tile edge for full-density tiling (0 = whole field)')
    ap.add_argument('--min-tile-rois', type=int, default=300)
    ap.add_argument('--only-tile', default=None, help='run a single tile by name (e.g. tile01) -- to split a long run into separate jobs')
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--no-sensitivity', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cfg = resolve_session(a.session)
    gate = a.gate or cfg['gate']
    tile_um = cfg['tile_um'] if a.tile_um is None else a.tile_um
    os.makedirs(OUTDIR, exist_ok=True)

    # tiles (full density) or whole field
    if tile_um and tile_um > 0:
        md = sessionio.load_metadata(cfg['path'])
        s2p = sessionio.load_suite2p(cfg['path'])
        xy_all = rm.calculate_roi_centroids_um(s2p['ROIs'], md['fov']['resolution_umpx'])
        x0, y0 = xy_all.min(0)
        nx = int(np.ceil((xy_all[:, 0].max() - x0) / tile_um)); ny = int(np.ceil((xy_all[:, 1].max() - y0) / tile_um))
        tiles = []
        for iy in range(ny):
            for ix in range(nx):
                sel = np.flatnonzero((xy_all[:, 0] >= x0 + ix * tile_um) & (xy_all[:, 0] < x0 + (ix + 1) * tile_um) &
                                     (xy_all[:, 1] >= y0 + iy * tile_um) & (xy_all[:, 1] < y0 + (iy + 1) * tile_um))
                if sel.size >= a.min_tile_rois:
                    tiles.append(('tile%d%d' % (iy, ix), sel))
        print('%s: %d ROIs total -> %d tiles of %g um with >= %d ROIs: %s' % (a.session, len(xy_all), len(tiles), tile_um, a.min_tile_rois,
                                                                            ', '.join('%s (%d)' % (t, s.size) for t, s in tiles)))
        if a.only_tile:
            tiles = [t for t in tiles if t[0] == a.only_tile]
    else:
        tiles = [('', None)]

    Fpc = None
    for tname, sel in tiles:
        data = rm.build_response_matrix(cfg['path'], denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7,
                                        roi_select=sel, **cfg['kw'])
        n_all = data['response'].shape[0]
        ck = np.array([cat_key(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
        if gate == 'zeta' and cfg['zeta'] and os.path.exists(cfg['zeta']):
            mask = pd.read_csv(cfg['zeta'])['p_zeta'].to_numpy() < 0.05
        elif gate == 'responsive':
            mask = rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)['responsive']
        else:
            mask = np.ones(n_all, bool)
        d = rm.apply_roi_mask(data, mask)
        resp, xy = d['response'], d['roi_xy_um']
        label = '%s%s (corrected 0.7, gate=%s, %d/%d cells)' % (a.session.upper(), ' ' + tname if tname else '', gate, resp.shape[0], n_all)
        if Fpc is None:
            Fpc = feature_pcs(data, cfg['stim_dir'])
        if a.session == 'pd':
            G0 = Geometry(xy)
            cls = rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)
            rel, sel_ = d['reliability'], cls['selective'][mask]
            q = np.nanpercentile(rel, [25, 50, 75, 90])
            print('split-half tuning reliability (3 splits): gated quartiles %.3f / %.3f / %.3f, p90 %.3f, frac above 0.2 = %.2f | '
                  'ANOVA-selective subset (n=%d) median %.3f, frac above 0.2 = %.2f'
                  % (q[0], q[1], q[2], q[3], np.nanmean(rel > 0.2), sel_.sum(), np.nanmedian(rel[sel_]), np.nanmean(rel[sel_] > 0.2)))
            near, far = G0.dd < v2.NEAR, G0.dd > v2.FAR
            for nd in (500, 5000):
                T = fs.random_projection_T(resp, G0.iu, near, far, n=nd, seed=1)
                print('random-projection T with %4d directions: median %+.4f  p95 %+.4f  p99 %+.4f' % (nd, np.median(T), np.percentile(T, 95), np.percentile(T, 99)))
        fig_path = os.path.join(OUTDIR, 'spatial_structure_%s%s.png' % (a.session, '_' + tname if tname else ''))
        analyse(resp, xy, ck, label, a.n_perm, Fpc, rng, fig_path)

        if a.session == 'pd' and not a.no_sensitivity:
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
                nh_, lc_, _ = G_.hotspots(z_)
                print('  neucoeff %-4s scan %.2f (R=%g, p_rproj %.3f) | hotspots %d (p %.3f) | largest cluster %d (p %.3f)'
                      % ('none' if cf is None else cf, s_, R_scan, pval(N_['scan'], s_), nh_, pval(N_['n_hot'], nh_), lc_, pval(N_['largest'], lc_)))
            sys.stdout.flush()


if __name__ == '__main__':
    main()
