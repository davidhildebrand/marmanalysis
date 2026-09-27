#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-SOM spatial-structure tests (roadmap #12; DH 2026-09-24/27): is there ANY structure beyond a smooth gradient in
the PD map -- a single column-like cluster, patches, or segregation between preferences -- without assuming
periodicity?

Maps (one scalar per ROI, z-scored across ROIs): the face axis (neural PC0 score), and face / body / object
preference (d' of the category vs the rest). Statistics:
  1. MARK-CORRELATION FUNCTION k(d) = mean(z_i z_j | d_ij in bin): the near-minus-far T as a full function of
     distance. Monotone decay = gradient; plateau then drop = patch of that size; a dip below the null = hole /
     periodicity.
  2. LOCAL HOTSPOTS (Getis-Ord Gi*): per cell the standardised sum of z over neighbours within R_GI (self included);
     a hotspot is |Gi*| > 2.58; we report how many cells are hotspots and the size of the largest connected (within
     R_GI, same-sign) hotspot cluster -- the "column-like cluster" candidate.
  3. SCAN STATISTIC: over circular windows centred on every cell with radii RADII, the standardised difference of
     the mean z inside vs outside; the maximum over windows is the scan statistic (the single most anomalous
     patch) and its p-value is the fraction of null maps whose own maximum over the same windows is as large.
  4. RIPLEY-K PAIR COUNTS of the top-TOPQ cells of a map at radii KR, vs a RANDOM-LABELLING null (the same number of
     cells drawn at random from the same positions): clustering of e.g. face-preferring cells at ANY scale
     (ratio > 1 above the null band).
  5. BIVARIATE CROSS-K between two subpopulations (face-top vs body-top): attraction (ratio > 1) vs segregation.
Nulls for 1-3 (all N_PERM maps): (a) POSITION SHUFFLE (any structure); (b) RANDOM PROJECTION in condition space
(the generic smoothness ANY tuning direction inherits); (c) RANDOM FEATURE DIRECTION in relu7-PC space (a random
but image-plausible feature -- the Chang & Tsao 2017 STA-style axis null: each cell's preference for a direction
in image-feature space). Also reported: how many random directions are needed (T p95/p99 for 500 vs 5000),
split-half reliability quantiles, and the face-axis scan / hotspot results at neucoeff none / 1.0.
Figure: output/spatial_structure_pd.png
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/spatial_structure.py [--no-sensitivity]
"""
import argparse
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
import som
import topography as tg

BINS = np.array([0, 25, 50, 75, 100, 150, 200, 300], float)
RADII = (40.0, 60.0, 80.0, 100.0)
R_GI = 50.0
GI_THR = 2.58
N_PERM = 1000
TOPQ = 0.2
KR = np.array([25, 50, 75, 100, 150, 200], float)
OUTDIR = 'output'


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
        self.M = {R: (self.dist <= R).astype(float) for R in RADII}
        self.n_in = {R: self.M[R].sum(1) for R in RADII}
        self.W_gi = (self.dist <= R_GI)
        self.n_gi = self.W_gi.sum(1).astype(float)
        self.area = float(ConvexHull(self.xy).volume)

    def mark_corr(self, z):
        p = z[self.iu[0]] * z[self.iu[1]]
        return np.array([p[m].mean() if m.any() else np.nan for m in self.bin_masks])

    def gistar(self, z):
        return (self.W_gi @ z) / np.sqrt(self.n_gi * (self.n - self.n_gi) / (self.n - 1.0))

    def hotspots(self, z):
        g = self.gistar(z)
        hot = np.abs(g) > GI_THR
        best = 0
        for sign in (1, -1):
            idx = np.flatnonzero(hot & (np.sign(g) == sign))
            if idx.size:
                _, lab = connected_components(csr_matrix(self.W_gi[np.ix_(idx, idx)]), directed=False)
                best = max(best, int(np.bincount(lab).max()))
        return int(hot.sum()), best, g

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
    """For a stack of null maps (n_maps x n): mark-corr curves, hotspot counts, largest clusters, scan maxima."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-sensitivity', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    data = nc.build(nc.PD, 0.7)
    zeta = pd.read_csv(nc.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    n = resp.shape[0]
    ck = np.array([cat_key(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
    G = Geometry(xy)
    print('PD corrected 0.7, %d ZETA ROIs, FOV hull area %.2f mm2, %d pairs' % (n, G.area / 1e6, G.dd.size))

    # ---- reliability question (DH: is 0.073 decent?) ----
    cls = rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)
    rel = d['reliability']
    sel = cls['selective'][zeta]
    q = np.nanpercentile(rel, [25, 50, 75, 90])
    print('split-half tuning reliability (3 splits): ZETA quartiles %.3f / %.3f / %.3f, p90 %.3f, frac > 0.2 = %.2f | '
          'ANOVA-selective subset (n=%d) median %.3f, frac > 0.2 = %.2f'
          % (q[0], q[1], q[2], q[3], np.nanmean(rel > 0.2), sel.sum(), np.nanmedian(rel[sel]), np.nanmean(rel[sel] > 0.2)))

    # ---- how many random directions (DH question) ----
    dd = G.dd
    near, far = dd < v2.NEAR, dd > v2.FAR
    for nd in (500, 5000):
        T = fs.random_projection_T(resp, G.iu, near, far, n=nd, seed=1)
        print('random-projection T with %4d directions: median %+.4f  p95 %+.4f  p99 %+.4f' % (nd, np.median(T), np.percentile(T, 95), np.percentile(T, 99)))

    # ---- maps ----
    maps = {'face axis (nPC0)': face_axis_map(resp, ck)}
    for k in ('face', 'body', 'object'):
        maps["%s d'" % k] = category_dprime(resp, ck, k)

    # ---- null map stacks ----
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    Uc = rng.standard_normal((resp.shape[1], N_PERM)); Uc /= np.linalg.norm(Uc, axis=0, keepdims=True)
    null_rp = (R @ Uc).T
    paths = som.condition_image_paths(data, v2.STIM_DIR)
    feats = dnn_som.extract_dnn_features(*dnn_som.prep_dnn_model('cpu'), dnn_som.load_images_rgb(paths), 'cpu').numpy()
    Fpc, _ = v2.pcs(feats)
    Uf = rng.standard_normal((Fpc.shape[1], N_PERM)); Uf /= np.linalg.norm(Uf, axis=0, keepdims=True)
    null_rf = np.column_stack([v2.rowcorr(resp, Fpc @ Uf[:, k]) for k in range(N_PERM)]).T
    print('computing nulls (%d maps each): random projection, random relu7-feature direction ...' % N_PERM); sys.stdout.flush()
    NRP, NRF = null_stats(G, null_rp), null_stats(G, null_rf)

    results = {}
    print('\n=== per map: mark-correlation k(d) [bins %s um] ===' % ' '.join('%g' % b for b in BINS))
    print('  null bands = random-projection p95 per bin: %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRP['mark_corr'], 95, axis=0)))
    print('               random-feature   p95 per bin: %s' % ' '.join('%+.3f' % v for v in np.nanpercentile(NRF['mark_corr'], 95, axis=0)))
    for nm, m in maps.items():
        z = zs(m)
        mc = G.mark_corr(z)
        nh, lc, g = G.hotspots(z)
        s, ci, sr = G.scan(z)
        shuf = null_stats(G, np.array([z[rng.permutation(n)] for _ in range(N_PERM)]))
        results[nm] = dict(z=z, mc=mc, n_hot=nh, largest=lc, g=g, scan=s, scan_i=ci, scan_r=sr, shuf=shuf)
        print('  %-18s k(d): %s' % (nm, ' '.join('%+.3f' % v for v in mc)))
    print('\n=== per map: hotspots (Gi*, R=%g um, |Gi*|>%.2f) and scan statistic (radii %s um) ===' % (R_GI, GI_THR, RADII))
    print('  %-18s %7s %7s | %6s %5s %5s | %6s %7s %7s | %s' % ('map', 'n_hot', 'largest', 'scan', 'n_in', 'R', 'p_shuf', 'p_rproj', 'p_rfeat', 'null medians: n_hot / largest / scan (shuf | rproj | rfeat)'))
    for nm, r in results.items():
        n_in = int(G.n_in[r['scan_r']][r['scan_i']])
        print('  %-18s %7d %7d | %6.2f %5d %5g | %6.3f %7.3f %7.3f | %d/%d/%.2f | %d/%d/%.2f | %d/%d/%.2f'
              % (nm, r['n_hot'], r['largest'], r['scan'], n_in, r['scan_r'],
                 pval(r['shuf']['scan'], r['scan']), pval(NRP['scan'], r['scan']), pval(NRF['scan'], r['scan']),
                 np.median(r['shuf']['n_hot']), np.median(r['shuf']['largest']), np.median(r['shuf']['scan']),
                 np.median(NRP['n_hot']), np.median(NRP['largest']), np.median(NRP['scan']),
                 np.median(NRF['n_hot']), np.median(NRF['largest']), np.median(NRF['scan'])))
        print('  %-18s hotspot count p: shuf %.3f rproj %.3f rfeat %.3f | largest-cluster p: shuf %.3f rproj %.3f rfeat %.3f'
              % ('', pval(r['shuf']['n_hot'], r['n_hot']), pval(NRP['n_hot'], r['n_hot']), pval(NRF['n_hot'], r['n_hot']),
                 pval(r['shuf']['largest'], r['largest']), pval(NRP['largest'], r['largest']), pval(NRF['largest'], r['largest'])))

    # ---- Ripley pair counts of the top-q cells (random labelling) + cross-K face vs body ----
    m_top = int(round(TOPQ * n))
    print('\n=== Ripley pair-count ratio of the top-%d%% cells vs random labelling (radii %s um): ratio (p) ===' % (100 * TOPQ, ' '.join('%g' % r for r in KR)))
    null_pc = np.array([G.pair_counts(rng.choice(n, m_top, replace=False)) for _ in range(N_PERM)])
    tops = {}
    for nm in ('face axis (nPC0)', "face d'", "body d'", "object d'"):
        idx = np.argsort(-results[nm]['z'])[:m_top]
        tops[nm] = idx
        pc = G.pair_counts(idx)
        ratio = pc / null_pc.mean(0)
        ps = [pval(null_pc[:, k], pc[k]) for k in range(len(KR))]
        print('  %-18s %s' % (nm, '  '.join('%.2f (%.3f)' % (ratio[k], ps[k]) for k in range(len(KR)))))
        results[nm]['kratio'] = ratio; results[nm]['kband'] = np.percentile(null_pc, [5, 95], axis=0) / null_pc.mean(0)
    fa, bo = tops["face d'"], tops["body d'"]
    both = np.intersect1d(fa, bo)
    fa, bo = np.setdiff1d(fa, both), np.setdiff1d(bo, both)
    cc = G.cross_counts(fa, bo)
    null_cc = []
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        null_cc.append(G.cross_counts(perm[:len(fa)], perm[len(fa):len(fa) + len(bo)]))
    null_cc = np.array(null_cc)
    ratio_cc = cc / null_cc.mean(0)
    p_lo = [(np.sum(null_cc[:, k] <= cc[k]) + 1) / (N_PERM + 1) for k in range(len(KR))]
    p_hi = [(np.sum(null_cc[:, k] >= cc[k]) + 1) / (N_PERM + 1) for k in range(len(KR))]
    print("\n=== cross-K face-top (n=%d) vs body-top (n=%d), ratio to random labelling: <1 segregation (p_low), >1 attraction (p_high) ===" % (len(fa), len(bo)))
    print('  ' + '  '.join('r<=%g: %.2f (lo %.3f / hi %.3f)' % (KR[k], ratio_cc[k], p_lo[k], p_hi[k]) for k in range(len(KR))))

    # ---- neucoeff sensitivity for the face axis ----
    if not a.no_sensitivity:
        print('\n=== face-axis scan / hotspots at other neuropil coefficients (random-projection null recomputed per build) ===')
        for cf in (None, 1.0):
            dd_ = rm.apply_roi_mask(nc.build(nc.PD, cf), zeta)
            r_ = dd_['response']
            G_ = Geometry(dd_['roi_xy_um'])
            z_ = zs(face_axis_map(r_, ck))
            R_ = np.nan_to_num(r_ - np.nanmean(r_, 0, keepdims=True))
            N_ = null_stats(G_, (R_ @ Uc[:, :300]).T)
            s_, i_, R_scan = G_.scan(z_)
            nh_, lc_, _ = G_.hotspots(z_)
            print('  neucoeff %-4s scan %.2f (R=%g, p_rproj %.3f) | hotspots %d (p %.3f) | largest cluster %d (p %.3f)'
                  % ('none' if cf is None else cf, s_, R_scan, pval(N_['scan'], s_), nh_, pval(N_['n_hot'], nh_), lc_, pval(N_['largest'], lc_)))

    # ---- figure ----
    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    r = results['face axis (nPC0)']
    sc = ax[0, 0].scatter(xy[:, 0], xy[:, 1], c=r['z'], cmap='coolwarm', vmin=-2, vmax=2, s=18)
    hot = np.abs(r['g']) > GI_THR
    ax[0, 0].scatter(xy[hot, 0], xy[hot, 1], facecolors='none', edgecolors='k', s=60, linewidths=1.0, label='Gi* hotspot cells (n=%d)' % hot.sum())
    ax[0, 0].add_patch(Circle(xy[r['scan_i']], r['scan_r'], fill=False, ec='k', lw=2, ls='--', label='best scan window (stat %.2f, R=%g)' % (r['scan'], r['scan_r'])))
    ax[0, 0].set_aspect('equal'); ax[0, 0].legend(fontsize=7, loc='lower left'); ax[0, 0].set_title('face axis z per cell; hotspots + best scan window', fontsize=9)
    plt.colorbar(sc, ax=ax[0, 0], fraction=0.04)
    ax[0, 1].scatter(xy[:, 0], xy[:, 1], c=r['g'], cmap='coolwarm', vmin=-4, vmax=4, s=18)
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
    fig.suptitle('PD (corrected 0.7, %d ZETA ROIs): non-gradient structure tests -- hotspots, scan window, mark-correlation, Ripley K, cross-K' % n, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'spatial_structure_pd.png'), dpi=120)
    print('\nfigure: output/spatial_structure_pd.png')


if __name__ == '__main__':
    main()
