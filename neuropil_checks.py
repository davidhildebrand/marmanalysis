#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neuropil correction: WHAT is subtracted (suite2p's neuropil mask geometry) and IS the subtraction removing
contamination rather than signal? (DH question 2026-09-24, roadmap item 13.)

--geometry   Figure of suite2p's neuropil zone for an example PD ROI: the ROI's pixels, the neighbouring ROIs'
             pixels, and the neuropil mask (a ring that starts ``inner_neuropil_radius``=2 px outside the ROI and
             is grown outward in 5-px steps until >= ``min_neuropil_pixels``=350 pixels that belong to NO detected
             ROI are collected; ``allow_overlap``=False). Plus the ring's inner/outer radius and the fraction of
             the ring occupied (and therefore excluded) by other ROIs, across all ROIs.
--pd         Certainty checks on the PD session, all three quantities through the IDENTICAL response pipeline:
             cell traces uncorrected (F), corrected (Fc = F - 0.7*(Fneu - median Fneu)), and the neuropil traces
             THEMSELVES treated as if they were cells (Fneu swapped in for F):
               1. how much of each cell's tuning is shared with its own surround (per-ROI corr of tuning vectors:
                  F vs Fneu; Fc vs Fneu -- near 0 = the shared part was removed, negative = over-corrected);
               2. is the cell's identity preserved (per-ROI corr F vs Fc; face-d' correlation across ROIs; sign
                  flips; split-half reliability with vs without);
               3. is the SURROUND signal itself spatially smooth and face-tuned (the neuropil matrix's own face-axis
                  T, topography score and random-projection T) -- if so, subtracting it removes exactly the shared
                  smooth field that inflated the uncorrected topography;
               4. over-correction signature in the raw traces at neucoeff 0 / 0.7 / 1.0: fraction of frames below
                  -3 sigma vs above +3 sigma (negative dips appear when too much neuropil is subtracted), skewness;
               5. responsive / selective fractions, overall and among suite2p-ACCEPTED ROIs (iscell) only.
--curly      The same with/without comparison on a large-FOV 3 um/px session (Curly 20231103d FOBmin, 2000-ROI
             subsample) where the cell<->Fneu trace correlation is low (0.23 vs PD 0.71): is the correction's effect
             on topography proportionally smaller there?
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/neuropil_checks.py --geometry --pd
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/neuropil_checks.py --curly
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
from scipy.stats import skew, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_specificity as fs
import feature_topography_v2 as v2
import response_matrix as rm
import response_table as rt
import sessionio
import signal_quality as sq
import topography as tg

PD = v2.SESSION
ZETA_CSV = v2.ZETA_CSV
CURLY_GLOB = 'suite2p_results/Curly/20231103d/*ImagesFOBmin*'
OUTDIR = 'output'


def _rowcorr(A, B):
    A = np.nan_to_num(np.asarray(A, float)); B = np.nan_to_num(np.asarray(B, float))
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (A * B).sum(1) / den


def build(session, neucoeff, **kw):
    if neucoeff is None:
        return rm.build_response_matrix(session, denoise=False, reliability_splits=3, neuropil_subtract=False, **kw)
    return rm.build_response_matrix(session, denoise=False, reliability_splits=3, neuropil_subtract=True,
                                    neucoeff=neucoeff, **kw)


def build_neuropil_as_cells(session, **kw):
    """Run the identical pipeline on the NEUROPIL traces (Fneu swapped in for F, no correction)."""
    orig = sessionio.load_suite2p

    def swapped(*a, **k):
        s = dict(orig(*a, **k))
        if s['Fneu'] is None:
            raise RuntimeError('no Fneu for %s' % session)
        s['Frois'] = s['Fneu'].copy()
        return s

    sessionio.load_suite2p = swapped
    try:
        return rm.build_response_matrix(session, denoise=False, reliability_splits=3, neuropil_subtract=False, **kw)
    finally:
        sessionio.load_suite2p = orig


def face_dprime(resp, cats):
    c = np.char.lower(np.asarray(cats).astype(str))
    isf = np.array(['face' in x for x in c])
    a, b = resp[:, isf], resp[:, ~isf]
    return (np.nanmean(a, 1) - np.nanmean(b, 1)) / np.sqrt(0.5 * (np.nanvar(a, 1) + np.nanvar(b, 1)) + 1e-12), isf


def topo_summary(resp, xy, mind=v2.MIND, maxd=v2.MAXD):
    """Face-axis (neural PC0) T, median random-projection T, and the full-tuning Spearman[sim, -dist] score."""
    dist = tg.pairwise_distance(xy)
    iu = np.triu_indices(resp.shape[0], 1)
    dd = dist[iu]
    near, far = dd < v2.NEAR, dd > v2.FAR
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    t_face = fs.T_of(U[:, :1] * S[:1], iu, near, far)[0]
    t_rand = float(np.median(fs.random_projection_T(resp, iu, near, far, n=200)))
    sim = tg.pairwise_tuning_similarity(resp)[iu]
    ok = (dd >= mind) & (dd <= maxd) & np.isfinite(sim)
    score = spearmanr(sim[ok], -dd[ok]).statistic
    return t_face, t_rand, score


def accepted_mask(data):
    s2p = data['ctx']['s2p']
    isc = np.asarray(s2p['iscell'])
    ci = np.asarray(s2p['cellinds'])
    return isc[ci, 0].astype(bool) if isc.shape[0] > ci.max() else isc[:, 0].astype(bool)


# ----------------------------------------------------------------------------------------------------------------
def geometry(session=PD, out=os.path.join(OUTDIR, 'neuropil_geometry_pd.png')):
    md = sessionio.load_metadata(session)
    s2p = sessionio.load_suite2p(session)
    ops, stat = s2p['ops'], s2p['ROIs']
    Ly, Lx = int(ops['Ly']), int(ops['Lx'])
    res = float(np.asarray(md['fov']['resolution_umpx'], float).ravel()[0])
    img = np.asarray(ops['meanImg'], float)
    cell_pix = np.zeros((Ly, Lx), bool)
    for s in stat:
        cell_pix[s['ypix'], s['xpix']] = True
    cy = np.array([s['med'][0] for s in stat], float); cx = np.array([s['med'][1] for s in stat], float)
    r_roi = np.array([s.get('radius', np.sqrt(s['npix'] / np.pi)) for s in stat], float)
    inner, outer, n_neu, frac_occ = [], [], [], []
    for s in stat:
        m = np.asarray(s['neuropil_mask'], int)
        yy, xx = np.unravel_index(m, (Ly, Lx))
        d = np.hypot(yy - s['med'][0], xx - s['med'][1])
        r_in, r_out = float(d.min()), float(d.max())
        inner.append(r_in); outer.append(r_out); n_neu.append(m.size)
        y0, y1 = int(max(0, s['med'][0] - r_out - 1)), int(min(Ly, s['med'][0] + r_out + 2))
        x0, x1 = int(max(0, s['med'][1] - r_out - 1)), int(min(Lx, s['med'][1] + r_out + 2))
        Y, X = np.mgrid[y0:y1, x0:x1]
        ann = (np.hypot(Y - s['med'][0], X - s['med'][1]) >= r_in) & (np.hypot(Y - s['med'][0], X - s['med'][1]) <= r_out)
        frac_occ.append(float(cell_pix[y0:y1, x0:x1][ann].mean()))
    inner, outer, n_neu, frac_occ = map(np.asarray, (inner, outer, n_neu, frac_occ))

    # example ROI: responsive (ZETA), away from the edges, with the most neighbours within 25 px
    n = len(stat)
    resp_mask = np.ones(n, bool)
    if os.path.exists(ZETA_CSV):
        z = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if z.size == n:
            resp_mask = z
    D = np.hypot(cy[:, None] - cy[None, :], cx[:, None] - cx[None, :])
    nn25 = (D < 25).sum(1) - 1
    ok = resp_mask & (cy > 70) & (cy < Ly - 70) & (cx > 70) & (cx < Lx - 70)
    i = int(np.argmax(np.where(ok, nn25, -1)))
    s = stat[i]
    half = 60
    y0, x0 = int(cy[i]) - half, int(cx[i]) - half
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    lo, hi = np.percentile(crop, [1, 99.5])
    layer = np.zeros(crop.shape + (4,))
    m = np.asarray(s['neuropil_mask'], int)
    yy, xx = np.unravel_index(m, (Ly, Lx))
    for (py, px), col in (((yy, xx), (1.0, 0.85, 0.0, 0.45)),):
        sel = (py >= y0) & (py < y0 + 2 * half) & (px >= x0) & (px < x0 + 2 * half)
        layer[py[sel] - y0, px[sel] - x0] = col
    for j in np.flatnonzero(D[i] < 2 * half):
        if j == i:
            continue
        py, px = stat[j]['ypix'], stat[j]['xpix']
        sel = (py >= y0) & (py < y0 + 2 * half) & (px >= x0) & (px < x0 + 2 * half)
        layer[py[sel] - y0, px[sel] - x0] = (0.2, 0.5, 1.0, 0.5)
    py, px = s['ypix'], s['xpix']
    sel = (py >= y0) & (py < y0 + 2 * half) & (px >= x0) & (px < x0 + 2 * half)
    layer[py[sel] - y0, px[sel] - x0] = (1.0, 0.15, 0.15, 0.65)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    ax[0].imshow(crop, cmap='gray', vmin=lo, vmax=hi)
    ax[0].imshow(layer)
    c0 = (cx[i] - x0, cy[i] - y0)
    for r, col, lab in ((r_roi[i], 'w', 'ROI radius %.1f px' % r_roi[i]),
                        (inner[i], 'orange', 'ring inner edge %.1f px (gap = %d px)' % (inner[i], int(ops.get('inner_neuropil_radius', 2)))),
                        (outer[i], 'yellow', 'ring outer edge %.1f px' % outer[i])):
        ax[0].add_patch(Circle(c0, r, fill=False, ec=col, ls='--', lw=1.2, label=lab))
    ax[0].legend(loc='lower left', fontsize=7, framealpha=0.85)
    ax[0].set_title('PD ROI #%d (%d neighbours < 25 px): ROI (red), other ROIs (blue), neuropil mask (yellow)\n'
                    '%d neuropil px; %.0f%% of the ring area is occupied by other ROIs (excluded)'
                    % (i, nn25[i], n_neu[i], 100 * frac_occ[i]), fontsize=8.5)
    ax[0].set_xlabel('px (= %.2f um)' % res); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].hist(inner * res, bins=30, alpha=0.7, label='ring inner edge (median %.1f um)' % np.median(inner * res))
    ax[1].hist(outer * res, bins=30, alpha=0.7, label='ring outer edge (median %.1f um)' % np.median(outer * res))
    ax[1].hist(r_roi * res, bins=30, alpha=0.5, label='ROI radius (median %.1f um)' % np.median(r_roi * res))
    ax[1].set_xlabel('distance from ROI centroid (um)'); ax[1].set_ylabel('ROIs'); ax[1].legend(fontsize=7.5)
    ax[1].set_title('neuropil ring radii, all %d PD ROIs' % n, fontsize=9)
    ax[2].hist(100 * frac_occ, bins=30, color='0.5')
    ax[2].set_xlabel('% of the ring annulus occupied by OTHER detected ROIs (excluded from Fneu)')
    ax[2].set_ylabel('ROIs')
    ax[2].set_title('median %.0f%% -- undetected somata are NOT excluded (the ribo-indicator concern)' % (100 * np.median(frac_occ)),
                    fontsize=9)
    fig.suptitle('suite2p neuropil zone (inner_neuropil_radius=%s px, min_neuropil_pixels=%s, allow_overlap=%s): Fneu = mean of the yellow pixels; '
                 'Fc = F - 0.7 * (Fneu - median Fneu)' % (ops.get('inner_neuropil_radius'), ops.get('min_neuropil_pixels'), ops.get('allow_overlap')),
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(out, dpi=130)
    print('GEOMETRY: example ROI %d | ring inner %.1f / outer %.1f um (medians over ROIs %.1f / %.1f um) | '
          'ROI radius median %.1f um | neuropil px median %d | ring occupied by other ROIs median %.0f%% -> %s'
          % (i, inner[i] * res, outer[i] * res, np.median(inner * res), np.median(outer * res), np.median(r_roi * res),
             int(np.median(n_neu)), 100 * np.median(frac_occ), out))


# ----------------------------------------------------------------------------------------------------------------
def pd_checks(session=PD):
    d0, d7 = build(session, None), build(session, 0.7)
    dn = build_neuropil_as_cells(session)
    n = d0['response'].shape[0]
    assert d7['response'].shape[0] == n == dn['response'].shape[0], 'ROI sets differ'
    cats = d0['conditions'].reindex(d0['condition_ids'])['cat'].to_numpy()
    zeta = np.ones(n, bool)
    if os.path.exists(ZETA_CSV):
        z = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
        if z.size == n:
            zeta = z
    acc = accepted_mask(d0)
    fr = d0['framerate']
    print('PD: n_roi=%d | ZETA-responsive=%d | suite2p-accepted=%d | categories=%s'
          % (n, zeta.sum(), acc.sum(), sorted(set(map(str, cats)))))
    R0, R7, Rn = d0['response'], d7['response'], dn['response']

    print('\n[1] SHARED TUNING cell vs its own surround (per-ROI corr of 60-condition tuning vectors; median over ZETA)')
    c0n, c7n, c07 = _rowcorr(R0, Rn), _rowcorr(R7, Rn), _rowcorr(R0, R7)
    print('    F  vs Fneu : %+.3f   (uncorrected cell tuning resembles its surround)' % np.nanmedian(c0n[zeta]))
    print('    Fc vs Fneu : %+.3f   (~0 = shared part removed; clearly negative = over-corrected)' % np.nanmedian(c7n[zeta]))
    print('    F  vs Fc   : %+.3f   (cell identity preserved)' % np.nanmedian(c07[zeta]))

    print('\n[2] FACE d-prime with vs without (ZETA): corr across ROIs and sign flips; split-half reliability')
    dp0, isf = face_dprime(R0, cats); dp7, _ = face_dprime(R7, cats); dpn, _ = face_dprime(Rn, cats)
    print('    corr(dprime without, with) = %.3f | sign flips %d/%d | |dprime| median without %.2f -> with %.2f'
          % (np.corrcoef(dp0[zeta], dp7[zeta])[0, 1], int((np.sign(dp0[zeta]) != np.sign(dp7[zeta])).sum()), zeta.sum(),
             np.nanmedian(np.abs(dp0[zeta])), np.nanmedian(np.abs(dp7[zeta]))))
    print('    neuropil (Fneu) face dprime median %.2f | corr(Fneu dprime, cell dprime without) = %.3f'
          % (np.nanmedian(dpn[zeta]), np.corrcoef(dpn[zeta], dp0[zeta])[0, 1]))
    rel0, rel7 = d0['reliability'], d7['reliability']
    print('    split-half tuning reliability median (ZETA): without %.3f -> with %.3f' % (np.nanmedian(rel0[zeta]), np.nanmedian(rel7[zeta])))

    print('\n[3] SPATIAL structure of the three signals (ZETA; T = near<%dum minus far>%dum z-product; score = Spearman[sim,-dist])'
          % (v2.NEAR, v2.FAR))
    xy = d0['roi_xy_um'][zeta]
    print('    %-22s %10s %14s %10s' % ('signal', 'T face-PC0', 'T random-proj', 'topo score'))
    for nm, R in (('cells uncorrected F', R0), ('cells corrected Fc', R7), ('NEUROPIL Fneu itself', Rn)):
        tf, tr, sc = topo_summary(R[zeta], xy)
        print('    %-22s %+10.4f %+14.4f %+10.4f' % (nm, tf, tr, sc))

    print('\n[4] OVER-CORRECTION signature in raw traces (all analysed ROIs): frames < -3 sigma vs > +3 sigma (noise sigma = diff-MAD)')
    s2p = d0['ctx']['s2p']
    F, Fneu = s2p['Frois'].astype(float), s2p['Fneu'].astype(float)
    print('    %-8s %12s %12s %10s %10s' % ('neucoeff', 'frac<-3sig', 'frac>+3sig', 'neg/pos', 'skewness'))
    for c in (0.0, 0.7, 1.0):
        tr = sessionio.compute_fluorescence_metrics(F, fr, fneu=Fneu, neucoeff=c, neuropil_subtract=c > 0)
        dff = tr['FdFF']
        sig = sq.noise_sigma(dff, fr, method='diff')[:, None]
        neg, pos = (dff < -3 * sig).mean(1), (dff > 3 * sig).mean(1)
        print('    %-8.1f %12.4f %12.4f %10.3f %10.2f' % (c, np.median(neg), np.median(pos), np.median(neg) / max(np.median(pos), 1e-9),
                                                        np.median(skew(dff, axis=1))))
    cc = _rowcorr(F, Fneu)
    print('    per-ROI corr(F, Fneu) median %.3f (QC metric cell_fneu_corr_med)' % np.nanmedian(cc))

    print('\n[5] RESPONSIVE / SELECTIVE fractions (ANOVA gate, alpha .05): all analysed vs suite2p-ACCEPTED ROIs')
    for nm, d in (('without', d0), ('with 0.7', d7)):
        c = rt.classify_responses(d['ds'], metric='Fzsc', framerate=fr, alpha=0.05)
        print('    %-9s all(n=%d): responsive %.3f selective %.3f | accepted(n=%d): responsive %.3f selective %.3f | non-accepted(n=%d): responsive %.3f'
              % (nm, n, c['responsive'].mean(), c['selective'].mean(), acc.sum(), c['responsive'][acc].mean(), c['selective'][acc].mean(),
                 (~acc).sum(), c['responsive'][~acc].mean()))
    sys.stdout.flush()


# ----------------------------------------------------------------------------------------------------------------
def curly_checks(subsample=2000):
    hits = [h for h in sorted(glob.glob(CURLY_GLOB)) if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
    session = hits[0]
    kw = dict(roi_subsample=subsample, equalize_repeats=True, eye_gate_mode='none')
    d0, d7 = build(session, None, **kw), build(session, 0.7, **kw)
    n = d0['response'].shape[0]
    fr = d0['framerate']
    acc = accepted_mask(d0)
    cats = d0['conditions'].reindex(d0['condition_ids'])['cat'].to_numpy()
    print('CURLY %s: n_roi=%d (subsample) | accepted=%d' % (os.path.basename(session)[:50], n, acc.sum()))
    c0 = rt.classify_responses(d0['ds'], metric='Fzsc', framerate=fr, alpha=0.05)
    c7 = rt.classify_responses(d7['ds'], metric='Fzsc', framerate=fr, alpha=0.05)
    for nm, c in (('without', c0), ('with 0.7', c7)):
        print('    %-9s responsive %.3f selective %.3f | accepted-only responsive %.3f selective %.3f'
              % (nm, c['responsive'].mean(), c['selective'].mean(), c['responsive'][acc].mean(), c['selective'][acc].mean()))
    resp_set = c0['responsive'] | c7['responsive']
    R0, R7 = d0['response'], d7['response']
    print('    per-ROI tuning corr F vs Fc median (responsive) %.3f | reliability median without %.3f -> with %.3f'
          % (np.nanmedian(_rowcorr(R0, R7)[resp_set]), np.nanmedian(d0['reliability'][resp_set]), np.nanmedian(d7['reliability'][resp_set])))
    dp0, _ = face_dprime(R0, cats); dp7, _ = face_dprime(R7, cats)
    print('    corr(face dprime without, with) over responsive = %.3f' % np.corrcoef(dp0[resp_set], dp7[resp_set])[0, 1])
    xy = d0['roi_xy_um'][resp_set]
    print('    %-22s %10s %14s %10s   (responsive set n=%d; score within %d-%d um)' % ('signal', 'T face-PC0', 'T random-proj', 'topo score', resp_set.sum(), 15, 2000))
    for nm, R in (('cells uncorrected F', R0), ('cells corrected Fc', R7)):
        tf, tr, sc = topo_summary(R[resp_set], xy, mind=15, maxd=2000)
        print('    %-22s %+10.4f %+14.4f %+10.4f' % (nm, tf, tr, sc))
    s2p = d0['ctx']['s2p']
    print('    per-ROI corr(F, Fneu) median %.3f' % np.nanmedian(_rowcorr(s2p['Frois'].astype(float), s2p['Fneu'].astype(float))))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--geometry', action='store_true')
    ap.add_argument('--pd', action='store_true')
    ap.add_argument('--curly', action='store_true')
    ap.add_argument('--subsample', type=int, default=2000)
    a = ap.parse_args()
    if a.geometry:
        geometry()
    if a.pd:
        pd_checks()
    if a.curly:
        curly_checks(a.subsample)


if __name__ == '__main__':
    main()
