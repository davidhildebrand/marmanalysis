#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Around the significant scan window of the PD face axis (spatial_structure.py; DH questions 2026-09-27):

 (1) GRADIENT OR BOUNDARY? Radial profiles vs distance from the window centre: mean face-axis z, the smooth-trend
     prediction (Gaussian-kernel regression, sigma = SIGMA_TREND um, fitted WITHOUT the cells within 2R of the
     centre so the cluster cannot inflate its own trend), the RESIDUAL (raw minus trend = what the cluster adds on
     top of the general smooth face preference), and the fraction of face-selective cells (d' > 1). Null band:
     the same profiles around N_CTRL random centre cells. A residual confined to the first annulus = sharp
     boundary; a residual that decays over several annuli = the cluster radiates outward.
 (2) EYE FEATURES? Within the 20 faces, each cell's responses correlated with the faces' pool5-PC0 scores (the
     direction whose saliency is the eyes; midlevel_features.py) -- for the window members vs all gated cells vs
     the 30 most face-selective cells -- plus each member's three best faces. Caveat: with 20 frontal faces of
     one type this can only say whether the members RANK faces along the eye direction, not that eyes drive them.
Figure: output/cluster_profile_pd.png
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/cluster_profile.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import feature_topography_v2 as v2
import neuropil_checks as nc
import response_matrix as rm
import som
import spatial_structure as ss

EDGES = np.array([0, 40, 80, 120, 160, 200, 250, 300, 400, 500], float)
SIGMA_TREND = 150.0
N_CTRL = 300
OUTDIR = 'output'


def radial_profile(vals, dist_from_centre):
    out = np.full(len(EDGES) - 1, np.nan)
    for k in range(len(EDGES) - 1):
        m = (dist_from_centre >= EDGES[k]) & (dist_from_centre < EDGES[k + 1])
        if m.sum() >= 3:
            out[k] = np.nanmean(vals[m])
    return out


def smooth_trend(z, dist, exclude):
    """Gaussian-kernel regression of z over position (leave-one-out), fitted without the ``exclude`` cells."""
    W = np.exp(-dist ** 2 / (2 * SIGMA_TREND ** 2))
    np.fill_diagonal(W, 0)
    W[:, exclude] = 0
    return (W @ z) / (W.sum(1) + 1e-12)


def main():
    rng = np.random.default_rng(0)
    data = nc.build(nc.PD, 0.7)
    zeta = pd.read_csv(nc.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    n = resp.shape[0]
    conds = data['conditions'].reindex(data['condition_ids'])
    ck = np.array([ss.cat_key(c) for c in conds['cat'].to_numpy()])
    names = conds['imagename'].astype(str).to_numpy()
    G = ss.Geometry(xy)
    z = ss.zs(ss.face_axis_map(resp, ck))
    fd = ss.category_dprime(resp, ck, 'face')
    s, ci, R = G.scan(z)
    members = np.flatnonzero(G.dist[ci] <= R)
    print('scan window: stat %.2f at (%.0f, %.0f) um, R=%g, %d members' % (s, xy[ci, 0], xy[ci, 1], R, members.size))

    # ---- (1) radial profiles ----
    dc = G.dist[ci]
    excl = dc <= 2 * R
    trend = smooth_trend(z, G.dist, excl)
    resid = z - trend
    prof = dict(raw=radial_profile(z, dc), trend=radial_profile(trend, dc), resid=radial_profile(resid, dc),
                frac_sel=radial_profile((fd > 1).astype(float), dc))
    ctrl_raw, ctrl_res, ctrl_frac = [], [], []
    for c in rng.choice(n, N_CTRL, replace=False):
        dcc = G.dist[c]
        tr_c = smooth_trend(z, G.dist, dcc <= 2 * R)
        ctrl_raw.append(radial_profile(z, dcc)); ctrl_res.append(radial_profile(z - tr_c, dcc)); ctrl_frac.append(radial_profile((fd > 1).astype(float), dcc))
    ctrl_raw, ctrl_res, ctrl_frac = map(np.array, (ctrl_raw, ctrl_res, ctrl_frac))
    p95_raw, p95_res, p95_frac = (np.nanpercentile(x, 95, axis=0) for x in (ctrl_raw, ctrl_res, ctrl_frac))
    p50_raw, p50_res, p50_frac = (np.nanpercentile(x, 50, axis=0) for x in (ctrl_raw, ctrl_res, ctrl_frac))
    print('\nradial profiles from the window centre (annulus um: n cells | face-z raw / smooth trend / RESIDUAL | frac face-selective d>1) ; '
          'control p95 in brackets')
    for k in range(len(EDGES) - 1):
        m = (dc >= EDGES[k]) & (dc < EDGES[k + 1])
        print('  %4g-%-4g  n=%3d | %+5.2f / %+5.2f / %+5.2f [%+.2f] | %.2f [%.2f]'
              % (EDGES[k], EDGES[k + 1], m.sum(), prof['raw'][k], prof['trend'][k], prof['resid'][k], p95_res[k], prof['frac_sel'][k], p95_frac[k]))
    above = [k for k in range(len(EDGES) - 1) if np.isfinite(prof['resid'][k]) and prof['resid'][k] > p95_res[k]]
    print('  residual exceeds the control p95 in annuli: %s -> the cluster\'s excess extends to ~%g um'
          % (', '.join('%g-%g' % (EDGES[k], EDGES[k + 1]) for k in above), EDGES[max(above) + 1] if above else 0))

    # ---- (2) eye direction within faces ----
    paths = som.condition_image_paths(data)          # images recorded in the session's stimulus log
    images = dnn_som.load_images_rgb(paths)
    model, _ = dnn_som.prep_dnn_model('cpu')
    f5 = dnn_som.extract_dnn_features(model, v2.LAYERS['pool5'], images, 'cpu').numpy()
    sc5, _ = v2.pcs(f5)
    isf = ck == 'face'
    p5 = sc5[:, 0] * np.sign(np.corrcoef(sc5[:, 0], isf.astype(float))[0, 1])
    rf = np.nan_to_num(resp[:, isf])
    eye_r = np.array([np.corrcoef(rf[i], p5[isf])[0, 1] if rf[i].std() > 0 else np.nan for i in range(n)])
    top30 = np.argsort(-fd)[:30]
    print('\nwithin-face correlation of responses with the pool5-PC0 (eye) direction across the %d faces:' % isf.sum())
    print('  all gated cells: median %+.2f (IQR %+.2f .. %+.2f) | top-30 face-selective: median %+.2f | window members: median %+.2f'
          % (np.nanmedian(eye_r), np.nanpercentile(eye_r, 25), np.nanpercentile(eye_r, 75), np.nanmedian(eye_r[top30]), np.nanmedian(eye_r[members])))
    print('  member  face-dprime  eye-r   best three faces')
    for c in members:
        order = np.argsort(-rf[c])[:3]
        print('  %5d   %+5.2f      %+.2f   %s' % (c, fd[c], eye_r[c], ', '.join(names[isf][order])))
    print('  (image-level: r(pool5-PC0, mean face response of members) = %+.2f; of all top-30 = %+.2f)'
          % (np.corrcoef(p5[isf], rf[members].mean(0))[0, 1], np.corrcoef(p5[isf], rf[top30].mean(0))[0, 1]))

    # ---- figure ----
    os.makedirs(OUTDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    box = 300
    ax[0].scatter(xy[:, 0], xy[:, 1], c=z, cmap='coolwarm', vmin=-2, vmax=2, s=30)
    ax[0].scatter(xy[members, 0], xy[members, 1], facecolors='none', edgecolors='k', s=90, linewidths=1.2)
    for e in EDGES[1:6]:
        ax[0].add_patch(Circle(xy[ci], e, fill=False, ec='0.4', ls=':', lw=0.8))
    ax[0].set_xlim(xy[ci, 0] - box, xy[ci, 0] + box); ax[0].set_ylim(xy[ci, 1] - box, xy[ci, 1] + box); ax[0].set_aspect('equal')
    ax[0].set_title('face-axis z around the window (members outlined; rings = annuli)', fontsize=9)
    ctr = 0.5 * (EDGES[1:] + EDGES[:-1])
    ax[1].plot(ctr, prof['raw'], '-o', color='tab:red', label='raw face-z')
    ax[1].plot(ctr, prof['trend'], '-s', color='tab:orange', label='smooth trend (sigma %g um, cluster excluded)' % SIGMA_TREND)
    ax[1].plot(ctr, prof['resid'], '-^', color='k', label='residual = raw - trend')
    ax[1].fill_between(ctr, np.nanpercentile(ctrl_res, 5, axis=0), p95_res, color='0.6', alpha=0.3, label='residual around %d random centres, p5-p95' % N_CTRL)
    ax[1].axhline(0, color='k', lw=0.5); ax[1].set_xlabel('distance from window centre (um)'); ax[1].set_ylabel('mean face-axis z'); ax[1].legend(fontsize=7)
    ax[1].set_title('gradient or boundary? residual excess vs distance', fontsize=9)
    ax[2].plot(ctr, prof['frac_sel'], '-o', color='tab:red', label='fraction face-selective (d-prime above 1)')
    ax[2].fill_between(ctr, np.nanpercentile(ctrl_frac, 5, axis=0), p95_frac, color='0.6', alpha=0.3, label='random centres p5-p95')
    ax[2].plot(ctr, p50_frac, ':', color='0.3', label='random centres median')
    ax[2].set_xlabel('distance from window centre (um)'); ax[2].set_ylabel('fraction'); ax[2].legend(fontsize=7)
    ax[2].set_title('fraction of face-selective cells per annulus', fontsize=9)
    ax[3].hist(eye_r[np.isfinite(eye_r)], bins=30, color='0.7', label='all gated cells')
    for c in members:
        ax[3].axvline(eye_r[c], color='tab:red' if fd[c] > 1 else 'tab:blue', lw=1.2)
    ax[3].axvline(np.nanmedian(eye_r[top30]), color='k', ls='--', label='top-30 face-selective median')
    ax[3].set_xlabel('within-face corr with pool5-PC0 (eye direction)'); ax[3].legend(fontsize=7)
    ax[3].set_title('members (red = face-selective, blue = not) vs population', fontsize=9)
    fig.suptitle('PD face-axis scan window: is it a sharp island or the peak of a gradient, and do its cells rank faces along the eye direction?', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'cluster_profile_pd.png'), dpi=120)
    print('figure: output/cluster_profile_pd.png')


if __name__ == '__main__':
    main()
