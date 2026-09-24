#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the ONE lead from feature_topography.py: the preference map for relu7 image-feature PC2 showed a
~198 um periodicity on PD (p=0.002, q=0.050 -- borderline, 1 of 20 features; its smooth autocorrelation a~0.01,
i.e. a periodic-only, column-like signature with no gradient). Before believing it:
  (a) WHAT is feature-2 -- which images/categories sit at its extremes;
  (b) tighter p (n_perm=2000) and early-vs-late SPLIT-HALF stability of the feature-2 map on PD;
  (c) REPLICATION: Dali 20230810d SongFOBonly (the SAME 60 images -> identical feature axis, 2.6 mm FOV), and if
      their image dirs resolve, Curly FOBmin / Dali FOBmany by projecting THEIR images onto PD's feature-2
      direction (fixed axis across sessions);
  (d) the Cell-2025-style embedding-distance vs physical-distance test done RIGHT: each ROI's tuning z-scored so
      Euclidean ~ correlation distance (the unnormalised run conflated per-cell gain with tuning shape).
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/feature2_verify.py
"""
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import response_matrix as rm
import response_table as rt
import som
import topography as tg

PD = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
      'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
STIM_PD = 'stimuli/Song_etal_Wang_2022_NatCommun/480288_equalized_RGBA_FOBonly'
ZETA_CSV = 'tmp/responsiveness_pd.csv'
LARGE = [('Dali', '20230810d', 'ImagesSongFOBonly'),
         ('Curly', '20231103d', 'ImagesFOBmin'),
         ('Dali', '20230910d', 'ImagesFOBmany')]
FEAT = 2
MIND = 15.0
_MODEL = {}


def rowcorr(A, v):
    A, v = np.asarray(A, float), np.asarray(v, float)
    Az, vz = A - np.nanmean(A, 1, keepdims=True), v - v.mean()
    with np.errstate(all='ignore'):
        return np.nansum(Az * vz, 1) / np.sqrt(np.nansum(Az ** 2, 1) * (vz ** 2).sum())


def resolve(animal, date, token):
    hits = [h for h in sorted(glob.glob('suite2p_results/%s/%s/*%s*' % (animal, date, token)))
            if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
    return hits[0] if hits else None


def resolve_stim_dir(imagenames):
    """Find the stimulus dir holding a session's images by locating the first few filenames under stimuli/."""
    for nm in list(imagenames)[:3]:
        hits = glob.glob('stimuli/**/' + os.path.basename(str(nm)), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def relu7_for(paths):
    if 'm' not in _MODEL:
        _MODEL['m'], _MODEL['l'] = dnn_som.prep_dnn_model('cpu')
    images = dnn_som.load_images_rgb(paths)
    return dnn_som.extract_dnn_features(_MODEL['m'], _MODEL['l'], images, 'cpu').numpy()


def znorm(M):
    M = np.nan_to_num(np.asarray(M, float))
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def dist_vs_phys(coords, xy, maxd, label):
    de, dp = pdist(np.nan_to_num(np.asarray(coords, float))), pdist(np.asarray(xy, float))
    ok = (dp >= MIND) & (dp <= maxd)
    print('      D_%-24s vs D_phys: Spearman r=%+.3f' % (label, spearmanr(de[ok], dp[ok]).statistic))


def main():
    data = rm.build_response_matrix(PD, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    conds = data['conditions'].reindex(data['condition_ids'])
    cats = np.array([str(c) for c in conds['cat'].to_numpy()])
    names = conds['imagename'].to_numpy()
    r7 = relu7_for(som.condition_image_paths(data, STIM_PD))
    mu = r7.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(r7 - mu, full_matrices=False)
    feat, fev = U[:, :20] * S[:20], (S[:20] ** 2) / (S ** 2).sum()
    f2 = feat[:, FEAT]
    print('PD | ZETA n=%d | verifying relu7 feature-PC%d (fev=%.3f)' % (resp.shape[0], FEAT, fev[FEAT]))

    # (a) what is feature-2?
    o = np.argsort(f2)
    print('  (a) feature-%d LOW  end: %s' % (FEAT, [(str(names[i])[:24], cats[i]) for i in o[:6]]))
    print('      feature-%d HIGH end: %s' % (FEAT, [(str(names[i])[:24], cats[i]) for i in o[-6:]]))
    print('      category means on feature-%d: %s'
          % (FEAT, {c: round(float(f2[cats == c].mean()), 2) for c in pd.unique(cats)}))

    # (b) tighter p + split-half stability
    pref = rowcorr(resp, f2)
    res = tg.scalar_map_periodicity(pref, xy, n_grid=24, n_perm=2000, min_wavelength_um=30)[0]
    print('  (b) PD feature-%d map: a=%.2f lam=%.0fum | periodicity peak_wl=%.0fum  p=%.4f (n_perm=2000)'
          % (FEAT, res['amplitude'], res['lambda_um'], res['peak_wavelength_um'], res['p']))
    tr = rt.trial_response(d['ds'], 'Fzsc').transpose('roi', 'condition', 'repeat').values[zeta]
    h = tr.shape[2] // 2
    pe = rowcorr(np.nanmean(tr[:, :, :h], 2), f2)
    pl = rowcorr(np.nanmean(tr[:, :, tr.shape[2] - h:], 2), f2)
    ok = np.isfinite(pe) & np.isfinite(pl)
    print('      split-half feature-%d map corr (early vs late trials) = %.3f' % (FEAT, np.corrcoef(pe[ok], pl[ok])[0, 1]))
    re_ = tg.scalar_map_periodicity(pe, xy, n_grid=24, n_perm=500, min_wavelength_um=30)[0]
    rl_ = tg.scalar_map_periodicity(pl, xy, n_grid=24, n_perm=500, min_wavelength_um=30)[0]
    print('      periodicity early-half: wl=%.0fum p=%.3f | late-half: wl=%.0fum p=%.3f'
          % (re_['peak_wavelength_um'], re_['p'], rl_['peak_wavelength_um'], rl_['p']))

    # (d) embedding distance vs physical, done right (z-scored tuning)
    print('  (d) embedding-distance vs physical (z-scored tuning => Euclidean ~ correlation distance; 15-600 um)')
    Z = znorm(resp)
    Uz, Sz, _ = np.linalg.svd(Z, full_matrices=False)
    dist_vs_phys(Uz[:, :20] * Sz[:20], xy, 600.0, 'neuralPC20 (znorm)')
    dist_vs_phys(Uz[:, :3] * Sz[:3], xy, 600.0, 'neuralPC3 (znorm)')
    dist_vs_phys(Z, xy, 600.0, 'fullTuning (znorm)')
    dist_vs_phys(znorm(np.column_stack([rowcorr(resp, feat[:, k]) for k in range(20)])), xy, 600.0,
                 'featurePref20 (znorm)')
    sys.stdout.flush()

    # (c) replication on the large-FOV sessions, same feature-2 direction
    for animal, date, token in LARGE:
        path = resolve(animal, date, token)
        print('\n=== (c) replication: %s / %s / %s ===' % (animal, date, token))
        if path is None:
            print('  (no session dir)'); continue
        try:
            dd = rm.build_response_matrix(path, denoise=False, reliability_splits=3, eye_gate_mode='none',
                                          neuropil_subtract=True, neucoeff=0.7, roi_subsample=2000,
                                          equalize_repeats=True)
            nm = dd['conditions'].reindex(dd['condition_ids'])['imagename'].to_numpy()
            sd = STIM_PD if token == 'ImagesSongFOBonly' else resolve_stim_dir(nm)
            if sd is None:
                print('  (stimulus dir for %s not found under stimuli/; skipping)' % token); continue
            pths = [os.path.join(sd, str(x)) for x in nm]
            keep = np.array([os.path.exists(p) for p in pths])   # conditions with an image file (drops e.g. blank)
            if keep.sum() < 0.8 * len(pths):
                print('  (%d/%d image files missing under %s; skipping)' % (int((~keep).sum()), len(pths), sd)); continue
            if not keep.all():
                print('  dropping %d condition(s) with no image file (e.g. blank): %s'
                      % (int((~keep).sum()), [str(nm[i])[:20] for i in np.where(~keep)[0]]))
            f2s = (relu7_for([p for p, k in zip(pths, keep) if k]) - mu) @ Vt[FEAT]   # coords on PD's feature-2
            rmask = rt.classify_responses(dd['ds'], metric='Fzsc', framerate=dd['framerate'], alpha=0.05)['responsive']
            fov = dd['roi_xy_um'].max(0) - dd['roi_xy_um'].min(0)
            print('  stim dir: %s | %d images | FOV~%.0fx%.0f um | subsample n=%d, responsive=%d'
                  % (sd, len(pths), fov[0], fov[1], dd['response'].shape[0], int(rmask.sum())))
            for lab, m in (('all-subsample', np.ones(dd['response'].shape[0], bool)), ('responsive', rmask)):
                sub = rm.apply_roi_mask(dd, m)
                pf = rowcorr(sub['response'][:, keep], f2s)
                rr = tg.scalar_map_periodicity(pf, sub['roi_xy_um'], n_grid=28, n_perm=1000, min_wavelength_um=30)[0]
                print('  %-14s n=%4d  feature-%d map a=%.2f lam=%4.0fum | periodicity peak_wl=%5.0fum  p=%.3f'
                      % (lab, sub['response'].shape[0], FEAT, rr['amplitude'], rr['lambda_um'],
                         rr['peak_wavelength_um'], rr['p']))
        except Exception as e:
            traceback.print_exc()
            print('  FAILED:', type(e).__name__, str(e)[:200])
        sys.stdout.flush()


if __name__ == '__main__':
    main()
