#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-outs for the suite2p settings sweep (roadmap item 1): the same metrics for every extraction variant of one
session, one CSV row per variant, so detection / neuropil settings can be compared on what matters downstream.

Per variant (the variant is a suite2p_* folder of the session or a detection-only folder made by suite2p_sweep.py):
  detection   ROIs total / accepted / analysed, density per mm^2, median cellpose probability, median detected ROI
              radius (um)                                                   [session_qc.light_metrics]
  neuropil    median cell<->Fneu trace correlation, median Fneu/F, ring geometry: inner / outer ring distance from
              the ROI centroid (um), ring pixels, fraction of the ring occupied by other ROIs
  signal      nu-SNR median, fraction active (k=7), median peak dF/F, bleaching slope
  drive       responsive / selective fractions (ANOVA gates, alpha 0.05 / 0.01), ZETA fraction (--zeta; slow),
              median split-half tuning reliability (all / responsive)          [response_matrix + response_table]
  topography  for the gates all / responsive (/ zeta): face-axis T (near < 75 um minus far > 150 um z-product of
              the neural-PC0 map) against the RANDOM-PROJECTION null (median, p95, p of the face axis), face-d' T,
              full-tuning topography score (Spearman of tuning similarity vs -distance, 15-600 um), and the SCAN
              statistic of the face-d' map (most anomalous window, radii 40-100 um) with its p against random
              projections and its mark-correlation at 0-25 um vs the null p95   [feature_specificity, spatial_structure]
All response-side numbers use the pipeline defaults the PD analyses used: Fzsc, neuropil correction 0.7, eye gate
'landing_window'. Writes tmp/suite2p_sweep/<slug>_readouts_<UTC>.csv (and updates <slug>_readouts_latest.csv, keyed by
variant) and prints a comparison table.
Run:  .venv/bin/python marmanalysis/suite2p_sweep_readouts.py \\
          [--session <session dir>] [--variants <dir or name> ...] [--zeta] [--n-rand 500] [--n-perm 500]
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_specificity as fs
import feature_topography_v2 as v2
import paths
import response_matrix as rm
import response_table as rt
import session_qc
import sessionio
import spatial_structure as ss
import topography as tg

PD = v2.SESSION
OUTDIR = paths.tmp('suite2p_sweep')
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')
KEY_COLS = ['roi_total', 'roi_accepted', 'roi_analysed', 'roi_radius_um_med', 'ring_outer_um_med', 'ring_occupied_frac_med',
            'cell_fneu_corr_med', 'fneu_over_f_med', 'responsive_frac', 'selective_frac_01', 'zeta_frac',
            'reliability_med_responsive', 'T_face_all', 'T_rand_p95_all', 'p_face_vs_rand_all', 'scan_face_all',
            'p_scan_vs_rand_all', 'T_face_responsive', 'p_face_vs_rand_responsive']


def discover_variants(session):
    """Default variant set: the session's suite2p_* folders plus every sweep folder with a plane0."""
    out = [d for d in sorted(glob.glob(os.path.join(session, 'suite2p*'))) if os.path.isdir(os.path.join(d, 'plane0'))]
    out += [d for d in sorted(glob.glob(os.path.join(session, 'sweep_suite2p', '*'))) if os.path.isdir(os.path.join(d, 'plane0'))]
    return out


def resolve_variant(session, v):
    for cand in (v, os.path.join(session, v), os.path.join(session, 'sweep_suite2p', v)):
        if os.path.isdir(os.path.join(cand, 'plane0')):
            return os.path.normpath(cand)
    return v                      # a regex on the suite2p_* folder names; sessionio resolves it


def category_keys(data):
    conds = data['conditions']
    ids = np.asarray(data['condition_ids'])
    cats = None
    try:
        cats = conds.reindex(ids)['cat'].to_numpy()
        if pd.isna(cats).all():
            cats = None
    except Exception:
        cats = None
    if cats is None:
        if len(conds) != len(ids):
            raise RuntimeError('cannot align condition metadata (%d rows) with the response columns (%d)' % (len(conds), len(ids)))
        cats = conds['cat'].to_numpy()
    return np.array([ss.cat_key(c) for c in cats])


def chunked_T(Z_maps, iu, near, far, chunk=50):
    """v2.topo_stat over many maps without materialising an (n_pairs x K) product for all K at once."""
    out = []
    for k0 in range(0, Z_maps.shape[1], chunk):
        out.append(v2.topo_stat(Z_maps[:, k0:k0 + chunk], iu, near, far))
    return np.concatenate(out)


def topography_block(resp, xy, ck, n_rand, n_perm, seed=0):
    """Face-axis / face-d' T vs random projections, the full-tuning topography score, and the face-d' scan statistic
    (+ its mark-correlation at 0-25 um) vs random projections. All on the ROIs passed in."""
    n = resp.shape[0]
    out = {'n': int(n)}
    if n < 30:
        return out
    dist = tg.pairwise_distance(xy)
    iu = np.triu_indices(n, 1)
    dd = dist[iu]
    near, far = dd < v2.NEAR, dd > v2.FAR
    R = np.nan_to_num(resp - np.nanmean(resp, 0, keepdims=True))
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((R.shape[1], n_rand)); U /= np.linalg.norm(U, axis=0, keepdims=True)
    t_rand = chunked_T(fs.zcols(R @ U), iu, near, far)
    face_axis = ss.face_axis_map(resp, ck)
    face_dp = ss.category_dprime(resp, ck, 'face')
    t_obs = fs.T_of(np.column_stack([face_axis, face_dp]), iu, near, far)
    out.update(T_face=float(t_obs[0]), T_facedp=float(t_obs[1]), T_rand_med=float(np.median(t_rand)),
               T_rand_p95=float(np.percentile(t_rand, 95)), T_rand_p99=float(np.percentile(t_rand, 99)),
               p_face_vs_rand=float((1 + (t_rand >= t_obs[0]).sum()) / (1 + t_rand.size)),
               p_facedp_vs_rand=float((1 + (t_rand >= t_obs[1]).sum()) / (1 + t_rand.size)))
    sim = tg.pairwise_tuning_similarity(resp)[iu]
    ok = (dd >= v2.MIND) & (dd <= v2.MAXD) & np.isfinite(sim)
    out['topo_score'] = float(spearmanr(sim[ok], -dd[ok]).statistic) if ok.sum() > 10 else np.nan
    face_r, pc0_var = ss.pc0_face_r(resp, ck)
    out.update(pc0_face_r=face_r, pc0_var=pc0_var)
    # scan statistic + mark-correlation of the face-d' map vs random projections (local statistics, pairs <= 300 um)
    G = ss.Geometry(xy, ss.RMAX)
    z = ss.zs(face_dp)
    scan_obs, scan_i, scan_r = G.scan(z)
    mc_obs = G.mark_corr(z)
    Up = rng.standard_normal((R.shape[1], n_perm)); Up /= np.linalg.norm(Up, axis=0, keepdims=True)
    scan_null, mc_null = [], []
    P = R @ Up
    for k in range(n_perm):
        zk = ss.zs(P[:, k])
        scan_null.append(G.scan(zk)[0])
        mc_null.append(G.mark_corr(zk))
    scan_null = np.array(scan_null); mc_null = np.array(mc_null)
    out.update(scan_face=float(scan_obs), scan_face_r_um=float(scan_r), scan_face_n_in=int(G.n_in[scan_r][scan_i]) if scan_i >= 0 else 0,
               scan_face_x_um=float(xy[scan_i, 0]) if scan_i >= 0 else np.nan, scan_face_y_um=float(xy[scan_i, 1]) if scan_i >= 0 else np.nan,
               p_scan_vs_rand=ss.pval(scan_null, scan_obs), scan_rand_med=float(np.median(scan_null)),
               k25_face=float(mc_obs[0]), k25_rand_p95=float(np.nanpercentile(mc_null[:, 0], 95)))
    return out


def readout(session, variant, n_rand, n_perm, zeta, reliability_splits, neucoeff=None):
    """One CSV row of read-outs for `variant`. `neucoeff` overrides the pipeline's neuropil coefficient (0 = no
    correction) for the response-side and topography numbers; the row is then named '<variant>__nc<value>'."""
    t0 = time.time()
    row = {'variant': os.path.basename(os.path.normpath(variant)) if os.path.isdir(variant) else variant,
           'variant_path': variant, 'session': sessionio.session_slug(session) if hasattr(sessionio, 'session_slug') else os.path.basename(session)}
    npil = {}
    if neucoeff is not None:
        npil = {'neucoeff': float(neucoeff), 'neuropil_subtract': float(neucoeff) != 0}
        row['variant'] += '__nc%g' % neucoeff
        row['neucoeff_override'] = float(neucoeff)
    vj = os.path.join(variant, 'variant.json') if os.path.isdir(variant) else None
    if vj and os.path.isfile(vj):
        with open(vj) as f:
            info = json.load(f)
        row['suite2p_version'] = info.get('suite2p_version')
        row['settings_changed'] = json.dumps(info.get('settings_changed_vs_anchor', {}), default=str)
    light = session_qc.light_metrics(session, variant=variant)
    row['suite2p_version'] = row.get('suite2p_version') or str(light.get('suite2p_version', ''))
    light.pop('session', None)
    row.update(light)
    s2p = sessionio.load_suite2p(session, variant=variant)
    res = float(np.asarray(sessionio.load_metadata(session)['fov']['resolution_umpx'], float).ravel()[0])
    row['roi_radius_um_med'] = float(np.median([np.sqrt(len(r['xpix']) / np.pi) for r in s2p['ROIs']])) * res
    row['suite2p_version'] = row['suite2p_version'] or str(s2p['ops'].get('suite2p_version', ''))

    data = rm.build_response_matrix(session, variant=variant, denoise=False, reliability_splits=reliability_splits, **npil)
    row.update(neuropil_subtract=bool(data['neuropil_subtract']), neucoeff=float(data['neucoeff']))
    fr = data['framerate']
    cls = rt.classify_responses(data['ds'], metric='Fzsc', framerate=fr, alpha=0.05)
    responsive = np.asarray(cls['responsive'], bool)
    selective = np.asarray(cls['selective'], bool)
    rel = np.asarray(data['reliability'], float)
    row.update(n_roi_table=int(responsive.size), n_responsive=int(responsive.sum()), responsive_frac=float(responsive.mean()),
               n_selective=int(selective.sum()), selective_frac=float(selective.mean()),
               reliability_med=float(np.nanmedian(rel)),
               reliability_med_responsive=float(np.nanmedian(rel[responsive])) if responsive.any() else np.nan,
               n_trials_excluded=data['n_trials_excluded'])
    p_sel = np.asarray(cls['p_selective'], float)
    row['selective_frac_01'] = float(np.mean((p_sel < 0.01) & np.isfinite(p_sel)))   # the CLAIM-level alpha (roadmap item 2)
    gates = {'all': np.ones(responsive.size, bool), 'responsive': responsive}
    if zeta:
        import compare_responsiveness as cr
        ctx = data['ctx']
        p_zeta = cr.zeta_pvalues(ctx, 'Fzsc', ctx['n_samp_isi'] + ctx['n_samp_stim'])
        zmask = p_zeta < 0.05
        row.update(zeta_frac=float(np.nanmean(zmask)), n_zeta=int(zmask.sum()))
        gates['zeta'] = zmask
        pd.DataFrame({'roi': np.arange(p_zeta.size), 'p_zeta': p_zeta}).to_csv(
            os.path.join(variant if os.path.isdir(variant) else OUTDIR, 'zeta_pvalues.csv'), index=False)
    ck = category_keys(data)
    for gname, mask in gates.items():
        d = rm.apply_roi_mask(data, mask)
        blk = topography_block(d['response'], d['roi_xy_um'], ck, n_rand, n_perm)
        row.update({'%s_%s' % (k, gname): v for k, v in blk.items()})
    row['readout_s'] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default=PD)
    ap.add_argument('--variants', nargs='*', default=None, help='variant dirs / names / folder regexes [default: all suite2p_* and sweep_suite2p/* of the session]')
    ap.add_argument('--n-rand', type=int, default=500, help='random projections for the T null')
    ap.add_argument('--n-perm', type=int, default=500, help='random projections for the scan / mark-correlation null')
    ap.add_argument('--zeta', action='store_true', help='also compute the ZETA gate per variant (slow)')
    ap.add_argument('--reliability-splits', type=int, default=20)
    ap.add_argument('--neucoeff', type=float, nargs='*', default=None,
                    help='neuropil coefficient(s) to use instead of the pipeline default 0.7 (0 = uncorrected); one row '
                         'per variant and value, named <variant>__nc<value>')
    ap.add_argument('--outdir', default=OUTDIR)
    a = ap.parse_args()
    session = a.session.rstrip('/')
    variants = [resolve_variant(session, v) for v in a.variants] if a.variants else discover_variants(session)
    jobs = [(v, nc) for v in variants for nc in (a.neucoeff if a.neucoeff else [None])]
    slug = sessionio.session_slug(session) if hasattr(sessionio, 'session_slug') else os.path.basename(session)
    os.makedirs(a.outdir, exist_ok=True)
    print('session %s: %d variants' % (slug, len(variants)))
    stamped = os.path.join(a.outdir, '%s_readouts_%s.csv' % (slug, STAMP))
    rows = []
    for v, nc in jobs:
        print('\n== %s%s' % (v, '' if nc is None else '  [neucoeff %g]' % nc))
        try:
            rows.append(readout(session, v, a.n_rand, a.n_perm, a.zeta, a.reliability_splits, neucoeff=nc))
            r = rows[-1]
            print('   ' + ' | '.join('%s=%s' % (k, ('%.3g' % r[k]) if isinstance(r[k], float) else r[k]) for k in KEY_COLS if k in r))
            pd.DataFrame(rows).set_index('variant').to_csv(stamped)      # incremental: a crash keeps the finished rows
        except Exception as exc:
            traceback.print_exc()
            print('   FAILED: %s: %s' % (type(exc).__name__, str(exc)[:200]))
        sys.stdout.flush()
    if not rows:
        return
    df = pd.DataFrame(rows).set_index('variant')
    df.to_csv(stamped)
    latest = os.path.join(a.outdir, '%s_readouts_latest.csv' % slug)
    if os.path.exists(latest):
        old = pd.read_csv(latest).set_index('variant')
        df = df.combine_first(old)
    df.to_csv(latest)
    cols = [c for c in KEY_COLS if c in df.columns]
    with pd.option_context('display.width', 250, 'display.max_columns', 60, 'display.float_format', lambda x: '%.3g' % x):
        print('\n' + df[cols].to_string())
    print('\nwrote %s and %s' % (stamped, latest))


if __name__ == '__main__':
    main()
