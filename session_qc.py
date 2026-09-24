#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-quality metrics (roadmap item 14) -- one row per session, from what the pipeline already produces, so
that inclusion can LATER be decided on stated thresholds set from the distribution across the catalog (the gate
itself is deliberately not implemented here).

Two tiers:
  LIGHT (always; suite2p ops/stat/F/Fneu + metadata; no stimulus log needed):
    imaging   -- mean-image contrast (p99-p1)/median; 'sharpness' = variance of the Laplacian of the
                 mean-normalised mean image (NB not a conventional calcium-imaging QC metric -- an auxiliary
                 defocus / z-drift proxy; motion, SNR, ROI density, drive and bleaching are the standard ones)
    motion    -- rigid-shift JITTER about the session median (median / 95th pct, px; the DC offset vs the reference
                 is reported separately), non-rigid block shift median, template-correlation median, fraction of
                 suite2p bad frames
    COMPARABILITY caveats: `frac_active_k7`/`nu_snr` are framerate-dependent (<4 Hz undersamples transient peaks and
    biases them low); `img_sharpness_lapvar` is resolution-dependent (use the `_3um` resampled variant across zooms).
    detection -- ROIs total / accepted / analysed (prob>=0 minus zero-variance), density per mm^2, cellprob median
    expression-- median ROI brightness (mean F), fraction of zero-variance ROIs
    snr       -- nu-SNR median (signal_quality.active_mask), fraction active (k=7), median peak dF/F
    stability -- bleaching: fractional change of the session-mean F per minute (linear fit)
    neuropil  -- median per-ROI corr(F, Fneu), median Fneu/F (contamination burden)
  FULL (--full; needs the stimulus log / response table): responsive fraction (classify_responses), median
    split-half tuning reliability, temporal split-half drift gap (responsive set), n conditions / repeats.

Writes/updates tmp/session_qc.csv keyed by session directory name.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/session_qc.py [--full]
                                              [--sessions 'suite2p_results/Cadbury/2022*/*Images*' ...] [--subsample N]
"""
import argparse
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import response_table as rt
import sessionio
import signal_quality as sq

DEFAULT = [('Cadbury', '20221016d', 'ImagesSongFOBonly'), ('Curly', '20231103d', 'ImagesFOBmin'),
           ('Dali', '20230810d', 'ImagesSongFOBonly'), ('Dali', '20230910d', 'ImagesFOBmany')]
OUT = 'tmp/session_qc.csv'


def resolve(animal, date, token):
    hits = [h for h in sorted(glob.glob('suite2p_results/%s/%s/*%s*' % (animal, date, token)))
            if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
    return hits[0] if hits else None


def _nanmed(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if x.size else np.nan


def light_metrics(path):
    md = sessionio.load_metadata(path)
    s2p = sessionio.load_suite2p(path)
    ops, F, Fneu = s2p['ops'], s2p['Frois'].astype(float), s2p['Fneu']
    fr = float(md['framerate'])
    n_fr = F.shape[1]
    res = np.asarray(md['fov']['resolution_umpx'], float).ravel()
    w_um, h_um = float(ops['Lx'] * res[0]), float(ops['Ly'] * res[-1])
    area_mm2 = w_um * h_um / 1e6
    m = {'session': os.path.basename(path.rstrip('/')), 'suite2p_variant': os.path.basename(s2p['path']),
         'fov_w_um': w_um, 'fov_h_um': h_um, 'res_umpx': float(res[0]), 'framerate_hz': fr,
         'n_frames': int(n_fr), 'duration_min': n_fr / fr / 60.0}

    # --- imaging: mean-image contrast + sharpness (auxiliary) ---
    img = np.asarray(ops.get('meanImg'), float)
    p1, p99 = np.percentile(img, [1, 99])
    m['img_contrast'] = float((p99 - p1) / (np.median(img) + 1e-9))
    m['img_sharpness_lapvar'] = float(ndimage.laplace(img / (img.mean() + 1e-9)).var())
    # Laplacian variance depends on pixel sampling, so also report it after block-averaging to ~3 um/px (the
    # coarsest resolution in the catalog) so sessions at different zooms are comparable.
    f = max(1, int(round(3.0 / max(float(res[0]), 1e-6))))
    if f > 1:
        h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
        img3 = img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    else:
        img3 = img
    m['img_sharpness_lapvar_3um'] = float(ndimage.laplace(img3 / (img3.mean() + 1e-9)).var())

    # --- motion (from suite2p registration) ---
    xo, yo = ops.get('xoff'), ops.get('yoff')
    if xo is not None and yo is not None:
        xo, yo = np.asarray(xo, float), np.asarray(yo, float)
        # Motion = JITTER about the session-median shift. A constant offset vs the reference image is not motion
        # (PD's raw median |shift| of 11 px was such a DC offset); it is reported separately.
        r = np.hypot(xo - np.median(xo), yo - np.median(yo))
        m['motion_rigid_med_px'], m['motion_rigid_p95_px'] = float(np.median(r)), float(np.percentile(r, 95))
        m['motion_rigid_offset_px'] = float(np.hypot(np.median(xo), np.median(yo)))
    else:
        m['motion_rigid_med_px'] = m['motion_rigid_p95_px'] = m['motion_rigid_offset_px'] = np.nan
    x1, y1 = ops.get('xoff1'), ops.get('yoff1')
    m['motion_nonrigid_med_px'] = (float(np.median(np.hypot(np.asarray(x1, float), np.asarray(y1, float))))
                                   if x1 is not None and y1 is not None else np.nan)
    m['template_corr_med'] = _nanmed(ops['corrXY']) if ops.get('corrXY') is not None else np.nan
    m['badframes_frac'] = float(np.mean(ops['badframes'])) if ops.get('badframes') is not None else np.nan

    # --- detection / expression ---
    iscell = s2p['iscell']
    m['roi_total'], m['roi_accepted'] = int(iscell.shape[0]), int(iscell[:, 0].sum())
    m['roi_analysed'] = int(len(s2p['cellinds']))
    m['roi_density_per_mm2'] = m['roi_analysed'] / area_mm2
    m['cellprob_med'] = float(np.median(iscell[:, 1]))
    m['frac_zero_variance'] = 1.0 - m['roi_analysed'] / max(1, m['roi_total'])
    m['brightness_medF'] = float(np.median(F.mean(1)))

    # --- SNR / activity (on neuropil-corrected dF/F, the pipeline default) ---
    tr = sessionio.compute_fluorescence_metrics(F, fr, fneu=Fneu, neucoeff=0.7)
    mask, snr, _, _ = sq.active_mask(tr['FdFF'], framerate=fr, method='nu', k=7.0)
    m['nu_snr_med'], m['frac_active_k7'] = _nanmed(snr), float(np.mean(mask))
    m['peak_dff_med'] = _nanmed(np.nanmax(tr['FdFF'], axis=1))

    # --- stability: bleaching as fractional change of the session-mean F per minute ---
    sm = F.mean(0)
    t_min = np.arange(n_fr) / fr / 60.0
    slope = np.polyfit(t_min, sm, 1)[0]
    m['bleach_frac_per_min'] = float(slope / (sm.mean() + 1e-9))

    # --- neuropil burden ---
    if Fneu is not None:
        Fz = F - F.mean(1, keepdims=True); Nz = Fneu - Fneu.mean(1, keepdims=True)
        with np.errstate(all='ignore'):
            cc = (Fz * Nz).sum(1) / np.sqrt((Fz ** 2).sum(1) * (Nz ** 2).sum(1))
            m['cell_fneu_corr_med'] = _nanmed(cc)
            m['fneu_over_f_med'] = _nanmed(Fneu.mean(1) / F.mean(1))
    else:
        m['cell_fneu_corr_med'] = m['fneu_over_f_med'] = np.nan
    return m


def full_metrics(path, subsample):
    ds, ctx = sessionio.build_session_response_table(path, eye_gate_mode='none', equalize_repeats=True,
                                                     roi_subsample=subsample)
    fr = ctx['md']['framerate']
    cls = rt.classify_responses(ds, metric='Fzsc', framerate=fr, alpha=0.05)
    rel = rt.split_half_reliability(ds, 'Fzsc', n_splits=20)
    st = rt.calculate_temporal_split_stability(ds, metric='Fzsc', roi_mask=cls['responsive'])
    return {'n_conditions': int(ds.sizes['condition']), 'n_repeats': int(ds.sizes['repeat']),
            'responsive_frac': float(np.mean(cls['responsive'])), 'selective_frac': float(np.mean(cls['selective'])),
            'reliability_med': _nanmed(rel), 'reliability_med_responsive': _nanmed(rel[cls['responsive']]),
            'temporal_drift_gap': float(st['gap'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', default=None, help='glob(s) of session dirs; default = the 4 raw-data sessions')
    ap.add_argument('--full', action='store_true', help='also compute the stimulus-driven (response-table) metrics')
    ap.add_argument('--subsample', type=int, default=3000, help='ROI subsample for the --full tier on large sessions')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    if a.sessions:
        paths = sorted({p for g in a.sessions for p in glob.glob(g) if os.path.isdir(p) and glob.glob(os.path.join(p, 'suite2p*'))})
    else:
        paths = [p for p in (resolve(*t) for t in DEFAULT) if p]
    rows = []
    for p in paths:
        print('== %s' % os.path.basename(p)[:80])
        try:
            m = light_metrics(p)
            if a.full:
                try:
                    m.update(full_metrics(p, a.subsample))
                except Exception as e:
                    print('   (full tier failed: %s: %s)' % (type(e).__name__, str(e)[:120]))
            rows.append(m)
            print('   ' + ' | '.join('%s=%s' % (k, (('%.3g' % v) if isinstance(v, float) else v))
                                     for k, v in m.items() if k not in ('session',)))
        except Exception as e:
            traceback.print_exc()
            print('   FAILED: %s: %s' % (type(e).__name__, str(e)[:160]))
        sys.stdout.flush()
    if not rows:
        return
    new = pd.DataFrame(rows).set_index('session')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if os.path.exists(a.out):
        old = pd.read_csv(a.out).set_index('session')
        new = new.combine_first(old)     # new values win; rows/columns not recomputed (e.g. full-tier) are kept
    new.to_csv(a.out)
    print('\nwrote %d session rows -> %s' % (len(new), a.out))


if __name__ == '__main__':
    main()
