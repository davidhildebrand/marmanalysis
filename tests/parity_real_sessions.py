#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-session parity check: response_table.py (xarray) vs the structured-array windowing.

For each session this:
  1. loads real suite2p fluorescence and computes F0 / dF/F / z-score exactly as
     analysis_for_images.py:984-994 (filters.calculate_baselines, method='medianbw');
  2. loads the real stimlog (CSV variant read directly, or older text .log via
     parsers.parse_log_stim_image) and derives n_samp_isi / n_samp_stim with the bincount
     logic of analysis_for_images.py:1314-1325;
  3. windows the traces into (n_conds, n_ROIs, n_reps, n_samp_trial) with the VERBATIM
     structured-array loop (analysis_for_images.py:1604-1657) -> the reference;
  4. builds the xarray container with response_table.build_response_table on the same inputs;
  5. asserts the windowed tensors match, and that the stim-window response matches both the
     reference tensor AND an INDEPENDENT per-condition computation straight from the traces.

This deliberately does NOT exercise the image-name->category parsing (d'/FSI are pure functions
of resp_vect_cond + category labels, already proven equal in test_response_table.py) nor the
CSV stim_mode/stim_class filter (broken for this CSV variant; a no-op for pure-image sessions).

Run:  ../.venv/bin/python marmanalysis/tests/parity_real_sessions.py
"""

import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filters
import parsers
import response_table as rt

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'
METRICS = ['FdFF', 'Fzsc', 'F0', 'Fraw']

SESSIONS = [
    ('Cadbury', '20231007d',
     '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'),
    ('Cadbury', '20221016d',
     '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'),
    ('Dali', '20230810d',
     '150108tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesSongFOBonly'),
]


def load_session(animal, date, session):
    """Reproduce the deterministic numeric loading of analysis_for_images.py for one session."""
    sp = os.path.join(BASE, animal, date, session)

    md_files = glob.glob(os.path.join(sp, '*_metadata.pickle'))
    with open(md_files[0], 'rb') as f:
        md = pickle.load(f)
    framerate = md['framerate']
    stim_locked = md.get('stim_locked_to_acqfr', True)

    # --- stimlog: CSV read directly (this variant lacks stim_mode/stim_class), else text log
    csvs = glob.glob(os.path.join(sp, '*_stimlog.csv'))
    logs = [f for f in glob.glob(os.path.join(sp, '*.log')) if 'disptimes' not in f]
    if csvs:
        stimlog = pd.read_csv(csvs[0])
        stimlog_src = 'csv'
    else:
        stimlog = parsers.parse_log_stim_image(open(logs[0]).read())
        stimlog_src = 'text-log'
    stimlog = stimlog.reset_index(drop=True)

    # --- suite2p (analysis_for_images.py:938-975)
    s2p = sorted(glob.glob(os.path.join(sp, 'suite2p*')))[0]
    plane = os.path.join(s2p, 'plane0')
    iscell = np.load(os.path.join(plane, 'iscell.npy'))
    F = np.load(os.path.join(plane, 'F.npy'))
    cellinds = np.where(iscell[:, 1] >= 0.0)[0]
    inactives = np.where(np.std(F, axis=1) == 0)[0]
    cellinds = np.setdiff1d(cellinds, inactives)
    Frois = F[cellinds]
    n_ROIs, n_frames = Frois.shape

    # --- F0, dF/F, z-score (analysis_for_images.py:985-994)
    F0 = filters.calculate_baselines(Frois, framerate=framerate, window=60, method='medianbw')
    FdFF_raw = (Frois - F0) / F0
    Fzsc_raw = ((Frois - F0 - np.mean(Frois - F0, axis=1)[:, np.newaxis])
                / np.std(Frois - F0, axis=1)[:, np.newaxis])
    traces = {'FdFF': FdFF_raw, 'Fzsc': Fzsc_raw, 'F0': F0, 'Fraw': Frois}

    return md, framerate, stim_locked, stimlog, stimlog_src, traces, n_ROIs, n_frames


def derive_timing(stimlog, stim_locked):
    """n_samp_isi / n_samp_stim per analysis_for_images.py:1314-1325 (after acqfr correction)."""
    stimlog = stimlog.copy()
    for ak in [c for c in stimlog.columns if 'acqfr' in c]:           # off-by-one fix (1268-1271)
        stimlog[ak] = stimlog[ak] - 1

    stim_span = (stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).dropna().astype(int)
    isi_span = (stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).dropna().astype(int)
    if stim_locked:
        n_samp_stim = np.bincount(stim_span).argmax()
    else:
        nz = np.bincount(stim_span).nonzero()[0]
        n_samp_stim = nz[0] if nz[0] != 0 else nz[1]
    nz = np.bincount(isi_span).nonzero()[0]
    n_samp_isi = nz[0] if nz[0] != 0 else nz[1]
    return stimlog, int(n_samp_isi), int(n_samp_stim)


def trim_in_bounds(stimlog, n_samp_isi, n_samp_stim, n_frames):
    """Drop trials with null/out-of-range onsets so every trial window fits (cf. 1274-1303)."""
    onset = stimlog['acqfr_stim_i']
    fr_end = onset + n_samp_stim + n_samp_isi
    keep = onset.notnull() & (fr_end <= n_frames) & (onset >= 0)
    return stimlog[keep].reset_index(drop=True)


def reference_windowing(traces, stimlog, n_samp_isi, n_samp_stim):
    """Verbatim structured-array windowing (analysis_for_images.py:1604-1657), metrics only."""
    n_ROIs = traces['Fraw'].shape[0]
    conditions = np.unique(stimlog['cond'].values)
    n_conds = conditions.size
    n_reps = int(stimlog['cond'].value_counts().iloc[0])
    n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi
    ref = {m: np.full((n_conds, n_ROIs, n_reps, n_samp_trial), np.nan) for m in METRICS}
    for ci, c in enumerate(conditions):
        sub = stimlog[stimlog['cond'] == c]
        for t in range(n_reps):
            fr_start = int(sub.iloc[t]['acqfr_stim_i'] - n_samp_isi)
            fr_end = int(sub.iloc[t]['acqfr_stim_i'] + n_samp_stim + n_samp_isi)
            if fr_start < 0 and t == 0:
                miss = abs(fr_start)
                for m in METRICS:
                    ref[m][ci, :, t, 0:miss] = np.repeat(traces[m][:, 0:1], miss, axis=1)
                    ref[m][ci, :, t, miss:n_samp_trial] = traces[m][:, 0:fr_end]
                continue
            for m in METRICS:
                ref[m][ci, :, t, :] = traces[m][:, fr_start:fr_end]
    return ref, conditions, n_reps


def independent_resp_vect(traces, stimlog, n_samp_isi, n_samp_stim, metric):
    """Per-(roi, condition) stim-window mean computed directly from traces (independent of the
    windowing loop and of the container), as a cross-check on resp_vect_cond."""
    conditions = np.unique(stimlog['cond'].values)
    n_ROIs = traces[metric].shape[0]
    out = np.full((n_ROIs, conditions.size), np.nan)
    for ci, c in enumerate(conditions):
        onsets = stimlog.loc[stimlog['cond'] == c, 'acqfr_stim_i'].astype(int).values
        per_trial = []
        for o in onsets:
            s = o + 0          # stim window starts at onset (= fr_start + n_samp_isi)
            per_trial.append(traces[metric][:, s:s + n_samp_stim].mean(axis=1))
        out[:, ci] = np.mean(per_trial, axis=0)
    return out


def check_session(animal, date, session):
    print('\n' + '=' * 78)
    print('{} / {} / {}'.format(animal, date, session[:40]))
    md, fr, stim_locked, stimlog, src, traces, n_ROIs, n_frames = load_session(animal, date, session)
    stimlog, n_samp_isi, n_samp_stim = derive_timing(stimlog, stim_locked)
    stimlog = trim_in_bounds(stimlog, n_samp_isi, n_samp_stim, n_frames)

    counts = stimlog['cond'].value_counts()
    n_conds = stimlog['cond'].nunique()
    print('  stimlog={}  framerate={:.3f}  n_ROIs={}  n_frames={}'.format(src, fr, n_ROIs, n_frames))
    print('  n_samp_isi={}  n_samp_stim={}  n_conds={}  reps={}'.format(
        n_samp_isi, n_samp_stim, n_conds, (int(counts.min()), int(counts.max()))))
    if counts.nunique() != 1:
        print('  SKIP: ragged repeats (aborted/partial session) -> ragged support is future work.')
        return None

    ref, conditions, n_reps = reference_windowing(traces, stimlog, n_samp_isi, n_samp_stim)
    ds = rt.build_response_table(traces, stimlog, n_samp_isi, n_samp_stim,
                                 framerate=fr, condition_coords={'cond': conditions})

    ok = True
    # (a) windowed tensors identical to the verbatim structured-array loop
    for m in METRICS:
        got = ds[m].transpose('condition', 'roi', 'repeat', 'time').values
        d = np.nanmax(np.abs(got - ref[m]))
        same_nan = np.array_equal(np.isnan(got), np.isnan(ref[m]))
        print('    window  {:5s}  max|diff|={:.2e}  nan-pattern-match={}'.format(m, d, same_nan))
        ok = ok and (d == 0.0) and same_nan
    # (b) stim-window response matches the reference tensor AND an independent computation, to
    #     floating-point precision. resp_vect_cond differs only by reduction order (xarray mean
    #     over (repeat, time) vs numpy nanmean), so tolerances are float-epsilon; Fraw is float32
    #     straight from suite2p (F.npy), so its absolute noise is ~1e-4 on values ~1e3.
    idx_stim = list(range(n_samp_isi, n_samp_isi + n_samp_stim))
    for m in METRICS:
        resp_container = rt.stim_window_response(ds, m).transpose('roi', 'condition').values
        resp_ref = np.nanmean(ref[m][:, :, :, idx_stim], axis=(2, 3)).T
        resp_indep = independent_resp_vect(traces, stimlog, n_samp_isi, n_samp_stim, m)
        d_ref = np.nanmax(np.abs(resp_container - resp_ref))
        d_indep = np.nanmax(np.abs(resp_container - resp_indep))
        print('    resp    {:5s}  vs-ref={:.2e}  vs-independent={:.2e}'.format(m, d_ref, d_indep))
        ok = (ok
              and np.allclose(resp_container, resp_ref, rtol=1e-5, atol=1e-5, equal_nan=True)
              and np.allclose(resp_container, resp_indep, rtol=1e-5, atol=1e-5, equal_nan=True))
    print('  RESULT:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    results = {}
    for a, d, s in SESSIONS:
        try:
            results[(a, d)] = check_session(a, d, s)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[(a, d)] = 'ERROR: {}'.format(e)
    print('\n' + '=' * 78)
    print('SUMMARY')
    for k, v in results.items():
        print('  {}: {}'.format('/'.join(k), {True: 'PASS', False: 'FAIL', None: 'skipped'}.get(v, v)))
