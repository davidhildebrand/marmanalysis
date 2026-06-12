#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-session parity: the full sessionio loader pipeline vs the structured-array windowing.

Loads each session end-to-end with sessionio.build_session_response_table (metadata -> suite2p ->
F0/dF-F -> stimlog -> acqfr correction / timing / trim -> response_table xarray container), then
checks the windowed tensors and resp_vect_cond against:
  * a VERBATIM reproduction of the structured-array windowing loop (analysis_for_images.py:1604-1657);
  * an INDEPENDENT per-condition stim-window mean computed straight from the traces.

This exercises every shared loader, so a 3/3 pass is the old-vs-new verification for task 2b.

Run:  ../.venv/bin/python marmanalysis/tests/parity_real_sessions.py
"""

import os
import sys
import warnings

import numpy as np

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import response_table as rt
import sessionio

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


def independent_resp_vect(traces, stimlog, n_samp_stim, metric):
    """Per-(roi, condition) stim-window mean from the traces, independent of the windowing loop."""
    conditions = np.unique(stimlog['cond'].values)
    out = np.full((traces[metric].shape[0], conditions.size), np.nan)
    for ci, c in enumerate(conditions):
        onsets = stimlog.loc[stimlog['cond'] == c, 'acqfr_stim_i'].astype(int).to_numpy()
        out[:, ci] = np.mean([traces[metric][:, o:o + n_samp_stim].mean(axis=1) for o in onsets], axis=0)
    return out


def check_session(animal, date, session):
    print('\n' + '=' * 78)
    print('{} / {} / {}'.format(animal, date, session[:40]))
    ds, ctx = sessionio.build_session_response_table(os.path.join(BASE, animal, date, session))
    traces, stimlog = ctx['traces'], ctx['stimlog']
    n_isi, n_stim = ctx['n_samp_isi'], ctx['n_samp_stim']
    print('  stimlog base={}  framerate={:.3f}  n_ROIs={}  n_frames={}'.format(
        ctx['stim_provenance']['base_source'], ctx['md']['framerate'],
        traces['Fraw'].shape[0], traces['Fraw'].shape[1]))
    print('  n_samp_isi={}  n_samp_stim={}  n_conds={}  n_reps={}'.format(
        n_isi, n_stim, ds.sizes['condition'], ds.sizes['repeat']))

    ref, conditions, n_reps = reference_windowing(traces, stimlog, n_isi, n_stim)
    ok = True
    for m in METRICS:
        got = ds[m].transpose('condition', 'roi', 'repeat', 'time').values
        d = np.nanmax(np.abs(got - ref[m]))
        same_nan = np.array_equal(np.isnan(got), np.isnan(ref[m]))
        print('    window  {:5s}  max|diff|={:.2e}  nan-match={}'.format(m, d, same_nan))
        ok = ok and (d == 0.0) and same_nan

    idx_stim = list(range(n_isi, n_isi + n_stim))
    for m in METRICS:
        resp = rt.stim_window_response(ds, m).transpose('roi', 'condition').values
        resp_ref = np.nanmean(ref[m][:, :, :, idx_stim], axis=(2, 3)).T
        resp_indep = independent_resp_vect(traces, stimlog, n_stim, m)
        print('    resp    {:5s}  vs-ref={:.2e}  vs-independent={:.2e}'.format(
            m, np.nanmax(np.abs(resp - resp_ref)), np.nanmax(np.abs(resp - resp_indep))))
        ok = (ok
              and np.allclose(resp, resp_ref, rtol=1e-5, atol=1e-5, equal_nan=True)
              and np.allclose(resp, resp_indep, rtol=1e-5, atol=1e-5, equal_nan=True))
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
        print('  {}: {}'.format('/'.join(k), {True: 'PASS', False: 'FAIL'}.get(v, v)))
