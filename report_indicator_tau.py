#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-driver for the empirical jGCaMP8s decay tau: derive the NON-stimulus mask, measure tau
(half_decay and/or curve_fit), and append a row to a markdown results table -- a per-session confirmation of
the sensor kinetics against the published value.

  .venv/bin/python marmanalysis/report_indicator_tau.py [SESSION] [--methods half_decay,curve_fit]

``half_decay`` is fast (~10 s/session) and accurate; ``curve_fit`` fits an exponential per transient and is
slower (minutes) but is the more literal decay-constant estimate -- run it when you want the rigorous check.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessionio
import signal_quality as sq
import eyetracking as et

LN2 = np.log(2.0)
PUBLISHED_TAU_SEC = et.INDICATOR_DECAY_TAU_SEC          # 0.29 s (jGCaMP8s, in vivo)
PUBLISHED_THALF_SEC = PUBLISHED_TAU_SEC * LN2           # ~0.20 s
DEFAULT_SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
                   'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')


def nonstim_mask(ctx, n_frames):
    """Per-frame NON-stimulus (spontaneous) mask on the continuous 2P timeline: the inverse of the union of
    each trial's stimulus window [acqfr_stim_i, acqfr_stim_i + n_samp_stim). Spontaneous decays are
    indicator-dominated; stimulus-driven sustained firing would bias the measured decay upward."""
    stim = np.zeros(n_frames, bool)
    n_stim = int(ctx['n_samp_stim'])
    for a in ctx['stimlog']['acqfr_stim_i'].values:
        i0 = int(round(float(a)))
        i1 = min(i0 + n_stim, n_frames)
        if 0 <= i0 < i1:
            stim[i0:i1] = True
    return ~stim


def measure_session_indicator_tau(session_path, method='half_decay', variant=None, use_nonstim=True, **kw):
    """Load a session, derive the non-stimulus isi_mask, and measure the empirical indicator decay tau in ONE
    call. Returns (result, ctx); ``result`` is the calculate_indicator_decay_tau dict augmented with
    'session' / 'method' / 'framerate' / 'nonstim_frac'."""
    _, ctx = sessionio.build_session_response_table(session_path, variant=variant)
    dff = ctx['traces']['FdFF']
    fr = ctx['md']['framerate']
    mask = nonstim_mask(ctx, dff.shape[1]) if use_nonstim else None
    res = sq.calculate_indicator_decay_tau(dff, fr, isi_mask=mask, method=method, **kw)
    res.update({'session': os.path.basename(session_path.rstrip('/')), 'method': method, 'framerate': fr,
                'nonstim_frac': float(mask.mean()) if mask is not None else 1.0})
    return res, ctx


def append_markdown_row(res, out_md='output/indicator_tau.md'):
    """Append one result row to the markdown table (writing the header + published reference if the file is
    new). Returns the path."""
    os.makedirs(os.path.dirname(out_md) or '.', exist_ok=True)
    fresh = not os.path.exists(out_md)
    thalf = res.get('t_half_median')
    if thalf is None or not np.isfinite(thalf):
        thalf = res['tau_median'] * LN2
    lo, hi = res['tau_iqr']
    with open(out_md, 'a') as f:
        if fresh:
            f.write('# Empirical jGCaMP8s decay tau per session\n\n')
            f.write('Published in vivo (Zhang et al. Looger 2023 Nature, '
                    'https://doi.org/10.1038/s41586-023-05828-9): **tau ~%.2f s** (half-decay t_half ~%.2f s). '
                    'Empirical = median over isolated spontaneous transients (non-stimulus frames); an UPPER '
                    'bound (observed decay = indicator convolved with residual firing).\n\n'
                    % (PUBLISHED_TAU_SEC, PUBLISHED_THALF_SEC))
            f.write('| session | method | tau (s) | tau IQR (s) | t_half (s) | n events | tau / published |\n')
            f.write('|---|---|---|---|---|---|---|\n')
        f.write('| %s | %s | %.3f | %.3f-%.3f | %.3f | %d | %.2f |\n'
                % (res['session'][:46], res['method'], res['tau_median'], lo, hi, thalf,
                   res['n_events'], res['tau_median'] / PUBLISHED_TAU_SEC))
    return out_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=DEFAULT_SESSION)
    ap.add_argument('--methods', default='half_decay')
    ap.add_argument('--out', default='output/indicator_tau.md')
    args = ap.parse_args()
    print('session:', os.path.basename(args.session.rstrip('/'))[:60])
    for m in [x.strip() for x in args.methods.split(',') if x.strip()]:
        if m == 'curve_fit':
            print('  [curve_fit] fitting an exponential to each transient -- this may take a few minutes...')
        res, _ = measure_session_indicator_tau(args.session, method=m)
        p = append_markdown_row(res, args.out)
        print('  %-10s tau %.3f s (t_half %.3f) IQR %.3f-%.3f | n=%d | tau/published %.2f -> %s'
              % (m, res['tau_median'], res['t_half_median'], res['tau_iqr'][0], res['tau_iqr'][1],
                 res['n_events'], res['tau_median'] / PUBLISHED_TAU_SEC, p))


if __name__ == '__main__':
    main()
