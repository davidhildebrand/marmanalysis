#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the per-trial response window (reduce x onset-latency offset) and tabulate responsive/selective.

The default scalar is the mean over the whole stimulus epoch, which dilutes a brief 2-3 frame calcium
transient and can undercount responsive cells. This sweeps ``reduce`` (mean/peak/auc) x an onset-latency
offset (shifting the window later to track the indicator rise + neural latency, and to tolerate
response-timing jitter e.g. from uncontrolled eye position) and reports the raw-gated responsive/selective
counts (nested; selective subset of responsive) for each, so the effect of the window choice is explicit.

  .venv/bin/python marmanalysis/compare_response_window.py [SESSION]
      [--metric Fzsc] [--alpha 0.05] [--reduces mean,peak] [--offsets 0,0.1,0.2,0.3,0.5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessionio
import response_table as rt

DEFAULT_SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
                   'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=DEFAULT_SESSION)
    ap.add_argument('--metric', default='Fzsc')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--reduces', default='mean,peak')
    ap.add_argument('--offsets', default='0,0.1,0.2,0.3,0.5')
    args = ap.parse_args()

    ds, ctx = sessionio.build_session_response_table(args.session)
    fr = ctx['md']['framerate']
    n = ds.sizes['roi']
    reduces = [r.strip() for r in args.reduces.split(',') if r.strip()]
    offsets = [float(x) for x in args.offsets.split(',')]

    print('session : %s' % os.path.basename(args.session.rstrip('/')))
    print('ROIs=%d  metric=%s  fr=%.3f Hz  alpha=%.3g   stim-epoch=%d frames'
          % (n, args.metric, fr, args.alpha, int((ds['epoch'].values == rt.EPOCH_STIM).sum())))
    print()
    print('  reduce   offset_s   offset_fr   responsive   selective')
    for red in reduces:
        for off in offsets:
            c = rt.classify_responses(ds, args.metric, framerate=fr, alpha=args.alpha,
                                      reduce=red, onset_offset_sec=off)
            print('  %-7s  %7.2f   %8d   %10d   %9d'
                  % (red, off, int(round(off * fr)), int(c['responsive'].sum()), int(c['selective'].sum())))
    print()
    print('  window = stimulus epoch shifted later by offset (into the post-stim ISI decay tail);')
    print('  responsive = differs-from-baseline OR differentiates;  selective is a subset of responsive.')


if __name__ == '__main__':
    main()
