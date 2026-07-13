#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the per-trial response window (reduce x window-length x onset offset, all in FRAMES) and tabulate
raw-gated responsive/selective counts (nested; selective subset of responsive).

Offsets and windows are swept in INTEGER FRAMES so the frame<->second mapping is exact (no rounding
collapse), and the window LENGTH is shown explicitly. ``win_fr = full`` uses the whole stimulus epoch shifted
by the offset; a numeric ``win_fr`` uses that many frames from the (shifted) onset -- a shorter peak-plateau
window. For ``reduce='peak'`` the baseline window is LENGTH-MATCHED to the stim window (peak grows with the
number of frames, so an unequal baseline inflates stim-vs-baseline responsiveness).

  .venv/bin/python marmanalysis/compare_response_window.py [SESSION]
      [--metric Fzsc] [--alpha 0.05] [--reduces mean,peak] [--offsets-fr 0,1,2,3,4,5] [--windows-fr full,4]
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
    ap.add_argument('--offsets-fr', default='0,1,2,3,4,5,6,7,8')
    ap.add_argument('--windows-fr', default='full,4')
    args = ap.parse_args()

    ds, ctx = sessionio.build_session_response_table(args.session)
    fr = ctx['md']['framerate']
    n = ds.sizes['roi']
    n_stim = int((ds['epoch'].values == rt.EPOCH_STIM).sum())
    reduces = [r.strip() for r in args.reduces.split(',') if r.strip()]
    offs = [int(x) for x in args.offsets_fr.split(',')]
    wins = [None if w.strip() == 'full' else int(w) for w in args.windows_fr.split(',')]

    print('session : %s' % os.path.basename(args.session.rstrip('/')))
    print('ROIs=%d  metric=%s  fr=%.3f Hz (%.0f ms/frame)  stim-epoch=%d fr (%.2f s)  alpha=%.3g'
          % (n, args.metric, fr, 1000.0 / fr, n_stim, n_stim / fr, args.alpha))
    print()
    print('  reduce  win_fr  win_s   off_fr  off_s    resp    sel')
    for red in reduces:
        for w in wins:
            w_fr = n_stim if w is None else w
            for off in offs:
                c = rt.classify_responses(ds, args.metric, framerate=fr, alpha=args.alpha, reduce=red,
                                          offset_response_window_sec=off / fr,
                                          response_window_sec=(None if w is None else w / fr),
                                          match_baseline_len=(red == 'peak'))
                print('  %-6s  %5s  %5.2f   %5d  %5.2f   %6d  %6d'
                      % (red, 'full' if w is None else str(w), w_fr / fr, off, off / fr,
                         int(c['responsive'].sum()), int(c['selective'].sum())))
        print()
    print('  window = stim epoch (win_fr=full) or an N-frame plateau, shifted later by off_fr frames;')
    print('  peak uses a length-matched baseline; responsive = differs-from-baseline OR differentiates;')
    print('  selective is a strict subset of responsive.')


if __name__ == '__main__':
    main()
