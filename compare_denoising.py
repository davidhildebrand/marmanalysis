#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Active / responsive / selective ROI counts: raw vs PSN-denoised response table (one session).

Answers "does response-level PSN denoising change the gate counts, and which way?". ``active`` is a
trace-level nu-SNR gate and PSN is a RESPONSE-level denoiser, so active is invariant by construction and
reported once; ``responsive`` and ``selective`` are recomputed on the PSN-denoised response table
(``denoising.denoise_psn``) and compared to raw, with gained/lost breakdown.

  .venv/bin/python marmanalysis/compare_denoising.py [SESSION]
      [--metric Fzsc] [--mode conservative] [--alpha 0.05] [--active-k 7.0] [--no-diagnostic]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessionio
import response_table as rt
import signal_quality as sq
from denoising import denoise_psn, denoise_psn_xval

DEFAULT_SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
                   'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')


def _sig(p, alpha):
    return (p < alpha) & np.isfinite(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=DEFAULT_SESSION)
    ap.add_argument('--metric', default='Fzsc')
    ap.add_argument('--mode', default='conservative')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--active-k', type=float, default=7.0)
    ap.add_argument('--active-metric', default='FdFF')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--no-diagnostic', action='store_true')
    args = ap.parse_args()

    ds, ctx = sessionio.build_session_response_table(args.session)
    fr = ctx['md']['framerate']
    n = ds.sizes['roi']
    m = args.metric

    # active: trace-level nu-SNR gate; PSN is response-level, so this is invariant to it.
    active, _, _, _ = sq.active_mask(ctx['traces'][args.active_metric], framerate=fr,
                                     method='nu', k=args.active_k)

    def gates(d):
        pr = rt.anova_responsive(d, m, framerate=fr)
        psl = rt.anova_selective(d, m)
        return _sig(pr, args.alpha), _sig(psl, args.alpha)

    resp0, sel0 = gates(ds)
    dsd, info = denoise_psn(ds, m, args.mode, diagnostic=not args.no_diagnostic, outdir='output',
                            tag=os.path.basename(args.session.rstrip('/'))[:24])
    resp1, sel1 = gates(dsd)
    dscv, _ = denoise_psn_xval(ds, m, args.mode, n_folds=args.folds)
    resp2, sel2 = gates(dscv)

    print('session   : %s' % os.path.basename(args.session.rstrip('/')))
    print('ROIs=%d  metric=%s  mode=%s  alpha=%.3g  active(nu k=%.1f on %s)  cv-folds=%d'
          % (n, m, args.mode, args.alpha, args.active_k, args.active_metric, args.folds))
    print('PSN       : retained %s signal dims | median NCSNR %.3f -> %.3f%s'
          % (info['n_signal_dims'], info['ncsnr_before_median'], info['ncsnr_after_median'],
             '' if args.no_diagnostic else ' | fig %s' % info['figure_path']))
    print()
    print('  gate          normal   psn(in-sample)   psn(CV, honest)')
    print('  %-11s %6d   %14s   %15s' % ('active', int(active.sum()), 'invariant', 'invariant'))
    print('  %-11s %6d   %14d   %15d'
          % ('responsive', int(resp0.sum()), int(resp1.sum()), int(resp2.sum())))
    print('  %-11s %6d   %14d   %15d'
          % ('selective', int(sel0.sum()), int(sel1.sum()), int(sel2.sum())))
    print()
    print('  nesting selective-not-responsive:  normal=%d  in-sample=%d  CV=%d'
          % (int(np.sum(sel0 & ~resp0)), int(np.sum(sel1 & ~resp1)), int(np.sum(sel2 & ~resp2))))


if __name__ == '__main__':
    main()
