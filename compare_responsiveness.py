#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare stimulus-responsiveness criteria on one imaging session.

Runs four responsiveness measures on the same (roi, condition, repeat, time) response table and
reports how much they agree, to quantify the effect of replacing the frame-pooled ANOVA -- which
pseudoreplicates strongly autocorrelated within-trial frames and is therefore anti-conservative --
with a trial-level test:

  frame ANOVA         images.responsive_anova              (pools trials x frames)
  trial ANOVA         response_table.anova_selective    (one stim-window scalar per trial)
  ZETA                zetapy.zetatstest                    (parameter-free, full trace vs onsets)
  split-half reliab   response_table.split_half_reliability (Spearman-Brown corrected)

Run from the workspace root (where .venv lives):
  .venv/bin/python marmanalysis/compare_responsiveness.py [SESSION_PATH]
      [--metric Fzsc] [--alpha 0.01] [--rel-thresh 0.4] [--max-rois N] [--no-zeta] [--out CSV]
"""
import argparse
import logging
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import images
import response_table as rt
import sessionio

DEFAULT_SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
                   'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')


def zeta_pvalues(context, metric, n_samp_window, max_rois=None):
    """Per-ROI ZETA p-value from the full trace vs stimulus-onset times (time-series ZETA)."""
    from zetapy import zetatstest
    fr = context['md']['framerate']
    traces = context['traces'][metric]
    n_roi, n_frames = traces.shape
    t = np.arange(n_frames) / fr
    onsets = context['stimlog']['acqfr_stim_i'].to_numpy(dtype=float)
    onsets = np.sort(onsets[np.isfinite(onsets)]) / fr
    max_dur = n_samp_window / fr
    # ZETA jitters events (~+/-2 s) and needs data around each; onsets at the trace edges otherwise
    # trigger an empty-interpolation error, so drop the first/last and any without a safe margin.
    if onsets.size > 2:
        onsets = onsets[1:-1]
    onsets = onsets[(onsets - max_dur - 2.0 >= t[0]) & (onsets + max_dur + 2.0 <= t[-1])]
    n = n_roi if max_rois is None else min(max_rois, n_roi)
    logging.getLogger().setLevel(logging.ERROR)  # silence per-trial jitter warnings
    p = np.full(n_roi, np.nan)
    for r in range(n):
        try:
            p[r] = zetatstest(t, traces[r].astype(float), onsets, max_duration=max_dur)[0]
        except Exception as exc:
            print('  ZETA roi %d failed: %s' % (r, exc))
        if (r + 1) % 100 == 0:
            print('  ZETA %d/%d' % (r + 1, n), flush=True)
    return p


def _neglog(p):
    return -np.log10(np.clip(p, 1e-300, 1.0))


def _report_pair(name_a, mask_a, name_b, mask_b):
    a, b = set(np.where(mask_a)[0]), set(np.where(mask_b)[0])
    both = len(a & b)
    union = len(a | b) or 1
    print('  %-12s(%4d) vs %-12s(%4d): both %4d | %s-only %4d | %s-only %4d | Jaccard %.2f'
          % (name_a, len(a), name_b, len(b), both, name_a, len(a - b), name_b, len(b - a),
             both / union))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=DEFAULT_SESSION)
    ap.add_argument('--metric', default='Fzsc')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--rel-thresh', type=float, default=0.4)
    ap.add_argument('--resp-q', type=float, default=0.05)
    ap.add_argument('--max-rois', type=int, default=None)
    ap.add_argument('--no-zeta', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ds, ctx = sessionio.build_session_response_table(args.session)
    m = args.metric
    n_roi = ds.sizes['roi']
    print('session :', os.path.basename(args.session.rstrip('/')))
    print('shape   : %d ROIs x %d conditions x %d repeats  (metric=%s)'
          % (n_roi, ds.sizes['condition'], ds.sizes['repeat'], m))

    p_frame = images.responsive_anova(ds, m)
    p_trial = rt.anova_selective(ds, m)
    p_resp = rt.anova_responsive(ds, m, framerate=ctx['md']['framerate'])
    rel = rt.split_half_reliability(ds, m)
    p_zeta = (np.full(n_roi, np.nan) if args.no_zeta
              else zeta_pvalues(ctx, m, ctx['n_samp_isi'] + ctx['n_samp_stim'], args.max_rois))

    a, rt_thr = args.alpha, args.rel_thresh
    crit = {
        'responsive-ANOVA': (p_resp < args.resp_q, np.isfinite(p_resp)),  # gate: stim+baseline ANOVA
        'ZETA RESP': (p_zeta < a, np.isfinite(p_zeta)),        # responsiveness (onset-locked)
        'frame-ANOVA sel': (p_frame < a, np.isfinite(p_frame)),  # selectivity (pseudorep'd)
        'trial-ANOVA sel': (p_trial < a, np.isfinite(p_trial)),  # selectivity
        'reliab>%.2f sel' % rt_thr: (rel > rt_thr, np.isfinite(rel)),  # selectivity
    }
    print('\nresponsive counts (alpha=%.3g):' % a)
    for k, (s, vld) in crit.items():
        s = s & vld
        print('  %-14s %5d / %5d  (%.1f%%)'
              % (k, int(s.sum()), int(vld.sum()), 100 * s.sum() / max(int(vld.sum()), 1)))

    print('\npairwise agreement (responsive sets):')
    sig = {k: (s & vld) for k, (s, vld) in crit.items()}
    k_frame, k_trial, k_resp = 'frame-ANOVA sel', 'trial-ANOVA sel', 'responsive-ANOVA'
    k_rel, k_zeta = 'reliab>%.2f sel' % rt_thr, 'ZETA RESP'
    _report_pair('frame', sig[k_frame], 'trial', sig[k_trial])
    _report_pair('resp', sig[k_resp], 'trial', sig[k_trial])       # is responsiveness a superset of selectivity?
    if not args.no_zeta:
        z = np.isfinite(p_zeta)
        print('  (ZETA computed on %d ROIs; comparisons below restricted to those)' % int(z.sum()))
        _report_pair('resp', sig[k_resp] & z, 'ZETA', sig[k_zeta])  # two drive measures agree?
        _report_pair('trial', sig[k_trial] & z, 'ZETA', sig[k_zeta])
        _report_pair('ZETA', sig[k_zeta], 'reliab', sig[k_rel] & z)

    print('\nrank correlations (Spearman) of the continuous scores:')
    scores = {'-log10 p_frame': _neglog(p_frame), '-log10 p_trial': _neglog(p_trial),
              '-log10 p_resp': _neglog(p_resp), 'reliability': rel}
    if not args.no_zeta:
        scores['-log10 p_zeta'] = _neglog(p_zeta)
    keys = list(scores)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = scores[keys[i]], scores[keys[j]]
            ok = np.isfinite(x) & np.isfinite(y)
            rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 2 else float('nan')
            print('  %-15s vs %-15s  rho=%+.3f  (n=%d)' % (keys[i], keys[j], rho, int(ok.sum())))

    if args.out:
        import csv
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['roi', 'p_frame_anova', 'p_trial_anova', 'p_stim_vs_base', 'p_zeta', 'reliability'])
            for r in range(n_roi):
                w.writerow([r, p_frame[r], p_trial[r], p_resp[r], p_zeta[r], rel[r]])
        print('\nwrote', args.out)


if __name__ == '__main__':
    main()
