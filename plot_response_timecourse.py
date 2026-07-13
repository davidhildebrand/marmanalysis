#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trial-averaged ΔF/F timecourse over the trial -- shows the response rise / plateau / decay directly, so the
per-trial response window can be placed on the plateau rather than inferred from the offset sweep. Saves a
figure to ``output/`` (Agg, UTC-stamped) AND prints the curve values at key times.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessionio
import response_table as rt

DEFAULT_SESSION = ('suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
                   'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', nargs='?', default=DEFAULT_SESSION)
    ap.add_argument('--metric', default='FdFF')
    ap.add_argument('--outdir', default='output')
    args = ap.parse_args()

    ds, ctx = sessionio.build_session_response_table(args.session)
    fr = ctx['md']['framerate']
    m = args.metric
    t = ds['time_s'].values
    stim = ds['epoch'].values == rt.EPOCH_STIM

    # responsive cells via the default gate (just to pick clearly-driven cells; shape is robust to selection)
    resp = rt.classify_responses(ds, 'Fzsc', framerate=fr)['responsive']
    ridx = np.where(resp)[0]

    tc = ds[m].mean(dim=('condition', 'repeat'), skipna=True).transpose('roi', 'time').values   # (roi, time)
    # per-ROI best condition (max stim-window response), then its timecourse (mean over repeats)
    cond_resp = ds[m].isel(time=np.where(stim)[0]).mean(('repeat', 'time')).transpose('roi', 'condition').values
    best = np.nanargmax(cond_resp, axis=1)
    tc_bc = ds[m].mean('repeat').transpose('roi', 'condition', 'time').values
    tc_bc = tc_bc[np.arange(tc_bc.shape[0]), best, :]                                            # (roi, time)

    curves = {'all cells (n=%d)' % ds.sizes['roi']: np.nanmean(tc, 0),
              'responsive, all cond (n=%d)' % len(ridx): np.nanmean(tc[ridx], 0),
              'responsive, best cond': np.nanmean(tc_bc[ridx], 0)}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvspan(t[stim].min(), t[stim].max(), color='orange', alpha=0.12, label='stimulus ON')
    ax.axvspan(0.5, 1.25, color='green', alpha=0.10, label='macaque conv. 0.5-1.25 s')
    ax.axvline(0, color='k', lw=0.8, ls=':')
    for (lbl, cv), col in zip(curves.items(), ['0.6', 'C0', 'C3']):
        ax.plot(t, cv, color=col, lw=1.7, label=lbl)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('ΔF/F  (%s)' % m)
    ax.set_title('trial-averaged response timecourse — %s' % os.path.basename(args.session.rstrip('/'))[:38])
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.2)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')
    path = os.path.join(args.outdir, 'response_timecourse_%s_%s.png'
                        % (os.path.basename(args.session.rstrip('/'))[:24], stamp))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print('responsive n=%d / %d | stim %.2f–%.2f s (%d fr) | fr=%.3f Hz'
          % (len(ridx), ds.sizes['roi'], t[stim].min(), t[stim].max(), stim.sum(), fr))
    ts = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    print('  t(s):      ' + ' '.join('%6.2f' % x for x in ts))
    for lbl, cv in curves.items():
        vals = [cv[np.argmin(abs(t - x))] for x in ts]
        pk = int(np.nanargmax(cv))
        print('  %-26s ' % lbl[:26] + ' '.join('%6.3f' % v for v in vals) + '   peak %.3f @ %+.2fs' % (cv[pk], t[pk]))
    print('saved', path)


if __name__ == '__main__':
    main()
