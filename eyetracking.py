#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gaze / eye-tracking gating from the analog eye-tracker (DAQ) record.

The per-session ``*_AIdata.p`` is a generic analog-input record -- eye X, Y plus accelerometer channels
piped to a DAQ, the shape any analog eye-tracker produces (EyeLoop for these sessions, but the tracker
identity is irrelevant to this reader). It is linked to the 2P acquisition frames via the stimulus log --
which stamps every trial event with both ``acqfr`` and the running ``AI_data.shape`` (there is NO
frame-trigger channel), so an (acqfr <-> AI-sample) interpolation aligns the continuous analog record to
frames. Produces per-trial gaze + eyes-open fraction, a DATA-DRIVEN fixation reference (session-median gaze
of eyes-open samples -- calibration is unreliable and fixation is not enforced), and a per-trial gate
(eyes-closed / off-fixation).

Channels (this era): ch0, ch1 = eye X, Y; ch2-4 = accelerometer (usually unused). Signal loss / eyes-closed
shows up as the DAQ rail (|v| ~ 9.5 V) and/or X,Y zeroed -- the signature varies by session, so both are
treated as lost. A tracker-SPECIFIC log (e.g. EyeLoop ``datalog.json``, ``'b'`` = blink) is a cleaner source
when present and would get its own reader (TODO; see [[eyetracking-data]]). NB: some calibration files carry
a legacy ``'coarse oculomatic values'`` key -- a misnomer; the tracker was EyeLoop.
"""
import glob
import os
import pickle
import re

import numpy as np

EYE_CH = (0, 1)     # eye X, Y channels (analog input; any tracker piped to the DAQ)
RAIL_V = 9.0        # |v| above this = DAQ rail (signal loss)


def _find(session_path, pat):
    hits = [f for f in glob.glob(os.path.join(session_path, pat)) if 'disptimes' not in f]
    return hits[0] if hits else None


def load_eye_ai(session_path):
    """Load analog eye data + parse the stimulus log. Returns dict: ``ai`` (n_samp, n_ch); ``anchors`` (k, 2)
    [acqfr, ai_sample] from the log's per-event stamps; ``trials`` {trial: {phase: (acqfr, ai_sample)}}."""
    aip = _find(session_path, '*_AIdata.p')
    logp = _find(session_path, '*Stimulus*.log')
    if aip is None or logp is None:
        raise FileNotFoundError('need *_AIdata.p and a stimulus *.log in %s' % session_path)
    ai = np.asarray(pickle.load(open(aip, 'rb')), float)
    anchors, trials = [], {}
    ev = re.compile(r'trial (\d+)/\d+, ([^,]+),')
    for line in open(logp):
        m = ev.search(line)
        if not m:
            continue
        a = re.search(r'acqfr=(\d+)', line)
        s = re.search(r'AI_data\.shape=\((\d+),', line)
        if not (a and s):
            continue
        tr, ph, acqfr, nai = int(m.group(1)), m.group(2).strip(), int(a.group(1)), int(s.group(1))
        anchors.append((acqfr, nai))
        trials.setdefault(tr, {})[ph] = (acqfr, nai)
    if not anchors:
        raise ValueError('no (acqfr, AI_data.shape) anchors parsed from %s' % logp)
    return {'ai': ai, 'anchors': np.array(sorted(set(anchors)), float), 'trials': trials,
            'ai_path': aip, 'log_path': logp}


def lost_mask(ai, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-sample signal-loss (eyes-closed/blink): either eye channel railed (|v|>rail_v), or both zeroed."""
    x, y = ai[:, eye_ch[0]], ai[:, eye_ch[1]]
    return (np.abs(x) > rail_v) | (np.abs(y) > rail_v) | ((x == 0) & (y == 0))


def acqfr_to_ai(anchors):
    """Function mapping acquisition-frame -> AI-sample index (linear interp over the log anchors)."""
    return lambda f: np.interp(f, anchors[:, 0], anchors[:, 1])


def _window_gaze(ai, lost, s0, s1, eye_ch):
    """(eyes_open_fraction, (mean_x, mean_y) over open samples) for AI-sample window [s0, s1)."""
    if s1 <= s0:
        return None
    ok = ~lost[s0:s1]
    if not ok.any():
        return 0.0, None
    return float(ok.mean()), (float(np.mean(ai[s0:s1, eye_ch[0]][ok])),
                              float(np.mean(ai[s0:s1, eye_ch[1]][ok])))


def trial_gaze(oc, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-trial gaze: mean eye X,Y + eyes-open fraction over the fixation window (if present) and the stim
    window ([fixation end | ISI end] -> stim end). Returns a list of per-trial dicts."""
    ai, trials = oc['ai'], oc['trials']
    f2a = acqfr_to_ai(oc['anchors'])
    lost = lost_mask(ai, eye_ch, rail_v)
    out = []
    for tr in sorted(trials):
        ph = trials[tr]
        rec = {'trial': tr}
        if 'fixation start' in ph and 'fixation end' in ph:
            g = _window_gaze(ai, lost, int(f2a(ph['fixation start'][0])), int(f2a(ph['fixation end'][0])), eye_ch)
            if g:
                rec['fixation_open'], rec['fixation_xy'] = g
        stim_start = ph.get('fixation end') or ph.get('ISI end')
        if stim_start and 'stim end' in ph:
            g = _window_gaze(ai, lost, int(f2a(stim_start[0])), int(f2a(ph['stim end'][0])), eye_ch)
            if g:
                rec['stim_open'], rec['stim_xy'] = g
        out.append(rec)
    return out


def gaze_gate(trial_recs, min_open=0.5, max_dev=None, ref=None):
    """Per-trial gate + deviations. The reference gaze is data-driven: median of fixation_xy across trials if
    fixation windows exist, else median of stim_xy (no calibration needed). A trial is KEPT if its stim-window
    eyes-open fraction >= ``min_open`` and (if ``max_dev`` given) its stim gaze is within ``max_dev`` of the
    reference. Returns (keep_mask, deviations, ref)."""
    fix = np.array([r['fixation_xy'] for r in trial_recs if r.get('fixation_xy')], float)
    stim = np.array([r['stim_xy'] for r in trial_recs if r.get('stim_xy')], float)
    if ref is None:
        ref = np.median(fix, axis=0) if len(fix) else (np.median(stim, axis=0) if len(stim) else np.array([np.nan, np.nan]))
    keep, devs = [], []
    for r in trial_recs:
        s = r.get('stim_xy')
        dev = float(np.hypot(s[0] - ref[0], s[1] - ref[1])) if s else np.nan
        devs.append(dev)
        keep.append(bool(r.get('stim_open', 0.0) >= min_open and
                         (max_dev is None or (np.isfinite(dev) and dev <= max_dev))))
    return np.array(keep), np.array(devs), np.asarray(ref, float)


def _demo(session_path, outdir='output'):
    from datetime import datetime, timezone
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    oc = load_eye_ai(session_path)
    ai, anchors = oc['ai'], oc['anchors']
    ai_per_fr = np.polyfit(anchors[:, 0], anchors[:, 1], 1)[0]
    lost = lost_mask(ai)
    recs = trial_gaze(oc)
    keep, devs, ref = gaze_gate(recs, min_open=0.5)
    stim_open = np.array([r.get('stim_open', np.nan) for r in recs], float)

    print('session   :', os.path.basename(session_path.rstrip('/'))[:60])
    print('AI        : %d samp x %d ch | ~%.1f samp/frame (~%.1f Hz @6.36)' % (ai.shape[0], ai.shape[1], ai_per_fr, ai_per_fr * 6.364))
    print('overall eyes-open fraction (whole session): %.3f' % (1 - lost.mean()))
    print('trials parsed: %d | with stim window: %d | with fixation window: %d'
          % (len(recs), int(np.isfinite(stim_open).sum()), sum('fixation_xy' in r for r in recs)))
    print('per-trial stim eyes-open: median %.3f | frac trials <0.5 open: %.3f'
          % (np.nanmedian(stim_open), np.nanmean(stim_open < 0.5)))
    print('data-driven gaze reference (eye V): (%.3f, %.3f)' % (ref[0], ref[1]))
    print('stim gaze deviation from ref (V): median %.3f | p90 %.3f' % (np.nanmedian(devs), np.nanpercentile(devs, 90)))
    print('gate keep (min_open=0.5): %d / %d (%.1f%%)' % (keep.sum(), len(keep), 100 * keep.mean()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].hist(stim_open[np.isfinite(stim_open)], bins=20, color='C0')
    ax[0].axvline(0.5, color='r', ls='--'); ax[0].set_xlabel('per-trial eyes-open fraction'); ax[0].set_ylabel('trials')
    sx = np.array([r['stim_xy'] for r in recs if r.get('stim_xy')], float)
    sc = ax[1].scatter(sx[:, 0], sx[:, 1], c=[r['stim_open'] for r in recs if r.get('stim_xy')], cmap='viridis', s=12)
    ax[1].plot(ref[0], ref[1], 'r+', ms=15, mew=2, label='data-driven ref')
    ax[1].set_xlabel('eye X (V)'); ax[1].set_ylabel('eye Y (V)'); ax[1].legend(fontsize=8); fig.colorbar(sc, ax=ax[1], label='eyes-open')
    fig.suptitle('eye-AI gaze — %s' % os.path.basename(session_path.rstrip('/'))[:44])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'eyetracking_%s_%s.png' % (os.path.basename(session_path.rstrip('/'))[:24],
                     datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1
          else 'suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
               'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
