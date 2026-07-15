#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eye-position gating from the analog eye-tracker (DAQ) record.

The per-session ``*_AIdata.p`` is a generic analog-input record -- eye X, Y plus accelerometer channels
piped to a DAQ, the shape any analog eye-tracker produces (EyeLoop for these sessions, but the tracker
identity is irrelevant to this reader). It is linked to the 2P acquisition frames via the stimulus log --
which stamps every trial event with both ``acqfr`` and the running ``AI_data.shape`` (there is NO
frame-trigger channel), so an (acqfr <-> AI-sample) interpolation aligns the continuous analog record to
frames. Produces per-trial eye position + eyes-open fraction, per-phase sample masks (stim / fixation / ISI), eye position
dispersion descriptors (BCEA + robust median-radial-dev), a DATA-DRIVEN on-target reference (stim-period
median eye position -- the stimulus sits at the fixation location, and the stimulus is what concentrates eye position), and a
per-trial gate (eyes-closed / off-fixation). An optional per-session calibration (a separate
``*_EyeTrackingCalibration`` recording) gives a ROUGH voltage->degree scale via an affine fit -- gated by a
quality check, since the grid is nonlinear + recorded well before the session (drift).

Channels (this era): ch0, ch1 = eye X, Y; ch2-4 = accelerometer (usually unused). Signal loss / eyes-closed
shows up as the DAQ rail (|v| ~ 9.5 V) and/or X,Y zeroed -- the signature varies by session, so both are
treated as lost. A tracker-SPECIFIC log (e.g. EyeLoop ``datalog.json``, ``'b'`` = blink) is a cleaner source
when present and would get its own reader (TODO; see [[eyetracking-data]]). NB: the calibration voltage key is
a legacy misnomer (``'coarse oculomatic values'``) -- the tracker was EyeLoop.
"""
import glob
import os
import pickle
import re

import numpy as np

EYE_CH = (0, 1)     # eye X, Y channels (analog input; any tracker piped to the DAQ)
RAIL_V = 9.0        # |v| above this = DAQ rail (signal loss)

# Calibration usability gate (affine voltage->degree fit). The grid is ~+/-5 deg; drop to VOLTS-only if the
# map folds, is too anisotropic, non-monotonic, or its residual is a large fraction of the grid half-range.
CAL_COND_MAX = 4.0          # max deg/V axis-ratio (anisotropy) before untrustworthy
CAL_MAX_INVERSIONS = 1      # allow at most this many non-monotonic grid edges
CAL_RESID_FRAC_GOOD = 0.15  # residual < this fraction of grid half-range => 'good'
CAL_RESID_FRAC_DROP = 0.50  # residual > this fraction => 'unusable' (report volts only)


def _find(session_path, pat):
    hits = [f for f in glob.glob(os.path.join(session_path, pat)) if 'disptimes' not in f]
    return hits[0] if hits else None


def load_eye_ai(session_path):
    """Load analog eye data + parse the stimulus log. Returns dict: ``ai`` (n_samp, n_ch); ``anchors`` (k, 2)
    [acqfr, ai_sample] from the log's per-event stamps; ``trials`` {trial: {phase: (acqfr, ai_sample)}};
    ``stim_pos`` {trial: (deg_x, deg_y)} = the stimulus screen position parsed from each 'stim start' line
    (enables task-based calibration anchors -- see ``calculate_stim_derived_eyepos_anchors``)."""
    aip = _find(session_path, '*_AIdata.p')
    logp = _find(session_path, '*Stimulus*.log')
    if aip is None or logp is None:
        raise FileNotFoundError('need *_AIdata.p and a stimulus *.log in %s' % session_path)
    ai = np.asarray(pickle.load(open(aip, 'rb')), float)
    anchors, trials, stim_pos = [], {}, {}
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
        if ph == 'stim start':
            pm = re.search(r'pos=[\(\[]\s*([-\d.eE]+)[,\s]+\s*([-\d.eE]+)', line)
            if pm:
                stim_pos[tr] = (float(pm.group(1)), float(pm.group(2)))
    if not anchors:
        raise ValueError('no (acqfr, AI_data.shape) anchors parsed from %s' % logp)
    return {'ai': ai, 'anchors': np.array(sorted(set(anchors)), float), 'trials': trials,
            'stim_pos': stim_pos, 'ai_path': aip, 'log_path': logp}


def calculate_eyepos_loss_mask(ai, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-sample signal-loss (eyes-closed/blink): either eye channel railed (|v|>rail_v), or both zeroed."""
    x, y = ai[:, eye_ch[0]], ai[:, eye_ch[1]]
    return (np.abs(x) > rail_v) | (np.abs(y) > rail_v) | ((x == 0) & (y == 0))


def map_acqfr_to_ai_sample(anchors):
    """Function mapping acquisition-frame -> AI-sample index (linear interp over the log anchors)."""
    return lambda f: np.interp(f, anchors[:, 0], anchors[:, 1])


def calculate_phase_sample_masks(oc):
    """Per-AI-sample boolean masks for the three trial phases, from the log's per-trial event stamps:
    ``stim`` (stim start -> stim end), ``fixation`` (fixation start -> end), ``isi`` (ISI start -> fixation
    start, i.e. the BLANK interval before the fixation spot -- the fixation spot occupies the ISI tail). Kept
    separate on purpose: whether the fixation spot actually tightens the eye-position cluster is session/animal-specific (in the
    Cadbury images session it does NOT -- fixation looks like blank ISI; only the stimulus concentrates eye position).
    Falls back gracefully for sessions lacking a fixation phase."""
    n = oc['ai'].shape[0]
    f2a = map_acqfr_to_ai_sample(oc['anchors'])
    masks = {k: np.zeros(n, bool) for k in ('stim', 'fixation', 'isi')}

    def span(ph, a, b, name):
        if a in ph and b in ph:
            i0, i1 = int(f2a(ph[a][0])), int(f2a(ph[b][0]))
            if 0 <= i0 < i1 <= n:
                masks[name][i0:i1] = True

    for ph in oc['trials'].values():
        stim_start = 'stim start' if 'stim start' in ph else ('fixation end' if 'fixation end' in ph else 'ISI end')
        span(ph, stim_start, 'stim end', 'stim')
        span(ph, 'fixation start', 'fixation end', 'fixation')
        isi_end = 'fixation start' if 'fixation start' in ph else ('ISI end' if 'ISI end' in ph else 'stim start')
        span(ph, 'ISI start', isi_end, 'isi')
    return masks


def calculate_stim_sample_mask(oc):
    """Boolean per-AI-sample mask, True during stimulus windows (convenience wrapper over
    ``calculate_phase_sample_masks``)."""
    return calculate_phase_sample_masks(oc)['stim']


def _window_eyepos(ai, lost, s0, s1, eye_ch):
    """(eyes_open_fraction, (mean_x, mean_y) over open samples) for AI-sample window [s0, s1)."""
    if s1 <= s0:
        return None
    ok = ~lost[s0:s1]
    if not ok.any():
        return 0.0, None
    return float(ok.mean()), (float(np.mean(ai[s0:s1, eye_ch[0]][ok])),
                              float(np.mean(ai[s0:s1, eye_ch[1]][ok])))


def calculate_trial_eyepos(oc, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-trial eye position: mean eye X,Y + eyes-open fraction over the fixation window (if present) and the stim
    window ([fixation end | ISI end] -> stim end). Returns a list of per-trial dicts."""
    ai, trials = oc['ai'], oc['trials']
    f2a = map_acqfr_to_ai_sample(oc['anchors'])
    lost = calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    out = []
    for tr in sorted(trials):
        ph = trials[tr]
        rec = {'trial': tr}
        if 'fixation start' in ph and 'fixation end' in ph:
            g = _window_eyepos(ai, lost, int(f2a(ph['fixation start'][0])), int(f2a(ph['fixation end'][0])), eye_ch)
            if g:
                rec['fixation_open'], rec['fixation_xy'] = g
        stim_start = ph.get('stim start') or ph.get('fixation end') or ph.get('ISI end')
        if stim_start and 'stim end' in ph:
            g = _window_eyepos(ai, lost, int(f2a(stim_start[0])), int(f2a(ph['stim end'][0])), eye_ch)
            if g:
                rec['stim_open'], rec['stim_xy'] = g
        out.append(rec)
    return out


def calculate_eyepos_bcea(x, y, p=0.68):
    """Bivariate Contour Ellipse Area at probability ``p`` -- the standard eye-tracking fixation-stability
    metric: the area of the covariance ellipse containing fraction ``p`` of samples,
    ``-2 ln(1-p) * pi * sqrt(det Cov)``. Units = input units squared (V^2, or deg^2 after calibration)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return float('nan')
    C = np.cov(x, y)
    return float(-2.0 * np.log(1 - p) * np.pi * np.sqrt(max(np.linalg.det(C), 0.0)))


def calculate_eyepos_dispersion(x, y, p=0.68):
    """Eye-position dispersion descriptors: robust ``median`` center, per-axis ``sd``, ``medrad`` (median radial
    deviation from the median -- robust to look-away outliers), and ``calculate_eyepos_bcea`` at probability ``p``. Lengths in
    input units (V, or deg after calibration); ``calculate_eyepos_bcea`` in units^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = float(np.median(x)), float(np.median(y))
    return {'median': (mx, my), 'sd': (float(x.std()), float(y.std())),
            'medrad': float(np.median(np.hypot(x - mx, y - my))), 'bcea': calculate_eyepos_bcea(x, y, p), 'n': int(x.size)}


def calculate_eyepos_kde_peak(x, y, grid=140, n_sub=25000, seed=0):
    """2D density MODE: the (x, y) maximizing a Gaussian KDE on a grid. Robust 'where they looked most'
    estimate -- sparse but concentrated fixations form a peak while wandering stays diffuse, so the mode
    recovers the target even when the animal only occasionally looks at it (unlike mean/median, which the
    wandering pulls off-target). Split-half stable to ~0.03 V on the Cadbury stim epoch. Subsamples for
    tractability. Returns (px, py)."""
    from scipy.stats import gaussian_kde
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, n_sub, replace=False) if x.size > n_sub else np.arange(x.size)
    k = gaussian_kde(np.vstack([x[idx], y[idx]]))
    xr, yr = np.percentile(x, [1, 99]), np.percentile(y, [1, 99])
    GX, GY = np.meshgrid(np.linspace(xr[0], xr[1], grid), np.linspace(yr[0], yr[1], grid))
    d = k(np.vstack([GX.ravel(), GY.ravel()]))
    i = int(d.argmax())
    return float(GX.ravel()[i]), float(GY.ravel()[i])


def calculate_stim_derived_eyepos_ref(oc, reduce='peak', eye_ch=EYE_CH, rail_v=RAIL_V):
    """Data-driven on-target eye position reference: the KDE mode (``reduce='peak'``, default) or median of pooled
    eyes-open eye position over ALL stim windows -- 'where the animal looked when looking at the stimulus'. The mode
    is preferred: it locks onto the (sparse but concentrated) fixation cluster and ignores wandering, and is
    far more reliable than the poorly-participated formal calibration. Returns (vx, vy)."""
    ai = oc['ai']
    stim = calculate_phase_sample_masks(oc)['stim'] & ~calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    xs, ys = ai[stim, eye_ch[0]], ai[stim, eye_ch[1]]
    if xs.size < 10:
        return np.array([np.nan, np.nan])
    return np.asarray(calculate_eyepos_kde_peak(xs, ys) if reduce == 'peak' else (np.median(xs), np.median(ys)), float)


def calculate_stim_derived_eyepos_anchors(oc, reduce='peak', min_samples=1000, conc_radius=1.0, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Task-based calibration anchors: for each DISTINCT stimulus screen position, the eye-voltage where the
    animal looked at it (KDE mode by default), plus a quality score. Because the stimulus -- unlike a bare
    calibration dot -- motivates a poorly-trained animal to look, these anchors beat the formal grid; for a
    multi-position session they can build a calibration outright, and for a single-position session (e.g.
    images at 0,0) they yield ONE anchor (the (0,0) supplement). Quality: ``n`` eyes-open samples and ``conc``
    = fraction within ``conc_radius`` V of the estimate (peak sharpness / participation rate). Returns a list
    of {'pos_deg', 'volt', 'n', 'conc'} sorted by decreasing ``conc``."""
    ai = oc['ai']
    x, y = ai[:, eye_ch[0]], ai[:, eye_ch[1]]
    lost = calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    f2a = map_acqfr_to_ai_sample(oc['anchors'])
    by_pos = {}
    for tr, ph in oc['trials'].items():
        pos = oc.get('stim_pos', {}).get(tr)
        s = ph.get('stim start') or ph.get('fixation end') or ph.get('ISI end')
        e = ph.get('stim end')
        if pos is None or not (s and e):
            continue
        i0, i1 = int(f2a(s[0])), int(f2a(e[0]))
        ok = ~lost[i0:i1]
        if ok.any():
            by_pos.setdefault(pos, []).append((x[i0:i1][ok], y[i0:i1][ok]))
    out = []
    for pos, chunks in by_pos.items():
        gx = np.concatenate([c[0] for c in chunks])
        gy = np.concatenate([c[1] for c in chunks])
        if gx.size < min_samples:
            continue
        v = calculate_eyepos_kde_peak(gx, gy) if reduce == 'peak' else (float(np.median(gx)), float(np.median(gy)))
        conc = float((np.hypot(gx - v[0], gy - v[1]) < conc_radius).mean())
        out.append({'pos_deg': pos, 'volt': (float(v[0]), float(v[1])), 'n': int(gx.size), 'conc': conc})
    return sorted(out, key=lambda a: -a['conc'])


def is_eye_near_stim(eyepos_x, eyepos_y, stim_derived_eyepos_ref, near_radius_v):
    """True where the (rough) eye position is within ``near_radius_v`` volts of the stimulus-derived
    eye-position reference -- i.e. the animal is plausibly looking at the stimulus. Per-sample boolean
    (scalars or arrays). Absolute eye-position calibration is unreliable here, so this is a RELATIVE proximity
    test to the data-driven reference, not an absolute-degrees judgement."""
    return (np.hypot(np.asarray(eyepos_x, float) - stim_derived_eyepos_ref[0],
                     np.asarray(eyepos_y, float) - stim_derived_eyepos_ref[1]) <= near_radius_v)


def calculate_eye_near_stim_sec(oc, stim_derived_eyepos_ref, near_radius_v, expected_response_dur_sec=0.5,
                                exclude_late_looks=True, framerate=6.364, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-trial SECONDS the (rough) eye position was near the stimulus during the countable part of the stim
    window -- the raw quantity a viewing-time gate thresholds (keep a trial when this >= a minimum, which
    defaults to ``expected_response_dur_sec``). Returns a per-trial numpy array (NaN if a trial has no stim
    window). Pass the session's ``framerate`` (acq frames/s) so the AI-sample count converts to real seconds.

    PROCESS: per trial, take the stim-window AI samples, keep those that are eyes-open AND ``is_eye_near_stim``
    (within ``near_radius_v`` of the reference), and convert the surviving sample count to seconds via the AI
    sample rate (samples/acq-frame from the log anchors x framerate).

    LOGIC (``exclude_late_looks``, default True): the response is measured as the full-stim-period mean, and
    jGCaMP8s has a fast rise but slow decay, so a look landing in the final ``expected_response_dur_sec`` of
    the stimulus drives calcium that develops mostly AFTER stim offset -- outside the averaging window -- and
    so is not captured. The countable window is therefore shortened to
    [stim_start, stim_end - expected_response_dur_sec]; near-stim time in that final stretch is not counted.
    Set False to count the whole stim window.

    OUTCOME: a trial whose only near-stim viewing is late tallies ~0 s and fails the gate; a trial viewed
    early enough for its response to register tallies its real on-stimulus time. This is an absolute seconds
    count (not a fraction of the stim window), so a 1 s and a 2 s stimulus session are treated consistently."""
    ai = oc['ai']
    lost = calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    near = is_eye_near_stim(ai[:, eye_ch[0]], ai[:, eye_ch[1]], stim_derived_eyepos_ref, near_radius_v) & ~lost
    f2a = map_acqfr_to_ai_sample(oc['anchors'])
    samp_per_sec = float(np.polyfit(oc['anchors'][:, 0], oc['anchors'][:, 1], 1)[0]) * framerate
    tail = int(expected_response_dur_sec * samp_per_sec) if exclude_late_looks else 0
    out = []
    for tr in sorted(oc['trials']):
        ph = oc['trials'][tr]
        s = ph.get('stim start') or ph.get('fixation end') or ph.get('ISI end')
        e = ph.get('stim end')
        if not (s and e):
            out.append(np.nan)
            continue
        i0, i1 = int(f2a(s[0])), int(f2a(e[0])) - tail
        out.append(float(near[i0:i1].sum()) / samp_per_sec if i1 > i0 else 0.0)
    return np.array(out)


def gate_trials_by_eyepos(trial_recs, min_open=0.5, max_dev=None, ref=None):
    """Per-trial gate + deviations. The on-target reference eye position is data-driven: median of per-trial STIM
    eye position (the engaged, tightest eye position -- the stimulus sits at the fixation location and is what concentrates
    eye position), falling back to fixation-window eye position only if no stim eye position exists. A trial is KEPT if its stim-window
    eyes-open fraction >= ``min_open`` (i.e. eyes open for at least that fraction of the stim period) AND (if
    ``max_dev`` given, in the same units as the eye position, i.e. volts) its mean stim eye position is within ``max_dev`` of
    the reference. Returns (keep_mask, deviations, ref)."""
    stim = np.array([r['stim_xy'] for r in trial_recs if r.get('stim_xy')], float)
    fix = np.array([r['fixation_xy'] for r in trial_recs if r.get('fixation_xy')], float)
    if ref is None:
        ref = (np.median(stim, axis=0) if len(stim)
               else (np.median(fix, axis=0) if len(fix) else np.array([np.nan, np.nan])))
    keep, devs = [], []
    for r in trial_recs:
        s = r.get('stim_xy')
        dev = float(np.hypot(s[0] - ref[0], s[1] - ref[1])) if s else np.nan
        devs.append(dev)
        keep.append(bool(r.get('stim_open', 0.0) >= min_open and
                         (max_dev is None or (np.isfinite(dev) and dev <= max_dev))))
    return np.array(keep), np.array(devs), np.asarray(ref, float)


def load_calibration(session_path, cal_glob='*EyeTrackingCalibration*/*calibration.p'):
    """Load a session's eye-tracking calibration (a separate ``*_EyeTrackingCalibration`` recording, typically
    in the SAME date dir as the session). Returns {``positions`` (k,2) screen deg, ``volts`` (k,2) eye V,
    ``accel_baseline``, ``path``} or None if absent. The voltage key is a legacy misnomer ('coarse oculomatic
    values') -- the tracker was EyeLoop."""
    base = os.path.dirname(session_path.rstrip('/'))
    hits = sorted(glob.glob(os.path.join(base, cal_glob)))
    if not hits:
        return None
    cal = pickle.load(open(hits[0], 'rb'))
    return {'positions': np.asarray(cal['calibration positions'], float),
            'volts': np.asarray(cal['coarse oculomatic values'], float),
            'accel_baseline': np.asarray(cal.get('accelerometer baseline', []), float),
            'path': hits[0]}


def fit_calibration(cal, extra_anchors=None, drop=None, weights=None, grid_half_deg=None):
    """Least-squares AFFINE eye-voltage -> screen-degree map. Optionally SUPPLEMENT the formal grid with
    task-derived ``extra_anchors`` (from ``calculate_stim_derived_eyepos_anchors`` -- any stimulus position, not just 0,0; each
    may carry a 'weight'), DROP unreliable grid positions (``drop`` = list of (deg_x, deg_y), e.g. the
    zero-completion targets from ``calculate_eyepos_calibration_quality``), and/or ``weights`` the formal points. Returns
    ``M`` (2x2 deg/V), ``offset``, residuals, ``deg_per_v`` (+ principal), ``n_points``, a ``quality`` verdict
    {'good','rough','unusable'} and ``reliable`` bool. Marmoset grids are typically nonlinear/poorly
    participated -> at best a ROUGH scale; 'unusable' => report VOLTS only, no degrees."""
    pos = [tuple(map(float, p)) for p in cal['positions']]
    volt = [tuple(map(float, v)) for v in cal['volts']]
    w = list(weights) if weights is not None else [1.0] * len(pos)
    if drop:
        drop = {tuple(map(float, d)) for d in drop}
        keep = [i for i, p in enumerate(pos) if p not in drop]
        pos, volt, w = [pos[i] for i in keep], [volt[i] for i in keep], [w[i] for i in keep]
    for a in (extra_anchors or []):
        pos.append(tuple(map(float, a['pos_deg'])))
        volt.append(tuple(map(float, a['volt'])))
        w.append(float(a.get('weight', 1.0)))
    pos, volt, w = np.array(pos, float), np.array(volt, float), np.array(w, float)
    A = np.hstack([volt, np.ones((len(volt), 1))])
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A * sw, pos * sw, rcond=None)
    M, offset = coef[:2], coef[2]
    resid = np.hypot(*(A @ coef - pos).T)
    detM = float(np.linalg.det(M))
    sv = np.linalg.svd(M, compute_uv=False)
    cond = float(sv[0] / sv[1]) if sv[1] > 0 else np.inf
    upos, ui = np.unique(pos, axis=0, return_inverse=True)   # aggregate duplicate positions (e.g. a task anchor
    uvolt = np.array([volt[ui == k].mean(0) for k in range(len(upos))])  # colliding with a grid point) for monotonicity
    inv = 0                                        # grid monotonicity: volt-X up along deg-X rows, volt-Y up along deg-Y cols
    for row in np.unique(upos[:, 1]):
        m = upos[:, 1] == row
        if m.sum() > 1:
            inv += int(np.any(np.diff(uvolt[m, 0][np.argsort(upos[m, 0])]) <= 0))
    for col in np.unique(upos[:, 0]):
        m = upos[:, 0] == col
        if m.sum() > 1:
            inv += int(np.any(np.diff(uvolt[m, 1][np.argsort(upos[m, 1])]) <= 0))
    half = grid_half_deg or float(np.abs(pos).max())
    resid_frac = float(resid.mean() / half) if half else np.inf
    if detM <= 0 or cond > CAL_COND_MAX or inv > CAL_MAX_INVERSIONS or resid_frac > CAL_RESID_FRAC_DROP:
        quality = 'unusable'
    elif resid_frac < CAL_RESID_FRAC_GOOD and inv == 0 and cond < 2:
        quality = 'good'
    else:
        quality = 'rough'
    return {'M': M, 'offset': offset, 'resid_mean': float(resid.mean()), 'resid_max': float(resid.max()),
            'deg_per_v': float(np.sqrt(abs(detM))), 'deg_per_v_principal': sv, 'cond': cond,
            'n_points': int(len(pos)), 'n_inversions': int(inv), 'resid_frac': resid_frac, 'quality': quality,
            'reliable': quality != 'unusable',
            'reason': 'n=%d det=%.2f cond=%.1f inv=%d resid=%.0f%%grid' % (len(pos), detM, cond, inv, 100 * resid_frac)}


def convert_volts_to_deg(fc, xy):
    """Map eye voltage(s) to screen degrees with a fitted affine calibration (``fit_calibration`` result).
    ``xy`` is (2,) or (n, 2). Returns the same shape. Meaningful only when ``fc['reliable']``."""
    xy = np.atleast_2d(np.asarray(xy, float))
    return np.squeeze(xy @ fc['M'] + fc['offset'])


def calculate_eyepos_calibration_quality(session_path, cal_glob='*EyeTrackingCalibration*/*.log'):
    """Per-target quality of the FORMAL calibration, parsed from the calibration session's grid-target phase
    (``grid target fixation start / completed`` per ``grid_target.pos``): fixations COMPLETED, RESTARTS, and
    the eye-voltage + dispersion of the successful holds. Poorly-trained animals leave many targets with 0
    completions / many restarts (esp. the periphery they won't saccade to) -- feed the zero-completion
    positions to ``fit_calibration(drop=...)``. Returns {pos_deg: {'n','done','restart','volt_hold','medrad'}}
    or {} if no calibration log is found."""
    base = os.path.dirname(session_path.rstrip('/'))
    logs = [l for l in glob.glob(os.path.join(base, cal_glob)) if 'disptimes' not in l]
    if not logs:
        return {}
    aip = glob.glob(os.path.join(os.path.dirname(logs[0]), '*_AIdata.p'))
    ai = np.asarray(pickle.load(open(aip[0], 'rb')), float) if aip else None
    lost = calculate_eyepos_loss_mask(ai) if ai is not None else None
    rx = re.compile(r'grid target trial (\d+), grid target ([a-z ]+?), grid_target\.pos = '
                    r'\[\s*([-\d.]+)\s+([-\d.]+)\], AI_data\.shape = \((\d+)')
    trials = {}
    for line in open(logs[0]):
        m = rx.search(line)
        if not m:
            continue
        tr, ev = int(m.group(1)), m.group(2).strip()
        p, s = (float(m.group(3)), float(m.group(4))), int(m.group(5))
        d = trials.setdefault(tr, {'pos': p, 'fix': [], 'done': None})
        if ev == 'fixation start':
            d['fix'].append(s)
        elif ev == 'fixation completed':
            d['done'] = s
    out, holds = {}, {}
    for d in trials.values():
        p = d['pos']
        q = out.setdefault(p, {'n': 0, 'done': 0, 'restart': 0})
        q['n'] += 1
        q['restart'] += max(0, len(d['fix']) - 1)
        if d['done'] and d['fix'] and ai is not None:
            q['done'] += 1
            s0 = max([f for f in d['fix'] if f <= d['done']] or [d['fix'][0]])
            g = ai[s0:d['done'], :2][~lost[s0:d['done']]]
            if len(g) >= 5:
                holds.setdefault(p, []).append(g)
    for p, q in out.items():
        if p in holds:
            g = np.vstack(holds[p])
            mx, my = float(np.median(g[:, 0])), float(np.median(g[:, 1]))
            q['volt_hold'], q['medrad'] = (mx, my), float(np.median(np.hypot(g[:, 0] - mx, g[:, 1] - my)))
        else:
            q['volt_hold'], q['medrad'] = None, None
    return out


def plot_eyepos_density(oc, which=('all', 'stim', 'nonstim'), outdir='output', tag=None, bins=100,
              eye_ch=EYE_CH, rail_v=RAIL_V):
    """2D eye-position density panels (log-scaled histogram2d + median & 1-SD ellipse) for the requested subsets over
    the whole session -- a vectorized, ~1e6-sample-friendly replacement for the per-trial scatter/KDE in
    analysis_for_images.py. ``which`` selects any of 'all' (whole session), 'stim' (during stimulus),
    'nonstim' (ISI/fixation). Signal-loss (railed/zeroed) samples are dropped. Returns the saved figure path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from datetime import datetime, timezone

    ai = oc['ai']
    x, y = ai[:, eye_ch[0]], ai[:, eye_ch[1]]
    good = ~calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    stim = calculate_stim_sample_mask(oc)
    subs = {'all': good, 'stim': good & stim, 'nonstim': good & ~stim}
    lab = {'all': 'all session', 'stim': 'during stimulus', 'nonstim': 'non-stimulus (ISI/fixation)'}
    which = [w for w in which if w in subs]
    xr = np.percentile(x[good], [0.5, 99.5])
    yr = np.percentile(y[good], [0.5, 99.5])

    fig, axes = plt.subplots(1, len(which), figsize=(4.3 * len(which), 4.4),
                             squeeze=False, sharex=True, sharey=True)
    for ax, w in zip(axes[0], which):
        m = subs[w]
        H, xe, ye = np.histogram2d(x[m], y[m], bins=bins, range=[xr, yr])
        ax.imshow(np.log1p(H.T), origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  aspect='auto', cmap='magma')
        mx, my = float(np.median(x[m])), float(np.median(y[m]))
        ax.plot(mx, my, 'c+', ms=12, mew=2)
        ax.add_patch(Ellipse((mx, my), 2 * np.std(x[m]), 2 * np.std(y[m]), fill=False, edgecolor='c', lw=1.5))
        ax.set_title('%s\n%d samp (%.1f%%)' % (lab[w], int(m.sum()), 100 * m.mean()))
        ax.set_xlabel('eye X (V)')
    axes[0][0].set_ylabel('eye Y (V)')
    fig.suptitle('eye position density (log color) — %s' % (tag or os.path.basename(oc.get('log_path', 'session'))))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'eyepos_density_%s_%s.png' % (tag or 'session',
                     datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_eyepos_by_phase(oc, subsets=('stim', 'fixation', 'isi'),
                     diffs=(('stim', 'isi'), ('fixation', 'isi'), ('stim', 'fixation')),
                     outdir='output', tag=None, grid=100, n_sub=15000, seed=0, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-phase eye position KDE (top row, each with its 68% BCEA ellipse + median marker) and pairwise normalized
    density DIFFERENCES (bottom row), to reveal how eye position concentration differs by trial phase -- differences
    the raw histogram flattens. KDE is on a random subsample (full ``gaussian_kde`` is O(N^2), infeasible at
    ~1e6 samples). Also prints the per-phase dispersion table (medRad + BCEA). Returns the saved figure path.
    """
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from datetime import datetime, timezone

    rng = np.random.default_rng(seed)
    ai = oc['ai']
    x, y = ai[:, eye_ch[0]], ai[:, eye_ch[1]]
    good = ~calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    masks = {k: v & good for k, v in calculate_phase_sample_masks(oc).items() if k in subsets}
    xr = np.percentile(x[good], [1, 99])
    yr = np.percentile(y[good], [1, 99])
    gx, gy = np.linspace(xr[0], xr[1], grid), np.linspace(yr[0], yr[1], grid)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.vstack([GX.ravel(), GY.ravel()])

    def kde(m):
        idx = np.where(m)[0]
        if idx.size > n_sub:
            idx = rng.choice(idx, n_sub, replace=False)
        d = gaussian_kde(np.vstack([x[idx], y[idx]]))(pts).reshape(GX.shape)
        return d / d.sum()

    dens = {k: kde(masks[k]) for k in subsets}
    ds = {k: calculate_eyepos_dispersion(x[masks[k]], y[masks[k]]) for k in subsets}
    print('  per-phase eye position dispersion (eyes-open samples):')
    for k in subsets:
        s = ds[k]
        print('    %-9s medRad %.3f V | BCEA68 %.3f V^2 | median (%.3f, %.3f) | n=%d'
              % (k, s['medrad'], s['bcea'], s['median'][0], s['median'][1], s['n']))

    ext = [xr[0], xr[1], yr[0], yr[1]]
    ncol = max(len(subsets), len(diffs))
    fig, ax = plt.subplots(2, ncol, figsize=(4.4 * ncol, 8.6), squeeze=False, sharex=True, sharey=True)
    for a_ in ax.ravel():
        a_.set_visible(False)
    for j, k in enumerate(subsets):
        a_ = ax[0, j]; a_.set_visible(True)
        a_.imshow(dens[k], origin='lower', extent=ext, aspect='auto', cmap='magma')
        s = ds[k]
        C = np.cov(x[masks[k]], y[masks[k]])
        vals, vecs = np.linalg.eigh(C)
        o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * np.sqrt(vals * (-2 * np.log(1 - 0.68)))
        a_.add_patch(Ellipse(s['median'], w, h, angle=ang, fill=False, edgecolor='c', lw=1.6))
        a_.plot(*s['median'], 'c+', ms=11, mew=2)
        a_.set_title('%s (KDE)\nmedRad %.2f V | BCEA68 %.2f' % (k, s['medrad'], s['bcea']))
        a_.set_ylabel('eye Y (V)')
    for j, (aN, bN) in enumerate(diffs):
        a_ = ax[1, j]; a_.set_visible(True)
        dd = dens[aN] - dens[bN]
        v = float(np.abs(dd).max())
        im = a_.imshow(dd, origin='lower', extent=ext, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v)
        a_.set_title('%s − %s (Δdensity)' % (aN, bN))
        a_.set_xlabel('eye X (V)')
        fig.colorbar(im, ax=a_, fraction=0.046)
    ax[1, 0].set_ylabel('eye Y (V)')
    fig.suptitle('eye position by trial phase — %s' % (tag or 'session'))
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'eyepos_phases_%s_%s.png' % (tag or 'session',
                     datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _demo(session_path, outdir='output'):
    from datetime import datetime, timezone
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    oc = load_eye_ai(session_path)
    ai, anchors = oc['ai'], oc['anchors']
    ai_per_fr = np.polyfit(anchors[:, 0], anchors[:, 1], 1)[0]
    lost = calculate_eyepos_loss_mask(ai)
    recs = calculate_trial_eyepos(oc)
    ref = calculate_stim_derived_eyepos_ref(oc, reduce='peak')
    keep, devs, ref = gate_trials_by_eyepos(recs, min_open=0.5, ref=ref)
    stim_open = np.array([r.get('stim_open', np.nan) for r in recs], float)

    print('session   :', os.path.basename(session_path.rstrip('/'))[:60])
    print('AI        : %d samp x %d ch | ~%.1f samp/frame (~%.1f Hz @6.36)' % (ai.shape[0], ai.shape[1], ai_per_fr, ai_per_fr * 6.364))
    print('overall eyes-open fraction (whole session): %.3f' % (1 - lost.mean()))
    print('trials parsed: %d | with stim window: %d | with fixation window: %d'
          % (len(recs), int(np.isfinite(stim_open).sum()), sum('fixation_xy' in r for r in recs)))
    print('per-trial stim eyes-open: median %.3f | frac trials <0.5 open: %.3f'
          % (np.nanmedian(stim_open), np.nanmean(stim_open < 0.5)))
    print('data-driven eye position reference (stim KDE-peak, eye V): (%.3f, %.3f)' % (ref[0], ref[1]))
    print('stim eye position deviation from ref (V): median %.3f | p90 %.3f' % (np.nanmedian(devs), np.nanpercentile(devs, 90)))
    print('gate keep (min_open=0.5): %d / %d (%.1f%%)' % (keep.sum(), len(keep), 100 * keep.mean()))

    masks = calculate_phase_sample_masks(oc)
    for k in ('stim', 'fixation', 'isi'):
        m = masks[k] & ~lost
        if m.any():
            s = calculate_eyepos_dispersion(ai[m, 0], ai[m, 1])
            print('  %-9s dispersion: medRad %.3f V | BCEA68 %.3f V^2 | n=%d' % (k, s['medrad'], s['bcea'], s['n']))
    stim_ok = masks['stim'] & ~lost
    near_radius_v = 2 * float(np.median(np.hypot(ai[stim_ok, 0] - ref[0], ai[stim_ok, 1] - ref[1])))
    near_sec = calculate_eye_near_stim_sec(oc, ref, near_radius_v, framerate=6.364)
    print('eye-near-stim radius (2x stim spread about ref): %.2f V | near-stim sec: median %.2f p10 %.2f'
          % (near_radius_v, np.nanmedian(near_sec), np.nanpercentile(near_sec, 10)))
    for mn in (0.5, 0.75, 1.0):
        print('  keep (eye near stim >= %.2fs, late looks excluded): %d / %d'
              % (mn, int(np.nansum(near_sec >= mn)), int(np.isfinite(near_sec).sum())))
    cal = load_calibration(session_path)
    if cal is not None:
        fc = fit_calibration(cal)
        print('calibration: quality=%s (%s) | ~%.2f deg/V (%.2f & %.2f principal) | resid %.2f deg'
              % (fc['quality'], fc['reason'], fc['deg_per_v'], fc['deg_per_v_principal'][0],
                 fc['deg_per_v_principal'][1], fc['resid_mean']))
    else:
        print('calibration: none found')
    anchors = calculate_stim_derived_eyepos_anchors(oc)
    if anchors:
        a0 = anchors[0]
        print('stim eye-position anchors: %d position(s) | best pos=%s volt=(%.3f, %.3f) conc=%.2f n=%d'
              % (len(anchors), a0['pos_deg'], a0['volt'][0], a0['volt'][1], a0['conc'], a0['n']))
        if cal is not None:
            fq = calculate_eyepos_calibration_quality(session_path)
            drop = [p for p, qq in fq.items() if qq['done'] == 0] + [a['pos_deg'] for a in anchors]
            fc2 = fit_calibration(cal, extra_anchors=anchors, drop=drop)
            print('  + supplement/clean fit: quality=%s | ~%.2f deg/V | resid %.2f deg (dropped %d, +%d anchors)'
                  % (fc2['quality'], fc2['deg_per_v'], fc2['resid_mean'], len(drop), len(anchors)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].hist(stim_open[np.isfinite(stim_open)], bins=20, color='C0')
    ax[0].axvline(0.5, color='r', ls='--'); ax[0].set_xlabel('per-trial eyes-open fraction'); ax[0].set_ylabel('trials')
    sx = np.array([r['stim_xy'] for r in recs if r.get('stim_xy')], float)
    sc = ax[1].scatter(sx[:, 0], sx[:, 1], c=[r['stim_open'] for r in recs if r.get('stim_xy')], cmap='viridis', s=12)
    ax[1].plot(ref[0], ref[1], 'r+', ms=15, mew=2, label='stim-median ref')
    ax[1].set_xlabel('eye X (V)'); ax[1].set_ylabel('eye Y (V)'); ax[1].legend(fontsize=8); fig.colorbar(sc, ax=ax[1], label='eyes-open')
    fig.suptitle('eye-AI position — %s' % os.path.basename(session_path.rstrip('/'))[:44])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'eyetracking_%s_%s.png' % (os.path.basename(session_path.rstrip('/'))[:24],
                     datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    print('saved', p)
    tag = os.path.basename(session_path.rstrip('/'))[:24]
    print('saved', plot_eyepos_density(oc, tag=tag, outdir=outdir))
    print('saved', plot_eyepos_by_phase(oc, tag=tag, outdir=outdir))


if __name__ == '__main__':
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1
          else 'suite2p_results/Cadbury/20221016d/152643tUTC_SP_depth200um_fov0730x0730um_'
               'res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly')
