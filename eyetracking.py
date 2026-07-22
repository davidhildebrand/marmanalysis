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

import indicators

EYE_CH = (0, 1)     # eye X, Y channels (analog input; any tracker piped to the DAQ)
RAIL_V = 9.0        # |v| above this = DAQ rail (signal loss)

# Calibration usability gate (affine voltage->degree fit). The grid is ~+/-5 deg; drop to VOLTS-only if the
# map folds, is too anisotropic, non-monotonic, or its residual is a large fraction of the grid half-range.
CAL_COND_MAX = 4.0          # max deg/V axis-ratio (anisotropy) before untrustworthy
CAL_MAX_INVERSIONS = 1      # allow at most this many non-monotonic grid edges
CAL_RESID_FRAC_GOOD = 0.15  # residual < this fraction of grid half-range => 'good'
CAL_RESID_FRAC_DROP = 0.50  # residual > this fraction => 'unusable' (report volts only)

# Indicator kinetics -> response timing.
# jGCaMP8s (Zhang et al. Looger 2023 Nature, https://doi.org/10.1038/s41586-023-05828-9): a few-MILLISECOND
# half-RISE -- negligible at our ~6 Hz imaging, so calcium tracks spike onset inside a single frame -- and a
# ~200 ms single-AP HALF-decay in mouse brain, with ~2x the 1-AP sensitivity of the best prior sensor.
# We parameterise the decay by its TIME CONSTANT tau (fall to 1/e), because tau is the standard quantity: it
# is what suite2p takes and what OASIS deconvolution / most reports use. tau = t_half / ln2, so the paper's
# t_half ~0.20 s -> tau ~0.29 s. The value + its provenance live in the indicators lookup (indicators.py) --
# the single source shared with process_with_suite2p.py -- and an empirical per-session tau can be measured
# via signal_quality.calculate_indicator_decay_tau. NB a ~0.29 s decay CANNOT manufacture the multi-second
# response plateau we observe: that plateau is sustained FIRING, not indicator ringing.
INDICATOR_DECAY_TAU_SEC = indicators.indicator_tau('jGCaMP8s')   # single source of truth: indicators.py
# The gate's response-timing parameter (``expected_response_dur_sec``: how long a look's response takes to
# register + linger -- used as BOTH the minimum viewing duration and the excluded tail before stim offset)
# defaults to ONE decay tau, i.e. INDICATOR_DECAY_TAU_SEC itself. It is deliberately NOT a separate module
# constant (it would just equal tau); pass a different multiple to the gate's ``expected_response_dur_sec``
# argument if you ever want > 1 tau.


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


def calculate_eye_near_stim_sec(oc, stim_derived_eyepos_ref, near_radius_v,
                                expected_response_dur_sec=INDICATOR_DECAY_TAU_SEC,
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


# Per-trial eye-position gating. Every criterion is SWITCHABLE via ``mode`` so its cost can be MEASURED
# rather than assumed, and the default is 'none' -- eye tracking excludes nothing unless explicitly asked.
# Use ``compare_eyepos_gate_modes`` to see all modes side by side on one session.
GATE_MODES = ('none', 'fraction_open', 'fraction_near', 'duration_near', 'landing_window')


def _default_near_radius_v(oc, ref, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Data-driven 'near the stimulus' radius: 2x the median radial deviation of eyes-open stim-window eye
    position about the reference. Calibration-free (absolute degrees are untrustworthy here) and self-scaling
    per session.

    TODO -- ANCHOR THIS TO RECEPTIVE-FIELD SIZE, not to stimulus size and not to session spread. The tolerance
    that matters is the one deciding whether an eye movement actually changed what a neuron saw, and that is
    set by RF size, not by how big the stimulus happens to be. The literature ties tolerance to STIMULUS size
    only informally and in contradictory directions: enforced-fixation studies deliberately set the window
    SMALLER than the stimulus -- Mansouri et al. Tanaka 2006 J Neurosci, 4 deg window vs 5-7 deg samples,
    https://doi.org/10.1523/jneurosci.5238-05.2006 -- whereas free-viewing studies set the tolerance EQUAL to
    it: Park et al. Leopold 2022 Sci Adv, tolerance "approximately the size of the movie stimulus",
    https://doi.org/10.1126/sciadv.abm2054. The papers that argue it properly compare eye scatter against RF
    size instead: Tang et al. Jiang 2018 Curr Biol report macaque V1 two-photon eye-position SD < 0.05 deg,
    "significantly smaller than the typical receptive field sizes" (0.3-0.8 deg at 3-5 deg eccentricity),
    https://doi.org/10.1016/j.cub.2017.11.039.

    No marmoset RF estimate exists for area PD specifically. Working proxies: (a) marmoset MT (nearby area) --
    Rosa & Elston 1998 J Comp Neurol give RF size (sqrt of RF area) = 2.40 * ecc^0.58 deg, i.e. ~2.4 deg at the
    fovea rising to ~7 deg at 6 deg eccentricity across our 12x7.2 deg stimulus (MTc slightly larger,
    2.92*ecc^0.59); https://doi.org/10.1002/(sici)1096-9861(19980420)393:4<505::aid-cne9>3.0.co;2-4 . (b) PD is
    a FACE patch, whose RFs are typically LARGER and more position-tolerant than MT (cf. macaque face-patch /
    Issa & DiCarlo 2012 above), so loose fixation should matter even less. So the on-stimulus tolerance ought
    to be a FEW degrees (MT-scale) to larger (face-patch); the session-spread heuristic below is a STAND-IN
    until a marmoset PD RF is measured directly -- sanity-check it against these scales, and do not read its
    volt value as a precise degree tolerance."""
    ai = oc['ai']
    m = calculate_phase_sample_masks(oc)['stim'] & ~calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    if not m.any():
        return float('nan')
    return 2.0 * float(np.median(np.hypot(ai[m, eye_ch[0]] - ref[0], ai[m, eye_ch[1]] - ref[1])))


def _eyepos_trial_quantities(oc, ref, near_radius_v, expected_response_dur_sec, exclude_late_looks,
                             framerate, eye_ch, rail_v):
    """Per-trial raw quantities every gate mode is built from -- computed once so modes are cheap to compare.
    ``fraction_open`` / ``fraction_near``: fraction of the stim window with eyes open / eye near the stimulus.
    ``eye_near_stim_sec``: ABSOLUTE seconds near the stimulus within the countable window (see
    ``exclude_late_looks``). ``landing_sec``: seconds from stim onset to the FIRST near-stim sample.
    ``drive_sec``: near-stim seconds from that landing to stim offset -- the real stimulus drive a
    landing-anchored response window would integrate. NaN where a trial has no usable stim window."""
    ai = oc['ai']
    lost = calculate_eyepos_loss_mask(ai, eye_ch, rail_v)
    near = is_eye_near_stim(ai[:, eye_ch[0]], ai[:, eye_ch[1]], ref, near_radius_v) & ~lost
    f2a = map_acqfr_to_ai_sample(oc['anchors'])
    samp_per_sec = float(np.polyfit(oc['anchors'][:, 0], oc['anchors'][:, 1], 1)[0]) * framerate
    tail = int(expected_response_dur_sec * samp_per_sec) if exclude_late_looks else 0
    keys = ('fraction_open', 'fraction_near', 'eye_near_stim_sec', 'landing_sec', 'drive_sec')
    q = {k: [] for k in keys}
    for tr in sorted(oc['trials']):
        ph = oc['trials'][tr]
        s = ph.get('stim start') or ph.get('fixation end') or ph.get('ISI end')
        e = ph.get('stim end')
        i0, i1 = (int(f2a(s[0])), int(f2a(e[0]))) if (s and e) else (0, 0)
        if i1 <= i0:
            for k in keys:
                q[k].append(np.nan)
            continue
        q['fraction_open'].append(float((~lost[i0:i1]).mean()))
        q['fraction_near'].append(float(near[i0:i1].mean()))
        j1 = i1 - tail
        q['eye_near_stim_sec'].append(float(near[i0:j1].sum()) / samp_per_sec if j1 > i0 else 0.0)
        hit = np.flatnonzero(near[i0:i1])
        q['landing_sec'].append(float(hit[0]) / samp_per_sec if hit.size else np.nan)
        q['drive_sec'].append(float(near[i0 + hit[0]:i1].sum()) / samp_per_sec if hit.size else 0.0)
    return {k: np.array(v, float) for k, v in q.items()}


def gate_trials_by_eyepos(oc, mode='none', stim_derived_eyepos_ref=None, near_radius_v=None,
                          min_fraction=0.5, min_eye_near_stim_sec=None,
                          expected_response_dur_sec=INDICATOR_DECAY_TAU_SEC, exclude_late_looks=True,
                          framerate=6.364, eye_ch=EYE_CH, rail_v=RAIL_V):
    """Per-trial eye-position gate with a SWITCHABLE criterion. Returns a dict: ``mode``, ``passed`` (bool array,
    one per trial -- True where the trial PASSED the gate criterion, i.e. is kept), the reference/radius/
    thresholds actually used, and EVERY per-trial quantity, so you can
    re-threshold or compare modes without recomputing. Default ``mode='none'`` -- eye tracking excludes nothing
    unless you ask, so any exclusion stays a deliberate, reversible choice.

    MODES
      'none'
          Keep every trial. The baseline: use it to measure what any other mode actually costs.
      'fraction_open'
          Keep if the eyes were open for >= ``min_fraction`` of the stim window. Unbiased about WHERE the
          animal looked -- it only rejects blinks / tracker dropout.
      'fraction_near'
          Keep if the eye position was near the stimulus for >= ``min_fraction`` of the stim window.
          CAVEAT for both fraction modes: a fraction is NOT comparable across sessions run with different
          stimulus durations -- the same fraction demands a different absolute amount of viewing. Prefer a
          duration mode when comparing across stim_dur conditions.
      'duration_near'
          Keep if the eye was near the stimulus for >= ``min_eye_near_stim_sec`` ABSOLUTE seconds (default
          ``expected_response_dur_sec``) -- stim_dur-fair by construction. With ``exclude_late_looks`` (the
          default) those seconds are counted only over [stim_start, stim_end - expected_response_dur_sec]:
          a look landing in that final stretch drives calcium that develops after stim offset, outside a
          stim-period-mean response window, so it cannot be measured and must not earn a trial its keep.
      'landing_window'
          The least exclusionary option, and the complement to 'duration_near'. Keep if the eye lands near the
          stimulus at ANY time in the stim window and then supplies >= ``min_eye_near_stim_sec`` of real drive
          before stim offset. Instead of discarding a late look it reports ``landing_sec``, so the response
          window can START at the landing and run past stim offset into the early ISI -- where that look's
          calcium actually appears (~1 half-decay, ~200 ms, after the drive). NB this function only GATES:
          handing ``landing_sec`` to the response calculation to actually shift the window is a separate,
          explicit step, because a per-trial window changes the response measure itself.

    ``stim_derived_eyepos_ref`` defaults to the stimulus-derived eye-position mode; ``near_radius_v`` to a
    data-driven radius (2x the stim-window spread about that reference). Pass the session ``framerate`` so
    sample counts convert to real seconds."""
    if mode not in GATE_MODES:
        raise ValueError('mode must be one of %s, got %r' % (GATE_MODES, mode))
    n_tr = len(oc['trials'])
    trial_ids = np.array(sorted(oc['trials']))               # aligns 1:1 with ``passed`` (and every q array)
    out = {'mode': mode, 'trial': trial_ids,
           'stim_derived_eyepos_ref': stim_derived_eyepos_ref, 'near_radius_v': near_radius_v}
    if mode == 'none':
        out['passed'] = np.ones(n_tr, bool)
        return out
    if stim_derived_eyepos_ref is None:
        stim_derived_eyepos_ref = calculate_stim_derived_eyepos_ref(oc, eye_ch=eye_ch, rail_v=rail_v)
    if near_radius_v is None:
        near_radius_v = _default_near_radius_v(oc, stim_derived_eyepos_ref, eye_ch, rail_v)
    if min_eye_near_stim_sec is None:
        min_eye_near_stim_sec = expected_response_dur_sec
    q = _eyepos_trial_quantities(oc, stim_derived_eyepos_ref, near_radius_v, expected_response_dur_sec,
                                 exclude_late_looks, framerate, eye_ch, rail_v)
    with np.errstate(invalid='ignore'):                     # NaN (no stim window) compares False = fails the gate
        if mode == 'fraction_open':
            passed = q['fraction_open'] >= min_fraction
        elif mode == 'fraction_near':
            passed = q['fraction_near'] >= min_fraction
        elif mode == 'duration_near':
            passed = q['eye_near_stim_sec'] >= min_eye_near_stim_sec
        else:                                               # 'landing_window'
            passed = np.isfinite(q['landing_sec']) & (q['drive_sec'] >= min_eye_near_stim_sec)
    out.update(q)
    out.update({'passed': np.asarray(passed, bool), 'stim_derived_eyepos_ref': stim_derived_eyepos_ref,
                'near_radius_v': near_radius_v, 'min_fraction': min_fraction,
                'min_eye_near_stim_sec': min_eye_near_stim_sec,
                'expected_response_dur_sec': expected_response_dur_sec})
    return out


def compare_eyepos_gate_modes(oc, min_fraction=0.5, min_eye_near_stim_sec=None,
                              expected_response_dur_sec=INDICATOR_DECAY_TAU_SEC, exclude_late_looks=True,
                              framerate=6.364, eye_ch=EYE_CH, rail_v=RAIL_V, verbose=True):
    """Run EVERY gate mode on one session and tabulate how many trials each keeps, so the cost of each
    criterion is visible rather than assumed (the point of keeping them switchable). The reference and radius
    are computed once and shared, so the modes differ only in their keep rule. Returns {mode: result-dict}."""
    ref = calculate_stim_derived_eyepos_ref(oc, eye_ch=eye_ch, rail_v=rail_v)
    radius = _default_near_radius_v(oc, ref, eye_ch, rail_v)
    res = {m: gate_trials_by_eyepos(oc, mode=m, stim_derived_eyepos_ref=ref, near_radius_v=radius,
                                    min_fraction=min_fraction, min_eye_near_stim_sec=min_eye_near_stim_sec,
                                    expected_response_dur_sec=expected_response_dur_sec,
                                    exclude_late_looks=exclude_late_looks, framerate=framerate,
                                    eye_ch=eye_ch, rail_v=rail_v) for m in GATE_MODES}
    if verbose:
        n = max(len(oc['trials']), 1)
        mn = expected_response_dur_sec if min_eye_near_stim_sec is None else min_eye_near_stim_sec
        print('  eye-position gate modes | ref=(%.3f, %.3f) V  near_radius=%.2f V  min_fraction=%.2f  '
              'min_near=%.2fs  expected_response_dur=%.2fs' % (ref[0], ref[1], radius, min_fraction, mn,
                                                              expected_response_dur_sec))
        crit = {'none': 'no eye-tracking exclusion (baseline)',
                'fraction_open': 'eyes open >= %.0f%% of stim window' % (100 * min_fraction),
                'fraction_near': 'eye near stim >= %.0f%% of stim window' % (100 * min_fraction),
                'duration_near': 'eye near stim >= %.2fs abs (late looks %s)'
                                 % (mn, 'excluded' if exclude_late_looks else 'counted'),
                'landing_window': 'lands near stim anytime, then >= %.2fs drive' % mn}
        print('    %-15s %6s %7s   %s' % ('mode', 'pass', 'excl', 'criterion'))
        for m in GATE_MODES:
            k = int(res[m]['passed'].sum())
            print('    %-15s %6d %6.1f%%   %s' % (m, k, 100 * (1 - k / n), crit[m]))
    return res


def _stim_onset_acqfr(ph):
    """Stimulus-onset acquisition frame (raw, from the text log) for one trial's phase dict, with the same
    stim-start fallback the rest of the module uses. NaN if the trial has no stim window."""
    s = ph.get('stim start') or ph.get('fixation end') or ph.get('ISI end')
    return float(s[0]) if s else np.nan


def calculate_eyepos_stimlog_keep(oc, gate, stimlog, acqfr_col='acqfr_stim_i', match_tol_frames=3):
    """Project a per-eye-trial gate result onto the response-table's stimlog rows: a per-row keep mask
    (True = keep) that ``response_table.exclude_trials_by_stimlog`` turns into (condition, repeat) exclusions.

    JOIN KEY = stimulus-onset acquisition frame. The eye tracker and the stimlog are stamped from the SAME
    'stim start' log line, so each eye trial's onset frame equals the stimlog's ``acqfr_stim_i`` (up to the
    global -1 acqfr correction applied to the stimlog). Matching on that physical frame -- nearest eye trial
    within ``match_tol_frames`` -- is robust to trial RENUMBERING between the structured (pickle) and text
    stimlog sources (which a trial-number join would silently get wrong) and self-validates: a bad join shows
    up as a low ``n_matched``. Inter-trial spacing is tens of frames, so a few-frame tolerance is unambiguous.

    A stimlog row with no eye trial within tolerance is KEPT (the eye record can't judge it) and counted in
    ``n_unmatched`` -- eye gating never drops a trial it has no evidence about. Returns (keep_rows, info),
    info summarizing the join (n_matched / n_unmatched / n_excluded) and the reference/radius the gate used."""
    eye_trials = sorted(oc['trials'])
    passed = np.asarray(gate['passed'], bool)
    if passed.shape[0] != len(eye_trials):
        raise ValueError('gate passed length %d != n eye trials %d' % (passed.shape[0], len(eye_trials)))
    onset = np.array([_stim_onset_acqfr(oc['trials'][t]) for t in eye_trials], float)
    onset = np.where(np.isfinite(onset), onset, np.inf)      # trials without a stim window never match
    acq = np.asarray(stimlog[acqfr_col].to_numpy(dtype=float, na_value=np.nan), float)
    landing = np.asarray(gate.get('landing_sec', np.full(len(eye_trials), np.nan)), float)
    keep_rows = np.ones(len(acq), bool)
    landing_rows = np.full(len(acq), np.nan)        # per-stimlog-row landing time (for a shifted response window)
    n_matched = n_unmatched = n_excluded = 0
    for i, a in enumerate(acq):
        if not np.isfinite(a):
            n_unmatched += 1
            continue
        j = int(np.argmin(np.abs(onset - a)))
        if abs(onset[j] - a) <= match_tol_frames:
            n_matched += 1
            if j < landing.size:
                landing_rows[i] = landing[j]
            if not passed[j]:
                keep_rows[i] = False
                n_excluded += 1
        else:
            n_unmatched += 1
    info = {'landing_sec_rows': landing_rows,
            'mode': gate.get('mode'), 'n_rows': int(len(acq)), 'n_matched': n_matched,
            'n_unmatched': n_unmatched, 'n_excluded': n_excluded, 'match_tol_frames': match_tol_frames,
            'stim_derived_eyepos_ref': gate.get('stim_derived_eyepos_ref'),
            'near_radius_v': gate.get('near_radius_v')}
    return keep_rows, info


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
    stim_open = np.array([r.get('stim_open', np.nan) for r in recs], float)
    devs = np.array([np.hypot(r['stim_xy'][0] - ref[0], r['stim_xy'][1] - ref[1])
                     if r.get('stim_xy') else np.nan for r in recs], float)

    print('session   :', os.path.basename(session_path.rstrip('/'))[:60])
    print('AI        : %d samp x %d ch | ~%.1f samp/frame (~%.1f Hz @6.36)' % (ai.shape[0], ai.shape[1], ai_per_fr, ai_per_fr * 6.364))
    print('overall eyes-open fraction (whole session): %.3f' % (1 - lost.mean()))
    print('trials parsed: %d | with stim window: %d | with fixation window: %d'
          % (len(recs), int(np.isfinite(stim_open).sum()), sum('fixation_xy' in r for r in recs)))
    print('per-trial stim eyes-open: median %.3f | frac trials <0.5 open: %.3f'
          % (np.nanmedian(stim_open), np.nanmean(stim_open < 0.5)))
    print('data-driven eye position reference (stim KDE-peak, eye V): (%.3f, %.3f)' % (ref[0], ref[1]))
    print('stim eye position deviation from ref (V): median %.3f | p90 %.3f' % (np.nanmedian(devs), np.nanpercentile(devs, 90)))

    masks = calculate_phase_sample_masks(oc)
    for k in ('stim', 'fixation', 'isi'):
        m = masks[k] & ~lost
        if m.any():
            s = calculate_eyepos_dispersion(ai[m, 0], ai[m, 1])
            print('  %-9s dispersion: medRad %.3f V | BCEA68 %.3f V^2 | n=%d' % (k, s['medrad'], s['bcea'], s['n']))
    compare_eyepos_gate_modes(oc, framerate=6.364)
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
