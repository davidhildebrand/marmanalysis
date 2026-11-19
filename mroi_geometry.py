#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry of ScanImage mROI strips from a resonant scanner: where does each raw pixel sit in space?
(Roadmap cross-cutting backlog item 4; DH 2026-11-17, starting from his res_scanner_sine_playground.m.)

The 2pRAM strips are saved with ``hScan2D.uniformSampling = True``: along the resonant (fast, x) axis the pixels are
evenly spaced in TIME. The mirror moves sinusoidally, so the pixels are NOT evenly spaced in space. With N pixels over
the temporal fill fraction f of a half period, pixel k (centre) sits at mirror phase

    phi_k = (pi/2) * f * u_k,    u_k = (2k + 1 - N) / N   in (-1, 1)

and at position  x_k = (w/2) * sin(phi_k) / sin(a),  a = (pi/2) * f,  w = strip width  (sin(a) is the spatial fill
fraction ScanImage reports). The local pixel size is  (w/N) * (a / sin a) * cos(phi_k): largest at the strip centre,
where the mirror is fastest, smallest at the edges. For f = 0.718 that is 1.25x the nominal pixel at the centre and
0.54x at the edges, a 2.3-fold change across every strip, and positions deviate from the evenly-spaced assumption by up
to 9.4% of the half width (20 um in a 438 um strip).

This module (1) states that model, (2) MEASURES the local pixel size from the data with no model, using the tissue's
own texture along the slow axis as a ruler, (3) undistorts strips by resampling each line onto an evenly spaced grid
(the fast-scan centre is upsampled by interpolation, the slow-scan edges are averaged), and (4) measures how adjacent
strips really sit relative to each other (overlap and vertical offset) from the columns they share.

(5) STITCHES the strips: each raw strip goes onto one evenly spaced canvas in a single interpolation (sinusoid undone,
odd/even lines aligned, seam position and row offset applied to a fraction of a pixel), and where two strips overlap
the two copies are combined as chosen: 'crossfade', 'split', 'average', 'left' or 'right'. The blending modes warn (or
raise) when the two copies do not agree, which means the seam position is wrong.

Run:  .venv/bin/python marmanalysis/mroi_geometry.py --session Curly20231103d162355tUTC [--frames 300]
      .venv/bin/python marmanalysis/mroi_geometry.py --session Cadbury20221016d --stitch --overlap-mode crossfade
Writes output/mroi_geometry_<slug>_<UTC>.png or output/mroi_stitch_<slug>_mode-<mode>_seam-<seam>_<UTC>.png and prints
the measurements.
"""
import argparse
import functools
import glob
import os
import re
import sys
from datetime import datetime, timezone
from warnings import warn

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sessionio

STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')


# ----------------------------------------------------------------------------------------------------- the model ----
def pixel_phase(n_px, fill_fraction_temporal, edges=False):
    """Mirror phase (rad) at each pixel centre, or at the n_px + 1 pixel edges, for uniform sampling in time."""
    u = (2.0 * np.arange(n_px + 1) - n_px) / n_px if edges else (2.0 * np.arange(n_px) + 1 - n_px) / n_px
    return 0.5 * np.pi * fill_fraction_temporal * u


def pixel_positions(n_px, fill_fraction_temporal, width=2.0, edges=False):
    """Position of each pixel centre (or edge) across a strip of ``width``, origin at the strip centre.
    ``fill_fraction_temporal = 0`` gives evenly spaced pixels (no sinusoid)."""
    if fill_fraction_temporal <= 0:
        phase = pixel_phase(n_px, 1.0, edges)
        return 0.5 * width * phase / (0.5 * np.pi)
    a = 0.5 * np.pi * fill_fraction_temporal
    return 0.5 * width * np.sin(pixel_phase(n_px, fill_fraction_temporal, edges)) / np.sin(a)


def pixel_sizes(n_px, fill_fraction_temporal, width=2.0):
    """Extent in space of each pixel: the distance between its two edges."""
    return np.diff(pixel_positions(n_px, fill_fraction_temporal, width, edges=True))


def resampling_matrix(n_px, fill_fraction_temporal, n_out=None, oversample=8, shift=0.0):
    """Sparse (n_out, n_px) matrix R that undistorts a line: ``line_out = R @ line_raw``. The raw line is linearly
    interpolated at ``oversample`` evenly spaced points per output pixel and those are averaged, so the fast-scan
    centre is upsampled smoothly and the finely sampled edges are averaged rather than thrown away. Rows sum to 1
    (intensity is preserved). ``n_out`` defaults to n_px: same pixel count, now evenly spaced.
    ``shift`` (raw pixels) reads the line that far to the right: the odd/even line correction, which is a constant
    shift in raw (time) pixels and therefore has to be applied here, before the positions are mapped to space."""
    n_out = n_px if n_out is None else int(n_out)
    centres = pixel_positions(n_px, fill_fraction_temporal)                      # in [-1, 1], increasing
    fine = (np.arange(n_out * oversample) + 0.5) / (n_out * oversample) * 2.0 - 1.0
    k = np.interp(fine, centres, np.arange(n_px)) + shift                        # fractional raw pixel index
    k0 = np.clip(np.floor(k).astype(int), 0, n_px - 2)
    w1 = np.clip(k - k0, 0.0, 1.0)
    rows = np.repeat(np.arange(n_out), oversample)
    R = sparse.coo_matrix((np.concatenate([1 - w1, w1]) / oversample,
                           (np.concatenate([rows, rows]), np.concatenate([k0, k0 + 1]))), shape=(n_out, n_px))
    return R.tocsr()


def undistort(strips, fill_fraction_temporal, n_out=None, line_offset=0.0):
    """Resample the last (x) axis of ``strips`` (..., rows, columns) onto evenly spaced pixels.
    ``line_offset`` (raw pixels, from bidirectional_offset) is how far the content of odd lines sits to the right of
    the even lines; each parity is read half that far towards the other, so the two meet in the middle."""
    strips = np.asarray(strips, np.float32)
    out = np.empty(strips.shape[:-1] + (strips.shape[-1] if n_out is None else int(n_out),), np.float32)
    for parity, shift in ((0, -0.5 * line_offset), (1, 0.5 * line_offset)):
        R = resampling_matrix(strips.shape[-1], fill_fraction_temporal, n_out, shift=shift)
        part = strips[..., parity::2, :]
        out[..., parity::2, :] = np.asarray((R @ part.reshape(-1, part.shape[-1]).T).T).reshape(part.shape[:-1] + (R.shape[0],))
    return out


# ---------------------------------------------------------------------------------------------------- the data ----
def header_fields(tif, names=('uniformSampling', 'bidirectional', 'linePhase', 'sampleRate', 'pixelBinFactor')):
    """A few ``SI.hScan2D`` values straight from the raw file's header text. Much faster on a long recording than
    metadata.get_metadata, which also walks every frame."""
    from ScanImageTiffReader import ScanImageTiffReader
    with ScanImageTiffReader(tif) as reader:
        text = reader.metadata()
    out = {}
    for name in names:
        m = re.search(r'^SI\.hScan2D\.%s\s*=\s*(.+?)\s*$' % name, text, re.M)
        value = m.group(1) if m else None
        if value in ('true', 'false'):
            value = value == 'true'
        elif value is not None:
            try:
                value = float(value)
            except ValueError:
                pass
        out[name] = value
    return out


@functools.lru_cache(maxsize=8)
def scan_settings(session):
    """The scan parameters that fix the strip geometry. Fill fractions, scanner frequency and strip layout come from
    the session metadata (metadata.extract_useful_metadata); fields that older metadata files do not hold are read
    from the raw file header. Cached per session."""
    md = sessionio.load_metadata(session)
    out = {'fill_fraction_temporal': float(md['fill_fraction_temporal']),
           'fill_fraction_spatial': float(md['fill_fraction_spatial']),
           'resonant_scanner_frequency': float(md['resonant_scanner_frequency']),
           'objective_resolution': float(md['objective_resolution']), 'n_strips': int(md['n_strips']),
           'strip_px': np.asarray(md['strip']['size_px'], int), 'strip_um': np.asarray(md['strip']['size_um'], float),
           'resolution_umpx': np.asarray(md['fov']['resolution_umpx'], float),
           'centres_deg': np.array([r['center_deg'][0] for r in md['mrois']['lrsort']], float),
           'size_deg': np.asarray(md['mrois']['lrsort'][0]['size_deg'], float),
           'rows': [tuple(int(v) for v in r['acqstrip_ys_px']) for r in md['mrois']['lrsort']]}
    extra = ('uniform_sampling', 'bidirectional', 'line_phase', 'sample_rate', 'pixel_bin_factor')
    if all(k in md for k in extra):
        out.update({k: md[k] for k in extra})
    else:
        h = header_fields(raw_file(session))
        out.update(uniform_sampling=h['uniformSampling'], bidirectional=h['bidirectional'], line_phase=h['linePhase'],
                   sample_rate=h['sampleRate'], pixel_bin_factor=h['pixelBinFactor'])
    return out


def raw_file(session):
    hits = sorted(f for f in glob.glob(os.path.join(session, '*_00001.tif')) if '_preprocd' not in f)
    if not hits:
        raise FileNotFoundError('no raw ScanImage file (*_00001.tif) in %s' % session)
    return hits[0]


def read_strips(session, n_frames=300, chunk=4, seed=0):
    """A random sample of RAW frames split into strips: float32 (n, n_strips, h, w), strips ordered left to right.
    Frames are read in short runs spread over the whole recording. Nothing is registered or filtered."""
    from ScanImageTiffReader import ScanImageTiffReader
    rows = scan_settings(session)['rows']
    with ScanImageTiffReader(raw_file(session)) as reader:
        total = reader.shape()[0]
        starts = np.sort(np.random.default_rng(seed).choice(total - chunk, max(1, n_frames // chunk), replace=False))
        frames = np.concatenate([reader.data(beg=int(s), end=int(s) + chunk) for s in starts]).astype(np.float32)
    return np.stack([frames[:, y0:y1] for y0, y1 in rows], axis=1)


# ---------------------------------------------------------------------------- measuring the pixel size, no model ----
def highpass(strips, sigma=10.0):
    """Remove structure coarser than ``sigma`` pixels from every frame (cell-scale texture is kept). ``strips`` is
    (..., n_strips, rows, columns). The strips are laid side by side before filtering, so that a strip edge is not a
    filter boundary: the neighbouring strip continues the tissue there (give or take the few-um seam overlap)."""
    s, h, w = strips.shape[-3:]
    out = np.empty(strips.shape, np.float32)
    for idx in np.ndindex(strips.shape[:-3]):
        wide = np.concatenate(list(strips[idx]), axis=1)                        # (rows, n_strips * columns)
        wide = wide - ndimage.gaussian_filter(wide, sigma)
        out[idx] = np.stack(np.split(wide, s, axis=1))
    return out


def texture_pixel_size(z, yres, lags=(4, 5, 6, 8), dmax=24):
    """Local pixel size along x from texture, with no geometric model.

    ``z`` (n, h, w): high-passed frames of ONE strip. Tissue texture is isotropic on average and the slow (y) axis has
    a known, constant pixel size ``yres``. For each column c and lag L, the covariance between columns c and c + L is
    compared with the same columns' covariance along y: the y-lag d* that gives the same value says how far apart in
    space the two columns are, so the local x pixel size is ``yres * d* / L`` (assigned to c + L/2).
    Only covariances at non-zero lags are used, so uncorrelated pixel noise does not enter, and only EVEN y-lags, so a
    bidirectional scan-phase offset between odd and even lines does not enter either.
    Returns (len(lags), w), NaN where the comparison is not defined."""
    n, h, w = z.shape
    ds = np.arange(2, dmax + 1, 2)
    cy = np.stack([np.einsum('fhc,fhc->c', z[:, :-d], z[:, d:]) / (n * (h - d)) for d in ds])       # (n_d, w)
    cy0 = np.where(cy[0] > 0, cy[0], np.nan)
    ry = cy / cy0                                                    # relative to the lag-2 covariance
    est = np.full((len(lags), w), np.nan)
    for i, lag in enumerate(lags):
        cx = np.einsum('fhc,fhc->c', z[:, :, :-lag], z[:, :, lag:]) / (n * h)
        rx = cx / np.sqrt(cy0[:-lag] * cy0[lag:])
        ref = 0.5 * (ry[:, :-lag] + ry[:, lag:])
        for c in range(w - lag):
            f = ref[:, c]
            if not np.all(np.isfinite(f)) or not np.isfinite(rx[c]):
                continue
            rising = np.nonzero(np.diff(f) >= 0)[0]
            stop = rising[0] + 1 if rising.size else len(f)
            if stop < 3 or not (f[stop - 1] < rx[c] <= f[0]):
                continue                                             # outside the monotone range of the reference
            d_star = float(PchipInterpolator(f[:stop][::-1], ds[:stop][::-1])(rx[c]))
            est[i, int(round(c + lag / 2.0))] = yres * d_star / lag
    return est


def pixel_size_profile(strips, yres, sigma=10.0, lags=(4, 5, 6, 8), dmax=24):
    """Median local x pixel size at each within-strip column, pooled over lags and strips: (w,), plus per-strip (s, w)."""
    z = highpass(strips, sigma)
    per_strip = np.array([np.nanmedian(texture_pixel_size(z[:, s], yres, lags, dmax), axis=0)
                          for s in range(strips.shape[1])])
    return np.nanmedian(per_strip, axis=0), per_strip


def binned(profile, n_bins=15, margin=0.06):
    """Median of a per-column profile in ``n_bins`` bins over the strip interior (``margin`` of the width is left out
    at each edge, where the beam is still being unblanked and the estimate is unreliable). Returns (centres, values)."""
    w = len(profile)
    edges = np.linspace(margin * w, (1 - margin) * w, n_bins + 1)
    k = np.arange(w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    values = np.array([np.nanmedian(profile[(k >= lo) & (k < hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
    return centres, values


def flatness(profile, margin=0.06):
    """How uneven a pixel-size profile is: the centre third relative to the two outer sixths, leaving out ``margin``
    of the width at each edge (1 = flat)."""
    w = len(profile)
    k = np.arange(w)
    centre = np.nanmedian(profile[np.abs(k - (w - 1) / 2) < w / 6])
    edge = np.nanmedian(profile[(np.abs(k - (w - 1) / 2) > w / 3) & (k >= margin * w) & (k < (1 - margin) * w)])
    return float(centre / edge)


def fit_fill_fraction(profile, n_px, grid=None):
    """The fill fraction whose sinusoid profile shape best matches a measured pixel-size profile (free overall scale,
    because blur and texture anisotropy bias the ruler's absolute value). Returns (best, grid, residuals)."""
    grid = np.linspace(0.0, 0.95, 96) if grid is None else np.asarray(grid)
    centres, values = binned(profile)                                 # robust: interior, binned medians
    ok = np.isfinite(values)
    res = []
    for f in grid:
        model = np.interp(centres, np.arange(n_px), pixel_sizes(n_px, f))[ok]
        scale = np.sum(values[ok] * model) / np.sum(model ** 2)
        res.append(np.sqrt(np.mean((values[ok] - scale * model) ** 2)) / np.mean(values[ok]))
    res = np.array(res)
    return float(grid[np.argmin(res)]), grid, res


# ------------------------------------------------------------------------------------------- odd / even lines ----
def bidirectional_offset(image, max_shift=4):
    """Shift along x (pixels) between odd and even lines of one strip image, by cross-correlation with a parabolic
    sub-pixel peak. Uniform sampling makes a scan-phase error a CONSTANT pixel shift across the line."""
    img = image - ndimage.gaussian_filter(image, 10)
    n = (img.shape[0] // 2) * 2
    even, odd = img[0:n:2], img[1:n:2]
    ref = 0.5 * (even[:-1] + even[1:])                                # even lines interpolated to the odd rows
    odd = odd[:-1]
    shifts = np.arange(-max_shift, max_shift + 1)
    cc = []
    for s in shifts:
        a = ref[:, max_shift:-max_shift]
        b = odd[:, max_shift + s:odd.shape[1] - max_shift + s]
        cc.append(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    cc = np.array(cc)
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        denom = cc[i - 1] - 2 * cc[i] + cc[i + 1]
        return float(shifts[i] + (0.5 * (cc[i - 1] - cc[i + 1]) / denom if denom else 0.0))
    return float(shifts[i])


# --------------------------------------------------------------------------------------------------- the seams ----
def seam_fit(left, right, fill_fraction_temporal, width, margin=28, max_dy=4, min_corr=0.3, n_parts=4):
    """Where does the right strip sit relative to the left one? ``left`` and ``right`` are high-passed mean images of
    two adjacent RAW strips.

    Every left-edge column is correlated (along y) with every right-edge column. A pair of columns that image the same
    tissue are each other's best match, and both columns' positions are known from the pixel model, so every such
    MUTUAL best match gives one estimate of the spacing between the two strip centres. The answer is their median.
    One-sided best matches are not used: a column beyond the overlap has no partner and simply picks the other strip's
    edge column, which would bias the estimate. Returns overlap (= width - spacing, same units as ``width``), the
    scatter of the matches in nominal pixels (``misfit``; lower = the pixel model describes the seam better), how many
    matches there were, the overlap from ``n_parts`` row blocks, and the vertical offset (whole and sub-pixel)."""
    n_px = left.shape[1]
    pos = pixel_positions(n_px, fill_fraction_temporal, width)
    xl, xr = pos[n_px - margin:], pos[:margin]

    def matrix(rows, dy):
        a = left[rows, n_px - margin:]
        b = right[rows, :margin]
        a = a[max_dy + dy:a.shape[0] - max_dy + dy]
        b = b[max_dy:b.shape[0] - max_dy]
        za, zb = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
        return za.T @ zb / za.shape[0]

    def fit(m):
        best_b, best_a = m.argmax(1), m.argmax(0)
        floor = max(min_corr, 0.6 * m.max())
        est = []
        for a in range(margin):
            b = int(best_b[a])
            if best_a[b] != a or m[a, b] < floor:
                continue
            frac = 0.0
            if 0 < b < margin - 1:
                denom = m[a, b - 1] - 2 * m[a, b] + m[a, b + 1]
                frac = float(np.clip(0.5 * (m[a, b - 1] - m[a, b + 1]) / denom, -0.5, 0.5)) if denom else 0.0
            est.append(xl[a] - np.interp(b + frac, np.arange(margin), xr))
        if len(est) < 2:
            return np.nan, np.nan, len(est)
        est = np.array(est)
        spacing = float(np.median(est))
        return spacing, float(1.4826 * np.median(np.abs(est - spacing))), len(est)

    everything = slice(None)
    dys = np.arange(-max_dy, max_dy + 1)
    peaks = np.array([matrix(everything, int(dy)).max() for dy in dys])
    j = int(np.argmax(peaks))
    dy = int(dys[j])
    dy_sub = float(dy)
    if 0 < j < len(dys) - 1:                                          # parabola through the peak and its neighbours
        denom = peaks[j - 1] - 2 * peaks[j] + peaks[j + 1]
        dy_sub = float(dy + (0.5 * (peaks[j - 1] - peaks[j + 1]) / denom if denom else 0.0))
    m = matrix(everything, dy)
    spacing, scatter, n_pairs = fit(m)
    h = left.shape[0]
    parts = [fit(matrix(slice(i * h // n_parts, (i + 1) * h // n_parts), dy))[0] for i in range(n_parts)]
    return {'overlap': float(width - spacing), 'overlap_parts': [float(width - s) for s in parts], 'dy': dy,
            'dy_subpixel': dy_sub, 'misfit': float(scatter / (width / n_px)), 'n_pairs': int(n_pairs),
            'peak': float(m.max()), 'matrix': m, 'spacing': float(spacing)}


# ------------------------------------------------------------------------------------- start-of-sweep ramp ----
def edge_gain(mean_strips, dark=0.0, floor=0.05):
    """Relative brightness of each raw pixel for even and odd lines: (2, n_px), 1 = full brightness.

    Every sweep of the resonant mirror STARTS with the beam still turning on, so the first ten or so pixels of a line
    are dim. Even lines start at one strip edge and odd lines at the other, so each edge has one dim parity and one
    good one. The two parities see the same tissue column, so their ratio isolates the ramp exactly: the dimmer parity
    gets ratio < 1 and the brighter one gets 1. Pooled over strips and rows; ``dark`` is the detector offset."""
    m = np.asarray(mean_strips, float) - dark
    even, odd = m[:, 0::2].mean((0, 1)), m[:, 1::2].mean((0, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.stack([np.minimum(1.0, even / odd), np.minimum(1.0, odd / even)])
    return np.clip(np.nan_to_num(g, nan=1.0), floor, 1.0)


def flyback_gaps(session, margin=4):
    """Row ranges of the flyback lines between strips, where the beam is blanked: [(first, last + 1), ...]."""
    rows = scan_settings(session)['rows']
    return [(rows[i][1] + margin, rows[i + 1][0] - margin) for i in range(len(rows) - 1)
            if rows[i + 1][0] - rows[i][1] > 2 * margin + 2]


def additive_level(frames, gaps, statistic='mean'):
    """The level a pixel reads with no fluorescence, per frame: ``statistic`` of the flyback rows of raw frames
    (n, H, W). Use the MEAN. It is the detector offset plus any light that does not come from the beam (ambient light,
    the stimulus monitor). That light arrives as sparse events, so it lifts the mean of the flyback rows and of every
    tissue pixel alike while leaving the median where it was: in Cadbury20221016d152643tUTC the mean sits about 23
    counts above the median for as long as the stimulus program runs."""
    if not gaps:
        return np.zeros(len(frames))
    fly = np.concatenate([frames[:, g0:g1].reshape(len(frames), -1) for g0, g1 in gaps], axis=1)
    return np.median(fly, axis=1) if statistic == 'median' else fly.mean(axis=1)


def dark_level(session, n_frames=300, seed=0, statistic='mean', chunk=4):
    """The additive level under the tissue, averaged over frames spread through the recording (the same frames
    read_strips samples by default): the mean of the flyback rows. See additive_level for why not the median."""
    from ScanImageTiffReader import ScanImageTiffReader
    gaps = flyback_gaps(session)
    if not gaps:
        return 0.0
    with ScanImageTiffReader(raw_file(session)) as reader:
        total = reader.shape()[0]
        starts = np.sort(np.random.default_rng(seed).choice(total - chunk, max(1, n_frames // chunk), replace=False))
        levels = [additive_level(reader.data(beg=int(b_), end=int(b_) + chunk).astype(np.float32), gaps, statistic)
                  for b_ in starts]
    return float(np.mean(np.concatenate(levels)))


RAMP_MODES = ('select', 'correct', 'none')
SELECT_POWER = 8          # 'select': a copy counts in proportion to its brightness to this power (0.5 -> 0.4%)


def ramp_mode(ramp):
    """Normalize the ``ramp`` argument (True and False are the older spellings of 'correct' and 'none')."""
    ramp = {True: 'correct', False: 'none'}.get(ramp, ramp) if isinstance(ramp, bool) else ramp
    if ramp not in RAMP_MODES:
        raise ValueError('ramp must be one of %s, got %r' % (RAMP_MODES, ramp))
    return ramp


def read_strip_blocks(session, n_blocks=6, frames_per_block=120):
    """Contiguous runs of RAW frames evenly spread from the start to the end of the recording. Returns a dict:
    strips float32 (n_blocks, frames, n_strips, h, w), starts (first frame of each block), and the additive level of
    every frame from the flyback rows, ``level`` (mean) and ``level_median``, both (n_blocks, frames)."""
    from ScanImageTiffReader import ScanImageTiffReader
    rows, gaps = scan_settings(session)['rows'], flyback_gaps(session)
    strips, level, level_median = [], [], []
    with ScanImageTiffReader(raw_file(session)) as reader:
        total = reader.shape()[0]
        frames_per_block = min(frames_per_block, total // n_blocks)
        starts = np.linspace(0, total - frames_per_block, n_blocks).astype(int)
        for b_ in starts:
            blk = reader.data(beg=int(b_), end=int(b_) + frames_per_block).astype(np.float32)
            strips.append(np.stack([blk[:, y0:y1] for y0, y1 in rows], axis=1))
            level.append(additive_level(blk, gaps))
            level_median.append(additive_level(blk, gaps, 'median'))
    return {'strips': np.stack(strips), 'starts': starts, 'level': np.stack(level), 'level_median': np.stack(level_median)}


def ramp_checks(session, n_blocks=8, frames_per_block=120):
    """Is the start-of-sweep ramp what ramp='correct' assumes: a FIXED, purely MULTIPLICATIVE gain about a KNOWN
    additive level? Flat-fielding computes (value - level) / gain + level for every frame, so it needs all three.
    Measured on runs of raw frames spread over the recording:

      additive level   the mean of the beam-blanked flyback rows, block by block. If it moves during the recording
                       (ambient or stimulus-monitor light does), a flat-field with one fixed level is wrong by the
                       change divided by the gain: ``level_error_at_dimmest`` is that error at the dimmest ramp pixel,
                       as a fraction of the median tissue signal.
      fixed gain       the gain profile of every block, each computed about its own additive level. ``drift`` is the
                       range across blocks over the ramp pixels (gain < 0.9), as a fraction of the gain; ``noise`` is
                       the same between the two interleaved halves of one block (what "no change" looks like).
                       ``drift_fixed_level`` repeats it with one level for the whole recording, i.e. what a fixed
                       flat-field would actually face.
      purely a gain    the gain measured separately on the dimmest and the brightest third of the tissue (sorted by
                       the neighbouring rows, so that the sort is independent of the pixels compared). A pure gain
                       gives the same number in both. ``level_implied`` is the additive level that would make them
                       equal: an estimate from the tissue itself, independent of the flyback rows.
    Returns a dict; print_ramp_checks formats it."""
    blk = read_strip_blocks(session, n_blocks, frames_per_block)
    strips, level = blk['strips'], blk['level'].mean(1)                          # per block
    level_all = float(level.mean())
    means = strips.mean(1)                                                       # (blocks, strips, rows, px)
    overall = edge_gain(np.mean([m - lv for m, lv in zip(means, level)], axis=0))
    zone = overall < 0.9
    per_block = np.stack([edge_gain(m, lv) for m, lv in zip(means, level)])
    per_block_fixed = np.stack([edge_gain(m, level_all) for m in means])
    halves = np.stack([np.stack([edge_gain(b_[h::2].mean(0), lv) for h in (0, 1)]) for b_, lv in zip(strips, level)])

    def spread(g):
        r = (g.max(0) - g.min(0))[zone] / overall[zone]
        return (float(np.median(r)), float(r.max())) if r.size else (np.nan, np.nan)
    noise = np.abs(halves[:, 0] - halves[:, 1])[:, zone] / overall[zone]
    m = np.mean([mm - lv for mm, lv in zip(means, level)], axis=0)               # level-subtracted, (strips, rows, px)
    even, odd = m[:, 0::2], m[:, 1::2]
    n_pair = min(even.shape[1], odd.shape[1])
    even, odd = even[:, :n_pair], odd[:, :n_pair]
    low, high, implied = [], [], []
    for parity, (dim, bright) in enumerate(((even, odd), (odd, even))):
        for k in np.flatnonzero(zone[parity]):
            d, b_ = dim[:, 1:-1, k].ravel(), bright[:, 1:-1, k].ravel()
            proxy = (bright[:, :-2, k] + bright[:, 2:, k]).ravel()               # the rows above and below
            order = np.argsort(proxy)
            third = max(1, order.size // 3)
            dl, bl, dh, bh = d[order[:third]].mean(), b_[order[:third]].mean(), d[order[-third:]].mean(), b_[order[-third:]].mean()
            low.append(dl / bl)
            high.append(dh / bh)
            denom = dl + bh - dh - bl
            implied.append((dl * bh - dh * bl) / denom if abs(denom) > 1e-9 else np.nan)
    low, high = np.array(low), np.array(high)
    signal = float(np.median(m))
    g_min = float(overall[zone].min()) if zone.any() else 1.0
    level_range = float(level.max() - level.min())
    return {'slug': sessionio.session_slug(session), 'gain': overall, 'n_ramp_px': int(zone.sum()), 'gain_min': g_min,
            'block_starts': blk['starts'], 'frames_per_block': int(strips.shape[1]),
            'level': level, 'level_median': blk['level_median'].mean(1), 'level_range': level_range,
            'level_frame_sd': float(np.median(blk['level'].std(1))),
            'signal_median': signal, 'level_error_at_dimmest': level_range / g_min / max(signal, 1e-9),
            'drift_median': spread(per_block)[0], 'drift_max': spread(per_block)[1],
            'drift_fixed_level_median': spread(per_block_fixed)[0], 'drift_fixed_level_max': spread(per_block_fixed)[1],
            'noise_median': float(np.median(noise)) if noise.size else np.nan,
            'noise_p95': float(np.percentile(noise, 95)) if noise.size else np.nan,
            'gain_dim_tissue': low, 'gain_bright_tissue': high,
            'brightness_dependence': float(np.median(np.abs(high - low) / (0.5 * (high + low)))) if low.size else np.nan,
            'level_implied': level_all + float(np.nanmedian(implied)) if implied else np.nan, 'level_all': level_all,
            'per_block': per_block}


def print_ramp_checks(c):
    print('%s | start-of-sweep ramp checks | %d blocks of %d frames, first frames %s'
          % (c['slug'], len(c['block_starts']), c['frames_per_block'], ', '.join(str(int(b_)) for b_ in c['block_starts'])))
    print('   ramp: %d pixels with gain < 0.9, smallest gain %.2f | median tissue signal above the additive level: %.1f counts'
          % (c['n_ramp_px'], c['gain_min'], c['signal_median']))
    print('   additive level (mean of the beam-blanked flyback rows) per block: %s'
          % ' '.join('%.1f' % v for v in c['level']))
    print('      the median of the same rows: %s' % ' '.join('%.1f' % v for v in c['level_median']))
    print('      range across blocks %.1f counts = %.0f%% of the median signal; frame-to-frame SD within a block %.1f counts'
          % (c['level_range'], 100 * c['level_range'] / max(c['signal_median'], 1e-9), c['level_frame_sd']))
    print('      a flat-field with ONE fixed level would be wrong at the dimmest ramp pixel by up to %.0f%% of the median signal'
          % (100 * c['level_error_at_dimmest']))
    print('   fixed gain?     range of the gain across blocks: median %.1f%%, largest %.1f%% of the gain (each block about its own '
          'level) | with one fixed level: median %.1f%%, largest %.1f%% | measurement noise: median %.1f%%, 95th percentile %.1f%%'
          % (100 * c['drift_median'], 100 * c['drift_max'], 100 * c['drift_fixed_level_median'], 100 * c['drift_fixed_level_max'],
             100 * c['noise_median'], 100 * c['noise_p95']))
    print('   purely a gain?  gain on the dimmest third of the tissue vs the brightest third differs by %.1f%% (median over ramp '
          'pixels) | additive level implied by the tissue %.1f vs %.1f from the flyback rows'
          % (100 * c['brightness_dependence'], c['level_implied'], c['level_all']))


# ---------------------------------------------------------------------------------------------------- stitching ----
OVERLAP_MODES = ('crossfade', 'split', 'average', 'left', 'right')


def overlap_weights(x, left_edges, width, mode):
    """Weight of every strip at canvas positions ``x``: (n_strips, len(x)), summing to 1 wherever a strip covers.

    Where two strips overlap, ``mode`` decides how the two copies of the tissue are combined:
      'crossfade'  the left strip fades out linearly across the overlap while the right one fades in
      'split'      a hard cut in the middle of the overlap: each strip keeps its own half
      'average'    both strips count equally across the whole overlap
      'left'       the left strip is used for the whole overlap
      'right'      the right strip is used for the whole overlap
    Where there is a gap instead (negative overlap), each side is extended to the middle of the gap."""
    if mode not in OVERLAP_MODES:
        raise ValueError('overlap mode must be one of %s, got %r' % (OVERLAP_MODES, mode))
    n = len(left_edges)
    w = np.zeros((n, len(x)))
    for i in range(n):
        lo, hi = left_edges[i], left_edges[i] + width
        if i > 0:
            lo = min(lo, 0.5 * (left_edges[i - 1] + width + left_edges[i]))      # a gap on the left: reach its middle
        if i < n - 1:
            hi = max(hi, 0.5 * (left_edges[i] + width + left_edges[i + 1]))      # a gap on the right
        w[i] = (x >= lo) & (x < hi) if i < n - 1 else (x >= lo) & (x <= hi)
    for i in range(n - 1):
        a, b = left_edges[i + 1], left_edges[i] + width                          # the overlap is [a, b]
        zone = (x >= a) & (x < b)
        if b <= a or not zone.any():
            continue
        if mode == 'crossfade':
            left = (b - x[zone]) / (b - a)
        elif mode == 'split':
            left = (x[zone] < 0.5 * (a + b)).astype(float)
        elif mode == 'average':
            left = np.full(zone.sum(), 0.5)
        else:
            left = np.full(zone.sum(), 1.0 if mode == 'left' else 0.0)
        w[i, zone], w[i + 1, zone] = left, 1.0 - left
    total = w.sum(0)
    return w / np.where(total > 0, total, 1.0)


def strip_to_canvas_matrix(n_px, fill_fraction_temporal, width, left_edge, n_canvas, pixel, weights=None, shift=0.0,
                           oversample=8):
    """Sparse (n_canvas, n_px) matrix that puts one RAW strip line onto the evenly spaced canvas in ONE interpolation:
    it undoes the sinusoid, applies the odd/even line ``shift`` (raw pixels), places the strip at ``left_edge`` to a
    fraction of a pixel, and scales each canvas column by its blending weight (default: 1 inside the strip)."""
    centres = pixel_positions(n_px, fill_fraction_temporal, width)               # raw pixel centres, strip-centred
    fine = (np.arange(n_canvas * oversample) + 0.5) / oversample * pixel - left_edge - 0.5 * width
    inside = (fine >= -0.5 * width) & (fine <= 0.5 * width)
    k = np.interp(fine, centres, np.arange(n_px)) + shift                        # clamps to the edge pixel outside
    k0 = np.clip(np.floor(k).astype(int), 0, n_px - 2)
    w1 = np.clip(k - k0, 0.0, 1.0)
    rows = np.repeat(np.arange(n_canvas), oversample)
    if weights is None:
        scale = inside.astype(float)
    else:
        scale = np.repeat(np.asarray(weights, float), oversample)                # a strip may be extended into a gap
    cover = np.bincount(rows, weights=(scale > 0).astype(float), minlength=n_canvas)
    norm = scale / np.where(cover[rows] > 0, cover[rows], 1.0)                   # average over the fine samples used
    R = sparse.coo_matrix((np.concatenate([(1 - w1) * norm, w1 * norm]),
                           (np.concatenate([rows, rows]), np.concatenate([k0, k0 + 1]))), shape=(n_canvas, n_px))
    return R.tocsr()


def stitch_plan(n_px, fill_fraction_temporal, width, overlaps, pixel, mode='crossfade', row_offsets=None,
                line_offset=0.0, gain=None, dark=0.0, ramp='select'):
    """Everything needed to turn raw strips into one evenly sampled plane.

    ``overlaps``: overlap between each pair of neighbours, in the units of ``width`` (negative = a gap), e.g. from
    seam_fit. ``pixel``: output pixel size along x. ``row_offsets``: per seam, how many rows (may be fractional) the
    right strip's content sits BELOW where the left strip shows the same tissue... as returned by seam_fit ('dy' or
    'dy_subpixel': left row r + dy matches right row r). ``line_offset``: odd/even line offset in raw pixels.
    ``gain`` (2, n_px), from edge_gain, and ``dark``: the start-of-sweep ramp. ``ramp`` says what to do about it:
      'select'   (default) NO pixel value is rescaled. Where two strips overlap, each line takes the copy that was
                 acquired at full brightness (a copy counts in proportion to its gain to the power SELECT_POWER), which
                 is possible because the two strips ramp on opposite line parities. Where only a ramped copy exists
                 (the outer edges of the field of view, or strips that do not overlap) the dimming stays in the data.
                 The output is a weighted average of raw values with fixed weights, and does not depend on ``dark``.
      'correct'  flat-field every frame: the light summed across strips is divided by the gain collected, so a single
                 strip is divided by its gain and, in an overlap, the dim copy counts for less (with mode 'average'
                 the photon-optimal combination). Brightness is right everywhere, at the price of dividing functional
                 data by a measured gain: see ramp_checks for what that assumes.
      'none'     the gain is ignored: copies are blended by geometry alone and the ramp stays wherever it was.
    Without a gain all three are the same.
    Returns a dict with the canvas size, each strip's left edge, weights, row shift, and its two sampling matrices
    (even and odd lines)."""
    overlaps = np.atleast_1d(np.asarray(overlaps, float))
    n = len(overlaps) + 1
    left_edges = np.concatenate([[0.0], np.cumsum(width - overlaps)])
    n_canvas = int(round((left_edges[-1] + width) / pixel))
    x = (np.arange(n_canvas) + 0.5) * pixel
    weights = overlap_weights(x, left_edges, width, mode)
    row_offsets = np.zeros(n - 1) if row_offsets is None else np.atleast_1d(np.asarray(row_offsets, float))
    row_shift = np.concatenate([[0.0], np.cumsum(row_offsets)])
    mats = [tuple(strip_to_canvas_matrix(n_px, fill_fraction_temporal, width, left_edges[i], n_canvas, pixel, weights[i], s)
                  for s in (-0.5 * line_offset, 0.5 * line_offset)) for i in range(n)]
    ramp = ramp_mode(ramp)
    gain = np.ones((2, n_px)) if gain is None else np.asarray(gain, float)
    cover = [tuple(np.asarray(R @ np.ones(n_px)) for R in pair) for pair in mats]        # blending weight on the canvas
    prefer = [tuple(np.where(c > 1e-9, np.asarray(R @ gain[q]) / np.where(c > 1e-9, c, 1.0), 0.0) ** SELECT_POWER
                    for q, (R, c) in enumerate(zip(pair, cov))) for pair, cov in zip(mats, cover)]
    return {'n_canvas': n_canvas, 'pixel': float(pixel), 'width': float(width), 'left_edges': left_edges, 'overlaps': overlaps,
            'weights': weights, 'row_shift': row_shift, 'matrices': mats, 'mode': mode,
            'fill_fraction_temporal': fill_fraction_temporal, 'n_px': n_px, 'line_offset': float(line_offset),
            'gain': gain, 'dark': float(dark), 'ramp': ramp, 'cover': cover, 'prefer': prefer}


def shift_rows(image, offset):
    """Content moved down by ``offset`` rows (fractional, linear interpolation, edges held): out[r] = in[r - offset]."""
    if abs(offset) < 1e-3:
        return image
    h = image.shape[-2]
    src = np.clip(np.arange(h) - offset, 0, h - 1)
    r0 = np.clip(np.floor(src).astype(int), 0, h - 2)
    f = (src - r0)[:, None]
    return image[..., r0, :] * (1 - f) + image[..., r0 + 1, :] * f


def strip_on_canvas(strip, plan, i, weighted=True):
    """One raw strip (..., rows, n_px) placed on the canvas (..., rows, n_canvas): undistorted, line-corrected, at its
    sub-pixel position, row-shifted. With ``weighted=True`` this is the strip's CONTRIBUTION to the stitched plane
    (dark-subtracted light times its blending weight, and with ramp 'select' times its brightness preference; stitch()
    sums these and divides by what gain_collected returns). With ``weighted=False`` it is the strip on its own,
    ALWAYS brightness-corrected whatever the plan's ramp mode: that is a measurement aid, for comparing the two copies
    of an overlap, and never reaches the stitched movie."""
    strip = np.asarray(strip, np.float32) - plan['dark']
    out = np.empty(strip.shape[:-1] + (plan['n_canvas'],), np.float32)
    for parity in (0, 1):
        R = plan['matrices'][i][parity]
        if not weighted:
            R = strip_to_canvas_matrix(plan['n_px'], plan['fill_fraction_temporal'], plan['width'], plan['left_edges'][i],
                                       plan['n_canvas'], plan['pixel'],
                                       shift=(-0.5, 0.5)[parity] * plan['line_offset'])
        part = strip[..., parity::2, :]
        res = np.asarray((R @ part.reshape(-1, part.shape[-1]).T).T).reshape(part.shape[:-1] + (R.shape[0],))
        if not weighted:
            own = R @ plan['gain'][parity]
            res = res / np.where(own > 1e-6, own, 1.0) + plan['dark']
        elif plan['ramp'] == 'select':
            res = res * plan['prefer'][i][parity]
        out[..., parity::2, :] = res
    return shift_rows(out, plan['row_shift'][i])


def gain_collected(plan, rows):
    """What the summed contributions have to be divided by at every canvas pixel, (rows, n_canvas), moved by each
    strip's row shift exactly as its light is. Ramp 'correct': the gain collected (each strip's weight times its even-
    or odd-line brightness), so the result is brightness-corrected. Ramp 'select' and 'none': the weight collected, so
    the result is a plain weighted average of raw values. Cached in the plan."""
    key = ('gain_collected', rows)
    if key not in plan:
        total = np.zeros((rows, plan['n_canvas']), np.float32)
        for i, (r_even, r_odd) in enumerate(plan['matrices']):
            own = np.empty((rows, plan['n_canvas']), np.float32)
            if plan['ramp'] == 'correct':
                own[0::2], own[1::2] = r_even @ plan['gain'][0], r_odd @ plan['gain'][1]
            elif plan['ramp'] == 'select':
                own[0::2], own[1::2] = (plan['cover'][i][0] * plan['prefer'][i][0],
                                        plan['cover'][i][1] * plan['prefer'][i][1])
            else:
                own[0::2], own[1::2] = plan['cover'][i]
            total += shift_rows(own, plan['row_shift'][i])
        plan[key] = np.where(total > 1e-6, total, 1.0)
    return plan[key]


def stitch(strips, plan):
    """Raw strips (..., n_strips, rows, n_px) -> one plane (..., rows, n_canvas), following ``plan``."""
    out = None
    for i in range(strips.shape[-3]):
        part = strip_on_canvas(strips[..., i, :, :], plan, i)
        out = part if out is None else out + part
    return out / gain_collected(plan, strips.shape[-2]) + plan['dark']


def overlap_agreement(mean_strips, plan, sigma=10.0):
    """Do the two copies of the tissue in each overlap agree? For each seam: the correlation between the left and the
    right strip's (high-passed) mean image over the overlap columns, after both were put on the canvas, and how many
    canvas columns the overlap spans. NaN where the overlap is narrower than one column."""
    out = []
    for i in range(len(plan['overlaps'])):
        a, b = plan['left_edges'][i + 1], plan['left_edges'][i] + plan['width']
        cols = np.nonzero(((np.arange(plan['n_canvas']) + 0.5) * plan['pixel'] >= a)
                          & ((np.arange(plan['n_canvas']) + 0.5) * plan['pixel'] < b))[0]
        if len(cols) < 1:
            out.append({'seam': i, 'columns': 0, 'r': np.nan})
            continue
        pad = int(3 * sigma)
        sl = slice(max(cols[0] - pad, 0), min(cols[-1] + pad + 1, plan['n_canvas']))
        pair = []
        for j in (i, i + 1):
            img = strip_on_canvas(mean_strips[j], plan, j, weighted=False)[:, sl]
            img = img - ndimage.gaussian_filter(img, sigma)
            pair.append(img[:, cols[0] - sl.start:cols[-1] - sl.start + 1])
        m = 2 * (int(np.ceil(abs(plan['row_shift']).max())) + 1)
        by_parity = [float(np.corrcoef(pair[0][m + q:-m:2].ravel(), pair[1][m + q:-m:2].ravel())[0, 1]) for q in (0, 1)]
        out.append({'seam': i, 'columns': int(len(cols)), 'r': float(np.mean(by_parity)), 'r_even': by_parity[0],
                    'r_odd': by_parity[1]})
    return out


def check_overlap_agreement(agreement, mode, min_agreement=0.7, on_disagreement='warn'):
    """Blending two copies that do not show the same thing blurs or doubles structure. For the blending modes, warn
    (or raise, ``on_disagreement='error'``) when an overlap's two copies correlate below ``min_agreement``."""
    if mode not in ('crossfade', 'average') or on_disagreement == 'ignore':
        return []
    bad = [a for a in agreement if np.isfinite(a['r']) and a['r'] < min_agreement]
    if bad:
        msg = ('overlap mode %r blends the two strips, but they disagree at seam(s) %s (correlation %s, wanted >= %.2f). '
               'The seam position is probably wrong: check the seam fit, or use mode "split".'
               % (mode, ', '.join('%d|%d' % (a['seam'], a['seam'] + 1) for a in bad),
                  ', '.join('%.2f' % a['r'] for a in bad), min_agreement))
        if on_disagreement == 'error':
            raise ValueError(msg)
        warn(msg)
    return bad


def fit_seams(mean_strips, fill_fraction_temporal, width, yres, max_misfit=1.5, min_pairs=3, gain=None, dark=0.0):
    """Per-seam overlap and vertical offset from the mean of raw strips. Even and odd lines are fitted SEPARATELY, each
    after its start-of-sweep ramp is divided out (``gain`` from edge_gain), and the two answers are averaged: with the
    parities mixed, the bright/dim alternation of lines at every strip edge dominates and pulls the fit one row off.
    A seam is trusted only when both parities found at least ``min_pairs`` matching column pairs that agree to within
    ``max_misfit`` nominal pixels. Strips that merely touch have no shared columns and cannot be fitted this way: an
    untrusted seam gets the median of the trusted ones, or the nominal 0 if there is none.
    Returns (overlaps, row_offsets, fits, trusted); each fit carries both parities' results."""
    cell_px = max(4, int(round(12.0 / yres)))
    margin = int(min(mean_strips.shape[-1] // 4, max(20, 3 * cell_px)))
    gain = np.ones((2, mean_strips.shape[-1])) if gain is None else gain
    per_parity = []
    for parity in (0, 1):
        img = (np.asarray(mean_strips, np.float32)[:, parity::2] - dark) / gain[parity]
        hp = highpass(img, max(8.0, 2.5 * cell_px))
        per_parity.append([seam_fit(hp[i], hp[i + 1], fill_fraction_temporal, width, margin) for i in range(len(hp) - 1)])
    fits = [{'overlap': float(np.mean([e['overlap'], o['overlap']])),
             'dy_subpixel': float(np.mean([e['dy_subpixel'], o['dy_subpixel']]) * 2.0),      # fitted on every other row
             'misfit': float(np.nanmax([e['misfit'], o['misfit']])) if np.isfinite([e['misfit'], o['misfit']]).all() else np.nan,
             'n_pairs': int(min(e['n_pairs'], o['n_pairs'])), 'peak': float(min(e['peak'], o['peak'])),
             'even': e, 'odd': o, 'matrix': e['matrix'], 'spacing': float(np.mean([e['spacing'], o['spacing']]))}
            for e, o in zip(*per_parity)]
    trusted = np.array([np.isfinite(f['overlap']) and f['n_pairs'] >= min_pairs and f['misfit'] <= max_misfit for f in fits])
    overlaps = np.array([f['overlap'] for f in fits])
    rows = np.array([f['dy_subpixel'] for f in fits])
    if not trusted.all():
        fill_o = float(np.median(overlaps[trusted])) if trusted.any() else 0.0
        fill_r = float(np.median(rows[trusted])) if trusted.any() else 0.0
        warn('seam fit not trusted at seam(s) %s (fewer than %d matching column pairs, or they disagree by more than '
             '%.1f px): using overlap %.1f and row offset %.2f there'
             % (', '.join('%d|%d' % (i, i + 1) for i in np.nonzero(~trusted)[0]), min_pairs, max_misfit, fill_o, fill_r))
        overlaps = np.where(trusted, overlaps, fill_o)
        rows = np.where(trusted, rows, fill_r)
    return overlaps, rows, fits, trusted


def refine_seams(mean_strips, fill_fraction_temporal, width, pixel, overlaps, row_offsets, line_offset=0.0, gain=None,
                 dark=0.0, span=2.5, step=0.25, row_span=0.0, row_step=0.25):
    """Sub-pixel refinement of every seam: around the fitted overlap (+-``span`` output pixels in steps of ``step``),
    keep the value at which the two copies of the overlap agree best once both are undistorted and
    brightness-corrected. A seam with less than one column of overlap is left as it is.

    The ROW offset is NOT refined by default (``row_span=0``): the fitted value stands. Agreement cannot place rows.
    A fractional row shift interpolates between neighbouring lines, which smooths the noisier copy and RAISES the
    correlation, so the search lands on a half row whatever the truth is. Measured on Cadbury20221016d152643tUTC
    (tmp/seam_row_offset_landscape.py, 2026-11-18): agreement is flat within 0.02 from -0.75 to +0.75 rows, dips at
    whole rows, and peaks at -0.5 and +0.5 alike. ``row_span`` > 0 turns the search back on.
    Returns (overlaps, row_offsets, agreement at the chosen values)."""
    n_px = mean_strips.shape[-1]
    overlaps, row_offsets = np.array(overlaps, float), np.array(row_offsets, float)
    best_r = np.full(len(overlaps), np.nan)
    for i in range(len(overlaps)):
        if not np.isfinite(overlaps[i]) or overlaps[i] < pixel:
            continue
        pair = mean_strips[i:i + 2]
        best = (-np.inf, overlaps[i], row_offsets[i])
        for ov in overlaps[i] + np.arange(-span, span + 1e-9, step) * pixel:
            if ov < pixel:
                continue
            for dy in row_offsets[i] + np.arange(-row_span, row_span + 1e-9, row_step):
                plan = stitch_plan(n_px, fill_fraction_temporal, width, [ov], pixel, 'average', [dy], line_offset, gain, dark)
                r = overlap_agreement(pair, plan)[0]['r']
                if np.isfinite(r) and r > best[0]:
                    best = (r, ov, dy)
        best_r[i], overlaps[i], row_offsets[i] = best
    return overlaps, row_offsets, best_r


def plan_session(session, mean_strips=None, mode='crossfade', seam='fit', pixel=None, min_agreement=0.7,
                 on_disagreement='warn', n_frames=300, subpixel=True, ramp='select', dark=None):
    """The stitch plan for a session. ``seam``: 'fit' measures every seam from the data, 'nominal' takes the header's
    layout (no overlap), a number is an overlap in um used for every seam. ``subpixel=False`` rounds the measured
    positions to whole output pixels and rows; otherwise each fitted seam is refined to a fraction of a pixel by
    maximising the agreement of its two copies (refine_seams). ``ramp``: what the stitched movie does about the
    start-of-sweep brightness ramp, 'select' (default: never rescale, take the full-brightness copy where there is
    one), 'correct' (flat-field every frame), or 'none' (see stitch_plan). The ramp is MEASURED in every mode, because
    the seams cannot be found without it. That measurement only touches the mean image used to place the strips.
    Returns (plan, info)."""
    s = scan_settings(session)
    w = int(s['strip_px'][0])
    width, yres, f_t = float(s['strip_um'][0]), float(s['resolution_umpx'][1]), s['fill_fraction_temporal']
    pixel = yres if pixel is None else float(pixel)
    if mean_strips is None:
        mean_strips = read_strips(session, n_frames).mean(0)
    line_offset = float(np.median([bidirectional_offset(m) for m in mean_strips]))
    n = mean_strips.shape[0]
    ramp = ramp_mode(ramp)
    dark = dark_level(session) if dark is None else float(dark)
    gain = edge_gain(mean_strips, dark)
    fits, trusted = None, None
    if seam == 'fit':
        overlaps, rows, fits, trusted = fit_seams(mean_strips, f_t, width, yres, gain=gain, dark=dark)
    else:
        overlaps = np.full(n - 1, 0.0 if seam == 'nominal' else float(seam))
        rows = np.zeros(n - 1)
    if subpixel and seam == 'fit':
        overlaps, rows, _ = refine_seams(mean_strips, f_t, width, pixel, overlaps, rows, line_offset, gain, dark)
    if not subpixel:
        overlaps, rows = np.round(overlaps / pixel) * pixel, np.round(rows)
    plan = stitch_plan(w, f_t, width, overlaps, pixel, mode, rows, line_offset, gain, dark, ramp)
    agreement = overlap_agreement(mean_strips, plan)
    bad = check_overlap_agreement(agreement, mode, min_agreement, on_disagreement)
    return plan, {'settings': s, 'mean_strips': mean_strips, 'fits': fits, 'trusted': trusted, 'agreement': agreement,
                  'disagree': bad, 'line_offset': line_offset, 'gain': gain, 'dark': dark, 'ramp': ramp,
                  'ramp_left': ramp_left_in(plan, mean_strips.shape[-2])}


def ramp_left_in(plan, rows):
    """How much of the start-of-sweep dimming is still in the stitched plane, (rows, n_canvas): the brightness of each
    canvas pixel relative to full brightness, 1 = none left. 1 everywhere with ramp 'correct'."""
    if plan['ramp'] == 'correct':
        return np.ones((rows, plan['n_canvas']), np.float32)
    light = np.zeros((rows, plan['n_canvas']), np.float32)
    for i, (r_even, r_odd) in enumerate(plan['matrices']):
        own = np.empty((rows, plan['n_canvas']), np.float32)
        own[0::2], own[1::2] = r_even @ plan['gain'][0], r_odd @ plan['gain'][1]
        if plan['ramp'] == 'select':
            own[0::2] *= plan['prefer'][i][0]
            own[1::2] *= plan['prefer'][i][1]
        light += shift_rows(own, plan['row_shift'][i])
    return light / gain_collected(plan, rows)


# ------------------------------------------------------------------------------------------------------- report ----
def analyse(session, n_frames=300, seed=0):
    s = scan_settings(session)
    w, h = int(s['strip_px'][0]), int(s['strip_px'][1])
    width, yres = float(s['strip_um'][0]), float(s['resolution_umpx'][1])
    f_t = s['fill_fraction_temporal']
    strips = read_strips(session, n_frames, seed=seed)
    cell_px = max(4, int(round(12.0 / yres)))                         # a cell body is about 12 um
    lags = tuple(sorted({max(2, int(round(cell_px * q))) for q in (0.5, 0.75, 1.0, 1.5)}))
    dmax = int(min(h // 4, max(16, 4 * cell_px)))
    sigma = max(8.0, 2.5 * cell_px)
    kw = dict(sigma=sigma, lags=lags, dmax=dmax)
    raw_profile, raw_by_strip = pixel_size_profile(strips, yres, **kw)
    best, grid, res = fit_fill_fraction(raw_profile, w)
    n_iso = int(round(width / yres))                                  # evenly spaced, square pixels
    mean_raw = strips.mean(0)
    bidi = [bidirectional_offset(m) for m in mean_raw]
    und = undistort(strips, f_t, n_iso, line_offset=float(np.median(bidi)))
    und_profile, und_by_strip = pixel_size_profile(und, yres, **kw)
    mean_und = und.mean(0)
    seams_model, seams_even = [], []
    margin = int(min(w // 4, max(20, 3 * cell_px)))
    hp = highpass(mean_raw, sigma)
    for i in range(strips.shape[1] - 1):
        seams_model.append(seam_fit(hp[i], hp[i + 1], f_t, width, margin))
        seams_even.append(seam_fit(hp[i], hp[i + 1], 0.0, width, margin))
    return {'settings': s, 'strips_shape': strips.shape, 'lags': lags, 'dmax': dmax, 'sigma': sigma,
            'raw_profile': raw_profile, 'raw_by_strip': raw_by_strip, 'und_profile': und_profile,
            'und_by_strip': und_by_strip, 'best_fill_fraction': best, 'ff_grid': grid, 'ff_residual': res,
            'n_iso': n_iso, 'mean_raw': mean_raw, 'mean_und': mean_und, 'bidi': bidi,
            'seams_model': seams_model, 'seams_even': seams_even, 'margin': margin}


def report(session, out_png=None, n_frames=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    slug = sessionio.session_slug(session)
    r = analyse(session, n_frames)
    s = r['settings']
    w = int(s['strip_px'][0])
    width, yres, f_t = float(s['strip_um'][0]), float(s['resolution_umpx'][1]), s['fill_fraction_temporal']
    nominal = width / w
    model = pixel_sizes(w, f_t, width)
    print('%s | %d strips of %d x %d px, %.1f um wide, nominal pixel %.3f x %.3f um' % (slug, s['n_strips'], w, s['strip_px'][1], width, nominal, yres))
    print('scan settings: uniform sampling %s | fill fraction temporal %.4f, spatial %.5f (sin(pi/2 * temporal) = %.5f) | '
          'bidirectional %s | line phase %.3g s | sample rate %.0f Hz | resonant %.1f Hz'
          % (s['uniform_sampling'], f_t, s['fill_fraction_spatial'], np.sin(0.5 * np.pi * f_t), s['bidirectional'],
             s['line_phase'], s['sample_rate'], s['resonant_scanner_frequency']))
    print('model: pixel size %.2f um at the strip centre, %.2f um at the edges (ratio %.2f); largest position error if '
          'pixels are taken as evenly spaced: %.1f um'
          % (model.max(), model.min(), model.max() / model.min(),
             np.max(np.abs(pixel_positions(w, f_t, width) - pixel_positions(w, 0.0, width)))))
    print('raw frames used: %s (frames, strips, rows, columns), unregistered | ruler lags %s px, reference y-lags to %d px'
          % (r['strips_shape'], r['lags'], r['dmax']))
    print('\nMEASURED local x pixel size from texture (um), median over strips, by position within the strip (interior):')
    print('      u      raw frames   model     after undistorting (target %.2f)' % (width / r['n_iso']))
    c_raw, v_raw = binned(r['raw_profile'], 13)
    c_und, v_und = binned(r['und_profile'], 13)
    ratios = []
    for c, v, v2 in zip(c_raw, v_raw, v_und):
        mod = float(np.interp(c, np.arange(w), model))
        ratios.append(v / mod)
        print('   %+.2f      %.2f        %.2f      %.2f' % ((c - w / 2) / (w / 2), v, mod, v2))
    print('measured / model over the interior: median %.2f (range %.2f to %.2f) | after undistorting: median %.2f um, '
          'range %.2f to %.2f' % (np.median(ratios), np.min(ratios), np.max(ratios), np.nanmedian(v_und), np.nanmin(v_und), np.nanmax(v_und)))
    print('centre-to-edge ratio of the measured pixel size: raw %.2f, model %.2f, after undistorting %.2f (1 = evenly spaced)'
          % (flatness(r['raw_profile']), flatness(model), flatness(r['und_profile'])))
    print('per strip, raw:          %s' % ' '.join('%.2f' % flatness(p) for p in r['raw_by_strip']))
    print('per strip, undistorted:  %s' % ' '.join('%.2f' % flatness(p) for p in r['und_by_strip']))
    print('fill fraction whose sinusoid best matches the measured profile shape: %.2f (file header: %.3f)' % (r['best_fill_fraction'], f_t))
    print('\nodd/even line offset per strip (raw px): %s | median %+.2f' % (' '.join('%+.2f' % b for b in r['bidi']), np.median(r['bidi'])))
    print('\nseams: where each strip sits relative to its left neighbour (overlap in um; nominal is 0):')
    for i, (sm, se) in enumerate(zip(r['seams_model'], r['seams_even'])):
        print('   strips %d|%d: peak r %.2f, %d matched column pairs | sinusoid model: overlap %5.1f um (row blocks %s), scatter %.2f px | '
              'evenly spaced pixels: overlap %5.1f um, scatter %.2f px'
              % (i, i + 1, sm['peak'], sm['n_pairs'], sm['overlap'], ' '.join('%.1f' % o for o in sm['overlap_parts']), sm['misfit'],
                 se['overlap'], se['misfit']))
    ov = np.array([sm['overlap'] for sm in r['seams_model']])
    print('   sinusoid model: median overlap %.1f um = %.1f%% of the strip width (seam-to-seam range %.1f to %.1f um) | '
          'median scatter of the matches %.2f px vs %.2f px for evenly spaced pixels'
          % (np.nanmedian(ov), 100 * np.nanmedian(ov) / width, np.nanmin(ov), np.nanmax(ov),
             np.nanmedian([s_['misfit'] for s_ in r['seams_model']]), np.nanmedian([s_['misfit'] for s_ in r['seams_even']])))

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    k = np.arange(w)
    ax[0, 0].plot(k, model, 'k-', lw=2, label='model: uniform sampling in time, fill fraction %.3f' % f_t)
    ax[0, 0].axhline(nominal, color='0.5', ls='--', label='assumed so far: evenly spaced (%.2f um)' % nominal)
    for p in r['raw_by_strip']:
        ax[0, 0].plot(k, p, '.', ms=2, color='tab:blue', alpha=0.35)
    ax[0, 0].plot(k, ndimage.uniform_filter1d(np.nan_to_num(r['raw_profile'], nan=np.nanmedian(r['raw_profile'])), 7), '-',
                  color='tab:blue', lw=2, label='measured from texture, raw frames (all strips)')
    ax[0, 0].set_xlabel('pixel within the strip (resonant axis)'); ax[0, 0].set_ylabel('local pixel size along x (um)')
    ax[0, 0].set_title('pixel size across a strip: centre %.2f um, edges %.2f um in the model' % (model.max(), model.min()), fontsize=10)
    ax[0, 0].legend(fontsize=8); ax[0, 0].set_ylim(0, None)
    ku = np.arange(r['n_iso'])
    for p in r['und_by_strip']:
        ax[0, 1].plot(ku, p, '.', ms=2, color='tab:green', alpha=0.35)
    ax[0, 1].plot(ku, ndimage.uniform_filter1d(np.nan_to_num(r['und_profile'], nan=np.nanmedian(r['und_profile'])), 7), '-',
                  color='tab:green', lw=2, label='measured after undistorting with the header fill fraction')
    ax[0, 1].axhline(width / r['n_iso'], color='k', ls='--', label='target: %.2f um everywhere' % (width / r['n_iso']))
    ax[0, 1].set_xlabel('pixel within the undistorted strip'); ax[0, 1].set_ylabel('local pixel size along x (um)')
    ax[0, 1].set_title('after undistorting: centre-to-edge ratio %.2f (was %.2f)' % (flatness(r['und_profile']), flatness(r['raw_profile'])), fontsize=10)
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_ylim(0, ax[0, 0].get_ylim()[1])
    ax[0, 2].plot(r['ff_grid'], r['ff_residual'], 'k-')
    ax[0, 2].axvline(f_t, color='tab:red', ls='--', label='file header: %.3f' % f_t)
    ax[0, 2].axvline(r['best_fill_fraction'], color='tab:blue', ls=':', label='best match to the data: %.2f' % r['best_fill_fraction'])
    ax[0, 2].set_xlabel('assumed temporal fill fraction (0 = evenly spaced pixels)')
    ax[0, 2].set_ylabel('misfit of the sinusoid profile shape to the measured one')
    ax[0, 2].set_title('which sinusoid do the data ask for?', fontsize=10); ax[0, 2].legend(fontsize=8)
    mid = r['mean_raw'].shape[0] // 2
    hh = r['mean_raw'].shape[1]
    rows = slice(hh // 2 - min(hh // 2, 2 * w), hh // 2 + min(hh // 2, 2 * w))
    for a_, img, title in ((ax[1, 0], r['mean_raw'][mid], 'raw strip %d (mean of raw frames): pixels evenly spaced in TIME' % mid),
                           (ax[1, 1], r['mean_und'][mid], 'same strip undistorted: pixels evenly spaced in SPACE')):
        v = np.percentile(img[rows], [1, 99.7])
        a_.imshow(img[rows], cmap='gray', vmin=v[0], vmax=v[1], aspect=(width / img.shape[1]) / yres, interpolation='nearest')
        a_.set_title(title, fontsize=10); a_.set_xlabel('x (pixels)'); a_.set_ylabel('y (pixels)')
    sm = max(r['seams_model'], key=lambda d: d['peak'])
    i = r['seams_model'].index(sm)
    im = ax[1, 2].imshow(sm['matrix'], cmap='magma', origin='lower', vmin=0, vmax=max(0.3, sm['matrix'].max()),
                         extent=[-0.5, r['margin'] - 0.5, w - r['margin'] - 0.5, w - 0.5], aspect='auto')
    b = np.arange(r['margin'])
    for f_, sp, col, lab in ((f_t, sm['spacing'], 'w', 'same tissue, sinusoid model'),
                             (0.0, r['seams_even'][i]['spacing'], 'lime', 'same tissue, evenly spaced pixels')):
        pp = pixel_positions(w, f_, width)
        ax[1, 2].plot(b, np.interp(pp[b] + sp, pp, np.arange(w), left=np.nan, right=np.nan), '-', color=col, lw=1.4, label=lab)
    ax[1, 2].set_ylim(w - r['margin'] - 0.5, w - 0.5)
    ax[1, 2].legend(fontsize=7, loc='lower right')
    ax[1, 2].set_xlabel('column of strip %d (its left edge)' % (i + 1)); ax[1, 2].set_ylabel('column of strip %d (its right edge)' % i)
    ax[1, 2].set_title('seam %d|%d: which columns image the same tissue (overlap %.1f um)' % (i, i + 1, sm['overlap']), fontsize=10)
    fig.colorbar(im, ax=ax[1, 2], fraction=0.046, label='correlation along y')
    fig.suptitle('%s: geometry of the mROI strips along the resonant axis (%d raw frames)' % (slug, r['strips_shape'][0]), fontsize=12)
    fig.tight_layout()
    out_png = out_png or paths.output('mroi_geometry_%s_%s.png' % (slug, STAMP))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)
    print('\nsaved %s' % out_png)
    return r


def stitch_movie(strips_by_frame, plan, chunk=200):
    """Stitch a whole movie: ``strips_by_frame`` is a sequence of n_strips arrays (frames, rows, n_px), e.g. views into
    the raw recording. Returns float32 (frames, rows, n_canvas). Works through the frames in chunks."""
    n_frames, rows = strips_by_frame[0].shape[:2]
    out = np.empty((n_frames, rows, plan['n_canvas']), np.float32)
    for f0 in range(0, n_frames, chunk):
        block = np.stack([np.asarray(s[f0:f0 + chunk], np.float32) for s in strips_by_frame], axis=1)
        out[f0:f0 + chunk] = stitch(block, plan)
    return out


def cut_and_butt(mean_strips, overlap_px):
    """The stitching used so far (reshape_mrois_to_planes.py --overlap N): drop N RAW pixels per seam, half from each
    strip, and butt the rest together. No undistortion, no row offset, the same N everywhere."""
    n, w = len(mean_strips), mean_strips[0].shape[1]
    lo, hi = int(np.ceil(overlap_px / 2)), int(np.floor(overlap_px / 2))
    parts = []
    for i, s in enumerate(mean_strips):
        if i == 0:
            parts.append(s[:, :w - hi])
        elif i < n - 1:
            parts.append(s[:, lo:lo + w - overlap_px])
        else:
            parts.append(s[:, lo:])
    return np.concatenate(parts, axis=1)


def stitch_report(session, mode='crossfade', seam='fit', n_frames=300, min_agreement=0.7, on_disagreement='warn',
                  subpixel=True, old_overlap_px=None, out_png=None, ramp='select'):
    """Plan and apply the stitching on the mean of sampled raw frames, print what was found, and save a figure that
    compares it with the cut-and-butt stitching used so far."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    slug = sessionio.session_slug(session)
    plan, info = plan_session(session, None, mode, seam, None, min_agreement, on_disagreement, n_frames, subpixel, ramp)
    s, mean_strips = info['settings'], info['mean_strips']
    w, width, yres = int(s['strip_px'][0]), float(s['strip_um'][0]), float(s['resolution_umpx'][1])
    new = stitch(mean_strips, plan)
    if old_overlap_px is None:                                   # what the existing extraction was stitched with
        try:
            ops = np.load(os.path.join(sessionio.find_suite2p_dir(session), 'plane0', 'ops.npy'), allow_pickle=True).item()
            old_overlap_px = int(round((len(mean_strips) * w - int(ops['Lx'])) / (len(mean_strips) - 1)))
        except Exception:                                        # noqa: BLE001
            old_overlap_px = 0
    old = cut_and_butt(mean_strips, old_overlap_px)
    print('%s | overlap mode %s | seams: %s | sub-pixel %s | odd/even line offset %+.2f raw px'
          % (slug, mode, seam, subpixel, info['line_offset']))
    print('stitched plane: %d x %d px of %.2f um = %.1f um wide (cut-and-butt with %d px per seam: %d px, nominally %.1f um)'
          % (new.shape[0], new.shape[1], plan['pixel'], new.shape[1] * plan['pixel'], old_overlap_px, old.shape[1],
             old.shape[1] * width / w))
    for i, a in enumerate(info['agreement']):
        fit = info['fits'][i] if info['fits'] else None
        print('   seam %d|%d: overlap %6.2f um, row offset %+.2f%s | overlap spans %d columns, the two copies correlate %s'
              % (i, i + 1, plan['overlaps'][i], plan['row_shift'][i + 1] - plan['row_shift'][i],
                 '' if fit is None else ' (%d matched column pairs, scatter %.2f px, %s)' % (
                     fit['n_pairs'], fit['misfit'], 'trusted' if info['trusted'][i] else 'NOT trusted, filled in'),
                 a['columns'], 'n/a' if not np.isfinite(a['r']) else '%.2f (even lines %.2f, odd %.2f)' % (a['r'], a['r_even'], a['r_odd'])))
    if info['gain'] is not None:
        g = info['gain']
        print('   start-of-sweep ramp (dark level %.1f): even lines dim over the first %d px of a line (first pixel %.2f), '
              'odd lines over the last %d px (last pixel %.2f)'
              % (info['dark'], int(np.sum(g[0, :w // 2] < 0.9)), g[0, 0], int(np.sum(g[1, w // 2:] < 0.9)), g[1, -1]))
        left = info['ramp_left']
        print('   ramp mode "%s": %s. Dimming left in the stitched plane: %.1f%% of its pixels are below 0.9 of full '
              'brightness, the dimmest at %.2f'
              % (info['ramp'], {'select': 'no value rescaled, full-brightness copy taken where one exists',
                                'correct': 'every frame flat-fielded', 'none': 'gain ignored'}[info['ramp']],
                 100 * float(np.mean(left < 0.9)), float(left.min())))
    if info['disagree']:
        print('   WARNING: %d seam(s) disagree below %.2f' % (len(info['disagree']), min_agreement))
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={'width_ratios': [2.2, 1]})
    seams_x_new = plan['left_edges'][1:] / plan['pixel']
    k = int(np.nanargmax([a['r'] if np.isfinite(a['r']) else -1 for a in info['agreement']])) if info['agreement'] else 0
    rows = slice(new.shape[0] // 2 - min(new.shape[0] // 2, 110), new.shape[0] // 2 + min(new.shape[0] // 2, 110))
    for r_, img, px, title, seams_x in (
            (0, old, width / w, 'so far: cut %d raw px per seam and butt (pixels evenly spaced in time)' % old_overlap_px,
             np.cumsum([p_ for p_ in ([w - old_overlap_px // 2] + [w - old_overlap_px] * (len(mean_strips) - 2))])),
            (1, new, plan['pixel'], 'undistorted, seams placed to a fraction of a pixel, overlap mode "%s"' % mode, seams_x_new)):
        v = np.percentile(img, [1, 99.7])
        ax[r_, 0].imshow(img, cmap='gray', vmin=v[0], vmax=v[1], aspect=yres / px if False else 'auto', interpolation='nearest',
                         extent=[0, img.shape[1] * px, img.shape[0] * yres, 0])
        for sx in seams_x:
            ax[r_, 0].axvline(sx * px, color='c', lw=0.5, alpha=0.6)
        ax[r_, 0].set_title(title, fontsize=10); ax[r_, 0].set_xlabel('x (um, at each image\'s own pixel size)'); ax[r_, 0].set_ylabel('y (um)')
        c = seams_x[k] * px
        half = 0.45 * width
        cols = slice(max(int((c - half) / px), 0), int((c + half) / px))
        ax[r_, 1].imshow(img[rows, cols], cmap='gray', vmin=v[0], vmax=v[1], aspect=px / yres, interpolation='nearest')
        ax[r_, 1].set_title('zoom on seam %d|%d' % (k, k + 1), fontsize=10); ax[r_, 1].set_xticks([]); ax[r_, 1].set_yticks([])
    fig.suptitle('%s: stitching the mROI strips (mean of %d raw frames), ramp mode "%s"' % (slug, n_frames, info['ramp']),
                 fontsize=12)
    fig.tight_layout()
    out_png = out_png or paths.output('mroi_stitch_%s_mode-%s_seam-%s_ramp-%s_%s.png' % (slug, mode, seam, info['ramp'], STAMP))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)
    print('saved %s' % out_png)
    return plan, info, new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', required=True, help='session slug (or a unique prefix of one), or a session folder')
    ap.add_argument('--frames', type=int, default=300, help='raw frames to sample')
    ap.add_argument('--stitch', action='store_true', help='plan and show the stitching instead of the geometry report')
    ap.add_argument('--overlap-mode', default='crossfade', choices=OVERLAP_MODES,
                    help='how the two copies of the tissue in an overlap are combined (see overlap_weights)')
    ap.add_argument('--seam', default='fit', help="'fit' (measure every seam), 'nominal' (header layout), or an overlap in um")
    ap.add_argument('--no-subpixel', action='store_true', help='round seam positions and row offsets to whole pixels')
    ap.add_argument('--min-agreement', type=float, default=0.7,
                    help='lowest correlation between the two copies in an overlap before a blending mode complains')
    ap.add_argument('--on-disagreement', default='warn', choices=['warn', 'error', 'ignore'])
    ap.add_argument('--ramp', default='select', choices=RAMP_MODES,
                    help="start-of-sweep brightness ramp: 'select' never rescales and takes the full-brightness copy "
                         "where strips overlap, 'correct' flat-fields every frame, 'none' ignores it (see stitch_plan)")
    ap.add_argument('--ramp-checks', action='store_true',
                    help='measure whether the ramp is a fixed, purely multiplicative gain (what --ramp correct assumes)')
    a = ap.parse_args()
    session = a.session if os.path.isdir(a.session) else paths.session(a.session)
    if a.ramp_checks:
        print_ramp_checks(ramp_checks(session))
    elif a.stitch:
        stitch_report(session, a.overlap_mode, a.seam, a.frames, a.min_agreement, a.on_disagreement, not a.no_subpixel,
                      ramp=a.ramp)
    else:
        report(session, n_frames=a.frames)


if __name__ == '__main__':
    main()
