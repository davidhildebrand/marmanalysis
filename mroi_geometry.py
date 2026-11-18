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

Run:  .venv/bin/python marmanalysis/mroi_geometry.py --session Curly20231103d162355tUTC [--frames 300]
Writes output/mroi_geometry_<slug>_<UTC>.png and prints the measurements.
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metadata
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
def scan_settings(session):
    """The scan parameters that fix the strip geometry. Fill fractions, scanner frequency and strip layout come from
    the session metadata (metadata.extract_useful_metadata); fields that older metadata files do not hold are read
    from the raw file header."""
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
        h = metadata.get_metadata(raw_file(session))['SI']['hScan2D']
        out.update(uniform_sampling=bool(h['uniformSampling']), bidirectional=bool(h['bidirectional']),
                   line_phase=float(h['linePhase']), sample_rate=float(h['sampleRate']),
                   pixel_bin_factor=h['pixelBinFactor'])
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
def column_correlation_vs_distance(image, positions, max_dist, step):
    """How the correlation between two columns' y-profiles falls off with the distance between the columns, measured
    INSIDE one strip (interior half): a monotone curve g on a grid of ``step``, with g(0) = 1."""
    w = image.shape[1]
    cols = np.arange(w // 4, 3 * w // 4)
    z = image[:, cols]
    z = (z - z.mean(0)) / z.std(0)
    c = z.T @ z / z.shape[0]
    d = np.abs(positions[cols][:, None] - positions[cols][None, :])
    grid = np.arange(0.0, max_dist + step, step)
    g = np.ones(len(grid))
    for i in range(1, len(grid)):
        m = (d > grid[i] - step) & (d <= grid[i] + step)
        g[i] = c[m].mean() if m.any() else np.nan
    ok = np.isfinite(g)
    g = np.interp(grid, grid[ok], g[ok])
    return grid, np.clip(np.minimum.accumulate(g), 0.0, None)     # monotone, and never negative: only nearness counts


def seam_fit(left, right, fill_fraction_temporal, width, margin=28, max_dy=4, max_dist=None, n_parts=4):
    """Where does the right strip sit relative to the left one? ``left`` and ``right`` are high-passed mean images of
    two adjacent RAW strips.

    The correlation between every left-edge column and every right-edge column is compared with what the within-strip
    fall-off of correlation with distance predicts, given each column's position under the pixel model and a trial
    spacing between the two strip centres. The spacing that fits best (least squares, free amplitude) is the answer.
    All column pairs contribute, including neighbours that do not overlap, so this also works when the overlap is only
    a pixel or two. Returns overlap (= width - spacing, same units as ``width``), its range over ``n_parts`` row
    blocks, the vertical offset, and the relative misfit (lower = the pixel model describes the seam better)."""
    n_px = left.shape[1]
    pos = pixel_positions(n_px, fill_fraction_temporal, width)
    step = width / n_px / 4.0
    max_dist = 6.0 * width / n_px * 3 if max_dist is None else max_dist
    grid, g = column_correlation_vs_distance(left, pos, max_dist, step)
    xl, xr = pos[n_px - margin:], pos[:margin]
    trial = np.arange(width - 0.35 * width, width + 0.1 * width, step)              # centre spacings to try

    def matrix(rows, dy):
        a = left[rows, n_px - margin:]
        b = right[rows, :margin]
        a = a[max_dy + dy:a.shape[0] - max_dy + dy]
        b = b[max_dy:b.shape[0] - max_dy]
        za, zb = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
        return za.T @ zb / za.shape[0]

    def fit(m):
        best = (np.inf, np.nan, np.nan)
        for s in trial:
            pred = np.interp(np.abs(xl[:, None] - (xr[None, :] + s)), grid, g, right=0.0)
            alpha = np.sum(m * pred) / max(np.sum(pred * pred), 1e-12)
            if alpha <= 0:
                continue                                         # shared tissue correlates positively, never negatively
            err = np.sum((m - alpha * pred) ** 2) / np.sum(m * m)
            if err < best[0]:
                best = (err, s, alpha)
        return best

    everything = slice(None)
    dys = np.arange(-max_dy, max_dy + 1)
    peaks = [matrix(everything, int(dy)).max() for dy in dys]
    dy = int(dys[int(np.argmax(peaks))])
    m = matrix(everything, dy)
    err, spacing, alpha = fit(m)
    h = left.shape[0]
    parts = [fit(matrix(slice(i * h // n_parts, (i + 1) * h // n_parts), dy))[1] for i in range(n_parts)]
    return {'overlap': float(width - spacing), 'overlap_parts': [float(width - s) for s in parts], 'dy': dy,
            'misfit': float(err), 'amplitude': float(alpha), 'peak': float(m.max()), 'matrix': m, 'spacing': float(spacing)}


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
        print('   strips %d|%d: peak r %.2f, vertical offset %+d px | sinusoid model: overlap %5.1f um (row blocks %s), misfit %.3f | '
              'evenly spaced pixels: overlap %5.1f um, misfit %.3f'
              % (i, i + 1, sm['peak'], sm['dy'], sm['overlap'], ' '.join('%.1f' % o for o in sm['overlap_parts']), sm['misfit'],
                 se['overlap'], se['misfit']))
    ov = np.array([sm['overlap'] for sm in r['seams_model']])
    print('   sinusoid model: median overlap %.1f um = %.1f%% of the strip width (seam-to-seam range %.1f to %.1f um) | '
          'mean misfit %.3f vs %.3f for evenly spaced pixels'
          % (np.median(ov), 100 * np.median(ov) / width, ov.min(), ov.max(), np.mean([s_['misfit'] for s_ in r['seams_model']]),
             np.mean([s_['misfit'] for s_ in r['seams_even']])))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', required=True, help='session slug (or a unique prefix of one), or a session folder')
    ap.add_argument('--frames', type=int, default=300, help='raw frames to sample')
    a = ap.parse_args()
    report(a.session if os.path.isdir(a.session) else paths.session(a.session), n_frames=a.frames)


if __name__ == '__main__':
    main()
