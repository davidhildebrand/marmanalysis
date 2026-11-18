"""Tests for mroi_geometry on synthetic scenes with a known geometry: the uniform-time sinusoid pixel model, the
undistortion, the odd/even line offset, the model-free texture ruler, and the seam fit. No real data is needed."""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import mroi_geometry as mg  # noqa: E402

N, F = 146, 0.72


def texture(rng, rows, width, corr=3.0, fine=4):
    """Isotropic random texture over true positions [0, width): rows x (width * fine) samples, unit row spacing."""
    t = ndimage.gaussian_filter(rng.standard_normal((rows, int(width * fine))), (corr, corr * fine))
    return t / t.std()


def sample(tex, x, fine=4):
    """Columns of ``tex`` at true positions ``x`` (linear interpolation)."""
    grid = (np.arange(tex.shape[1]) + 0.5) / fine
    return np.stack([np.interp(x, grid, row) for row in tex])


def test_pixel_model():
    sizes = mg.pixel_sizes(N, F, 146.0)
    assert np.isclose(sizes.sum(), 146.0)                        # the pixels tile the strip exactly
    assert np.argmax(sizes) in (N // 2 - 1, N // 2) and sizes[0] == sizes.min()
    a = 0.5 * np.pi * F
    assert np.isclose(sizes[N // 2] / (146.0 / N), a / np.sin(a), rtol=1e-3)           # centre pixel vs nominal
    pos = mg.pixel_positions(N, F, 146.0)
    assert np.all(np.diff(pos) > 0) and np.isclose(pos[0], -pos[-1])
    assert np.allclose(np.diff(mg.pixel_positions(N, 0.0, 146.0)), 1.0)                # no sinusoid: evenly spaced
    worst = np.max(np.abs(mg.pixel_positions(2001, 0.718) - mg.pixel_positions(2001, 0.0)))
    assert abs(worst - 0.0935) < 0.001                           # 9.4% of the half width at this fill fraction


def test_undistort_recovers_a_scene_sampled_on_the_sinusoid():
    def scene(x):
        return np.sin(7 * x) + 0.5 * np.cos(19 * x + 1)
    raw = np.repeat(scene(mg.pixel_positions(N, F))[None], 6, axis=0)
    even = scene(mg.pixel_positions(N, 0.0))
    assert np.max(np.abs(raw[0] - even)) > 0.5                   # the raw line is badly off
    out = mg.undistort(raw, F)
    assert out.shape == raw.shape and np.max(np.abs(out[:, 3:-3] - even[3:-3])) < 0.06
    R = mg.resampling_matrix(N, F, 200)
    assert R.shape == (200, N) and np.allclose(np.asarray(R.sum(1)).ravel(), 1.0)       # intensity preserved
    assert np.allclose(mg.undistort(np.ones((4, N)), F, 180), 1.0)


def test_line_offset_is_measured_and_removed():
    rng = np.random.default_rng(0)
    tex = texture(rng, 300, N + 8)
    x = np.arange(N) + 4.5
    img = sample(tex, x)
    img[1::2] = sample(tex[1::2], x - 0.4)                       # content of odd lines sits 0.4 px to the right
    off = mg.bidirectional_offset(img)
    assert abs(off - 0.4) < 0.08
    fixed = mg.undistort(img, 0.0, line_offset=off)             # fill fraction 0: only the line correction acts
    assert abs(mg.bidirectional_offset(fixed)) < 0.08


def test_texture_ruler_finds_the_sinusoid_and_undistorting_flattens_it():
    rng = np.random.default_rng(1)
    pos = mg.pixel_positions(N, F, float(N)) + N / 2.0
    strips = np.stack([np.stack([sample(texture(rng, 400, N), pos) for _ in range(3)]) for _ in range(6)])
    strips = strips.astype(np.float32) + 0.5 * rng.standard_normal(strips.shape).astype(np.float32)   # pixel noise
    kw = dict(sigma=12.0, lags=(4, 6, 8), dmax=20)
    profile, per_strip = mg.pixel_size_profile(strips, 1.0, **kw)
    assert per_strip.shape == (3, N)
    centres, values = mg.binned(profile)
    model = np.interp(centres, np.arange(N), mg.pixel_sizes(N, F, float(N)))
    assert np.nanmedian(np.abs(values / model - 1)) < 0.1        # the ruler reads the model's pixel size, in um
    best, _, _ = mg.fit_fill_fraction(profile, N)
    assert abs(best - F) < 0.08
    assert mg.flatness(profile) > 1.35
    flat, _ = mg.pixel_size_profile(mg.undistort(strips, F), 1.0, **kw)
    assert abs(mg.flatness(flat) - 1) < 0.1 and abs(np.nanmedian(mg.binned(flat)[1]) - 1) < 0.1


def test_seam_fit_recovers_a_known_overlap():
    rng = np.random.default_rng(2)
    width, overlap = float(N), 8.0
    tex = texture(rng, 600, 2 * N + 20)
    pos = mg.pixel_positions(N, F, width)
    left = sample(tex, pos + 10 + width / 2)
    right = sample(tex, pos + 10 + width / 2 + (width - overlap))
    right = np.roll(right, 2, axis=0)                           # and the right strip sits 2 rows lower
    hp = mg.highpass(np.stack([left, right]), 12.0)
    good = mg.seam_fit(hp[0], hp[1], F, width, margin=30)
    even = mg.seam_fit(hp[0], hp[1], 0.0, width, margin=30)
    assert abs(good['overlap'] - overlap) < 1.0 and abs(good['dy']) == 2
    assert good['misfit'] < even['misfit']                       # the sinusoid describes the seam better
    assert abs(even['overlap'] - overlap) > 2.0                  # evenly spaced pixels get the overlap wrong
