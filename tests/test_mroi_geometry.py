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


# ------------------------------------------------------------------------------------------------ stitching ----
import pytest  # noqa: E402

OVERLAPS, ROW_OFFSETS = (8.0, 5.0), (1.5, -0.5)


def scene_and_strips(rng, rows=420, overlaps=OVERLAPS, row_offsets=ROW_OFFSETS):
    """A scene, and three raw strips of it: sinusoid-sampled, overlapping by ``overlaps``, each neighbour's content
    offset by ``row_offsets`` rows (left row r + offset shows what the right strip shows at row r)."""
    width = float(N)
    left = np.concatenate([[0.0], np.cumsum(width - np.array(overlaps))])
    down = np.concatenate([[0.0], np.cumsum(row_offsets)])
    tex = texture(rng, rows + 12, left[-1] + width + 20)
    pos = mg.pixel_positions(N, F, width)

    def view(x, row_shift):
        cols = sample(tex, x + 10.0)                                       # (rows + 12, len(x))
        r = np.arange(rows) + 6.0 + row_shift
        r0 = np.floor(r).astype(int)
        f = (r - r0)[:, None]
        return cols[r0] * (1 - f) + cols[r0 + 1] * f

    strips = np.stack([view(pos + left[i] + width / 2, down[i]) for i in range(3)])
    return strips, view, left, width


def test_overlap_weights_for_every_mode():
    width, left = 100.0, np.array([0.0, 90.0, 185.0])                     # overlaps of 10 and 5
    x = np.arange(0.5, 285, 1.0)
    w = {m: mg.overlap_weights(x, left, width, m) for m in mg.OVERLAP_MODES}
    for m, wm in w.items():
        assert np.allclose(wm.sum(0), 1.0), m                              # every column is fully accounted for
    outside = (x < 90) | ((x >= 100) & (x < 185)) | (x >= 200)
    for m in mg.OVERLAP_MODES:
        assert np.allclose(w[m][:, outside], w['split'][:, outside])       # modes differ only inside an overlap
    zone = (x >= 90) & (x < 100)
    assert np.allclose(w['crossfade'][0, zone], (100 - x[zone]) / 10)      # a linear fade
    assert np.allclose(w['average'][:2, zone], 0.5)
    assert np.allclose(w['left'][0, zone], 1) and np.allclose(w['right'][1, zone], 1)
    assert np.allclose(w['split'][0, zone], x[zone] < 95)
    gap = mg.overlap_weights(x, np.array([0.0, 104.0]), width, 'crossfade')   # a 4-wide gap: each side reaches its middle
    assert np.allclose(gap.sum(0)[x < 204], 1.0) and gap[0, x < 102].all() and gap[1, (x >= 102) & (x < 204)].all()
    with pytest.raises(ValueError):
        mg.overlap_weights(x, left, width, 'blend')


@pytest.mark.parametrize('mode', mg.OVERLAP_MODES)
def test_stitch_rebuilds_the_scene_in_every_mode(mode):
    strips, view, left, width = scene_and_strips(np.random.default_rng(3))
    plan = mg.stitch_plan(N, F, width, OVERLAPS, 1.0, mode, ROW_OFFSETS)
    out = mg.stitch(strips, plan)
    truth = view((np.arange(plan['n_canvas']) + 0.5) * 1.0, 0.0)
    assert out.shape == truth.shape == (420, int(round(left[-1] + width)))
    inner = (slice(8, -8), slice(4, -4))
    assert np.corrcoef(out[inner].ravel(), truth[inner].ravel())[0, 1] > 0.985
    # leaving the strips where the header puts them (edge to edge, no row offsets) doubles the tissue at the seams
    naive = mg.stitch(strips, mg.stitch_plan(N, F, width, (0.0, 0.0), 1.0, mode))
    seam = slice(int(left[1]) - 6, int(left[1]) + 14)
    good = np.corrcoef(out[8:-8, seam].ravel(), truth[8:-8, seam].ravel())[0, 1]
    assert good > 0.97
    cols = slice(int(width) - 6, int(width) + 14)
    wide = view((np.arange(naive.shape[1]) + 0.5) * 1.0, 0.0) if naive.shape[1] <= truth.shape[1] + 20 else None
    assert naive.shape[1] > out.shape[1] and (wide is None or np.corrcoef(naive[8:-8, cols].ravel(), wide[8:-8, cols].ravel())[0, 1] < good)


def test_seams_are_found_to_a_fraction_of_a_pixel_and_disagreement_is_caught():
    strips, view, left, width = scene_and_strips(np.random.default_rng(4))
    overlaps, rows, fits, trusted = mg.fit_seams(strips, F, width, 1.0)
    assert trusted.all()
    assert np.max(np.abs(overlaps - OVERLAPS)) < 1.0 and np.max(np.abs(rows - ROW_OFFSETS)) < 0.35
    plan = mg.stitch_plan(N, F, width, overlaps, 1.0, 'crossfade', rows)
    agree = mg.overlap_agreement(strips, plan)
    assert all(a['r'] > 0.85 for a in agree)
    assert all(abs(a['columns'] - o) <= 1 for a, o in zip(agree, overlaps))     # the overlap in whole canvas columns
    assert mg.check_overlap_agreement(agree, 'crossfade') == []
    wrong = mg.stitch_plan(N, F, width, np.array(OVERLAPS) + 9.0, 1.0, 'crossfade', rows)   # seams 9 um off
    bad = mg.overlap_agreement(strips, wrong)
    assert all(a['r'] < 0.6 for a in bad)
    with pytest.warns(UserWarning, match='disagree'):
        assert len(mg.check_overlap_agreement(bad, 'crossfade')) == 2
    with pytest.raises(ValueError, match='disagree'):
        mg.check_overlap_agreement(bad, 'average', on_disagreement='error')
    assert mg.check_overlap_agreement(bad, 'split', on_disagreement='error') == []        # nothing is blended there


def test_shift_rows():
    img = np.arange(40, dtype=float).reshape(8, 5)
    assert np.allclose(mg.shift_rows(img, 0.0), img)
    assert np.allclose(mg.shift_rows(img, 2.0)[2:], img[:-2])              # content moved down by two rows
    assert np.allclose(mg.shift_rows(img, 0.5)[1:], 0.5 * (img[:-1] + img[1:]))


def test_start_of_sweep_ramp_is_measured_and_removed():
    """Every line starts dim: even lines at the left strip edge, odd lines at the right one. The ramp is recovered from
    the ratio of the two parities, the seams are still found, and the stitched plane has no bright/dim line pattern."""
    strips, view, left, width = scene_and_strips(np.random.default_rng(5), row_offsets=(0.0, 0.0))
    dark, level = 30.0, 6.0
    ramp = np.ones((2, N))
    ramp[0, :12] = np.linspace(0.2, 0.95, 12)                              # even lines: first pixels of the sweep
    ramp[1, -12:] = np.linspace(0.95, 0.2, 12)                             # odd lines sweep the other way
    raw = np.empty_like(strips)
    for parity in (0, 1):
        raw[:, parity::2] = dark + (level + strips[:, parity::2]) * ramp[parity]
    gain = mg.edge_gain(raw, dark)
    assert gain.shape == (2, N) and np.max(np.abs(gain - ramp)) < 0.05
    overlaps, rows, fits, trusted = mg.fit_seams(raw, F, width, 1.0, gain=gain, dark=dark)
    assert trusted.all() and np.max(np.abs(overlaps - OVERLAPS)) < 1.0 and np.max(np.abs(rows)) < 0.35
    truth = None
    scores = {}
    for name, g in (('corrected', gain), ('uncorrected', None)):
        plan = mg.stitch_plan(N, F, width, OVERLAPS, 1.0, 'average', gain=g, dark=dark if g is not None else 0.0,
                              ramp='correct')
        out = mg.stitch(raw, plan)
        if truth is None:
            truth = dark + level + view((np.arange(plan['n_canvas']) + 0.5) * 1.0, 0.0)
        seam = slice(int(left[1]) - 8, int(left[1]) + 16)
        scores[name] = np.corrcoef(out[8:-8, seam].ravel(), truth[8:-8, seam].ravel())[0, 1]
        if g is not None:
            assert np.max(np.abs(out[8:-8, 20:-20] - truth[8:-8, 20:-20])) < 1.0     # right brightness, no line pattern
    assert scores['corrected'] > 0.97 and scores['corrected'] > scores['uncorrected'] + 0.02


def test_ramp_modes_select_takes_the_bright_copy_and_never_rescales():
    """'select' reaches full brightness wherever a strip overlaps its neighbour without dividing anything by a gain:
    its output is a fixed-weight average of raw values, so it is linear in the data and blind to the dark level.
    'correct' flat-fields everything and therefore depends on the dark level. 'none' leaves the line pattern in."""
    import pytest
    strips, view, left, width = scene_and_strips(np.random.default_rng(6), row_offsets=(0.0, 0.0))
    dark, level = 30.0, 6.0
    ramp = np.ones((2, N))
    ramp[0, :12] = np.linspace(0.2, 0.95, 12)
    ramp[1, -12:] = np.linspace(0.95, 0.2, 12)
    raw = np.empty_like(strips)
    for parity in (0, 1):
        raw[:, parity::2] = dark + (level + strips[:, parity::2]) * ramp[parity]
    gain = mg.edge_gain(raw, dark)
    plans = {m: mg.stitch_plan(N, F, width, OVERLAPS, 1.0, 'average', gain=gain, dark=dark, ramp=m) for m in mg.RAMP_MODES}
    out = {m: mg.stitch(raw, plans[m]) for m in plans}
    truth = dark + level + view((np.arange(plans['select']['n_canvas']) + 0.5) * 1.0, 0.0)
    rows = slice(8, -8)
    zone = slice(int(left[1]) + 1, int(left[1]) + 7)                       # inside the first overlap (8 wide)
    err = {m: float(np.max(np.abs(out[m][rows, zone] - truth[rows, zone]))) for m in out}
    assert err['select'] < 1.0 and err['correct'] < 1.0                    # both right in the overlap ...
    assert err['none'] > 2.0 * err['select']                               # ... the plain blend is not
    interior = slice(30, 100)                                              # no ramp here: the modes agree
    assert np.allclose(out['select'][rows, interior], out['correct'][rows, interior], atol=0.05)
    assert np.allclose(out['select'][rows, interior], out['none'][rows, interior], atol=1e-3)

    # the outer edge has one, ramped, copy: 'select' leaves it dim and says so, 'correct' restores it
    left_in = mg.ramp_left_in(plans['select'], raw.shape[-2])
    assert left_in.shape == out['select'].shape and left_in[:, zone].min() > 0.95 and left_in[:, :3].min() < 0.5
    assert np.allclose(mg.ramp_left_in(plans['correct'], raw.shape[-2]), 1.0)
    assert np.mean(truth[8:-8:2, :3] - out['select'][8:-8:2, :3]) > 2.0    # even lines start dim at the left edge
    assert np.max(np.abs(out['correct'][rows, 2:6] - truth[rows, 2:6])) < 1.5

    # 'select' and 'none' are fixed-weight averages of raw values: linear, and independent of the dark level
    for m in ('select', 'none'):
        assert np.allclose(mg.stitch(2.0 * raw + 7.0, plans[m]), 2.0 * out[m] + 7.0, atol=2e-2)
        off = mg.stitch_plan(N, F, width, OVERLAPS, 1.0, 'average', gain=gain, dark=dark + 10.0, ramp=m)
        assert np.allclose(mg.stitch(raw, off), out[m], atol=2e-2)
    # 'correct' is not: a dark level wrong by 10 counts moves the ramped edge by tens of counts
    off = mg.stitch_plan(N, F, width, OVERLAPS, 1.0, 'average', gain=gain, dark=dark + 10.0, ramp='correct')
    assert np.max(np.abs(mg.stitch(raw, off)[rows, :3] - out['correct'][rows, :3])) > 10.0

    assert mg.ramp_mode(True) == 'correct' and mg.ramp_mode(False) == 'none' and mg.ramp_mode('select') == 'select'
    with pytest.raises(ValueError):
        mg.ramp_mode('flat')
