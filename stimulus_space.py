#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The stimulus feature space and each ROI's preferred axis in it, as a picture and as a map.

Ported from the "stimulus space" sections of old_analysis_scripts/analysis_for_images.py (3541-3740), which followed
the object-space idea of Bao et al. 2020 (https://doi.org/10.1038/s41586-020-2350-5): take a deep network's features
of the presented images, reduce them to their first two principal components, find for each ROI the direction in that
plane along which its response grows, and color the ROI by that direction's angle relative to the face images.

What is kept: the 2-D plane of the first two feature components, one preferred-axis vector per ROI, the angle measured
from the face vector, and the ROI map colored by that angle.

What is changed, and why:
  * Features are computed here from the session's own images as they were shown (som.condition_image_paths), at a
    chosen AlexNet layer, instead of being read from a precomputed fc6 record. ``--layer fc6`` reproduces the old layer.
  * The two components are scaled to unit variance and the responses are z-scored per ROI. The preferred axis is then
    the pair of correlations between the ROI's tuning and the two components. In the old code the response-weighted
    sum and the least-squares weights gave different angles, because PC1 has more variance than PC2; after scaling
    they are the same vector, so the angle no longer depends on that choice.
  * The vector's LENGTH now means something: its square is the share of the ROI's tuning variance that the plane
    explains. Under no relation it is below sqrt(-2 ln(alpha) / n_images) with probability 1 - alpha, drawn as a circle;
    ROIs inside it are shown gray, because their angle is noise.
  * Several hundred arrows from one origin hide each other, so the picture shows the arrow tips, faint spokes for the
    ROIs outside the circle, and a polar histogram of the angles. The same color wheel is used in the ROI map.

Run:  .venv/bin/python marmanalysis/stimulus_space.py --session Cadbury20221016d [--layer relu7] [--gate zeta]
Writes output/stimulus_space_<slug>_layer-<layer>_gate-<gate>_<UTC>.png and prints the summary numbers.
"""
import argparse
import colorsys
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnn_som
import layer_alignment as la
import paths
import response_matrix as rm
import response_table as rt
import sessionio
import som
import spatial_structure as ss
import topography as tg

LAYERS = {'fc6': 'classifier.1', 'relu6': 'classifier.2', 'fc7': 'classifier.4', 'relu7': 'classifier.5'}
CAT_COLORS = {'face': 'tab:red', 'body': 'tab:green', 'object': 'tab:blue', 'animal': 'tab:olive', 'scramble': '0.5',
              'blank': 'k'}
NEAR, FAR = 75.0, 150.0                                    # um: "nearby" and "distant" ROI pairs
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')


def feature_plane(features, n_components=2):
    """First principal components of image features, each scaled to unit variance: (n_images, n_components), and the
    share of the feature variance each one carries."""
    f = np.asarray(features, float)
    f = f - f.mean(0)
    u, s, _ = np.linalg.svd(f, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    return scores / scores.std(0), (s[:n_components] ** 2) / (s ** 2).sum()


def preferred_axes(response, plane):
    """Each ROI's preferred axis in the feature plane: the correlation of its tuning (z-scored over images) with each
    unit-variance component. (n_roi, n_components); the squared length is the tuning variance the plane explains."""
    r = np.asarray(response, float)
    z = (r - np.nanmean(r, 1, keepdims=True)) / (np.nanstd(r, 1, keepdims=True) + 1e-12)
    p = (plane - plane.mean(0)) / plane.std(0)
    return np.nan_to_num(z) @ p / plane.shape[0]


def noise_radius(n_images, alpha=0.05):
    """Length that a preferred axis exceeds with probability ``alpha`` when tuning and features are unrelated
    (n * length^2 is then chi-square with 2 degrees of freedom)."""
    return float(np.sqrt(-2.0 * np.log(alpha) / n_images))


def signed_angle(vectors, reference):
    """Angle (degrees, -180..180) from ``reference`` to each 2-D vector, counterclockwise positive."""
    ref = np.arctan2(reference[1], reference[0])
    ang = np.arctan2(vectors[:, 1], vectors[:, 0]) - ref
    return np.degrees((ang + np.pi) % (2 * np.pi) - np.pi)


def angle_colors(angles_deg, strength=None):
    """Hue from the angle (0 = along the face vector = red), saturation from ``strength`` in [0, 1]."""
    s = np.ones(len(angles_deg)) if strength is None else np.clip(strength, 0, 1)
    return np.array([colorsys.hsv_to_rgb((a % 360) / 360.0, sat, 1.0) for a, sat in zip(angles_deg, s)])


def angle_smoothness(angles_deg, xy, n_perm=500, seed=0):
    """Do nearby ROIs prefer similar directions? Mean cos(angle difference) for pairs closer than NEAR minus pairs
    farther than FAR, with a position-shuffle p-value."""
    a = np.radians(angles_deg)
    iu = np.triu_indices(len(a), 1)
    d = tg.pairwise_distance(xy)[iu]
    near, far = d < NEAR, d > FAR
    if near.sum() < 10 or far.sum() < 10:
        return np.nan, np.nan

    def stat(v):
        c = np.cos(v[iu[0]] - v[iu[1]])
        return c[near].mean() - c[far].mean()
    obs = stat(a)
    rng = np.random.default_rng(seed)
    null = np.array([stat(a[rng.permutation(len(a))]) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + n_perm))


def analyse(session, layer='relu7', gate='auto'):
    slug = sessionio.session_slug(session)
    data = rm.build_response_matrix(session, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    ck = np.array([ss.cat_key(c) for c in data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()])
    image_paths = som.condition_image_paths(data)
    keep = np.array([p is not None and os.path.exists(p) for p in image_paths]) & ~np.isin(ck, ('blank', 'other'))
    zeta_csv = paths.tmp('responsiveness_%s.csv' % slug)
    if gate == 'auto':
        gate = 'zeta' if os.path.exists(zeta_csv) else 'responsive'
    if gate == 'zeta':
        mask = pd.read_csv(zeta_csv)['p_zeta'].to_numpy() < 0.05
    elif gate == 'responsive':
        mask = np.asarray(rt.classify_responses(data['ds'], metric='Fzsc', framerate=data['framerate'], alpha=0.05)['responsive'], bool)
    else:
        mask = np.ones(data['response'].shape[0], bool)
    d = rm.apply_roi_mask(data, mask)
    resp, xy, ck = d['response'][:, keep], d['roi_xy_um'], ck[keep]
    model, _ = dnn_som.prep_dnn_model()
    images = dnn_som.load_images_rgb([p for p, k in zip(image_paths, keep) if k])
    feats = la.layer_activations(model, images, [LAYERS[layer]])[LAYERS[layer]]
    plane, share = feature_plane(feats)
    face = ck == 'face'
    face_vec = plane[face].mean(0) if face.any() else np.array([1.0, 0.0])
    axes = preferred_axes(resp, plane)
    length = np.linalg.norm(axes, axis=1)
    radius = noise_radius(plane.shape[0])
    angle = signed_angle(axes, face_vec)
    return {'slug': slug, 'session': session, 'layer': layer, 'gate': gate, 'plane': plane, 'share': share, 'ck': ck,
            'face_vec': face_vec, 'axes': axes, 'length': length, 'radius': radius, 'angle': angle, 'xy': xy,
            'n_total': int(mask.size), 'n_gated': int(mask.sum())}


def report(session, layer='relu7', gate='auto', out_png=None):
    r = analyse(session, layer, gate)
    plane, axes, length, angle, radius, xy = r['plane'], r['axes'], r['length'], r['angle'], r['radius'], r['xy']
    sig = length > radius
    resultant = np.abs(np.mean(np.exp(1j * np.radians(angle[sig])))) if sig.any() else np.nan
    mean_dir = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(angle[sig]))))) if sig.any() else np.nan
    smooth, p_smooth = angle_smoothness(angle[sig], xy[sig]) if sig.sum() > 30 else (np.nan, np.nan)
    print('%s | layer %s | gate %s: %d of %d ROIs | %d images' % (r['slug'], layer, r['gate'], r['n_gated'], r['n_total'], plane.shape[0]))
    print('feature plane: PC1 %.1f%% and PC2 %.1f%% of the feature variance | face vector at %.0f deg from PC1, length %.2f'
          % (100 * r['share'][0], 100 * r['share'][1], np.degrees(np.arctan2(r['face_vec'][1], r['face_vec'][0])), np.linalg.norm(r['face_vec'])))
    print('preferred axes: median length %.2f (the plane explains %.0f%% of the median ROI\'s tuning variance) | noise radius %.2f | '
          '%d of %d ROIs (%.0f%%) lie outside it (5%% expected by chance)'
          % (np.median(length), 100 * np.median(length) ** 2, radius, sig.sum(), sig.size, 100 * sig.mean()))
    print('angles of those ROIs relative to the face vector: mean direction %+.0f deg, concentration %.2f (0 = uniform, 1 = identical) | '
          'within 45 deg of the face vector %.0f%%, within 45 deg of its opposite %.0f%%'
          % (mean_dir, resultant, 100 * np.mean(np.abs(angle[sig]) < 45), 100 * np.mean(np.abs(angle[sig]) > 135)))
    print('do nearby ROIs prefer similar directions? cos(angle difference) for pairs < %.0f um minus pairs > %.0f um: %+.3f (position shuffle p = %.3f)'
          % (NEAR, FAR, smooth, p_smooth))

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    for c in sorted(set(r['ck'])):
        m = r['ck'] == c
        ax.scatter(plane[m, 0], plane[m, 1], s=42, color=CAT_COLORS.get(c, '0.3'), edgecolor='k', linewidth=0.5, label='%s (%d)' % (c, m.sum()))
    ax.annotate('', xy=r['face_vec'], xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='k', lw=2))
    ax.text(r['face_vec'][0] * 0.5, r['face_vec'][1] * 0.5 - 0.22, 'face vector', fontsize=8, ha='center')
    ax.axhline(0, color='0.8', lw=0.5); ax.axvline(0, color='0.8', lw=0.5); ax.set_aspect('equal')
    ax.set_xlabel('feature PC1 (unit variance, %.0f%% of the feature variance)' % (100 * r['share'][0]))
    ax.set_ylabel('feature PC2 (%.0f%%)' % (100 * r['share'][1]))
    ax.set_title('the %d images in the AlexNet %s feature plane' % (plane.shape[0], layer), fontsize=10); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    colors = angle_colors(angle, (length - radius) / max(length.max() - radius, 1e-9) * 0.7 + 0.3)
    ax.scatter(axes[~sig, 0], axes[~sig, 1], s=7, color='0.75', label='inside the noise radius (%d)' % (~sig).sum())
    for a_, c_ in zip(axes[sig], colors[sig]):
        ax.plot([0, a_[0]], [0, a_[1]], '-', color=c_, lw=0.4, alpha=0.25)
    ax.scatter(axes[sig, 0], axes[sig, 1], s=14, color=colors[sig], edgecolor='k', linewidth=0.2, label='preferred axis (%d ROIs)' % sig.sum())
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(radius * np.cos(th), radius * np.sin(th), 'k--', lw=0.8, label='noise radius (95% of unrelated ROIs inside)')
    unit = r['face_vec'] / np.linalg.norm(r['face_vec'])
    lim = max(length.max() * 1.1, radius * 1.5)
    ax.annotate('', xy=unit * lim * 0.95, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('correlation of tuning with feature PC1'); ax.set_ylabel('correlation of tuning with feature PC2')
    ax.set_title('each ROI\'s preferred axis (tip of its vector); black arrow = face direction', fontsize=10); ax.legend(fontsize=7, loc='lower left')

    ax = fig.add_subplot(gs[1, 0], projection='polar')
    bins = np.linspace(-np.pi, np.pi, 25)
    h, _ = np.histogram(np.radians(angle[sig]), bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers, h, width=np.diff(bins), color=angle_colors(np.degrees(centers)), edgecolor='k', linewidth=0.3)
    ax.set_theta_zero_location('E'); ax.set_title('preferred-axis angle relative to the face vector (0 deg)\nROIs outside the noise radius', fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(length, bins=40, color='0.6'); ax.axvline(radius, color='k', ls='--')
    ax.set_xlabel('preferred-axis length (length$^2$ = tuning variance explained by the plane)'); ax.set_ylabel('ROIs')
    ax.set_title('%.0f%% of ROIs outside the noise radius' % (100 * sig.mean()), fontsize=10)

    ax = fig.add_subplot(gs[:, 2])
    try:
        bg, rx, ry = ss.mean_image(session)
        v = np.percentile(bg, [1, 99.5])
        ax.imshow(bg, cmap='gray', vmin=v[0], vmax=v[1], extent=[0, bg.shape[1] * rx, bg.shape[0] * ry, 0], alpha=0.55)
    except Exception:                                        # noqa: BLE001
        pass
    ax.scatter(xy[~sig, 0], xy[~sig, 1], s=12, facecolor='none', edgecolor='w', linewidth=0.5)
    ax.scatter(xy[sig, 0], xy[sig, 1], s=26, color=colors[sig], edgecolor='k', linewidth=0.3)
    ss._frame(ax, xy)
    ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
    ax.set_title('ROI map colored by preferred-axis angle relative to the face vector\n(open circles: inside the noise '
                 'radius)\nnearby ROIs more alike than distant ones: %+.3f, p = %.3f' % (smooth, p_smooth), fontsize=9)
    wheel = fig.add_axes([0.915, 0.86, 0.06, 0.06], projection='polar')
    tw = np.linspace(-np.pi, np.pi, 73)
    wheel.bar(tw[:-1], 1, width=np.diff(tw), bottom=0.5, color=angle_colors(np.degrees(tw[:-1])), linewidth=0)
    wheel.set_yticks([]); wheel.set_xticks([0, np.pi]); wheel.set_xticklabels(['face', 'opposite'], fontsize=7); wheel.spines['polar'].set_visible(False)
    fig.suptitle('%s: stimulus feature space and preferred axes (AlexNet %s, gate %s, %d ROIs)'
                 % (r['slug'], layer, r['gate'], r['n_gated']), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = out_png or paths.output('stimulus_space_%s_layer-%s_gate-%s_%s.png' % (r['slug'], layer, r['gate'], STAMP))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)
    print('saved %s' % out_png)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', default=ss.DEFAULT_SLUG, help='session slug (or a unique prefix of one), or a session folder')
    ap.add_argument('--layer', default='relu7', choices=sorted(LAYERS), help='AlexNet layer for the image features')
    ap.add_argument('--gate', default='auto', choices=['auto', 'zeta', 'responsive', 'all'],
                    help="which ROIs: 'auto' uses the session's ZETA table when there is one, else ANOVA-responsive")
    a = ap.parse_args()
    report(a.session if os.path.isdir(a.session) else paths.session(a.session), a.layer, a.gate)


if __name__ == '__main__':
    main()
