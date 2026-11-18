"""Tests for the feature-plane and preferred-axis arithmetic in stimulus_space, on synthetic tuning."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import stimulus_space as sp  # noqa: E402


def test_feature_plane_is_centred_with_unit_variance():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((60, 200)) * np.linspace(5, 0.1, 200)        # decaying variance across units
    plane, share = sp.feature_plane(feats)
    assert plane.shape == (60, 2) and np.allclose(plane.mean(0), 0) and np.allclose(plane.std(0), 1)
    assert abs(np.corrcoef(plane[:, 0], plane[:, 1])[0, 1]) < 1e-9 and share[0] > share[1] > 0


def test_preferred_axis_recovers_the_direction_whatever_the_component_variances():
    rng = np.random.default_rng(1)
    n = 400
    plane, _ = sp.feature_plane(rng.standard_normal((n, 30)))            # uncorrelated, unit-variance components
    directions = np.radians([0, 45, 90, 135, 180, 270])
    resp = np.stack([np.cos(d) * plane[:, 0] + np.sin(d) * plane[:, 1] for d in directions])
    resp = resp * rng.uniform(0.5, 20, (len(directions), 1)) + rng.uniform(-3, 3, (len(directions), 1))   # gain, offset
    axes = sp.preferred_axes(resp, plane)
    got = np.arctan2(axes[:, 1], axes[:, 0])
    assert np.max(np.abs(np.degrees(np.angle(np.exp(1j * (got - directions)))))) < 1.0     # compared on the circle
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=0.03)         # the plane explains all of the tuning
    stretched = plane * np.array([7.0, 0.3])                                 # unequal variances must not turn the axis
    again = sp.preferred_axes(resp, stretched)
    assert np.allclose(again, axes, atol=1e-9)


def test_noise_radius_holds_for_unrelated_tuning():
    rng = np.random.default_rng(2)
    n = 60
    plane = rng.standard_normal((n, 2))
    plane = (plane - plane.mean(0)) / plane.std(0)
    length = np.linalg.norm(sp.preferred_axes(rng.standard_normal((4000, n)), plane), axis=1)
    assert abs(np.mean(length > sp.noise_radius(n, 0.05)) - 0.05) < 0.015


def test_signed_angle_and_colors():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    assert np.allclose(sp.signed_angle(v, np.array([1.0, 0.0])), [0, 90, -180, -90, 45])
    assert np.allclose(sp.signed_angle(v, np.array([0.0, 2.0])), [-90, 0, 90, -180, -45])
    c = sp.angle_colors(np.array([0.0, 120.0, 240.0]))
    assert np.allclose(c, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])               # along the face vector = red


def test_angle_smoothness_detects_a_gradient():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 600, (300, 2))
    smooth_map = xy[:, 0] / 600 * 180                                      # the preferred angle changes across the field
    obs, p = sp.angle_smoothness(smooth_map, xy, n_perm=200)
    assert obs > 0.2 and p < 0.02
    obs0, p0 = sp.angle_smoothness(rng.uniform(-180, 180, 300), xy, n_perm=200)
    assert abs(obs0) < 0.1 and p0 > 0.02
