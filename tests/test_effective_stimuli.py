#!/usr/bin/env python3
"""Unit tests for effective_stimuli: text-log parsing, psychopy-style compositing, seeded pink noise.

Run:  .venv/bin/python -m pytest marmanalysis/tests/test_effective_stimuli.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import effective_stimuli as es

HEADER = """1.5009 \tEXP \tCreated stimulus_window = Window(allowGUI=False, blendMode='avg', bpc=(8, 8, 8), color=array([0, 0, 0]), colorSpace='rgb', gamma=array([[1.9101917],
       [1.9050918],
       [1.8659667]], dtype=float32), name='stimulus_window', pos=[0.0, 0.0], screen=1, size=array([1920, 1080]), units='deg', waitBlanking=True, winType='pyglet')
2.0214 \tEXP \tCreated background0:pinksame115 = ImageStim(color=array([1, 1, 1]), colorSpace='rgb', contrast=1.0, interpolate=False, name='background0:pinksame115', opacity=1.0, ori=0.0, pos=array([0., 0.]), size=array([65.95053948, 37.09717846]), texRes=512, units='deg', win=Window(...))
11.7379 \tEXP \tCreated image0:blank.png = ImageStim(color=array([1., 1., 1.]), colorSpace='rgb', contrast=1.0, flipHoriz=False, flipVert=False, image=str(...), interpolate=False, name='image0:blank.png', opacity=1.0, ori=0.0, pos=array([0., 0.]), size=array([8., 8.]), texRes=512, units='deg', win=Window(...))
12.0000 \tINFO \ttrial 0/10, stim start
"""


@pytest.fixture
def text_log(tmp_path):
    path = tmp_path / '000000tUTC_Stimulus_Images.log'
    path.write_text(HEADER)
    return str(path)


def test_parse_display_reads_window_background_and_images(text_log):
    display = es.parse_display(text_log)
    assert display['window']['size_px'] == [1920.0, 1080.0]
    assert display['window']['color'] == [0.0, 0.0, 0.0]
    assert display['window']['blendMode'] == 'avg'
    assert display['window']['gamma'] == pytest.approx([1.9101917, 1.9050918, 1.8659667])
    assert display['background'] == {'type': 'pinksame', 'intensity': 115, 'n_logged': 1}
    assert display['geometry']['px_per_deg'] == pytest.approx(1920 / 65.95053948)
    image = display['images'][0]
    assert image['name'] == 'blank.png' and image['size'] == [8.0, 8.0] and image['interpolate'] is False


def test_geometry_without_background_matches_psychopy_deg2pix(tmp_path):
    path = tmp_path / 'x_Stimulus_Images.log'
    path.write_text(HEADER.splitlines()[0] + '\n' + '\n'.join(HEADER.splitlines()[1:3]) + '\n')
    display = es.parse_display(str(path))
    assert display['background']['type'] is None
    # 30 cm * 0.017455 cm/deg * 1920 px / 34.535 cm
    assert display['geometry']['px_per_deg'] == pytest.approx(30 * 0.017455 * 1920 / 34.535)


def test_pink_noise_is_seeded_and_in_range():
    first = es.generate_pinknoise_image(64, 48, 115, seed=3)
    again = es.generate_pinknoise_image(64, 48, 115, seed=3)
    other = es.generate_pinknoise_image(64, 48, 115, seed=4)
    assert first.shape == (48, 64)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert first.min() >= -1 and first.max() <= 1


def test_render_composites_alpha_over_background(tmp_path):
    from PIL import Image
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[:2, :, :3] = 255                    # top half white, opaque
    rgba[:2, :, 3] = 255
    rgba[2:, :, :3] = 200                    # bottom half hidden color, fully transparent
    path = tmp_path / 'img.png'
    Image.fromarray(rgba, 'RGBA').save(path)
    screen = np.zeros((40, 40, 3))           # mid-gray window
    geometry = {'px_per_deg': 2.0}
    out = es.render_condition(screen, str(path), {'size': [2.0, 2.0], 'pos': [0.0, 0.0]}, geometry)
    assert out.shape == (4, 4, 3)
    assert (out[:2] == 255).all()            # opaque white stays white
    assert (out[2:] == 128).all()            # transparent pixels show the gray background, not the hidden color
    half = es.render_condition(screen, str(path), {'size': [2.0, 2.0], 'opacity': 0.5, 'contrast': 1.0}, geometry)
    assert (half[:2] == 191).all()           # 0.5 * white + 0.5 * gray in -1..1 space


NEWER_HEADER = """1.5009 \tEXP \tCreated stimulus_window = Window(allowGUI=False, blendMode='avg', bpc=(8, 8, 8), color=array([0, 0, 0]), colorSpace='rgb', gamma=array([[1.9101917],
       [1.9050918],
       [1.8659667]], dtype=float32), name='stimulus_window', pos=[0.0, 0.0], screen=1, size=array([1920, 1080]), units='deg', waitBlanking=True, winType='pyglet')
1.6206 \tEXP \tCreated background_pinksame_intensity140 = ImageStim(color=array([1, 1, 1]), colorSpace='rgb', contrast=1.0, interpolate=False, name='background_pinksame_intensity140', opacity=1.0, ori=0.0, pos=array([0., 0.]), size=array([65.95053948, 37.09717846]), texRes=512, units='deg', win=Window(...))
1.6439 \tEXP \tCreated a1.png = ImageStim(color=array([1., 1., 1.]), colorSpace='rgb', contrast=1.0, flipHoriz=False, flipVert=False, image=str(...), interpolate=False, name='a1.png', opacity=1.0, ori=0.0, pos=array([0., 0.]), size=array([12. ,  7.2]), texRes=512, units='deg', win=Window(...))
12.0000 \tINFO \ttrial 0/10, stim start
"""


def test_parse_display_reads_the_newer_log_format(tmp_path):
    """The multimodal stimcode logs 'Created background_<type>_intensity<n>' and 'Created <name>.png' (no image<k>:
    prefix); recipe 1 missed both and rendered an 8 x 8 deg square on gray instead of 12 x 7.2 deg on pink noise."""
    path = tmp_path / '000000tUTC_Stimulus_Multimodal.log'
    path.write_text(NEWER_HEADER)
    display = es.parse_display(str(path))
    assert display['background'] == {'type': 'pinksame', 'intensity': 140, 'n_logged': 1}
    assert display['geometry']['px_per_deg'] == pytest.approx(1920 / 65.95053948)
    image = next(iter(display['images'].values()))
    assert image['name'] == 'a1.png' and image['size'] == [12.0, 7.2]


def test_render_is_square_with_the_real_surround(tmp_path):
    """A non-square stimulus is rendered inside the square of screen surround around it (nothing for a DNN's center
    crop to cut), at its presented size and aspect; square=False gives the old stimulus-only extent."""
    from PIL import Image
    rgba = np.full((2, 4, 4), 255, np.uint8)                # an opaque white 4 x 2 image
    path = tmp_path / 'wide.png'
    Image.fromarray(rgba, 'RGBA').save(path)
    screen = np.zeros((40, 40, 3))                          # mid-gray window
    out = es.render_condition(screen, str(path), {'size': [4.0, 2.0], 'pos': [0.0, 0.0]}, {'px_per_deg': 2.0})
    assert out.shape == (8, 8, 3)                           # 8 x 4 px stimulus inside an 8 x 8 square
    assert (out[2:6] == 255).all() and (out[:2] == 128).all() and (out[6:] == 128).all()
    old = es.render_condition(screen, str(path), {'size': [4.0, 2.0]}, {'px_per_deg': 2.0}, square=False)
    assert old.shape == (4, 8, 3)


def test_trial_params_ignore_non_numeric_reprs():
    stimlog = pd.DataFrame({'cond': [1], 'size': ['[8. 8.]'], 'pos': ['[0. 0.]'],
                            'ori': ["<method-wrapper '__getattribute__' of attributeSetter object at 0x7f09c2122f70>"],
                            'contrast': [1.0], 'opacity': [np.nan], 'color': ['[1. 1. 1.]'], 'units': ['deg']})
    params = es._trial_params(stimlog, 1)
    assert params == {'size': [8.0, 8.0], 'pos': [0.0, 0.0], 'contrast': 1.0, 'color': [1.0, 1.0, 1.0],
                      'units': 'deg'}
