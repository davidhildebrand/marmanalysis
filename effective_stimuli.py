"""Per-session effective stimuli: every presented image condition rendered as it appeared on the screen.

The image files a stimulus log names are not what the animal saw. Object PNGs carry transparency, and psychopy
drew them over the session's background: the window color (``background_type = None``, e.g.
Cadbury20221016d152643tUTC, uniform mid-gray) or, in most image sessions, a full-screen 1/f pink-noise image
(``background_type = 'pinksame'``: one noise image behind every condition; ``'pink'``: one per condition) made by
the stimulus code's ``generate_pinknoise_image`` with UNSEEDED ``np.random`` and never saved. The exact noise
pixels are therefore unrecoverable; this module regenerates statistically equivalent noise with the same generator
and an explicit seed (several seeds can be rendered and their model features averaged).

Everything is read from the session's own logs (never assumed):
  * the text ``.log`` header: the psychopy Window (size in pixels, color and colorSpace, gamma, blendMode), the
    background stimuli (``Created background<k>:<type><intensity> = ImageStim(... size=array([w, h]) ...)``,
    whose full-field size in degrees against the window size in pixels gives pixels per degree), and the image
    stimuli (``Created image<k>:<name> = ImageStim(...)``: size, pos, ori, color, contrast, opacity, interpolate);
  * the trial log: per-condition size / pos / ori / contrast / opacity when recorded (these override the
    creation values);
  * geometry fallback when the log has no background stimulus: the session's ``*_stimcode.py`` copy
    (``disp_pixels``, ``disp_width``, ``disp_distance``), else the rig's ASUS XG16AHPE (1920 x 1080 px,
    34.535 cm wide, 30 cm away); psychopy's ``deg2pix`` (small-angle, ``correctFlat=False``) converts.

Rendering per condition (psychopy semantics, in its -1..1 rgb space): the image is stretched to its logged size
in screen pixels (nearest-neighbour when ``interpolate=False``, as logged), its rgb multiplied by ``color`` and
``contrast``, its alpha multiplied by ``opacity``, and alpha-blended over the background pixels it covered
(``blendMode='avg'``). The render is the on-screen SQUARE around the image (side = its longer dimension, plus
``margin_deg``), filled with the real surround -- so a non-square image keeps its presented size and aspect and a
DNN's center crop removes nothing -- saved as RGB PNG without alpha. Both log formats are parsed: the older
``Created image<k>:<name>`` / ``Created background<k>:<type><intensity>`` and the newer ``Created <name>.png`` /
``Created background_<type>_intensity<n>`` (multimodal stimcode; recipe 1 missed these and fell back to an 8 x 8
deg square on gray). Code-generated blank conditions render the background alone. Monitor
gamma is recorded, not applied: renders are the intended frame-buffer values, as the image files are.

Output (written inside the imaging session folder, as David asked):
``<session>/effective_stimuli/<text log stem>_<UTC stamp>/seed<k>/cond<id>_<image stem>.png`` plus
``manifest.json`` with the recipe. ``effective_image_paths`` resolves the newest matching render (and renders
when none exists), aligned with ``data['condition_ids']`` like ``som.condition_image_paths``.
"""
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np

RECIPE_VERSION = 2       # 2 (2026-10-18): square canvas of real surround; newer log formats parsed (sizes, pink noise)
DEFAULT_DISPLAY = {'pixels': (1920, 1080), 'width_cm': 34.535, 'distance_cm': 30.0,
                   'source': 'default rig display ASUS XG16AHPE'}
_ARRAY = r'array\(\[([^\]]*)\]\)'


def _numbers(text):
    return [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text))]


def _field(spec, key):
    """Value of ``key=`` in a psychopy repr: numbers for arrays/numbers, text for quoted strings, else raw."""
    match = re.search(r'\b{}=({}|\'[^\']*\'|[^,()]+)'.format(re.escape(key), _ARRAY), spec)
    if not match:
        return None
    raw = match.group(1)
    if raw.startswith('array('):
        return _numbers(match.group(2))
    if raw.startswith("'"):
        return raw.strip("'")
    values = _numbers(raw)
    if raw.strip() in ('True', 'False'):
        return raw.strip() == 'True'
    if raw.strip() == 'None':
        return None
    return values[0] if len(values) == 1 else raw.strip()


def _header_text(text_log, max_lines=20000):
    lines = []
    with open(text_log, errors='replace') as handle:
        for number, line in enumerate(handle):
            if number >= max_lines or re.search(r'\btrial 0/', line):
                break
            lines.append(line)
    return ''.join(lines)


def parse_display(text_log, stimcode=None):
    """Window, background and image-stimulus settings recorded in a session's text log header."""
    header = _header_text(text_log)
    window_match = re.search(r'Created (\S+) = Window\((.*?)winType=', header, re.S)
    if not window_match:
        raise ValueError('no psychopy Window in the header of {}'.format(text_log))
    window_spec = window_match.group(2)
    window = {'name': window_match.group(1), 'size_px': _field(window_spec, 'size'),
              'color': _field(window_spec, 'color'), 'colorSpace': _field(window_spec, 'colorSpace'),
              'blendMode': _field(window_spec, 'blendMode'), 'units': _field(window_spec, 'units')}
    gamma = re.search(r'gamma=array\((\[.*?\])\s*,\s*dtype', window_spec, re.S)
    window['gamma'] = _numbers(gamma.group(1)) if gamma else None
    if window['colorSpace'] not in (None, 'rgb'):
        raise NotImplementedError('window colorSpace {!r}'.format(window['colorSpace']))

    backgrounds = {}
    for match in re.finditer(r'Created background(\d+):([A-Za-z]+?)(\d*) = ImageStim\((.*)', header):
        spec = match.group(4)
        backgrounds[int(match.group(1))] = {'type': match.group(2), 'intensity': (int(match.group(3))
                                            if match.group(3) else None),
                                            'size_deg': _field(spec, 'size'), 'pos_deg': _field(spec, 'pos'),
                                            'units': _field(spec, 'units')}
    # newer logs (e.g. the multimodal stimcode): "Created background_pinksame_intensity140 = ImageStim(..."
    for k, match in enumerate(re.finditer(r'Created background_([A-Za-z]+)_intensity(\d+) = ImageStim\((.*)', header)):
        spec = match.group(3)
        backgrounds[len(backgrounds) + k] = {'type': match.group(1), 'intensity': int(match.group(2)),
                                             'size_deg': _field(spec, 'size'), 'pos_deg': _field(spec, 'pos'),
                                             'units': _field(spec, 'units')}
    image_lines = [(int(m.group(1)), m.group(2), m.group(3))
                   for m in re.finditer(r'Created image(\d+):(.+?) = ImageStim\((.*)', header)]
    # newer logs: "Created a1.png = ImageStim(..." (no image<k>: prefix)
    image_lines += [(10 ** 6 + k, m.group(1), m.group(2)) for k, m in enumerate(
        re.finditer(r'Created ([^\s=:]+\.(?:png|jpe?g|bmp|tiff?)) = ImageStim\((.*)', header, re.I))]
    images = {}
    for key, name, spec in image_lines:
        images[key] = {
            'name': name, 'units': _field(spec, 'units'), 'size': _field(spec, 'size'),
            'pos': _field(spec, 'pos'), 'ori': _field(spec, 'ori'), 'color': _field(spec, 'color'),
            'colorSpace': _field(spec, 'colorSpace'), 'contrast': _field(spec, 'contrast'),
            'opacity': _field(spec, 'opacity'), 'interpolate': _field(spec, 'interpolate'),
            'flipHoriz': _field(spec, 'flipHoriz'), 'flipVert': _field(spec, 'flipVert')}

    types = {b['type'] for b in backgrounds.values()}
    if len(types) > 1:
        raise ValueError('several background types in {}: {}'.format(text_log, types))
    background = {'type': types.pop() if types else None,
                  'intensity': next((b['intensity'] for b in backgrounds.values()), None),
                  'n_logged': len(backgrounds)}
    geometry = _geometry(window, backgrounds, stimcode)
    return {'text_log': text_log, 'window': window, 'background': background, 'geometry': geometry,
            'images': images}


def _geometry(window, backgrounds, stimcode):
    """Pixels per degree: from a full-field background's logged degrees, else the stimcode display, else default."""
    size_px = window['size_px']
    for b in backgrounds.values():
        if b['size_deg'] and b['units'] == 'deg' and size_px:
            per_axis = [size_px[0] / b['size_deg'][0], size_px[1] / b['size_deg'][1]]
            return {'px_per_deg': float(np.mean(per_axis)), 'px_per_deg_xy': per_axis,
                    'window_px': size_px, 'source': 'full-field background size in the text log'}
    display = dict(DEFAULT_DISPLAY)
    if stimcode and os.path.isfile(stimcode):
        code = open(stimcode, errors='replace').read()
        found = {}
        for key, name in (('disp_width', 'width_cm'), ('disp_distance', 'distance_cm'), ('disp_pixels', 'pixels')):
            block = re.search(r'^{}\s*=\s*\{{(.*?)\}}'.format(key), code, re.S | re.M)
            entry = re.search(r"'MarmoScope'\s*:\s*([^,\n#]+(?:,\s*\d+\))?)", block.group(1)) if block else None
            if entry:
                values = _numbers(entry.group(1))
                found[name] = tuple(int(v) for v in values) if name == 'pixels' else values[0]
        if len(found) == 3:
            display = {**found, 'source': 'stimcode copy {}'.format(os.path.basename(stimcode))}
    pixels = size_px or display['pixels']
    # psychopy.tools.monitorunittools.deg2pix with correctFlat=False: deg -> cm by distance*0.017455, cm -> px.
    px_per_deg = display['distance_cm'] * 0.017455 * pixels[0] / display['width_cm']
    return {'px_per_deg': float(px_per_deg), 'px_per_deg_xy': [float(px_per_deg)] * 2, 'window_px': list(pixels),
            'source': display['source'], 'display': display}


def generate_pinknoise_image(width=1920, height=1080, intensity=115, seed=0):
    """The stimulus code's generator (ported from Xindong Song's MATLAB 'pinkinoiseimage'), with a seed.

    Same algorithm and value mapping as the stimulus scripts; only the random phases come from
    ``np.random.RandomState(seed)`` instead of the unseeded global state. Returns height x width in -1..1."""
    random = np.random.RandomState(seed)
    beta = 1
    fhw = round((width / 2) + 1)
    fhh = round((height / 2) + 1)
    fhimagea = np.arange(0, fhh)[:, np.newaxis] @ np.ones([1, fhw])
    fhimageb = np.ones([fhh, 1]) @ np.arange(0, fhw)[np.newaxis, :]
    fhimagef = (fhimagea ** 2 + fhimageb ** 2) ** (1 / 2)
    with np.errstate(divide='ignore'):
        fhimagefbeta = fhimagef ** (-beta)
    fhimagefbeta[0, 0] = 0
    fhimageagl1 = random.rand(fhh, fhw) * 2 * np.pi
    fhimageagl1[-1, 0] = 0
    fhimageagl1[0, -1] = 0
    fhimageagl1[-1, -1] = 0
    fhimageagl2 = random.rand(fhh, fhw) * 2 * np.pi
    fhimagecomp1 = fhimagefbeta * np.exp(1j * fhimageagl1)
    fhimagecomp2 = fhimagefbeta * np.exp(1j * fhimageagl2)
    fimagecomp = np.concatenate((fhimagecomp1,
                                 np.fliplr(np.concatenate((np.conj(fhimagecomp1[0, 1:-1][np.newaxis, :]),
                                                           fhimagecomp2[1:-1, 1:-1],
                                                           np.conj(fhimagecomp1[-1, 1:-1][np.newaxis, :]))))), 1)
    fimagecomp = np.concatenate((fimagecomp,
                                 np.conj(np.flipud(np.concatenate((fhimagecomp1[1:-1, 0][:, np.newaxis],
                                                                   np.fliplr(fimagecomp[1:-1, 1::])), 1)))))
    imagetmp = np.fft.ifft2(fimagecomp)
    imagemaxn = np.abs(np.max(imagetmp) / (255 - intensity))
    imageminn = np.abs(np.min(imagetmp) / (intensity - 0))
    image = (imagetmp.real / max(imagemaxn, imageminn)) + intensity
    # skimage.exposure.rescale_intensity(image, in_range=(0, 255), out_range=(-1, 1)): clip, then map linearly.
    return np.clip(image, 0, 255) / 255.0 * 2.0 - 1.0


def _screen_background(display, seed, condition=None):
    """Full-screen background in -1..1 rgb (height x width x 3)."""
    width, height = (int(v) for v in display['geometry']['window_px'])
    kind = display['background']['type']
    if kind is None:
        color = np.asarray(display['window']['color'] or (0, 0, 0), float)
        return np.broadcast_to(color[None, None, :], (height, width, 3))
    if kind not in ('pink', 'pinksame'):
        raise NotImplementedError('background type {!r}'.format(kind))
    intensity = display['background']['intensity'] or 115
    noise_seed = seed if kind == 'pinksame' else int(np.random.SeedSequence([seed, int(condition)]).generate_state(1)[0])
    noise = generate_pinknoise_image(width, height, intensity, noise_seed)
    return np.repeat(noise[:, :, None], 3, axis=2)


def render_condition(screen, image_path, params, geometry, margin_deg=0.0, square=True):
    """One condition as drawn: returns uint8 RGB of the on-screen region it covered (plus margin). With ``square``
    (default) the region is the square around the stimulus (side = its longer dimension), filled with the screen
    surround it had -- the stimulus at its presented size and aspect inside a square of the real background, i.e.
    the square-padded alpha version (e.g. Song's 480288_equalized_RGBAsq) composited as presented. A DNN's
    Resize + CenterCrop then crops nothing (a 480x288 image shown at 12 x 7.2 deg used to lose 40% of its width)."""
    from PIL import Image
    ppd = geometry['px_per_deg']
    height, width = screen.shape[:2]
    size_deg = np.asarray(params.get('size') or (8.0, 8.0), float)
    pos_deg = np.asarray(params.get('pos') or (0.0, 0.0), float)
    if params.get('units', 'deg') not in ('deg', None, ''):
        raise NotImplementedError('stimulus units {!r}'.format(params.get('units')))
    stim_w, stim_h = (max(1, int(round(v * ppd))) for v in size_deg)
    margin = int(round(margin_deg * ppd))
    pad_x = pad_y = 0
    if square:
        side = max(stim_w, stim_h)
        pad_x, pad_y = (side - stim_w) // 2, (side - stim_h) // 2
    centre_x, centre_y = width / 2 + pos_deg[0] * ppd, height / 2 - pos_deg[1] * ppd
    left = int(round(centre_x - stim_w / 2)) - margin - pad_x
    top = int(round(centre_y - stim_h / 2)) - margin - pad_y
    out_w, out_h = stim_w + 2 * (margin + pad_x), stim_h + 2 * (margin + pad_y)
    if square:
        out_w = out_h = max(out_w, out_h)
    canvas = np.empty((out_h, out_w, 3))
    canvas[:] = screen[height // 2, width // 2]          # outside the screen: never expected; keep finite
    sx0, sy0 = max(left, 0), max(top, 0)
    sx1, sy1 = min(left + out_w, width), min(top + out_h, height)
    canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = screen[sy0:sy1, sx0:sx1]
    if image_path is not None:
        source = Image.open(image_path).convert('RGBA')
        if params.get('flipHoriz'):
            source = source.transpose(Image.FLIP_LEFT_RIGHT)
        if params.get('flipVert'):
            source = source.transpose(Image.FLIP_TOP_BOTTOM)
        resample = Image.BILINEAR if params.get('interpolate') else Image.NEAREST
        source = source.resize((stim_w, stim_h), resample)
        ori = float(params.get('ori') or 0.0)
        if ori:
            source = source.rotate(-ori, resample=resample, expand=False)   # psychopy ori: clockwise degrees
        rgba = np.asarray(source, float) / 255.0
        rgb = rgba[..., :3] * 2.0 - 1.0
        color = np.asarray(params.get('color') or (1.0, 1.0, 1.0), float)
        contrast = 1.0 if params.get('contrast') is None else float(params['contrast'])
        opacity = 1.0 if params.get('opacity') is None else float(params['opacity'])
        rgb = rgb * color[None, None, :] * contrast
        alpha = rgba[..., 3:4] * opacity
        y0, x0 = margin + pad_y, margin + pad_x
        region = canvas[y0:y0 + stim_h, x0:x0 + stim_w]
        canvas[y0:y0 + stim_h, x0:x0 + stim_w] = alpha * rgb + (1.0 - alpha) * region
    return np.clip(np.round((canvas + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)


def _trial_params(stimlog, cond, cond_col='cond'):
    """Presentation parameters a condition's first trial row recorded (None where not recorded)."""
    rows = stimlog[stimlog[cond_col] == cond]
    if not len(rows):
        return {}
    row = rows.iloc[0]
    out = {}
    clean = re.compile(r'^\s*[\[(]?\s*[-+\d.eE\s,]+[\])]?\s*$')    # numbers only, not e.g. a method-wrapper repr
    for key in ('size', 'pos', 'ori', 'contrast', 'opacity', 'color', 'units'):
        if key not in row.index or row[key] is None or (isinstance(row[key], float) and np.isnan(row[key])):
            continue
        value = row[key]
        if key == 'units':
            if str(value).strip():
                out[key] = str(value).strip()
            continue
        if not isinstance(value, (int, float, np.number)) and not clean.match(str(value)):
            continue
        numbers = _numbers(value)
        if key in ('size', 'pos', 'color'):
            if (key == 'color' and len(numbers) != 3) or not numbers:
                continue
            out[key] = numbers if len(numbers) > 1 else numbers * 2
        elif numbers:
            out[key] = numbers[0]
    return out


def _git_hash():
    try:
        return subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def render_session(session_path, conditions, stimlog, text_log, seeds=(0,), margin_deg=0.0, cond_col='cond'):
    """Render every image or blank condition of a session; returns the manifest (also written to disk)."""
    from PIL import Image
    stimcode = sorted(glob.glob(os.path.join(os.path.dirname(text_log), '*_stimcode.py')))
    display = parse_display(text_log, stimcode[0] if stimcode else None)
    stem = os.path.splitext(os.path.basename(text_log))[0]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')
    root = os.path.join(session_path, 'effective_stimuli', '{}_{}'.format(stem, stamp))
    by_name = {v['name']: v for v in display['images'].values()}
    seeds = tuple(int(s) for s in seeds)
    if display['background']['type'] is None:
        seeds = seeds[:1]                          # a uniform background does not depend on the seed
    records = []
    for cond, row in conditions.iterrows():
        path = row.get('imagepath')
        path = None if path is None or (isinstance(path, float) and np.isnan(path)) else str(path)
        blank = bool(row.get('blank_source')) and str(row.get('blank_source')) not in ('nan', 'None')
        if path is None and not blank:
            continue                                # not an image condition
        name = row.get('imagename')
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = os.path.basename(path) if path else 'blank'
        name = str(name)
        params = dict(by_name.get(os.path.basename(path), {}) if path else {})
        params.update(_trial_params(stimlog, cond, cond_col))
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError('condition {} image {} is missing'.format(cond, path))
        records.append({'cond': int(cond) if np.isscalar(cond) else str(cond), 'imagename': name,
                        'imagepath': path, 'imagepath_source': row.get('imagepath_source'),
                        'blank_source': row.get('blank_source') if blank else None,
                        'params': params, 'outputs': {}})
    if not records:
        raise ValueError('no image or blank conditions to render in {}'.format(session_path))
    # Code-generated blanks have no logged size: use the median size of the session's image conditions.
    sizes = [r['params']['size'] for r in records if r['params'].get('size')]
    default_size = list(np.median(np.asarray(sizes, float), axis=0)) if sizes else [8.0, 8.0]
    for seed in seeds:
        os.makedirs(os.path.join(root, 'seed{}'.format(seed)), exist_ok=True)
        shared = _screen_background(display, seed) if display['background']['type'] != 'pink' else None
        for record in records:
            params = dict(record['params'])
            params.setdefault('size', default_size)
            screen = shared if shared is not None else _screen_background(display, seed, record['cond'])
            rendered = render_condition(screen, record['imagepath'], params, display['geometry'], margin_deg)
            label = os.path.splitext(record['imagename'])[0] if record['imagename'] else 'blank'
            relative = os.path.join('seed{}'.format(seed), 'cond{}_{}.png'.format(record['cond'],
                                                                                  re.sub(r'[^\w.-]+', '_', label)))
            Image.fromarray(rendered, 'RGB').save(os.path.join(root, relative))
            record['outputs'][str(seed)] = relative
            record['size_px'] = list(rendered.shape[:2][::-1])
    manifest = {'recipe_version': RECIPE_VERSION, 'created_utc': stamp, 'marmanalysis_git': _git_hash(),
                'session_path': session_path, 'text_log': text_log, 'stimcode': stimcode[0] if stimcode else None,
                'window': display['window'], 'geometry': display['geometry'], 'background': {
                    **display['background'], 'seeds': list(seeds),
                    'generator': 'generate_pinknoise_image (stimulus code port), np.random.RandomState(seed)'
                                 + ('; per condition SeedSequence([seed, cond])' if display['background']['type'] == 'pink' else '')
                                 if display['background']['type'] else 'uniform window color'},
                'margin_deg': float(margin_deg), 'gamma_applied': False, 'conditions': records}
    with open(os.path.join(root, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1, default=str)
    return {**manifest, 'root': root}


def find_render(session_path, text_log, seeds=(0,), margin_deg=0.0):
    """Newest render of this log with the current recipe, the margin and all requested seeds; else None."""
    stem = os.path.splitext(os.path.basename(text_log))[0]
    for root in sorted(glob.glob(os.path.join(session_path, 'effective_stimuli', stem + '_*')), reverse=True):
        try:
            manifest = json.load(open(os.path.join(root, 'manifest.json')))
        except (OSError, ValueError):
            continue
        rendered = set(manifest['background']['seeds'])
        wanted = set(int(s) for s in seeds) if manifest['background']['type'] else set(list(rendered)[:1])
        if (manifest.get('recipe_version') == RECIPE_VERSION and float(manifest.get('margin_deg', 0)) == margin_deg
                and wanted <= rendered):
            return {**manifest, 'root': root}
    return None


def effective_image_paths(data, seed=0, margin_deg=0.0, render_missing=True, seeds_to_render=None):
    """Effective-stimulus paths aligned with ``data['condition_ids']`` (None for non-image conditions).

    ``data`` is a ``response_matrix.build_response_matrix`` result (needs ``conditions`` and ``ctx`` with the
    stimlog and its provenance). Renders live in the folder of the text log they come from: the session's own
    folder, or for a continuation session the earlier session whose log it shares (one render per log).
    Renders when no matching render exists and ``render_missing``."""
    text_log = data['ctx']['stim_provenance'].get('text_file')
    if not text_log:
        raise FileNotFoundError('no text .log for {}: display settings unknown'.format(data.get('session')))
    session_path = os.path.dirname(text_log)
    manifest = find_render(session_path, text_log, (seed,), margin_deg)
    if manifest is None:
        if not render_missing:
            raise FileNotFoundError('no effective-stimulus render for {}'.format(session_path))
        manifest = render_session(session_path, data['conditions'], data['ctx']['stimlog'], text_log,
                                  seeds=seeds_to_render or (seed,), margin_deg=margin_deg)
    key = str(seed) if manifest['background']['type'] else str(manifest['background']['seeds'][0])
    by_cond = {str(r['cond']): os.path.join(manifest['root'], r['outputs'][key]) for r in manifest['conditions']}
    return [by_cond.get(str(int(c)) if np.isscalar(c) and not isinstance(c, str) else str(c))
            for c in data['condition_ids']]
