#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDANN (Margalit et al. 2024 Neuron, https://doi.org/10.1016/j.neuron.2024.04.018) as a model-side alternative to
the DNN-SOM: a topographic deep network (ResNet-18 trained with SimCLR plus a spatial-correlation loss) whose units
each sit at a position on a simulated cortical sheet. Its VTC-like layer (layer4.1, 512 x 7 x 7 = 25,088 units on a
70 mm sheet) grows face / body / place / word patches from visual experience alone.

The repository is a reference-only submodule (marmanalysis/external/tdann); nothing from it is installed. Its demo
(demo/src/model.py) shows the checkpoint is a plain torchvision ResNet-18 trunk, so we load it into our own venv.
Main model = spatial weight alpha 0.25 (folder without an "lwx" suffix; spacetorch/analyses/alpha.py), with the
retinotopically initialized positions the demo names for SimCLR checkpoints.

Probe of one session (mirrors SOM steps 3-5):
  1. RSA: cortical RDM vs the TDANN layer4.1 RDM over the presented images (correlation distance), raw and with
     the category model partialled out, plus AlexNet relu7 on the same images and a TDANN-minus-relu7 bootstrap.
  2. SHEET MAPS: each unit's face / body / object selectivity t-statistic over our images (spacetorch tstat,
     blanks excluded), smoothed on the sheet with the paper's Gaussian (sigma 2.4 mm, 100 x 100 anchors) and
     thresholded at t = 2 for patches (paper: >= 100 mm2, >= 10 units; area here = above-threshold grid area, the
     paper uses a concave hull). The face-selectivity profile around the largest face patch shows whether the model
     predicts a sharp core or a graded patch.
  3. UNIT -> MAP ALIGNMENT (SOM step 5 analog): each ROI is assigned the unit whose response profile over the
     images best matches its tuning (som.assign_bmu_units); Mantel-style Spearman between cortical and assigned-unit
     sheet distances with a position-shuffle and a within-category-preference null (som.map_alignment).
Images: the session's effective stimuli (som.condition_image_paths), ImageNet-normalized as in TDANN training.
Weights + positions: OSF node 64qv3 (tdann_data/tdann/...), fetched with ``--fetch`` into TDANN_ROOT and verified
against the OSF hashes. The checkpoint is read with ``torch.load(weights_only=True)``: no code runs from the pickle.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/tdann_probe.py --selftest
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/tdann_probe.py --fetch
      /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python marmanalysis/tdann_probe.py --session pd
"""
import argparse
import hashlib
import json
import os
import pickletools
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import som

TDANN_ROOT = 'Margalit_etal_2024_Neuron/tdann_data'      # mirrors the OSF layout, so ST_BASE_FS could point here
FILES = {   # key: (OSF file id, path under TDANN_ROOT)
    'checkpoint': ('exugm', 'tdann/checkpoints/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_checkpoints/'
                            'model_final_checkpoint_phase199.torch'),
    'layer4.1': ('dnh7g', 'tdann/positions/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/'
                          'resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019_NBVER2/layer4.1.npz'),
}
LAYER = 'layer4.1'
SMOOTH_SIGMA_MM = 2.4
PATCH_T = 2.0
MIN_PATCH_MM2 = 100.0
MIN_PATCH_UNITS = 10
N_ANCHORS = 100
PROFILE_EDGES_MM = np.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30], float)
OUTDIR = 'output'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')


def path_of(key):
    return os.path.join(TDANN_ROOT, FILES[key][1])


# ------------------------------------------------------------------------------------------------ fetching ----
def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(force=False, retries=3):
    """Download the main checkpoint and the layer4.1 positions from OSF, verifying size and the OSF hash."""
    for key, (osf_id, rel) in FILES.items():
        dest = os.path.join(TDANN_ROOT, rel)
        meta = json.load(urllib.request.urlopen('https://api.osf.io/v2/files/%s/' % osf_id, timeout=60))['data']['attributes']
        size = int(meta['size'])
        hashes = (meta.get('extra') or {}).get('hashes') or {}
        algo = 'sha256' if hashes.get('sha256') else ('md5' if hashes.get('md5') else None)
        if os.path.exists(dest) and not force and os.path.getsize(dest) == size and (algo is None or _hash(dest, algo) == hashes[algo]):
            print('  %s already present and verified (%.1f MB): %s' % (key, size / 1e6, dest))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + '.part'
        for attempt in range(1, retries + 1):
            try:
                print('  downloading %s (%.1f MB) from https://osf.io/download/%s/ (attempt %d)' % (key, size / 1e6, osf_id, attempt))
                sys.stdout.flush()
                with urllib.request.urlopen('https://osf.io/download/%s/' % osf_id, timeout=300) as r, open(tmp, 'wb') as f:
                    for chunk in iter(lambda: r.read(1 << 20), b''):
                        f.write(chunk)
                break
            except OSError as e:
                print('    transient failure: %s' % e)
                if attempt == retries:
                    raise
                time.sleep(10 * attempt)
        if os.path.getsize(tmp) != size or (algo and _hash(tmp, algo) != hashes[algo]):
            raise RuntimeError('%s: size or %s mismatch after download (%d vs %d bytes)' % (key, algo, os.path.getsize(tmp), size))
        os.replace(tmp, dest)
        print('  verified (%s) -> %s' % (algo or 'size only', dest))
        time.sleep(2)


# --------------------------------------------------------------------------------------------- model + sheet ----
def _pickle_globals(path):
    """The classes a torch zip checkpoint's pickle refers to, read with pickletools (nothing is executed)."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith('data.pkl')][0]
        data = z.read(name)
    found, strings = set(), []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
            strings.append(arg)
        elif op.name == 'GLOBAL':
            found.add(arg.replace(' ', '.'))
        elif op.name == 'STACK_GLOBAL' and len(strings) >= 2:
            found.add('%s.%s' % (strings[-2], strings[-1]))
    return sorted(found)


def load_tdann(ckpt_path=None, device='cpu'):
    """torchvision ResNet-18 with the TDANN trunk weights (demo/src/model.py); ``ckpt_path=None`` = random weights."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Identity()
    if ckpt_path is not None:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except Exception as e:
            print('weights_only load refused (%s: %s). Classes the pickle references:' % (type(e).__name__, str(e)[:200]))
            for g in _pickle_globals(ckpt_path):
                print('   ', g)
            raise
        trunk = ckpt['classy_state_dict']['base_model']['model']['trunk']
        sd = {k.split('base_model.')[-1]: v for k, v in trunk.items() if k.startswith('base_model') and 'fc.' not in k}
        model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(device)


def load_positions(path):
    """Unit coordinates (mm) and CHW dims of one layer; coordinate i belongs to unit i of the row-major flattened
    (channel, row, column) layer output. Arrays only, no pickled objects."""
    st = np.load(path, allow_pickle=False)
    dims = tuple(int(d) for d in st['dims'])
    xy = np.asarray(st['coordinates'], float)
    if int(np.prod(dims)) != len(xy):
        raise ValueError('positions %s: dims %s do not match %d coordinates' % (path, dims, len(xy)))
    return xy, dims


def layer_responses(model, images, layer=LAYER, batch=16):
    """(n_images, n_units) responses of ``layer``, flattened in (channel, row, column) order like the positions."""
    mod = dict(model.named_modules())[layer]
    cap = []
    h = mod.register_forward_hook(lambda _m, _i, out: cap.append(out.detach().flatten(1).float().cpu().clone()))
    try:
        with torch.no_grad():
            for k in range(0, len(images), batch):
                model(images[k:k + batch])
    finally:
        h.remove()
    return torch.cat(cap).numpy()


def tstat(on, off):
    """spacetorch.utils.array_utils.tstat: Welch-style t of each unit's responses, 0 where the variance is 0."""
    m_on, m_off = np.nanmean(on, 0), np.nanmean(off, 0)
    denom = np.sqrt(np.nanstd(on, 0) ** 2 / len(on) + np.nanstd(off, 0) ** 2 / len(off))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom == 0, 0.0, (m_on - m_off) / denom)


def smooth_on_sheet(values, xy, sigma=SMOOTH_SIGMA_MM, n=N_ANCHORS):
    """Gaussian-weighted average of a unit map on an n x n anchor grid spanning the sheet (rows = y, columns = x)."""
    ax = np.linspace(xy.min(), xy.max(), n)
    out = np.zeros((n, n))
    for r in range(n):
        d2 = (xy[None, :, 0] - ax[:, None]) ** 2 + (xy[None, :, 1] - ax[r]) ** 2      # anchors (ax[c], ax[r]) x units
        w = np.exp(-d2 / (2 * sigma ** 2))
        out[r] = (w @ values) / (w.sum(1) + 1e-300)
    return out, ax


def find_patches(smoothed, ax, xy, thr=PATCH_T):
    """Connected above-threshold regions of the smoothed map that pass the paper's size criteria."""
    lab, n = ndimage.label(smoothed > thr)
    step = ax[1] - ax[0]
    cell = step ** 2
    ix = np.clip(np.round((xy[:, 0] - ax[0]) / step).astype(int), 0, len(ax) - 1)
    iy = np.clip(np.round((xy[:, 1] - ax[0]) / step).astype(int), 0, len(ax) - 1)
    unit_lab = lab[iy, ix]
    patches = []
    for k in range(1, n + 1):
        m = lab == k
        units = np.flatnonzero(unit_lab == k)
        area = float(m.sum() * cell)
        if area >= MIN_PATCH_MM2 and units.size >= MIN_PATCH_UNITS:
            r, c = np.unravel_index(np.argmax(np.where(m, smoothed, -np.inf)), m.shape)
            patches.append({'label': k, 'area_mm2': area, 'n_units': int(units.size), 'peak_xy': (ax[c], ax[r]),
                            'peak_t': float(smoothed[r, c]), 'units': units})
    return sorted(patches, key=lambda p: -p['area_mm2']), lab, unit_lab


def radial_profile(values, xy, center):
    d = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
    return np.array([np.nanmean(values[(d >= a) & (d < b)]) if ((d >= a) & (d < b)).any() else np.nan
                     for a, b in zip(PROFILE_EDGES_MM[:-1], PROFILE_EDGES_MM[1:])])


def cat_key(c):
    c = str(c).lower()
    if 'blank' in c:
        return 'blank'
    return 'face' if 'face' in c else ('body' if 'body' in c else 'object')


def roi_category(response, ck):
    """Each ROI's preferred category (largest mean response; blanks excluded), as in som_alignment.py."""
    cats = [c for c in ('face', 'body', 'object') if (ck == c).any()]
    means = np.column_stack([np.nanmean(response[:, ck == c], 1) for c in cats])
    return np.array([cats[i] for i in np.nanargmax(means, axis=1)])


# ------------------------------------------------------------------------------------------------- analysis ----
def run_session(session, n_perm=1000):
    import dnn_som
    import feature_topography_v2 as v2
    import response_matrix as rm
    if session != 'pd':
        raise NotImplementedError('only the PD session is wired so far')
    for key in FILES:
        if not os.path.exists(path_of(key)):
            raise FileNotFoundError('%s missing (%s); run with --fetch first' % (key, path_of(key)))
    data = rm.build_response_matrix(v2.SESSION, denoise=False, reliability_splits=3, neuropil_subtract=True, neucoeff=0.7)
    zeta = pd.read_csv(v2.ZETA_CSV)['p_zeta'].to_numpy() < 0.05
    d = rm.apply_roi_mask(data, zeta)
    resp, xy = d['response'], d['roi_xy_um']
    cats = data['conditions'].reindex(data['condition_ids'])['cat'].astype(str).to_numpy()
    ck = np.array([cat_key(c) for c in cats])
    paths = som.condition_image_paths(data)                       # effective stimuli
    images = dnn_som.load_images_rgb(paths, normalize=True)       # TDANN was trained on ImageNet-normalized images
    model = load_tdann(path_of('checkpoint'))
    F = layer_responses(model, images)
    sheet_xy, dims = load_positions(path_of(LAYER))
    if F.shape[1] != len(sheet_xy):
        raise ValueError('layer has %d units but %d positions' % (F.shape[1], len(sheet_xy)))
    live = F.std(0) > 0
    print('PD corrected 0.7, %d ZETA ROIs x %d images | TDANN %s %s = %d units (%d respond to these images), sheet %.0f x %.0f mm'
          % (resp.shape[0], len(paths), LAYER, dims, F.shape[1], live.sum(), np.ptp(sheet_xy[:, 0]), np.ptp(sheet_xy[:, 1])))

    # 1. RSA
    rdm_cx, rdm_t, rdm_cat = som.cortical_rdm(resp), som.build_rdm(F[:, live]), som.category_model_rdm(cats)
    alex, alex_layer = dnn_som.prep_dnn_model('cpu')
    r7 = dnn_som.extract_dnn_features(alex, alex_layer, images, 'cpu').numpy()
    rdm_7 = som.build_rdm(r7)
    print('\n[1] RSA with the cortical RDM (Spearman of RDM upper triangles, %d permutations)' % n_perm)
    for nm, rdm in (('TDANN %s' % LAYER, rdm_t), ('AlexNet relu7', rdm_7), ('category model', rdm_cat)):
        a = som.rsa(rdm_cx, rdm, n_perm=n_perm)
        line = '    %-18s r=%+.3f (p=%.3f)' % (nm, a['r'], a['p'])
        if nm != 'category model':
            b = som.rsa(rdm_cx, rdm, control=rdm_cat, n_perm=n_perm)
            line += ' | partial on category r=%+.3f (p=%.3f)' % (b['r'], b['p'])
        print(line)
    diff = som.rsa_difference_bootstrap(rdm_cx, rdm_t, rdm_7, control=rdm_cat, n_boot=n_perm)
    print('    TDANN minus relu7 (partial on category): %+.3f, 95%% CI [%+.3f, %+.3f], p(TDANN not better)=%.3f'
          % (diff['diff_mean'], diff['ci'][0], diff['ci'][1], diff['p_le0']))
    sys.stdout.flush()

    # 2. sheet maps and patches
    print('\n[2] category patches on the TDANN sheet (t > %g after %.1f mm smoothing; >= %g mm2, >= %d units)'
          % (PATCH_T, SMOOTH_SIGMA_MM, MIN_PATCH_MM2, MIN_PATCH_UNITS))
    maps = {}
    for k in ('face', 'body', 'object'):
        t = tstat(F[ck == k], F[(ck != k) & (ck != 'blank')])
        sm, ax = smooth_on_sheet(t, sheet_xy)
        patches, lab, unit_lab = find_patches(sm, ax, sheet_xy)
        maps[k] = dict(t=t, sm=sm, ax=ax, patches=patches, unit_lab=unit_lab)
        print('    %-6s units with t>2: %5d (%.1f%%) | patches: %d | areas mm2: %s'
              % (k, int((t > 2).sum()), 100 * (t > 2).mean(), len(patches), [round(p['area_mm2']) for p in patches][:8]))
    prof = None
    if maps['face']['patches']:
        top = maps['face']['patches'][0]
        prof = radial_profile(maps['face']['t'], sheet_xy, top['peak_xy'])
        print('    largest face patch: %.0f mm2, %d units, peak smoothed t %.1f at (%.1f, %.1f) mm'
              % (top['area_mm2'], top['n_units'], top['peak_t'], top['peak_xy'][0], top['peak_xy'][1]))
        print('    mean unit face t by distance from its peak (mm): %s'
              % '  '.join('%g-%g: %+.2f' % (a, b, v) for a, b, v in zip(PROFILE_EDGES_MM[:-1], PROFILE_EDGES_MM[1:], prof)))
    sys.stdout.flush()

    # 3. unit -> map alignment
    bmu, match = som.assign_bmu_units(resp, F[:, live])
    units = np.flatnonzero(live)[bmu]
    grp = roi_category(resp, ck)
    ap = som.map_alignment(xy, sheet_xy[units], n_perm=n_perm)
    ag = som.map_alignment(xy, sheet_xy[units], n_perm=n_perm, groups=grp)
    print('\n[3] unit -> map alignment (each ROI to its best-matching TDANN unit; Mantel on distances)')
    print('    match median r=%+.2f | %d distinct units for %d ROIs | align r=%+.3f p(position shuffle)=%.3f p(within category)=%.3f'
          % (float(np.median(match)), len(np.unique(units)), len(units), ap['r'], ap['p'], ag['p']))
    in_face = maps['face']['unit_lab'][units] > 0
    for g in ('face', 'body', 'object'):
        m = grp == g
        if m.any():
            print('    ROIs preferring %-6s n=%3d: assigned unit inside a face-selective region %.0f%%' % (g, m.sum(), 100 * in_face[m].mean()))

    # figure
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13, 12))
    f = maps['face']
    sc = axs[0, 0].scatter(sheet_xy[:, 0], sheet_xy[:, 1], c=f['t'], cmap='coolwarm', vmin=-4, vmax=4, s=1, rasterized=True)
    plt.colorbar(sc, ax=axs[0, 0], fraction=0.04, label='face t (unit)')
    axs[0, 0].set_title('TDANN %s units on the sheet: face selectivity for our images' % LAYER, fontsize=9)
    ext = (f['ax'][0], f['ax'][-1], f['ax'][0], f['ax'][-1])
    im = axs[0, 1].imshow(f['sm'], origin='lower', extent=ext, cmap='coolwarm', vmin=-4, vmax=4)
    plt.colorbar(im, ax=axs[0, 1], fraction=0.04, label='smoothed face t')
    for k, col in (('face', 'darkred'), ('body', 'darkgreen'), ('object', 'navy')):
        axs[0, 1].contour(maps[k]['ax'], maps[k]['ax'], maps[k]['sm'], levels=[PATCH_T], colors=[col], linewidths=1.2)
    axs[0, 1].set_title('smoothed face map (sigma %.1f mm); t = %g contours: face red, body green, object blue'
                        % (SMOOTH_SIGMA_MM, PATCH_T), fontsize=9)
    if prof is not None:
        ctr = 0.5 * (PROFILE_EDGES_MM[1:] + PROFILE_EDGES_MM[:-1])
        axs[1, 0].plot(ctr, prof, '-o', color='tab:red')
        axs[1, 0].axhline(0, color='k', lw=0.5)
        axs[1, 0].set_xlabel('distance from the largest face patch peak (mm)')
        axs[1, 0].set_ylabel('mean unit face t')
        axs[1, 0].set_title('model face patch: core-to-surround profile', fontsize=9)
    colors = {'face': 'tab:red', 'body': 'tab:green', 'object': 'tab:blue'}
    axs[1, 1].contour(f['ax'], f['ax'], f['sm'], levels=[PATCH_T], colors=['darkred'], linewidths=1)
    for g, col in colors.items():
        m = grp == g
        axs[1, 1].scatter(sheet_xy[units[m], 0], sheet_xy[units[m], 1], s=10, color=col, alpha=0.6, label='ROIs preferring %s' % g)
    axs[1, 1].legend(fontsize=7)
    axs[1, 1].set_title('where the %d PD ROIs land: their best-matching units (Mantel r=%+.3f)' % (len(units), ap['r']), fontsize=9)
    for a in axs.ravel():
        a.set_aspect('equal' if a is not axs[1, 0] else 'auto')
    fig.suptitle('TDANN (Margalit et al. 2024, alpha 0.25 SimCLR) probed with the PD stimuli (effective, ImageNet-normalized)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'tdann_probe_%s_%s.png' % (session, STAMP))
    fig.savefig(out, dpi=120)
    print('\nfigure: %s' % out)


# ------------------------------------------------------------------------------------------------ self-test ----
def selftest():
    """No download needed: random-weight network, synthetic sheet, synthetic tuning."""
    rng = np.random.default_rng(0)
    model = load_tdann(None)
    x = torch.randn(6, 3, 224, 224)
    F = layer_responses(model, x, batch=4)
    assert F.shape == (6, 512 * 7 * 7), F.shape
    cap = {}
    h = model.layer4[1].register_forward_hook(lambda _m, _i, out: cap.setdefault('t', out.detach().clone()))
    with torch.no_grad():
        model(x[:1])
    h.remove()
    assert np.isclose(F[0, 3 * 49 + 2 * 7 + 5], float(cap['t'][0, 3, 2, 5]), atol=1e-5), 'flatten order differs from (C, H, W)'
    print('  layer4.1 responses %s; unit (c=3, row=2, col=5) sits at flat index %d as the positions assume'
          % (F.shape, 3 * 49 + 2 * 7 + 5))
    xy = rng.uniform(0, 70, (25088, 2))
    dc = np.hypot(xy[:, 0] - 20, xy[:, 1] - 30)
    vals = 6 * np.exp(-dc ** 2 / (2 * 6 ** 2)) + rng.normal(0, 0.5, len(xy))
    sm, ax = smooth_on_sheet(vals, xy)
    patches, _, _ = find_patches(sm, ax, xy)
    print('  synthetic bump (peak 6, sigma 6 mm at (20, 30)): %d patch, %.0f mm2 (expected ~250), peak at (%.1f, %.1f)'
          % (len(patches), patches[0]['area_mm2'] if patches else 0, *(patches[0]['peak_xy'] if patches else (np.nan, np.nan))))
    assert len(patches) == 1 and 150 < patches[0]['area_mm2'] < 400
    n_cond, n_roi = 40, 300
    U = rng.standard_normal((n_cond, 2000))
    uxy = rng.uniform(0, 70, (2000, 2))
    pick = rng.choice(2000, n_roi, replace=False)
    resp = U[:, pick].T + 0.3 * rng.standard_normal((n_roi, n_cond))
    cortex_xy = uxy[pick] * 10 + rng.normal(0, 1, (n_roi, 2))
    bmu, match = som.assign_bmu_units(resp, U)
    al = som.map_alignment(cortex_xy, uxy[bmu], n_perm=200)
    print('  synthetic alignment: %.0f%% of ROIs matched to their source unit, Mantel r=%+.3f p=%.3f'
          % (100 * np.mean(bmu == pick), al['r'], al['p']))
    assert np.mean(bmu == pick) > 0.9 and al['r'] > 0.8 and al['p'] < 0.01
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--fetch', action='store_true', help='download + verify the checkpoint and layer4.1 positions (173 MB)')
    ap.add_argument('--session', default=None, choices=['pd'])
    ap.add_argument('--n-perm', type=int, default=1000)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.fetch:
        fetch()
    if a.session:
        run_session(a.session, a.n_perm)


if __name__ == '__main__':
    main()
