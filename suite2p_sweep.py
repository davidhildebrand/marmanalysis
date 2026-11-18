#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection-only suite2p variants on ONE registered movie (roadmap item 1, the suite2p settings sweep).

Registration is the expensive step (~1 h for the 17,000-frame PD session) and is also the step whose result every
variant should share, so the sweep is "register once, detect many": an ANCHOR extraction made by
process_with_suite2p.py with --keep-bin provides plane0/data.bin (the registered movie), plane0/ops.npy (the
registration outputs: yrange / xrange / badframes / mean image / shifts) and ops_in.npy (the settings that were asked
for). Every variant here re-runs ONLY cell detection, trace extraction, deconvolution and classification on that
binary, with its own detection / neuropil settings, into its own folder

    <out-root>/<variant name>/plane0/{stat,F,Fneu,spks,iscell,ops}.npy  +  <out-root>/<variant name>/variant.json

The default out-root is <session>/sweep_suite2p/. Its folders deliberately do NOT match the 'suite2p*' glob that
sessionio.find_suite2p_dir uses to pick a session's default extraction, so a sweep can never silently change which
extraction the analyses load; a variant is addressed explicitly, e.g.
    sessionio.build_session_response_table(session, variant='sweep_suite2p/<variant name>')

Two suite2p generations are supported, chosen by the interpreter this script runs under:
  * suite2p 0.14.x + cellpose 3 (venv_s2p014, Python 3.10): the flat-ops API, cellpose models 'cyto' / 'cyto2' /
    'cyto3' / 'nuclei', anatomical_only 1-4 as in the preprint (v9) pipeline -- the PARITY generation;
  * suite2p 1.x + cellpose 4 (.venv): the nested-settings API and the Cellpose-SAM model ('cpsam'); anatomical_only 3
    (enhanced mean image) no longer exists there and is mapped to the mean image plus the spatial high-pass.
The binary is only read (0.14 opens it read/write because that is all its BinaryFile offers; nothing writes to it when
registration is skipped). The anchor folder itself is never modified.

Run (parity generation):
  venv_s2p014/bin/python marmanalysis/suite2p_sweep.py <anchor dir> \\
        --cellprob -6 --flow 0 --diameter 14 --anatomical-only 3 --max-overlap 0.75 --spatial-hp-cp 0.5
  ... or several variants from a plan file (a JSON list of override dicts, keys as below, optional "name"):
  venv_s2p014/bin/python marmanalysis/suite2p_sweep.py <anchor dir> --plan tmp/suite2p_sweep_plan.json
Run (Cellpose-SAM):
  .venv/bin/python marmanalysis/suite2p_sweep.py <anchor dir> --model cpsam --cellprob 0 --flow 0.4
Override keys (plan file / CLI): anatomical_only, diameter, cellprob_threshold, flow_threshold, spatial_hp_cp,
pretrained_model, max_overlap, threshold_scaling, sparse_mode, min_neuropil_pixels, inner_neuropil_radius,
allow_overlap, lam_percentile, neucoeff, tau. Anything not overridden is taken from the anchor's ops_in.npy.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

# suite2p 1.x's OASIS (numba, parallel) segfaults on this Intel Mac with numba's default / OpenMP threading layer once torch
# has loaded its own OpenMP runtime (smoke test 2026-11-17: crash inside "SPIKE DECONVOLUTION", leaked semaphore). The
# workqueue layer is safe and the deconvolution is a few seconds of work, so it is selected before numba is imported.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
# Only matters for --torch-device mps: torch 2.2.2 lacks a few ops on Metal (Cellpose-SAM's relative-position
# interpolation, aten::upsample_linear1d); with the fallback those ops run on the CPU instead of raising.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import suite2p  # noqa: E402

S2P_VERSION = str(getattr(suite2p, 'version', None) or getattr(suite2p, '__version__', '0'))
V1 = int(S2P_VERSION.split('.')[0]) >= 1

DETECTION_KEYS = ('anatomical_only', 'diameter', 'cellprob_threshold', 'flow_threshold', 'spatial_hp_cp',
                  'pretrained_model', 'max_overlap', 'threshold_scaling', 'sparse_mode')
EXTRACTION_KEYS = ('min_neuropil_pixels', 'inner_neuropil_radius', 'allow_overlap', 'lam_percentile', 'neucoeff')
OTHER_KEYS = ('tau',)
OVERRIDE_KEYS = DETECTION_KEYS + EXTRACTION_KEYS + OTHER_KEYS
# suite2p 0.14 defaults for keys the anchor's ops_in may not carry (process_with_suite2p.py leaves them to suite2p).
FALLBACK = {'spatial_hp_cp': 0, 'pretrained_model': 'cyto', 'threshold_scaling': 1.0, 'sparse_mode': True,
            'min_neuropil_pixels': 350, 'inner_neuropil_radius': 2, 'allow_overlap': False, 'lam_percentile': 50.0,
            'neucoeff': 0.7, 'max_overlap': 0.75, 'anatomical_only': 0, 'diameter': 0, 'cellprob_threshold': 0.0,
            'flow_threshold': 0.4, 'tau': 1.0}
# Registration outputs carried from the anchor into every variant (suite2p 1.x reads them from reg_outputs.npy;
# session_qc.py reads the shifts from ops.npy).
REG_KEYS = ('yrange', 'xrange', 'badframes', 'meanImg', 'meanImgE', 'refImg', 'xoff', 'yoff', 'corrXY', 'xoff1', 'yoff1',
            'corrXY1', 'xblock', 'yblock', 'nblocks', 'NRsm', 'regDX', 'regPC', 'tPC', 'bidiphase', 'Ly', 'Lx', 'nframes',
            'fs', 'nchannels', 'nplanes', 'functional_chan', 'batch_size')
V1_IMG = {1: 'max_proj / meanImg', 2: 'meanImg', 3: 'meanImg', 4: 'max_proj'}


def _g(v):
    """Compact number formatting for folder names: 14.0 -> '14', -3.5 -> '-3p5', 0.75 -> '0p75'."""
    if isinstance(v, (bool, np.bool_)):
        return '1' if v else '0'
    return ('%g' % float(v)).replace('.', 'p')


def load_anchor(anchor):
    plane = os.path.join(anchor, 'plane0')
    ops = np.load(os.path.join(plane, 'ops.npy'), allow_pickle=True).item()
    reg_file = os.path.join(plane, 'data.bin')
    if not os.path.isfile(reg_file):
        reg_file = ops.get('reg_file', '')
    if not os.path.isfile(reg_file):
        raise FileNotFoundError('no registered binary for %s (plane0/data.bin missing; was the anchor run with --keep-bin?)'
                                % anchor)
    ops_in_path = os.path.join(anchor, 'ops_in.npy')
    if os.path.isfile(ops_in_path):
        asked = np.load(ops_in_path, allow_pickle=True).item()
    else:
        print('WARNING: %s has no ops_in.npy; taking the base settings from plane0/ops.npy, whose "diameter" is the '
              'MEDIAN DETECTED diameter, not the requested one -- pass --diameter explicitly.' % anchor)
        asked = ops
    base = {}
    for k in OVERRIDE_KEYS:
        base[k] = asked[k] if k in asked else ops.get(k, FALLBACK[k])
    if isinstance(base['diameter'], (list, tuple, np.ndarray)):
        base['diameter'] = float(np.asarray(base['diameter']).ravel()[-1])
    return ops, reg_file, base


def effective_settings(base, over):
    eff = dict(base)
    for k, v in over.items():
        if k not in OVERRIDE_KEYS:
            raise KeyError('unknown override %r (allowed: %s)' % (k, ', '.join(OVERRIDE_KEYS)))
        if v is not None:
            eff[k] = v
    if V1 and int(eff['anatomical_only']) > 0 and eff['pretrained_model'] in ('cyto', 'cyto2', 'cyto3', 'nuclei'):
        print('NOTE: suite2p %s / cellpose 4 only ship the Cellpose-SAM model; using pretrained_model="cpsam" instead of %r'
              % (S2P_VERSION, eff['pretrained_model']))
        eff['pretrained_model'] = 'cpsam'
    return eff


def variant_name(eff, base):
    ao = int(eff['anatomical_only'])
    if ao > 0:
        name = 'cp%d_%s_d%s_pt%s_ft%s_mo%s_hp%s' % (ao, eff['pretrained_model'], _g(eff['diameter']), _g(eff['cellprob_threshold']),
                                                   _g(eff['flow_threshold']), _g(eff['max_overlap']), _g(eff['spatial_hp_cp']))
    else:
        name = 'func_ts%s_mo%s' % (_g(eff['threshold_scaling']), _g(eff['max_overlap']))
    if eff['min_neuropil_pixels'] != base['min_neuropil_pixels']:
        name += '_np%s' % _g(eff['min_neuropil_pixels'])
    if eff['inner_neuropil_radius'] != base['inner_neuropil_radius']:
        name += '_in%s' % _g(eff['inner_neuropil_radius'])
    if bool(eff['allow_overlap']) != bool(base['allow_overlap']):
        name += '_olap%s' % _g(eff['allow_overlap'])
    if eff['lam_percentile'] != base['lam_percentile']:
        name += '_lp%s' % _g(eff['lam_percentile'])
    if eff['neucoeff'] != base['neucoeff']:
        name += '_nc%s' % _g(eff['neucoeff'])
    if abs(float(eff['tau']) - float(base['tau'])) > 1e-9:
        name += '_tau%s' % _g(eff['tau'])
    return name + ('_s2p%s' % S2P_VERSION.replace('.', ''))


def _prune_detection_outputs(ops):
    """Drop the anchor's own detection products so a variant's ops.npy carries only its own."""
    for k in ('Vcorr', 'Vmap', 'Vmax', 'Vsplit', 'ihop', 'max_proj', 'spatscale_pix', 'timing', 'meanImg_crop'):
        ops.pop(k, None)
    return ops


def enhanced_mean_image(ops, diameter):
    """Recompute suite2p 0.14's enhanced mean image (the anatomical_only=3 detection input) for `diameter`.

    A full run computes ops['meanImgE'] during REGISTRATION with that run's diameter (median-filter window
    4*ceil(d)+1 px), and detection then uses it as-is: anatomical.select_rois never recomputes it. A detection-only
    variant that changes the diameter must therefore redo it here to match a full run. The function is deterministic,
    so for the anchor's own diameter it reproduces the anchor's meanImgE bit for bit (checked on the PD anchor,
    2026-11-17). suite2p 0.14.x indexes the diameter as a pair unless it is a Python int, hence the coercion.
    """
    from suite2p.registration.register import compute_enhanced_mean_image
    d = float(diameter)
    tmp = {'meanImg': np.asarray(ops['meanImg'], dtype=np.float32),
           'diameter': int(d) if d == int(d) else [d, d]}          # no spatscale_pix / aspect -> derived from diameter
    return compute_enhanced_mean_image(tmp['meanImg'], tmp)


# ----------------------------------------------------------------------------------------------------------------
def run_v014(anchor_ops, reg_file, out_plane, eff):
    """suite2p 0.14.x: flat ops; run_s2p.pipeline() with registration skipped saves stat / F / Fneu / spks / iscell
    and ops.npy into ops['save_path'] itself."""
    from suite2p.io import BinaryFile
    from suite2p.run_s2p import pipeline
    ops = _prune_detection_outputs(dict(anchor_ops))
    ops.update(eff)
    ops.update({'save_path': out_plane, 'ops_path': os.path.join(out_plane, 'ops.npy'), 'reg_file': reg_file,
                'do_registration': False, 'roidetect': True, 'spikedetect': True, 'delete_bin': False, 'move_bin': False,
                'reg_tif': False, 'save_mat': False, 'save_NWB': False, 'preclassify': 0.0})
    ops['diameter_set'] = eff['diameter']                 # select_rois overwrites 'diameter' with the detected median
    if int(eff.get('anatomical_only', 0)) == 3 and 'meanImg' in ops:
        # Detection reads meanImgE as-is, so it must correspond to THIS variant's diameter (see enhanced_mean_image).
        ops['meanImgE'] = enhanced_mean_image(ops, eff['diameter'])
        ops['meanImgE_diameter'] = eff['diameter']
    with BinaryFile(Ly=int(ops['Ly']), Lx=int(ops['Lx']), filename=reg_file) as f_reg:
        ops = pipeline(f_reg, run_registration=False, ops=ops)
    return ops


def extract_traces_numpy(f_in, cell_masks, neuropil_masks, batch_size=500, device=None):
    """Drop-in replacement for suite2p 1.x ``extraction.extract.extract_traces`` without torch: the dense-batch times
    sparse-CSC products are done with scipy.sparse. The torch version segfaulted on this machine (torch 2.2.2, Intel
    macOS, CPU) during the smoke test on a 128 x 128 synthetic movie; the arithmetic is identical (ROI trace = lam-
    weighted pixel sum, neuropil trace = mean over the ring pixels)."""
    from scipy.sparse import csc_matrix
    n_frames, Ly, Lx = f_in.shape
    ncells = len(cell_masks)
    rows = np.concatenate([np.asarray(cm[0], np.int64) for cm in cell_masks])
    cols = np.concatenate([np.full(len(cm[0]), k, np.int64) for k, cm in enumerate(cell_masks)])
    lam = np.concatenate([np.asarray(cm[1], np.float32) for cm in cell_masks])
    cmasks = csc_matrix((lam, (rows, cols)), shape=(Ly * Lx, ncells), dtype=np.float32)
    if neuropil_masks is not None:
        nrows = np.concatenate([np.asarray(nm, np.int64) for nm in neuropil_masks])
        ncols = np.concatenate([np.full(len(nm), k, np.int64) for k, nm in enumerate(neuropil_masks)])
        npix = np.array([max(len(nm), 1) for nm in neuropil_masks], np.float32)
        nmasks = csc_matrix((np.ones(len(nrows), np.float32), (nrows, ncols)), shape=(Ly * Lx, ncells), dtype=np.float32)
    F = np.zeros((ncells, n_frames), np.float32)
    Fneu = np.zeros((ncells, n_frames), np.float32)
    batch_size = int(min(batch_size, 1000))
    for t0 in range(0, n_frames, batch_size):
        t1 = min(t0 + batch_size, n_frames)
        data = np.asarray(f_in[t0:t1]).reshape(t1 - t0, Ly * Lx).astype(np.float32)
        F[:, t0:t1] = (cmasks.T @ data.T)
        if neuropil_masks is not None:
            Fneu[:, t0:t1] = (nmasks.T @ data.T) / npix[:, None]
    return F, Fneu


def run_v1(anchor_ops, reg_file, out_plane, eff, device, torch_extract=False, cellpose_bf16=False):
    """suite2p 1.x: nested settings; pipeline_s2p.pipeline() on a read-only BinaryFile, registration outputs supplied
    through reg_outputs.npy in the variant folder; ops.npy written here in the flat form the analyses read."""
    import torch
    from suite2p import default_settings, pipeline
    from suite2p.io import BinaryFile
    from suite2p.run_s2p import logger_setup
    logger_setup(None)                                        # suite2p 1.x logs through `logging`; show its progress lines
    if torch.get_num_threads() < (os.cpu_count() or 1):
        # torch comes up with ONE intra-op thread here (MKL_NUM_THREADS=1 inside the process); 16 threads make the CPU
        # Cellpose-SAM pass ~5x faster (tmp/torch_threads_test.py).
        torch.set_num_threads(int(os.environ.get('SWEEP_TORCH_THREADS', os.cpu_count() or 1)))
    print('torch threads %d, device %s' % (torch.get_num_threads(), device))
    if not torch_extract:
        import suite2p.extraction.extract as s2p_extract
        s2p_extract.extract_traces = extract_traces_numpy        # extraction_wrapper resolves the name at call time
    try:
        import functools
        from cellpose import core as cp_core
        import suite2p.detection.anatomical as s2p_anat
        if not cellpose_bf16:
            # cellpose 4 loads Cellpose-SAM in bfloat16 by default (CellposeModel(use_bfloat16=True)); x86 CPUs emulate
            # bf16 (the synthetic 120 x 120 smoke test took 23 min) and Metal (MPS) rejects it outright. Build the model
            # in float32 instead; suite2p's roi_detect creates it as CellposeModel(pretrained_model=..., gpu=...).
            s2p_anat.CellposeModel = functools.partial(s2p_anat.CellposeModel, use_bfloat16=False)
        if device == 'cpu':
            # suite2p 1.x asks cellpose for gpu=core.use_gpu(), which is True here because torch reports the Metal (MPS)
            # backend; honor --torch-device cpu by keeping cellpose on the CPU too.
            cp_core.use_gpu = lambda *a, **k: False
    except Exception as exc:                                      # cellpose missing: suite2p falls back to sparsery itself
        print('NOTE: could not configure cellpose (%s)' % exc)
    s = default_settings()
    s['torch_device'] = device
    s['tau'], s['fs'] = float(eff['tau']), float(anchor_ops['fs'])
    d = float(eff['diameter'])
    s['diameter'] = [d, d] if d > 0 else [30.0, 30.0]      # cpsam rescales the image to a 30-px cell; 0 = its native scale
    s['run'].update(do_registration=0, do_regmetrics=False, do_detection=True, do_deconvolution=True)
    s['io'].update(delete_bin=False, move_bin=False, save_ops_orig=False, save_mat=False, save_NWB=False)
    det = s['detection']
    ao = int(eff['anatomical_only'])
    det['algorithm'] = 'cellpose' if ao > 0 else 'sparsery'
    det['max_overlap'] = float(eff['max_overlap'])
    det['threshold_scaling'] = float(eff['threshold_scaling'])
    det['nbins'] = int(anchor_ops.get('nbinned', 5000))
    det['highpass_time'] = int(anchor_ops.get('high_pass', 100))
    det['denoise'] = bool(anchor_ops.get('denoise', False))
    det['soma_crop'] = bool(anchor_ops.get('soma_crop', True))
    det['sparsery_settings']['highpass_neuropil'] = int(anchor_ops.get('spatial_hp_detect', 25))
    det['sparsery_settings']['spatial_scale'] = int(anchor_ops.get('spatial_scale', 0))
    cp = det['cellpose_settings']
    cp['cellpose_model'] = eff['pretrained_model']
    cp['img'] = V1_IMG[ao] if ao > 0 else cp['img']
    cp['highpass_spatial'] = float(eff['spatial_hp_cp'])
    cp['flow_threshold'] = float(eff['flow_threshold'])
    cp['cellprob_threshold'] = float(eff['cellprob_threshold'])
    s['extraction'].update(neuropil_coefficient=float(eff['neucoeff']), inner_neuropil_radius=int(eff['inner_neuropil_radius']),
                           min_neuropil_pixels=int(eff['min_neuropil_pixels']), lam_percentile=float(eff['lam_percentile']),
                           allow_overlap=bool(eff['allow_overlap']), snr_threshold=0.0)
    for k in ('baseline', 'win_baseline', 'sig_baseline', 'prctile_baseline'):
        if k in anchor_ops:
            s['dcnv_preprocess'][k] = anchor_ops[k]
    reg = {k: anchor_ops[k] for k in REG_KEYS if k in anchor_ops}
    reg['badframes'] = np.asarray(anchor_ops['badframes'], bool)
    np.save(os.path.join(out_plane, 'reg_outputs.npy'), reg)
    np.save(os.path.join(out_plane, 'settings.npy'), s)
    with BinaryFile(Ly=int(anchor_ops['Ly']), Lx=int(anchor_ops['Lx']), filename=reg_file, write=False) as f_reg:
        outputs = pipeline(out_plane, f_reg, run_registration=False, settings=s, device=torch.device(device))
    detect_outputs = outputs[1] or {}
    stat = outputs[2]
    if stat is not None and len(stat) and 'neuropil_mask' not in stat[0]:
        # suite2p 1.x keeps the neuropil ring pixels out of stat; 0.14 stored them as stat['neuropil_mask'], which the
        # ring-geometry read-outs (session_qc.neuropil_ring_metrics) rely on -> rebuild them with the run's settings.
        from suite2p.extraction.masks import create_masks
        ex = s['extraction']
        _, neuropil_masks = create_masks(stat, int(anchor_ops['Ly']), int(anchor_ops['Lx']), lam_percentile=ex['lam_percentile'],
                                         allow_overlap=ex['allow_overlap'], neuropil_extract=ex['neuropil_extract'],
                                         inner_neuropil_radius=ex['inner_neuropil_radius'],
                                         min_neuropil_pixels=ex['min_neuropil_pixels'], circular_neuropil=ex['circular_neuropil'])
        for st, nm in zip(stat, neuropil_masks):
            st['neuropil_mask'] = np.asarray(nm)
        np.save(os.path.join(out_plane, 'stat.npy'), stat)
    ops = _prune_detection_outputs(dict(anchor_ops))
    ops.update(eff)
    ops.update({k: v for k, v in detect_outputs.items() if k != 'diameter'})
    ops.update({'diameter_set': d, 'save_path': out_plane, 'ops_path': os.path.join(out_plane, 'ops.npy'), 'reg_file': reg_file,
                'do_registration': False, 'delete_bin': False, 'suite2p_version': S2P_VERSION, 'settings_v1': s,
                'cellpose_img': cp['img'], 'anatomical_only_note': ('suite2p 1.x has no enhanced mean image; '
                                                                     'anatomical_only 3 ran on the mean image' if ao == 3 else '')})
    ops['plane_times'] = outputs[-1]
    np.save(ops['ops_path'], ops)
    return ops


def run_variant(anchor, out_root, over, device='cpu', force=False, dry_run=False, name=None, torch_extract=False,
                cellpose_bf16=False):
    anchor_ops, reg_file, base = load_anchor(anchor)
    eff = effective_settings(base, over)
    name = name or variant_name(eff, base)
    out = os.path.join(out_root, name)
    out_plane = os.path.join(out, 'plane0')
    changed = {k: eff[k] for k in OVERRIDE_KEYS if eff[k] != base[k]}
    print('== variant %s' % name)
    print('   anchor %s' % anchor)
    print('   binary %s (%d frames, %d x %d)' % (reg_file, anchor_ops['nframes'], anchor_ops['Ly'], anchor_ops['Lx']))
    print('   changed vs anchor: %s' % (json.dumps(changed, default=str) if changed else '(none: a re-detection with the anchor settings)'))
    if dry_run:
        return None
    if os.path.isfile(os.path.join(out_plane, 'iscell.npy')) and not force:
        iscell = np.load(os.path.join(out_plane, 'iscell.npy'))
        print('   exists: %d ROIs (%d accepted); use --force to redo' % (len(iscell), int(iscell[:, 0].sum())))
        return out
    os.makedirs(out_plane, exist_ok=True)
    t0 = time.time()
    try:
        import cellpose
        cellpose_version = str(getattr(cellpose, 'version', None) or getattr(cellpose, '__version__', '?'))
    except Exception:
        cellpose_version = None
    ops = (run_v1(anchor_ops, reg_file, out_plane, eff, device, torch_extract=torch_extract, cellpose_bf16=cellpose_bf16) if V1
           else run_v014(anchor_ops, reg_file, out_plane, eff))
    iscell_path = os.path.join(out_plane, 'iscell.npy')
    n_roi, n_acc = (0, 0)
    if os.path.isfile(iscell_path):
        iscell = np.load(iscell_path)
        n_roi, n_acc = int(len(iscell)), int(iscell[:, 0].sum())
    info = {'name': name, 'anchor': os.path.abspath(anchor), 'reg_file': os.path.abspath(reg_file), 'out': os.path.abspath(out),
            'suite2p_version': S2P_VERSION, 'cellpose_version': cellpose_version, 'python': platform.python_version(),
            'torch_device': device if V1 else 'n/a', 'cellpose_bf16': bool(cellpose_bf16) if V1 else None,
            'settings_effective': {k: eff[k] for k in OVERRIDE_KEYS},
            'settings_changed_vs_anchor': changed, 'anchor_base': base, 'n_rois': n_roi, 'n_accepted': n_acc,
            'detected_median_diameter_px': float(np.asarray(ops.get('diameter', np.nan)).ravel()[-1]) if ops is not None else None,
            'runtime_s': round(time.time() - t0, 1), 'finished_utc': datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')}
    with open(os.path.join(out, 'variant.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print('   done: %d ROIs (%d accepted) in %.0f s -> %s' % (n_roi, n_acc, info['runtime_s'], out))
    return out


def list_variants(out_root):
    rows = []
    for d in sorted(os.listdir(out_root)) if os.path.isdir(out_root) else []:
        vj = os.path.join(out_root, d, 'variant.json')
        if os.path.isfile(vj):
            with open(vj) as f:
                info = json.load(f)
            rows.append((d, info.get('suite2p_version'), info.get('n_rois'), info.get('n_accepted'), info.get('runtime_s'),
                         json.dumps(info.get('settings_changed_vs_anchor', {}), default=str)))
        elif os.path.isdir(os.path.join(out_root, d, 'plane0')):
            rows.append((d, '?', '?', '?', '?', '(incomplete: no variant.json)'))
    print('%-60s %-8s %6s %6s %7s  %s' % ('variant', 's2p', 'rois', 'acc', 'time_s', 'changed vs anchor'))
    for r in rows:
        print('%-60s %-8s %6s %6s %7s  %s' % r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('anchor', help='anchor extraction folder (holds plane0/data.bin, plane0/ops.npy, ops_in.npy)')
    ap.add_argument('--out-root', default=None, help='where variant folders go [default: <session>/sweep_suite2p]')
    ap.add_argument('--name', default=None, help='variant folder name [default: built from the effective settings]')
    ap.add_argument('--plan', default=None, help='JSON list of override dicts (keys: %s; optional "name")' % ', '.join(OVERRIDE_KEYS))
    ap.add_argument('--anatomical-only', dest='anatomical_only', type=int, default=None)
    ap.add_argument('--diameter', type=float, default=None)
    ap.add_argument('--cellprob', dest='cellprob_threshold', type=float, default=None)
    ap.add_argument('--flow', dest='flow_threshold', type=float, default=None)
    ap.add_argument('--spatial-hp-cp', dest='spatial_hp_cp', type=float, default=None)
    ap.add_argument('--model', dest='pretrained_model', default=None, help="cellpose model: cyto / cyto2 / cyto3 / nuclei (0.14) or cpsam (1.x)")
    ap.add_argument('--max-overlap', dest='max_overlap', type=float, default=None)
    ap.add_argument('--threshold-scaling', dest='threshold_scaling', type=float, default=None)
    ap.add_argument('--min-neuropil-pixels', dest='min_neuropil_pixels', type=int, default=None)
    ap.add_argument('--inner-neuropil-radius', dest='inner_neuropil_radius', type=int, default=None)
    ap.add_argument('--allow-overlap', dest='allow_overlap', action='store_const', const=True, default=None)
    ap.add_argument('--lam-percentile', dest='lam_percentile', type=float, default=None)
    ap.add_argument('--neucoeff', type=float, default=None)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--torch-device', default='cpu',
                    help='suite2p 1.x only: cpu / mps / cuda [default: cpu]. On this Mac mps fails inside cellpose '
                         '(dynamics.flow_error IndexError after the MPS fallback), so cpu + float32 is the working route.')
    ap.add_argument('--torch-extract', action='store_true',
                    help='suite2p 1.x only: keep its torch sparse trace extraction instead of the scipy replacement '
                         '(the torch version segfaults on this Intel Mac)')
    ap.add_argument('--cellpose-bf16', action='store_true',
                    help='suite2p 1.x only: keep cellpose 4\'s default bfloat16 weights (very slow on x86 CPUs, '
                         'unsupported on MPS) instead of float32')
    ap.add_argument('--force', action='store_true', help='redo a variant whose outputs exist')
    ap.add_argument('--dry-run', action='store_true', help='print the effective settings and folder name, run nothing')
    ap.add_argument('--list', action='store_true', help='list the variants already in --out-root and exit')
    a = ap.parse_args()

    anchor = os.path.normpath(a.anchor)
    out_root = a.out_root or os.path.join(os.path.dirname(anchor), 'sweep_suite2p')
    print('suite2p %s (%s API), python %s' % (S2P_VERSION, '1.x nested-settings' if V1 else '0.14 flat-ops', platform.python_version()))
    if a.list:
        list_variants(out_root)
        return
    if a.plan:
        with open(a.plan) as f:
            plan = json.load(f)
        if not isinstance(plan, list):
            raise ValueError('--plan must be a JSON list of override dicts')
        print('plan %s: %d variants -> %s' % (a.plan, len(plan), out_root))
        for i, over in enumerate(plan):
            over = dict(over)
            name = over.pop('name', None)
            print('--- [%d/%d]' % (i + 1, len(plan)))
            try:
                run_variant(anchor, out_root, over, device=a.torch_device, force=a.force, dry_run=a.dry_run, name=name,
                            torch_extract=a.torch_extract, cellpose_bf16=a.cellpose_bf16)
            except Exception as exc:
                print('   FAILED: %s: %s' % (type(exc).__name__, exc))
            sys.stdout.flush()
        if not a.dry_run:
            list_variants(out_root)
        return
    over = {k: getattr(a, k, None) for k in OVERRIDE_KEYS if getattr(a, k, None) is not None}   # sparse_mode: plan file only
    run_variant(anchor, out_root, over, device=a.torch_device, force=a.force, dry_run=a.dry_run, name=a.name,
                torch_extract=a.torch_extract, cellpose_bf16=a.cellpose_bf16)


if __name__ == '__main__':
    main()
