#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-paradigm driver helpers on top of the shared loaders (sessionio) and the labeled
response container (response_table).

Factors the image-specific logic out of analysis_for_images.py into reusable functions:
  * classify_image            -- stimulus image basename -> (cond, cat, identity, pitch, yaw, roll)
  * build_condition_metadata  -- per-condition classification table from a stimlog
  * attach_condition_metadata -- put that metadata onto an xarray response Dataset as coords
  * supercategory_bools       -- face / non-face / non-face-object masks over conditions
  * face_dprime_fsi           -- face d' (Vinken 2023) and FSI (Freiwald 2010) per metric
  * responsive_anova          -- per-ROI one-way ANOVA across conditions (responsiveness)

d'/FSI reuse response_table.face_dprime / face_selectivity_index (already parity-tested).
"""

import os
import re
from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

import response_table
import sessionio


# Plot/template ordering for categories (analysis_for_images.py:74-77).
TEMPLATE = np.array([b'blank', b'scram_s', b'scram_p',
                     b'face_mrm', b'face_rhe', b'face_hum', b'face_ctn',
                     b'obj', b'food', b'body_mrm', b'animal'], dtype='|S8')

# Marmoset head pose (pitch, yaw, roll) by view index (analysis_for_images.py:1457-1493).
_MARM_HEAD_POSE = {
    1: (0, 0, 0), 2: (0, 180, 0), 3: (0, 0, -45), 4: (0, 0, 45), 5: (0, -90, 0),
    6: (0, -45, 0), 7: (0, 45, 0), 8: (0, 90, 0), 9: (0, 0, 180),
}

_PATTERN_FREI = (r'^(Freiwald(FOB)?([0-9]*)?)?_?([^_]+)_([^_]+)_?([^_]+)?_([0-9]+)_?'
                 r'([^_]*erode[^_]*)?_?(inverted)?$')
_PATTERN_SONG = (r'^(Song_(etal_Wang_2022_NatCommun)?(_selected20230509d)?)?_?'
                 r'([aobmufps]{1})([0-9]{1,2})$')


def classify_image(imn):
    """Classify a stimulus image basename (no extension) into
    (cond, cat, identity, pitch, yaw, roll).

    Faithful extraction of analysis_for_images.py:1413-1590. cond/cat/identity are bytes (or None);
    pitch/yaw/roll are ints or None (the structured array used an int16 sentinel for "unset" --
    here that is just None, which does not affect d'/FSI, which depend only on cat).
    """
    cond = cat = identity = None
    pitch = yaw = roll = None

    m_frei = re.match(_PATTERN_FREI, imn)
    m_song = re.match(_PATTERN_SONG, imn)

    if m_frei is not None:
        sp, ct, di = m_frei.group(4), m_frei.group(5), m_frei.group(6)
        nm = m_frei.group(7)
        ed = 'e' if m_frei.group(8) is not None else ''
        if nm.isnumeric():
            nm = int(float(nm))
        iv = m_frei.group(9) is not None
        if sp == 'Human':
            if ct == 'Head':
                cond = bytes('fh{:02}{}'.format(nm, ed), 'ascii'); cat = b'face_hum'
                identity = bytes('Hum{:02}{}'.format(nm, ed), 'ascii'); pitch = yaw = roll = 0
        elif sp == 'MacaqueRhesus':
            if ct == 'Head':
                cond = bytes('fr{:02}{}'.format(nm, ed), 'ascii'); cat = b'face_rhe'
                identity = bytes('Rhe{:02}{}'.format(nm, ed), 'ascii'); pitch = yaw = roll = 0
        elif sp == 'Marm':
            if ct == 'Head':
                if iv is True:
                    nm = 9
                cond = bytes('fm{}{:02}{}'.format(di[0:3], nm, ed), 'ascii'); cat = b'face_mrm'
                identity = bytes(di[0:8], 'ascii')
                if nm in _MARM_HEAD_POSE:
                    pitch, yaw, roll = _MARM_HEAD_POSE[nm]
                else:
                    warn('Could not recognize pitch, yaw, or roll of head image from filename.')
            if ct == 'Body':
                cond = bytes('bm{}{:02}{}'.format(di[0:3], nm, ed), 'ascii'); cat = b'body_mrm'
                identity = bytes(di[0:8], 'ascii')
        elif sp == 'Objects':
            m_ct = re.match(r'^([^0-9]+)([0-9])$', ct)
            if m_ct is not None:
                ct, ct_p2 = m_ct.group(1), m_ct.group(2)
                ct_p2 = int(float(ct_p2)) if ct_p2.isnumeric() else 0
            else:
                warn('Could not recognize object details from filename.')
                ct_p2 = 0
            if 'Manmade' in ct:
                cond = bytes('om{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii'); cat = b'obj'
            elif 'FruitVeg' in ct:
                cond = bytes('vf{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii'); cat = b'food'
            elif 'MultipartGeon' in ct:
                cond = bytes('og{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
                identity = bytes('Geon{:01}'.format(ct_p2), 'ascii'); cat = b'obj'
            elif 'Pairwise' in ct:
                cond = bytes('op{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii'); cat = b'obj'
            elif 'String' in ct:
                cond = bytes('os{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii'); cat = b'obj'
            else:
                warn('Could not recognize category of object image from filename.')
        else:
            warn('Could not recognize type of image from filename.')

    elif m_song is not None:
        tg, ng = m_song.group(4), m_song.group(5)
        cond = bytes('S{}{}'.format(tg, ng.zfill(2)), 'ascii')
        if tg == 'a':
            cat = b'animal'
        elif tg == 'o':
            cat = b'obj'
        elif tg == 'b':
            cat = b'blank' if imn == 'blank' else b'body_mrm'
        elif tg == 'm':
            cat = b'face_mrm'; pitch = yaw = roll = 0
        elif tg == 'u':
            cat = b'obj'
        elif tg == 'f':
            cat = b'food'
        elif tg == 'p':
            cat = b'scram_p'
        elif tg == 's':
            cat = b'scram_s'

    elif imn == 'blank':
        cond = b'blank'; cat = b'blank'

    elif 'Cartoon' in imn:
        m_ctn = re.match(r'^[^_]*Cartoon_([0-9]+)_?[^_]*_?(inverted)?$', imn)
        if m_ctn is not None:
            nm = m_ctn.group(1)
            cond = bytes('fcm{:04}'.format(int(nm)), 'ascii'); cat = b'face_ctn'
            identity = bytes(nm, 'ascii'); pitch = yaw = roll = 0

    else:
        warn('Could not recognize category or condition of image from filename ({}).'.format(imn))

    return cond, cat, identity, pitch, yaw, roll


def build_condition_metadata(stimlog, cond_col='cond', image_col='image'):
    """Per-condition image classification table, indexed by condition id, with columns
    cond, cat, id, pitch, yaw, roll, imagename. One row per unique condition (the image is
    expected to be constant within a condition)."""
    conditions = np.unique(stimlog[cond_col].dropna().to_numpy())
    rows = []
    for c in conditions:
        names = pd.unique(stimlog.loc[stimlog[cond_col] == c, image_col].dropna().values)
        if len(names) != 1:
            warn('Condition {} is associated with {} distinct images.'.format(c, len(names)))
        imagename = names[0] if len(names) else None
        imn = os.path.splitext(str(imagename))[0] if imagename is not None else ''
        cond, cat, identity, pitch, yaw, roll = classify_image(imn)
        rows.append({'condition': c, 'cond': cond, 'cat': cat, 'id': identity,
                     'pitch': pitch, 'yaw': yaw, 'roll': roll, 'imagename': imagename})
    return pd.DataFrame(rows).set_index('condition')


def attach_condition_metadata(ds, meta):
    """Attach per-condition metadata (from build_condition_metadata) as coords on ds.condition."""
    meta = meta.reindex(ds['condition'].values)
    coords = {col: ('condition', meta[col].to_numpy())
              for col in ['cond', 'cat', 'id', 'pitch', 'yaw', 'roll', 'imagename']}
    return ds.assign_coords(coords)


def supercategory_bools(cats):
    """Per-condition (face, non-face, non-face-object) boolean masks from the cat bytes array
    (analysis_for_images.py:1915-1925)."""
    cats = [b'' if c is None else bytes(c) for c in cats]
    is_face = np.array([b'face' in c and b'blank' not in c and b'scram' not in c and b'ctn' not in c
                        for c in cats])
    is_nonface = np.array([b'face' not in c and b'blank' not in c and b'scram' not in c
                           for c in cats])
    is_nonface_object = np.array([b'face' not in c and b'blank' not in c and b'scram' not in c
                                  and b'body' not in c for c in cats])
    return is_face, is_nonface, is_nonface_object


def face_dprime(ds, metric='Fzsc'):
    """Face discriminability index d' per ROI, using the cat coord on ds.

    Vinken et al. Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
    Faces vs ALL non-faces, SD-normalized: d' = (mu_F - mu_NF) / sqrt((sigma_F**2 + sigma_NF**2)/2).
    Distinct from FSI (different comparison group and formula) -- see face_selectivity_index.
    """
    is_face, is_nonface, _ = supercategory_bools(ds['cat'].values)
    return response_table.face_dprime(
        response_table.stim_window_response(ds, metric), is_face, is_nonface)


def face_selectivity_index(ds, metric='Fzsc'):
    """Face-selectivity index FSI per ROI, using the cat coord on ds.

    Freiwald and Tsao 2010 Science https://doi.org/10.1126/science.1194908
    Faces vs non-face OBJECTS only (bodies/scrambles/blank excluded), a mean-response ratio with
    no variance term: FSI = (R_F - R_O) / (R_F + R_O), clamped to +/-1 on sign disagreement.
    Distinct from d' (different comparison group and formula) -- see face_dprime.
    """
    is_face, _, is_nonface_object = supercategory_bools(ds['cat'].values)
    return response_table.face_selectivity_index(
        response_table.stim_window_response(ds, metric), is_face, is_nonface_object)


def responsive_anova(ds, metric='Fzsc'):
    """Per-ROI one-way ANOVA p-value across conditions over the stimulus window
    (analysis_for_images.py:1800-1802). Excluded trials are dropped via the mask."""
    stim = ds[metric].where(~ds['excluded']).where(ds['epoch'] == response_table.EPOCH_STIM)
    arr = stim.transpose('roi', 'condition', 'repeat', 'time').values
    n_roi, n_cond = arr.shape[0], arr.shape[1]
    p = np.full(n_roi, np.nan)
    for r in range(n_roi):
        groups = [arr[r, c][~np.isnan(arr[r, c])] for c in range(n_cond)]
        _, p[r] = f_oneway(*groups)
    return p


def roi_stats(ds, rois, resolution_umpx, metric='Fzsc'):
    """Per-ROI image-response statistics: spatial centroid, peak-driving condition and category,
    and face d'/FSI. Reproduces the non-WIP fields of analysis_for_images.py:2070-2106.

    ``rois`` is the suite2p stat array (each entry has 'xpix'/'ypix'); ``resolution_umpx`` is
    md['fov']['resolution_umpx']. Returns a DataFrame indexed by ROI.
    """
    resp_cond = response_table.stim_window_response(ds, metric).transpose('roi', 'condition').values
    cond_labels = ds['cond'].values
    cats = ds['cat'].values
    categories = pd.unique(cats)
    cat_masks = [np.array([c == k for c in cats]) for k in categories]
    resp_cat = np.column_stack([resp_cond[:, m].mean(axis=1) for m in cat_masks])

    dprime = face_dprime(ds, metric).values
    fsi = face_selectivity_index(ds, metric).values
    peak_cond_idx = np.nanargmax(resp_cond, axis=1)
    peak_cat_idx = np.nanargmax(resp_cat, axis=1)

    rows = []
    for r in range(resp_cond.shape[0]):
        centroid_px = np.array([rois[r]['xpix'].mean(), rois[r]['ypix'].mean()])
        rows.append({
            'roi': r,
            'centroid_px': centroid_px,
            'centroid_um': np.asarray(resolution_umpx) * centroid_px,
            'peak_cond': cond_labels[peak_cond_idx[r]],
            'peak_cond_val': resp_cond[r, peak_cond_idx[r]],
            'cat_of_peak_cond': cats[peak_cond_idx[r]],
            'peak_cat': categories[peak_cat_idx[r]],
            'peak_cat_val': resp_cat[r, peak_cat_idx[r]],
            'dprime': dprime[r],
            'fsi': fsi[r],
        })
    return pd.DataFrame(rows).set_index('roi')


def process_session(session_path, metrics=('FdFF', 'Fzsc'), responsiveness_metric='Fzsc',
                    roi_stats_metric='Fzsc', variant=None, baseline_method='medianbw'):
    """Thin image-paradigm driver: a session path -> the per-ROI image statistics, via the shared
    loaders + container (sessionio.build_session_response_table + the image helpers here).

    Returns a results dict with the response Dataset, the load context, per-metric face d'/FSI,
    the responsiveness ANOVA p-values, and the per-ROI stats table. Figures are intentionally left
    to plots.py -- call those on the returned arrays.
    """
    ds, ctx = sessionio.build_session_response_table(
        session_path, variant=variant, baseline_method=baseline_method)
    ds = attach_condition_metadata(ds, build_condition_metadata(ctx['stimlog']))
    return {
        'dataset': ds,
        'context': ctx,
        'dprime': {m: face_dprime(ds, m) for m in metrics},
        'fsi': {m: face_selectivity_index(ds, m) for m in metrics},
        'p_anova': responsive_anova(ds, responsiveness_metric),
        'roi_stats': roi_stats(ds, ctx['s2p']['ROIs'], ctx['md']['fov']['resolution_umpx'],
                               roi_stats_metric),
    }
