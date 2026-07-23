#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble a session's per-ROI (roi x condition) response matrix + positions + condition metadata.

General-purpose (any 2P imaging session, not only cortex/PD): the SOM-vs-cortex topography comparison (SOM
plan, main-track step 2) is one consumer. Produces the things a tuning/topography analysis needs:

  1. the **(roi x condition) response matrix** -- each ROI's response profile over the conditions;
  2. **per-ROI position** (centroid in microns) -- the spatial axis of a topography analysis;
  3. the **condition -> image/stimulus map** -- ties each column to the stimulus, so a model side
     (e.g. ``dnn_som``) can be indexed consistently against the responses.

Plus the per-ROI **response reliability**, usable either as a soft weight or to justify a gate.

DEFAULTS encode decisions already made and measured (see analysis_roadmap.md items 2/6/12):
  * ``eye_gate_mode='landing_window'`` -- trials where the animal never looked at the stimulus add pure
    response variance; gating them lifts split-half tuning reliability 0.251 -> 0.319 on the responsive set,
    for the least data loss of the modes tested.
  * ``roi_mask=None`` (NO gate) is the default, and the gate is a caller-supplied boolean, so the same
    assembly can be re-run across gates for the gate-robustness curve rather than baking one choice in.
  * ``denoise=True`` runs PSN in the ORDER that was validated: fit on the COMPLETE (ungated) trial tensor,
    then apply the eye-gate exclusions only when averaging trials into the profile. Denoising an already
    gated table feeds PSN NaN trials (it assumes complete trials) and understates its benefit.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import images
import response_table as rt
import sessionio


def calculate_roi_centroids_um(rois, resolution_umpx):
    """Per-ROI spatial centroid in MICRONS, ``(n_roi, 2)``. ``rois`` is the suite2p stat array (each entry has
    'xpix'/'ypix'); ``resolution_umpx`` is ``md['fov']['resolution_umpx']``. Matches ``images.roi_stats``."""
    res = np.asarray(resolution_umpx, float)
    return np.array([[r['xpix'].mean(), r['ypix'].mean()] for r in rois], float) * res


def build_response_matrix(session_path, metric='Fzsc', eye_gate_mode='landing_window', variant=None,
                          denoise=False, psn_mode='conservative', normalize=None, reliability_splits=100):
    """Assemble the (roi x condition) response matrix + positions + condition map for one session.

    Returns a dict for ALL ROIs (no gate applied) so a caller can subset for several gates without
    rebuilding: ``response`` (n_roi, n_cond), ``roi_xy_um`` (n_roi, 2), ``reliability`` (n_roi,),
    ``conditions`` (DataFrame indexed by condition: imagename/cat/id/...), plus ``metric``,
    ``eye_gate_mode``, ``denoise``, ``n_trials_excluded``, ``framerate`` and ``session``.

    ``normalize='peak'`` divides each ROI's profile by its peak |response| (``response_table.normalize_response``)
    -- removes per-cell GAIN while preserving tuning shape. Pair it with a real gate or reliability weighting:
    normalising hands a weak noisy cell the same weight as a strongly driven one.
    """
    ds, ctx = sessionio.build_session_response_table(session_path, variant=variant,
                                                     eye_gate_mode=eye_gate_mode)
    excluded = ds['excluded'].transpose('condition', 'repeat').values

    if denoise:
        # PSN assumes COMPLETE trials -> fit on the ungated tensor, apply the gate only at averaging time.
        import denoising
        ds_full, _ = sessionio.build_session_response_table(session_path, variant=variant,
                                                            eye_gate_mode='none')
        ds_den, _ = denoising.denoise_psn(ds_full, metric=metric, mode=psn_mode, diagnostic=False)
        trials = rt.trial_response(ds_den, metric).transpose('roi', 'condition', 'repeat').values.copy()
        trials[:, excluded] = np.nan
    else:
        # exclusions are already NaN via the response table's mask
        trials = rt.trial_response(ds, metric).transpose('roi', 'condition', 'repeat').values

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)     # conditions fully excluded -> NaN
        response = np.nanmean(trials, axis=2)

    if normalize:
        da = xr.DataArray(response, dims=('roi', 'condition'),
                          coords={'roi': ds['roi'], 'condition': ds['condition']})
        response = rt.normalize_response(da, method=normalize).values

    return {
        'response': response,
        'roi_xy_um': calculate_roi_centroids_um(ctx['s2p']['ROIs'], ctx['md']['fov']['resolution_umpx']),
        # Reliability is ALWAYS measured on the RAW (non-denoised) table, even when denoise=True. In-sample
        # denoising collapses within-condition trial scatter -- the very quantity split-half reliability
        # measures -- so reliability computed on denoised trials is spuriously inflated (the same mechanism
        # that drove selective 62 -> 791). Raw reliability is the honest weight for BOTH variants.
        'reliability': rt.split_half_reliability(ds, metric, n_splits=reliability_splits),
        'conditions': images.build_condition_metadata(ctx['stimlog']),
        'condition_ids': ds['condition'].values,
        'metric': metric, 'eye_gate_mode': eye_gate_mode, 'denoise': bool(denoise),
        'normalize': normalize, 'n_trials_excluded': int(excluded.sum()),
        'framerate': ctx['md']['framerate'], 'session': os.path.basename(session_path.rstrip('/')),
        'ds': ds, 'ctx': ctx,
    }


def apply_roi_mask(data, roi_mask):
    """Subset a ``build_cortical_dataset`` result to the ROIs in ``roi_mask`` (boolean, one per ROI).
    Returns a shallow copy with ``response`` / ``roi_xy_um`` / ``reliability`` subset and ``n_roi`` set."""
    m = np.asarray(roi_mask, bool)
    out = dict(data)
    out.update({'response': data['response'][m], 'roi_xy_um': data['roi_xy_um'][m],
                'reliability': data['reliability'][m], 'roi_mask': m, 'n_roi': int(m.sum())})
    return out


def _category_blocked_condition_order(data):
    """Condition order that groups conditions by category into contiguous blocks, plus the per-block extents
    for x-axis tick labelling. Returns (cond_order, blocks) where blocks is a list of (category, i0, i1)."""
    cats = data['conditions'].reindex(data['condition_ids'])['cat'].to_numpy()
    cats = np.array(['' if c is None else (c.decode(errors='ignore') if isinstance(c, bytes) else str(c))
                     for c in cats])
    order, blocks, pos = [], [], 0
    for cat in pd.unique(cats):                          # preserve stimulus-set category order (not alphabetical)
        idx = np.where(cats == cat)[0]
        order.extend(idx.tolist())
        blocks.append((cat, pos, pos + len(idx)))
        pos += len(idx)
    return np.array(order, int), blocks


def plot_response_heatmap(datasets, labels, outdir='output', tag=None, sort_metric='dprime_f',
                          vlim=1.0, cmap='bwr', threshold=0.2):
    """Side-by-side (roi x condition) response heatmaps in the ``analysis_for_images.py`` house style -- e.g.
    raw vs PSN-denoised.

    Rows are ROIs **sorted by ``sort_metric`` descending** (default ``dprime_f`` = face d', the original
    ordering), columns are conditions grouped into category blocks, colour is ``cmap`` (default ``bwr``) at
    fixed ``vmin/vmax = -/+vlim`` (default 1.0). A left side-bar shows the per-ROI sort statistic, and dotted
    lines mark the +/-``threshold`` boundaries on it. The ROI order + statistic are taken from the FIRST
    dataset and reused for every panel, so raw and denoised are comparable row-for-row.

    ``sort_metric`` is a key computed per ROI on the first dataset: 'dprime_f' (face d', via
    ``images.face_dprime``), 'reliability', or 'peak' (max response). d'-sorting is meaningful structure (unlike
    a peak-condition sort, whose diagonal appears even in noise), which is why it is the default here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone
    import images

    first = datasets[0]
    cond_order, blocks = _category_blocked_condition_order(first)

    # per-ROI sort statistic on the first dataset
    if sort_metric == 'dprime_f':
        ds0 = images.attach_condition_metadata(first['ds'], first['conditions'])
        stat = images.face_dprime(ds0, metric=first['metric']).transpose('roi').values
        stat_label, thr = "$d^\\prime_F$", threshold
    elif sort_metric == 'reliability':
        stat, stat_label, thr = first['reliability'], 'reliability', threshold
    else:
        with np.errstate(all='ignore'):
            stat, stat_label, thr = np.nanmax(first['response'], axis=1), 'peak', None
    roi_order = np.argsort(np.where(np.isfinite(stat), stat, -np.inf))[::-1]
    stat_sorted = stat[roi_order]

    n_panel = len(datasets)
    fig, axes = plt.subplots(1, 1 + n_panel, figsize=(1.2 + 4.6 * n_panel, 7.0),
                             gridspec_kw={'width_ratios': [0.6] + [4.6] * n_panel})
    ax_stat = axes[0]
    n_roi = len(roi_order)
    ax_stat.barh(range(n_roi), stat_sorted, height=1.0, color='0.5')
    ax_stat.axvline(0, color='0.0', linewidth=0.5)
    ax_stat.set_xlabel(stat_label)
    ax_stat.set_ylim(n_roi - 0.5, -0.5)
    ax_stat.set_yticks([])
    for s in ('right', 'left', 'top'):
        ax_stat.spines[s].set_visible(False)

    img_hm = None
    for ax, d, lab in zip(axes[1:], datasets, labels):
        M = d['response'][:, cond_order][roi_order]
        img_hm = ax.imshow(M, vmin=-vlim, vmax=vlim, aspect='auto', cmap=cmap, interpolation='none')
        ax.set_title('%s (%d ROIs x %d cond)' % (lab, M.shape[0], M.shape[1]), fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([i0 - 0.5 for _, i0, _ in blocks])
        ax.set_xticks([(i0 + i1) / 2 - 0.5 for _, i0, i1 in blocks], minor=True)
        ax.set_xticklabels([b[0] for b in blocks], minor=True, rotation=90, fontsize=7)
        ax.set_xticklabels([])
        ax.tick_params(which='minor', length=0)
        if thr is not None:
            for edge, sign in ((np.where(stat_sorted > thr)[0], 1), (np.where(stat_sorted < -thr)[0], -1)):
                if edge.size:
                    y = edge.max() if sign > 0 else edge.min()
                    for a in (ax, ax_stat):
                        a.axhline(y, color='0.2', linestyle='dotted', linewidth=0.5)
    axes[1].set_ylabel('ROI (sorted by %s, descending)' % stat_label)
    fig.colorbar(img_hm, ax=list(axes[1:]), fraction=0.03, pad=0.02, label=first['metric'])
    fig.suptitle('response matrix — %s' % (tag or first['session'])[:60])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'response_heatmap_%s_%s.png'
                     % (tag or 'session', datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p
