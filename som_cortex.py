#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cortical side of the SOM-vs-cortex topography comparison (SOM plan, main-track step 2).

Assembles, for one session, the three things the comparison needs:

  1. the **(roi x condition) response matrix** -- each ROI's tuning profile over the presented images;
  2. **per-ROI cortical position** (centroid in microns) -- the spatial axis of the topography analysis;
  3. the **condition -> image map** -- ties each column to the stimulus file the DNN/SOM side consumes
     (``dnn_som``), so cortex and model are indexed consistently.

Plus the per-ROI **tuning reliability**, usable either as a soft weight or to justify a gate.

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
import xarray as xr

import images
import response_table as rt
import sessionio


def calculate_roi_centroids_um(rois, resolution_umpx):
    """Per-ROI spatial centroid in MICRONS, ``(n_roi, 2)``. ``rois`` is the suite2p stat array (each entry has
    'xpix'/'ypix'); ``resolution_umpx`` is ``md['fov']['resolution_umpx']``. Matches ``images.roi_stats``."""
    res = np.asarray(resolution_umpx, float)
    return np.array([[r['xpix'].mean(), r['ypix'].mean()] for r in rois], float) * res


def build_cortical_dataset(session_path, metric='Fzsc', eye_gate_mode='landing_window', variant=None,
                           denoise=False, psn_mode='conservative', normalize=None, reliability_splits=100):
    """Assemble the cortical response matrix + positions + condition map for one session.

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


def plot_response_heatmap(datasets, labels, outdir='output', tag=None, clim_pct=98, sort_rois=True):
    """Side-by-side (roi x condition) response heatmaps -- e.g. raw vs PSN-denoised.

    Conditions are ordered by category so category blocks are visible; ROIs are ordered by their
    peak-driving condition (``sort_rois``) so tuning structure shows as a diagonal rather than noise. The
    ROI order is taken from the FIRST dataset and reused for all panels, so the panels are comparable
    row-for-row. Colour limits are symmetric at the ``clim_pct`` percentile of the first panel, shared across
    panels. Returns the saved figure path.

    READING THE FIGURE -- the DIAGONAL IS A SORTING ARTIFACT. Ordering ROIs by their peak-driving condition
    produces a diagonal even in PURE NOISE (every cell has *some* argmax), so the diagonal is NOT evidence of
    tuning. What IS interpretable: VERTICAL bands (conditions that drive many cells irrespective of row order)
    and HORIZONTAL stripes (broadly responsive cells) -- neither can be manufactured by the row sort. Set
    ``sort_rois=False`` for an unsorted view, and compare against a condition-label-shuffled table
    (``response_table.shuffle_condition_labels``) to see the artifact alone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone

    first = datasets[0]
    cats = first['conditions'].reindex(first['condition_ids'])['cat'].to_numpy()
    cats = np.array([b'' if c is None else (c if isinstance(c, bytes) else str(c).encode()) for c in cats])
    cond_order = np.argsort([c.decode(errors='ignore') for c in cats], kind='stable')

    R0 = first['response'][:, cond_order]
    if sort_rois:
        with np.errstate(all='ignore'):
            peak = np.nanargmax(np.where(np.isfinite(R0), R0, -np.inf), axis=1)
        roi_order = np.argsort(peak, kind='stable')
    else:
        roi_order = np.arange(R0.shape[0])

    v = np.nanpercentile(np.abs(R0), clim_pct)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 6.0), squeeze=False)
    for ax, d, lab in zip(axes[0], datasets, labels):
        M = d['response'][:, cond_order][roi_order]
        im = ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title('%s\n%d ROIs x %d conditions' % (lab, M.shape[0], M.shape[1]), fontsize=10)
        ax.set_xlabel('condition (sorted by category)')
        fig.colorbar(im, ax=ax, fraction=0.046, label=first['metric'])
    axes[0][0].set_ylabel('ROI (sorted by peak-driving condition)')
    fig.suptitle('cortical response matrix — %s' % (tag or first['session'])[:60])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, 'som_cortex_heatmap_%s_%s.png'
                     % (tag or 'session', datetime.now(timezone.utc).strftime('%Y%m%dd%H%M%StUTC')))
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
