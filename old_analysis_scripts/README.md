# Old analysis scripts

Defunct scripts kept for reference and inspiration. Nothing here is imported, run, or tested, and nothing here is
maintained. New work goes into the shared library one folder up.

## The `analysis_for_*.py` monoliths

`analysis_for_images.py`, `analysis_for_dots.py`, and `analysis_for_gratings.py` were the working analysis scripts
before the shared library (moved here on the `vibe` branch, 2026-11-17). The library cites them by line number, for
example `analysis_for_images.py:1604-1667` in `response_table.py`. Those citations refer to the files in this folder,
which are unchanged since the move, so the line numbers still match.

They run cell by cell (`# %%`) with their sibling modules on the path and a host-name block for the data location, so
they no longer run from here as they are.

### What the shared library has taken over

| Monolith section | Now in |
|---|---|
| Find files, load metadata, suite2p output, stimulus log | `sessionio.py`, `parsers.py`, `metadata.py` |
| Fluorescence metrics, baseline, z-score | `sessionio.compute_fluorescence_metrics`, `filters.py` |
| SNR estimate, indicator decay | `signal_quality.py`, `indicators.py` |
| Stimulus timing, trial windows, the response data table | `sessionio.py`, `response_table.py` |
| Image classification, categories, head pose, template order | `images.py` |
| Responsive cells, per-ROI statistics (d', FSI) | `images.py`, `response_table.py` (parity-tested) |
| Tuning similarity against cortical distance | `topography.py`, `spatial_structure.py` |
| Response heatmap | `response_matrix.plot_response_heatmap` (one generalized version) |
| Eye position | `eyetracking.py` (a different, calibration-free approach) |
| Stimulus feature space: per-ROI feature preference, encoding weights, preference against distance | `feature_topography.py`, `feature_topography_v2.py`, `layer_alignment.py` (re-implemented, see below) |
| Direction and orientation indices, von Mises fits | `tuning.py` |
| Gratings: trial selection, OSI and DSI per ROI | `gratings.py` |
| Dots: stimulus-log parsing only | `parsers.parse_log_stim_dots`, `sessionio` |

### What still lives only here

- **Per-session settings** (`analysis_for_images.py` lines 96-640, and the same block in the other two): the list of
  sessions with their per-session overrides. This is the de facto session catalog.
- **Moving dots analysis** (`analysis_for_dots.py`): the `StimulusDots` class, the per-ROI direction statistics, the
  tuned and untuned split, tuning histograms, tuning maps, and the preferred-direction difference against distance with
  its exponential fits (lines 1290-1518). The library has no dots driver yet.
- **Gratings figures** (`analysis_for_gratings.py`): tuning histograms, tuning maps, direction difference against
  distance, and per-condition response plots. Note that this script never loaded its data and computed the DSI under an
  OSI name. `gratings.py` is a rebuild, not a port.
- **Deconvolution** (`analysis_for_images.py` lines 1035-1125).
- **Eye-tracking calibration plots** (lines 1126-1264).
- **Most figures** (lines 2163-3206): tuning histograms, example-ROI traces, the heatmap variants, and the ROI-mask and
  stimulus-image overlays on the mean image. The overlay helpers themselves are in `plots.py`.
- **Simulated discontinuous-boundary control** (lines 3207-3303).
- **Parts of the stimulus-space analyses** (lines 3541-4009). The core idea is re-implemented in the library: image
  features from AlexNet, their principal components, each ROI's preference along them, and how that relates to
  cortical distance. The library uses relu7 features of the session's own images, correlation and ridge weights, and
  its own topography statistics. Still only here: the precomputed stimulus-space record on fc6 features
  (`generate_stimulus_space.py`), the 2-D picture of the space with each ROI's preferred-axis vector (response-weighted
  and least-squares), the ROI map colored by preferred-axis angle relative to the face vector, and the
  response-weighted position against distance.

## The other files

`trialsorting_2pRAM_*.py` are the per-session ancestors of the monoliths. `reshape_SP.py`,
`reshaping_stitching_aligning.py`, and `reshaping_and_aligning_planes_shortcut_appendingPlanes.py` are earlier attempts
at reshaping, stitching, and aligning ScanImage mROI strips; the current reshaping is `reshape_mrois_to_planes.py`.
