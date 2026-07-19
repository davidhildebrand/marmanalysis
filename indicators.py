#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcium-indicator decay time constants (suite2p ``tau``, seconds) keyed by indicator name, with per-entry
provenance -- so the sensor kinetics are selected by a single ``indicator`` choice rather than a bare magic
number, and can be swapped/refined in one place (e.g. for OASIS spike deconvolution later).

suite2p's ``tau`` is the exponential-decay TIME CONSTANT of the deconvolution kernel (exp(-t/tau)); the
principled value is the published SINGLE-ACTION-POTENTIAL half-decay time t_half divided by ln2 -- papers
report t_half (half-decay), NOT tau, so tau = t_half / ln2 (LN2 = 0.693). 1-AP kinetics are the convention
(multi-AP decays run slower from sensor supralinearity). An empirically-refined per-preparation value (e.g.
for the ribo-/soma-targeted variants, once measured with signal_quality.calculate_indicator_decay_tau /
report_indicator_tau.py) can replace the published default for that entry.
"""
import numpy as np

LN2 = float(np.log(2.0))                       # 0.693; tau = t_half / LN2

# indicator name -> (tau_sec, provenance note). Extend as values become available.
INDICATOR_TAU_SEC = {
    'jGCaMP8s': (
        0.29,
        '1-AP half-decay t_half ~0.20 s in vivo (mouse brain) -> tau = t_half/ln2 = 0.29 s. '
        'Zhang et al. Looger 2023 Nature, https://doi.org/10.1038/s41586-023-05828-9. '
        'Confirmed on Cadbury PD (empirical t_half ~0.19 s via calculate_indicator_decay_tau).'),
    'ribo-jGCaMP8s': (
        0.29,
        'Ribosome-tethered jGCaMP8s (ribo-L1-jGCaMP8s) -- our default construct. Uses the jGCaMP8s value '
        'pending a per-preparation empirical measurement (tethering may shift kinetics slightly).'),
    'soma-jGCaMP8s': (
        0.29,
        'Soma-targeted jGCaMP8s. Uses the jGCaMP8s value pending an empirical measurement.'),
    'GCaMP6s': (
        1.0,
        'suite2p default/recommended tau for GCaMP6s (~1.0-1.25 s). NB the 1-AP half-decay is only ~0.55 s '
        '(Chen et al. 2013 Nature Fig 3f, https://doi.org/10.1038/nature12354 -> t_half/ln2 ~0.8 s), but '
        't_half is strongly AP-count-dependent (~0.5 s at 1 AP to ~2 s at 10 AP) and suite2p uses a pragmatic '
        'effective decay ~1.0 s -- kept here for OASIS/suite2p compatibility.'),
}
DEFAULT_INDICATOR = 'jGCaMP8s'


def indicator_tau(indicator=DEFAULT_INDICATOR):
    """suite2p decay time constant tau (seconds) for a named indicator. Raises KeyError (listing the known
    names) if the indicator is unknown."""
    try:
        return INDICATOR_TAU_SEC[indicator][0]
    except KeyError:
        raise KeyError('unknown indicator %r; known indicators: %s'
                       % (indicator, ', '.join(sorted(INDICATOR_TAU_SEC)))) from None


def indicator_tau_note(indicator=DEFAULT_INDICATOR):
    """Provenance note (reference / basis) for the indicator's tau."""
    return INDICATOR_TAU_SEC[indicator][1]
