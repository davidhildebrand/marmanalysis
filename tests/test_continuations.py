#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuation sessions must never include a stimulus shown while no frames were acquired (DH 2026-10-18: "there
should be a test for continuations to consistently test that they have no stimulus during the gap because handling
may have changed over time").

Every continuation-style folder in the catalog (``*ontinuation*``: the log-less ``_Continuation`` kind served from an
earlier same-date log, and the ``_withContinuation`` kind with its own log) is discovered automatically, its stimulus
log is scanned with ``sessionio.find_acquisition_gaps`` (frame counter vs stimulus-PC clock), and the test asserts
that no stimulus falls in any acquisition gap. For the log-less kind it also asserts that the restart is visible in
the parent log and that every trial the loader serves sits on the continuation's frame clock. Real-data cases are
skipped when the data are not present; the synthetic cases always run.

Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python -m pytest marmanalysis/tests/test_continuations.py -v
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sessionio

BASE = '/Users/davidh/Data/Vibe/Analysis_Freiwald/suite2p_results'


def _continuation_cases():
    cases = []
    for d in sorted(glob.glob(os.path.join(BASE, '*', '*', '*ontinuation*'))):
        if not os.path.isdir(d):
            continue
        own_log = any(sessionio.find_stimlog_sources(d).values())
        src = d if own_log else sessionio.find_continuation_parent(d)
        if src:
            cases.append((d, src))
    return cases


CASES = _continuation_cases()


def _log_with_gap(gap_trial=False):
    """Two recordings on one frame counter: 20 trials, a 40 s restart, 20 more trials; optionally one stimulus shown
    during the restart (stimulus clock inside the gap, frame counter frozen)."""
    fr = 4.0
    onset = np.arange(40) * 12.0 + 5.0
    t = 10.0 + onset / fr
    t[20:] += 40.0
    rows = {'cond': np.arange(40) % 8, 'acqfr_stim_i': onset, 't_stim_i': t}
    log = pd.DataFrame(rows)
    if gap_trial:
        extra = pd.DataFrame({'cond': [99], 'acqfr_stim_i': [onset[19] + 4.0], 't_stim_i': [t[19] + 20.0]})
        log = pd.concat([log, extra], ignore_index=True)
    return log, fr


def test_find_acquisition_gaps_synthetic():
    log, fr = _log_with_gap()
    gaps = sessionio.find_acquisition_gaps(log, fr)
    assert len(gaps) == 1 and abs(gaps[0]['gap_sec'] - 40.0) < 0.5 and gaps[0]['trials_in_gap'] == []
    log, fr = _log_with_gap(gap_trial=True)
    gaps = sessionio.find_acquisition_gaps(log, fr)
    assert len(gaps) == 1 and gaps[0]['trials_in_gap'] == [40]           # the stimulus shown during the restart
    assert sessionio.find_acquisition_gaps(log.drop(columns='t_stim_i'), fr) == []


def test_session_slug():
    """Same string as the main-branch scripts' save_pfix (animal_str + date_str + session_abbrev_str)."""
    p = BASE + '/Curly/20240130d/172813tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow060p2mW_stimMultimodal'
    assert sessionio.session_slug(p) == 'Curly20240130d172813tUTC'
    assert sessionio.session_slug(p + '/') == 'Curly20240130d172813tUTC'
    assert sessionio.session_slug(BASE + '/Dali/20230910d/180205tUTC_Continuation') == 'Dali20230910d180205tUTC'
    assert sessionio.session_slug(BASE + '/Dali/20230910d/nounderscore') == 'Dali20230910d'


@pytest.mark.parametrize('session, log_session', CASES, ids=[sessionio.session_slug(c[0]) for c in CASES])
def test_no_stimulus_shown_during_acquisition_gap(session, log_session):
    log, _ = sessionio.load_stimlog(log_session)
    fr = float(sessionio.load_metadata(log_session)['framerate'])
    gaps = sessionio.find_acquisition_gaps(log, fr)
    shown = [(g['gap_sec'], g['trials_in_gap']) for g in gaps if g['trials_in_gap']]
    assert not shown, 'stimuli shown during an acquisition gap in %s: %s' % (sessionio.session_slug(log_session), shown)
    if session != log_session:                       # log-less _Continuation served from the parent's log
        assert gaps, 'no acquisition restart found in the parent log of %s' % sessionio.session_slug(session)
        _, prov = sessionio.load_stimlog(session)
        chk = prov['continuation_checks']
        assert chk['n_off_clock'] == 0 and chk['n_duplicate_onsets'] == 0 and chk['n_degenerate_stim_window'] == 0
