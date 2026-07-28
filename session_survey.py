#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-catalog survey (roadmap backlog: session catalog). Enumerates sessions under `suite2p_results/`,
parses the stimulus + FOV tokens from each directory name, and lists each session's `suite2p_*` extraction
variants with ROI counts (from `iscell.npy`, memory-mapped so it's cheap). Read-only, popup-free. Sorted by FOV
area and with a LARGE-FOV shortlist for the low-spatial-frequency topography test (roadmap 12e / #2).
Run:  .venv/bin/python marmanalysis/session_survey.py
"""
import glob
import os
import re

import numpy as np

ROOT = 'suite2p_results'


def fov_um(name):
    m = re.search(r'fov(\d+)x(\d+)um', name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def stim_token(name):
    m = re.search(r'stim([A-Za-z0-9]+)', name)
    return m.group(1) if m else '?'


def n_roi(variant_dir):
    ic = os.path.join(variant_dir, 'plane0', 'iscell.npy')
    try:
        return int(np.load(ic, mmap_mode='r').shape[0]) if os.path.isfile(ic) else None
    except Exception:
        return None


def main():
    rows = []
    for sess in sorted(glob.glob(os.path.join(ROOT, '*', '*', '*'))):
        if not os.path.isdir(sess):
            continue
        parts = sess.split(os.sep)
        animal, date, name = parts[-3], parts[-2], parts[-1]
        variants = [(os.path.basename(v).replace('suite2p_', ''), n_roi(v))
                    for v in sorted(glob.glob(os.path.join(sess, 'suite2p*'))) if os.path.isdir(v)]
        w, h = fov_um(name)
        rows.append((animal, date, stim_token(name), w, h, variants))

    rows.sort(key=lambda r: -(r[3] * r[4]))
    print('%d sessions under %s/  (sorted by FOV area)\n' % (len(rows), ROOT))
    print('%-9s %-10s %-20s %-12s  variants (n_roi)' % ('animal', 'date', 'stim', 'fov_um'))
    for an, dt, st, w, h, vs in rows:
        vtxt = ', '.join('%s=%s' % (n, r) for n, r in vs) or '(none)'
        print('%-9s %-10s %-20s %5dx%-5d  %s' % (an, dt, st[:20], w, h, vtxt))

    print('\nLARGE-FOV shortlist (>= 1500 um) for the low-frequency topography test:')
    for an, dt, st, w, h, vs in rows:
        if w >= 1500:
            vtxt = ', '.join('%s=%s' % (n, r) for n, r in vs) or '(none)'
            print('  %s/%s  stim=%s  %dx%d um  %s' % (an, dt, st, w, h, vtxt))


if __name__ == '__main__':
    main()
