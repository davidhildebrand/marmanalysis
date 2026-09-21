#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the original processing-machine /Data/... paths (from each session's suite2p ops) for the sessions
recently worked on, so the raw movies (input .h5 / registered data.bin) can be transferred to this machine.
Run:  .venv/bin/python marmanalysis/get_data_paths.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sessionio

TARGETS = [
    ('Cadbury', '20221016d', 'ImagesSongFOBonly'),   # the PD topography session
    ('Curly', '20231103d', 'ImagesFOBmin'),          # #2 large-FOV, 8-category
    ('Dali', '20230810d', 'ImagesSongFOBonly'),      # #2 same-stimulus large-FOV
    ('Dali', '20230910d', 'ImagesFOBmany'),          # #2 large-FOV, 8-category
]


def resolve(animal, date, token):
    hits = [h for h in sorted(glob.glob('suite2p_results/%s/%s/*%s*' % (animal, date, token)))
            if os.path.isdir(h) and glob.glob(os.path.join(h, 'suite2p*'))]
    return hits[0] if hits else None


for animal, date, token in TARGETS:
    path = resolve(animal, date, token)
    print('\n=== %s / %s / %s ===' % (animal, date, token))
    if path is None:
        print('  (session dir not found)'); continue
    print('  in-project dir : %s' % path)
    try:
        s2p_dir = sessionio.find_suite2p_dir(path)
        print('  suite2p variant: %s' % os.path.basename(s2p_dir))
        ops = np.load(os.path.join(s2p_dir, 'plane0', 'ops.npy'), allow_pickle=True).item()
        h5 = ops.get('h5list') or []
        print('  save_path0     : %s' % ops.get('save_path0'))
        print('  raw h5 (input) : %s' % (h5[0] if h5 else None))
        print('  reg data.bin   : %s' % ops.get('reg_file'))
        print('  data_path      : %s' % ops.get('data_path'))
    except Exception as e:
        print('  ERROR: %s: %s' % (type(e).__name__, e))
