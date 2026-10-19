#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""layer_alignment's Bao et al.-style decoder: a planted linear mapping from features to population responses is
decoded far above chance, and responses unrelated to the features decode at chance.
Run:  /Users/davidh/Data/Vibe/Analysis_Freiwald/.venv/bin/python -m pytest marmanalysis/tests/test_layer_alignment.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layer_alignment as la


def test_decode_accuracy_planted_and_null():
    rng = np.random.default_rng(0)
    n_img, n_cell, k = 80, 300, 20
    Y = rng.standard_normal((n_img, k)) * np.linspace(3, 0.5, k)          # PC-like scores, decreasing variance
    X = Y @ rng.standard_normal((k, n_cell)) + 0.5 * rng.standard_normal((n_img, n_cell))
    acc = la.decode_accuracy(la.loo_ridge_folds(X), Y, np.random.default_rng(1))
    assert acc > 0.8
    X_null = rng.standard_normal((n_img, n_cell))
    acc_null = la.decode_accuracy(la.loo_ridge_folds(X_null), Y, np.random.default_rng(1))
    assert acc_null < 0.1                                             # chance is 1/40 = 0.025
