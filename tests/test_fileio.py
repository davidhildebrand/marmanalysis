#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for fileio.is_tiff -- the dependency-free TIFF signature check that replaced python-magic.

Crucially covers BigTIFF (the case the `filetype` package misses and the reason we use a direct
signature check rather than `filetype`).

Run:  ../.venv/bin/python -m pytest marmanalysis/tests/test_fileio.py -v
"""

import os
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fileio


def _write_tiff(path, **kw):
    tifffile.imwrite(str(path), (np.random.rand(16, 16) * 255).astype('uint8'), **kw)
    return str(path)


def test_is_tiff_classic(tmp_path):
    assert fileio.is_tiff(_write_tiff(tmp_path / 'classic.tif')) is True


def test_is_tiff_bigtiff(tmp_path):
    # BigTIFF (signature II+\0) -- what `filetype` returns None for; ScanImage writes it for big stacks.
    assert fileio.is_tiff(_write_tiff(tmp_path / 'big.tif', bigtiff=True)) is True


def test_is_tiff_big_endian(tmp_path):
    assert fileio.is_tiff(_write_tiff(tmp_path / 'be.tif', byteorder='>')) is True


def test_is_tiff_rejects_non_tiff(tmp_path):
    p = str(tmp_path / 'not_a_tiff.bin')
    with open(p, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 64)   # a PNG header, not TIFF
    assert fileio.is_tiff(p) is False


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
