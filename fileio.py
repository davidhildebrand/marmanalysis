#!/usr/bin/env python3

import argparse
# import hashlib
import os


# def calculate_sha256_hash(file_path):
#     hash_sha256 = hashlib.sha256()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             hash_sha256.update(chunk)
#     return hash_sha256.hexdigest()
#
#
# def calculate_md5_hash(file_path):
#     hash_md5 = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()
#
#
# def checksum_file(file_path):
#     h_sha256 = hashlib.sha256()
#     h_md5 = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             h_sha256.update(chunk)
#             h_md5.update(chunk)
#     return h_sha256.hexdigest(), h_md5.hexdigest()


# TIFF signatures: classic (II*\0 / MM\0*) and BigTIFF (II+\0 / MM\0+), both byte orders.
_TIFF_SIGNATURES = (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+')


def is_tiff(filepath: str) -> bool:
    """True if the file is a TIFF (classic or BigTIFF, either byte order), from its 4-byte signature.

    A direct signature check, chosen deliberately over both python-magic (which needs the libmagic
    system library) and the pure-Python `filetype` package. `filetype` is NOT used here because it
    does NOT recognize BigTIFF -- filetype.guess_mime() returns None for the BigTIFF signature
    (II+ / MM+) -- and ScanImage writes BigTIFF for large recordings, so it would falsely reject
    exactly the files this pipeline cares about. A signature check is dependency-free and covers
    every TIFF variant (classic and BigTIFF, little- and big-endian).
    """
    with open(filepath, 'rb') as f:
        signature = f.read(4)
    return signature in _TIFF_SIGNATURES


if __name__ == '__main__':
    # Parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('source',
                        help='Path to a ScanImage TIFF data file. [required]')
    opts = parser.parse_args()

    if os.path.isfile(opts.source):
        source = opts.source
        source_path = os.path.split(source)[0] + os.path.sep
        source_base = os.path.basename(source)
        source_name = os.path.splitext(source_base)[0]
        source_ext = os.path.splitext(source_base)[1]
    else:
        m = 'Source file does not exist ({}).'.format(opts.source)
        raise argparse.ArgumentTypeError(m)

    if not is_tiff(source):
        m = 'Source file must be a TIFF stack (with extension .tif or .tiff).'
        raise RuntimeError(m)

