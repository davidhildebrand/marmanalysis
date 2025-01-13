#!/usr/bin/env python3

import argparse
# import hashlib
import magic
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


def is_tiff(filepath: str) -> bool:
    allowed_types = ['image/tiff', 'image/tif']
    if magic.from_file(filepath, mime=True) not in allowed_types:
        return False
    return True

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
    m = 'Source file does not exist ({}).'.format(opts.sourcefile)
    raise argparse.ArgumentTypeError(m)

if not is_tiff(source): # source_ext != '.tif' and source_ext != '.tiff':
    m = 'Source file must be a TIFF stack (with extension .tif or .tiff).'
    raise RuntimeError(m)

