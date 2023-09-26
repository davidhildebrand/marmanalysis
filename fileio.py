#!/usr/bin/env python3

import argparse
import magic
import os

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

