#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import magic
import os
from warnings import warn

import metadata


def is_tiff(filepath: str) -> bool:
    # source_ext != '.tif' and source_ext != '.tiff':
    allowed_types = ['image/tiff', 'image/tif']
    if magic.from_file(filepath, mime=True) not in allowed_types:
        return False
    return True


def json_serializer(obj):
    from datetime import date, datetime, time
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    else:
        warn('Type {} not serializable.'.format(type(obj)))
        return str(obj)


# Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('source',
                    help='Path to a ScanImage TIFF data file. [required]')
opts = parser.parse_args()

if os.path.isfile(opts.source):
    source = opts.source
    source_path = os.path.split(source)[0]
    source_base = os.path.basename(source)
    source_name = os.path.splitext(source_base)[0]
    source_ext = os.path.splitext(source_base)[1]
else:
    raise argparse.ArgumentTypeError('Source file does not exist ({}).'.format(opts.sourcefile))

if not is_tiff(source):
    raise RuntimeError('Source file must be a TIFF stack.')


p = dict()
p['save'] = dict()
p['save']['hdf5'] = True
p['save']['tif'] = True
p['save']['metadata'] = True
p['save']['mean'] = True
p['save']['video'] = True

simd = metadata.get_metadata(source)
md = metadata.extract_useful_metadata(simd)

if md['mrois']['overlap'] is not False or md['mrois']['overlap_px'] is not None:
    raise Exception('Handling overlapping MROIs is not yet implemented.')
if md['n_planes'] != 1:
    RuntimeError('Handing multi-plane data is not yet implemented.')

sp = source_path + os.path.sep

if p['save']['metadata']:
    import pickle

    save_path_mdp = sp + source_name + '_metadata.pickle'
    if os.path.isfile(save_path_mdp):
        warn('Metadata file already exists, overwriting ({}).'.format(save_path_mdp))
    with open(save_path_mdp, 'wb') as mdpf:
        pickle.dump(md, mdpf)

    save_path_mdj = sp + source_name + '_metadata.json'
    if os.path.isfile(save_path_mdj):
        warn('Metadata file already exists, overwriting ({}).'.format(save_path_mdj))
    with open(save_path_mdj, 'w') as mdjf:
        json.dump(md, mdjf, indent=4, sort_keys=True, default=json_serializer)
