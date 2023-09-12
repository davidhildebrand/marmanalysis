#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta
import magic
import numpy as np
import os

import metadata # *** TODO *** make relative?

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

md = metadata.get_scanimage_metadata(source)

n_frames = md['n_frames']
n_strips = len(md['json']['RoiGroups']['imagingRoiGroup']['rois'])
obj_res = md['SI']['objectiveResolution']  # deg
framerate = md['SI']['hRoiManager']['scanFrameRate']
framerate_str = '{:06.3f}'.format(framerate).replace('.', 'p')
strip_size_px = np.array(md['json']['RoiGroups']['imagingRoiGroup']['rois'][0]['scanfields']['pixelResolutionXY'])
strip_w_px = strip_size_px[0]
strip_h_px = strip_size_px[1]
strip_size_deg = np.array(md['json']['RoiGroups']['imagingRoiGroup']['rois'][0]['scanfields']['sizeXY'])
strip_w_deg = strip_size_deg[0]
strip_h_deg = strip_size_deg[1]
#px_ratio = strip_size_px / strip_size_deg  # px/deg
res_w_umppx = obj_res / (strip_w_px / strip_w_deg)
res_h_umppx = obj_res / (strip_h_px / strip_h_deg)
res_w_str = '{:03.2f}'.format(res_w_umppx).replace('.', 'p')
res_h_str = '{:03.2f}'.format(res_h_umppx).replace('.', 'p')
roi_size_px = strip_size_px * np.array([n_strips, 1])
roi_w_px = strip_w_px * n_strips
roi_h_px = strip_h_px
roi_w_um = roi_w_px * res_w_umppx
roi_h_um = roi_h_px * res_h_umppx

start = md['frame0desc']['epoch'] - timedelta(seconds=framerate)
start_str = start.strftime('%H%M%StUTC')

fntxt_fov = 'fov{:04d}x{:04d}um'.format(round(roi_w_um), round(roi_h_um))
fntxt_part = '{}_SP_depthum_{}_res{}x{}umpx_fr{}Hz_powmW'.format(start_str, fntxt_fov, res_w_str, res_h_str, framerate_str)

print('{} :'.format(os.path.split(os.path.dirname(source))[-1]))
print('{} >>>'.format(os.path.basename(source)))
print('{}_00001.tif'.format(fntxt_part))

