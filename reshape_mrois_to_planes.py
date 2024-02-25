#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import json
import magic
import numpy as np
import os
from ScanImageTiffReader import ScanImageTiffReader
from warnings import warn

import metadata


def is_tiff(filepath: str) -> bool:
    allowed_types = ['image/tiff', 'image/tif']
    if magic.from_file(filepath, mime=True) not in allowed_types:
        return False
    return True


def json_serializer(obj):
    from datetime import date, datetime, time
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    else:
        # warn('Type {} not serializable.'.format(type(obj)))
        return str(obj)


# Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    'source',
    help='Path to a ScanImage TIFF data file. [required]')
parser.add_argument(
    '-o', '--overlap', type=int, default=0,
    help='Overlapping pixels between MROIs. [optional, default: 0]')
opts = parser.parse_args()

if os.path.isfile(opts.source):
    source = opts.source
    source_path = os.path.split(source)[0]
    source_base = os.path.basename(source)
    source_name = os.path.splitext(source_base)[0]
    source_ext = os.path.splitext(source_base)[1]
else:
    raise argparse.ArgumentTypeError('Source file does not exist ({}).'.format(opts.sourcefile))

overlap_px = opts.overlap

if not is_tiff(source):
    raise RuntimeError('Source file must be a TIFF stack.')


p = dict()
p['save'] = dict()
p['save']['hdf5'] = True
p['save']['tif'] = True
p['save']['metadata'] = True
p['save']['mean'] = True
p['save']['video'] = True
p['overwrite_warn'] = False

simd = metadata.get_metadata(source)
md = metadata.extract_useful_metadata(simd)

if md['mrois']['overlap'] is not False or md['mrois']['overlap_px'] is not None:
    warn('Overlap between MROIs may require calculation, which is not yet supported.')
    # raise Exception('Handling overlapping MROIs is not yet implemented.')
if md['n_planes'] != 1:
    RuntimeError('Handing multi-plane data is not yet implemented.')

data = ScanImageTiffReader(source).data()
data = np.expand_dims(data, 1)
data = np.swapaxes(data, 1, 3)

# Divide long acquisition strip into MROI chunks.
planes_mrois = np.empty((md['n_planes'], md['n_mrois']), dtype=np.ndarray)
for i_plane in range(md['n_planes']):
    y_start = 0
    for i_mroi in range(md['n_mrois']):
        y_start, y_end = md['mrois']['lrsort'][i_mroi]['acqstrip_ys_px']
        planes_mrois[i_plane, i_mroi] = data[:, :, y_start:y_end, i_plane]

# Clear original image data from memory.
del data

# Generate volume from MROIs.
if type(overlap_px) is not int:
    overlap_px = int(overlap_px)
n_f = md['n_frames']
n_x = md['fov']['w_px'] - (overlap_px * (md['n_mrois'] - 1))
n_y = md['fov']['h_px']  # [len(md['fov']['positions_deg'][a]) for a in range(2)]
n_z = md['n_planes']
mroi_sizes_px = np.array([r['size_px'] for r in md['mrois']['lrsort']], dtype=int)
mroi_corners_tl_px = np.array([r['corner_tl_px'] for r in md['mrois']['lrsort']], dtype=int)

volume = np.full((n_f, n_x, n_y, n_z), np.nan, dtype=np.float32)
# volume = np.empty((n_f, n_x, n_y, n_z), dtype=np.int16)
# print('volume shape: {}'.format(volume.shape))

for i_plane in range(n_z):
    plane_w = n_x
    # plane_w = n_x - (overlap_px * (md['n_mrois'] - 1))
    plane_h = n_y
    # canvas = np.zeros((n_f, plane_w, plane_h), dtype=np.float32)
    canvas = np.full((n_f, plane_w, plane_h), np.nan, dtype=np.float32)
    # print('canvas shape: {}'.format(canvas.shape))
    xe_canv = plane_w
    for i_mroi in range(md['n_mrois']):
        if i_mroi == 0:
            xs_canv = 0
            xe_canv = xs_canv + mroi_sizes_px[i_mroi, 0] - int(np.floor(overlap_px / 2))
            xs_mroi = xs_canv
            xe_mroi = xe_canv
        elif i_mroi != md['n_mrois'] - 1:
            xs_canv = xe_canv
            xe_canv = xs_canv + mroi_sizes_px[i_mroi, 0] - overlap_px
            x_mroi_width = mroi_sizes_px[i_mroi][0] - overlap_px
            xs_mroi = int(np.ceil(overlap_px / 2))
            xe_mroi = xs_mroi + x_mroi_width
        else:
            xs_canv = xe_canv
            xe_canv = plane_w
            xs_mroi = np.ceil(overlap_px / 2).astype(int)
            xe_mroi = mroi_sizes_px[i_mroi][0]
        # print(xs_canv, xe_canv, xs_mroi, xe_mroi)

        ys_canv = mroi_corners_tl_px[i_mroi, 1]
        ye_canv = ys_canv + mroi_sizes_px[i_mroi, 1]

        canvas[:, xs_canv:xe_canv, ys_canv:ye_canv] = planes_mrois[i_plane, i_mroi][:, xs_mroi:xe_mroi]

    # shift_x_varied_seams = int(round((overlap_seams_this_plane - min(overlaps_planes)) * (n_mrois - 1) / 2))
    shift_x = 0  # accumulated_shifts[i_plane, 0] + shift_x_varied_seams
    shift_y = 0  # accumulated_shifts[i_plane, 1]

    end_x = shift_x + canvas.shape[1]
    end_y = shift_y + canvas.shape[2]

    volume[:, shift_x:end_x, shift_y:end_y, i_plane] = canvas

if np.any(np.isnan(volume)):
    raise Exception('NaNs found in preprocessed volume.')

if volume.dtype != np.int16:
    volume = volume.astype(np.int16)
volume = np.squeeze(np.swapaxes(volume, 1, 2))

md_str = json.dumps(md, default=json_serializer)
sp = source_path + os.path.sep

if p['save']['hdf5']:
    save_path_h5 = sp + source_name + '_preprocd_olap{:02d}px.h5'.format(overlap_px)
    if os.path.isfile(save_path_h5) and p['overwrite_warn']:
        warn('Preprocessed HDF5 ouput already exists, overwriting ({}).'.format(save_path_h5))
    h5f = h5py.File(save_path_h5, 'w')
    h5f.create_dataset('data', data=volume)
    h5f.attrs.create('metadata', str(md_str))
    h5f.close()
    del h5f

if p['save']['tif']:
    save_path_tif = sp + source_name + '_preprocd_olap{:02d}px.tif'.format(overlap_px)
    if os.path.isfile(save_path_tif) and p['overwrite_warn']:
        warn('Preprocessed TIF ouput already exists, overwriting ({}).'.format(save_path_tif))

    import tifffile

    tifffile.imwrite(save_path_tif, volume, description=md_str)

if p['save']['metadata']:
    import pickle

    save_path_mdp = sp + source_name + '_metadata.pickle'
    if os.path.isfile(save_path_mdp) and p['overwrite_warn']:
        warn('Metadata file already exists, overwriting ({}).'.format(save_path_mdp))
    with open(save_path_mdp, 'wb') as mdpf:
        pickle.dump(md, mdpf)

    save_path_mdj = sp + source_name + '_metadata.json'
    if os.path.isfile(save_path_mdj) and p['overwrite_warn']:
        warn('Metadata file already exists, overwriting ({}).'.format(save_path_mdj))
    with open(save_path_mdj, 'w') as mdjf:
        json.dump(md, mdjf, indent=4, sort_keys=True, default=json_serializer)

if p['save']['mean']:
    save_path_mean = sp + source_name + '_preprocd_olap{:02d}px_mean.png'.format(overlap_px)

    from cv2 import imwrite
    from skimage.exposure import rescale_intensity
    from skimage.io import imsave
    from skimage.util import img_as_ubyte

    volume_mean = np.mean(volume, axis=0)
    pl, ph = np.percentile(volume_mean, [1, 99.9])
    volume_mean_rescale = img_as_ubyte(rescale_intensity(volume_mean, in_range=(pl, ph)))

    if os.path.isfile(save_path_mean) and p['overwrite_warn']:
        warn('Preprocessed mean image already exists, overwriting ({}).'.format(save_path_mean))
    imwrite(save_path_mean, volume_mean_rescale)

    volume_mean_rescale_median = np.median(volume_mean_rescale)
    if volume_mean_rescale_median < 20:
        save_path_mean_rescaled = sp + source_name + '_preprocd_olap{:02d}px_mean_rescaled.png'.format(overlap_px)

        while volume_mean_rescale_median < 20:
            print('Rescaling mean image.  Current mean: {}'.format(np.mean(volume_mean_rescale)))
            pl, ph = np.percentile(volume_mean_rescale, [1, 99.0])
            volume_mean_rescale = img_as_ubyte(rescale_intensity(volume_mean_rescale, in_range=(pl, ph)))
            volume_mean_rescale_median = np.median(volume_mean_rescale)

        if os.path.isfile(save_path_mean_rescaled) and p['overwrite_warn']:
            warn('Preprocessed mean rescaled image already exists, overwriting ({}).'.format(save_path_mean_rescaled))
        imwrite(save_path_mean_rescaled, volume_mean_rescale)

if p['save']['video']:
    save_path_video = sp + source_name + '_preprocd_olap{:02d}px_clip.mp4'.format(overlap_px)
    if os.path.isfile(save_path_video) and p['overwrite_warn']:
        warn('Video clip already exists, overwriting ({}).'.format(save_path_video))

    from cv2 import VideoWriter_fourcc, VideoWriter
    from skimage.exposure import rescale_intensity
    from skimage.io import imsave
    from skimage.transform import rescale
    from skimage.util import img_as_ubyte

    def write_video(filepath, stack, framerate=20.0):
        # Not sure what the actual limits are.
        video_max_w = 2 ** 13  # 4096
        video_max_h = 2 ** 13  # 2160
        (z, h, w) = stack.shape[0:3]
        ratio_w = w / video_max_w
        ratio_h = h / video_max_h
        ratio_max = max(ratio_w, ratio_h)
        if ratio_max > 1:
            warn('Frame size exceeds the codec maximum. The video output will be downsampled.')
            dw = round(w / ratio_max)
            dh = round(h / ratio_max)
            print('resizing to w={} h={}'.format(dw, dh))
            dstack = np.empty([z, dh, dw], dtype=np.float64)
            dstack.fill(np.nan)
            for dz in range(z):
                dstack[dz] = rescale(stack[dz], (1 / ratio_max), anti_aliasing=True)
            stack = dstack
        (z, h, w) = stack.shape[0:3]
        if stack.dtype != 'uint8':
            stack = rescale_intensity(stack, out_range=(0, 1))
            stack = img_as_ubyte(stack)
        codec = VideoWriter_fourcc(*'mp4v')
        video_writer = VideoWriter(filepath, codec, framerate, (w, h), isColor=False)
        for f in range(z):
            video_writer.write(stack[f])
        video_writer.release()

    n_f = np.min([np.ceil(30 * md['framerate']).astype(int), volume.shape[0]])
    v = volume[0:n_f]
    pl, ph = np.percentile(v, [1, 99.9])
    v = rescale_intensity(v, in_range=(pl, ph))
    write_video(save_path_video, v, framerate=md['framerate'])
