#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import json
import numpy as np
import os
from ScanImageTiffReader import ScanImageTiffReader
from warnings import warn

import metadata
from fileio import is_tiff


def json_serializer(obj):
    from datetime import date, datetime, time
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    else:
        # warn('Type {} not serializable.'.format(type(obj)))
        return str(obj)


# Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('source',
                    help='Path to a ScanImage TIFF data file. [required]')
parser.add_argument('-o', '--overlap', type=int, default=0,
                    help='Overlapping pixels between MROIs. [optional, default: 0]')
parser.add_argument('-fv', '--fill_value', type=int, default=0,
                    help='Intensity value with which to fill empty regions. [optional, default: 0, type: uint16]')
parser.add_argument('--undistort', action='store_true',
                    help='Undo the resonant-axis sinusoid, the odd/even line offset and the start-of-sweep ramp on each '
                         'strip first, then place the strips to a fraction of a pixel and combine their overlaps as '
                         '--overlap_mode says (see mroi_geometry.py). Writes *_preprocd_undist_<mode>.* and leaves the '
                         'session metadata files untouched. [optional, default: off = cut --overlap pixels and butt]')
parser.add_argument('--overlap_mode', default='crossfade', choices=['crossfade', 'split', 'average', 'left', 'right'],
                    help='With --undistort: how the two copies of the tissue in an overlap are combined. '
                         '[optional, default: crossfade]')
parser.add_argument('--seam', default='fit',
                    help="With --undistort: 'fit' measures every seam from the data, 'nominal' uses the header layout, "
                         "a number is an overlap in um for every seam. [optional, default: fit]")
parser.add_argument('--no_subpixel', action='store_true',
                    help='With --undistort: round seam positions and row offsets to whole pixels. [optional]')
parser.add_argument('--min_agreement', type=float, default=0.7,
                    help='With --undistort: lowest correlation between the two copies of an overlap before a blending '
                         'mode complains. [optional, default: 0.7]')
parser.add_argument('--on_disagreement', default='warn', choices=['warn', 'error', 'ignore'],
                    help='With --undistort: what to do when the two copies disagree. [optional, default: warn]')
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
fill_value = opts.fill_value


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

# TODO use metadata overlap if exists?
if md['mrois']['overlap'] and md['mrois']['overlap_px'] is None:
    warn('Overlap between MROIs may require calculation, which is not yet supported.')
if md['n_planes'] != 1:
    RuntimeError('Handing multi-plane data is not yet implemented.')

try:
    with ScanImageTiffReader(source) as tif:
        data = tif.data()  # initial order: frames, y/h/rows, x/w/cols, z/planes
except:
    warn('ScanImageTiffReader failed to load the data, trying with tifffile.')
    import tifffile
    data = tifffile.imread(source)  # initial order: frames, y/h/rows, x/w/cols, z/planes

if data.ndim == 4:
    # Assume data consists of multiple 3-D volumes sampled across time.
    data = np.swapaxes(data, 1, 2)  # yields: frames, x/w/cols, y/h/rows, z/planes
elif data.ndim == 3:
    # Assume data consists of a 2-D plane sampled across time.
    data = np.swapaxes(data, 1, 2)  # yields: frames, x/w/cols, y/h/rows
    data = np.expand_dims(data, 3)  # yields: frames, x/w/cols, y/h/rows, z/planes
elif data.ndim == 2:
    # Assume data consists of a 2-D plane sampled once.
    data = np.swapaxes(data, 0, 1)  # yields: x/w/cols, y/h/rows
    data = np.expand_dims(data, (0, 3))  # yields: frames, x/w/cols, y/h/rows, z/planes
else:
    raise Exception('Unsupported or misconfigured data dimensionality.')

# Divide long acquisition strip into MROI chunks.
planes_mrois = np.empty((md['n_planes'], md['n_mrois']), dtype=np.ndarray)
for i_plane in range(md['n_planes']):
    y_start = 0
    for i_mroi in range(md['n_mrois']):
        y_start, y_end = md['mrois']['lrsort'][i_mroi]['acqstrip_ys_px']
        planes_mrois[i_plane, i_mroi] = data[:, :, y_start:y_end, i_plane]

# Clear original image data from memory.
del data

if opts.undistort:
    # Undistort every strip, then stitch (mroi_geometry.py). Order: line offset, sinusoid and ramp first, then the
    # seams. Registration comes afterwards, in suite2p.
    import mroi_geometry
    if md['n_planes'] != 1:
        raise RuntimeError('--undistort handles single-plane data only so far.')
    strips = [np.swapaxes(planes_mrois[0, i_mroi], 1, 2) for i_mroi in range(md['n_mrois'])]  # views: frames, rows, px
    frame_step = max(1, md['n_frames'] // 300)
    mean_strips = np.stack([s[::frame_step].mean(0, dtype=np.float64) for s in strips]).astype(np.float32)
    seam = opts.seam if opts.seam in ('fit', 'nominal') else float(opts.seam)
    plan, stitch_info = mroi_geometry.plan_session(source_path, mean_strips, mode=opts.overlap_mode, seam=seam,
                                                   min_agreement=opts.min_agreement,
                                                   on_disagreement=opts.on_disagreement,
                                                   subpixel=not opts.no_subpixel)
    volume = np.swapaxes(mroi_geometry.stitch_movie(strips, plan), 1, 2)[..., np.newaxis]  # frames, x, y, z
    del planes_mrois, strips
    preproc_tag = 'undist_{}'.format(opts.overlap_mode)
    md['stitch'] = {'overlap_mode': opts.overlap_mode, 'seam': opts.seam, 'pixel_um': plan['pixel'],
                    'w_px': plan['n_canvas'], 'overlaps_um': plan['overlaps'].tolist(),
                    'row_shift_px': plan['row_shift'].tolist(), 'line_offset_px': plan['line_offset'],
                    'dark': plan['dark'], 'fill_fraction_temporal': plan['fill_fraction_temporal'],
                    'agreement': [a['r'] for a in stitch_info['agreement']]}
    p['save']['metadata'] = False  # the session metadata files describe the classic geometry: leave them alone
    with open(source_path + os.path.sep + source_name + '_preprocd_' + preproc_tag + '_stitch.json', 'w') as sjf:
        json.dump(md['stitch'], sjf, indent=4, sort_keys=True)
else:
    # Generate volume from MROIs.
    if type(overlap_px) is not int:
        overlap_px = int(overlap_px)
    n_f = md['n_frames']
    n_x = md['fov']['w_px'] - (overlap_px * (md['n_mrois'] - 1))
    n_y = md['fov']['h_px']  # [len(md['fov']['positions_deg'][a]) for a in range(2)]
    n_z = md['n_planes']
    mroi_sizes_px = np.array([r['size_px'] for r in md['mrois']['lrsort']], dtype=int)
    mroi_corners_tl_px = np.array([r['corner_tl_px'] for r in md['mrois']['lrsort']], dtype=int)
    # print('n_f={} n_x={} n_y={} n_z={}'.format(n_f, n_x, n_y, n_z))

    volume = np.full((n_f, n_x, n_y, n_z), np.nan, dtype=np.float32)
    # volume = np.empty((n_f, n_x, n_y, n_z), dtype=np.uint16)
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
            # print('xs_canv={} xe_canv={}'.format(xs_canv, xe_canv))

            ys_canv = mroi_corners_tl_px[i_mroi, 1]
            ye_canv = ys_canv + mroi_sizes_px[i_mroi, 1]

            canvas[:, xs_canv:xe_canv, ys_canv:ye_canv] = planes_mrois[i_plane, i_mroi][:, xs_mroi:xe_mroi]

        # shift_x_varied_seams = int(round((overlap_seams_this_plane - min(overlaps_planes)) * (n_mrois - 1) / 2))
        shift_x = 0  # accumulated_shifts[i_plane, 0] + shift_x_varied_seams
        shift_y = 0  # accumulated_shifts[i_plane, 1]

        end_x = shift_x + canvas.shape[1]
        end_y = shift_y + canvas.shape[2]

        volume[:, shift_x:end_x, shift_y:end_y, i_plane] = canvas
    del canvas, planes_mrois
    preproc_tag = 'olap{:02d}px'.format(overlap_px)

if np.any(np.isnan(volume)):
    warn('Empty regions (NaNs) found in processed volume, ' +
         'filling with pixels with intensity {} (uint16).'.format(fill_value))
    volume = np.nan_to_num(volume, nan=fill_value)

final_dtype = np.uint16
if np.any(volume < np.iinfo(final_dtype).min):
    warn('Intensity values below the minimum ({} < {}), '.format(np.min(volume), np.iinfo(final_dtype).min) +
         'adjusted by adding absolute value of minimum value to all pixels.')
    volume = volume + np.abs(np.min(volume))
if np.any(volume > np.iinfo(final_dtype).max):
    warn('Intensity values above maximum ({} > {}) clipped.'.format(np.max(volume), np.iinfo(final_dtype).max))
    volume = np.clip(volume, np.iinfo(final_dtype).min, np.iinfo(final_dtype).max)
if volume.dtype != final_dtype:
    volume = volume.astype(final_dtype)

volume = np.squeeze(np.swapaxes(volume, 1, 2))

md_str = json.dumps(md, default=json_serializer)
sp = source_path + os.path.sep

if p['save']['hdf5']:
    save_path_h5 = sp + source_name + '_preprocd_' + preproc_tag + '.h5'
    if os.path.isfile(save_path_h5) and p['overwrite_warn']:
        warn('Preprocessed HDF5 ouput already exists, overwriting ({}).'.format(save_path_h5))
    h5f = h5py.File(save_path_h5, 'w')
    h5f.create_dataset('data', data=volume)
    h5f.attrs.create('metadata', str(md_str))
    h5f.close()
    del h5f

if p['save']['tif']:
    save_path_tif = sp + source_name + '_preprocd_' + preproc_tag + '.tif'
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
    save_path_mean = sp + source_name + '_preprocd_' + preproc_tag + '_mean.png'

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

    volume_rescale_mean = volume_mean_rescale
    if np.median(volume_mean_rescale) < 20:
        save_path_mean_rescaled = sp + source_name + '_preprocd_' + preproc_tag + '_mean_rescaled.png'

        subvolume = volume[0:np.min([500, volume.shape[0]]).astype(int)]
        high = 100.0
        while np.median(volume_rescale_mean) < 20 and high > 0:
            high = high - 0.5
            print('Rescaling mean image. (median = {}, high = {})'.format(np.median(volume_mean_rescale), high))
            pl, ph = np.percentile(subvolume, [0, high])
            volume_rescale = img_as_ubyte(rescale_intensity(subvolume, in_range=(pl, ph)))
            volume_rescale_mean = np.mean(volume_rescale, axis=0)

        if os.path.isfile(save_path_mean_rescaled) and p['overwrite_warn']:
            warn('Preprocessed mean rescaled image already exists, overwriting ({}).'.format(save_path_mean_rescaled))
        imwrite(save_path_mean_rescaled, volume_rescale_mean)

if p['save']['video']:
    save_path_video = sp + source_name + '_preprocd_' + preproc_tag + '_clip.mp4'
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
