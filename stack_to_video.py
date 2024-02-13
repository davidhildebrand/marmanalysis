#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Grant Colburn Hildebrand
"""

# *** TODO create option for saving moving mean
# *** TODO save moving mean output!
# *** TODO allow dynamic selection of averaging mode
# *** TODO allow dynamic selection of rescaling and params
# *** TODO search for framerate in metadata?

import argparse
import cv2
import json
import math
import numpy as np
import os
import skimage
from ScanImageTiffReader import ScanImageTiffReader
import tifffile
import time
from warnings import warn


def directory(path):
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    if path is not None and not os.path.isdir(path):
        m = 'path is not a directory (%s)'
        raise argparse.ArgumentTypeError(m)
    return path


def write_video(filepath, stack, framerate=20.0):
    # Not sure what the actual limits are.
    video_max_w = 2 ** 13  # 4096
    video_max_h = 2 ** 13  # 2160
    (z, h, w) = stack.shape[0:3]
    ratio_w = w / video_max_w
    ratio_h = h / video_max_h
    ratio_max = max(ratio_w, ratio_h)
    if ratio_max > 1:
        m = 'Frame size exceeds the maximum allowed by the codec. ' + \
            'The video output will be downsampled.'
        warn(m)
        dw = round(w / ratio_max)
        dh = round(h / ratio_max)
        print('resizing to w={} h={}'.format(dw, dh))
        dstack = np.empty([z, dh, dw], dtype=np.float64)
        dstack.fill(np.nan)
        for dz in range(z):
            dstack[dz] = skimage.transform.rescale(stack[dz], (1 / ratio_max), anti_aliasing=True)
        stack = dstack
    (z,h,w) = stack.shape[0:3]
    if stack.dtype != 'uint8':
        stack = skimage.exposure.rescale_intensity(stack, out_range=(0, 1))
        stack = skimage.util.img_as_ubyte(stack)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filepath, codec, framerate, (w,h), isColor=False)
    for f in range(z):
        video_writer.write(stack[f])
    video_writer.release()


def save_stack(save_path, save_stack, overwrite=False):
    if not os.path.isfile(save_path):
        print('  Saving {}... '.format(save_path), end='')
        tifffile.imwrite(save_path, save_stack)
        print('done.')
    elif os.path.isfile(save_path) and overwrite:
        print('  Overwriting {}... '.format(gZps_path), end='')
        tifffile.imwrite(save_path, save_stack)
        print('done.')
    elif os.path.isfile(save_path) and not overwrite:
        print('  Not saving {}, file already exists.'.format(save_path))


def save_video(save_path, save_stack, framerate=20.0, overwrite=False):
    if not os.path.isfile(save_path):
        print('  Saving {}... '.format(save_path), end='')
        write_video(save_path, save_stack, framerate=framerate)
        print('done.')
    elif os.path.isfile(save_path) and overwrite:
        print('  Overwriting {}... '.format(save_path), end='')
        write_video(save_path, save_stack, framerate=framerate)
        print('done.')
    elif os.path.isfile(save_path) and not overwrite:
        print('  Not saving {}, file already exists.'.format(save_path))


#%% Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    'sourcefile',
    help='Path to a source image data file. [required]')
parser.add_argument(
    '-d', '--dest', required=False, type=directory, default=None,
    help='Output directory path. [optional, default: source path]')
parser.add_argument(
    '-gp', '--groupproj', action='store_true',
    help='Perform an mean grouped projection in which every <windowsize> ' + \
         'frames are averaged into a single frame. [optional, default: true]')
parser.add_argument(
    '-sw', '--slidewin', action='store_true',
    help='Perform a central moving average (mean) over a sliding window ' + \
         'that incorporates <windowsize>/2 frames on either side of the ' + \
         'center frame (except at the stack edges). Preserves original ' + \
         'stack size. [optional, default: true]')
parser.add_argument(
    '-v', '--videos', action='store_true',
    help='Save outputs as compressed videos. [optional, default: False]')
parser.add_argument(
    '-s', '--stacks', action='store_true',
    help='Save outputs as uncompressed image stacks. [optional, default: False]')
parser.add_argument(
    '-ws', '--windowsize', type=int, default=10,
    help='Window size for grouping or averaging. [optional, default: 10]' + \
         '  Note that for sliding window averages, windowsize/2 frames on' + \
         '  either side of the (included) centered frame are processed.' + \
         '  This implies that each resulting frame with windowsize=10 is' + \
         '  an actually an average of 11 frames.')
parser.add_argument(
    '-f', '--frames', type=int, default=100,
    help='Frames to process from the original image stack. [optional, default: 100]')
parser.add_argument(
    '-fr', '--framerate', type=float, default=None,
    help='Framerate for generated video output. [optional, default: 20.0]')
parser.add_argument(
    '-ip', '--intenspct', type=float, default=[1.0, 99.98], nargs=2,
    help='Intensity min and max cutoff percentiles for rescaling. [optional, ' + \
         'default: 1.0 99.98]')
parser.add_argument(
    '-o', '--overwrite', action='store_true',
    help='Overwrite existing output files. [optional, default: false]')
opts = parser.parse_args()

if os.path.isfile(opts.sourcefile):
    source = opts.sourcefile
    source_path = os.path.split(source)[0] + os.path.sep
    source_name = os.path.splitext(os.path.split(source)[1])[0]
    source_ext = os.path.splitext(os.path.split(source)[1])[1]
else:
    m = 'Source file does not exist ({}).'.format(opts.sourcefile)
    raise argparse.ArgumentTypeError(m)
if source_ext != '.tif' and source_ext != '.tiff':
    m = 'Source file must be a TIFF stack (with extension .tif or .tiff).'
    raise RuntimeError(m)

if opts.dest is not None:
    dest_path = opts.dest
else:
    dest_path = source_path

grouping = opts.groupproj
sliding = opts.slidewin
window_size = opts.windowsize
save_videos = opts.videos
save_stacks = opts.stacks
overwrite = opts.overwrite
n_frames = opts.frames
framerate = opts.framerate
intens_pctiles = tuple(opts.intenspct)

if not grouping and not sliding:
    m = 'No processing type (grouped projection or sliding window) was specified.'
    raise RuntimeError(m)

if not save_videos and not save_stacks:
    m = 'No output type (videos or stacks) was specified.'
    raise RuntimeError(m)

if framerate is not None:
    if not save_videos:
        warn('Without the videos output argument (-v/--videos), ' + \
             'the framerate argument is ignored.')
        framerate = None


#%%

print('Processing {}{}{}'.format(source_path, source_name, source_ext))
print('  Using a window size of {}.'.format(window_size))
if save_videos:
    print('  Configured to save video outputs.')
    print('    Framerate set to {} fps.'.format(framerate))
    if intens_pctiles:
        print('    Intensity percentile range set to {}.'.format(intens_pctiles))
if save_stacks:
    print('  Configured to save image stack outputs.')

print('  Loading full image stack... ', end='')
metadata = ScanImageTiffReader(source).metadata()
if metadata != '':
    metadata = metadata[metadata.find('\n{'):]
    metadata = json.loads(metadata)
    # *** TODO search for framerate in metadata?
ts = time.time()
data = ScanImageTiffReader(source).data()
te = time.time()
print('done.  Took {:.02f} sec.'.format(te - ts))

print('  Separating subset of image frames for processing... ', end='')
data_subset = data[0:n_frames]
frames = skimage.util.img_as_float64(data_subset)
print('done.')

if grouping:
    print('  Generating grouped projection stack with a window size of ' + \
          '{} frames... '.format(window_size), end='')
    bins = math.floor(frames.shape[0] / window_size)
    groupZproj = np.empty([bins, frames.shape[1], frames.shape[2]], dtype=frames.dtype)
    groupZproj.fill(np.nan)
    for idx in range(bins):
        idx_rng = range(idx * window_size, (idx * window_size) + window_size)
        groupZproj[idx,:,:] = np.mean(frames[idx_rng], axis=0, dtype=frames.dtype)
    groupZproj_int = skimage.util.img_as_int(groupZproj)
    print('done.')
    
    temp_path = source_path + source_name + '_frames0to{}'.format(n_frames) + \
                '_groupZavg{}'.format(window_size)
    if save_stacks:
        gZps_path = temp_path + source_ext
        save_stack(gZps_path, groupZproj_int, overwrite=overwrite)
    if save_videos:
        gZpm_path = temp_path + '.mp4'
        save_video(gZpm_path, groupZproj_int, framerate=framerate, 
                   overwrite=overwrite)
    
    print('  Rescaling grouped projection stack intensity... ', end='')
    gZp_low, gZp_high = np.percentile(groupZproj, intens_pctiles)
    groupZproj_rescale  = skimage.exposure.rescale_intensity(groupZproj, 
                                                             in_range=(gZp_low, gZp_high))
    groupZproj_rescale_int = skimage.util.img_as_int(groupZproj_rescale)
    print('done.')
    
    temp_path = source_path + source_name + '_frames0to{}'.format(n_frames) + \
                '_groupZavg{}'.format(window_size) + \
                '_rescale{:.2f}t{:.2f}'.format(intens_pctiles[0], intens_pctiles[1]).replace('.', 'p')
    if save_stacks:
        gZpsr_path = temp_path + source_ext
        save_stack(gZpsr_path, groupZproj_rescale_int, overwrite=overwrite)
    if save_videos:
        gZpmr_path = temp_path + '.mp4'
        save_video(gZpmr_path, groupZproj_rescale_int, framerate=framerate,
                   overwrite=overwrite)

if sliding:
    print('  Generating sliding window stack with a window size of ' + \
          '{} frames... '.format(window_size), end='')
    frames_movmean = np.empty(frames.shape, dtype=np.float64)
    frames_movmean.fill(np.nan)
    for idx in range(frames.shape[0]):
        # The first and last window/2 frames of the sliding mean average
        # are the same.
        # All others include the current frame ±(window_size / 2).
        if idx < math.ceil(window_size / 2):
            idx_rng = range(window_size)
        elif idx > (frames.shape[0] - math.floor(window_size / 2) - 1):
            idx_rng = range(frames.shape[0] - window_size, frames.shape[0])
        else:
            idx_rng = range(idx - math.floor(window_size / 2), idx + math.ceil(window_size / 2))
        frames_movmean[idx,:,:] = np.mean(frames[idx_rng], axis=0, dtype=float)
    frames_movmean_int = skimage.util.img_as_int(frames_movmean)
    print('done.')
    
    temp_path = source_path + source_name + '_frames0to{}'.format(n_frames) + \
                '_slidewin{}'.format(window_size)
    if save_stacks:
        sws_path = temp_path + source_ext
        save_stack(sws_path, frames_movmean_int, overwrite=overwrite)
    if save_videos:
        swm_path = temp_path + '.mp4'
        save_video(swm_path, frames_movmean_int, framerate=framerate,
                   overwrite=overwrite)
    
    print('  Rescaling sliding window stack intensity... ', end='')
    sw_low, sw_high = np.percentile(frames_movmean, intens_pctiles)
    sw_rescale = skimage.exposure.rescale_intensity(frames_movmean, 
                                                    in_range=(sw_low, sw_high))
    sw_rescale_int = skimage.util.img_as_int(sw_rescale)
    print('done.')
    
    temp_path = source_path + source_name + '_frames0to{}'.format(n_frames) + \
                '_slidewin{}'.format(window_size) + \
                '_rescale{:.2f}t{:.2f}'.format(intens_pctiles[0], intens_pctiles[1]).replace('.', 'p')
    if save_stacks:
        swsr_path = temp_path + source_ext
        save_stack(swsr_path, sw_rescale_int, overwrite=overwrite)
    if save_videos:
        swmr_path = temp_path + '.mp4'
        save_video(swmr_path, sw_rescale_int, framerate=framerate,
                   overwrite=overwrite)
