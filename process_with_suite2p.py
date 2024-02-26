#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from glob import glob
import numpy as np
import os
import suite2p
from warnings import warn

import metadata


# TODO: implement parameter sweeping options
# TODO Write a function to merge registered TIF files into a single file and also save as registered h5.


# Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    'source',
    help='Path to a ScanImage TIFF data file. [required]')
parser.add_argument(
    '-ps', '--preprocstr', type=str, default='*preprocd_olap00px*.h5',
    help='String contained in filename of preprocessed data file. [optional, default: \'*preprocd_olap00px*\']')
opts = parser.parse_args()

if os.path.isfile(opts.source):
    source = opts.source
    source_path = os.path.split(source)[0]
    source_base = os.path.basename(source)
    source_name = os.path.splitext(source_base)[0]
    source_ext = os.path.splitext(source_base)[1]
else:
    raise argparse.ArgumentTypeError('Source file does not exist ({}).'.format(opts.sourcefile))

preproc_str = opts.preprocstr

# Find preprocessed file.
# source_preproc = os.path.splitext(os.path.basename(source_name))[0] + '_preprocd_olap00px.h5'
# if not os.path.isfile(os.path.join(source_path, source_preproc)):
#     warn('Preprocessed file does not exist ({}).'.format(source_preproc))

preproc_list = [f for f in glob(os.path.join(source_path, preproc_str))
                if os.path.isfile(f)]

if len(preproc_list) > 0:
    pp_path = preproc_list[0]
    if len(preproc_list) > 1:
        warn('Found multiple matching preprocessed data files, using the first one: {}'.format(pp_path))
    if os.path.isfile(pp_path):
        source_preproc = pp_path
    else:
        raise RuntimeError('Could not find preprocessed data file.')
else:
    raise RuntimeError('Could not find preprocessed data file.')

# Load metadata.
amd = metadata.get_metadata(source)
md = metadata.extract_useful_metadata(amd)

# Initialize options without suite2p defaults.
ops = dict()
db = dict()
# Optionally import saved ops.
# ops = np.load('ops.npy', allow_pickle=True).item()

# Set input options specific to this data.
# db['data_path'] = [source_path]
db['save_path0'] = source_path
# db['save_folder'] is set at the end to label with important settings.
db['fast_disk'] = []  # Path for storing temporary binary, defaults to 'save_path0'.
db['subfolders'] = []
db['look_one_level_down'] = False
db['move_bin'] = False  # Move binary file from 'fast_disk' to 'save_folder'.
ppext = os.path.splitext(os.path.basename(source_preproc))[1]
if ppext == '.h5' or ppext == '.hdf5':
    db['h5py'] = [os.path.join(source_path, source_preproc)]
    db['h5py_key'] = 'data'
elif ppext == '.tif' or ppext == '.tiff':
    db['tiff_list'] = [source_preproc]
elif ppext == '.nwb':
    db['nwb_file'] = source_preproc
    # db['nwb_driver'] = ''
    # db['nwb_series'] = ''
db['ignore_flyback'] = []  # Planes to be ignored as flyback.
db['force_sktiff'] = False  # Force use of scikit-image for reading TIFFs.
db['bruker'] = False
db['bruker_bidirectional'] = False

# Set processing options specific to system.
db['multiplane_parallel'] = False  # Run parallel pipeline on server.

# Set options related to a variety of categories.
# - Imaging and indicator settings
db['nplanes'] = 1
db['nchannels'] = 1
db['functional_chan'] = 1
db['tau'] = 0.6
db['fs'] = md['framerate']
db['mesoscan'] = False  # Load json file containing mesoscope metadata.
# db['frames_include'] = -1  # Process only a subset of # frames.

# - Bidirectional phase offset settings
ops['do_bidiphase'] = True
# db['bidiphase'] = 0
# db['bidi_corrected'] = False  # Do bidirectional scan correction during registration?

# - Rigid registration settings
ops['do_registration'] = True
# db['align_by_chan'] = 1
ops['keep_movie_raw'] = False  # Save binary file of non-registered frames.
ops['delete_bin'] = True  # Delete binary file of registered frames.
ops['reg_tif'] = True  # Save registered image stacks.
ops['reg_tif_chan2'] = False
# ops['force_refImg'] = False  # Use refImg from path stored in saved ops.
ops['two_step_registration'] = False  # Run registration twice (for low SNR data), requires 'keep_movie_raw' to be True.
ops['nimg_init'] = 500  # Template image frame size.
if md['n_frames'] > 2000:
    batch_size = 2000
elif md['n_frames'] > 1000:
    batch_size = 1000
else:
    batch_size = md['n_frames']
ops['batch_size'] = batch_size
# ops['subpixel'] = 10  # Precision of subpixel registration (in 1/subpixel steps).
# ops['maxregshift'] = 0.1  # Max allowed rigid shift as a fraction of frame size.
# ops['smooth_sigma'] = 1.15  # Gaussian SD (px) for smoothing phase correlation between the template and frame.
# ops['smooth_sigma_time'] = 0  # Gaussian SD (frames) for smoothing phase correlation between the template and frame.
# ops['th_badframes'] = 1.0  # Set threshold for throwing out bad frames, with lower values excluding more frames.
# ops['norm_frames'] = True  # Normalize frames before shift detection.
# ops['pad_fft'] = False  # Pad image before running FFT.

# - Single-photon (1P) registration settings
ops['1Preg'] = False
# ops['spatial_hp_reg'] = 42  # Spatial high-pass filtering window before registration.
# ops['pre_smooth'] = 0  # Gaussian SD for smoothing before spatial high-pass filtering.
# ops['spatial_taper'] = 40  # Amount (px) to taper image edges before registration. Set > 3*ops[‘smooth_sigma’].

# - Non-rigid registration settings
ops['nonrigid'] = True
ops['block_size'] = [128, 128]  # Edge size (px) of blocks.
ops['snr_thresh'] = 1.2  # SNR threshold for phase correlation peak to noise. Smooths until above. Set 1 no smoothing.
ops['maxregshiftNR'] = 5.0  # Max non-rigid pixel shift relative to rigid result.

# - Cell detection settings
# Anatomical cell detection settings to use cellpose to detect ROIs (if anatomical_only > 0)
#     Options for anatomical_only are: 1 = max_proj / mean_img, 2 = mean_img, 3 = mean_img_enhanced, 4 = max_proj
ops['anatomical_only'] = 2
if ops['anatomical_only'] > 0:
    if md['fov']['neurondiameter_px'] is not None:
        # Set estimated cell diameter (px) for cellpose.
        ops['diameter'] = md['fov']['neurondiameter_px']
        print('Estimated diameter for cellpose anatomical ROI detection is {}px.'.format(md['fov']['neurondiameter_px']))
    else:
        # Set diameter to 0 for automatic estimation.
        ops['diameter'] = 0
    ops['cellprob_threshold'] = -3.5  # Threshold of input to sigmoid cell probability function, varying from -6 to 6.
    ops['flow_threshold'] = 2.0  # Maximum error of flows for each mask. Increase for more ROIs, decrease for fewer.
    # ops['spatial_hp_cp'] = 0  # Spatial high-pass filtering window size.
    # ops['pretrained_model'] = 'cyto'  # Path to pretrained model.
    # ops['chan2_thres']  # Threshold for detecting an ROI in channel 2.
# Functional cell detection settings
ops['roidetect'] = True
ops['sparse_mode'] = True  # Use 'sparse_mode' algorithm.
ops['denoise'] = False  # Denoise before cell detection in 'sparse_mode'.
# Documentation says 'smooth_masks' defaults to True, but it is not set in default_ops().
# ops['smooth_masks'] = True  # Smooth ROI masks in final pass of cell detection.
ops['connected'] = True  # Require ROI pixels to be fully connected.
# Check resolution to determine spatial scale.
#     Options for spatial_scale are: 0 = multi-scale, 1 = 6px, 2 = 12px, 3 = 24px, 4 = 48px
spatial_scales = {1: 6, 2: 12, 3: 24, 4: 48}
if ops['anatomical_only'] <= 0:
    if md['fov']['neurondiameter_px'] is not None:
        ssv = min(spatial_scales.values(), key=lambda x: abs(x - md['fov']['neurondiameter_px']))
        ssk = list(spatial_scales.keys())[list(spatial_scales.values()).index(ssv)]
        ops['spatial_scale'] = ssk
        print('Estimated diameter for functional ROI detection is {} pixels, '.format(md['fov']['neurondiameter_px']) +
              'using spatial scale {} ({}px).'.format(ssk, ssv))
    else:
        spatial_scales = None
        ops['spatial_scale'] = 0
ops['nbinned'] = 5000  # Max binned frames for cell detection, default 5000.
ops['max_iterations'] = 25  # 50
ops['threshold_scaling'] = 0.2  # Multiplier for ROI detection threshold. Lower values yield more ROIs. Default '1.0'.
ops['max_overlap'] = 0.9  # Allowed overlap proportion between ROIs. Default '0.75'.
ops['high_pass'] = 100  # Mean subtraction across time is performed with window of size ‘high_pass’ (frames?).
ops['spatial_hp_detect'] = 25.0  # Spatial high-pass window size for neuropil subtraction.

# - Neuropil extraction settings
ops['neuropil_extract'] = True
ops['allow_overlap'] = False  # Extract fluorescence signal from overlapping regions of ROIs.
ops['inner_neuropil_radius'] = 2  # Number of pixels to keep between ROI and neuropil donut. Default '2'.
ops['min_neuropil_pixels'] = 350  # Minimum number (px) used to compute neuropil.
ops['lam_percentile'] = 50.0  # Percentile of neuropil area to ignore when excluding cell ROIs. Default '50.0'.

# - Spike deconvolution settings
ops['spikedetect'] = True
# Method for computing baseline of teach fluorescence trace.
#     Options are: 'maximin', 'prctile', 'constant', 'constant_percentile'.
ops['baseline'] = 'maximin'
ops['win_baseline'] = 60  # Maximin filter window size (sec).
ops['sig_baseline'] = 10  # Gaussian filter width (sec) for filtering before baseline estimation.
ops['prctile_baseline'] = 8  # Percentile of trace to use as 'constant_percentile' baseline.
ops['neucoeff'] = 0.7  # Neuropil signal subtraction coefficient. Default '0.7'.

# - Cell classifier settings
ops['soma_crop'] = True  # Crop dendrites for cell classification stats like compactness.
# ops['use_builtin_classifier'] = False
# ops['classifier_path'] = ''

# - Output settings
ops['report_time'] = True  # Output processing time metrics for each plane in timing dictionary.
ops['save_nwb'] = False
ops['save_mat'] = False
if ops['roidetect'] and ops['anatomical_only'] <= 0 and ops['spatial_scale'] != 0:
    db['save_folder'] = 'suite2p_func{}px'.format(spatial_scales[ops['spatial_scale']])
elif ops['anatomical_only'] > 0:
    if ops['diameter'] > 0:
        d_str = '{}px'.format(ops['diameter'])
    else:
        d_str = '0'
    cpt_str = '{}'.format(ops['cellprob_threshold']).replace('.', 'p')
    ft_str = '{}'.format(ops['flow_threshold']).replace('.', 'p')
    db['save_folder'] = 'suite2p_cellpose{}_d{}_pt{}_ft{}'.format(ops['anatomical_only'], d_str, cpt_str, ft_str)
# ops['reg_file'] = os.path.join(db['save_path0'], db['save_folder'], 'plane0', 'data.bin')

# Run suite2p.
print('Running suite2p with save_folder: {}'.format(db['save_folder']))
s2pops = suite2p.default_ops()
runops = {**s2pops, **ops}
try:
    ops_out = suite2p.run_s2p(ops=runops, db=db)
    print(set(ops_out.keys()).difference(ops.keys()))
except ValueError as value_error:
    warn('suite2p failed to run properly: {}'.format(value_error))
except Exception as error:
    warn('suite2p failed to run properly: {}'.format(error))

# At least for hdf5 inputs, suite2p creates its default folder for converting image data to a binary file.
# Remove it if it exists and is empty.
default_suite2p_folder = 'suite2p'
default_suite2p_path = os.path.join(source_path, default_suite2p_folder)
if db['save_folder'] != default_suite2p_folder:
    if os.path.isdir(default_suite2p_path):
        contents = os.listdir(default_suite2p_path)
        for c in contents:
            cp = os.path.join(default_suite2p_path, c)
            if os.path.isdir(cp):
                subcontents = os.listdir(cp)
                if not subcontents:
                    os.rmdir(cp)
                else:
                    warn('Not removing suite2p sub-directory ({}), not empty.'.format(cp))
            else:
                warn('Not removing suite2p default directory ({}), not empty.'.format(default_suite2p_path))
                break
        contents = os.listdir(default_suite2p_path)
        if not contents:
            os.rmdir(default_suite2p_path)
        else:
            warn('Not removing suite2p default directory ({}), not empty.'.format(default_suite2p_path))

# Save options separately from suite2p.
save_folder_path = os.path.join(db['save_path0'], db['save_folder'])
save_opsin_path = os.path.join(save_folder_path, 'ops_in.npy')
save_opsnodb_path = os.path.join(save_folder_path, 'ops_in_nodb.npy')
save_opsdbonly_path = os.path.join(save_folder_path, 'ops_in_dbonly.npy')
save_opsout_path = os.path.join(save_folder_path, 'ops_out.npy')
ops_in = {**ops, **db}
np.save(save_opsin_path, ops_in)
np.save(save_opsnodb_path, ops)
np.save(save_opsdbonly_path, db)
np.save(save_opsout_path, ops_out)
