#!/usr/bin/env python3

import json
import numpy as np
import os
import re
from ScanImageTiffReader import ScanImageTiffReader
import suite2p
#import tifffile

import metadata # *** TODO *** make relative?

session_path = r'/Data/Cadbury/20221016d/161100tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow059p0mW'
file_name_raw = '161100tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow059p0mW_00001.tif'

file_path_raw = session_path + os.path.sep + file_name_raw
file_name_preproc = os.path.splitext(os.path.basename(file_name_raw))[0] + '_preprocessed.h5'
file_path_preproc = session_path + os.path.sep + file_name_preproc

def dict_generator(string, sep):
    if string.count('=') != 1:
        print('error: do not know how to handle strings with more than one equals sign')
    eqpos = string.find('=')
    if sep in string[eqpos+1:-1]:
        # if necessary, first replace sep in value with unit separator
        stringA, stringB = string.split('=')
        stringBnosep = stringB.replace(sep, chr(31))
        string = '='.join([stringA, stringBnosep])
    if sep not in string:
        k1s, vs = string.split('=')
        k1 = k1s.strip()
        # replace any instances of unit separator with sep
        v = vs.strip().replace(chr(31), sep)
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'nan':
            v = float('nan')
        elif v.lower() == 'none':
            v = None
        elif v.lower() == 'inf':
            v = float('inf')
        elif v.lower() == '-inf':
            v = float('-inf')
        #elif re.match("^[+-]?\d+.?\d*$", v) is not None: # handles +/-/. but not scientific notation
        #elif v.isnumeric(): # does not handle +/-/.
        #    v = float(v)
        #    if v.is_integer():
        #        v = int(v)
        elif re.match("^[+-]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?$", v) is not None: 
            v = float(v)
            if v.is_integer():
                v = int(v)
        elif '[' in v and ']' in v:
            if v == '[]':
                v = []
            else:
                bs = v.find('[')
                be = v.find(']', bs)
                v = np.fromstring(v[bs+1:be], dtype=float, sep=' ')
        return {k1: v}
    k1, k2 = string.split(sep, 1)
    return {k1: dict_generator(k2, sep)}

def merge_dicts(a, b):
    """
    from https://stackoverflow.com/a/56177639
    Merge two values, with `b` taking precedence over `a`.
    Semantics:
    - If either `a` or `b` is not a dictionary, `a` will be returned only if
    `b` is `None`. Otherwise `b` will be returned.
    - If both values are dictionaries, they are merged as follows:
    * Each key that is found only in `a` or only in `b` will be included in
    the output collection with its value intact.
    * For any key in common between `a` and `b`, the corresponding values
    will be merged with the same semantics.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a if b is None else b
    else:
        # If we're here, both a and b must be dictionaries or subtypes thereof.
        # Compute set of all keys in both dictionaries.
        keys = set(a.keys()) | set(b.keys())
        # Build output dictionary, merging recursively values with common keys,
        # where `None` is used to mean the absence of a value.
        return {key: merge_dicts(a.get(key), b.get(key)) for key in keys}

md_dict = {}
with ScanImageTiffReader(file_path_raw) as reader:
    print(reader.description(0))
    n_frames = reader.shape()[0]
    print('{} frames '.format(n_frames))
    md = reader.metadata()
    md_js = md.find('\n{')
    md_je = md.find('}\n', -1)
    md_nonjson = md[0:md_js]
    for line in md_nonjson.splitlines():
        md_dict = merge_dicts(md_dict, dict_generator(line, '.'))
    md_json_str = md[md_js+1:md_je]
    md_json = json.loads(md_json_str)

dur = n_frames * framerate  # sec
dur_min = dur / 60 / 60

n_strips = len(md_json['RoiGroups']['imagingRoiGroup']['rois'])
obj_res = md_dict['SI']['objectiveResolution']  # deg
framerate = md_dict['SI']['hRoiManager']['scanFrameRate']
framerate_str = '{:06.3f}'.format(framerate).replace('.', 'p')
strip_size_px = np.array(md_json['RoiGroups']['imagingRoiGroup']['rois'][0]['scanfields']['pixelResolutionXY'])
strip_w_px = strip_size_px[0]
strip_h_px = strip_size_px[1]
strip_size_deg = np.array(md_json['RoiGroups']['imagingRoiGroup']['rois'][0]['scanfields']['sizeXY'])
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

fntxt_fov = 'fov{0:04d}umx{0:04d}um'.format(round(roi_w_um), round(roi_h_um))
fntxt_part = '_{}_res{}x{}umpx_fr{}Hz'.format(fntxt_fov, res_w_str, res_h_str, framerate_str)

mtime = os.path.getmtime(file_path_raw)

#md_dict = {}
#for line in md.splitlines():
#    if line.find('SI.') == 0:
#        print(line)
#        md_dict.update(dict_generator(line, '.'))

#with tifffile.TiffFile(file_path) as tif:
#    tfmd = {}
#    for tag in tif.pages[0].tags.values():
#        tag_name, tag_value = tag.name, tag.value
#        tfmd[tag_name] = tag_value


# if loading from a common set of ops saved elsewhere...
# ops = np.load('ops.npy', allow_pickle=True).item()
ops = suite2p.default_ops()
db = {}

# input/output settings
db['data_path'] = [session_path]
ppext = os.path.splitext(os.path.basename(file_name_preproc))[1]
if ppext == '.h5' or ppext == '.hdf5':
    db['h5py'] = file_name_preproc
    db['h5py_key'] = 'data'
if ppext == '.tif' or ppext == '.tiff':
    db['tiff_list'] = [file_name_preproc]
db['save_path0'] = session_path
db['save_folder'] = 'suite2p_testcmdline'
#ops['mesoscan'] = False  # look for json containing ScanImage mROI information?

# imaging and indicator settings
ops['nplanes'] = 1 
ops['nchannels'] = 1
ops['functional_chan'] = 1
ops['tau'] = 0.6 
ops['fs'] = 6.36368

# bidirectional phase offset settings
ops['do_bidiphase'] = True
#ops['bidi_corrected'] = False  # do bidirectional correction during registration?

# rigid registration settings
ops['do_registration'] = True
ops['two_step_registration'] = False
ops['nimg_init'] = 300  # frames subsampled reference image
ops['batch_size'] = 1000 #500
#ops['maxregshift'] = 
#ops['smooth_sigma_time'] = 
#ops['smooth_sigma'] = 
ops['reg_tif'] = True  # save registered tiffs

# non-rigid registration settings
ops['nonrigid'] = True
ops['block_size'] = [128, 128]
ops['snr_thresh'] = 1.2  # if non-rigid block is below threshold, smooth it until above, set to 1.0 for no smoothing
ops['maxregshiftNR'] = 5.0  # max non-rigid pixel shift relative to rigid

# functional cell detection settings
ops['roidetect'] = True
ops['spikedetect'] = True
ops['sparse_mode'] = True
ops['spatial_scale'] = 2  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
ops['connected'] = True
ops['nbinned'] = 5000  # max binned frames for cell detection
ops['max_iterations'] = 25
ops['threshold_scaling'] = 0.2  # adjust automatically determined threshold by this multiplier
ops['max_overlap'] = 0.9  # ROIs with greater overlap get removed during triage, before refinement
ops['high_pass'] = 100
ops['spatial_hp_detect'] = 25.0  # window for spatial high-pass filtering for neuropil subtraction before detection
ops['denoise'] = False

# anatomical cell detection settings (only used if anatomical_only > 0)
ops['anatomical_only'] = 0  # get ROIs from cellpose via 1: max_proj / mean_img ; 2: mean_img ; 3: mean_img enhanced ; 4: max_proj
ops['diameter'] = 0  # diameter for cellpose, automatically estimates if 0
#ops['cellprob_threshold'] = 
#ops['flow_threshold'] = 
#ops['spatial_hp_cp'] = 
#ops['pretrained_model'] = 'cyto'

# classification settings
ops['soma_crop'] = True  # crop dendrites for cell classification stats like compactness

# ROI extraction settings
ops['neuropil_extract'] = True
ops['inner_neuropil_radius'] = 2  # number of pixels to keep between ROI and neuropil donut
ops['min_neuropil_pixels'] = 350
ops['lam_percentile'] = 50.0
ops['allow_overlap'] = False  # px overlapping another ROI are thrown out (False) or included in both (True)

# deconvolution settings
#ops['baseline'] = 'maximin'  # or 'prctile'
#ops['win_baseline'] = 60
#ops['sig_baseline'] = 10
#ops['prctile_baseline'] = 8
#ops['neucoeff'] = 0.7


# Run the pipeline
output_ops = suite2p.run_s2p(ops=ops, db=db)
print(set(output_ops.keys()).difference(ops.keys()))

