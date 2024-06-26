#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
import imagesize
import json
import numpy as np
import os
import re
from ScanImageTiffReader import ScanImageTiffReader
from warnings import warn
from zoneinfo import ZoneInfo


def metadata_line_to_dict(string, sep, tz='America/New_York'):
    if string.count('=') != 1:
        warn('Do not know how to handle strings with more than one equals sign.')
        print(string)
    equal_position = string.find('=')

    # Replace 'sep' instances in value with unit separator.
    if sep in string[equal_position+1:-1]:
        str_a, str_b = string.split('=')
        str_b_nosep = str_b.replace(sep, chr(31))
        string = '='.join([str_a, str_b_nosep])

    if sep not in string:
        k1s, vs = string.split('=')
        k1 = k1s.strip()

        # Replace unit separator instances with 'sep'.
        v = vs.strip().replace(chr(31), sep)

        # Import values from MATLAB ScanImage into python format.
        pattern_num = r'^[+-]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?$'
        pattern_datetime = r'^\[\ *[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\.[0-9]+\ *\]$'
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
        elif re.match(pattern_num, v) is not None:
            v = float(v)
            if v.is_integer():
                v = int(v)
        elif re.match(pattern_datetime, v) is not None:
            # e.g., [2023  8  4 14  2 8.937]
            v = ' '.join(v.split())
            dt = datetime.strptime(v, '[%Y %m %d %H %M %S.%f]').replace(tzinfo=ZoneInfo(tz))
            v = dt.astimezone(timezone.utc)
        elif '[' in v and ']' in v:
            if v == '[]':
                v = []
            else:
                bs = v.find('[')
                be = v.find(']', bs)
                v = ' '.join(v.split())
                v = np.fromstring(v[bs + 1:be], dtype=float, sep=' ')
        return {k1: v}
    k1, k2 = string.split(sep, 1)
    return {k1: metadata_line_to_dict(k2, sep)}


def merge_metadata_dicts(d1, d2):
    """
    Merge two values, with `b` taking precedence over `a`.
    modified from https://stackoverflow.com/a/56177639
    Semantics:
    - If either `a` or `b` is not a dictionary, `a` will be returned only if
    `b` is `None`, otherwise `b` will be returned.
    - If both values are dictionaries, they are merged as follows:
    * Each key that is found only in `a` or only in `b` will be included in
    the output collection with its value intact.
    * For any key in common between `a` and `b`, the corresponding values
    will be merged with the same semantics.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 if d2 is None else d2
    # Compute set of all keys in both dictionaries.
    keys = set(d1.keys()) | set(d2.keys())
    # Build output dictionary, merging recursively values with common keys,
    # where `None` is used to mean the absence of a value.
    return {key: merge_metadata_dicts(d1.get(key), d2.get(key)) for key in keys}


def parse_scanimage_desc(desc):
    desc_dict = dict()
    for line in desc.splitlines():
        if '=' not in line:
            continue
        desc_dict = merge_metadata_dicts(desc_dict,
                                         metadata_line_to_dict(line, '.'))
    return desc_dict


def get_mode_str(mode):
    if mode == 'max30':
        mode_str = 'Max30'
    elif mode == 'max15':
        mode_str = 'Max15'
    elif mode == 'sp':
        mode_str = 'SP'
    else:
        mode_str = 'unknown'
    return mode_str


def default_metadata() -> dict:
    dmd = dict()
    dmd['stim_locked_to_acqfr'] = False
    return dmd


def get_metadata(filepath):
    md_dict = dict()

    md_dict['acqstrip_w'], md_dict['acqstrip_h'] = imagesize.get(filepath)

    with (ScanImageTiffReader(filepath) as reader):
        fn = os.path.basename(filepath)

        # Attempt to determine acquisition time from filename.
        pattern_t0 = r'^.*([0-9]{6}tUTC).*$'
        if re.match(pattern_t0, fn) is not None:
            m = re.match(pattern_t0, fn)
        else:
            m = None
        if m is not None:
            g = m.groups()
            if len(g) > 1:
                warn('Found more than one start time in filename. Using the first.')
            start_time = g[0]
            pattern_t00 = r'^([0-9]{4}00tUTC).*$'
            if re.match(pattern_t00, fn) is not None:
                warn('Start time string in filename is suspect, will calculate.')
                start_time = None
        else:
            warn('Could not determine start time from filename.')
            start_time = None
        md_dict['start_time_str'] = start_time

        # Attempt to determine imaging mode and plane number filename.
        fnci = fn.lower()
        if 'max30' in fnci:
            n_planes = 30
            mode = 'max30'
        elif 'max15' in fnci:
            n_planes = 15
            mode = 'max15'
        elif 'sp' in fnci:
            n_planes = 1
            mode = 'sp'
        else:
            warn('Could not determine number of imaging planes from filename. ' +
                 'Assuming single plane (n_planes = 1).')
            n_planes = 1
            mode = 'sp'
        md_dict['n_planes'] = n_planes
        md_dict['mode'] = mode

        # Attempt to determine imaging depth (of deepest plane) from filename.
        pattern_d0 = r'^.*depth([0-9]+)um.*$'
        pattern_d1 = r'^.*[^0-9]([0-9]+)umdeep.*$'
        if re.match(pattern_d0, fn) is not None:
            m = re.match(pattern_d0, fn)
        elif re.match(pattern_d1, fn) is not None:
            m = re.match(pattern_d1, fn)
        else:
            m = None
        if m is not None:
            g = m.groups()
            if len(g) > 1:
                warn('Found more than one depth in filename. Using the first.')
            depth = g[0]
        else:
            warn('Could not determine depth imaging plane from filename.')
            depth = None
        if depth is not None:
            depth = float(depth)
            if depth.is_integer():
                depth = int(depth)
        md_dict['depth'] = depth

        # Attempt to determine laser power from filename.
        pattern_p0 = r'^.*pow([0-9]+p?[0-9]+?)mW.*$'
        pattern_p1 = r'^.*[^0-9]([0-9]+p?[0-9]+?)mW.*$'
        if re.match(pattern_p0, fn) is not None:
            m = re.match(pattern_p0, fn)
        elif re.match(pattern_p1, fn) is not None:
            m = re.match(pattern_p1, fn)
        else:
            m = None
        if m is not None:
            g = m.groups()
            if len(g) > 1:
                warn('Found more than one power in filename. Using the first.')
            power = g[0]
        else:
            warn('Could not determine power from filename. ')
            power = None
        if power is not None:
            if type(power) == str:
                if 'p' in power:
                    power = power.replace('p', '.')
            power = float(power)
            if power.is_integer():
                power = int(power)
        md_dict['power'] = power

        md_dict['n_frames'] = reader.shape()[0]
        desc = parse_scanimage_desc(reader.description(0))
        md_dict['frame0desc'] = desc

        md_raw = reader.metadata()
        md_json_start = md_raw.find('\n{')
        # md_json_end = md_raw.find('}\n', -2) + 1

        if md_json_start != -1:
            md_json_str = md_raw[md_json_start + 1:]
            md_json = json.loads(md_json_str)
            md_dict['json_str'] = md_json_str
            md_dict['json'] = md_json
            md_nonjson = md_raw[0:md_json_start]
        else:
            md_dict['json_str'] = None
            md_dict['json'] = None
            md_nonjson = md_raw

        for line in md_nonjson.splitlines():
            md_dict = merge_metadata_dicts(md_dict,
                                           metadata_line_to_dict(line, '.'))

        return md_dict


def roi_from_scanfield(scanfield):
    r = dict()
    r['center_deg'] = np.array(scanfield['centerXY'], dtype=float)
    r['size_deg'] = np.array(scanfield['sizeXY'], dtype=float)
    r['size_px'] = np.array(scanfield['pixelResolutionXY'], dtype=int)
    return r


def extract_useful_metadata(scanimage_metadata):
    # NOTE assumes data acquired before 20230505d have strip overlap and after do not
    simd = scanimage_metadata
    date_strip_overlap_fix = datetime(2023, 5, 5, tzinfo=ZoneInfo('America/New_York'))
    date_stim_lock_acqfr_inc = datetime(2023, 5, 5, tzinfo=ZoneInfo('America/New_York'))

    umd = dict()

    umd['n_planes'] = simd['n_planes']
    umd['mode'] = simd['mode']
    umd['mode_str'] = get_mode_str(umd['mode'])
    if simd['depth'] is not None:
        umd['depth'] = simd['depth']
        umd['depth_str'] = 'depth{:03}um'.format(umd['depth'])
    else:
        umd['depth'] = None
        umd['depth_str'] = 'depthZZZum'
    if simd['power'] is not None:
        umd['power'] = simd['power']
        umd['power_str'] = 'pow{:05.1f}mW'.format(umd['power']).replace('.', 'p')
    else:
        umd['power'] = None
        umd['power_str'] = 'powPPPpPmW'
    umd['n_planes'] = simd['n_planes']
    umd['n_frames'] = simd['n_frames']
    umd['objective_resolution'] = simd['SI']['objectiveResolution']  # um/deg
    umd['framerate'] = simd['SI']['hRoiManager']['scanFrameRate']
    umd['framerate_str'] = 'fr{:06.3f}Hz'.format(umd['framerate']).replace('.', 'p')
    umd['fill_fraction_temporal'] = simd['SI']['hScan2D']['fillFractionTemporal']
    umd['fill_fraction_spatial'] = simd['SI']['hScan2D']['fillFractionSpatial']
    umd['resonant_scanner_frequency'] = simd['SI']['hScan2D']['scannerFrequency']

    # Estimate start time.
    sdt = simd['frame0desc']['epoch'] - timedelta(seconds=umd['framerate'])
    if simd['start_time_str'] is not None:
        # Replace calculated start time with that from the filename.
        sd = sdt.date()
        st = datetime.strptime(simd['start_time_str'], '%H%M%StUTC').time()
        sdt = datetime.combine(sd, st).replace(tzinfo=timezone.utc)
    umd['start_datetime'] = sdt
    umd['start_datetime_str'] = umd['start_datetime'].strftime('%Y%m%dd%H%M%StUTC')
    umd['start_date_str'] = umd['start_datetime'].strftime('%Y%m%dd')
    umd['start_time_str'] = umd['start_datetime'].strftime('%H%M%StUTC')

    # Extract MROI information.
    umd['mrois'] = dict()
    if simd['json'] is not None:
        mrois_raw = simd['json']['RoiGroups']['imagingRoiGroup']['rois']
    else:
        mrois_raw = json.loads(simd["Artist"])['RoiGroups']['imagingRoiGroup']['rois']
    if type(mrois_raw) is not dict:
        mrois_orig = []
        for roi in mrois_raw:
            if type(roi['scanfields']) is not list:
                scanfield = roi['scanfields']
            else:
                scanfield = roi['scanfields'][np.where(np.array(roi['zs']) == 0)[0][0]]
            r = roi_from_scanfield(scanfield)
            mrois_orig.append(r)
    else:
        scanfield = mrois_raw['scanfields']
        r = roi_from_scanfield(scanfield)
        mrois_orig = [r]
    umd['mrois']['raw'] = mrois_raw
    umd['mrois']['orig'] = mrois_orig
    umd['n_mrois'] = len(mrois_orig)
    mroi_sizes_px = np.array([r['size_px'] for r in umd['mrois']['orig']], dtype=int)
    if np.all(np.isclose(mroi_sizes_px[:, 0], mroi_sizes_px[0, 0])):
        mroi_w_px = mroi_sizes_px[0, 0]
    else:
        raise Exception('Not all MROIs have the same width.')

    # Check that fill fractions are correctly related.
    spatial_from_temporal = np.cos((1 - umd['fill_fraction_temporal']) * np.pi/2)
    if not np.isclose(umd['fill_fraction_spatial'], spatial_from_temporal):
        warn('Fill fractions do not match expected relationship [spatial = cos((1-temporal) * pi/2)]. ' +
             'temporal: {:.3f}, '.format(umd['fill_fraction_temporal']) +
             'spatial: {:.3f}, '.format(umd['fill_fraction_spatial']) +
             'spatial calculated from temporal: {:.3f}'.format(spatial_from_temporal))

    # Extract acquisition strip information.
    umd['acqstrip'] = dict()
    umd['acqstrip']['w_px'] = simd['acqstrip_w']
    umd['acqstrip']['h_px'] = simd['acqstrip_h']
    umd['acqstrip']['size_px'] = np.array([umd['acqstrip']['w_px'], umd['acqstrip']['h_px']], dtype=int)
    umd['acqstrip']['flyback_h_px'] = (umd['acqstrip']['h_px'] - (mroi_sizes_px[:, 1].sum())) // (umd['n_mrois'] - 1)

    # Compare acquisition strip details to MROI details and conform to acquisition strip if necessary.
    acqstrip_hcalc_px = mroi_sizes_px[:, 1].sum() + ((umd['n_mrois'] - 1) * umd['acqstrip']['flyback_h_px'])
    if umd['acqstrip']['h_px'] != acqstrip_hcalc_px:
        warn('Acquisition strip height does not match expectation from MROI sizes.')
    if umd['acqstrip']['w_px'] != mroi_w_px:
        warn('Acquisition strip width does not match expectation from MROI widths. ' +
             'Forcing conformation to acquisition strip size.')
        for r in range(umd['n_mrois']):
            umd['mrois']['orig'][r]['size_px'][0] = umd['acqstrip']['w_px']
    del mroi_w_px

    # Calculate MROI resolutions.  Note that the objective resolution is in um/deg.
    for r in umd['mrois']['orig']:
        if type(umd['objective_resolution']) is float:
            r['resolution_umpx'] = umd['objective_resolution'] / (r['size_px'] / r['size_deg'])
            r['resolution_degpx'] = r['size_deg'] / r['size_px']
        else:
            r['resolution_umpx'] = None
            r['resolution_degpx'] = None

    # Sort MROIs by physical left-to-right position.
    # Keep un-sorted 'orig' version to recover data location in the acquisition strip.
    mroi_centers_deg = np.array([r['center_deg'] for r in umd['mrois']['orig']])
    mroi_centers_lrsort_arg = np.argsort(mroi_centers_deg[:, 0])
    mrois_lrsort = [umd['mrois']['orig'][a] for a in mroi_centers_lrsort_arg]
    mroi_centers_lrsort_deg = [mroi_centers_deg[a] for a in mroi_centers_lrsort_arg]
    umd['mrois']['lrsort_arg'] = mroi_centers_lrsort_arg
    umd['mrois']['lrsort'] = mrois_lrsort
    del mroi_centers_deg, mroi_centers_lrsort_arg, mrois_lrsort

    mroi_sizes_deg = np.array([r['size_deg'] for r in umd['mrois']['lrsort']])
    mroi_sizes_px = np.array([r['size_px'] for r in umd['mrois']['lrsort']], dtype=int)
    mroi_resolutions_umpx = np.array([r['resolution_umpx'] for r in umd['mrois']['lrsort']])
    mroi_resolutions_degpx = np.array([r['resolution_degpx'] for r in umd['mrois']['lrsort']])
    mroi_centers_deg = np.array([r['center_deg'] for r in umd['mrois']['lrsort']])

    if umd['start_datetime'] > date_strip_overlap_fix:
        umd['mrois']['overlap'] = False
        umd['mrois']['overlap_px'] = None
    else:
        umd['mrois']['overlap'] = True
        # TODO calculate overlap between mrois
        umd['mrois']['overlap_px'] = 'unknown'

    if umd['start_datetime'] > date_stim_lock_acqfr_inc:
        umd['stim_locked_to_acqfr'] = True
    else:
        umd['stim_locked_to_acqfr'] = False

    # Extract coordinates for MROI positions within long acquisition strip excluding flybacks.
    for i_plane in range(umd['n_planes']):
        y_start = 0
        for i_mroi in range(umd['n_mrois']):
            i_m = umd['mrois']['lrsort_arg'][i_mroi]
            y_end = y_start + mroi_sizes_px[i_mroi, 1]
            umd['mrois']['orig'][i_m]['acqstrip_ys_px'] = np.array([y_start, y_end], dtype=int)
            umd['mrois']['lrsort'][i_mroi]['acqstrip_ys_px'] = np.array([y_start, y_end], dtype=int)
            y_start += mroi_sizes_px[i_mroi, 1] + umd['acqstrip']['flyback_h_px']

    # Calculate the field of view parameters that fit all MROIs.
    umd['fov'] = dict()
    mroi_corners_tl_deg = np.array([(r['center_deg'] - (r['size_deg'] / 2)) for r in umd['mrois']['lrsort']])
    mroi_corners_br_deg = np.array([(r['center_deg'] + (r['size_deg'] / 2)) for r in umd['mrois']['lrsort']])
    umd['fov']['corner_tl_deg'] = np.array([mroi_corners_tl_deg[:, 0].min(),
                                            mroi_corners_tl_deg[:, 1].min()])
    umd['fov']['corner_br_deg'] = np.array([mroi_corners_br_deg[:, 0].max(),
                                            mroi_corners_br_deg[:, 1].max()])
    if np.all(np.isclose(mroi_resolutions_umpx, mroi_resolutions_umpx[0])) and \
       np.all(np.isclose(mroi_resolutions_degpx, mroi_resolutions_degpx[0])):
        umd['fov']['resolution_umpx'] = mroi_resolutions_umpx[0]
        umd['fov']['xres_umpx'] = umd['fov']['resolution_umpx'][0]
        umd['fov']['yres_umpx'] = umd['fov']['resolution_umpx'][1]
        umd['fov']['resolution_degpx'] = mroi_resolutions_degpx[0]
    else:
        warn('Not all MROIs have the same resolution.')
        umd['fov']['resolution_umpx'] = None
        umd['fov']['resolution_degpx'] = None
    fov_positions_deg = [np.arange(umd['fov']['corner_tl_deg'][0],  # x_deg min
                                   umd['fov']['corner_br_deg'][0],  # x_deg max
                                   umd['fov']['resolution_degpx'][0]),
                         np.arange(umd['fov']['corner_tl_deg'][1],  # y_deg min
                                   umd['fov']['corner_br_deg'][1],  # y_deg max
                                   umd['fov']['resolution_degpx'][1])]

    # Calculate the pixel coordinates for MROIs in reconstructed volume.
    mroi_corners_tl_px = np.empty((umd['n_mrois'], 2), dtype=int)
    for i_xy in range(2):
        dimax = 'width' if i_xy == 0 else 'height'
        for i_mroi in range(umd['n_mrois']):
            closest_xy_px = np.argmin(np.abs(fov_positions_deg[i_xy] - mroi_corners_tl_deg[i_mroi, i_xy])).astype(int)
            mroi_corners_tl_px[i_mroi, i_xy] = closest_xy_px
            closest_xy_deg = fov_positions_deg[i_xy][closest_xy_px]
            if not np.isclose(closest_xy_deg, mroi_corners_tl_deg[i_mroi, i_xy]):
                warn('Fit of MROI into reconstructed image {} is imperfect: '.format(dimax) +
                     'MROI {}, top-left corner {:.4f}, closest {:.4f}'.format(i_mroi,
                                                                              mroi_corners_tl_deg[i_mroi, i_xy],
                                                                              closest_xy_deg))
    del dimax

    # If necessary, remove extra pixel added to reconstruction width due to rounding errors or ScanImage errors.
    if len(fov_positions_deg[0]) == np.sum(mroi_sizes_px[:, 0]) + 1:
        warn('Removed extra pixel from reconstructed image width.')
        fov_positions_deg[0] = fov_positions_deg[0][:-1]
    if np.any(len(fov_positions_deg[1]) == mroi_sizes_px[:, 1] + 1):
        warn('Removed extra pixel from reconstructed image height.')
        fov_positions_deg[1] = fov_positions_deg[1][:-1]

    umd['fov']['positions_deg'] = fov_positions_deg
    umd['fov']['w_px'] = len(fov_positions_deg[0])
    umd['fov']['h_px'] = len(fov_positions_deg[1])
    if 'resolution_umpx' in umd['fov']:
        umd['fov']['w_um'] = umd['fov']['w_px'] * umd['fov']['resolution_umpx'][0]
        umd['fov']['h_um'] = umd['fov']['h_px'] * umd['fov']['resolution_umpx'][1]
    for i_mroi in range(umd['n_mrois']):
        i_m = umd['mrois']['lrsort_arg'][i_mroi]
        umd['mrois']['orig'][i_m]['corner_tl_deg'] = mroi_corners_tl_deg[i_mroi]
        umd['mrois']['orig'][i_m]['corner_tl_px'] = mroi_corners_tl_px[i_mroi]
        umd['mrois']['lrsort'][i_mroi]['corner_tl_deg'] = mroi_corners_tl_deg[i_mroi]
        umd['mrois']['lrsort'][i_mroi]['corner_tl_px'] = mroi_corners_tl_px[i_mroi]

    # Estimate neuron size (i.e. cell body diameter) in px, assuming average diameter of 15um.
    if umd['fov']['resolution_umpx'] is not None:
        neuron_diameter_um = 15
        umd['fov']['neurondiameter_px'] = int(np.mean(neuron_diameter_um / umd['fov']['resolution_umpx']))
    else:
        umd['fov']['neurondiameter_px'] = None

    # Calculate additional values if all MROIs have the same size, resolution, and vertical position.
    if np.all(np.isclose(mroi_centers_deg[:, 1], mroi_centers_deg[0, 1])) and \
            np.all(np.isclose(mroi_sizes_deg, mroi_sizes_deg[0])) and \
            np.all(np.isclose(mroi_sizes_px, mroi_sizes_px[0])) and \
            np.all(np.isclose(mroi_resolutions_umpx, mroi_resolutions_umpx[0])):
        # print('All MROIs have the same size, resolution, and vertical position.')
        umd['strip'] = dict()
        umd['n_strips'] = umd['n_mrois']
        umd['strip']['size_deg'] = mroi_sizes_deg[0]
        umd['strip']['w_deg'] = umd['strip']['size_deg'][0]
        umd['strip']['h_deg'] = umd['strip']['size_deg'][1]
        umd['strip']['size_px'] = mroi_sizes_px[0]
        umd['strip']['w_px'] = umd['strip']['size_px'][0]
        umd['strip']['h_px'] = umd['strip']['size_px'][1]
        umd['strip']['resolution_umpx'] = mroi_resolutions_umpx[0]
        umd['strip']['xres_umpx'] = umd['strip']['resolution_umpx'][0]
        umd['strip']['yres_umpx'] = umd['strip']['resolution_umpx'][1]
        xres_str = '{:03.2f}'.format(umd['strip']['xres_umpx']).replace('.', 'p')
        yres_str = '{:03.2f}'.format(umd['strip']['yres_umpx']).replace('.', 'p')
        umd['strip']['resolution_str'] = 'res{}x{}umpx'.format(xres_str, yres_str)
        umd['strip']['size_um'] = umd['strip']['size_px'] * umd['strip']['resolution_umpx']
        umd['strip']['w_um'] = umd['strip']['size_um'][0]
        umd['strip']['h_um'] = umd['strip']['size_um'][1]
    else:
        umd['strip'] = None
        umd['n_strips'] = None

    if umd['strip'] is not None:
        umd['plane'] = dict()
        if umd['mrois']['overlap'] is False and umd['mrois']['overlap_px'] is None:
            umd['plane']['size_deg'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_deg']
            umd['plane']['w_deg'] = umd['plane']['size_deg'][0]
            umd['plane']['h_deg'] = umd['plane']['size_deg'][1]
            umd['plane']['size_px'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_px']
            umd['plane']['w_px'] = umd['plane']['size_px'][0]
            umd['plane']['h_px'] = umd['plane']['size_px'][1]
            umd['plane']['size_um'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_um']
            umd['plane']['w_um'] = round(umd['plane']['size_um'][0])
            umd['plane']['h_um'] = round(umd['plane']['size_um'][1])
            umd['plane']['size_um_str'] = 'fov{:04d}x{:04d}um'.format(umd['plane']['w_um'],
                                                                      umd['plane']['h_um'])
        elif umd['mrois']['overlap'] is True and type(umd['mrois']['overlap_px']) is int:
            umd['plane']['size_deg'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_deg']
            # TODO reduce plane size by the overlap amount
            # umd['plane']['w_deg'] = umd['plane']['size_deg'][0]
            # umd['plane']['h_deg'] = umd['plane']['size_deg'][1]
            # umd['plane']['size_px'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_px']
            # umd['plane']['w_px'] = umd['plane']['size_px'][0]
            # umd['plane']['h_px'] = umd['plane']['size_px'][1]
            # umd['plane']['size_um'] = np.array([umd['n_strips'], 1]) * umd['strip']['size_um']
            # umd['plane']['w_um'] = round(umd['plane']['size_um'][0])
            # umd['plane']['h_um'] = round(umd['plane']['size_um'][1])
            # umd['plane']['size_um_str'] = 'fov{:04d}x{:04d}um'.format(round(umd['plane']['w_um']),
            #                                                           round(umd['plane']['h_um']))
    else:
        warn('Only MROI strips are currently supported.')
        umd['plane'] = None

    return umd
