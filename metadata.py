#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
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
    eqpos = string.find('=')

    # Replace 'sep' instances in value with unit separator.
    if sep in string[eqpos + 1:-1]:
        stringA, stringB = string.split('=')
        stringBnosep = stringB.replace(sep, chr(31))
        string = '='.join([stringA, stringBnosep])

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
    modified from https://stackoverflow.com/a/56177639
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
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 if d2 is None else d2
    # Compute set of all keys in both dictionaries.
    keys = set(d1.keys()) | set(d2.keys())
    # Build output dictionary, merging recursively values with common keys,
    # where `None` is used to mean the absence of a value.
    return {key: merge_metadata_dicts(d1.get(key), d2.get(key)) for key in keys}


def parse_scanimage_desc(desc):
    desc_dict = {}
    for line in desc.splitlines():
        if '=' not in line:
            continue
        desc_dict = merge_metadata_dicts(desc_dict,
                                         metadata_line_to_dict(line, '.'))
    return desc_dict


def get_scanimage_metadata(filepath):
    md_dict = {}
    with ScanImageTiffReader(filepath) as reader:
        # Attempt to determine number of imaging planes from filename
        fn = os.path.basename(filepath)
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
            warn('Could not determine number of imaging planes from filename. ' + \
                 'Assuming single plane (n_planes = 1).')
            n_planes = 1
            mode = 'sp'
        md_dict['n_planes'] = n_planes
        md_dict['mode'] = mode

        # Attempt to determine depth of deepest imaging plane from filename
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
            warn('Could not determine depth of deepest imaging plane from filename. ')
            depth = None
        if depth is not None:
            depth = float(depth)
            if depth.is_integer():
                depth = int(depth)
        md_dict['depth'] = depth

        # Attempt to determine laser power from filename
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
        if power is not None and 'p' in power:
            power = power.replace('p', '.')
            power = float(power)
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


def roi_from_scanfield(scanfield, objective_resolution):
    r = {}
    r['center_deg'] = np.array(scanfield['centerXY'])
    r['size_deg'] = np.array(scanfield['sizeXY'])
    r['size_px'] = np.array(scanfield['pixelResolutionXY'])
    if type(objective_resolution) is float:
        r['resolution_umpx'] = objective_resolution / (r['size_px'] / r['size_deg'])
    else:
        r['resolution_umpx'] = None

    return r

def extract_useful_metadata(scanimage_metadata):
    # NOTE assumes ROIs strips of a larger plane ROI and are all the same size
    # NOTE assumes data acquired before 20230505d have strip overlap and after do not
    simd = scanimage_metadata
    date_strip_overlap_fix = datetime(2023, 5, 5, tzinfo=ZoneInfo('America/New_York'))

    umd = {}

    umd['n_planes'] = simd['n_planes']
    umd['mode'] = simd['mode']
    umd['depth'] = simd['depth']
    umd['power'] = simd['power']
    umd['n_planes'] = simd['n_planes']
    umd['n_frames'] = simd['n_frames']
    umd['objective_resolution'] = simd['SI']['objectiveResolution']  # um/deg
    umd['framerate'] = simd['SI']['hRoiManager']['scanFrameRate']
    umd['framerate_str'] = '{:06.3f}'.format(umd['framerate']).replace('.', 'p') + 'Hz'
    umd['datetime_start'] = simd['frame0desc']['epoch'] - timedelta(seconds=umd['framerate'])
    umd['date_str'] = umd['datetime_start'].strftime('%Y%m%dd')
    umd['time_start_str'] = umd['datetime_start'].strftime('%H%M%StUTC')

    # Extract ScanImage MROI details.
    umd['mrois'] = {}
    if umd['datetime_start'] > date_strip_overlap_fix:
        umd['mrois']['overlap'] = False
        umd['mrois']['overlap_px'] = None
    else:
        umd['mrois']['overlap'] = True
        umd['mrois']['overlap_px'] = 'unknown'

    if simd['json'] is not None:
        mrois_raw = simd['json']['RoiGroups']['imagingRoiGroup']['rois']
    else:
        mrois_raw = json.loads(simd["Artist"])['RoiGroups']['imagingRoiGroup']['rois']
    if type(mrois_raw) != dict:
        mrois_orig = []
        for roi in mrois_raw:
            if type(roi['scanfields']) != list:
                scanfield = roi['scanfields']
            else:
                scanfield = roi['scanfields'][np.where(np.array(roi['zs']) == 0)[0][0]]
            r = roi_from_scanfield(scanfield, umd['objective_resolution'])
            mrois_orig.append(r)
    else:
        scanfield = mrois_raw['scanfields']
        r = roi_from_scanfield(scanfield, umd['objective_resolution'])
        mrois_orig = [r]
    umd['mrois']['orig'] = mrois_orig
    umd['n_mrois'] = len(mrois_orig)

    # Sort MROIs by physical left-to-right position.
    # Keep un-sorted 'orig' version to recover data location in the long-tif-strip.
    mroi_centers_deg = np.array([r['center_deg'] for r in mrois_orig])
    mroi_centers_lrsort_arg = np.argsort(mroi_centers_deg[:, 0])
    mrois_lrsort = [mrois_orig[a] for a in mroi_centers_lrsort_arg]
    mroi_centers_lrsort_deg = [mroi_centers_deg[a] for a in mroi_centers_lrsort_arg]
    umd['mrois']['lrsort'] = mrois_lrsort

    mroi_sizes_deg = np.array([r['size_deg'] for r in mrois_orig])
    mroi_sizes_px = np.array([r['size_px'] for r in mrois_orig])
    mroi_resolutions_umpx = np.array([r['resolution_umpx'] for r in mrois_orig])
    if np.all(np.isclose(mroi_sizes_deg, mroi_sizes_deg[0])) and \
            np.all(np.isclose(mroi_sizes_px, mroi_sizes_px[0])) and \
            np.all(np.isclose(mroi_sizes_px, mroi_sizes_px[0])):
        print('All MROIs are the same size and resolution.')
        umd['strip'] = {}
        umd['n_strips'] = len(umd['mrois']['orig'])
        umd['strip']['size_deg'] = mroi_sizes_deg[0]
        umd['strip']['w_deg'] = umd['strip']['size_deg'][0]
        umd['strip']['h_deg'] = umd['strip']['size_deg'][1]
        umd['strip']['size_px'] = mroi_sizes_px[0]
        umd['strip']['w_px'] = umd['strip']['size_px'][0]
        umd['strip']['h_px'] = umd['strip']['size_px'][1]
        umd['strip']['resolution_umpx'] = mroi_resolutions_umpx[0]
        umd['strip']['xres_umpx'] = umd['strip']['resolution_umpx'][0]
        umd['strip']['yres_umpx'] = umd['strip']['resolution_umpx'][1]
        xres_str = '{:03.2f}'.format(umd['xres_umpx']).replace('.', 'p')
        yres_str = '{:03.2f}'.format(umd['yres_umpx']).replace('.', 'p')
        umd['strip']['resolution_str'] = 'res{}x{}umpx'.format(xres_str, yres_str)
    else:
        umd['strip'] = None
        umd['n_strips'] = None

    # if umd['strip'] is not None:
    #     if umd['mrois']['overlap'] == False and \
    #        umd['mrois']['overlap_px'] is None:
    #         umd['plane']['size_deg'] =
    #         umd['plane']['size_px'] = umd['strip_size_px'] * np.array([umd['n_strips'], 1])
    #     elif umd['mrois']['overlap'] == True and \
    #          type(umd['mrois']['overlap_px']) is int:
    #     umd['plane']
    #     umd['plane_size_px'] = umd['strip_size_px'] * np.array([umd['n_strips'], 1])
    #     umd['plane_w_px'] = umd['plane_size_px'][0]
    #     umd['plane_h_px'] = umd['plane_size_px'][1]
    #     umd['plane_size_um'] = umd['res_umpx'] * umd['plane_size_px']
    #     umd['plane_w_um'] = umd['plane_size_um'][0]
    #     umd['plane_h_um'] = umd['plane_size_um'][1]
    #     umd['plane_size_um_str'] = 'fov{:04d}x{:04d}um'.format(round(umd['plane_w_um']),
    #                                                            round(umd['plane_h_um']))
    # else:
    #     warn('Only MROI strips are currently supported.')
    #     umd['plane'] = None

    return umd
