#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
import json
import numpy as np
import os
import re
from ScanImageTiffReader import ScanImageTiffReader
from warnings import warn
from zoneinfo import ZoneInfo


def metadata_line_to_dict(string, sep, acqtz='America/New_York'):
    if string.count('=') != 1:
        warn('Do not know how to handle strings with more than one equals sign.')
        print(string)
    eqpos = string.find('=')

    # Replace 'sep' instances in value with unit separator
    if sep in string[eqpos+1:-1]:
        stringA, stringB = string.split('=')
        stringBnosep = stringB.replace(sep, chr(31))
        string = '='.join([stringA, stringBnosep])

    if sep not in string:
        k1s, vs = string.split('=')
        k1 = k1s.strip()

        # Replace unit separator instances with sep
        v = vs.strip().replace(chr(31), sep)
        
        # Import values from MATLAB ScanImage into python format
        ptrn_num = r'^[+-]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?$'
        ptrn_dt = r'^\[\ *[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\.[0-9]+\ *\]$'
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
        elif re.match(ptrn_num, v) is not None:
            v = float(v)
            if v.is_integer():
                v = int(v)
        elif re.match(ptrn_dt, v) is not None:
            # e.g., [2023  8  4 14  2 8.937]
            #m = re.match("^\[\ *([0-9]+)\ +([0-9]+)\ +([0-9]+)\ +([0-9]+)\ +([0-9]+)\ +([0-9]+)(\.[0-9]+)\ *\]$", v)
            #g = m.groups()
            #year, month, day = (int(g[x]) for x in range(0, 3))
            #hour, minute, sec = (int(g[x]) for x in range(3, 6))
            #sec_mantissa = float(g[-1])
            #microsec = int(millisec * 10**6)
            #dt = datetime(year, month, day, hour,minute, sec, microsec, tzinfo=ZoneInfo(acqtz))
            v = ' '.join(v.split())
            dt = datetime.strptime(v, '[%Y %m %d %H %M %S.%f]').replace(tzinfo=ZoneInfo(acqtz))
            v = dt.astimezone(timezone.utc)
        elif '[' in v and ']' in v:
            if v == '[]':
                v = []
            else:
                bs = v.find('[')
                be = v.find(']', bs)
                v = ' '.join(v.split())
                v = np.fromstring(v[bs+1:be], dtype=float, sep=' ')
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
        ptrn_d0 = r'^.*depth([0-9]+)um.*$'
        ptrn_d1 = r'^.*[^0-9]([0-9]+)umdeep.*$'
        if re.match(ptrn_d0, fn) is not None:
            m = re.match(ptrn_d0, fn)
        elif re.match(ptrn_d1, fn) is not None:
            m = re.match(ptrn_d1, fn)
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
        ptrn_p0 = r'^.*pow([0-9]+p?[0-9]+?)mW.*$'
        ptrn_p1 = r'^.*[^0-9]([0-9]+p?[0-9]+?)mW.*$'
        if re.match(ptrn_p0, fn) is not None:
            m = re.match(ptrn_p0, fn)
        elif re.match(ptrn_p1, fn) is not None:
            m = re.match(ptrn_p1, fn)
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
        #md_json_end = md_raw.find('}\n', -2) + 1

        if md_json_start != -1:
            md_json_str = md_raw[md_json_start+1:]
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


def extract_useful_metadata(scanimage_metadata):
    # NOTE assumes ROIs strips of a larger plane ROI and are all the same size
    # NOTE assumes acqusitions before 20230505d have strip overlap and after do not
    simd = scanimage_metadata
    date_strip_overlap_fix = datetime(2023, 5, 5, tzinfo=ZoneInfo('America/New_York'))
    
    umd = {}
    
    umd['n_planes'] = simd['n_planes']
    umd['mode'] = simd['mode']
    umd['depth'] = simd['depth']
    umd['power'] = simd['power']
    umd['n_planes'] = simd['n_planes']
    umd['n_frames'] = simd['n_frames']
    umd['n_strips'] = len(simd['json']['RoiGroups']['imagingRoiGroup']['rois'])
    umd['objective_resolution'] = simd['SI']['objectiveResolution']  # um/deg
    umd['framerate'] = simd['SI']['hRoiManager']['scanFrameRate']
    umd['framerate_str'] = '{:06.3f}'.format(umd['framerate']).replace('.', 'p') + 'Hz'
    umd['datetime_start'] = simd['frame0desc']['epoch'] - timedelta(seconds=umd['framerate'])
    umd['date_str'] = umd['datetime_start'].strftime('%Y%m%dd')
    umd['time_start_str'] = umd['datetime_start'].strftime('%H%M%StUTC')
    
    scanfields = simd['json']['RoiGroups']['imagingRoiGroup']['rois'][0]['scanfields']
    umd['strip_size_px'] = np.array(scanfields['pixelResolutionXY'])
    umd['strip_w_px'] = umd['strip_size_px'][0]
    umd['strip_h_px'] = umd['strip_size_px'][1]
    umd['strip_size_deg'] = np.array(scanfields['sizeXY'])
    umd['strip_w_deg'] = umd['strip_size_deg'][0]
    umd['strip_h_deg'] = umd['strip_size_deg'][1]
    if umd['datetime_start'] < date_strip_overlap_fix:
        umd['strip_overlap'] = False
        umd['strip_overlap_px'] = None
    else:
        umd['strip_overlap'] = True
        umd['strip_overlap_px'] = None
    
    umd['res_x_umppx'] = umd['objective_resolution'] / (umd['strip_w_px'] / umd['strip_w_deg'])
    res_x_umppx_str = '{:03.2f}'.format(umd['res_x_umppx']).replace('.', 'p')
    umd['res_y_umppx'] = umd['objective_resolution'] / (umd['strip_h_px'] / umd['strip_h_deg'])
    res_y_umppx_str = '{:03.2f}'.format(umd['res_y_umppx']).replace('.', 'p')
    umd['res_umpx'] = np.array([umd['res_x_umppx'], umd['res_y_umppx']])
    umd['res_str'] = 'res{}x{}umpx'.format(res_x_umppx_str, res_y_umppx_str)
    
    umd['plane_size_px'] = umd['strip_size_px'] * np.array([umd['n_strips'], 1])
    umd['plane_w_px'] = umd['plane_size_px'][0]
    umd['plane_h_px'] = umd['plane_size_px'][1]
    umd['plane_size_um'] = umd['res_umpx'] * umd['plane_size_px']
    umd['plane_w_um'] = umd['plane_size_um'][0]
    umd['plane_h_um'] = umd['plane_size_um'][1]
    umd['plane_size_um_str'] = 'fov{:04d}x{:04d}um'.format(round(umd['plane_w_um']), 
                                                           round(umd['plane_h_um']))

    return umd
