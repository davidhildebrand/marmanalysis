#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
import json
import numpy as np
import re
from ScanImageTiffReader import ScanImageTiffReader
from zoneinfo import ZoneInfo

def metadata_line_to_dict(string, sep, acqtz='America/New_York'):
    if string.count('=') != 1:
        print('error: do not know how to handle strings with more than one equals sign')
    eqpos = string.find('=')
    if sep in string[eqpos+1:-1]:
        # replace sep instances in value with unit separator
        stringA, stringB = string.split('=')
        stringBnosep = stringB.replace(sep, chr(31))
        string = '='.join([stringA, stringBnosep])
    if sep not in string:
        k1s, vs = string.split('=')
        k1 = k1s.strip()
        # replace unit separator instances with sep
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
        elif re.match("^\[\ *[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\ +[0-9]+\.[0-9]+\ *\]$", v) is not None:
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
        md_dict['n_frames'] = reader.shape()[0]

        desc = parse_scanimage_desc(reader.description(0))
        md_dict['frame0desc'] = desc

        md_raw = reader.metadata()
        md_json_start = md_raw.find('\n{')
        md_json_end = md_raw.find('}\n', -1)
        md_nonjson = md_raw[0:md_json_start]
        for line in md_nonjson.splitlines():
            md_dict = merge_metadata_dicts(md_dict, 
                                           metadata_line_to_dict(line, '.'))
        md_json_str = md_raw[md_json_start+1:md_json_end]
        md_json = json.loads(md_json_str)
        md_dict['json_str'] = md_json_str
        md_dict['json'] = md_json

        return md_dict

