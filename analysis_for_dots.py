#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from datetime import datetime
from glob import glob
# import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.optimize import least_squares
from scipy.stats import binned_statistic
import socket
from warnings import warn

import filters
from metadata import default_metadata
import parsers
# import plots
from tuning import calculate_dsi


# %% Settings

# Direction-Selectivity Index (DSI) threshold
# based on Pattadkal etal Priebe 2022 bioRxiv
#   https://doi.org/10.1101/2022.06.23.497220
# "All cell pairs with DSI ≥ 0.15 are considered."
threshold_dsi = 0.15

threshold_cellprob = 0.0
threshold_Zscore = 0.5

## Exclusion criteria
exclude_by_movement = True

## Plotting parameters
plot_eyecal = False
plt.rcParams['figure.dpi'] = 600

## Data handling parameters
# Fix off-by-one issue caused by acqusition frame counter ('acqfr') index starting at 1 rather than 0.
correct_acqfr_index = True

# Metrics to consider for plots and calculations
metrics = ['FdFF', 'Fzsc']
metric_labels = {'FdFF': 'dF/F',
                 'Fzsc': 'Z-score'}

# Remove stale metadata
if 'md' in locals():
    md = dict()
    del md


# %% Specify data locations

# savepath_str = 'analysis'
# save_path = r''
save_ext = ['.png', '.svg']

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Cadbury
animal_str = 'Cadbury'

# # 20221016d Right ribo-jGCaMP8s
date_str = '20221016d'
# -- MT 200um (good)
session_str = '163736tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p362Hz_pow059p0mW_stimMovingDots8dirFF'
md = dict()
md['framerate'] = 6.362
md['fov'] = dict()
md['fov']['resolution_umpx'] = np.array([2.0, 2.0])
md['fov']['w_px'] = 730
md['fov']['h_px'] = 730
# dirstr_suite2p = 'suite2p_old*'
dirstr_suite2p = 'suite2p_cellpose2_d7px_pt-3p5_ft1p5'

# # 20231001d Right ribo-jGCaMP8s (~430um fluid)
# date_str = '20231001d'
# -- MT 200um (| movement exclusions needed)
# session_str = '205309tUTC_SP_depth200um_fov1460x1200um_res2p00x2p00umpx_fr07p685Hz_pow060p3mW_stimMovingDots16dirFF'

# # 20231007d Right ribo-jGCaMP8s (~120um fluid)
# date_str = '20231007d'
# # -- PD 200um ( | eyetrack crashed) ... PD but included dot stims
# session_str = '180407tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p359Hz_pow080p2mW_stimMultimodal'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Cashew
# animal_str = 'Cashew'

# # 20230728d Right ribo-jGCaMP8s (negligible fluid)
# date_str = '20230728d'
# # -- MT 200um ( | drowsy, YouTube audio)
# session_str = '153957tUTC_SP_depth200um_fov2190x2600um_res3p00x3p00umpx_fr05p381Hz_pow059p9mW_stimMovingDots16dirFF

# # 20230806d Right ribo-jGCaMP8s (up to 250um fluid around MD and STS)
# date_str = '20230806d'
# # -- MT 200um ( | )
# session_str = '161055tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow059p9mW_stimMovingDots16dirFF

# # 20230809d Right ribo-jGCaMP8s
# date_str = '20230809d'
# # -- MT 200um ( | )
# session_str = '162517tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimMovingDots16dirFF'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Dali
# animal_str = 'Dali'

# # 20230622d Right ribo-jGCaMP8s
# date_str = '20230622d'
# # -- MT? 300um ( | )
# session_str = '165308tUTC_SP_depth300um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow069p8mW_stimMovingDots8dirFF'

# # 20230627d Right ribo-jGCaMP8s
# date_str = '20230627d'
# # -- MT? 200um ( | )
# session_str = '152654tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p3mW_stimMovingDots8dirFF'

# # 20230704d Right ribo-jGCaMP8s
# date_str = '20230704d'
# # -- MT? 200um ( | )
# session_str = '160506tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p0mW_stimMovingDots16dirFF'

# # 20230906d Right ribo-jGCaMP8s
# date_str = '20230906d'
# # -- PD 200um ( | )
# session_str = '180300tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow049p6mW_stimMovingDots16dirFF'

# # 20240923d Left ribo-jGCaMP8s
# date_str = '20240923d'
# # -- aud? 200um ( | )
# session_str = '180320tUTC_SP_depth200um_fov0742x0700um_res1p00x1p00umpx_fr04p734Hz_pow039p7mW_stimMultimodalTonesDotsGratingsFlashes'

# # 20240926d Left ribo-jGCaMP8s
# date_str = '20240926d'
# # -- PD-MT? 200um ( | )
# session_str = '171222tUTC_SP_depth260um_fov1060x1800um_res2p00x2p00umpx_fr05p196Hz_pow050p4mW_stimMultimodalImagesDotsGratingsTonesVocalizations'
# # -- PD-MT? 200um ( | )
# session_str = '171222tUTC_SP_depth260um_fov1060x1800um_res2p00x2p00umpx_fr05p196Hz_pow050p4mW_stimMultimodalImagesDotsGratingsTonesVocalizations'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Larry
# animal_str = 'Larry'

# # 20231007d Right ribo-jGCaMP8s
# date_str = '20231007d'
# # -- MT 200um ( | movement exclusions needed)
# session_str = '211856tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimMovingDots16dirFF'
# # -- MT 200um ( | )
# session_str = '213733tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow060p1mW_stimMovingDots16dirFF'

# # 20241101d Left soma-jGCaMP8s
# date_str = '20241101d'
# # -- MT 200um ( | )
# session_str = '161123tUTC_SP_depth200um_fov1184x1184um_res2p00x2p00umpx_fr09p733Hz_pow030p1mW_stimMovingDots8dirFF'

# # 20241104d Left soma-jGCaMP8s
# date_str = '20241104d'
# # -- MT 200um ( | )
# session_str = '163451tUTC_SP_depth200um_fov3626x3626um_res3p50x3p50umpx_fr03p236Hz_pow029p5mW_stimMultimodal'
# session_str = '173725tUTC_Max15_depth400um_fov3626x3626um_res3p50x3p50umpx_fr03p236Hz_pow279p8mW_stimMultimodal'

# # 20241218d Left soma-jGCaMP8s
# date_str = '20241218d'
# # -- MT 200um ( | )
# session_str = '153342tUTC_SP_depth200um_fov0740x0740um_res1p00x1p00umpx_fr06p283Hz_pow030p4mW_stimMovingDots16dirFF'
# # -- MT 150um ( | )
# session_str = '155459tUTC_SP_depth150um_fov0740x0740um_res1p00x1p00umpx_fr06p283Hz_pow025p3mW_stimMovingDots16dirFF'

# # 20241227d Left soma-jGCaMP8s
# date_str = '20241227d'
# # -- MT 150um ( | )
# session_str = '170445tUTC_SP_depth150um_fov0740x0740um_res1p00x1p00umpx_fr06p282Hz_pow015p2mW_stimMultimodal'
# # -- MT 150um ( | )
# session_str = '174301tUTC_SP_depth150um_fov0740x0740um_res1p00x1p00umpx_fr06p282Hz_pow015p2mW_stimMovingDots16dirFF'
# # -- MT 200um ( | )
# session_str = '175431tUTC_SP_depth200um_fov0740x0740um_res1p00x1p00umpx_fr06p282Hz_pow019p9mW_stimMovingDots16dirFF'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if 'save_path' not in locals():
    save_path = ''
if 'savepath_str' not in locals():
    savepath_str = ''
if isinstance(save_ext, str):
    save_ext = [save_ext]
if 'stimimage_path' not in locals():
    stimimage_path = ''

if session_str.find('_') != -1:
    session_abbrev_str = session_str[0:session_str.find('_')]
    title_str = animal_str + '_' + date_str + '_' + session_abbrev_str
else:
    session_abbrev_str = ''
    title_str = animal_str + '_' + date_str + '_' + session_str
save_pfix = animal_str + date_str + session_abbrev_str

filestr_metadata = '*_metadata.pickle'
filestr_image_data = '*_00001.tif'
filestr_session_log = '*.log'
pattern_session_log = r'.*^((?!disptimes).)*$'  # Exclude log files whose names contain 'disptimes'
filestr_stimulus_log = '*_stimlog.*'
filestr_eyetrack_data = '*_AIdata.p'
dirstr_eyecal = '*_EyeTrackingCalibration'
filestr_eyecal_log = '*_EyeTrackingCalibration.log'
filestr_eyecal_aidata = '*_EyeTrackingCalibration_AIdata.p'
if 'dirstr_suite2p' not in locals():
    dirstr_suite2p = 'suite2p*'
dirstr_suite2p_pref = 'suite2p_cellpose3_d[0-9]+px_pt-3p5_ft1p5'
dirstr_suite2p_plane = 'plane0'


# %% Define functions and classes


deg_symbol = u'\N{DEGREE SIGN}'


class StimulusDots(object):
    """Representation of stimulus dots."""

    # 'dots_translation_dir': None,
    # 'dots_opticflow_dir': None,
    # 'dots_rotation_dir': None,
    # 'dots_nDots': None,
    # 'dots_coherence': None,
    # 'dots_fieldPos': None,
    # 'dots_fieldSize': None,
    # 'dots_fieldShape': None,
    # 'dots_dotSize': None,
    # 'dots_dotLife': None,
    # 'dots_dir': None,
    # 'dots_speed': None,
    # 'dots_color': None,
    # 'dots_opacity': None,
    # 'dots_contrast': None,
    # 'dots_signalDots': None,
    # 'dots_noiseDots': None,

    def __init__(self, 
                 condition, 
                 # stim_class,  # maybe use group and subgroup instead?
                 subclass, 
                 units='',
                 nDots=None,
                 coherence=0.5,
                 fieldPos=(0.0, 0.0),
                 fieldSize=(1.0, 1.0),
                 fieldShape='sqr',
                 # fieldAnchor='center',
                 dotSize=2.0,
                 dotLife=3,
                 dir=0.0,
                 speed=0.5,
                 color=(1.0, 1.0, 1.0),
                 colorSpace='rgb',
                 opacity=None,
                 contrast=1.0,
                 element=None,
                 signalDots='same',
                 noiseDots='direction'):
        self.condition = condition
        self.subclass = subclass
        self.units = units
        self.nDots = nDots
        self.coherence = coherence
        self.fieldPos = fieldPos
        self.fieldSize = fieldSize
        self.fieldShape = fieldShape
        self.dotSize = dotSize
        self.dotLife = dotLife
        self.dir = dir
        self.speed = speed
        self.color = color
        self.colorSpace = colorSpace
        self.opacity = opacity
        self.contrast = contrast
        self.element = element
        self.signalDots = signalDots
        self.noiseDots = noiseDots

    def __repr__(self):
        return str((self.condition, self.subclass, self.dir))

    def __eq__(self, other):
        return self.condition == other.condition and \
            self.subclass == other.subclass and \
            self.dir == other.dir

    # *** TODO handle clockwise/counterclockwise?
    def __lt__(self, other):
        return self.dir < other.dir

    def __gt__(self, other):
        return self.dir > other.dir


# %% Find files and load data

system_name = socket.gethostname()
if 'galactica' in system_name.lower():
    base_path = r'/Users/davidh/Data/Freiwald/suite2p_results'
    stim_path = r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets'
elif 'obsidian' in system_name.lower():
    base_path = r'F:\Data'
    stim_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets'
elif 'dobbin' in system_name.lower():
    base_path = r'D:\Data'
    stim_path = r'C:\Users\DavidH\Sync\Freiwald\MarmoScope\Stimulus\Sets'
else:
    base_path = None
    stim_path = None

date_path = os.path.join(base_path, animal_str, date_str)
session_path = os.path.join(base_path, animal_str, date_str, session_str)

if savepath_str != '' and save_path == '':
    save_path = os.path.join(session_path, savepath_str)
    os.makedirs(save_path, exist_ok=True)
if save_path == '':
    saving = False
else:
    saving = True

filelist_metadata = [f for f in glob(os.path.join(session_path, filestr_metadata)) if os.path.isfile(f)]
filelist_image_data = [f for f in glob(os.path.join(session_path, filestr_image_data)) if os.path.isfile(f)]
if 'md' not in locals():
    if not filelist_metadata and not filelist_image_data:
        raise RuntimeError('Could not find metadata file or image data file.')
    if len(filelist_metadata) > 0:
        if len(filelist_metadata) > 1:
            warn('Found multiple metadata files, using the first: {}'.format(filelist_metadata[0]))
        md_path = filelist_metadata[0]
        with open(md_path, 'rb') as mdf:
            md = pickle.load(mdf)
    elif len(filelist_image_data) > 0:
        warn('Could not find metadata file, loading from image data file.')
        df_path = filelist_image_data[0]
        import metadata
        simd = metadata.get_metadata(df_path)
        md = metadata.extract_useful_metadata(simd)
md = {**default_metadata(), **md}

if 'fov' in md:
    if 'w_um' not in md['fov'] and 'resolution_umpx' in md['fov'] and 'w_px' in md['fov']:
        md['fov']['w_um'] = md['fov']['resolution_umpx'][0] * md['fov']['w_px']
    if 'h_um' not in md['fov'] and 'resolution_umpx' in md['fov'] and 'h_px' in md['fov']:
        md['fov']['h_um'] = md['fov']['resolution_umpx'][1] * md['fov']['h_px']

filelist_session_log = [f for f in glob(os.path.join(session_path, filestr_session_log))
                        if re.search(pattern_session_log, f) and os.path.isfile(f)]
if len(filelist_session_log) > 0:
    lf_path = filelist_session_log[0]
    if len(filelist_session_log) > 1:
        warn('Found multiple log files, using the first: {}'.format(lf_path))
    lf = open(lf_path, 'r')
    session_log = lf.read()
    lf.close()
    del lf
else:
    lf_path = None
    session_log = None

filelist_stimulus_log = [f for f in glob(os.path.join(session_path, filestr_stimulus_log)) if os.path.isfile(f)]
if len(filelist_stimulus_log) > 0:
    pkls = [f for f in filelist_stimulus_log if f.endswith('.pickle') or f.endswith('.pkl') or f.endswith('.p')]
    hdf5s = [f for f in filelist_stimulus_log if f.endswith('.h5') or f.endswith('.hdf5')]
    csvs = [f for f in filelist_stimulus_log if f.endswith('.csv')]
    if len(pkls) > 0:
        if len(pkls) > 1:
            warn('Found multiple stimlog pickle files, using the first: {}'.format(pkls[0]))
        slf_path = pkls[0]
        sl = pd.read_pickle(slf_path)
    elif len(hdf5s) > 0:
        if len(hdf5s) > 1:
            warn('Found multiple stimlog hdf5 files, using the first: {}'.format(hdf5s[0]))
        slf_path = hdf5s[0]
        sl = pd.read_hdf(slf_path)
    elif len(csvs) > 0:
        if len(hdf5s) > 1:
            warn('Found multiple stimlog csv files, using the first: {}'.format(csvs[0]))
        slf_path = csvs[0]
        sl = pd.read_csv(slf_path)
    else:
        sl = None
    if sl is not None:
        stimlog = parsers.create_stimulus_record(trials=len(sl))
        stimlog.update(sl)
    else:
        stimlog = None
    del pkls, hdf5s, csvs, sl
else:
    if session_log is not None:
        stimlog = parsers.parse_log_stim_dots(session_log)
    else:
        stimlog = None
        raise RuntimeError('Could not load stimulus record from session log file.')

# Fill in missing stimulus log values by calculation or from the session log
if stimlog is not None:
    if stimlog['dur_isi_pre'].isnull().values.any():
        if not stimlog['t_isi_i'].isnull().values.any() and not stimlog['t_isi_f'].isnull().values.any():
            stimlog['dur_isi_pre'] = stimlog['t_isi_f'] - stimlog['t_isi_i']
    if stimlog['dur_stim'].isnull().values.any():
        if not stimlog['t_stim_i'].isnull().values.any() and not stimlog['t_stim_f'].isnull().values.any():
            stimlog['dur_stim'] = stimlog['t_stim_f'] - stimlog['t_stim_i']
    if stimlog['dur_isi_post'].isnull().values.any():
        if not stimlog['t_isi_i'].isnull().values.any() and not stimlog['t_isi_f'].isnull().values.any():
            for t in range(len(stimlog['dur_isi_post']) - 1):
                stimlog.at[t, 'dur_isi_post'] = stimlog['t_isi_f'].loc[t + 1] - stimlog['t_isi_i'].loc[t + 1]
            del t
    if stimlog.isnull().values.any() and session_log is not None:
        sl = parsers.parse_log_stim_dots(session_log)
        stimlog.update(sl, overwrite=False)
        del sl

dirlist_eyecal = [d for d in glob(os.path.join(date_path, dirstr_eyecal)) if os.path.isdir(d)]
if len(dirlist_eyecal) > 0:
    ecd_path = dirlist_eyecal[0]
    if len(dirlist_eyecal) > 1:
        warn('Found multiple eye tracking calibration directories, using the first one: {}'.format(ecd_path))
    filelist_eyecal_log = [f for f in glob(os.path.join(ecd_path, filestr_eyecal_log))
                           if os.path.isfile(f)]
    filelist_eyecal_aidata = [f for f in glob(os.path.join(ecd_path, filestr_eyecal_aidata))
                              if os.path.isfile(f)]
else:
    ecd_path = None
    filelist_eyecal_log = []
    filelist_eyecal_aidata = []

if len(filelist_eyecal_log) > 0:
    ec_lf_path = filelist_eyecal_log[0]
    if len(filelist_eyecal_log) > 1:
        warn('Found multiple eye tracking calibration log files, using the first one: {}'.format(ec_lf_path))
    eclf = open(ec_lf_path, 'r')
    eyecal_log = eclf.read()
    eclf.close()
    del eclf
else:
    ec_lf_path = None
    eyecal_log = None

if len(filelist_eyecal_aidata) > 0:
    ec_df_path = filelist_eyecal_aidata[0]
    if len(filelist_eyecal_aidata) > 1:
        warn('Found multiple eye tracking calibration data files, using the first one: {}'.format(ec_df_path))
    with open(ec_df_path, 'rb') as ec_df:
        eyecal_aidata = pickle.load(ec_df)
    del ec_df
else:
    etf_path = None
    eyecal_aidata = None

if eyecal_log is not None:
    if eyecal_aidata is not None:
        eyecal_data = parsers.parse_log_eyecal(eyecal_log, eyecal_aidata)
    else:
        eyecal_data = parsers.parse_log_eyecal(eyecal_log)
else:
    eyecal_data = None

filelist_eyetrack_data = [f for f in glob(os.path.join(session_path, filestr_eyetrack_data)) if os.path.isfile(f)]
if len(filelist_eyetrack_data) > 0:
    etf_path = filelist_eyetrack_data[0]
    if len(filelist_eyetrack_data) > 1:
        warn('Found multiple log files, using the first: {}'.format(etf_path))
    with open(etf_path, 'rb') as etf:
        eyetrk_data = pickle.load(etf)
    del etf
else:
    etf_path = None

dirlist_suite2p = [d for d in glob(os.path.join(session_path, dirstr_suite2p)) if os.path.isdir(d)]
if len(dirlist_suite2p) > 0:
    dirlist_suite2p_idx = 0
    if len(dirlist_suite2p) > 1:
        if np.any([re.match(dirstr_suite2p_pref, os.path.basename(d)) is not None for d in dirlist_suite2p]):
            dirlist_suite2p_idx = np.where([re.match(dirstr_suite2p_pref, os.path.basename(d)) is not None
                                             for d in dirlist_suite2p])[0][0]
        warn('Found multiple suite2p folders, using: {}'.format(os.path.basename(dirlist_suite2p[dirlist_suite2p_idx])))
    s2p_path = dirlist_suite2p[dirlist_suite2p_idx]
    s2p_plane_path = os.path.join(s2p_path, dirstr_suite2p_plane)
    if not os.path.isdir(s2p_plane_path):
        raise RuntimeError('Could not load suite2p plane0 folder.')
else:
    raise RuntimeError('Could not find suite2p folder.')


# Load suite2p data

s2p_iscell = np.load(os.path.join(s2p_plane_path, 'iscell.npy'))
s2p_F = np.load(os.path.join(s2p_plane_path, 'F.npy'))
s2p_stat = np.load(os.path.join(s2p_plane_path, 'stat.npy'), allow_pickle=True)
s2p_ops = np.load(os.path.join(s2p_plane_path, 'ops.npy'), allow_pickle=True).item()
s2p_badframes = np.where(s2p_ops['badframes'])[0]

if 'threshold_cellprob' not in locals():
    threshold_cellprob = 0.0
cellinds = np.where(s2p_iscell[:, 1] >= threshold_cellprob)[0]
inactives = np.where(np.std(s2p_F, axis=1) == 0)[0]
if len(inactives) > 0:
    warn('Excluded {} inactive ROIs: {}'.format(len(inactives), inactives))
cellinds = np.setdiff1d(cellinds, inactives)
ROIs = s2p_stat[cellinds]
Frois = s2p_F[cellinds]
fov_h = s2p_ops['Ly']
fov_w = s2p_ops['Lx']
fov_size = (fov_h, fov_w)  # rows/height/y, columns/width/x
fov_image = s2p_ops['meanImg']
n_ROIs, n_frames = Frois.shape


# %% Process fluorescence signal

# # Inspect fluorescence baseline filters
# filters.plot_example_baselines(Frois, rois=2, frames=1000, framerate=md['framerate'], window=60, include_mpfi=False,
#                                percentile=10, rank=10, sigma=10)

# Calculate baseline fluorescence (F0)
F0_filt_win_sec = 60  # sec
F0_filt_win_frames = round(F0_filt_win_sec * md['framerate'])  # frames
F0 = filters.calculate_baselines(Frois, framerate=md['framerate'], window=F0_filt_win_sec, method='meanbw')

# Compute dF/F and z-scored dF/F
FdF = Frois - F0
FdFF_raw = (Frois - F0) / F0
Fzsc_raw = (Frois - F0 - np.mean(Frois - F0, axis=1)[:, np.newaxis]) / np.std(Frois - F0, axis=1)[:, np.newaxis]

FdFF = FdFF_raw
Fzsc = Fzsc_raw


# %% Estimate SNR

# v_noiselev = np.median(np.abs(np.diff(FdFF_raw)), axis=1) / np.sqrt(md['framerate'])

# # Deconvolve fluorescence signals.
# from oasis.oasis_methods import oasisAR1
# from oasis.functions import deconvolve, estimate_parameters, GetSn
#
# caconc = np.full((n_ROIs, n_frames), np.nan)
# spikes = np.full((n_ROIs, n_frames), np.nan)
# basels = np.full(n_ROIs, np.nan)
# decays = np.full(n_ROIs, np.nan)
# lamdas = np.full(n_ROIs, np.nan)
# sigmas = np.full(n_ROIs, np.nan)
# for r in range(n_ROIs):
#     # FdFF_c[r], _, b, g, lam = deconvolve(FdFF_raw[r], penalty=1)
#     # caconc[r], spikes[r], basels[r], decays[r], lamdas[r] = deconvolve(FdF[r], penalty=0, optimize_g=5)
#     sigmas[r] = GetSn(FdF[r], range_ff=[0.25, 0.5], method='mean')
#     # c[r], _ = oasisAR1(FdF[r].astype('float64'), g=.95)
#     # c[r], _ = oasisAR1(FdF[r], g=.95)
#
# r = 0
# plt.plot(FdF[r, 0:500], 'b', alpha=0.5)
# plt.plot(caconc[r, 0:500] + basels[r], 'm')


# %% Deconvolve fluorescence signals

# Friedrich et al 2017 Paninski https://doi.org/10.1371/journal.pcbi.1005423
# "An AR(1) process models the calcium response to a spike as an instantaneous increase followed by an exponential
# decay. This is a good description when the fluorescence rise time constant is small compared to the length of a
# time-bin, e.g. when using GCaMP6f [36] with a slow imaging rate. For fast imaging rates and slow indicators such
# as GCaMP6s it is more accurate to explicitly model the finite rise time. Typically we choose an AR(2) process,
# though more structured responses (e.g. multiple decay time constants) can also be modeled with higher values for
# the order p."

# import oasis

# n_plot_ROIs = 1
# n_samp_inspect = 1000
# plot_ROIs = np.random.choice(n_ROIs, n_plot_ROIs)
# frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
# frame_end = frame_start + n_samp_inspect

# fig = plt.figure()
# # fig.suptitle('mean response by condition (each trial plotted)')
# axes = fig.subplots(nrows=n_plot_ROIs, ncols=1)
# for r in range(n_plot_ROIs):
#     ridx = plot_ROIs[r]
#     # frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
#     # frame_end = frame_start + n_samp_inspect
#     Fr = Frois[ridx, frame_start:frame_end]

#     # Fr = Frois[ridx, frame_start:frame_end]
#     Fr_dFF = (Fr - np.mean(Fr)) / np.mean(Fr)
#     # F0_rnk = filters.rank_order_filter(Fr, p=filter_percentile, n=round(filter_window * md['framerate']))
#     # Fr_dFF_rnk = (Fr - F0_rnk) / F0_rnk
#     # F0_pct = filters.percentile_filter_1d(Fr, p=filter_percentile, n=round(filter_window * md['framerate']))
#     # Fr_dFF_pct = (Fr - F0_pct) / F0_pct
#     # F0_rnkbw = filters.butterworth_filter(F0_rnk, fs=md['framerate'], p=filter_percentile)
#     # Fr_dFF_rnkbw = (Fr - F0_rnkbw) / F0_rnkbw
#     # F0_pctbw = filters.butterworth_filter(Fr, fs=md['framerate'], p=filter_percentile)
#     # Fr_dFF_pctbw = (Fr - F0_pctbw) / F0_pctbw
#     # # F0_med = np.median(np.lib.stride_tricks.sliding_window_view(Fr, (round(filter_window * md['framerate']),)), axis=1)
#     # # Fr_dFF_med = (Fr - F0_med) / F0_med
#     # F0_ma = np.convolve(Fr, np.ones(round(filter_window * md['framerate'])), mode='same') / round(filter_window * md['framerate'])
#     # Fr_dFF_ma = Fr_dFF - np.convolve(Fr_dFF, np.ones(round(filter_window * md['framerate'])), mode='same') / round(
#     #     filter_window * md['framerate'])

#     # y: observed fluorescence
#     # c: calcium concentration
#     # s: neural activity / spike train
#     # b: baseline
#     # "To produce calcium trace c, spike train s is filtered with the inverse filter
#     # of g, an infinite impulse response
#     # h, c = s * h."
#     # decay factor γ, regularization parameter λ, data y, sigma noise

#     oasisL0_c, oasisL0_s, oasisL0_b, oasisL0_g, oasisL0_lam = oasis.functions.deconvolve(Fr, penalty=0)
#     Fr_oasisL0 = oasisL0_c + oasisL0_b
#     Fr_dFF_oasisL0 = (Fr_oasisL0 - oasisL0_b) / oasisL0_b

#     ymin = np.min(Fr_dFF)
#     ymax = np.max(Fr_dFF)
#     xs = range(n_samp_inspect)

#     if n_plot_ROIs > 1:
#         ax = axes[r]
#     else:
#         ax = axes
#     ax.set_ylabel('dF/F')
#     ax.set_xlabel('Frames')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
#     ax.set_xticks([0, n_samp_inspect])
#     ax.set_xticklabels([frame_start, frame_end])
#     ax.plot(xs, Fr_dFF, label='FdFF', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_rnk, label='FdFF_rnk', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_pct, label='FdFF_pct', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_rnkbw, label='FdFF_rnkbw', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_pctbw, label='FdFF_pctbw', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_ma, label='Fr_dFF_med', linewidth=0.5, alpha=0.5, zorder=3)
#     # ax.plot(xs, Fr_dFF_ma, label='FdFF_ma', linewidth=0.5, alpha=0.5, zorder=3)
#     ax.plot(xs, Fr_dFF_oasisL0, label='FdFF_oasisL0', color='g', linewidth=1, alpha=0.8, zorder=10)

#     ax.legend(ncol=len(ax.get_lines()), frameon=False, loc=(.02, .85))
# plt.show()

# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
# from oasis.plotting import simpleaxis
# from oasis.oasis_methods import oasisAR1, oasisAR2

# c, s, b, g, lam = deconvolve(Fr)  # , penalty=1)


# %% Plot eye-tracking calibration results

if plot_eyecal and eyecal_data is not None:
    # f = plt.figure()
    # ecx, ecy = eyecal_data['zero']['AIdata']
    # plt.scatter(ecx, ecy, s=1, c='m')
    # ecx, ecy = eyecal_data['circ']['data'][0]['AIdata']
    # plt.scatter(ecx, ecy, s=1, c='k')
    # plt.show()

    f = plt.figure()
    circ = np.array([])
    for trl in range(eyecal_data['circ']['n_trials']):
        ecx, ecy = np.transpose(eyecal_data['circ']['data'][trl]['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            circ = eyecal_data['circ']['data'][trl]['AIdata']
        else:
            circ = np.concatenate((circ, eyecal_data['circ']['data'][trl]['AIdata']))
    f.show()

    f = plt.figure()
    circ = np.array([])
    plt.scatter(circ.T[0], circ.T[1], s=1)
    plt.scatter(np.median(circ.T[0]), np.median(circ.T[1]), s=5, c='m')
    ax = plt.gca()
    from matplotlib.patches import Ellipse
    c1 = Ellipse((np.median(circ.T[0]), np.median(circ.T[1])),
                 width=np.std(circ.T[0]), height=np.std(circ.T[1]), lw=2, edgecolor='m', fc='None')
    ax.add_patch(c1)
    f.show()

    # # Plot point density
    # from matplotlib.patches import Ellipse
    # from scipy.stats import gaussian_kde
    # etx = circ.T[0]
    # ety = circ.T[1]
    # etxy = np.vstack([etx, ety])
    # et_ptdensity = gaussian_kde(etxy)(etxy)
    # idx = et_ptdensity.argsort() # Sort by density, so that the densest points are plotted last
    # etx, ety, et_ptdensity = etx[idx], ety[idx], et_ptdensity[idx]
    # fig, ax = plt.subplots()
    # ax.scatter(etx, ety, s=1, c=et_ptdensity)
    # ax.scatter(np.median(etx), np.median(ety), s=5, c='m')
    # c1 = Ellipse((np.median(etx), np.median(ety)), width=np.std(etx),
    #              height=np.std(ety), lw=2, edgecolor='m', fc='None')
    # ax.add_patch(c1)
    # plt.show()

    f = plt.figure()
    grdf = np.array([])
    for trl in range(eyecal_data['grdf']['n_trials']):
        ecx, ecy = np.transpose(eyecal_data['grdf']['data'][trl]['face']['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            grdf = eyecal_data['grdf']['data'][trl]['face']['AIdata']
        else:
            grdf = np.concatenate((grdf, eyecal_data['grdf']['data'][trl]['face']['AIdata']))
    plt.show()
    f = plt.figure()
    plt.scatter(grdf.T[0], grdf.T[1], s=1)
    plt.scatter(np.median(grdf.T[0]), np.median(grdf.T[1]), s=5, c='m')
    # ax = plt.gca()
    # from matplotlib.patches import Ellipse
    # c1 = Ellipse((np.median(grdf.T[0]), np.median(grdf.T[1])), width=np.std(grdf.T[0]),
    #              height=np.std(grdf.T[1]), lw=2, edgecolor='m', fc='None')
    # ax.add_patch(c1)
    plt.show()
    from scipy.stats import gaussian_kde
    etx = grdf.T[0]
    ety = grdf.T[1]
    etxy = np.vstack([etx, ety])
    et_ptdensity = gaussian_kde(etxy)(etxy)
    idx = et_ptdensity.argsort()  # Sort by density, so that the densest points are plotted last
    etx, ety, et_ptdensity = etx[idx], ety[idx], et_ptdensity[idx]
    fig, ax = plt.subplots()
    ax.scatter(etx, ety, s=1, c=et_ptdensity)
    ax.scatter(np.median(etx), np.median(ety), s=5, c='m')
    f.show()

    f = plt.figure()
    crse = np.array([])
    crsecv = np.array([])
    for trl in range(eyecal_data['crse']['n_trials']):
        ecx, ecy = np.transpose(eyecal_data['crse']['data'][trl]['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            crse = eyecal_data['crse']['data'][trl]['AIdata']
            crsecv = eyecal_data['crse']['data'][trl]['cvals']
        else:
            crse = np.concatenate((crse, eyecal_data['crse']['data'][trl]['AIdata']))
            crsecv = np.vstack((crsecv, eyecal_data['crse']['data'][trl]['cvals']))
    plt.show()
    f = plt.figure()
    plt.scatter(crse.T[0], crse.T[1], s=1)
    plt.scatter(crsecv.T[0], crsecv.T[1], s=10)
    plt.scatter(np.median(crse.T[0]), np.median(crse.T[1]), s=5, c='m')
    # ax = plt.gca()
    # from matplotlib.patches import Ellipse
    # c1 = Ellipse((np.median(grdf.T[0]), np.median(grdf.T[1])), width=np.std(grdf.T[0]),
    #              height=np.std(grdf.T[1]), lw=2, edgecolor='m', fc='None')
    # ax.add_patch(c1)
    plt.show()
    from scipy.stats import gaussian_kde
    etx = crse.T[0]
    ety = crse.T[1]
    etxy = np.vstack([etx, ety])
    et_ptdensity = gaussian_kde(etxy)(etxy)
    idx = et_ptdensity.argsort()  # Sort by density, so that the densest points are plotted last
    etx, ety, et_ptdensity = etx[idx], ety[idx], et_ptdensity[idx]
    fig, ax = plt.subplots()
    ax.scatter(etx, ety, s=1, c=et_ptdensity)
    ax.scatter(np.median(etx), np.median(ety), s=5, c='m')
    f.show()

    # # Plot at eye tracking data
    # # Take a look at this for density plotting:
    # # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
    # f = plt.figure()
    # etx, ety = np.transpose(eyetrk_data[:, :2])
    # plt.scatter(eyetrk_data[:, 0], eyetrk_data[:, 1], s=1)
    # plt.show()
    # # [np.std(eyetrk_data[:, d]) for d in range(0, eyetrk_data.shape[1])]

    # from scipy.stats import gaussian_kde
    #
    # # Calculate point density
    # etxy = np.vstack([etx, ety])
    # et_ptdensity = gaussian_kde(etxy)(etxy)
    #
    # # Sort the points by density, so that the densest points are plotted last
    # idx = et_ptdensity.argsort()
    # etx, ety, et_ptdensity = etx[idx], ety[idx], et_ptdensity[idx]
    #
    # fig, ax = plt.subplots()
    # ax.scatter(etx, ety, c=et_ptdensity, s=1)
    # plt.show()


# %% Process stimulus information

# Fix off-by-one issue caused by acqusition frame counter ('acqfr') index starting at 1 rather than 0.
if correct_acqfr_index:
    acqfr_keys = [c for c in stimlog.columns if 'acqfr' in c]
    for ak in acqfr_keys:
        stimlog[ak] = stimlog[ak] - 1

# Determine basic stimulus presentation information
#   For sessions using older stimulus code, the exact number of stimulus or ISI frames could
#   vary slightly because the stim start was not locked to an acqusition frame increment.
dur_stim = np.round(np.mean(stimlog['dur_stim'].values), 2)
dur_isi = np.round(np.min(stimlog['dur_isi_pre'].values), 2)
dur_trial = dur_isi + dur_stim + dur_isi
if md['stim_locked_to_acqfr']:
    n_samp_stim = np.bincount(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).argmax()
else:
    if np.bincount(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).nonzero()[0][0] != 0:
        n_samp_stim = np.bincount(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).nonzero()[0][0]
    else:
        n_samp_stim = np.bincount(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).nonzero()[0][1]
if np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][0] != 0:
    n_samp_isi = np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][0]
else:
    n_samp_isi = np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][1]
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

# Calculate the timing mismatch (contraction) introduced by rounding stim and/or isi frame samples down
acqfr_dilation_factor = (dur_trial * md['framerate']) / (n_samp_trial - 1)

# *** TODO: take only conds with supported presentations
if np.unique(stimlog[:]['stim_mode'].values).size != 1:
    warn('More than one stimulus mode was presented and is not yet fully supported.')

n_metrics = len(metrics)
n_conds = len(np.unique(stimlog['cond'].values))
n_trials = len(stimlog)
n_reps = int(len(stimlog) / n_conds)

if len(np.unique(stimlog['acqfr_stim_i'])) != len(stimlog['acqfr_stim_i']):
    warn('Acquisition was started after stimulus, interrupted, or stopped before stimulus.')


# %% Organize fluorescence signals into a structured array data table

dlist = [('cond', 'S8'),
         ('stimulus', object),
         ('cat', 'S8'),
         ('dir', 'f8'),
         ('coherence', 'f8'),
         ('speed', 'f8'),
         ('contrast', 'f8')]

for m in metrics:
    dlist.append((m, 'f8', (n_ROIs, n_reps, n_samp_trial)))
del m

data = np.zeros(n_conds, dtype=dlist)
del dlist

for m in metrics:
    data[m] = np.nan
del m

# 'dots_translation_dir': None,
# 'dots_opticflow_dir': None,
# 'dots_rotation_dir': None,

# Currently supported dot subclasses:
# 'translation'  # ... rotation and optic flow and etc possible

trials_movement = []
for c in range(n_conds):
    tmp_cond = None
    tmp_class = None
    tmp_subclass = None
    
    tmp_settings = dict()
    tmp_settings['units'] = None
    tmp_settings['nDots'] = None
    tmp_settings['coherence'] = None
    tmp_settings['fieldPos'] = None
    tmp_settings['fieldSize'] = None
    tmp_settings['fieldShape'] = None
    tmp_settings['dotSize'] = None
    tmp_settings['dotLife'] = None
    tmp_settings['dir'] = None
    tmp_settings['speed'] = None
    tmp_settings['color'] = None
    tmp_settings['colorSpace'] = None
    tmp_settings['opacity'] = None
    tmp_settings['contrast'] = None
    tmp_settings['element'] = None
    tmp_settings['signalDots'] = None
    tmp_settings['noiseDots'] = None
    
    if np.unique(stimlog[stimlog['cond'] == c]['stim_class'].values).size == 1:
        tmp_class = np.unique(stimlog[stimlog['cond'] == c]['stim_class'].values)[0]
    else:
        warn('Not all dot motion subclasses were the same for condition {}.'.format(c))
        tmp_class = None
    
    if np.unique(stimlog[stimlog['cond'] == c]['stim_subclass'].values).size == 1:
        tmp_subclass = np.unique(stimlog[stimlog['cond'] == c]['stim_subclass'].values)[0]
    else:
        warn('Not all dot motion subclasses were the same for condition {}.'.format(c))
        tmp_subclass = None

    if np.unique(stimlog[stimlog['cond'] == c]['dots_dir'].values).size == 1:
        tmp_dir = np.unique(stimlog[stimlog['cond'] == c]['dots_dir'].values)[0]
    else:
        raise Exception('Not all dot directions were the same for condition {}.'.format(c))
    
   
    for k in tmp_settings.keys():
        ks = 'dots_' + k
        if ks in stimlog[stimlog['cond'] == c].columns:
            val = stimlog[stimlog['cond'] == c][ks].values
            if val.dtype == np.dtype('O') and isinstance(val[0], np.ndarray):
                val = np.array(list(val), dtype=float)
            if np.all(val[0] == val):
                tmp_settings[k] = val[0]
            else:
                warn('Not all values were the same for condition {} setting {}.'.format(c, ks))
                tmp_settings[k] = None
    del k, ks, val
    
    if tmp_class == 'dots':
        match tmp_subclass:
            case 'translation':
                tmp_cond = bytes('dt' + '{:03.1f}'.format(tmp_settings['dir']).replace('.','p'), 'ascii')
            case _, None:
                warn('Unsupported stimulus skipped. cond {} class {} subclass {}'.format(tmp_cond, tmp_class, tmp_subclass))
                continue

    data[c]['cond'] = tmp_cond    
    data[c]['dir'] = tmp_settings['dir']
    data[c]['coherence'] = tmp_settings['coherence']
    data[c]['speed'] = tmp_settings['speed']
    data[c]['contrast'] = tmp_settings['contrast']
    data[c]['stimulus'] = StimulusDots(tmp_cond, 
                                       tmp_subclass, 
                                       units=tmp_settings['units'], 
                                       nDots=tmp_settings['nDots'], 
                                       coherence=tmp_settings['coherence'], 
                                       fieldPos=tmp_settings['fieldPos'], 
                                       fieldSize=tmp_settings['fieldSize'], 
                                       fieldShape=tmp_settings['fieldShape'], 
                                       dotSize=tmp_settings['dotSize'], 
                                       dotLife=tmp_settings['dotLife'], 
                                       dir=tmp_settings['dir'], 
                                       speed=tmp_settings['speed'], 
                                       color=tmp_settings['color'], 
                                       colorSpace=tmp_settings['colorSpace'], 
                                       opacity=tmp_settings['opacity'], 
                                       contrast=tmp_settings['contrast'], 
                                       element=tmp_settings['element'], 
                                       signalDots=tmp_settings['signalDots'], 
                                       noiseDots=tmp_settings['noiseDots'])
    
    for t in range(n_reps):
        fr_start = stimlog[stimlog['cond'] == c].iloc[t]['acqfr_stim_i'] - n_samp_isi
        if isinstance(fr_start, float):
            if fr_start.is_integer():
                fr_start = int(fr_start)
            else:
                raise ValueError('Non-integer acquisition frame index.')
        fr_end = stimlog[stimlog['cond'] == c].iloc[t]['acqfr_stim_i'] + n_samp_stim + n_samp_isi
        if isinstance(fr_end, float):
            if fr_end.is_integer():
                fr_end = int(fr_end)
            else:
                raise ValueError('Non-integer acquisition frame index.')

        # Identify trials (condition-repeat pairs) with movement (via suite2p badframes)
        if np.any(np.isin(range(fr_start, fr_end + 1), s2p_badframes)):
            trials_movement.append((c, t))

        if fr_start < 0 and t == 0:
            # TODO exclude trials at start of imaging if ISI is too short
            warn('Period before first trial was shorter than inter-stimulus interval. ' +
                 'Copied first present value to prevent error. ' +
                 'But this trial could instead be excluded.')
            n_samp_miss = abs(fr_start)
            if 'FdFF' in metrics:
                data[c]['FdFF'][:, t, 0:n_samp_miss] = np.array([FdFF_raw[:, 0],] * n_samp_miss).T
            if 'Fzsc' in metrics:
                data[c]['Fzsc'][:, t, 0:n_samp_miss] = np.array([Fzsc_raw[:, 0],] * n_samp_miss).T
            fr_start = 0
            if 'FdFF' in metrics:
                data[c]['FdFF'][:, t, n_samp_miss:n_samp_trial] = FdFF_raw[:, fr_start:fr_end]
            if 'Fzsc' in metrics:
                data[c]['Fzsc'][:, t, n_samp_miss:n_samp_trial] = Fzsc_raw[:, fr_start:fr_end]
            del n_samp_miss
            continue
        if fr_end > n_frames:
            # TODO: support throwing away trials after imaging stops
            raise RuntimeError('Imaging was stopped before stimulus. Handling this is not yet implemented.')
        if 'FdFF' in metrics:
            data[c]['FdFF'][:, t, :] = FdFF_raw[:, fr_start:fr_end]
        if 'Fzsc' in metrics:
            data[c]['Fzsc'][:, t, :] = Fzsc_raw[:, fr_start:fr_end]
    for m in metrics:
        if np.any(np.isnan(data[c][m])):
            warn('Some {} values in cond {} are NaNs'.format(m, c))
del tmp_cond, tmp_class, tmp_subclass, tmp_settings

# Sanity check for NaN values after loading data
for m in metrics:
    if np.any(np.isnan(data[m])):
        raise Exception('Found NaNs after loading {} data.'.format(m))
del m


# %% Sort data table according to template, define categories and conditions, and perform exclusions

sort_by_cond = lambda x: (x[1].dir,
                          x[1].coherence,
                          x[1].speed,
                          x[1].contrast,
                          x[1].condition.decode().lower())
# sort_by_cond = lambda x: (np.where(template == x[1].category)[0][0]
#                           if np.where(template == x[1].category)[0].size > 0
#                           else np.iinfo(np.where(template == x[1].category)[0].dtype).max,
#                           np.abs(x[1].roll),
#                           x[1].roll,
#                           x[1].yaw,
#                           x[1].condition.decode().lower())
# sort_by_cat = lambda x: (np.where(template == x[1])[0][0]
#                          if np.where(template == x[1])[0].size > 0
#                          else np.iinfo(np.where(template == x[1])[0].dtype).max)

data = data[[i for i, _ in sorted(enumerate(data['stimulus']), key=sort_by_cond)]]

# categories = pd.unique(data['cat'])  # Use pandas instead of numpy to avoid automatic sorting
# # categories = categories[[i for i, _ in sorted(enumerate(categories), key=sort_by_cat)]]
# cat_to_catidx = {k: i for i, k in enumerate(categories)}
# n_cats = len(categories)
conditions = pd.unique(data['cond'])  # Use pandas instead of numpy to avoid automatic sorting
# conditions = conditions[[i for i, _ in sorted(enumerate(conditions), key=sort_by_cond)]]
cond_to_condidx = {k: i for i, k in enumerate(conditions)}
# cat_to_cond = {cat: [cnd for cnd in data[data['cat'] == cat]['cond']] 
#                for cat in categories}
# cat_to_condidx = {cat: [cond_to_condidx[cnd] for cnd in data[data['cat'] == cat]['cond']] 
#                   for cat in categories}
# cond_to_cat = {cnd: cat for cat, cndlist in cat_to_cond.items() for cnd in cndlist}
# condidx_to_cat = {icnd: cat for cat, icndlist in cat_to_condidx.items() for icnd in icndlist}
directions = pd.unique(data['dir'])

if n_conds != conditions.shape[0]:
    u, c = np.unique(data['cond'], return_counts=True)
    mult = u[c > 1]
    warn('Some different stimulus conditions were combined into the same condition.' +
         'May not be able to handle that yet...')
    del u, c, mult


# Exclude trials where with various problems

trials_exclude = []
excluded = {}
excluded_movement = {}
excluded_preacq = {}

exclude_by_acqstart = True
trials_preacq = []
if exclude_by_acqstart:
    tidx_acqstart = []
    if np.where(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i'] < n_samp_stim)[0].size > 0:
        tidx_acqstart.append(np.where(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i'] < n_samp_stim)[0].max())
    if np.any(np.where(np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']) != 0)[0] < n_samp_isi):
        tidx_acqstart.append(np.where(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i'] < n_samp_isi)[0].max())
    if tidx_acqstart:
        tidx_acqstart = max(tidx_acqstart)
        warn('Acquisition frames for some trials were less than the expected number. '
             'Assuming that stimulus was started before acquisition, '
             'excluding first {} trials. '.format(tidx_acqstart) +
             'Also adjusting acqfr values to start after excluded trials.')
        if stimlog.iloc[tidx_acqstart]['acqfr_isi_f'] - stimlog.iloc[tidx_acqstart]['acqfr_isi_i'] > 0:
            # Acquisition seems to have started during ISI period of last exluded trial.
            acqfr_diff = stimlog.iloc[tidx_acqstart]['acqfr_isi_i']
        elif stimlog.iloc[tidx_acqstart]['acqfr_stim_f'] - stimlog.iloc[tidx_acqstart]['acqfr_stim_i'] > 0:
            # Acquisition seems to have started during stim period of last exluded trial.
            acqfr_diff = stimlog.iloc[tidx_acqstart]['acqfr_stim_i']
        if float(acqfr_diff).is_integer():
            acqfr_diff = int(acqfr_diff)
        else:
            warn('Unexpectedly got non-integer value for an acqfr, rounding to int.')
            acqfr_diff = int(acqfr_diff)

        acqfr_keys = [c for c in stimlog.columns if 'acqfr' in c]
        for ak in acqfr_keys:
            stimlog[ak] = stimlog[ak] - acqfr_diff

        cond_count = {}
        for t in range(tidx_acqstart):
            cnd = stimlog.iloc[t]['cond']
            if cnd in cond_count:
                cond_count[cnd] += 1
            else:
                cond_count[cnd] = 1
            rep = cond_count[cnd]
            trials_preacq.append((cnd, cond_count[cnd]))
        del cond_count

        for cnd, rep in trials_preacq:
            trials_exclude.append((cnd, rep))
            if cnd not in excluded_preacq:
                excluded_preacq[cnd] = []
            excluded_preacq[cnd].append(rep)
        print('Excluding {} trials presumably presented before acquisition start.'.format(len(trials_preacq)))
        for ek in excluded_preacq:
            print('  {}/{} excluded for {} ({})'.format(len(excluded_preacq[ek]), n_reps,
                                                        conditions[ek].decode(), condidx_to_cat[ek].decode()))

if exclude_by_movement and len(trials_movement) > 0:
    for cnd, rep in trials_movement:
        trials_exclude.append((cnd, rep))
        if cnd not in excluded_movement:
            excluded_movement[cnd] = []
        excluded_movement[cnd].append(rep)
    print('Excluding {} trials due to movement (via suite2p badframes).'.format(len(trials_movement)))
    for ek in excluded_movement:
        print('  {}/{} excluded for {} ({})'.format(len(excluded_movement[ek]), n_reps,
                                                    conditions[ek].decode(), condidx_to_cat[ek].decode()))

excluded_blinks = {}
# TODO * * * add exclusion by blinks
# stimlog['ai_stim_f'] - stimlog['ai_stim_i']
# eyetrk_data

# Set excluded trial data table values to NaN.
for cnd, rep in trials_exclude:
    for m in metrics:
        data[cnd][m][:, rep, :] = np.nan
    if cnd not in excluded:
        excluded[cnd] = []
    excluded[cnd].append(rep)

print('In total, {} trials excluded.'.format(len(trials_movement)))
for ek in excluded:
    print('  {}/{} excluded for {} ({})'.format(len(excluded[ek]), n_reps,
                                                conditions[ek].decode(), condidx_to_cat[ek].decode()))

# # F__by_cond = [roi, cond, t, F]
# FdFF_by_cond = np.full([n_ROIs, n_conds, n_trials, n_samp_isi+n_samp_stim+n_samp_isi], np.nan)
# FdFF_by_cond_top_decile = np.full([FdFF.shape[0], n_conds], np.nan)
# for c in range(n_conds):
#     for r in range(n_ROIs):
#         for t in range(n_trials):
#             fr_start = acqfr_by_conds[c][t] - n_samp_isi
#             fr_end = acqfr_by_conds[c][t] + n_samp_stim + n_samp_isi
#             if fr_start < 0 and t == 0:
#                 warn('Period before first trial was shorter than inter-stimulus interval.' + \
#                       'Copied first present value to prevent error. ' + \
#                       'But in the future this trial should be excluded.')
#                 n_missing = abs(fr_start)
#                 FdFF_by_cond[r, c, t, 0:n_missing] = np.array([FdFF[r, 0],] * n_missing).transpose()
#                 fr_start = 0
#                 FdFF_by_cond[r, c, t, n_missing:n_samp_trial] = FdFF[r, fr_start:fr_end]
#                 continue
#             FdFF_by_cond[r,c,t,:] = FdFF[r, fr_start:fr_end]
# FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,n_samp_isi:(n_samp_isi+n_samp_stim)]
# FdFF_by_cond_meanR = np.mean(FdFF_by_cond_Rstim, axis=2) #mean across trials and selecting stimulus window
# #FFdFF_by_cond_meanR = FdFF_by_cond_Rstim.reshape([FdFF_by_cond_Rstim.shape[0], FdFF_by_cond_Rstim.shape[1], -1]) # susceptible to noise


# %% Compute statistics for each ROI

idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)


# FdFF_by_cond[:,:,:,n_samp_isi:(n_samp_isi+n_samp_stim)]
# FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,n_samp_isi:(n_samp_isi+n_samp_stim)]

# dsiT(__by_roi) = [roi, [dsi, T]]
Ts = np.repeat(directions, n_reps)
# distxs = np.repeat(conds, n_reps)
Rs = np.full([n_ROIs, (n_conds * n_reps)], np.nan)
dsiT = np.full([n_ROIs, 2], np.nan)
# for r in range(10):
#     Rs[r] = np.ravel(np.mean(FdFF_by_cond_Rstim[r], axis=2))
#     dsiT[r] = calculate_dsi(Ts, Rs[r], plotting=True)
for r in range(n_ROIs):
    # Rs[r] = np.ravel(np.mean(FdFF_by_cond_Rstim[r], axis=2))
    Rs[r] = np.ravel(np.mean(data['FdFF'][:, r, :, idx_stim], axis=0))
    dsiT[r] = calculate_dsi(Ts, Rs[r])


# %% Plot one experimentally measured distribution used for DSI fitting
#
# plt.figure()
# r = np.random.randint(0, n_ROIs)
# plt.scatter(Ts,
#             Rs[r],  # mean of stimon frames
#             s=4,
#             facecolors='none',
#             edgecolors='k')
# plt.plot()


# %% Define ROIs as tuned or untuned
print('DSI tuning threshold: {}' .format(threshold_dsi))
tunidx_dsi = dsiT[:, 0]
DSI = dsiT[:, 0]
Tprefs = dsiT[:, 1]
Tprefs_norm = Tprefs / 360  # normalized to [0,1] range
tunidx_dsi_argsrt = np.argsort(tunidx_dsi)[::-1]
n_ROIs_tuned = np.argwhere(tunidx_dsi[tunidx_dsi_argsrt] >= threshold_dsi).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


# %% Plot histogram for the number of ROIs with corresponding tuning values

f_hist0 = plt.figure()
plt.hist(tunidx_dsi, bins=100)
plt.xlabel('Direction-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([0, 1])
plt.axvline(threshold_dsi, color='m')
# plt.axvline(-dsi_tuning_thresh, color='m')
f_hist0.show()
if saving:
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(threshold_dsi).replace('.', 'p') + '.svg'
    f_hist0.savefig(save_path + os.path.sep + save_name, dpi=plt.rcParams['figure.dpi'], transparent=True)
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(threshold_dsi).replace('.', 'p') + '.png'
    f_hist0.savefig(save_path + os.path.sep + save_name, dpi=plt.rcParams['figure.dpi'], transparent=True)

#
# f_hist1 = plt.figure()
# plt.hist(Tprefs[DSI >= dsi_tuning_thresh], bins=100)
# plt.xlabel('Preferred Direction (rad)')
# plt.ylabel('ROIs')
# plt.xlim([0, 1])
# # plt.axvline(Tprefs, color='m')
# # plt.axvline(-dsi_tuning_thresh, color='m')
# f_hist1.show()


# %% Plot tuning map

import plots

plots.plot_overlays_orig(ROIs, Tprefs_norm, DSI, tuning_thresh=threshold_dsi, title=title_str,
                         fov_size=fov_size, circular=True, ref_image=fov_image, save_path=save_path)


# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) if not np.isnan(h) else (0, 0, 0)
#                        for h in np.divide([supcat_to_supcatidx[catidx_to_supcat[icat]]
#                                            if icat in catidx_to_supcat else np.nan
#                                            for icat in stats_df[m]['peak_cat_idx'].values.astype(int)],
#                                           n_supcats)])
ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)  # if not np.isnan(h) else (0, 0, 0)
                       for h in Tprefs_norm])


# resp_vect_FOB = np.array([muR_F[m], muR_O[m], muR_B[m]]).swapaxes(0, 1)
# # Subtract the value corresponding to the least responsive category to make it relative
# # (otherwise, an ROI that responds to all categories would show up as white)
# resp_vect_FOB_rel = np.subtract(resp_vect_FOB.T, np.min(resp_vect_FOB, axis=1)).T

# # Saturate colors at specified value (corresponding to the across-stimulus mean of trial-averaged responses)
# ROI_colors_saturateval = 0.2
# ROI_colors = resp_vect_FOB_rel / ROI_colors_saturateval
# ROI_colors[ROI_colors > 1] = 1


# ...only for ROIs with |DSI| >= threshold
above_threshold = np.where(DSI >= threshold_dsi)[0]
sn = save_pfix + '__ROIplot_ColorByPreferredDotMotionDirection' + \
    '_threshDSI{:0.2f}'.format(threshold_dsi).replace('.', 'p')
    # '_max{}{:0.2f}'.format(m, ROI_colors_saturateval).replace('.', 'p') + \
sp = os.path.join(save_path, sn + save_ext) if saving else ''
plots.plot_overlays_roi(ROIs[above_threshold],
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip=None, rotate=0,
                        # flip='lr', rotate=-90,
                        title='preferred direction,\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dsi),
                        save_path=sp)


# %% Plot preferred direction difference relative to distance for every ROI pair


# TODO *** note that the >= might not be general (e.g. with FSI)
tuned_index = DSI >= threshold_dsi
ROIs_tuned = ROIs[tuned_index]
tuning_tuned = Tprefs[tuned_index]
tuning_mag_tuned = DSI[tuned_index]

if Tprefs_norm.max() > 1:
    warn('Provided tuning index has out-of-range values > 1.')

assert len(ROIs_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
n_regions_tuned = len(ROIs_tuned)


roi_centers_px = np.empty([n_ROIs_tuned, 2])
roi_centers_um = np.empty([n_ROIs_tuned, 2])
roi_colors = np.empty([n_ROIs_tuned, 3])
for r in range(n_ROIs_tuned):
    region = ROIs_tuned[r]
    rxs = region['xpix']
    rys = region['ypix']
    rxys = np.array(list(zip(rxs, rys)))
    roi_centers_px[r] = np.average(rxys, axis=0)
    roi_centers_um[r] = md['fov']['resolution_umpx'] * roi_centers_px[r]
    roi_colors[r] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
    # if r > 0:
    #     roi_dists[r - 1] = np.sqrt((roi_centers[r, 0] - last_roi_center[0])**2 + (roi_centers[r, 1] - last_roi_center[1])**2)
    #     roi_theta_diffs_degs[r - 1] = np.abs(np.degrees(tuning_tuned[r]) - np.degrees(last_roi_theta))
    # last_roi_center = roi_centers[r]
    # last_roi_theta = np.radians(tuning_tuned[r])

# direction difference (deg) vs distance (um)
# import itertools
# roi_dists = scipy.spatial.distance.cdist(roi_centers, roi_centers, 'euclidean')
# roi_tuning_diffs = np.array([abs(a - b) % 180 for (a, b) in itertools.product(tuning_tuned_deg, tuning_tuned_deg)])
# roi_tuning_diffs = np.array([abs(a - b) % 180 for (a, b) in itertools.permutations(tuning_tuned, 2)])

roi_distances_um = np.empty([n_ROIs_tuned, n_ROIs_tuned])
roi_tuning_differences = np.empty([n_ROIs_tuned, n_ROIs_tuned])
for r1 in range(n_ROIs_tuned):
    r1_c = roi_centers_um[r1]
    r1_t = tuning_tuned[r1]
    for r2 in range(n_ROIs_tuned):
        r2_c = roi_centers_um[r2]
        r2_t = tuning_tuned[r2]
        roi_distances_um[r1, r2] = np.sqrt((r1_c[0] - r2_c[0])**2 + (r1_c[1] - r2_c[1])**2)
        # TODO: make sure this makes sense
        roi_tuning_differences[r1, r2] = np.abs(r1_t - r2_t) % 180

# TODO: CHECK THESE
np.allclose(roi_distances_um, roi_distances_um.T)
np.allclose(roi_tuning_differences, roi_tuning_differences.T)

assert roi_distances_um.shape == roi_tuning_differences.shape
rinds, cinds = np.triu_indices_from(roi_distances_um, k=1)
distprefs = np.array([[roi_distances_um[r, c], roi_tuning_differences[r, c]] for r, c in zip(rinds, cinds)])

if 'resolution_umpx' in md['fov']:
    if 'w_um' not in md['fov'] or 'h_um' not in md['fov']:
        md['fov']['w_um'] = md['fov']['w_px'] * md['fov']['resolution_umpx'][0]
        md['fov']['h_um'] = md['fov']['h_px'] * md['fov']['resolution_umpx'][1]

fov_diagonal = np.sqrt(md['fov']['w_um']**2 + md['fov']['h_um']**2)
if np.any(distprefs > fov_diagonal):
    warn('Distance between some ROIs exceeds expected FOV diagonal.')


def dir_dist_dep_exp_equation(params, x):
    # based on Pattadkal et al Priebe 2022 bioRxiv
    #     https://doi.org/10.1101/2022.06.23.497220
    # y: fitted direction difference
    # x: distance between cells
    C = params[0]  # saturation value
    A = params[1]  # start value
    k = params[2]  # decay space constant
    y = C - (A * np.exp(-k * x))
    return y

def dir_dist_dep_exp_equation_wcos(params, x):
    # based on Pattadkal et al Priebe 2022 bioRxiv
    #     https://doi.org/10.1101/2022.06.23.497220
    # y: fitted direction difference
    # x: distance between cells
    C = params[0]  # saturation value
    A = params[1]  # start value
    k = params[2]  # decay space constant
    B = params[3]
    d = params[4]
    e = params[5]
    y = C - (A * np.exp(-k * x)) - B * np.cos((x + d) / e)
    return y

def dirdist_objective(params, xs, measured_dirdiff):
    predicted_dirdiff = dir_dist_dep_exp_equation(params, xs)
    mse = np.square(np.subtract(predicted_dirdiff, measured_dirdiff)).mean()
    return mse

def dirdist_objective_wcos(params, xs, measured_dirdiff):
    predicted_dirdiff = dir_dist_dep_exp_equation_wcos(params, xs)
    mse = np.square(np.subtract(predicted_dirdiff, measured_dirdiff)).mean()
    return mse

# Calculate median values for 25 um distance bins
w_bin_um = 25
n_bins = int(np.ceil(distprefs[:, 0].max() / w_bin_um))
bin_edges = np.linspace(0, n_bins * w_bin_um, n_bins + 1)
bin_centers = np.linspace(w_bin_um / 2, (n_bins * w_bin_um) - (w_bin_um / 2), n_bins)
bin_medians, _, _ = binned_statistic(distprefs[:, 0], distprefs[:, 1], statistic='median', bins=bin_edges)
bin_stds, _, _ = binned_statistic(distprefs[:, 0], distprefs[:, 1], statistic='std', bins=bin_edges)

# TODO convert all this to take radians like other functions, then convert to deg

# params = [C, A, k]
# guess = [70, 80, 0.018]
guess = [70, 50, 0.03]
# result = minimize(dirdist_objective, guess, args=(distprefs[:, 0], distprefs[:, 1]), method='L-BFGS-B')
# result = minimize(dirdist_objective, guess, args=(bin_centers, bin_medians), method='Nelder-Mead')  # , method='L-BFGS-B')
# result = least_squares(dirdist_objective, guess, args=(distprefs[:, 0], distprefs[:, 1]), max_nfev=10000)
result = least_squares(dirdist_objective, guess, args=(bin_centers, bin_medians), max_nfev=100000)
fit = result['x']
C_f = fit[0]
A_f = fit[1]
k_f = fit[2]
ddxs = np.linspace(0, np.round(np.max(bin_edges)), int(np.round(np.max(bin_edges)) + 1))
ddys = C_f - (A_f * np.exp(-k_f * ddxs))

# params = [C, A, k]
# guess = [70, 80, 0.018]
# guess = [70, 50, 0.03]
# guess = [70, 50, 0.03, 8, 50, 10]
# guess = [70, 50, 0.03, 8, 20, 80]
# maybe find model with chirp or damping?
guess0 = [70.99, 57.46, 0.01294, 6.581, -3.221, 60]
guess1 = [7.099e+01, 5.746e+01, 1.294e-02, 6.581e+00, -3.221e+00, 7.514e+01]
guess2 = [7.101e+01, 6.917e+01, 1.599e-02, 3.525e+00, -9.966e+01, 5.435e+01]
result = least_squares(dirdist_objective_wcos, guess,
                            args=(bin_centers, bin_medians), 
                            xtol=1e-10,
                            ftol=1e-12,
                            #bounds=([50, 0, 0.01, 1, 0, 10], [90, 40, 0.05, 30, 1000, 300]),
                            max_nfev=1000000000)
fit = result['x']
C_f = fit[0]
A_f = fit[1]
k_f = fit[2]
B_f = fit[3]
d_f = fit[4]
e_f = fit[5]
ddxs = np.linspace(0, np.round(np.max(bin_edges)), int(np.round(np.max(bin_edges)) + 1))
ddys = C_f - (A_f * np.exp(-k_f * ddxs)) - B_f * np.cos((ddxs + d_f) / e_f)



# Jagruti paper
# Error bars represent the angular standard deviation.
# Blue line is an exponential fit to the data and
# red line is an exponential fit to the shuffled data.
#
# The exponential fit to the data measuring dependence of preferred direction difference
# between cells with the distance between cells used the following equation:
# 𝑦 = 𝐶 − 𝐴𝑒 −𝑘𝑥
# Where y is the fitted direction difference for x distance between cells, C is the saturation
# value, A is the start value and k is the decay space constant. The parameters were
# estimated using least squares curve fitting to individual data points.
# The shuffled data for measuring this dependence was generated by keeping the same
# cell positions but shuffling their preferred directions

f1 = plt.figure()
ax = f1.subplots(1, 1)
ax.set_ylabel('Preferred direction difference (' + deg_symbol + ')', fontsize=10)
ax.set_xlabel('Distance (µm)', fontsize=10)
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim((0, 1000))
ax.set_ylim((0, 180))

# Plot all pairs of direction difference and distance difference
ax.scatter(distprefs[:, 0], distprefs[:, 1], marker='.', s=0.5, color='k', edgecolor='None')

# Plot median values for 25um distance bins
# ax.scatter(bin_centers, bin_medians, marker='o', s=5, edgecolor='k', facecolor='w')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='r', markerfacecolor='w', markersize=3, capsize=0,
            fmt='o', elinewidth=1, ecolor='r')
# TODO investigate whether it is an issue that direction difference is only accurate to 1º

# ddxs = np.linspace(0, np.round(np.max(bin_edges)), int(np.round(np.max(bin_edges)) + 1))
# ddys = C_f - (A_f * np.exp(-k_f * ddxs)) - 8 * np.cos((ddxs + 20) / 80)
if result['success']:
    ax.plot(ddxs, ddys, 'r')

plt.show()

if saving:
    sn = save_pfix + '_RelationshipPlot_TprefDiff_by_Distance'
    f1.savefig(os.path.join(save_path, sn + save_ext),
               dpi=plt.rcParams['figure.dpi'], transparent=True)



# %% Plot responses across conditions for direction-tuned ROIs

for r in range(0, n_ROIs_tuned, 1):
    ridx = tunidx_dsi_argsrt[r]
    # ridx = tunidx_dsi_argsrt[-2]
    print('ROI {} with '.format(ridx) +
          'DSI={:.2f} and '.format(DSI[ridx]) +
          'Tpref={:.2f}'.format(Tprefs[ridx]))
    ipd = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=((8+2)*2*150*ipd, (4+1)*300*ipd))
    # fig.subplots(nrows=2, ncols=8)
    fig.clf()
    fig.suptitle('ROI {} Tpref={:.2f} DSI={:0.2f})'.format(ridx, Tprefs[ridx], DSI[ridx]), fontsize=8)
    axes = fig.subplots(nrows=2, ncols=n_conds)
    m = 'Fzsc'
    for c in range(n_conds):
        ax = axes[0, c]
        ax.set_title(str(data[c]['dir']) + deg_symbol, fontsize=6)
        if c == 0:
            # plt.xlabel('Frame (@'+str(md['framerate'])+'Hz)', fontsize=8)
            # ax.set_xlabel('Frame (@'+str(md['framerate'])+'Hz)', fontsize=8)
            ax.set_ylabel('Z-score', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xticks([x * md['framerate'] for x in range(5)])
            ax.set_xticklabels([])
            # ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            # ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(n_samp_isi, (n_samp_isi + n_samp_stim), color='0.9')
        # ax.set_ylim((-2,8))
        # ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
        #              np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
        ymin = np.min(data[m][:, ridx, :, :])
        ymax = np.max(data[m][:, ridx, :, :])
        ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
        for t in range(n_reps):
            # ax.plot(range(n_samp_trial), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
            ax.plot(range(n_samp_trial), data[m][c, ridx, t, :], color=str((0.4)+0.4*t/15))
        # ax.plot(range(n_samp_trial), np.mean(Fzsc_by_cond[ridx,c,:,:], axis=0), color='tab:green')
        ax.plot(range(n_samp_trial), np.mean(data[m][c, ridx, :, :], axis=0), color='tab:green')
    m = 'FdFF'
    for c in range(n_conds):
        ax = axes[1,c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * md['framerate'] for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(n_samp_isi, (n_samp_isi + n_samp_stim), color='0.9')
        ax.set_xlim((0,4*md['framerate']))
        # ax.set_ylim((-1,3))
        # ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.1,
        #              np.max(FdFF_by_cond[ridx,:,:,:]) + 0.1))
        ymin = np.min(data[m][:, ridx, :, :])
        ymax = np.max(data[m][:, ridx, :, :])
        ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
        for t in range(n_reps):
            # ax.plot(range(n_samp_trial), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
            ax.plot(range(n_samp_trial), data[m][c, ridx, t, :], color=str((0.4)+0.4*t/15))
        # ax.plot(range(n_samp_trial), np.mean(FdFF_by_cond[ridx,c,:,:], axis=0), color='tab:blue')
        ax.plot(range(n_samp_trial), np.mean(data[m][c, ridx, :, :], axis=0), color='tab:blue')
        # ## Plot p_value of the mean
        # t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
        # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
        # p_val = t_test[1]
        # plt.title('p value = ' + str(p_val.round(2)))
        
        # fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
        # fig.waitforbuttonpress()
        # print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
    plt.show()
    # if saving:
    #    fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_roi{}.svg'.format(r), format='svg', dpi=1200)
    plt.pause(0.05)

