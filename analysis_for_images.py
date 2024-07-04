#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
import pandas as pd
import re
import socket
from warnings import warn

import filters
from metadata import default_metadata
import parsers
import plots


# TODO: add consistent pyplot theme handling for plots https://github.com/raybuhr/pyplot-themes
#       including colorblind palette options https://personal.sron.nl/~pault/
# TODO: exclude suite2p badframes
# TODO: exclude based on eye tracking, at least when eyes are not open


# %% Settings

# FSI threshold 
# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci (https://doi.org/10.1038/nn.2363):
# """
# For |face-selectivity index| = 1/3, that is, if the response to faces was at least twice (or 
# at most half) that of nonface objects, a cell was classed as being face selective.
# [...] neurons (94%) were face selective (that is, face-selectivity index larger than 1/3 or 
# smaller than -1/3).
# """
threshold_fsi = 1 / 3

# dprime_F threshold 
# based on Shi et al Tsao bioRxiv (Fig 1g, https://doi.org/10.1101/2023.12.06.570341):
# "The dotted vertical line marks d’ = 0.2, which we used as our threshold for identifying face-selective units."
threshold_dprime = 0.2

threshold_cellprob = 0.0

plot_eyecal = False


plt.rcParams['figure.dpi'] = 300
dpi = plt.rcParams['figure.dpi']

# Template for ordering and labeling plots
template = np.array([b'blank', b'scram_s', b'scram_p',
                     b'face_mrm', b'face_rhe', b'face_hum', b'face_ctn',
                     b'obj', b'food',
                     b'body_mrm', b'animal'], dtype='|S8')
template_labels = {b'blank': 'Blank',
                   b'scram_s': 'Scramble (Spatial)',
                   b'scram_p': 'Scrambles (Phase)',
                   b'face_mrm': 'Faces (Marmo)',
                   b'face_rhe': 'R',  # 'Faces (Rhesus)',
                   b'face_hum': 'H',  # 'Faces (Human)',
                   b'face_ctn': 'Ctn',  # 'Faces (Cartoon)',
                   b'obj': 'Objects',
                   b'food': 'Foods',
                   b'body_mrm': 'Bodies (Marmo)',
                   b'animal': 'Animals'}

# Metrics for plots and calculations
metrics = ['FdFF', 'Fzsc']
metric_labels = {'FdFF': 'dF/F',
                 'Fzsc': 'Z-score'}

# Remove stale metadata
if 'md' in locals():
    md = dict()
    del md

# %% Specify data locations

# # Cadbury 20221016d
animal_str = 'Cadbury'
# date_str = '20221016d_olds2p'
date_str = '20221016d'
# --  PD GOOD
# ... |FSI| threshold: 0.25
# ... Tuned ROIs: 894. Total ROIs: 6020.   (note: using threshold_cellprob old = 0.0)
# ... Percentage of tuned ROIs: 14.85%
session_str = '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'
md = dict()
md['framerate'] = 6.364
md['fov'] = dict()
md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
md['fov']['w_px'] = 730
md['fov']['h_px'] = 730
# dirstr_suite2p = 'suite2p_old*'
dirstr_suite2p = 'suite2p_cellpose2_d14px_pt-3p5_ft1p5*'
dirstr_stimset = 'Song_etal_Wang_2022_NatCommun|480288_equalized_RGBA_FOBonly'.replace('|', os.path.sep)

# # Cadbury 20230510d
# animal_str = 'Cadbury'
# date_str = '20230510d'
# # -- UNCLEAR (fluid, z drift across time)
# session_str = '155713tUTC_SP_depth200um_fov2190x2000um_res3p00x3p02umpx_fr06p993Hz_pow059p9mW_stimImagesSong230509dSel'

# # Cadbury 20230809d
# animal_str = 'Cadbury'
# date_str = '20230809d'
# # -- OBJ ()
# session_str = '173936tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # Cadbury 20231001d
# animal_str = 'Cadbury'
# date_str = '20231001d'
# # -- PD ()
# session_str = '190608tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p3mW_stimImagesFOBmany'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmany\Images\20230728d'
# # -- PD ()
# session_str = '200422tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow060p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # Cadbury 20231003d
# animal_str = 'Cadbury'
# date_str = '20231003d'
# # -- PD (200um, )
# session_str = '142836tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- PD (150um, )
# session_str = '145031tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p0mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- PD (200um, )
# session_str = '153340tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- PD (250um, )
# session_str = '154955tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow089p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- MD (150um, )
# session_str = '162025tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- MD (200um, )
# session_str = '163738tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow079p9mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- MD (200um, )
# session_str = '165634tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow079p9mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d
# # -- OBJ (200um, )
# session_str = '173850tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # Cadbury 20231007d
# animal_str = 'Cadbury'
# date_str = '20231007d'
# # -- OBJ (200um, )
# session_str = '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmany\Images\20230728d'
# # -- OBJ (150um, )
# session_str = '162705tUTC_SP_depth150um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow049p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- OBJ (250um, )
# session_str = '164258tUTC_SP_depth250um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow069p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- OBJ (300um, )
# session_str = '170046tUTC_SP_depth300um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow099p7mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- PD (200um, )
# session_str = '174147tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow080p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- PD (200um, )
# session_str = '180407tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p359Hz_pow080p2mW_stimMultimodal'
# stimimage_path = r'multimodal'

# # Cadbury 20231018d
# animal_str = 'Cadbury'
# date_str = '20231018d'
# # -- OBJ (200um, )
# session_str = '185135tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- OBJ ( )
# session_str = '190745tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow075p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'
# # -- OBJ ( )
# session_str = '192426tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # Dali 20230511d
# animal_str = 'Dali'
# date_str = '20230511d'
# # -- PD (headpost not tight)
# session_str = '134800tUTC_SP_depth200um_fov2190x2000um_res3p02x3p02umpx_fr06p993Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'
# # -- PD ()
# session_str = '150200tUTC_SP_depth300um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow065p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'

# # Dali 20230515d
# # -- Dali ? () CHECK MEANIM
# animal_str = 'Dali'
# date_str = '20230515d'
# session_str = '135100tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'
# # -- Dali ? () CHECK MEANIM
# session_str = '144500tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'

# # Dali 20230522d
# animal_str = 'Dali'
# date_str = '20230522d'
# # -- Dali ? (ANISO) CHECK MEANIM
# session_str = '153415tUTC_SP_depth200um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow051p8mW_stimImagesFOBsel230517dAniso'
# ????? stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\MarmosetFOB2018\20230517d\Scaled512x512_MaskEroded3_SHINEdLum_RGBA_reorg_subset'
# # -- Dali ? () CHECK MEANIM
# session_str = '170053tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow051p8mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'

# ... Dali .... aniso until 20230804d


if 'save_path' not in locals():
    save_path = ''
if 'stimimage_path' not in locals():
    stimimage_path = ''

if session_str.find('_') != -1:
    session_abbrev_str = session_str[0:session_str.find('_')]
    title_str = animal_str + '_' + date_str + '_' + session_abbrev_str
else:
    session_abbrev_str = ''
    title_str = animal_str + '_' + date_str + '_' + session_str
save_pfix = animal_str + date_str + session_abbrev_str
save_ext = '.png'

filestr_metadata = '*_metadata.pickle'
filestr_image_data = '*_00001.tif'
filestr_session_log = '*.log'
pattern_session_log = r'.*^((?!disptimes).)*$'  # Exclude log files whose names contain 'disptimes'
filestr_stimulus_log = '*_stimlog.*'
filestr_eyetrack_data = '*_AIdata.p'
dirstr_eyecal = '*_EyeTrackingCalibration'
filestr_eyecal_log = '*_EyeTrackingCalibration.log'
filestr_eyecal_data = '*_EyeTrackingCalibration_AIdata.p'
if 'dirstr_suite2p' not in locals():
    dirstr_suite2p = 'suite2p*'
dirstr_suite2p_plane = 'plane0'


# Object defininitions


class StimulusImage(object):
    """Representation of stimulus images."""

    def __init__(self, condition, category, orientation, identity=None, filename=None, filepath=None):
        self.condition = condition
        self.category = category
        self.identity = identity
        self.pitch = orientation[0]
        self.yaw = orientation[1]
        self.roll = orientation[2]
        self.orientation = orientation
        self.filename = filename
        self.filepath = filepath

    def __repr__(self):
        return str((self.condition, self.category, self.identity, self.orientation, self.filename))

    def __eq__(self, other):
        return self.category == other.category and self.condition == other.condition

    def __lt__(self, other):
        return self.yaw < other.yaw


# %% Load data

system_name = socket.gethostname()
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/suite2p_results'
    stim_path = r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data'
    stim_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data'
    stim_path = r'C:\Users\DavidH\Sync\Freiwald\MarmoScope\Stimulus\Sets'
else:
    base_path = None
    stim_path = None

if save_path == '':
    saving = False
else:
    saving = True

stimimage_path = os.path.join(stim_path, dirstr_stimset)
if not os.path.isdir(stimimage_path):
    warn('Could not find stimulus image source path.')
    stimimage_path = None

date_path = os.path.join(base_path, animal_str, date_str)
session_path = os.path.join(base_path, animal_str, date_str, session_str)

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
        stimlog = pd.read_pickle(slf_path)
    elif len(hdf5s) > 0:
        if len(hdf5s) > 1:
            warn('Found multiple stimlog hdf5 files, using the first: {}'.format(hdf5s[0]))
        slf_path = hdf5s[0]
        stimlog = pd.read_hdf(slf_path)
    elif len(csvs) > 0:
        if len(hdf5s) > 1:
            warn('Found multiple stimlog csv files, using the first: {}'.format(csvs[0]))
        slf_path = csvs[0]
        stimlog = pd.read_csv(slf_path)
    else:
        stimlog = None
    del pkls, hdf5s, csvs
else:
    stimlog = None

dirlist_eyecal = [d for d in glob(os.path.join(date_path, dirstr_eyecal)) if os.path.isdir(d)]
if len(dirlist_eyecal) > 0:
    ecd_path = dirlist_eyecal[0]
    if len(dirlist_eyecal) > 1:
        warn('Found multiple eye tracking calibration directories, using the first one: {}'.format(ecd_path))
    filelist_eyecal_log = [f for f in glob(os.path.join(ecd_path, filestr_eyecal_log))
                           if os.path.isfile(f)]
    filelist_eyecal_data = [f for f in glob(os.path.join(ecd_path, filestr_eyecal_data))
                            if os.path.isfile(f)]
else:
    ecd_path = None
    filelist_eyecal_log = []
    filelist_eyecal_data = []

if len(filelist_eyecal_log) > 0:
    ec_lf_path = filelist_eyecal_log[0]
    if len(filelist_eyecal_log) > 1:
        warn('Found multiple eye tracking calibration log files, using the first one: {}'.format(ec_lf_path))
    eclf = open(ec_lf_path, 'r')
    eyecal_log = eclf.read()
    eclf.close()
else:
    ec_lf_path = None
    eclf = None
    eyecal_log = None
del eclf

if len(filelist_eyecal_data) > 0:
    ec_df_path = filelist_eyecal_data[0]
    if len(filelist_eyecal_data) > 1:
        warn('Found multiple eye tracking calibration data files, using the first one: {}'.format(ec_df_path))
    with open(ec_df_path, 'rb') as ec_df:
        eyecal_data = pickle.load(ec_df)
else:
    etf_path = None
    eyecal_data = None
del ec_df

filelist_eyetrack_data = [f for f in glob(os.path.join(session_path, filestr_eyetrack_data)) if os.path.isfile(f)]
if len(filelist_eyetrack_data) > 0:
    etf_path = filelist_eyetrack_data[0]
    if len(filelist_eyetrack_data) > 1:
        warn('Found multiple log files, using the first: {}'.format(etf_path))
    with open(etf_path, 'rb') as etf:
        eyetrk_data = pickle.load(etf)
else:
    etf_path = None
del etf

dirlist_suite2p = [d for d in glob(os.path.join(session_path, dirstr_suite2p)) if os.path.isdir(d)]
if len(dirlist_suite2p) > 0:
    if len(dirlist_suite2p) > 1:
        warn('Found multiple suite2p folders, using the first one: {}'.format(os.path.basename(dirlist_suite2p[0])))
    s2p_path = dirlist_suite2p[0]
    s2p_plane_path = os.path.join(s2p_path, dirstr_suite2p_plane)
    if not os.path.isdir(s2p_plane_path):
        raise RuntimeError('Could not load suite2p plane0 folder.')
else:
    raise RuntimeError('Could not find suite2p folder.')


# % Load suite2p outputs

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


# Estimate SNR

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
# # fig.suptitle('mean response by condition (each trial plotted)', fontsize=8)
# axes = fig.subplots(nrows=n_plot_ROIs, ncols=1)
# for r in range(n_plot_ROIs):
#     ridx = plot_ROIs[r]
#     # frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
#     # frame_end = frame_start + n_samp_inspect
#     Fr = Frois[ridx, frame_start:frame_end]

#     # Fr = Frois[ridx, frame_start:frame_end]
#     Fr_dFF = (Fr - np.mean(Fr)) / np.mean(Fr)
#     # F0_rnk = filters.rank_order_filter(Fr, p=filter_percentile, n=round(filter_window * fr))
#     # Fr_dFF_rnk = (Fr - F0_rnk) / F0_rnk
#     # F0_pct = filters.percentile_filter_1d(Fr, p=filter_percentile, n=round(filter_window * fr))
#     # Fr_dFF_pct = (Fr - F0_pct) / F0_pct
#     # F0_rnkbw = filters.butterworth_filter(F0_rnk, fs=fr, p=filter_percentile)
#     # Fr_dFF_rnkbw = (Fr - F0_rnkbw) / F0_rnkbw
#     # F0_pctbw = filters.butterworth_filter(Fr, fs=fr, p=filter_percentile)
#     # Fr_dFF_pctbw = (Fr - F0_pctbw) / F0_pctbw
#     # # F0_med = np.median(np.lib.stride_tricks.sliding_window_view(Fr, (round(filter_window * fr),)), axis=1)
#     # # Fr_dFF_med = (Fr - F0_med) / F0_med
#     # F0_ma = np.convolve(Fr, np.ones(round(filter_window * fr)), mode='same') / round(filter_window * fr)
#     # Fr_dFF_ma = Fr_dFF - np.convolve(Fr_dFF, np.ones(round(filter_window * fr)), mode='same') / round(
#     #     filter_window * fr)

#     # y: observed fluorescence
#     # c: calcium concentration
#     # s: neural activity / spike train
#     # b: baseline
#     # "To produce calcium trace c, spike train s is filtered with the inverse filter of g, an infinite impulse response
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
#     ax.set_ylabel('dF/F', fontsize=6)
#     ax.set_xlabel('Frames', fontsize=6)
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

#     ax.legend(fontsize=4, ncol=len(ax.get_lines()), frameon=False, loc=(.02, .85))
# plt.show()

# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
# from oasis.plotting import simpleaxis
# from oasis.oasis_methods import oasisAR1, oasisAR2

# c, s, b, g, lam = deconvolve(Fr)  # , penalty=1)


# Extract eye tracking calibration information from log file

if eyecal_log is not None:
    ecdata = parsers.parse_log_eyecal(eyecal_log, eyecal_data)
else:
    ecdata = None


# Plot eye-tracking calibration results

if plot_eyecal and ecdata is not None:
    # f = plt.figure()
    # ecx, ecy = ecdata['zero']['AIdata']
    # plt.scatter(ecx, ecy, s=1, c='m')
    # ecx, ecy = ecdata['circ']['data'][0]['AIdata']
    # plt.scatter(ecx, ecy, s=1, c='k')
    # plt.show()

    f = plt.figure()
    for trl in range(ecdata['circ']['n_trials']):
        ecx, ecy = np.transpose(ecdata['circ']['data'][trl]['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            circ = ecdata['circ']['data'][trl]['AIdata']
        else:
            circ = np.concatenate((circ, ecdata['circ']['data'][trl]['AIdata']))
    plt.show()
    f = plt.figure()
    plt.scatter(circ.T[0], circ.T[1], s=1)
    plt.scatter(np.median(circ.T[0]), np.median(circ.T[1]), s=5, c='m')
    ax = plt.gca()
    from matplotlib.patches import Ellipse
    c1 = Ellipse((np.median(circ.T[0]), np.median(circ.T[1])), width=np.std(circ.T[0]), height=np.std(circ.T[1]), lw=2, edgecolor='m', fc='None')
    ax.add_patch(c1)
    plt.show()

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
    # c1 = Ellipse((np.median(etx), np.median(ety)), width=np.std(etx), height=np.std(ety), lw=2, edgecolor='m', fc='None')
    # ax.add_patch(c1)
    # plt.show()


    f = plt.figure()
    for trl in range(ecdata['grdf']['n_trials']):
        ecx, ecy = np.transpose(ecdata['grdf']['data'][trl]['face']['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            grdf = ecdata['grdf']['data'][trl]['face']['AIdata']
        else:
            grdf = np.concatenate((grdf, ecdata['grdf']['data'][trl]['face']['AIdata']))
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
    plt.show()


    f = plt.figure()
    for trl in range(ecdata['crse']['n_trials']):
        ecx, ecy = np.transpose(ecdata['crse']['data'][trl]['AIdata'])
        plt.scatter(ecx, ecy, s=1)
        if trl == 0:
            crse = ecdata['crse']['data'][trl]['AIdata']
            crsecv = ecdata['crse']['data'][trl]['cvals']
        else:
            crse = np.concatenate((crse, ecdata['crse']['data'][trl]['AIdata']))
            crsecv = np.vstack((crsecv, ecdata['crse']['data'][trl]['cvals']))
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
    plt.show()


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


# Load stimulus information

# *** TODO load from a pandas dataframe instead of a text log

if stimlog is None:
    if session_log is not None:
        stimlog = parsers.parse_log_stim_image(session_log)
    else:
        raise RuntimeError('Could not load stimulus record from session log file.')

# Identify stimulus image set from file paths
if np.unique([os.path.dirname(p) for p in stimlog['image_path'].values]).size == 1:
    image_dirpath = os.path.dirname(stimlog.iloc[0]['image_path'])
    if 'FOBmin_MarmOnly' in image_dirpath or 'MinFOB_MarmOnly' in image_dirpath:
        image_set = 'FOBmin'
    elif 'FOBmin' in image_dirpath or 'MinFOB' in image_dirpath:
        image_set = 'FOBmin'
    elif 'FOBmany' in image_dirpath:
        image_set = 'FOBmany'
    elif 'MarmosetFOB2018' in image_dirpath:
        image_set = 'FOBmin'
    elif '480288_equalized_RGBA_FOBonly' in image_dirpath:
        image_set = 'Song_etal_Wang_2022_FOBonly'
    elif '480288_equalized_RGBA_selected20230509d' in image_dirpath:
        image_set = 'Song_etal_Wang_2022_FOBonly'
    else:
        warn('Image set not recognized from image paths. Set to last directory in path.')
        image_set = os.path.split(os.path.dirname(stimlog.iloc[0]['image_path']))[-1]
else:
    # *** TODO: default to images without category separation in this case
    warn('Images are not all from the same set.')
    image_set = None


# Determine basic stimulus presentation information
dur_stim = np.round(np.mean(stimlog['dur_stim'].values), 2)
dur_isi = np.round(np.min(stimlog['dur_isi_pre'].values), 2)
dur_trial = dur_isi + dur_stim + dur_isi
n_samp_stim = int(np.floor(np.mean(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i'])))
# n_samp_stim = int(np.ceil(dur_stim * md['framerate']))
n_samp_isi = int(np.min(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']))
# n_samp_isi = int(np.round(dur_isi * md['framerate']))
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

# Calculate the timing mismatch (contraction) introduced by rounding stim and/or isi frame samples down
acqfr_dilation_factor = (dur_trial * md['framerate']) / (n_samp_trial - 1)

n_metrics = len(metrics)
n_conds = len(np.unique(stimlog['cond'].values))
n_trials = len(stimlog)
n_reps = int(len(stimlog) / n_conds)

if len(np.unique(stimlog['acqfr_stim_i'])) != len(stimlog['acqfr_stim_i']):
    raise RuntimeError('Imaging was interrupted or stopped before stimulus. ' +
                       'Handling this is not yet implemented.')

# % Organize and average fluorescence traces

data = np.zeros(n_conds, dtype=[('cond', 'S8'),
                                ('stimulus', object),
                                ('cat', 'S8'),
                                ('id', 'S8'),
                                ('pitch', 'i2'),
                                ('yaw', 'i2'),
                                ('roll', 'i2'),
                                ('imagename', np.unicode_, 256),
                                ('FdFF', 'f4', (n_ROIs,
                                                n_reps,
                                                n_samp_trial)),
                                ('Fzsc', 'f4', (n_ROIs,
                                                n_reps,
                                                n_samp_trial)),
                                ('FdFF_meant', 'f4', (n_ROIs,
                                                      n_samp_trial)),
                                ('Fzsc_meant', 'f4', (n_ROIs,
                                                      n_samp_trial))])
data[:]['FdFF'] = np.nan
data[:]['Fzsc'] = np.nan
data[:]['FdFF_meant'] = np.nan
data[:]['Fzsc_meant'] = np.nan

# Currently supported image sets:
# 'FOBmin_MarmOnly', 'FOBmin', 'FOBmany', 'Song_etal_Wang_2022_FOBonly'
for c in range(n_conds):
    tmp_cond = None
    tmp_cat = None
    tmp_id = None
    tmp_pitch = -32768
    tmp_yaw = -32768
    tmp_roll = -32768
    if np.unique(stimlog[stimlog['cond'] == c]['image'].values).size == 1:
        tmp_imagename = np.unique(stimlog[stimlog['cond'] == c]['image'].values)[0]
        imn = os.path.splitext(tmp_imagename)[0]
    else:
        warn('Not all images were the same for condition {}.'.format(c))
        tmp_imagename = ''
        imn = ''
    tmp_ip = os.path.join(stimimage_path, tmp_imagename)
    tmp_imagepath = tmp_ip if os.path.isfile(tmp_ip) else None
    if image_set == 'FOBmin' or image_set == 'FOBmany':
        pattern_imn = r'^(Freiwald(FOB)?([0-9]*)?)?_?([^_]+)_([^_]+)_?([^_]+)?_([0-9]+)_?[^_]*_?(inverted)?$'
        if re.match(pattern_imn, imn) is not None:
            sp = re.match(pattern_imn, imn).group(4)
            ct = re.match(pattern_imn, imn).group(5)
            di = re.match(pattern_imn, imn).group(6)
            nm = re.match(pattern_imn, imn).group(7)
            if nm.isnumeric():
                nm = float(nm)
                if nm.is_integer():
                    nm = int(nm)
                else:
                    warn('View index in filename incorrect ({}). '.format(tmp_imagename) +
                         'Expected integer, not float.')
            iv = re.match(pattern_imn, imn).group(8) is not None
            match sp:
                case 'Human':
                    if ct == 'Head':
                        tmp_cond = bytes('fh{:02}'.format(nm), 'ascii')
                        tmp_cat = b'face_hum'
                        tmp_id = bytes('Hum{:02}'.format(nm), 'ascii')
                        tmp_pitch = 0
                        tmp_roll = 0
                        tmp_yaw = 0
                case 'MacaqueRhesus':
                    if ct == 'Head':
                        tmp_cond = bytes('fr{:02}'.format(nm), 'ascii')
                        tmp_cat = b'face_rhe'
                        tmp_id = bytes('Rhe{:02}'.format(nm), 'ascii')
                        tmp_pitch = 0
                        tmp_roll = 0
                        tmp_yaw = 0
                case 'Marm':
                    if ct == 'Head':
                        if iv is True:
                            nm = 9
                        tmp_cond = bytes('fm{}{:02}'.format(di[0:3], nm), 'ascii')
                        tmp_cat = b'face_mrm'
                        tmp_id = bytes(di[0:8], 'ascii')
                        match nm:
                            case 1:
                                tmp_pitch = 0
                                tmp_yaw = 0
                                tmp_roll = 0
                            case 2:
                                tmp_pitch = 0
                                tmp_yaw = 180
                                tmp_roll = 0
                            case 3:
                                tmp_pitch = 0
                                tmp_yaw = 0
                                tmp_roll = -45
                            case 4:
                                tmp_pitch = 0
                                tmp_yaw = 0
                                tmp_roll = 45
                            case 5:
                                tmp_pitch = 0
                                tmp_yaw = -90
                                tmp_roll = 0
                            case 6:
                                tmp_pitch = 0
                                tmp_yaw = -45
                                tmp_roll = 0
                            case 7:
                                tmp_pitch = 0
                                tmp_yaw = 45
                                tmp_roll = 0
                            case 8:
                                tmp_pitch = 0
                                tmp_yaw = 90
                                tmp_roll = 0
                            case 9:
                                tmp_pitch = 0
                                tmp_yaw = 0
                                tmp_roll = 180
                            case _:
                                warn('Could not recognize pitch, yaw, or roll of head image from filename.')
                                tmp_pitch = None
                                tmp_yaw = None
                                tmp_roll = None
                    if ct == 'Body':
                        tmp_cond = bytes('bm{}{:02}'.format(di[0:3], nm), 'ascii')
                        tmp_cat = b'body_mrm'
                        tmp_id = bytes(di[0:8], 'ascii')
                case 'Objects':
                    pattern_ct = r'^([^0-9]+)([0-9])$'
                    if re.match(pattern_ct, ct) is not None:
                        ct_p1 = re.match(pattern_ct, ct).group(1)
                        ct_p2 = re.match(pattern_ct, ct).group(2)
                        ct = ct_p1
                        if ct_p2.isnumeric():
                            ct_p2 = float(ct_p2)
                            if ct_p2.is_integer():
                                ct_p2 = int(ct_p2)
                            else:
                                warn('Object identity index in filename incorrect ({}). '.format(tmp_imagename) +
                                     'Expected integer, not float.')
                    else:
                        warn('Could not recognize object details from filename.')
                        ct_p2 = 0
                    if 'Manmade' in ct:
                        tmp_cond = bytes('om{:01}{:03}'.format(ct_p2, nm), 'ascii')
                        tmp_cat = b'obj'
                    elif 'FruitVeg' in ct:
                        tmp_cond = bytes('vf{:01}{:03}'.format(ct_p2, nm), 'ascii')
                        tmp_cat = b'food'
                    elif 'MultipartGeon' in ct:
                        tmp_cond = bytes('og{:01}{:03}'.format(ct_p2, nm), 'ascii')
                        tmp_id = bytes('Geon{:01}'.format(ct_p2), 'ascii')
                        tmp_cat = b'obj'
                    elif 'Pairwise' in ct:
                        tmp_cond = bytes('op{:01}{:03}'.format(ct_p2, nm), 'ascii')
                        tmp_cat = b'obj'
                    elif 'String' in ct:
                        tmp_cond = bytes('os{:01}{:03}'.format(ct_p2, nm), 'ascii')
                        tmp_cat = b'obj'
                    else:
                        warn('Could not recognize category of object image from filename.')
                case _:
                    warn('Could not recognize type of image from filename.')
        elif imn == 'blank':
            tmp_cond = bytes('blank', 'ascii')
            tmp_cat = b'blank'
        elif 'Cartoon' in imn:
            pattern_ctn = r'^[^_]*Cartoon_([0-9]+)_?[^_]*_?(inverted)?$'
            if re.match(pattern_ctn, imn) is not None:
                nm = re.match(pattern_ctn, imn).group(1)
                tmp_cond = bytes('fcm{:04}'.format(nm), 'ascii')
                tmp_cat = b'face_ctn'
                tmp_id = bytes(nm, 'ascii')
                tmp_pitch = 0
                tmp_yaw = 0
                tmp_roll = 0
        else:
            warn('Could not recognize category or condition of image from filename.')
            tmp_cond = None
            tmp_cat = None
    elif image_set == 'Song_etal_Wang_2022_FOBonly':
        pattern_zerocheck = r'^([aobmufps]{1})([0-9]{1})$'
        if re.match(pattern_zerocheck, imn) is not None:
            tg = re.match(pattern_zerocheck, imn).group(1)
            ng = re.match(pattern_zerocheck, imn).group(2)
            lfc = '{}{}'.format(tg, ng.zfill(2))
        else:
            lfc = imn
        tmp_cond = bytes(lfc, 'ascii')
        match imn[0]:
            case 'a':
                tmp_cat = b'animal'
            case 'o':
                tmp_cat = b'obj'
            case 'b':
                if imn == 'blank':
                    tmp_cat = b'blank'
                else:
                    tmp_cat = b'body_mrm'
            case 'm':
                tmp_cat = b'face_mrm'
                tmp_pitch = 0
                tmp_yaw = 0
                tmp_roll = 0
            case 'u':
                tmp_cat = b'obj'
            case 'f':
                tmp_cat = b'food'
            case 'p':
                tmp_cat = b'scram_p'
            case 's':
                tmp_cat = b'scram_s'
            case _:
                tmp_cat = None
    data[c]['cond'] = tmp_cond
    data[c]['cat'] = tmp_cat
    data[c]['id'] = tmp_id
    data[c]['pitch'] = tmp_pitch
    data[c]['yaw'] = tmp_yaw
    data[c]['roll'] = tmp_roll
    data[c]['imagename'] = tmp_imagename
    data[c]['stimulus'] = StimulusImage(tmp_cond, tmp_cat, (tmp_pitch, tmp_yaw, tmp_roll),
                                        identity=tmp_id, filename=tmp_imagename, filepath=tmp_imagepath)
    for t in range(n_reps):
        fr_start = stimlog[stimlog['cond'] == c].iloc[t]['acqfr_stim_i'] - n_samp_isi
        fr_end = stimlog[stimlog['cond'] == c].iloc[t]['acqfr_stim_i'] + n_samp_stim + n_samp_isi
        if fr_start < 0 and t == 0:
            # TODO exclude trials at start of imaging if ISI is too short
            warn('Period before first trial was shorter than inter-stimulus interval. ' +
                 'Copied first present value to prevent error. ' +
                 'But this trial could instead be excluded.')
            n_missing = abs(fr_start)
            data[c]['FdFF'][:, t, 0:n_missing] = np.array([FdFF_raw[:, 0],] * n_missing).transpose()
            data[c]['Fzsc'][:, t, 0:n_missing] = np.array([Fzsc_raw[:, 0],] * n_missing).transpose()
            fr_start = 0
            data[c]['FdFF'][:, t, n_missing:n_samp_trial] = FdFF_raw[:, fr_start:fr_end]
            data[c]['Fzsc'][:, t, n_missing:n_samp_trial] = Fzsc_raw[:, fr_start:fr_end]
            continue
        if fr_end > n_frames:
            # TODO: support throwing away trials after imaging stops
            raise RuntimeError('Imaging was stopped before stimulus. Handling this is not yet implemented.')
        data[c]['FdFF'][:, t, :] = FdFF_raw[:, fr_start:fr_end]
        data[c]['Fzsc'][:, t, :] = Fzsc_raw[:, fr_start:fr_end]
    if not np.any(np.isnan(data[c]['FdFF'])):
        data[c]['FdFF_meant'] = np.mean(data[c]['FdFF'], axis=1)
    else:
        warn('Cond {} had FdFF values that are NaNs'.format(c))
        data[c]['FdFF_meant'] = np.nanmean(data[c]['FdFF'], axis=1)
    if not np.any(np.isnan(data[c]['Fzsc'])):
        data[c]['Fzsc_meant'] = np.mean(data[c]['Fzsc'], axis=1)
    else:
        warn('Cond {} had Fzsc values that are NaNs'.format(c))
        data[c]['Fzsc_meant'] = np.nanmean(data[c]['Fzsc'], axis=1)

categories = np.unique(data[:]['cat'])
n_cats = len(categories)
conditions = np.unique(data[:]['cond'])
# condition_inds = {c: i for i, c in enumerate(data['cond'])}

if n_conds != conditions.shape[0]:
    u, c = np.unique(data[:]['cond'], return_counts=True)
    mult = u[c > 1]
    warn('Some different image files were combined into the same condition. ' +
         '{}'.format(data[data['cond'] == mult]['imagename']))
    del u, c, mult


# Plot population mean response (similar to PSTH)
fr = md['framerate']
fig_psth = plt.figure()
fig_psth.suptitle('mean population response (across all ROIs) by category')
axes = fig_psth.subplots(nrows=n_metrics, ncols=1)
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * fr)
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
for mi, m in enumerate(metrics):
    ymin = np.min(np.array([np.mean(data[data['cat'] == categories[c]][m], axis=(0, 1, 2)) for c in range(n_cats)]))
    ymax = np.max(np.array([np.mean(data[data['cat'] == categories[c]][m], axis=(0, 1, 2)) for c in range(n_cats)]))
    ax = axes[mi]
    ax.set_ylabel(metric_labels[m])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xticks = [x * fr for x in range(np.ceil(dur_trial).astype('int') + 1)]
    xticklabels = ['' if not np.isclose(xt, dur_isi * fr) and not np.isclose(xt, (dur_isi + dur_stim) * fr)
                   else '{}'.format(np.round(xt / fr).astype('int')) for xt in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.axvspan(dur_isi * fr, (dur_isi + dur_stim) * fr, color='0.9', zorder=0)
    ax.set_xlim((0, np.ceil(dur_trial) * fr))
    ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
    ax.plot(xs,
            np.mean(data[m], axis=(0, 1, 2)), 
            label='All', 
            color='0', linestyle='dotted', linewidth=1, zorder=4)
    for cat in range(n_cats):
        n_cnd_in_cat = data[data['cat'] == categories[cat]]['cond'].shape[0]
        # Fmean = np.mean(data[data['cat'] == categories[cat]][m], axis=(0, 1, 2))
        # Fsem = np.std(data[data['cat'] == categories[cat]][m], axis=(0, 1, 2)) / np.sqrt(n_ROIs)
        ax.plot(xs,
                np.mean(data[data['cat'] == categories[cat]][m], axis=(0, 1, 2)),
                'o-', markersize=2,
                label=template_labels[categories[cat]], 
                color=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), zorder=3)
        # ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, 
        #                 color=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), alpha=0.2, zorder=2)
    del cat
    ax.legend(fontsize=4, frameon=False, loc=(.02, .7))
plt.show()
del mi, m, xs, xticks, xticklabels


# TODO: check for NaN values rather than using nanmean?
# if np.any(np.isnan(volume)):
#     raise Exception('NaNs found in preprocessed volume.')

idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)

# Define booleans for face and non-face conditions
#   *** TODO: Also consider yaw and roll... and perhaps excluding cartoons?
bool_F = np.logical_or.reduce([data['cat'] == fc for fc in categories 
                               if 'face' in fc.decode() 
                               and 'blank' not in fc.decode() and 'scram' not in fc.decode()])
bool_NF = np.logical_or.reduce([data['cat'] == fc for fc in categories 
                                if 'face' not in fc.decode() 
                                and 'blank' not in fc.decode() and 'scram' not in fc.decode()])
bool_NFobj = np.logical_or.reduce([data['cat'] == fc for fc in categories 
                                   if 'face' not in fc.decode() 
                                   and 'blank' not in fc.decode() and 'scram' not in fc.decode()
                                   and 'body' not in fc.decode()])
bool_B = np.logical_or.reduce([data['cat'] == fc for fc in categories 
                               if 'body' in fc.decode() 
                               and 'blank' not in fc.decode() and 'scram' not in fc.decode()])


# Calculate R across-stimulus, trial-averaged mean responses
muR_F = {}
muR_NF = {}
muR_NFobj = {}
muR_B = {}
sigma_F = {}
sigma_NF = {}
sigma_NFobj = {}
sigma_B = {}
for m in metrics:
    muR_F[m] = np.full(n_ROIs, np.nan)
    muR_NF[m] = np.full(n_ROIs, np.nan)
    muR_NFobj[m] = np.full(n_ROIs, np.nan)
    muR_B[m] = np.full(n_ROIs, np.nan)
    sigma_F[m] = np.full(n_ROIs, np.nan)
    sigma_NF[m] = np.full(n_ROIs, np.nan)
    sigma_NFobj[m] = np.full(n_ROIs, np.nan)
    sigma_B[m] = np.full(n_ROIs, np.nan)

    # Calculate across-stimulus mean responses.
    #   Note that the ordering of mean calculations matters here because the mean of a set is only the
    #   same as the mean of the mean of subsets if the subsets share the same sample size.
    muR_F[m] = np.mean(np.mean(data[bool_F][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_NF[m] = np.mean(np.mean(data[bool_NF][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_NFobj[m] = np.mean(np.mean(data[bool_NFobj][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_B[m] = np.mean(np.mean(data[bool_B][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)

    # Calculate across-stimulus standard deviation.
    #   Take the mean across trials and frames (okay because same n_samp_stim in each), 
    #   then take std across stimulus conditions.
    sigma_F[m] = np.std(np.mean(data[bool_F][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_NF[m] = np.std(np.mean(data[bool_NF][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_NFobj[m] = np.std(np.mean(data[bool_NFobj][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_B[m] = np.std(np.mean(data[bool_B][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
del m


# Calculate face discriminability index d′
# based on Vinken et al Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
# """
# Face selectivity was quantified by computing the d′ sensitivity index comparing trial-averaged responses to faces
# and non-faces:
# d′ = (μ_F - μ_NF) / sqrt((σ_F^2 + σ_NF^2) / 2)
# where μ_F and μ_NF are the across-stimulus averages of the trial-averaged responses to faces and non-faces, and
# σ_F and σ_NF are the across-stimulus SDs. This face d′ value quantifies how much higher (positive d′) or lower
# (negative d′) the response to a face is expected to be compared to a non-face, in SD units.
# """

dprime = {}
sort_idx_dprime = {}
for m in metrics:
    dprime[m] = []
    sort_idx_dprime[m] = []
    dprime[m] = (muR_F[m] - muR_NF[m]) / np.sqrt((sigma_F[m]**2 + sigma_NF[m]**2) / 2)
    sort_idx_dprime[m] = np.argsort(dprime[m])[::-1]
del m


# Calculate face selectivity index (FSI)
# based on Freiwald and Tsao 2010 Science https://doi.org/10.1126/science.1194908
# """
# ... the Face Selectivity Index (FSI) was defined by FSI = (Rfaces – Rnonfaceobjects) / (Rface + Rnonfaceobjects), 
# where Rfaces was the mean response above baseline to faces and Rnonfaceobjects the mean response above baseline 
# to non-face objects. An FSI of 0 indicates equal responses to face and non-face objects. An FSI of 0.33 indicated 
# twice as strong response to faces as to non-face objects. 
# For cases where (Rface > 0) and (Rnonfaceobjects < 0), FSI was set to 1; for cases where (Rface < 0) and 
# (Rnonfaceobjects > 0), FSI was set to -1.
# """

FSI = {}
for m in metrics:
    FSI[m] = np.full(n_ROIs, np.nan)

    bool_same = (np.sign(muR_F[m]) == np.sign(muR_NFobj[m]))
    bool_FposNFneg = (np.sign(muR_F[m]) > np.sign(muR_NFobj[m]))
    bool_FnegNFpos = (np.sign(muR_F[m]) < np.sign(muR_NFobj[m]))
    
    FSI[m][bool_same] = (muR_F[m][bool_same] - muR_NFobj[m][bool_same]) / \
        (muR_F[m][bool_same] + muR_NFobj[m][bool_same])
    FSI[m][bool_FposNFneg] = 1.0
    FSI[m][bool_FnegNFpos] = -1.0
del m, bool_same, bool_FposNFneg, bool_FnegNFpos


# Calculate stimulus response vectors for each ROI
# conds_in_cat = {cat: data[data['cat'] == cat]['cond'] for cat in template.tolist()}
# cond_inds_in_cat = {cat: [condition_inds[ci] for ci in conds_in_cat[cat] if ci.tolist()] 
#                     for cat in template.tolist()
#                     if [condition_inds[ci] for ci in conds_in_cat[cat] if ci.tolist()]}
cond_resp_vect = {}
cat_resp_vect = {}
for m in metrics:
    cond_resp_vect[m] = np.mean(data[m][:, :, :, idx_stim], axis=(2,3)).transpose()
    cat_resp_vect[m] = np.array([np.mean(data[data['cat'] == c][m][:, :, :, idx_stim], axis=(0, 2, 3)) 
                                 for c in categories]).transpose()
del m


# Store response values and tuning metrics for each ROI
ROI_stats = {}
for m in metrics:
    ROI_stats[m] = {}
    for r in range(n_ROIs):
        ROI_stats[m][r] = {}
    
        ROI_stats[m][r]['cond_resp_vect'] = cond_resp_vect[m][r]
        ROI_stats[m][r]['peak_cond_idx'] = ROI_stats[m][r]['cond_resp_vect'].argmax()
        ROI_stats[m][r]['peak_cond'] = conditions[ROI_stats[m][r]['peak_cond_idx']]
        ROI_stats[m][r]['peak_cond_val'] = ROI_stats[m][r]['cond_resp_vect'].max()
        ROI_stats[m][r]['cat_of_peak_cond'] = data[data['cond'] == ROI_stats[m][r]['peak_cond']]['cat']
    
        ROI_stats[m][r]['cat_resp_vect'] = cat_resp_vect[m][r]
        ROI_stats[m][r]['peak_cat_idx'] = ROI_stats[m][r]['cat_resp_vect'].argmax()
        ROI_stats[m][r]['peak_cat'] = categories[ROI_stats[m][r]['peak_cat_idx']]
        ROI_stats[m][r]['peak_cat_val'] = ROI_stats[m][r]['cat_resp_vect'].max()
        
        ROI_stats[m][r]['dprime_f'] = dprime[m][r]
        ROI_stats[m][r]['fsi'] = FSI[m][r]
    del r
del m

ROI_stats_df = {}
for m in metrics:
    ROI_stats_df[m] = {}
    
    ROI_stats_df[m] = pd.DataFrame({'roi': range(n_ROIs),
                                    'peak_cond': None,
                                    'peak_cond_idx': None,
                                    'peak_cond_val': None,
                                    'cat_of_peak_cond': None,
                                    'peak_cat': None,
                                    'peak_cat_idx': None,
                                    'peak_cat_val': None,
                                    'dprime_f': None,
                                    'fsi': None,
                                    'cond_resp_vect': None,
                                    'cat_resp_vect': None})
    ROI_stats_df[m].set_index(['roi'])

    for r in range(n_ROIs):
        ROI_stats_df[m].at[r, 'cond_resp_vect'] = np.mean(data[m][:, r, :, :][:, :, idx_stim], axis=(1, 2))
        ROI_stats_df[m].at[r, 'peak_cond_idx'] = ROI_stats_df[m].loc[r]['cond_resp_vect'].argmax()
        ROI_stats_df[m].at[r, 'peak_cond'] = conditions[ROI_stats_df[m].at[r, 'peak_cond_idx']]
        ROI_stats_df[m].at[r, 'peak_cond_val'] = ROI_stats_df[m].loc[r]['cond_resp_vect'].max()
        ROI_stats_df[m].at[r, 'cat_of_peak_cond'] = data[data['cond'] == conditions[ROI_stats_df[m].at[r, 'peak_cond_idx']]]['cat']

        ROI_stats_df[m].at[r, 'cat_resp_vect'] = cat_resp_vect[m][r]
        ROI_stats_df[m].at[r, 'peak_cat_idx'] = ROI_stats_df[m].loc[r]['cat_resp_vect'].argmax()
        ROI_stats_df[m].at[r, 'peak_cat'] = categories[ROI_stats_df[m].at[r, 'peak_cat_idx']]
        ROI_stats_df[m].at[r, 'peak_cat_val'] = ROI_stats_df[m].loc[r]['cat_resp_vect'].max()
        
        ROI_stats_df[m].at[r, 'dprime_f'] = dprime[m][r]
        ROI_stats_df[m].at[r, 'fsi'] = FSI[m][r]
    del r
del m

del cond_resp_vect, cat_resp_vect

if n_metrics > 1:
    for mi, m in enumerate(metrics):
        if mi + 1 < n_metrics:
            bool_cnd = (ROI_stats_df[m]['peak_cond'].values != ROI_stats_df[metrics[mi+1]]['peak_cond'].values)
            ROIs_diff_peak_cnd = np.where(bool_cnd)[0]
            if ROIs_diff_peak_cnd.size > 0:
                warn('peak_cond mismatch ({} vs {}) for ROIs: {}'.format(m, metrics[mi+1], ROIs_diff_peak_cnd))
            bool_cat = (ROI_stats_df[m]['peak_cat'].values != ROI_stats_df[metrics[mi+1]]['peak_cat'].values)
            ROIs_diff_peak_cat = np.where(bool_cat)[0]
            if ROIs_diff_peak_cat.size > 0:
                warn('peak_cat mismatch ({} vs {}) for ROIs: {}'.format(m, metrics[mi+1], ROIs_diff_peak_cat))
    del mi, m, bool_cnd, ROIs_diff_peak_cnd, bool_cat, ROIs_diff_peak_cat


# Output information about ROI tuning based on defined thresholds
print('|FSI| threshold: {}'.format(threshold_fsi))
tunidx_fsi = FSI['Fzsc']
tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).squeeze()
n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).shape[0]
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(round(((100 * n_ROIs_tuned) / n_ROIs), 2)))
del ROIs_tuned_idx, n_ROIs_tuned

# TODO regorganize

# Plot ROI tuning histograms
sp = os.path.join(save_path, save_pfix + '_Histogram_FSIs_fromFdFF' + save_ext) if saving else ''
plots.plot_hist_fsi(FSI['FdFF'], threshold=threshold_fsi, title='FSIs calculated from FdFF values', save_path=sp)
sp = os.path.join(save_path, save_pfix + '_Histogram_FSIs_fromZscr' + save_ext) if saving else ''
plots.plot_hist_fsi(FSI['Fzsc'], threshold=threshold_fsi, title='FSIs calculated from z-scored values', save_path=sp)

sp = os.path.join(save_path, save_pfix + '_Histogram_dprimes_fromFzsc' + save_ext) if saving else ''
plots.plot_hist_dprime(dprime['Fzsc'], threshold=threshold_dprime, title='dprimes calculated from Fzsc values', save_path=sp)
sp = os.path.join(save_path, save_pfix + '_Histogram_dprimes_fromFdFF' + save_ext) if saving else ''
plots.plot_hist_dprime(dprime['FdFF'], threshold=threshold_dprime, title='dprimes calculated from FdFF values', save_path=sp)


# Summarize responsiveness of each ROI

# Ensure that category order is FOB for consistent RGB color code
fob = np.array([b'face_mrm', b'obj', b'body_mrm'], dtype='|S8')
if all(x in categories for x in [b'face_mrm', b'obj', b'body_mrm']):
    if np.setdiff1d(categories, fob).size == 0:
        categories = fob
    else:
        categories = np.concatenate((fob, np.setdiff1d(categories, fob)))

# # ONLY CONSIDER FOB MARM IMAGES
cat_subset = fob
cond_subset = np.where((data['cat'] == b'face_mrm') | (data['cat'] == b'obj') | (data['cat'] == b'body_mrm'))[0]
cond_names = np.hstack([np.sort(data[data['cat'] == cat]['cond']) for cat in cat_subset])
cond_idx = np.array([np.where(data['cond'] == cond)[0][0] for cond in cond_names])

# TODO improve variable naming here for clarity

# Determine for each ROI which condition (image) elicited the largest response
m = 'Fzsc'
above_threshold = np.where(ROI_stats_df[m]['peak_cond_val'] > 0.5)[0]
# TODO THIS SHOULD NOT BE HARD CODED
at_sortidx = (-np.mean(data[cond_idx[0:19]]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1))[above_threshold]).argsort()


# Establish ordering for heatmaps

lambda_sort = lambda x: (np.where(template == x[1].category)[0][0]
                         if np.where(template == x[1].category)[0].size > 0
                         else np.iinfo(np.where(template == x[1].category)[0].dtype).max,
                         np.abs(x[1].roll),
                         x[1].roll,
                         x[1].yaw,
                         x[1].condition.decode().lower())
stimarr = data[:]['stimulus']
stimcond = [i for i, x in sorted(enumerate(stimarr), key=lambda_sort)]
stimsort = stimarr[stimcond]

tickinfo = {t.decode(): {} for t in template}
for t in template:
    ts = t.decode()
    wheret = np.where(data[stimcond]['cat'] == t)[0]
    if wheret.size > 0:
        tickinfo[ts]['start'] = wheret[0]
        tickinfo[ts]['end'] = wheret[-1]
        tickinfo[ts]['labelpos'] = (tickinfo[ts]['start'] + tickinfo[ts]['end']) / 2
        tickinfo[ts]['label'] = template_labels[t]
    else:
        tickinfo.pop(ts)


# # Plot heatmap of mean responses to all presented conditions (images) for ROIs
# # with at least one stimulus period z-score > 0.5
# fig_hm = plt.figure()
# plt.xlabel('Image')
# plt.ylabel('ROI')
# ax = plt.gca()
# xtick_majors = []
# xtick_majorlabels = []
# xtick_minors = []
# xtick_minorlabels = []
# for i, t in enumerate(tickinfo):
#     ti = tickinfo[t]
#     if ti['start'] == ti['end']:
#         xtick_majors.append(ti['start'] + 0.5)
#         xtick_majorlabels.append(ti['label'])
#     elif ti['end'] > ti['start']:
#         # https://stackoverflow.com/questions/13576805/matplotlib-hiding-specific-ticks-on-x-axis
#         # if ti['end'] - ti['start'] > 10:
#         #     # some logic here to create a tick and hide it 
#         #     xtmn = ax.xaxis.get_minor_ticks()
#         #     xtmn[3].label1.set_visible(False)
#         xtick_majors.append(ti['start'] + 0.5)
#         xtick_majorlabels.append(None)
#         xtick_minors.append(ti['labelpos'] + 0.5)
#         xtick_minorlabels.append(ti['label'])
#         if i == len(tickinfo) - 1:
#             xtick_majors.append(ti['end'] + 0.5)
#             xtick_majorlabels.append(None)
#     else:
#         warn('Heatmap plot tick issue for category {}'.format(ti))
# ax.set_xticks(xtick_majors)
# ax.set_xticklabels(xtick_majorlabels)
# ax.set_xticks(xtick_minors, minor=True)
# ax.set_xticklabels(xtick_minorlabels, minor=True)
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
# ax.tick_params(which='minor', length=0)
# # plt.imshow(np.mean(data[:]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[above_threshold],
# #            vmin=0.5-0.0001, vmax=0.5+0.0001, aspect='auto', cmap='gray', interpolation='none')
# # plt.imshow(np.mean(data[cond_idx]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[above_threshold[at_sortidx]],
# #            vmin=-1.0, vmax=1.0,
# #            aspect='auto', cmap='bwr', interpolation='none')
# plt.imshow(np.mean(data[stimcond]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[above_threshold],
#            vmin=-1.0, vmax=1.0,
#            aspect='auto', cmap='bwr', interpolation='none')
# ax.invert_yaxis()
# # ax = plt.gca()
# # ax.axvline(x=20)
# cbar = plt.colorbar()
# # cbar.ax.set_yticks(['0','1','2','>3'])
# # cbar.ax.set_yticklabels(['0','1','2','>3'])
# cbar.set_label('mean Zscore across stimulus period')
# fig_hm.tight_layout()
# fig_hm.show()
# if saving:
#     fig_hm.savefig(os.path.join(save_path, save_pfix + '_Heatmap_byCondition_sortMeanFace_threshZgt0p5' + save_ext),
#                 dpi=dpi, transparent=True)


# Plot the data


# Define a subset of ROIs to plot.
n_plot_ROIs = 9
n_plot_ROIs_div = np.round(n_plot_ROIs / 3).astype('int')
plot_ROI_subset = np.concatenate((range(0, n_plot_ROIs_div),
                                  range(np.floor(n_ROIs / 2 - n_plot_ROIs_div / 2).astype('int'),
                                        np.ceil(n_ROIs / 2 + n_plot_ROIs_div / 2).astype('int')),
                                  range(n_ROIs - n_plot_ROIs_div, n_ROIs)))
if len(plot_ROI_subset) > n_plot_ROIs:
    n_diff = len(plot_ROI_subset) - n_plot_ROIs
    plot_ROI_subset = np.concatenate((range(0, n_plot_ROIs_div),
                                      range(np.floor(n_ROIs / 2 - n_plot_ROIs_div / 2).astype('int'),
                                            np.ceil(n_ROIs / 2 + n_plot_ROIs_div / 2).astype('int') - n_diff),
                                      range(n_ROIs - n_plot_ROIs_div, n_ROIs)))

# Plot summary of each single ROI's responses ...
            
# ... by category, including the average for each condition within that category.
fr = md['framerate']
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * fr)
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
for r in range(n_plot_ROIs):
    ridx = sort_idx_dprime['zsc'][plot_ROI_subset[r]]
    dp = dprime[ridx]
    fig = plt.figure()
    fig.suptitle('ROI {} dprime={:0.2f}: mean response by category (each cond mean plotted)'.format(ridx, dp))
    axes = fig.subplots(nrows=n_metrics, ncols=n_cats)
    for mi, m in enumerate(metrics):
        ymin = np.min(np.mean(data[m][:, ridx, :, :], axis=1))
        ymax = np.max(np.mean(data[m][:, ridx, :, :], axis=1))
        for cat in range(n_cats):
            ax = axes[mi, cat]
            if mi == 0:
                ax.set_title(template_labels[categories[cat]])
            if cat == 0:
                ax.set_ylabel(metric_labels[m])
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                xticks = [x * fr for x in range(np.ceil(dur_trial).astype('int') + 1)]
                xticklabels = ['' if not np.isclose(xt, dur_isi * fr) and not np.isclose(xt, (dur_isi + dur_stim) * fr)
                               else '{}'.format(np.round(xt / fr).astype('int')) for xt in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.axis('off')
            ax.axvspan(dur_isi * fr, (dur_isi + dur_stim) * fr, color='0.9', zorder=0)
            ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
            n_cnd_in_cat = data[data['cat'] == categories[cat]]['cond'].shape[0]
            for cnd in range(n_cnd_in_cat):
                ax.plot(xs,
                        np.mean(data[data['cat'] == categories[cat]][m][cnd, ridx, :, :], axis=0),
                        linewidth=0.5, markersize=0.5, color=str(np.linspace(0.4, 0.7, n_cnd_in_cat)[cnd]), zorder=1)
            Fmean = np.mean(data[data['cat'] == categories[cat]][m][:, ridx, :, :], axis=(0, 1))
            Fsem = np.std(data[data['cat'] == categories[cat]][m][:, ridx, :, :], axis=(0, 1)) / np.sqrt(n_cnd_in_cat)
            ax.plot(xs, Fmean, color='0.0', zorder=3)
            ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, facecolor='0.2', alpha=0.6, zorder=2)
    plt.show()
del mi, m, xs, xticks, xticklabels


# ... by conditions (selected subset), including the average for each trial within that condition
fr = md['framerate']
fig = plt.figure()
fig.suptitle('mean response by condition (each trial plotted)', fontsize=8)
m = 'Fzsc'
focus_cat = b'face_mrm'
bool_focuscat = data['cat'] == focus_cat
stims = data[bool_focuscat]['stimulus']
stimconds = [i for i, x in sorted(enumerate(stims), key=lambda_sort)]
sortedstims = stims[stimconds]
conds_in_fcat = [i for i, x in sorted(enumerate(data[bool_focuscat]['stimulus']), key=lambda_sort)]
n_cnd_in_fcat = len(sortedstims)
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * fr)
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
axes = fig.subplots(nrows=(n_plot_ROIs + 1), ncols=(n_cnd_in_fcat + 1), sharey='row')
for r in range(n_plot_ROIs):
    ridx = sort_idx_dprime[m][plot_ROI_subset[r]]
    if r == 0:
        ax = axes[0, 0]
        ax.axis('off')
        for cnd in range(20):
            bool_cnd = data['cond'] == sortedstims[cnd].condition
            ax = axes[0, cnd + 1]
            ax.axis('off')
            ax.imshow(mpimg.imread(data[bool_cnd]['stimulus'][0].filepath))
    pr = r + 1
    
    # Leftmost summary plot of category averages.
    ymin = np.min(np.mean(data[m][:, ridx, :, :], axis=1))
    ymax = np.max(np.mean(data[m][:, ridx, :, :], axis=1))
    # if m == 0:
    #     ax.set_title(template_labels[categories[cat]], fontsize=10)
    ax = axes[pr, 0]
    ax.axis('off')
    # if r == 0:
    #     ax.set_ylabel(metric_labels[m], fontsize=8)
    # elif r == n_plot_ROIs - 1:
    #     ax.set_xlabel('Time (sec)', fontsize=6)
    # ax.tick_params(axis='both', which='major', length=2, labelsize=6)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # xticks = [x * fr for x in range(np.ceil(dur_trial).astype('int') + 1)]
    # xticklabels = ['' if not np.isclose(xt, dur_isi * fr) and not np.isclose(xt, (dur_isi + dur_stim) * fr) 
    #                else '{}'.format(np.round(xt / fr).astype('int')) for xt in xticks]
    # ax.set_xticks(xticks)            
    # ax.set_xticklabels(xticklabels)
    ax.axvspan(dur_isi * fr, (dur_isi + dur_stim) * fr, color='0.9', zorder=0)
    ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
    for cat in range(n_cats):
        bool_cat = data['cat'] == categories[cat]
        Fmean = np.mean(data[bool_cat][m][:, ridx, :, :], axis=(0, 1))
        Fsem = np.std(data[bool_cat][m][:, ridx, :, :], axis=(0, 1)) / np.sqrt(n_cnd_in_cat)
        ax.plot(xs, Fmean, color=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), linewidth=1, zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem,
                        facecolor=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), alpha=0.6, zorder=2)

    # Plot each cond
    for cnd in range(n_cnd_in_fcat):
        bool_cnd = data['cond'] == sortedstims[cnd].condition
        ax = axes[pr, cnd + 1]
        ax.axis('off')
        ax.axvspan(dur_isi * fr, (dur_isi + dur_stim) * fr, color='0.9', zorder=0)
        ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
        for t in range(n_reps):
            ax.plot(xs, 
                    data[bool_cnd][m][0, ridx, t, :],
                    color=str(np.linspace(0.4, 0.7, n_reps)[t]), linewidth=0.1)
        Fmean = np.mean(data[bool_cnd][m][0, ridx, :, :], axis=0)
        Fsem = np.std(data[bool_cnd][m][0, ridx, :, :], axis=0) / np.sqrt(n_cnd_in_cat)
        ax.plot(xs, Fmean, color='0.0', linewidth=1, zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, facecolor='0.0', alpha=0.6, zorder=2)
del m, xs, bool_cat, bool_cnd
plt.show()


# ... by conditions (selected subset), as a trial-averaged heatmap
fr = md['framerate']
fig = plt.figure()
fig.suptitle('trial-averaged heat maps by condition', fontsize=8)
m = 'Fzsc'
focus_cat = b'face_mrm'
bool_focuscat = data['cat'] == focus_cat
stims = data[bool_focuscat]['stimulus']
stimconds = [i for i, x in sorted(enumerate(stims), key=lambda_sort)]
sortedstims = stims[stimconds]
conds_in_fcat = [i for i, x in sorted(enumerate(data[bool_focuscat]['stimulus']), key=lambda_sort)]
n_cnd_in_fcat = len(sortedstims)
axes = fig.subplots(nrows=2, ncols=(n_cnd_in_fcat + 1), height_ratios=[0.25, 2.75], sharey='row')

pr = 0
ax = axes[pr, 0]
ax.axis('off')
for cnd in range(20):
    bool_cnd = data['cond'] == sortedstims[cnd].condition
    ax = axes[pr, cnd + 1]
    ax.axis('off')
    img_st = ax.imshow(mpimg.imread(data[bool_cnd]['stimulus'][0].filepath))
    # ax.set_ylim((0, img_st.get_size()[0]))

pr = 1
ax_dp = axes[pr, 0]
ax_dp.set_xlabel('Face d′')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(linestyle='--', linewidth=0.5, color='0.75')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=1)
        ax_dp.axhline(np.where(dprime[sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=1)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[sort_idx_dprime[m]], threshold_dprime), atol=0.05),
                      color='0.2', linestyle='dotted', linewidth=1)

for cnd in range(n_cnd_in_fcat):
    bool_cnd = data['cond'] == sortedstims[cnd].condition
    ax = axes[pr, cnd + 1]
    ax.axis('off')
    img_hm = ax.imshow(np.mean(data[bool_cnd][m][0, :, :, :], axis=1)[sort_idx_dprime[m]],
                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
    xlines = [dur_isi * fr, (dur_isi + dur_stim) * fr]
    for xl in xlines:
        ax.axvline(x=xl, linestyle='--', linewidth=0.5, color='0.6')

del m, bool_cnd
plt.show()


# Plot heatmap of mean responses to all presented conditions (images) for ROIs
# with at least one stimulus period z-score > 0.5
m = 'Fzsc'
# fig_hm, (ax_hm, ax_dp, ax_fsi) = plt.subplots(1, 3, width_ratios=[7.5, 0.75, 0.75], sharey=True)
fig_hm, (ax_hm, ax_dp) = plt.subplots(1, 2, width_ratios=[7.5, 0.75], sharey=True)
plt.subplots_adjust(wspace=0.05)
ax_hm.set_xlabel('Stimulus Image')
ax_hm.set_ylabel('ROI')
xtick_majors = []
xtick_majorlabels = []
xtick_minors = []
xtick_minorlabels = []
for i, t in enumerate(tickinfo):
    ti = tickinfo[t]
    if ti['start'] == ti['end']:
        xtick_majors.append(ti['start'] + 0.5)
        xtick_majorlabels.append(ti['label'])
    elif ti['end'] > ti['start']:
        # https://stackoverflow.com/questions/13576805/matplotlib-hiding-specific-ticks-on-x-axis
        # if ti['end'] - ti['start'] > 10:
        #     # some logic here to create a tick and hide it 
        #     xtmn = ax.xaxis.get_minor_ticks()
        #     xtmn[3].label1.set_visible(False)
        xtick_majors.append(ti['start'] - 0.5)
        xtick_majorlabels.append(None)
        xtick_minors.append(ti['labelpos'])
        xtick_minorlabels.append(ti['label'])
        if i == len(tickinfo) - 1:
            xtick_majors.append(ti['end'] + 0.5)
            xtick_majorlabels.append(None)
    else:
        warn('Heatmap plot tick issue for category {}'.format(ti))
ax_hm.set_xticks(xtick_majors)
ax_hm.set_xticklabels(xtick_majorlabels)
ax_hm.set_xticks(xtick_minors, minor=True)
ax_hm.set_xticklabels(xtick_minorlabels, minor=True)
plt.setp(ax_hm.xaxis.get_majorticklabels(), rotation=90)
ax_hm.tick_params(which='minor', length=0)
img_hm = ax_hm.imshow(np.mean(data[stimcond]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[sort_idx_dprime[m]],
                      vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
# ax_hm.invert_yaxis()
# ax_hm.axvline(x=20)

cbar = plt.colorbar(img_hm, ax=ax_hm, shrink=0.6)  # , location='bottom', shrink=0.6)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('mean Zscore during stimulus')

ax_dp.set_xlabel('Face d′')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(linestyle='--', linewidth=0.5, color='0.75')  # axis='x'
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=1)
        ax_dp.axhline(np.where(dprime[sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=1)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                      color='0.2', linestyle='dotted', linewidth=1)
    
# ax_fsi.set_xlabel('FSI')
# ax_fsi.set_axisbelow(True)
# ax_fsi.set_xlim([-1, 1])
# ax_fsi.barh(range(0, n_ROIs), FSIs_zsc[sort_idx_dprime[m]], height=1.0, color='0.5')
# ax_fsi.axvline(x=0, color='0.0', linewidth=0.5)
# ax_fsi.spines['right'].set_visible(False)
# ax_fsi.spines['left'].set_visible(False)
# ax_fsi.grid(linestyle='--', linewidth=0.5, color='0.75')  # axis='x'
# for tick in ax_fsi.yaxis.get_major_ticks():
#     tick.tick1line.set_visible(False)
#     tick.tick2line.set_visible(False)
#     tick.label1.set_visible(False)
#     tick.label2.set_visible(False)

plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=8)
# fig_hm.tight_layout()
fig_hm.show()
if saving:
    fig_hm.savefig(os.path.join(save_path, save_pfix + '_Heatmap_byCondition_sortMeanFace' + save_ext),
                   dpi=dpi, transparent=True)
del m


# # Plot sorted heatmap of mean responses to all presented conditions (images) for ROIs 
# # with at least one stimulus period z-score > 0.5
# sortidx = np.argsort(Fzsc_allfaces_meanRstimall - Fzsc_allobjs_meanRstimall - Fzsc_allbodies_meanRstimall)
# plt.xlabel('Image')
# plt.ylabel('ROI')
# ax = plt.gca()
# ax.set_xticks([20, 40])
# ax.set_xticklabels([None, None])
# ax.set_xticks([10, 30, 50], minor=True)
# ax.set_xticklabels(['faces', 'objects', 'bodies'], minor=True)
# xticks = ax.xaxis.get_major_ticks()
# ax.tick_params(which='minor', length=0)
# plt.imshow(np.mean(data[:]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[sortidx][above_threshold], 
#            vmin=-1.0, vmax=1.0, 
#            aspect='auto', cmap='bwr', interpolation='none')
# cbar = plt.colorbar()
# # cbar.ax.set_yticks(['0','1','2','>3'])
# # cbar.ax.set_yticklabels(['0','1','2','>3'])
# cbar.set_label('mean Zscore across stimulus period')
# plt.show()

# Determine for each ROI the category of the condition (image) that elicited the largest response
m = 'Fzsc'
top_cat_id = [np.argwhere(categories == ROI_stats_df[m]['peak_cat'][r])[0][0] for r in range(n_ROIs)]
top_cat_idn = np.divide(top_cat_id, len(cat_subset))

# Plot for each ROI the category of the condition (image) eliciting the largest response
above_threshold = np.where(ROI_stats_df[m]['peak_cond_val'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response, z > 0.5', save_path=sp)

# Plot for each ROI the category of the condition (image) eliciting the largest response
above_threshold = np.where(np.abs(FSI[m]) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response, ' +
                              'FSI > {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# Determine for each ROI which category elicited the largest average response
# TODO improve variable naming here for clarity

mean_by_cat = np.array([np.nanmean(data[data['cat'] == cat]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1)) for cat in cat_subset]).swapaxes(0, 1)
top_cat_mean = categories[np.argmax(mean_by_cat, axis=-1)]

# top_cat_mean_id = [np.argwhere(categories == top_cat_mean[r])[0][0] for r in range(n_ROIs)]
# top_cat_mean_idn = np.divide(top_cat_mean_id, len(categories))
# TODO check if fob here can be cat_subset
top_cat_mean_id = [np.argwhere(fob == top_cat_mean[r])[0][0] for r in range(n_ROIs)]
top_cat_mean_idn = np.divide(top_cat_mean_id, len(cat_subset))


# Plot heatmap of mean responses to all presented conditions (images) for ROIs 
# with at least one stimulus period z-score > 0.5
above_threshold = np.where(ROI_stats_df[m]['peak_cond_val'] > 0.5)[0]
fhm = plt.figure()
plt.xlabel('Image Category')
plt.ylabel('ROI')
ax = plt.gca()
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['faces', 'objects', 'bodies'])
# plt.imshow(mean_by_cat[above_threshold],
#            vmin=0.1-0.0001, vmax=0.1+0.0001, aspect='auto', cmap='bwr', interpolation='none')
plt.imshow(mean_by_cat[above_threshold[at_sortidx]],
           vmin=-0.5, vmax=0.5, aspect='auto', cmap='bwr', interpolation='none')
cbar = plt.colorbar()
# cbar.ax.set_yticks(['0','1','2','>3'])
# cbar.ax.set_yticklabels(['0','1','2','>3'])
cbar.set_label('mean Zscore across stim period and images')
plt.show()
if saving:
    fhm.savefig(os.path.join(save_path, save_pfix + '_Heatmap_byCategory_sortMeanFace_threshZgt0p5' + save_ext),
                dpi=dpi, transparent=True)


# Plot for each ROI the category eliciting the largest average response
above_threshold = np.where(ROI_stats_df[m]['peak_cond_val'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category eliciting the largest average response, z > 0.5', save_path=sp)

# Plot for each ROI the category eliciting the largest average response
# for only ROIs with FSI > threshold
above_threshold = np.where(np.abs(FSI[m]) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category eliciting the largest average response, ' +
                              'FSI > {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# Plot relative response strength

# TODO make this more dynamic

Fzsc_fob = np.array([muR_F[m],
                     muR_NFobj[m],
                     muR_B[m]]).swapaxes(0, 1)

# Subtract the response to the least-tuned category to make it relative
# (otherwise, an ROI that responds to all categories would show up as white)
Fzsc_mean_min_cat = np.min(Fzsc_fob, axis=1)
for col_i in range(3):
    Fzsc_fob[:, col_i] = Fzsc_fob[:, col_i] - Fzsc_mean_min_cat

max_Fzsc = 0.5
Fzsc_fob_norm = Fzsc_fob / max_Fzsc
Fzsc_fob_norm[Fzsc_fob_norm > 1] = 1

above_threshold = np.where(ROI_stats_df[m]['peak_cond_val'] > 0.5)[0]
sn = save_pfix + '_ROIplot_ColorByRelativeResponseStrength_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='relative response strength, z > 0.5', save_path=sp)

above_threshold = np.where(FSI[m] > threshold_fsi)[0]
sn = save_pfix + '_ROIplot_ColorByRelativeResponseStrength_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='relative response strength, FSI > {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

#
# OLD VERSION OF CONTINUOUS PLOT
# Fzsc_for_plot_bfo = np.array([Fzsc_allbodies_meanRstimall,
#                               Fzsc_allfaces_meanRstimall,
#                               Fzsc_allobjs_meanRstimall]).swapaxes(0, 1)


# # % Plot tuned cells with continuous tuning-wheel

# # parameters
# RGB_multiplier = 2.5
# plotting_threshold_continuous = 0 # 0.25

# # Fzsc_for_plot_continuous = copy.deepcopy(Fzsc_by_cat_meanRstimallnorm)
# Fzsc_for_plot_continuous = copy.deepcopy(Fzsc_for_plot_bfo)

# # Subtract the response to the least-tuned category to make it relative
# # (otherwise, an ROI that responds to all categories would show up as white)
# Fzsc_least_tuned = np.min(Fzsc_for_plot_continuous, axis=1)
# for col_i in range(3):
#     Fzsc_for_plot_continuous[:, col_i] = Fzsc_for_plot_continuous[:, col_i] - Fzsc_least_tuned

# Fzsc_for_plot_continuous[Fzsc_for_plot_continuous > 1] = 1  # Cap the RGB values to 1
# Fzsc_for_plot_continuous = Fzsc_for_plot_continuous * RGB_multiplier  # This highlights ROIs with less tuning, at the expense of dynamic range
# # Fzsc_for_plot_continuous[:, [0, 1, 2]] = Fzsc_for_plot_continuous[:, [key_bodies,
# #                                                                       key_faces,
# #                                                                       key_objs]]  # Swap orders to change colors.

# # TODO : fix this because it gives more than the number stated as 
# thresholding_logical_vector_continuous = np.max(Fzsc_for_plot_continuous,
#                                                 axis=1) > plotting_threshold_continuous  # Threshold so we don't plot un-tuned neurons (particularly important if using RGB_multiplier > 1)
# # thresholding_logical_vector_continuous = ROIs_tuned_idx

# ROIs_for_plot_continuous = ROIs[thresholding_logical_vector_continuous]
# Fzsc_for_plot_continuous = Fzsc_for_plot_continuous[thresholding_logical_vector_continuous]


# plots.plot_roi_overlays(ROIs_for_plot_continuous, 
#                         Fzsc_for_plot_continuous,
#                         size=fov_size, 
#                         image=fov_image, 
#                         save_path=save_path)

# # plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous,
# #               size=fov_size, image=fov_image, title=title_str, save_path=save_path)

# # plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous,
# #               size=fov_size, image=fov_image, save_path=r'F:\Sync\Transient\Science\Conferences\20231111d-20231115d_SocietyForNeuroscience_AnnualMeeting_WashingtonDC\media')


# % Plot stimulus images

n_subconds = len(cond_subset)

if n_conds % 20 == 0:
    n_cols = 20
    n_rows = int(n_subconds / 20)
else:
    n_cols = 20
    n_rows = round(n_subconds / 20)

fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
                        figsize=(n_cols * 512 / dpi, n_rows * 512 / dpi),
                        layout='constrained')
for row in range(n_rows):
    for col in range(n_cols):
        i = col + (row * n_cols)
        ax = axs[row, col]
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        imp = os.path.join(stimimage_path, data[cond_idx[i]]['imagename'])
        ax.imshow(mpimg.imread(imp))
if saving:
    fig.savefig(os.path.join(save_path, save_pfix + '_StimulusImages' + save_ext),
                dpi=dpi, transparent=True)





# %% Plot tuned cells with discreete tuning-wheel

# # parameters
# plotting_threshold_discrete = 0.15
# subtract_responses_to_other_stim = True
# subtract_least_or_secondLeast_preferred_stim_responses = 0  # If set to 0, will subtract the responses to the least-preferred stim. If set to 1, will subtract the responses to the second-least (in this case, second-most) preferred stim


# Fzsc_for_plot_discrete = Fzsc_fob_norm  # Fzsc_for_plot_bfo

# # Subtract to all responses the responses to the second-preferred stimulus
# if subtract_responses_to_other_stim:
#     for row_i in range(len(Fzsc_for_plot_discrete)):
#         this_row = Fzsc_for_plot_discrete[row_i]
#         response_to_non_preferred_stim = sorted(set(this_row))[
#             subtract_least_or_secondLeast_preferred_stim_responses]  # This sorts the responses and selects the lowest(0) or second-lowest(1) response
#         Fzsc_for_plot_discrete[row_i] = this_row - response_to_non_preferred_stim


# above_threshold = np.where(FSIs_zsc > threshold_fsi)[0]

# Fzsc_for_plot_preferredKey = np.argmax(Fzsc_for_plot_discrete, axis=1)
# Fzsc_for_plot_discrete[:] = 0  # We will re-fill the preferredkeys with 1s in the folowwing for loop
# for roi_i in range(len(Fzsc_for_plot_discrete)):
#     Fzsc_for_plot_discrete[roi_i, Fzsc_for_plot_preferredKey[roi_i]] = 1

# # Fzsc_for_plot_discrete[:, [0, 1, 2]] = Fzsc_for_plot_discrete[:, [key_bodies, key_faces,
# #                                                                   key_objs]]  # Swap face indexes to be on the first column, making face-cells be red
# # Fzsc_for_plot_discrete[:,[1,2]] = Fzsc_for_plot_discrete[:,[2,1]] #Swap face indexes to be on the first column, making face-cells be red

# # plots.plot_ROIs_RGB(ROIs_for_plot_discrete, Fzsc_for_plot_discrete,
# #                     size=fov_size, image=fov_image, save_path=save_path)

# plots.plot_roi_overlays(ROIs[above_threshold], 
#                         Fzsc_for_plot_discrete[above_threshold],
#                         image=plots.auto_level_s2p_image(fov_image))


# TODO Plot responses from top, middle, bottom 10 sorted ROIs


# %% Other approaches for measuring/approximating tuning

# e.g. from https://www.biorxiv.org/content/10.1101/2022.03.06.483186v1.full.pdf
# Face selectivity was quantified by computing the d’ sensitivity index
# comparing trial averaged responses to faces and to non-faces:
# [eq]
# where 𝜇f and 𝜇nf are the across-stimulus averages of the trial-averaged
# responses to faces and non-faces, and 𝜎f and 𝜎nf are the across-stimulus
# standard deviations. This face d’ value quantifies how much higher
# (positive d’) or lower (negative d’) the response to a face is expected
# to be compared to an object, in standard deviation units.


# if normalize == 'dF/F':
#     Ftest_cond = FdFF_by_cond_meanRstim
#     Ftest_cat = FdFF_by_cat_meanRstim
# elif normalize == 'Zscore':
#     Ftest_cond = Fzsc_by_cond_meanRstim
#     Ftest_cat = Fzsc_by_cat_meanRstim
#
# if tuning == 't-test':
#     t_test_cond = scipy.stats.ttest_1samp(Ftest_cond, 0, axis=2)
#     p_vals_cond = t_test_cond[1]
#     p_vals_min_cond = np.min(p_vals_cond, axis=1)
#     tuning_index = 1 - p_vals_min_cond
#     t_test_cat = scipy.stats.ttest_1samp(Ftest_cat, 0, axis=2)
#     p_vals_cat = t_test_cat[1]
#     p_vals_min_cat = np.min(p_vals_cat, axis=1)
#     tuning_index_cat = 1 - p_vals_min_cat
# elif tuning == 'percentile':
#     first_quantile_all_conds = np.percentile(Ftest_cond, percentile, axis=2)
#     first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
#     tuning_index_cond = first_quantile_max_cond
#     first_quantile_all_cats = np.percentile(Ftest_cat, percentile, axis=2)
#     first_quantile_max_cat = np.max(first_quantile_all_cats, axis = 1)
#     tuning_index_cat = first_quantile_max_cat
# elif tuning == 'average':
#     average_cond = np.abs(np.mean(Ftest_cond, axis=-1))
#     average_max_cond = np.max(average_cond, axis=1)
#     tuning_index_cond = average_max_cond
#     average_cat = np.abs(np.mean(Ftest_cat, axis=-1))
#     average_max_cat = np.max(average_cat, axis=1)
#     tuning_index_cat = average_max_cat
# elif tuning == 'fsi':
#     #FSI = (mean responsefaces – mean responsenonface objects)/(mean responsefaces + mean responsenonface objects)
#     #average_cond = np.abs(np.mean(Frois_by_cond_meanRstim, axis=-1))
#     Rcatframes = np.mean(Ftest_cat, axis=-1)
#     Rcatnorm = Rcatframes + np.abs(np.min(Rcatframes))
#     catidx_face = [c for c in categories if categories[c]=='m'][0]
#     catidx_obj = [c for c in categories if categories[c]=='u'][0]
#     Rfaces = Rcatframes[:,catidx_face]
#     Robjs = Rcatnorm[:,catidx_obj]
#     fsi = (Rfaces - Robjs) / (Rfaces + Robjs)
#     tuning_index_cat = fsi
#     tuning_index_cond = fsi


# %% Define ROIs as tuned or untuned using the FSI

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For
# |face-selectivity index| = 1/3, that is, if the response to faces was at
# least twice (or at most half) that of nonface objects, a cell was classed
# as being face selective45–47.

# print('|FSI| threshold: {}'.format(threshold_fsi))
# tunidx_fsi = FSIs_zsc
# tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
# ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).squeeze()
# n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).shape[0]
# pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
# print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
# print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

# tuning_index_cond = tunidx_fsi
# tuning_index_cat = tunidx_fsi


# %% 

# # Non-zero hack
# FdFF_absmin = -np.inf
# for cd in np.unique(data['cond']):
#     absmintmp = np.abs(np.min(np.nanmean(data[data['cond'] == cd]['FdFF_meant'][:, :, idx_stim], axis=(0, -1))))
#     if absmintmp > FdFF_absmin:
#         FdFF_absmin = absmintmp
# Fzsc_absmin = -np.inf
# for cd in np.unique(data['cond']):
#     absmintmp = np.abs(np.min(np.nanmean(data[data['cond'] == cd]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1))))
#     if absmintmp > Fzsc_absmin:
#         Fzsc_absmin = absmintmp

# ImM_zsc = np.empty([n_ROIs_tuned, len(data['cond'][(data['cat'] == b'face_mrm')])])
# ImSIs_zsc = np.empty([n_ROIs_tuned, len(data['cond'][(data['cat'] == b'face_mrm')])])
# for cid, cim in enumerate(data['cond'][(data['cat'] == b'face_mrm')]):
#     Fzsc_nowface_meanRstimall = np.nanmean(data[data['cond'] == cim]['Fzsc_meant'][:, :, idx_stim],
#                                            axis=(0, -1)) + Fzsc_absmin
#     Fzsc_otherfaces_meanRstimall = np.nanmean(data[data['cond'] != cim]['Fzsc_meant'][:, :, idx_stim],
#                                               axis=(0, -1)) + Fzsc_absmin
#     for r in range(n_ROIs_tuned):
#         rti = ROIs_tuned_idx[r]
#         ImM_zsc[r, cid] = Fzsc_nowface_meanRstimall[rti]
#         ImSIs_zsc[r, cid] = (Fzsc_nowface_meanRstimall[rti] - Fzsc_otherfaces_meanRstimall[rti]) / \
#                             (Fzsc_nowface_meanRstimall[rti] + Fzsc_otherfaces_meanRstimall[rti])

# # idx_trial = range(n_samp_trial)
# # Fzsc_allfaces_meanRstim = np.nanmean(data[(data['cat'] == b'face_mrm') & (data['yaw'] == 0) & (data['roll'] == 0)]['Fzsc_meant'][:, :, idx_trial],
# #                                      axis=0) + Fzsc_absmin
# # sorting_ind = ROIs_tuned_idx
# # Fzsc_allfaces_meanRstim_sorted = Fzsc_allfaces_meanRstim[sorting_ind]
# # # cats = categories_sorted

# # plt.figure(dpi=1000)
# # plt.imshow(Fzsc_allfaces_meanRstim_sorted, cmap='bwr')

# # plt.clim(-1,1)
# # #plt.title('2-D Heat Map in Matplotlib')
# # #plt.colorbar()
# # plt.tick_params(left=False)
# # ax = plt.gca()
# # ax.tick_params(left=False, right=False, labelleft=False)
# # ax.set_xticks([c for c in range(0,60+1,20)])#, categories_sorted)
# # ax.set_xticklabels([], fontsize=3, rotation=90)
# # plt.show()


# plt.figure()
# plt.imshow(ImM_zsc[range(20)], cmap='bwr')
# #plt.clim(-1,1)
# #plt.title('2-D Heat Map in Matplotlib')
# plt.colorbar()
# plt.tick_params(left=False, bottom=False)
# ax = plt.gca()
# ax.tick_params(left=False, right=False, bottom=False, labelleft=False)
# ax.set_xticks([])  # c for c in range(0,20,20)])#, categories_sorted)
# ax.set_xticklabels([], fontsize=3, rotation=90)
# plt.xlabel('Face Image')
# plt.ylabel('ROI')
# plt.show()


# # tunidx_fsi = ImSIs_zsc
# # tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
# # ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).squeeze()
# # n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).shape[0]
# # pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
# # print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
# # print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

# maxim = np.unravel_index(ImSIs_zsc[:,:].argmax(), ImSIs_zsc.shape)[1]

# for it in range(20):
#     ImSIs_zsc_argsort_imt = ImSIs_zsc[:, it].argsort()
#     rois_sel = np.argwhere(np.abs(ImSIs_zsc[ImSIs_zsc_argsort_imt, it]) > threshold_fsi).squeeze()
#     cn = data['cond'][(data['cat'] == b'face_mrm')][it].decode()
#     sn = save_pfix + '_ROIplot_ColorByDiscrete_' + cn + '_inclFSIthrs' + save_ext
#     sp = os.path.join(save_path, sn) if saving else ''
#     plots.plot_roi_overlays(ROIs[above_threshold[rois_sel]],
#                             Fzsc_for_plot_discrete[above_threshold[rois_sel]],
#                             image=plots.auto_level_s2p_image(fov_image), save_path=sp)
