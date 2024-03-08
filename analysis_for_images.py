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
# from scipy.optimize import minimize as scipy_minimize
# from scipy.signal import find_peaks as find_peaks
import socket
from warnings import warn

import filters
import plots

# TODO: exclude suite2p badframes
# TODO: exclude when eyes are not open
# TODO: set path for images to include in figures during processing
# TODO: plot neuron traces over all conds or cats


# % Settings

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# [...] neurons (94%) were face selective (that is, face-selectivity index
# larger than 1/3 or smaller than -1/3, dotted lines).
fsi_tuning_thresh = 1 / 3

cell_probability_thresh = 0.0

save_path = ''
# save_path = r'F:\Sync\Transient\Science\Conferences\20240301d_FreiwaldLabMeeting\media'
stimimage_path = r''

plt.rcParams['figure.dpi'] = 600
dpi = plt.rcParams['figure.dpi']


# Remove stale metadata
if 'md' in locals():
    md = dict()
    del md


# % Specify data locations

# --  GOOD OLD Cadbury PD  20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx
# |FSI| threshold: 0.25
# Tuned ROIs: 894. Total ROIs: 6020.   (note: using cell_probability_threshold = 0.0)
# Percentage of tuned ROIs: 14.85%
# animal_str = 'Cadbury'
# # date_str = '20221016d_olds2p'
# date_str = '20221016d'
# session_str = '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'
# md = dict()
# md['framerate'] = 6.364
# md['fov'] = dict()
# md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
# md['fov']['w_px'] = 730
# md['fov']['h_px'] = 730
# # suite2p_str = 'suite2p_old*'
# suite2p_str = 'suite2p_cellpose2_d14px_pt-3p5_ft1p5*'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_FOBonly'


# # -- Cadbury UNCLEAR (fluid, z drift across time)
# animal_str = 'Cadbury'
# date_str = '20230510d'
# session_str = '155713tUTC_SP_depth200um_fov2190x2000um_res3p00x3p02umpx_fr06p993Hz_pow059p9mW_stimImagesSong230509dSel'


# # -- Cadbury OBJ ()
animal_str = 'Cadbury'
date_str = '20230809d'
session_str = '173936tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'
stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'


# # -- Cadbury PD ()
# animal_str = 'Cadbury'
# date_str = '20231001d'
# session_str = '190608tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p3mW_stimImagesFOBmany'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmany\Images\20230728d'

# # -- Cadbury PD ()
# animal_str = 'Cadbury'
# date_str = '20231001d'
# session_str = '200422tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow060p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'


# # -- Cadbury PD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '142836tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury PD (150um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '145031tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p0mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury PD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '153340tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury PD (250um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '154955tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow089p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury MD (150um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '162025tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury MD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '163738tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow079p9mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury MD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '165634tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow079p9mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury OBJ (200um, )
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '173850tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow070p3mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'


# # -- Cadbury OBJ (200um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmany\Images\20230728d'

# # -- Cadbury OBJ (150um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '162705tUTC_SP_depth150um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow049p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury OBJ (250um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '164258tUTC_SP_depth250um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow069p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury OBJ (300um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '170046tUTC_SP_depth300um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow099p7mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury PD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '174147tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow080p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury PD (200um, )
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '180407tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p359Hz_pow080p2mW_stimMultimodal'
# stimimage_path = r'multimodal'


# # -- Cadbury OBJ (200um, )
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '185135tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury OBJ ( )
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '190745tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow075p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'

# # -- Cadbury OBJ ( )
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '192426tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'


# # -- Dali PD (headpost not tight)
# animal_str = 'Dali'
# date_str = '20230511d'
# session_str = '134800tUTC_SP_depth200um_fov2190x2000um_res3p02x3p02umpx_fr06p993Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'

# # -- Dali PD ()
# animal_str = 'Dali'
# date_str = '20230511d'
# session_str = '150200tUTC_SP_depth300um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow065p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'


# # -- Dali ? () CHECK MEANIM
# animal_str = 'Dali'
# date_str = '20230515d'
# session_str = '135100tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'

# # -- Dali ? () CHECK MEANIM
# animal_str = 'Dali'
# date_str = '20230515d'
# session_str = '144500tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'


# # -- Dali ? (ANISO) CHECK MEANIM
# animal_str = 'Dali'
# date_str = '20230522d'
# session_str = '153415tUTC_SP_depth200um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow051p8mW_stimImagesFOBsel230517dAniso'
# ????? stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\MarmosetFOB2018\20230517d\Scaled512x512_MaskEroded3_SHINEdLum_RGBA_reorg_subset'

# # -- Dali ? () CHECK MEANIM
# animal_str = 'Dali'
# date_str = '20230522d'
# session_str = '170053tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow051p8mW_stimImagesSong230509dSel'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\Song_etal_Wang_2022_NatCommun\480288_equalized_RGBA_selected20230509d'


# ... Dali .... aniso until 20230804d




# 
# # 
# # # -- MAYBE Cadbury MD?BOD?
# # #   - not sure what to make of this, area tuning looks very jumbled now
# # # animal_str = 'Cadbury'
# # # date_str = '20231003d'
# # # session_str = '165634tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow079p9mW_stimImagesFOBmin'

# # # -- DECENT Cadbury OBJ
# # # animal_str = 'Cadbury'
# # # date_str = '20231007d'
# # # session_str = '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'
# # # -- OKAY Cadbury OBJ 
# # # animal_str = 'Cadbury'
# # # date_str = '20231007d'
# # # session_str = '164258tUTC_SP_depth250um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow069p8mW_stimImagesFOBmin'

# # # -- OKAY Cadbury OBJ
# # # animal_str = 'Cadbury'
# # # date_str = '20231018d'
# # # session_str = '185135tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# # # -- OKAY Cadbury  OBJ
# # # animal_str = 'Cadbury'
# # # date_str = '20231018d'
# # # session_str = '190745tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow075p8mW_stimImagesFOBmin'
# # # -- MAYBE Cadbury OBJ?PV? 
# # # animal_str = 'Cadbury'
# # # date_str = '20231018d'
# # # session_str = '192426tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'

# # # -- MAYBE Dali PD mixed with anisoStim
# # # animal_str = 'Dali'
# # # date_str = '20230606d'
# # # session_str = '135543tUTC_SP_depth300um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow091p0mW_stimImagesFOBsel230517dAniso'

# # # -- MAYBE Dali PD mixed with anisoStim
# # # animal_str = 'Dali'
# # # date_str = '20230608d'
# # # session_str = '124502tUTC_SP_depth350um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow092p7mW_stimImagesFOBsel230517dAniso'
# # # -- MAYBE Dali PD mixed with anisoStim
# # # animal_str = 'Dali'
# # # date_str = '20230608d'
# # # session_str = '134220tUTC_SP_depth400um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow122p3mW_stimImagesFOBsel230517dAniso'

if session_str.find('_') != -1:
    session_abbrev_str = session_str[0:session_str.find('_')]
    title_str = animal_str + '_' + date_str + '_' + session_abbrev_str
else:
    session_abbrev_str = ''
    title_str = animal_str + '_' + date_str + '_' + session_str
savepfix_str = animal_str + date_str + session_abbrev_str
mdfile_str = '*_metadata.pickle'
datafile_str = '*_00001.tif'
logfile_str = '*.log'
logfile_re = r'.*^((?!disptimes).)*$'  # Exclude log files whose names contain 'disptimes'
stimlogfile_str = '*_stimlog.csv'
if 'suite2p_str' not in locals():
    suite2p_str = 'suite2p*'
suite2p_plane_str = 'plane0'

# Load imaging session information
system_name = socket.gethostname()
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/suite2p_results'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data'
else:
    base_path = r'F:\Data'

if save_path == '':
    saving = False
else:
    saving = True

session_path = os.path.join(base_path, animal_str, date_str, session_str)
mdfile_list = glob(os.path.join(session_path, mdfile_str))
datafile_list = [f for f in glob(os.path.join(session_path, datafile_str))
                 if os.path.isfile(f)]
logfile_list = [f for f in glob(os.path.join(session_path, logfile_str))
                if re.search(logfile_re, f) and os.path.isfile(f)]
stimlogfile_list = [f for f in glob(os.path.join(session_path, stimlogfile_str))
                    if os.path.isfile(f)]
suite2p_list = [d for d in glob(os.path.join(session_path, suite2p_str))
                if os.path.isdir(d)]

if 'md' not in locals():
    if not mdfile_list and not datafile_list:
        raise RuntimeError('Could not find metadata file or image data file.')
    if len(mdfile_list) > 0:
        if len(mdfile_list) > 1:
            warn('Found multiple metadata files, using the first one: {}'.format(mdfile_list[0]))
        md_path = mdfile_list[0]
        if os.path.isfile(md_path):
            with open(md_path, 'rb') as mdf:
                md = pickle.load(mdf)
            # jf = open(md_path, 'r')
            # md = json.load(jf)
            # jf.close()
        else:
            raise RuntimeError('Could not load metadata from file.')
    elif len(datafile_list) > 0:
        warn('Could not find metadata file, using image data file.')
        df_path = datafile_list[0]
        if os.path.isfile(df_path):
            import metadata
            simd = metadata.get_metadata(df_path)
            md = metadata.extract_useful_metadata(simd)
    else:
        raise RuntimeError('Could not load metadata from either file.')

if len(logfile_list) > 0:
    lf_path = logfile_list[0]
    if len(logfile_list) > 1:
        warn('Found multiple log files, using the first one: {}'.format(lf_path))
    if os.path.isfile(lf_path):
        lf = open(lf_path, 'r')
        log = lf.read()
        lf.close()
    else:
        raise RuntimeError('Could not find log file.')
else:
    lf_path = None

if len(stimlogfile_list) > 0:
    slf_path = stimlogfile_list[0]
    if len(stimlogfile_list) > 1:
        warn('Found multiple stimlog files, using the first one: {}'.format(slf_path))
    if os.path.isfile(slf_path):
        stimlog = pd.read_csv(slf_path)
    else:
        stimlog = None

if len(suite2p_list) > 0:
    if len(suite2p_list) > 1:
        warn('Found multiple suite2p folders, using the first one: {}'.format(os.path.basename(suite2p_list[0])))
    s2p_path = suite2p_list[0]
    s2p_plane_path = os.path.join(s2p_path, suite2p_plane_str)
    if not os.path.isdir(s2p_plane_path):
        raise RuntimeError('Could not load suite2p plane0 folder.')
else:
    raise RuntimeError('Could not find suite2p folder.')


# % Load suite2p outputs

s2p_iscell = np.load(os.path.join(s2p_plane_path, 'iscell.npy'))
s2p_F = np.load(os.path.join(s2p_plane_path, 'F.npy'))
s2p_stat = np.load(os.path.join(s2p_plane_path, 'stat.npy'), allow_pickle=True)
s2p_ops = np.load(os.path.join(s2p_plane_path, 'ops.npy'), allow_pickle=True).item()

cellinds = np.where(s2p_iscell[:, 1] >= cell_probability_thresh)[0]
# cellinds = np.where(s2p_iscell[:,0] == 1.0)[0]
# cellinds = np.logical_and(s2p_iscell[:, 1] >= cell_probability_thresh, np.std(s2p_F, axis=1) != 0)
# cellinds = np.logical_and(s2p_iscell[:, 0] == 1.0, np.std(s2p_F, axis=1) != 0)
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
n_frames = Frois.shape[1]
bad_frames = np.where(s2p_ops['badframes'])

# Compute dF/F and z-scored dF/F
n_ROIs = Frois.shape[0]
FdFF_raw = (Frois - np.mean(Frois, axis=1)[:, np.newaxis]) / np.mean(Frois, axis=1)[:, np.newaxis]
Fzsc_raw = (Frois - np.mean(Frois, axis=1)[:, np.newaxis]) / np.std(Frois, axis=1)[:, np.newaxis]

# Approaches for computing dF/F using more sophisticated baseline fluorescence (F0) calculations.
# from Gordon Smith (https://doi.org/10.1038/s41592-023-02098-1):
#   Baseline fluorescence (F0) was calculated by applying a rank-order filter to the raw fluorescence trace (tenth
#   percentile) with a rolling time window of 60 sec.
# from Gordon Smith (https://doi.org/10.1016/j.jneumeth.2023.110051):
#   The baseline fluorescence (F0) for each pixel was obtained by applying a rank-order filter to the raw fluorescence
#   trace with a rank 70 samples and a time window of 30 sec (451 samples).
# from Wilson et al Fitzpatrick (e.g. https://doi.org/10.1038/s41586-018-0354-1):
#   ΔF/F0 was computed by defining F0 using a 60 sec percentile filter (typically 10th percentile), which was then
#   low-pass filtered at 0.01 Hz.
#
# filter_percentile = 10
# filter_window = 60  # sec
#
# # F0_rnk = np.zeros((n_ROIs, n_frames))
# # FdFF_rnk = np.zeros((n_ROIs, n_frames))
# F0_pct = np.zeros((n_ROIs, n_frames))
# FdFF_pct = np.zeros((n_ROIs, n_frames))
# # F0_rnkbw = np.zeros((n_ROIs, n_frames))
# # FdFF_rnkbw = np.zeros((n_ROIs, n_frames))
# F0_pctbw = np.zeros((n_ROIs, n_frames))
# FdFF_pctbw = np.zeros((n_ROIs, n_frames))
# for r in range(n_ROIs):
#     # F0_rnk[r] = filters.rank_order_filter(Frois[r], p=filter_percentile, n=round(filter_window * md['framerate']))
#     # FdFF_rnk[r] = (Frois[r] - F0_rnk[r]) / F0_rnk[r]
#     F0_pct[r] = filters.percentile_filter_1d(Frois[r], p=filter_percentile, n=round(filter_window * md['framerate']))
#     FdFF_pct[r] = (Frois[r] - F0_pct[r]) / F0_pct[r]
#     # F0_rnkbw[r] = filters.butterworth_filter(F0_rnk[r], fs=md['framerate'], p=filter_percentile)
#     # FdFF_rnkbw[r] = (Frois[r] - F0_rnkbw[r]) / F0_rnkbw[r]
#     F0_pctbw[r] = filters.butterworth_filter(Frois[r], fs=md['framerate'], p=filter_percentile)
#     FdFF_pctbw[r] = (Frois[r] - F0_pctbw[r]) / F0_pctbw[r]
#
# n_rolling_average = 120
# FdFF_sw = np.zeros((n_ROIs, n_frames))
# for r in range(n_ROIs):
#     FdFF_sw[r] = FdFF_raw[r] - np.convolve(FdFF_raw[r], np.ones(n_rolling_average)/n_rolling_average, mode='same')

# r = 8
# plt.plot(FdFF_raw[r, 0:249], 'k', alpha=0.5)
# plt.plot(FdFF_sw[r, 0:249], 'g', alpha=0.5)
# plt.plot(FdFF_rnk[r, 0:249], 'c', alpha=0.5)
# plt.plot(FdFF_pct[r, 0:249], 'y', alpha=0.5)
# plt.plot(FdFF_rnkbw[r, 0:249], 'b', alpha=0.5)
# plt.plot(FdFF_pctbw[r, 0:249], 'r', alpha=0.5)

# # Deconvolve fluorescence signals.
# from oasis.oasis_methods import oasisAR1
# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
#
# FdFF_c = np.zeros((n_ROIs, n_frames))
# for r in range(n_ROIs):
#     # c[r], _ = oasisAR1(Frois[r].astype('float64'), g=.95, s_min=.55)
#     FdFF_c[r], _, b, g, lam = deconvolve(FdFF_raw[r], penalty=1)
# # plot_trace(True)
# r = 0
# plt.plot(Frois[r, 0:500], 'b', alpha=0.5)
# plt.plot(FdFF_raw[r, 0:500], 'k', alpha=0.5)
# plt.plot(FdFF_c[r, 0:500], 'm')


# % Extract stimulus information from log file

# *** TODO load from a pickle file or pandas frame instead of a text log

# if stimlog is None:

file = open(os.path.join(lf_path), 'r')
lines = file.read().splitlines()
file.close()

# 37.1533         EXP     trial 0/240, stim start, image, cond=7, name=image7:b16.png,
# path=/FreiwaldSync/MarmoScope/Stimulus/Images/Song_etal_Wang_2020_NatCommun/480288_equalized_RGBA_FOBonly/b16.png,
# units=deg, pos=[0. 0.], size=[12.   7.2], ori=0.0, color=[1. 1. 1.], colorSpace=rgb, contrast=1.0,
# opacity=1.0, texRes=512, acqfr=23, AI_data.shape=(1336, 5)
trialdata = {}
ims = {}
impaths = {}
lf_categories = {}
tmp_image = None
tmp_imagepath = None
tmp_category = None
tmp_catid = None
tmp_cond = None
tmp_acqfr = None
tmp_stimtimestr = ''
stimtime_mode = False
tmp_isitimestr = ''
isitime_mode = False
for line in lines:
    if 'EXP \tstim_times:' in line:
        stimtime_mode = True
    if 'EXP \tinterstim_times:' in line:
        isitime_mode = True
    if stimtime_mode:
        tmp_stimtimestr = tmp_stimtimestr + line
        if ']' in line:
            stimtime_mode = False
    if isitime_mode:
        tmp_isitimestr = tmp_isitimestr + line
        if ']' in line:
            isitime_mode = False

    if 'stim start' not in line or 'image' not in line:
        continue

    col = line.split('trial')
    if not col:
        continue
    subcol = [sc.strip() for sc in col[1].split(',')]
    tmp_trial = int(subcol[0].split('/')[0].strip())
    if 'cond' in subcol[3]:
        tmp_cond = int(subcol[3].split('=')[1].strip())
    else:
        print('could not get cond from log entry')
    if 'image' in subcol[4]:
        tmp_image = subcol[4].split(':')[1].strip()
        if tmp_image not in ims:
            ims[tmp_image] = tmp_cond
        tmp_category = tmp_image[0]
        if tmp_category not in lf_categories:
            lf_categories[tmp_category] = len(lf_categories)
        tmp_catid = lf_categories[tmp_category]
    else:
        print('could not get image name from log entry')
    if 'path' in subcol[5]:
        tmp_imagepath = subcol[5].split('=')[1].strip()
        if tmp_imagepath not in impaths:
            impaths[tmp_imagepath] = tmp_cond
    else:
        print('could not get image name from log entry')
    if 'acqfr' in subcol[15]:
        tmp_acqfr = int(subcol[15].split('=')[1].strip())
    else:
        print('could not get acqfr from log entry')
    trialdata[tmp_trial] = {'cond': tmp_cond,  # effectively image_id
                            'image': tmp_image,
                            'category': tmp_category,
                            'catid': tmp_catid,
                            'acqfr': tmp_acqfr}

if tmp_stimtimestr != '':
    s_si = tmp_stimtimestr.find('[')
    s_ei = tmp_stimtimestr.find(']')
    stimtimes = np.fromstring(tmp_stimtimestr[s_si+1:s_ei].strip(' []'), sep=' ')
    dur_stim = np.round(np.mean(stimtimes), 2)
else:
    warn('Could not automatically detect stimulus duration, assuming default value of 1.0 sec.')
    dur_stim = 1.0

if tmp_isitimestr != '':
    s_si = tmp_isitimestr.find('[')
    s_ei = tmp_isitimestr.find(']')
    isitimes = np.fromstring(tmp_isitimestr[s_si+1:s_ei].strip(' []'), sep=' ')
    dur_isi = np.round(np.min(isitimes), 2)
else:
    warn('Could not automatically detect interstimulus duration, assuming default value of 1.0 sec.')
    dur_isi = 1.0

n_samp_stim = int(np.ceil(dur_stim * md['framerate']))
n_samp_isi = int(np.round(dur_isi * md['framerate']))
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

lf_categories = {v: k for k, v in lf_categories.items()}
images_filename = {v: k for k, v in ims.items()}
# image_names = {v: k.split('.')[0] for k, v in ims.items()}
image_paths = {v: k for k, v in impaths.items()}
image_dirpaths = {v: os.path.dirname(k) for k, v in impaths.items()}
image_folders = {v: os.path.split(os.path.dirname(k))[-1] for k, v in impaths.items()}
same_dirpaths = np.unique(list(image_dirpaths.values())).size == 1
if same_dirpaths:
    image_dirpath = image_dirpaths[0]
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
        image_set = image_folders[0]
else:
    raise RuntimeError('Images are not all from the same folder. Processing for this not supported yet.')
image_filenames = {v: os.path.basename(k) for k, v in impaths.items()}
image_names = {v: os.path.splitext(os.path.basename(k))[0] for k, v in impaths.items()}
lf_conditions = image_names

# trialdataarr[trial_idx] = [cond/imageid, category_id, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['catid'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)
all_stim_start_frames = trialdataarr[:, 2]

if len(np.unique(all_stim_start_frames)) != len(all_stim_start_frames):
    raise RuntimeError('Imaging was interrupted or stopped before stimulus. ' + \
                       'Handling this is not yet implemented.')                                

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:, 0])
n_conds = len(conds)
n_trials = int(len(trialdata) / n_conds)
cats = np.unique(trialdataarr[:, 1])
n_cats = len(cats)
conds_per_cat = int(n_conds / n_cats)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdataarr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]
fridx = acqfr_by_conds


# % Organize and average fluorescence traces

data = np.zeros(n_conds, dtype=[('cond', 'S8'),
                                ('cat', 'S8'),
                                ('id', 'S8'),
                                ('view', 'i2'),
                                ('roll', 'i2'),
                                ('imagename', np.unicode_, 256),
                                ('FdFF', 'f4', (n_ROIs,
                                                n_trials,
                                                n_samp_trial)),
                                ('Fzsc', 'f4', (n_ROIs,
                                                n_trials,
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
    tmp_view = -32768
    tmp_roll = -32768
    imn = image_names[c]
    tmp_imagename = image_filenames[c]
    if image_set == 'FOBmin' or image_set == 'FOBmany':
        # pattern_fn = r'^([0-9]{6}tUTC).*$'
        # FreiwaldFOB2018_Marm_Body_Spring_7_erode3px
        # FreiwaldFOB2012_Human_Head_10_erode3px
        # FreiwaldFOB2018_Marm_Head_Hunter_8_erode3px
        # FreiwaldFOB2018_Objects_Manmade1_8_erode3px
        # FreiwaldMarmosetCartoon_0_erode3px
        # pattern_imn = r'^[^_]*_?([^_]+)_([^_]+)_?([^_]+)?_([0-9]+)_?[^_]*_?(inverted)?$'
        pattern_imn = r'^(Freiwald(FOB)?([0-9]*)?)?_?([^_]+)_([^_]+)_?([^_]+)?_([0-9]+)_?[^_]*_?(inverted)?$'
        if re.match(pattern_imn, imn) is not None:
            sp = re.match(pattern_imn, imn).group(4)
            ct = re.match(pattern_imn, imn).group(5)
            di = re.match(pattern_imn, imn).group(6)
            nm = re.match(pattern_imn, imn).group(7)
            iv = re.match(pattern_imn, imn).group(8) is not None
            match sp:
                case 'Human':
                    if ct == 'Head':
                        tmp_cond = bytes('fh{:02}'.format(nm), 'ascii')
                        tmp_cat = b'face_hum'
                        tmp_id = bytes('Hum{:02}'.format(nm), 'ascii')
                        tmp_view = 0
                        tmp_roll = 0
                case 'MacaqueRhesus':
                    if ct == 'Head':
                        tmp_cond = bytes('fr{:02}'.format(nm), 'ascii')
                        tmp_cat = b'face_rhe'
                        tmp_id = bytes('Rhe{:02}'.format(nm), 'ascii')
                        tmp_view = 0
                        tmp_roll = 0
                case 'Marm':
                    if ct == 'Head':
                        tmp_cond = bytes('fm{}{:02}'.format(di[0:3], nm), 'ascii')
                        tmp_cat = b'face_mrm'
                        tmp_id = bytes(di[0:8], 'ascii')
                        if iv is True:
                            nm = '9'
                        match nm:
                            case '1':
                                tmp_view = 0
                                tmp_roll = 0
                            case '2':
                                tmp_view = 180
                                tmp_roll = 0
                            case '3':
                                tmp_view = 0
                                tmp_roll = -45
                            case '4':
                                tmp_view = 0
                                tmp_roll = 45
                            case '5':
                                tmp_view = -90
                                tmp_roll = 0
                            case '6':
                                tmp_view = -45
                                tmp_roll = 0
                            case '7':
                                tmp_view = 45
                                tmp_roll = 0
                            case '8':
                                tmp_view = 90
                                tmp_roll = 0
                            case '9':
                                tmp_view = 0
                                tmp_roll = 180
                            case _:
                                warn('Could not recognize view or roll of head image from filename.')
                                tmp_view = None
                                tmp_roll = None
                    if ct == 'Body':
                        tmp_cond = bytes('bm{}{:02}'.format(ct[0:3], nm), 'ascii')
                        tmp_cat = b'body_mrm'
                        tmp_id = bytes(di[0:8], 'ascii')
                case 'Objects':
                    pattern_ct = r'^([^0-9]+)([0-9])$'
                    if re.match(pattern_ct, ct) is not None:
                        ct_p1 = re.match(pattern_ct, ct).group(1)
                        ct_p2 = re.match(pattern_ct, ct).group(2)
                        ct = ct_p1
                        nm = ct_p2 + nm
                    if 'Manmade' in ct:
                        tmp_cond = bytes('om{:02}'.format(nm), 'ascii')
                        tmp_cat = b'obj'
                    elif 'FruitVeg' in ct:
                        tmp_cond = bytes('vf{:02}'.format(nm), 'ascii')
                        tmp_cat = b'food'
                    elif 'MultipartGeon' in ct:
                        tmp_cond = bytes('og{:02}'.format(nm), 'ascii')
                        tmp_cat = b'obj'
                    elif 'Pairwise' in ct:
                        tmp_cond = bytes('op{:02}'.format(nm), 'ascii')
                        tmp_cat = b'obj'
                    elif 'String' in ct:
                        tmp_cond = bytes('os{:02}'.format(nm), 'ascii')
                        tmp_cat = b'obj'
                case _:
                    warn('Could not recognize type of image from filename.')
        elif imn == 'blank':
            tmp_cond = bytes('blank', 'ascii')
            tmp_cat = b'blank'
        elif 'Cartoon' in imn:
            pattern_ctn = r'^[^_]*Cartoon_([0-9]+)_?[^_]*_?(inverted)?$'
            if re.match(pattern_ctn, imn) is not None:
                nm = re.match(pattern_ctn, imn).group(1)
                tmp_cond = bytes('fcm{:02}'.format(nm), 'ascii')
                tmp_cat = b'face_ctn'
                tmp_id = bytes(nm, 'ascii')
                tmp_view = 0
                tmp_roll = 0
        else:
            warn('Could not recognize category or condition of image from filename.')
            tmp_cond = None
            tmp_cat = None
    elif image_set == 'Song_etal_Wang_2022_FOBonly':
        pattern_zerocheck = r'^([aobmufps]{1})([0-9]{1})$'
        if re.match(pattern_zerocheck, lf_conditions[c]) is not None:
            tg = re.match(pattern_zerocheck, lf_conditions[c]).group(1)
            ng = re.match(pattern_zerocheck, lf_conditions[c]).group(2)
            lfc = '{}{}'.format(tg, ng.zfill(2))
        else:
            lfc = lf_conditions[c]
        tmp_cond = bytes(lfc, 'ascii')
        match lf_conditions[c][0]:
            case 'a':
                tmp_cat = b'animal'
            case 'o':
                tmp_cat = b'obj'
            case 'b':
                if lf_conditions[c] == 'blank':
                    tmp_cat = b'blank'
                else:
                    tmp_cat = b'body_mrm'
            case 'm':
                tmp_cat = b'face_mrm'
                tmp_view = 0
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
    data[c]['view'] = tmp_view
    data[c]['roll'] = tmp_roll
    data[c]['imagename'] = tmp_imagename
    for t in range(n_trials):
        fr_start = fridx[c, t] - n_samp_isi
        fr_end = fridx[c, t] + n_samp_stim + n_samp_isi
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
            raise RuntimeError('Imaging was stopped before stimulus. ' +
                               'Handling this is not yet implemented.')
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
conditions = np.unique(data[:]['cond'])


# % Calculate tuning properties for each ROI (i.e. compute face selectivity index)
# ? ? ? and find preferred face(s)?
#
# # FSI = (meanR_faces – meanR_nonfaceobj) / (meanR_faces + meanR_nonfaceobj)
# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For
# |face-selectivity index| = 1/3, that is, if the response to faces was at
# least twice (or at most half) that of nonface objects, a cell was classed
# as being face selective [45–47].

idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)

# TODO: check for NaN values rather than using nanmean?
# if np.any(np.isnan(volume)):
#     raise Exception('NaNs found in preprocessed volume.')

# for reference...
# data[cond]['FdFF'][roi, trial, frame]
# data[cond]['FdFF_meant'][roi, frame]

# Non-zero hack
FdFF_absmin = -np.inf
for ct in np.unique(data['cat']):
    absmintmp = np.abs(np.min(np.nanmean(data[data['cat'] == ct]['FdFF_meant'][:, :, idx_stim], axis=(0, -1))))
    if absmintmp > FdFF_absmin:
        FdFF_absmin = absmintmp
Fzsc_absmin = -np.inf
for ct in np.unique(data['cat']):
    absmintmp = np.abs(np.min(np.nanmean(data[data['cat'] == ct]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1))))
    if absmintmp > Fzsc_absmin:
        Fzsc_absmin = absmintmp

FdFF_allfaces_meanRstimall = np.nanmean(data[(data['cat'] == b'face_mrm') & (data['view'] == 0) & (data['roll'] == 0)]['FdFF_meant'][:, :, idx_stim],
                                        axis=(0, -1)) + FdFF_absmin
FdFF_allobjs_meanRstimall = np.nanmean(data[data['cat'] == b'obj']['FdFF_meant'][:, :, idx_stim],
                                       axis=(0, -1)) + FdFF_absmin
FdFF_allbodies_meanRstimall = np.nanmean(data[data['cat'] == b'body_mrm']['FdFF_meant'][:, :, idx_stim],
                                       axis=(0, -1)) + FdFF_absmin


Fzsc_allfaces_meanRstimall = np.nanmean(data[data['cat'] == b'face_mrm']['Fzsc_meant'][:, :, idx_stim],
                                        axis=(0, -1)) + Fzsc_absmin
Fzsc_allobjs_meanRstimall = np.nanmean(data[data['cat'] == b'obj']['Fzsc_meant'][:, :, idx_stim],
                                       axis=(0, -1)) + Fzsc_absmin
Fzsc_allbodies_meanRstimall = np.nanmean(data[data['cat'] == b'body_mrm']['Fzsc_meant'][:, :, idx_stim],
                                       axis=(0, -1)) + Fzsc_absmin

# FdFFn_allfaces_meanRstimall = np.nanmean(data[data['cat'] == b'face_mrm']['FdFFn_meant'][:, :, idx_stim],
#                                          axis=(0, -1))
# FdFFn_allobjs_meanRstimall = np.nanmean(data[data['cat'] == b'obj']['FdFFn_meant'][:, :, idx_stim], axis=(0, -1))
# Fzscn_allfaces_meanRstimall = np.nanmean(data[data['cat'] == b'face_mrm']['Fzscn_meant'][:, :, idx_stim],
#                                          axis=(0, -1))
# Fzscn_allobjs_meanRstimall = np.nanmean(data[data['cat'] == b'obj']['Fzscn_meant'][:, :, idx_stim], axis=(0, -1))


# FSIs(_by_roi) = [roi, fsi]
FSIs_dFF = (FdFF_allfaces_meanRstimall - FdFF_allobjs_meanRstimall) / \
           (FdFF_allfaces_meanRstimall + FdFF_allobjs_meanRstimall)
FSIs_zsc = (Fzsc_allfaces_meanRstimall - Fzsc_allobjs_meanRstimall) / \
           (Fzsc_allfaces_meanRstimall + Fzsc_allobjs_meanRstimall)

# FSIs_wbody_dFF = (FdFF_allfaces_meanRstimall - FdFF_allobjs_meanRstimall - FdFF_allbodies_meanRstimall) / \
#                  (FdFF_allfaces_meanRstimall + FdFF_allobjs_meanRstimall + FdFF_allbodies_meanRstimall)
# FSIs_wbody_zsc = (Fzsc_allfaces_meanRstimall - Fzsc_allobjs_meanRstimall - Fzsc_allbodies_meanRstimall) / \
#                  (Fzsc_allfaces_meanRstimall + Fzsc_allobjs_meanRstimall + Fzsc_allbodies_meanRstimall)


# % Define ROIs as tuned or untuned using the FSI

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For
# |face-selectivity index| = 1/3, that is, if the response to faces was at
# least twice (or at most half) that of nonface objects, a cell was classed
# as being face selective45–47.

print('|FSI| threshold: {}'.format(fsi_tuning_thresh))
tunidx_fsi = FSIs_zsc
tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).squeeze()
n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


# TODO regorganize

# Plot FSI histogram
f0 = plt.figure()
plt.hist(FSIs_zsc, bins=100)
plt.xlabel('Face-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([-1, 1])
plt.axvline(fsi_tuning_thresh, color='m')
plt.axvline(-fsi_tuning_thresh, color='m')
f0.show()
if saving:
    save_name = savepfix_str + '_Histogram_FSIdFF_thresh' + \
                '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)

f0 = plt.figure()
plt.hist(FSIs_zsc, bins=100)
plt.xlabel('Face-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([-1, 1])
plt.axvline(fsi_tuning_thresh, color='m')
plt.axvline(-fsi_tuning_thresh, color='m')
f0.show()
if saving:
    save_name = savepfix_str + '_Histogram_FSIzsc_thresh' + \
                '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)

# Summarize responsiveness of each ROI

# Ensure that category order is FOB for consistent RGB color code
fob = np.array([b'face_mrm', b'obj', b'body_mrm'], dtype='|S8')
if all(x in categories for x in [b'face_mrm', b'obj', b'body_mrm']):
    if np.setdiff1d(categories, fob).size == 0:
        categories = fob
    else:
        categories = np.concatenate((fob, np.setdiff1d(categories, fob)))

# TODO improve variable naming here for clarity

ROIinfo = np.zeros(n_ROIs, dtype=[('top_cat', 'S8'),
                                  ('top_cond', 'S8'),
                                  ('top_cond_FdFF', 'f4'),
                                  ('top_cond_Fzsc', 'f4'),
                                  ('FSI_byFdFF', 'f4'),
                                  ('FSI_byFzsc', 'f4')
                                  ])

# ROIinfo = np.zeros(n_ROIs, dtype=[('cat_meanFdFF_meanTFstim', 'S8'),
#                                   ('cond_meanFzsc_meanTFstim', 'S8'),
#                                   ('cond_maxFdFF_meanTFstim', 'S8'),
#                                   ('cond_maxFzsc_meanTFstim', 'S8'),
#                                   ('cat_cond_maxFdFF_meanTFstim', 'S8'),
#                                   ('cat_cond_maxFzsc_meanTFstim', 'S8'),
#                                   ('val_cond_maxFdFF_meanTFstim', 'f4'),
#                                   ('val_cond_maxFzsc_meanTFstim', 'f4'),
#                                   ('FSI_byFdFF', 'f4'),
#                                   ('FSI_byFzsc', 'f4')
#                                   ])
ROIinfo[:]['top_cond_FdFF'] = np.nan
ROIinfo[:]['top_cond_Fzsc'] = np.nan
ROIinfo[:]['FSI_byFdFF'] = np.nan
ROIinfo[:]['FSI_byFzsc'] = np.nan

# ONLY CONSIDER FOB MARM IMAGES
cat_subset = fob
cond_subset = np.where((data['cat'] == b'face_mrm') | (data['cat'] == b'obj') | (data['cat'] == b'body_mrm'))[0]
cond_names = np.hstack([np.sort(data[data['cat'] == cat]['cond']) for cat in cat_subset])
cond_idx = np.array([np.where(data['cond'] == cond)[0][0] for cond in cond_names])

top_cond_byFdFF = np.argmax(np.mean(data[cond_idx]['FdFF_meant'][:, :, idx_stim], axis=-1), axis=0)
top_meantstim_FdFF = np.max(np.mean(data[cond_idx]['FdFF_meant'][:, :, idx_stim], axis=-1), axis=0)
top_cond_byFzsc = np.argmax(np.mean(data[cond_idx]['Fzsc_meant'][:, :, idx_stim], axis=-1), axis=0)
top_meantstim_Fzsc = np.max(np.mean(data[cond_idx]['Fzsc_meant'][:, :, idx_stim], axis=-1), axis=0)
# top_cond_byFdFF = np.argmax(np.mean(data[:]['FdFF_meant'][:, :, idx_stim], axis=-1), axis=0)
# top_meantstim_FdFF = np.max(np.mean(data[:]['FdFF_meant'][:, :, idx_stim], axis=-1), axis=0)
# top_cond_byFzsc = np.argmax(np.mean(data[:]['Fzsc_meant'][:, :, idx_stim], axis=-1), axis=0)
# top_meantstim_Fzsc = np.max(np.mean(data[:]['Fzsc_meant'][:, :, idx_stim], axis=-1), axis=0)

if np.any(top_cond_byFdFF != top_cond_byFzsc):
    print('top_cond mismatch from FdFF and Fzsc for ROIs: {}'.format(np.where(top_cond_byFdFF != top_cond_byFzsc)[0]))

for r in range(n_ROIs):
    # ROIinfo[r]['top_cat'] = data[top_cond_byFzsc[r]]['cat']
    # ROIinfo[r]['top_cond'] = data[top_cond_byFzsc[r]]['cond']
    ROIinfo[r]['top_cat'] = data[cond_idx[top_cond_byFzsc[r]]]['cat']
    ROIinfo[r]['top_cond'] = data[cond_idx[top_cond_byFzsc[r]]]['cond']
    ROIinfo[r]['top_cond_FdFF'] = top_meantstim_FdFF[r]
    ROIinfo[r]['top_cond_Fzsc'] = top_meantstim_Fzsc[r]
    ROIinfo[r]['FSI_byFdFF'] = FSIs_dFF[r]
    ROIinfo[r]['FSI_byFzsc'] = FSIs_zsc[r]

# Determine for each ROI which condition (image) elicited the largest response
# TODO improve variable naming here for clarity
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
# TODO THIS SHOULD NOT BE HARD CODED
at_sortidx = (-np.mean(data[cond_idx[0:19]]['Fzsc_meant'][:, :, idx_stim], axis=(0,-1))[above_threshold]).argsort()

# Plot heatmap of mean responses to all presented conditions (images) for ROIs 
# with at least one stimulus period z-score > 0.5
fhm = plt.figure()
plt.xlabel('Image')
plt.ylabel('ROI')
ax = plt.gca()
match image_set:
    case 'Song_etal_Wang_2022_FOBonly':
        if n_conds == 60:
            ax.set_xticks([19.5, 39.5])
            ax.set_xticklabels([None, None])
            ax.set_xticks([10, 30, 50], minor=True)
            ax.set_xticklabels(['faces', 'objects', 'bodies'], minor=True)
        if n_conds == 130 or n_conds == 131:
            ax.set_xticks([19.5, 59.5])
            ax.set_xticklabels([None, None])
            ax.set_xticks([10, 40, 70], minor=True)
            ax.set_xticklabels(['faces', 'objects', 'bodies'], minor=True)
# xticks = ax.xaxis.get_major_ticks()
ax.tick_params(which='minor', length=0)
# plt.imshow(np.mean(data[:]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[above_threshold], vmin=0.5-0.0001, vmax=0.5+0.0001, aspect='auto', cmap='gray', interpolation='none')
plt.imshow(np.mean(data[cond_idx]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[above_threshold[at_sortidx]], 
           vmin=-1.0, vmax=1.0, 
           aspect='auto', cmap='bwr', interpolation='none')
# ax = plt.gca()
# ax.axvline(x=20)
cbar = plt.colorbar()
# cbar.ax.set_yticks(['0','1','2','>3'])
# cbar.ax.set_yticklabels(['0','1','2','>3'])
cbar.set_label('mean Zscore across stimulus period')
plt.show()
if saving:
    fhm.savefig(os.path.join(save_path, savepfix_str + '_Heatmap_byCondition_sortMeanFace_threshZgt0p5.png'),
                dpi=dpi, transparent=True)

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
# top_cat_id = [np.argwhere(categories == ROIinfo[rat]['top_cat'])[0][0] for rat in above_threshold]
top_cat_id = [np.argwhere(categories == ROIinfo[r]['top_cat'])[0][0] for r in range(n_ROIs)]
# top_cat_idn = np.divide(top_cat_id, len(categories))
top_cat_idn = np.divide(top_cat_id, len(cat_subset))


# Plot for each ROI the category of the condition (image) that elicited the largest response
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
plots.plot_roi_overlays(ROIs[above_threshold], 
                        ROI_colors[above_threshold], 
                        image=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        save_path=save_path,
                        imn=savepfix_str + '_CategoryOfMostActivatingCondition_threshZgt0p5')

# Plot for each ROI the category of the condition (image) that elicited the largest response  
above_threshold = np.where(np.abs(FSIs_zsc) > fsi_tuning_thresh)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
plots.plot_roi_overlays(ROIs[above_threshold], 
                        ROI_colors[above_threshold], 
                        image=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        save_path=save_path,
                        imn=savepfix_str + '_CategoryOfMostActivatingCondition_threshFSIgtlt0p33')

# Determine for each ROI which category elicited the largest average response
# TODO improve variable naming here for clarity

# above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]

mean_by_cat = np.array([np.nanmean(data[data['cat'] == cat]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1)) for cat in cat_subset]).swapaxes(0, 1)
top_cat_mean = categories[np.argmax(mean_by_cat, axis=-1)]

# top_cat_mean_id = [np.argwhere(categories == top_cat_mean[r])[0][0] for r in range(n_ROIs)]
# top_cat_mean_idn = np.divide(top_cat_mean_id, len(categories))
# TODO check if fob here can be cat_subset
top_cat_mean_id = [np.argwhere(fob == top_cat_mean[r])[0][0] for r in range(n_ROIs)]
top_cat_mean_idn = np.divide(top_cat_mean_id, len(cat_subset))


# Plot heatmap of mean responses to all presented conditions (images) for ROIs 
# with at least one stimulus period z-score > 0.5
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
fhm = plt.figure()
plt.xlabel('Image Category')
plt.ylabel('ROI')
ax = plt.gca()
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['faces','objects','bodies'])
# plt.imshow(mean_by_cat[above_threshold], vmin=0.1-0.0001, vmax=0.1+0.0001, aspect='auto', cmap='bwr', interpolation='none')
plt.imshow(mean_by_cat[above_threshold[at_sortidx]], 
           vmin=-0.5, vmax=0.5, aspect='auto', cmap='bwr', interpolation='none')
cbar = plt.colorbar()
# cbar.ax.set_yticks(['0','1','2','>3'])
# cbar.ax.set_yticklabels(['0','1','2','>3'])
cbar.set_label('mean Zscore across stim period and images')
plt.show()
if saving:
    fhm.savefig(os.path.join(save_path, savepfix_str + '_Heatmap_byCategory_sortMeanFace_threshZgt0p5.png'), 
                dpi=dpi, transparent=True)


# Plot for each ROI the category elicited the largest average response
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
plots.plot_roi_overlays(ROIs[above_threshold], 
                        ROI_colors[above_threshold], 
                        image=plots.auto_level_s2p_image(fov_image))  #,
                        # save_path=save_path,
                        # imn=savepfix_str + '_CategoryOfMostActivatingCategoryOnAverage_threshZgt0p5')

# Plot for each ROI the category elicited the largest average response
# for only ROIs with FSI > threshold
above_threshold = np.where(np.abs(FSIs_zsc) > fsi_tuning_thresh)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
plots.plot_roi_overlays(ROIs[above_threshold], 
                        ROI_colors[above_threshold], 
                        image=plots.auto_level_s2p_image(fov_image),  #,
                        save_path=save_path,
                        imn=savepfix_str + '_CategoryOfMostActivatingCategoryOnAverage_threshFSIgtlt0p33')

# Plot relative response strength

# TODO make this more dynamic
Fzsc_fob = np.array([Fzsc_allfaces_meanRstimall,
                     Fzsc_allobjs_meanRstimall,
                     Fzsc_allbodies_meanRstimall]).swapaxes(0, 1)

# Subtract the response to the least-tuned category to make it relative
# (otherwise, an ROI that responds to all categories would show up as white)
Fzsc_mean_min_cat = np.min(Fzsc_fob, axis=1)
for col_i in range(3):
    Fzsc_fob[:, col_i] = Fzsc_fob[:, col_i] - Fzsc_mean_min_cat

max_Fzsc = 0.5
Fzsc_fob_norm = Fzsc_fob / max_Fzsc
Fzsc_fob_norm[Fzsc_fob_norm > 1] = 1


above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
plots.plot_roi_overlays(ROIs[above_threshold], 
                        Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        save_path=save_path,
                        imn=savepfix_str + '_RelativeResponseStrength_threshZgt0p5')


above_threshold = np.where(FSIs_zsc > fsi_tuning_thresh)[0]
plots.plot_roi_overlays(ROIs[above_threshold],
                        Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        save_path=save_path,
                        imn=savepfix_str + '_RelativeResponseStrength_threshFSIgtlt0p33')







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
    fig.savefig(os.path.join(save_path, savepfix_str + '_StimulusImages.png'), 
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


# above_threshold = np.where(FSIs_zsc > fsi_tuning_thresh)[0]

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

# TODO *** Could also calculate d’
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

# print('|FSI| threshold: {}'.format(fsi_tuning_thresh))
# tunidx_fsi = FSIs_zsc
# tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
# ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).squeeze()
# n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).shape[0]
# pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
# print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
# print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

# tuning_index_cond = tunidx_fsi
# tuning_index_cat = tunidx_fsi


# %%
# plt.figure()
# plt.hist(tuning_index_cond, bins=1000)
# plt.xlabel('Tuning index (cond/perimage): {}'.format(tuning))
# plt.ylabel('Neurons')
#
# tuning_index_cond_tuned_neurons = tuning_index_cond[tuning_index_cond > tuning_index_thresh]
# FdFF_by_cond_tuned = FdFF_by_cond[tuning_index_cond > tuning_index_thresh]
# if plot_least_tuned_neurons_first:
#     FdFF_by_cond_tuned = FdFF_by_cond_tuned[(+tuning_index_cond_tuned_neurons).argsort()]
# else:
#     FdFF_by_cond_tuned = FdFF_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]
# Fzsc_by_cond_tuned = Fzsc_by_cond[tuning_index_cond > tuning_index_thresh]
# if plot_least_tuned_neurons_first:
#     Fzsc_by_cond_tuned = Fzsc_by_cond_tuned[(+tuning_index_cond_tuned_neurons).argsort()]
# else:
#     Fzsc_by_cond_tuned = Fzsc_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]

# plt.figure()
# plt.hist(tuning_index_cat, bins=1000)
# plt.xlabel('Tuning index (category): {}'.format(tuning))
# plt.ylabel('Neurons')
#
# tuning_index_cat_tuned_neurons = tuning_index_cat[tuning_index_cat > tuning_index_thresh]
# FdFF_by_cat_tuned = FdFF_by_cat[tuning_index_cat > tuning_index_thresh]
# if plot_least_tuned_neurons_first:
#     FdFF_by_cat_tuned = FdFF_by_cat_tuned[(+tuning_index_cat_tuned_neurons).argsort()]
# else:
#     FdFF_by_cat_tuned = FdFF_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
# Fzsc_by_cat_tuned = Fzsc_by_cat[tuning_index_cat > tuning_index_thresh]
# if plot_least_tuned_neurons_first:
#     Fzsc_by_cat_tuned = Fzsc_by_cat_tuned[(+tuning_index_cat_tuned_neurons).argsort()]
# else:
#     Fzsc_by_cat_tuned = Fzsc_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
#
# if normalize == 'dF/F':
#     Frois_by_cond = FdFF_by_cond
#     Frois_by_cat = FdFF_by_cat
# elif normalize == 'Zscore':
#     Frois_by_cond = Fzsc_by_cond
#     Frois_by_cat = Fzsc_by_cat

# redundant with cat
# plt.figure()
# plt.hist(tuning_index_cond, bins=1000)
# plt.xlabel('FSI (per image)')
# plt.ylabel('ROIs')
# plt.xlim([-1, 1])
#
# tuning_index_cond_tuned_neurons = tuning_index_cond[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# #Frois_by_cond_tuned = Frois_by_cond[tuning_index_cond > tuning_index_thresh]
# FdFF_by_cond_tuned = FdFF_by_cond[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# Fzsc_by_cond_tuned = Fzsc_by_cond[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# #Frois_by_cond_tuned = Frois_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]
# FdFF_by_cond_tuned = FdFF_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]
# Fzsc_by_cond_tuned = Fzsc_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]

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
# # Fzsc_allfaces_meanRstim = np.nanmean(data[(data['cat'] == b'face_mrm') & (data['view'] == 0) & (data['roll'] == 0)]['Fzsc_meant'][:, :, idx_trial],
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
# # ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).squeeze()
# # n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).shape[0]
# # pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
# # print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
# # print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

# maxim = np.unravel_index(ImSIs_zsc[:,:].argmax(), ImSIs_zsc.shape)[1]

# for it in range(20):
#     ImSIs_zsc_argsort_imt = ImSIs_zsc[:, it].argsort()
#     rois_sel = np.argwhere(np.abs(ImSIs_zsc[ImSIs_zsc_argsort_imt, it]) > fsi_tuning_thresh).squeeze()
    
#     plots.plot_roi_overlays(ROIs[above_threshold[rois_sel]], 
#                             Fzsc_for_plot_discrete[above_threshold[rois_sel]],
#                             image=plots.auto_level_s2p_image(fov_image)) 
#                             # save_path=save_path, 
#                             # imn=data['cond'][(data['cat'] == b'face_mrm')][it].decode())


# %% Compatibility with old variable names
# tuning_index_cat_tuned_neurons = tuning_index_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# #Frois_by_cat_tuned = Frois_by_cat[tuning_index_cat > tuning_index_thresh]
# FdFF_by_cat_tuned = FdFF_by_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# Fzsc_by_cat_tuned = Fzsc_by_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# #Frois_by_cat_tuned = Frois_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
# FdFF_by_cat_tuned = FdFF_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
# Fzsc_by_cat_tuned = Fzsc_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]


# %% Plot tuning map

# import time

# for c in range(0, 11):
#     c = 0.1 * c
#     plots.plot_map(ROIs, c * np.ones(FSIs_zsc.shape), FSIs_zsc, tuning_thresh=fsi_tuning_thresh,
#              size=fov_size, image=fov_image, circular=False, save_path=save_path)
#     time.sleep(1.2)

# %%

# for r in range(Frois_by_cat_tuned.shape[0]):
#     print(r)
#     plt.pause(0.05)
#     plt.subplots(1, 3, constrained_layout=True)
#     #if plot_rando_neurons == True:
#     #    r = np.random.randint(Frois_by_cat_tuned.shape[0])
#     for c in range(n_cats):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, n_samp_isi:n_samp_isi+n_samp_stim]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:n_samp_isi]):
#         #     continue
#         plt.subplot(1, 3, c+1)
#         plt.title('Stim: ' + str(categories[c]), fontsize=7)
#         plt.axvspan(n_samp_isi, (n_samp_isi + n_samp_stim), color='0.9')
#         plt.ylim((-1,5))
#         plt.xlabel('Frame # (@'+str(md['framerate'])+'Hz)', fontsize=6)
#         plt.ylabel(normalize, fontsize=6)
#         plt.tick_params(axis='both', which='major', labelsize=5)
#         for t in range(conds_per_cat * n_trials):
#             plt.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
#         plt.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0))
#         plt.suptitle('roi {} '.format(r), fontsize=10)
#         #plt.waitforbuttonpress()
#         print(np.std(np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0)))

# for r in range(Frois_by_cat_tuned.shape[0]):
for r in range(n_ROIs_tuned):
    ridx = tunidx_fsi_argsrt[r]
    print(r)
    ipd = 1 / dpi
    fig = plt.figure(figsize=(2 * 300 * ipd, 3 * 300 * ipd))
    fig.clf()
    fig.suptitle('roi {} '.format(ridx), fontsize=12)
    axes = fig.subplots(nrows=2, ncols=3)
    for c in range(n_cats):
        # plt.subplot(2, 3, c+1)
        ax = axes[0, c]
        ax.set_title(categories[cats[c]], fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
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
        # ax.set_ylim((-3,10))
        ax.set_ylim((np.min(Fzsc_by_cat[ridx, :, :, :]) - 0.2,
                     np.max(Fzsc_by_cat[ridx, :, :, :]) + 0.2))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), Fzsc_by_cat[ridx, c, t, :], color=str(0.4 + 0.4 * t / Fzsc_by_cat.shape[2]))
        ax.plot(range(totalframes), np.mean(Fzsc_by_cat[ridx, c, :, :], axis=0), color='tab:green')
    for c in range(n_cats):
        ax = axes[1, c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
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
        ax.set_xlim((0, 4 * md['framerate']))
        # ax.set_ylim((-1,2))
        ax.set_ylim((np.min(FdFF_by_cat[ridx, :, :, :]) - 0.1,
                     np.max(FdFF_by_cat[ridx, :, :, :]) + 0.1))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), FdFF_by_cat[ridx, c, t, :], color=str((0.4) + 0.4 * t / FdFF_by_cat.shape[2]))
        ax.plot(range(totalframes), np.mean(FdFF_by_cat[ridx, c, :, :], axis=0), color='tab:blue')
    plt.show()
    plt.pause(0.05)

# %%
for r in range(Frois_by_cat_tuned.shape[0]):
    print(r)
    ipd = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(2 * 300 * ipd, 4 * 300 * ipd))
    fig.clf()
    fig.suptitle('roi {} '.format(r), fontsize=12)
    axes = fig.subplots(nrows=4, ncols=3)
    for c in range(n_cats):
        ax = axes[0, c]
        ax.set_title(str(cats[c]), fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
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
        ax.set_ylim((-3, 10))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), Frois_by_cat_tuned[r, c, t, :],
                    color=str((0.4) + 0.4 * t / Frois_by_cat_tuned.shape[2]))
        ax.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r, c, :, :], axis=0), color='tab:green')
    for c in range(n_cats):
        ax = axes[1, c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
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
        ax.set_xlim((0, 4 * md['framerate']))
        ax.set_ylim((-1, 2))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), FdFF_by_cat_tuned[r, c, t, :],
                    color=str((0.4) + 0.4 * t / Frois_by_cat_tuned.shape[2]))
        ax.plot(range(totalframes), np.mean(FdFF_by_cat_tuned[r, c, :, :], axis=0), color='tab:blue')
    for c in range(n_cats):
        ax = axes[2, c]
        ax.set_title(str(cats[c]), fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
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
        ax.set_ylim((-0.1, 0.5))
        # for t in range(conds_per_cat * n_trials):
        #    ax.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        Fmean = np.mean(FdFF_by_cat_tuned[r, c, :, :], axis=0)
        Fsem = np.std(FdFF_by_cat_tuned[r, c, :, :], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
        ax.plot(range(totalframes), Fmean, color='tab:green')
        ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:green', alpha=0.25)
    for c in range(n_cats):
        ax = axes[3, c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
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
        ax.set_xlim((0, 4 * md['framerate']))
        ax.set_ylim((-0.1, 0.5))
        # for t in range(conds_per_cat * n_trials):
        #    ax.plot(range(totalframes), FdFF_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        Fmean = np.mean(FdFF_by_cat_tuned[r, c, :, :], axis=0)
        Fsem = np.std(FdFF_by_cat_tuned[r, c, :, :], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
        ax.plot(range(totalframes), Fmean, color='tab:blue')
        ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:blue', alpha=0.25)
    plt.show()
    plt.pause(0.05)
