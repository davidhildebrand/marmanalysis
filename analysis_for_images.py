#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import oasis
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

# TODO: add consistent pyplot theme handling for plots https://github.com/raybuhr/pyplot-themes
# including colorblind palette options https://personal.sron.nl/~pault/
# TODO: exclude suite2p badframes
# TODO: exclude based on eye tracking, at least when eyes are not open
# TODO: plot neuron traces over conds or cats


# % Settings

# FSI threshold based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci (https://doi.org/10.1038/nn.2363):
# [...] neurons (94%) were face selective (that is, face-selectivity index
# larger than 1/3 or smaller than -1/3, dotted lines).
threshold_fsi = 1 / 3
# dprime_F threshold based on Shi et al Tsao bioRxiv (Fig 1g, https://doi.org/10.1101/2023.12.06.570341):
# "The dotted vertical line marks d’ = 0.2, which we used as our threshold for identifying face-selective units."
threshold_dprime = 0.2

threshold_cellprob = 0.0

plt.rcParams['figure.dpi'] = 600
dpi = plt.rcParams['figure.dpi']


# Remove stale metadata
if 'md' in locals():
    md = dict()
    del md

# % Specify data locations

# --  GOOD OLD Cadbury PD  20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx
# |FSI| threshold: 0.25
# Tuned ROIs: 894. Total ROIs: 6020.   (note: using threshold_cellprob old = 0.0)
# Percentage of tuned ROIs: 14.85%
animal_str = 'Cadbury'
# date_str = '20221016d_olds2p'
date_str = '20221016d'
session_str = '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'
md = dict()
md['framerate'] = 6.364
md['fov'] = dict()
md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
md['fov']['w_px'] = 730
md['fov']['h_px'] = 730
# suite2p_str = 'suite2p_old*'
suite2p_str = 'suite2p_cellpose2_d14px_pt-3p5_ft1p5*'
stimset_str = 'Song_etal_Wang_2022_NatCommun|480288_equalized_RGBA_FOBonly'.replace('|', os.path.sep)

# # -- Cadbury UNCLEAR (fluid, z drift across time)
# animal_str = 'Cadbury'
# date_str = '20230510d'
# session_str = '155713tUTC_SP_depth200um_fov2190x2000um_res3p00x3p02umpx_fr06p993Hz_pow059p9mW_stimImagesSong230509dSel'


# # -- Cadbury OBJ ()
# animal_str = 'Cadbury'
# date_str = '20230809d'
# session_str = '173936tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'
# stimimage_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets\FOBmin\Images\20230728d'


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

mdfile_str = '*_metadata.pickle'
datafile_str = '*_00001.tif'
logfile_str = '*.log'
logfile_re = r'.*^((?!disptimes).)*$'  # Exclude log files whose names contain 'disptimes'
stimlogfile_str = '*_stimlog.csv'
eyefile_str = '*_AIdata.p'
eyecalib_dir_str = '*_EyeTrackingCalibration'
eyecalib_logfile_str = '*_EyeTrackingCalibration.log'
eyecalib_datafile_str = '*_EyeTrackingCalibration_AIdata.p'
if 'suite2p_str' not in locals():
    suite2p_str = 'suite2p*'
suite2p_plane_str = 'plane0'

# Load imaging session information
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

stimimage_path = os.path.join(stim_path, stimset_str)
if not os.path.isdir(stimimage_path):
    warn('Could not find stimulus image source path.')
    stimimage_path = None

date_path = os.path.join(base_path, animal_str, date_str)
session_path = os.path.join(base_path, animal_str, date_str, session_str)
mdfile_list = glob(os.path.join(session_path, mdfile_str))
datafile_list = [f for f in glob(os.path.join(session_path, datafile_str)) if os.path.isfile(f)]
logfile_list = [f for f in glob(os.path.join(session_path, logfile_str))
                if re.search(logfile_re, f) and os.path.isfile(f)]
stimlogfile_list = [f for f in glob(os.path.join(session_path, stimlogfile_str)) if os.path.isfile(f)]
eyefile_list = [f for f in glob(os.path.join(session_path, eyefile_str)) if os.path.isfile(f)]
eyecalib_dir_list = [d for d in glob(os.path.join(date_path, eyecalib_dir_str)) if os.path.isdir(d)]
if len(eyecalib_dir_list) > 0:
    ecd_path = eyecalib_dir_list[0]
    if len(eyecalib_dir_list) > 1:
        warn('Found multiple eye tracking calibration directories, using the first one: {}'.format(ecd_path))
    eyecalib_logfile_list = [f for f in glob(os.path.join(ecd_path, eyecalib_logfile_str))
                             if os.path.isfile(f)]
    eyecalib_datafile_list = [f for f in glob(os.path.join(ecd_path, eyecalib_datafile_str))
                              if os.path.isfile(f)]
else:
    ecd_path = None
    eyecalib_logfile_list = []
    eyecalib_datafile_list = []

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
    log = None

if len(stimlogfile_list) > 0:
    slf_path = stimlogfile_list[0]
    if len(stimlogfile_list) > 1:
        warn('Found multiple stimlog files, using the first one: {}'.format(slf_path))
    if os.path.isfile(slf_path):
        stimlog = pd.read_csv(slf_path)
    else:
        stimlog = None

if len(eyefile_list) > 0:
    etf_path = eyefile_list[0]
    if len(eyefile_list) > 1:
        warn('Found multiple log files, using the first one: {}'.format(etf_path))
    if os.path.isfile(etf_path):
        with open(etf_path, 'rb') as etf:
            et = pickle.load(etf)
    else:
        raise RuntimeError('Could not find eye tracking data file.')
else:
    etf_path = None

if len(eyecalib_logfile_list) > 0:
    ec_lf_path = eyecalib_logfile_list[0]
    if len(eyecalib_logfile_list) > 1:
        warn('Found multiple eye tracking calibration log files, using the first one: {}'.format(ec_lf_path))
    if os.path.isfile(ec_lf_path):
        eclf = open(ec_lf_path, 'r')
        eclog = eclf.read()
        eclf.close()
    else:
        raise RuntimeError('Could not find eye tracking calibration log file.')
else:
    ec_lf_path = None
    eclf = None
    eclog = None

if len(eyecalib_datafile_list) > 0:
    ec_df_path = eyecalib_datafile_list[0]
    if len(eyecalib_datafile_list) > 1:
        warn('Found multiple eye tracking calibration data files, using the first one: {}'.format(ec_df_path))
    if os.path.isfile(ec_df_path):
        with open(ec_df_path, 'rb') as ec_df:
            ecdf = pickle.load(ec_df)
    else:
        raise RuntimeError('Could not find eye tracking calibration data file.')
else:
    etf_path = None
    ecdf = None

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
s2p_badframes = np.where(s2p_ops['badframes'])[0]

cellinds = np.where(s2p_iscell[:, 1] >= threshold_cellprob)[0]
# cellinds = np.where(s2p_iscell[:,0] == 1.0)[0]
# cellinds = np.logical_and(s2p_iscell[:, 1] >= threshold_cellprob, np.std(s2p_F, axis=1) != 0)
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
n_ROIs, n_frames = Frois.shape


# # Inspect fluorescence baseline filters
# filters.plot_example_baselines(Frois, rois=2, frames=1000, framerate=md['framerate'], window=60, include_mpfi=False,
#                                percentile=10, rank=10, sigma=10)

# Calculate baseline fluorescence (F0)
F0_filt_win_sec = 60  # sec
F0_filt_win_frames = round(F0_filt_win_sec * md['framerate'])  # frames
F0 = filters.calculate_baselines(Frois, framerate=md['framerate'], window=F0_filt_win_sec, method='meanbw')

# Compute dF/F and z-scored dF/F
FdFF_raw = (Frois - F0) / F0
Fzsc_raw = (Frois - F0 - np.mean(Frois - F0, axis=1)[:, np.newaxis]) / np.std(Frois - F0, axis=1)[:, np.newaxis]


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

# Friedrich et al 2017 Paninski https://doi.org/10.1371/journal.pcbi.1005423
# "An AR(1) process models the calcium response to a spike as an instantaneous increase followed by an exponential
# decay. This is a good description when the fluorescence rise time constant is small compared to the length of a
# time-bin, e.g. when using GCaMP6f [36] with a slow imaging rate. For fast imaging rates and slow indicators such
# as GCaMP6s it is more accurate to explicitly model the finite rise time. Typically we choose an AR(2) process,
# though more structured responses (e.g. multiple decay time constants) can also be modeled with higher values for
# the order p."

n_plot_ROIs = 1
n_samp_inspect = 1000
plot_ROIs = np.random.choice(n_ROIs, n_plot_ROIs)
frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
frame_end = frame_start + n_samp_inspect

fig = plt.figure()
# fig.suptitle('mean response by condition (each trial plotted)', fontsize=8)
axes = fig.subplots(nrows=n_plot_ROIs, ncols=1)
for r in range(n_plot_ROIs):
    ridx = plot_ROIs[r]
    # frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
    # frame_end = frame_start + n_samp_inspect
    Fr = Frois[ridx, frame_start:frame_end]

    # Fr = Frois[ridx, frame_start:frame_end]
    Fr_dFF = (Fr - np.mean(Fr)) / np.mean(Fr)
    # F0_rnk = filters.rank_order_filter(Fr, p=filter_percentile, n=round(filter_window * fr))
    # Fr_dFF_rnk = (Fr - F0_rnk) / F0_rnk
    # F0_pct = filters.percentile_filter_1d(Fr, p=filter_percentile, n=round(filter_window * fr))
    # Fr_dFF_pct = (Fr - F0_pct) / F0_pct
    # F0_rnkbw = filters.butterworth_filter(F0_rnk, fs=fr, p=filter_percentile)
    # Fr_dFF_rnkbw = (Fr - F0_rnkbw) / F0_rnkbw
    # F0_pctbw = filters.butterworth_filter(Fr, fs=fr, p=filter_percentile)
    # Fr_dFF_pctbw = (Fr - F0_pctbw) / F0_pctbw
    # # F0_med = np.median(np.lib.stride_tricks.sliding_window_view(Fr, (round(filter_window * fr),)), axis=1)
    # # Fr_dFF_med = (Fr - F0_med) / F0_med
    # F0_ma = np.convolve(Fr, np.ones(round(filter_window * fr)), mode='same') / round(filter_window * fr)
    # Fr_dFF_ma = Fr_dFF - np.convolve(Fr_dFF, np.ones(round(filter_window * fr)), mode='same') / round(
    #     filter_window * fr)

    oasisL0_c, oasisL0_s, oasisL0_b, oasisL0_g, oasisL0_lam = oasis.functions.deconvolve(Fr, penalty=0)
    Fr_oasisL0 = oasisL0_c + oasisL0_b
    Fr_dFF_oasisL0 = (Fr_oasisL0 - oasisL0_b) / oasisL0_b

    oasisL1_c, oasisL1_s, oasisL1_b, oasisL1_g, oasisL1_lam = oasis.functions.deconvolve(Fr, penalty=1)
    Fr_oasisL1 = oasisL1_c + oasisL1_b
    Fr_dFF_oasisL1 = (Fr_oasisL1 - oasisL1_b) / oasisL1_b

    # c, _ = oasisAR1(Fr, g=.95, s_min=.55)
    # oasisAR1_c, oasisAR1_s, oasisAR1_b, oasisAR1_g, oasisAR1_lam = deconvolve(Fr, penalty=1)

    ymin = np.min(Fr_dFF)
    ymax = np.max(Fr_dFF)
    xs = range(n_samp_inspect)

    if n_plot_ROIs > 1:
        ax = axes[r]
    else:
        ax = axes
    ax.set_ylabel('dF/F', fontsize=6)
    ax.set_xlabel('Frames', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
    ax.set_xticks([0, n_samp_inspect])
    ax.set_xticklabels([frame_start, frame_end])
    ax.plot(xs, Fr_dFF, label='FdFF', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_rnk, label='FdFF_rnk', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_pct, label='FdFF_pct', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_rnkbw, label='FdFF_rnkbw', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_pctbw, label='FdFF_pctbw', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_ma, label='Fr_dFF_med', linewidth=0.5, alpha=0.5, zorder=3)
    # ax.plot(xs, Fr_dFF_ma, label='FdFF_ma', linewidth=0.5, alpha=0.5, zorder=3)
    ax.plot(xs, Fr_dFF_oasisL0, label='FdFF_oasisL0', color='g', linewidth=1, alpha=0.8, zorder=10)
    ax.plot(xs, Fr_dFF_oasisL1, label='FdFF_oasisL1', color='b', linewidth=1, alpha=0.8, zorder=10)

    ax.legend(fontsize=4, ncol=len(ax.get_lines()), frameon=False, loc=(.02, .85))
plt.show()

# from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
# from oasis.plotting import simpleaxis
# from oasis.oasis_methods import oasisAR1, oasisAR2

# c, s, b, g, lam = deconvolve(Fr)  # , penalty=1)



# % Look at eye tracking data
# Take a look at this for density plotting:
# https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
f = plt.figure()
etx, ety = np.transpose(et[:, :2])
plt.scatter(et[:, 0], et[:, 1], s=1)
plt.show()
[np.std(et[:, d]) for d in range(0, et.shape[1])]

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


# % Extract eye tracking calibration information from log file

if eclog is not None:
    eclines = eclog.splitlines()

#
# 234.5829 	DATA 	Keypress: right
# 234.7007 	EXP 	oculomatic zeroing, hiding face, AI_data.shape = (133928, 6)
# 234.7010 	EXP 	oculomatic zeroing end, AI_data.shape = (133928, 6)
# 234.8201 	EXP 	coarse eye-tracking calibration start, AI_data.shape = (134078, 6)
# 234.8261 	EXP 	coarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (134078, 6)
# 243.3137 	DATA 	Keypress: left
# 243.3274 	EXP 	coarse trial 0, hiding face, AI_data.shape = (138942, 6)
# 243.8351 	DATA 	Keypress: right
# 243.8488 	EXP 	coarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (139240, 6)
# 247.9851 	DATA 	Keypress: right
# 247.9988 	EXP 	coarse trial 0, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.825335  -2.4251535] for face.pos = [ 5. -5.], AI_data.shape = (141609, 6)
# 248.4397 	DATA 	Keypress: right
# 248.4444 	EXP 	coarse trial 0, candidate values accepted,coarse_oculomatic_calib_values[0] = [ 0.82533503 -2.42515349], AI_data.shape = (141873, 6)
# 248.4576 	EXP 	coarse trial 1, showing face,face.pos = [-5.  0.], AI_data.shape = (141878, 6)
# 250.8841 	DATA 	Keypress: right
# 250.8974 	EXP 	coarse trial 1, hiding face, coarse_oculomatic_calib_values_candidate = [-2.6176107  -0.39334545] for face.pos = [-5.  0.], AI_data.shape = (143265, 6)
# 251.1865 	DATA 	Keypress: right
# 251.1903 	EXP 	coarse trial 1, candidate values accepted,coarse_oculomatic_calib_values[1] = [-2.61761069 -0.39334545], AI_data.shape = (143444, 6)
# 251.2032 	EXP 	coarse trial 2, showing face,face.pos = [ 5. -5.], AI_data.shape = (143449, 6)
# 251.8365 	DATA 	Keypress: right
# 251.8499 	EXP 	coarse trial 2, hiding face, coarse_oculomatic_calib_values_candidate = [2.3267987 2.5205238] for face.pos = [ 5. -5.], AI_data.shape = (143813, 6)
# 252.8094 	DATA 	Keypress: right
# 252.8099 	EXP 	coarse trial 2, candidate values accepted,coarse_oculomatic_calib_values[2] = [2.32679868 2.52052379], AI_data.shape = (144374, 6)
# 252.8232 	EXP 	coarse trial 3, showing face,face.pos = [5. 0.], AI_data.shape = (144379, 6)
# 253.7063 	DATA 	Keypress: right
# 253.7198 	EXP 	coarse trial 3, hiding face, coarse_oculomatic_calib_values_candidate = [ 1.0137813  -0.71914685] for face.pos = [5. 0.], AI_data.shape = (144888, 6)
# 253.9685 	DATA 	Keypress: right
# 253.9713 	EXP 	coarse trial 3, candidate values accepted,coarse_oculomatic_calib_values[3] = [ 1.01378131 -0.71914685], AI_data.shape = (145042, 6)
# 253.9843 	EXP 	coarse trial 4, showing face,face.pos = [ 0. -5.], AI_data.shape = (145047, 6)
# 254.6653 	DATA 	Keypress: right
# 254.6790 	EXP 	coarse trial 4, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.6928083 -1.7855618] for face.pos = [ 0. -5.], AI_data.shape = (145438, 6)
# 255.0501 	DATA 	Keypress: right
# 255.0549 	EXP 	coarse trial 4, candidate values accepted,coarse_oculomatic_calib_values[4] = [ 0.69280827 -1.7855618 ], AI_data.shape = (145665, 6)
# 255.0684 	EXP 	coarse trial 5, showing face,face.pos = [-5.  5.], AI_data.shape = (145669, 6)
# 255.6733 	DATA 	Keypress: right
# 255.6874 	EXP 	coarse trial 5, hiding face, coarse_oculomatic_calib_values_candidate = [-3.3262887  1.0924864] for face.pos = [-5.  5.], AI_data.shape = (146016, 6)
# 255.9472 	DATA 	Keypress: right
# 255.9522 	EXP 	coarse trial 5, candidate values accepted,coarse_oculomatic_calib_values[5] = [-3.3262887   1.09248638], AI_data.shape = (146179, 6)
# 255.9650 	EXP 	coarse trial 6, showing face,face.pos = [-5.  0.], AI_data.shape = (146184, 6)
# 256.6258 	DATA 	Keypress: right
# 256.6391 	EXP 	coarse trial 6, hiding face, coarse_oculomatic_calib_values_candidate = [-3.0205312 -0.5992691] for face.pos = [-5.  0.], AI_data.shape = (146564, 6)
# 257.4876 	DATA 	Keypress: left
# 257.5012 	EXP 	coarse trial 6, showing face,face.pos = [-5.  0.], AI_data.shape = (147067, 6)
# 259.8440 	DATA 	Keypress: right
# 259.8577 	EXP 	coarse trial 6, hiding face, coarse_oculomatic_calib_values_candidate = [-1.2768182 -3.5795538] for face.pos = [-5.  0.], AI_data.shape = (148413, 6)
# 260.9100 	DATA 	Keypress: left
# 260.9283 	EXP 	coarse trial 6, showing face,face.pos = [-5.  0.], AI_data.shape = (149033, 6)
# 262.9931 	DATA 	Keypress: right
# 263.0067 	EXP 	coarse trial 6, hiding face, coarse_oculomatic_calib_values_candidate = [-2.4761193 -0.6247902] for face.pos = [-5.  0.], AI_data.shape = (150216, 6)
# 263.3433 	DATA 	Keypress: right
# 263.3480 	EXP 	coarse trial 6, candidate values accepted,coarse_oculomatic_calib_values[6] = [-2.47611928 -0.62479019], AI_data.shape = (150423, 6)
# 263.3613 	EXP 	coarse trial 7, showing face,face.pos = [5. 0.], AI_data.shape = (150428, 6)
# 263.8549 	DATA 	Keypress: right
# 263.8686 	EXP 	coarse trial 7, hiding face, coarse_oculomatic_calib_values_candidate = [ 1.0083752 -0.6322563] for face.pos = [5. 0.], AI_data.shape = (150710, 6)
# 264.1260 	DATA 	Keypress: right
# 264.1266 	EXP 	coarse trial 7, candidate values accepted,coarse_oculomatic_calib_values[7] = [ 1.00837517 -0.63225633], AI_data.shape = (150868, 6)
# 264.1397 	EXP 	coarse trial 8, showing face,face.pos = [-5. -5.], AI_data.shape = (150873, 6)
# 266.1769 	DATA 	Keypress: right
# 266.1903 	EXP 	coarse trial 8, hiding face, coarse_oculomatic_calib_values_candidate = [-1.1659087 -1.782281 ] for face.pos = [-5. -5.], AI_data.shape = (152041, 6)
# 266.5713 	DATA 	Keypress: right
# 266.5740 	EXP 	coarse trial 8, candidate values accepted,coarse_oculomatic_calib_values[8] = [-1.16590869 -1.78228104], AI_data.shape = (152272, 6)
# 266.5868 	EXP 	coarse trial 9, showing face,face.pos = [-5.  0.], AI_data.shape = (152277, 6)
# 267.2959 	DATA 	Keypress: right
# 267.3096 	EXP 	coarse trial 9, hiding face, coarse_oculomatic_calib_values_candidate = [-2.3525956   0.22579882] for face.pos = [-5.  0.], AI_data.shape = (152684, 6)
# 267.7708 	DATA 	Keypress: right
# 267.7758 	EXP 	coarse trial 9, candidate values accepted,coarse_oculomatic_calib_values[9] = [-2.35259557  0.22579882], AI_data.shape = (152962, 6)
# 267.7892 	EXP 	coarse trial 10, showing face,face.pos = [-5.  5.], AI_data.shape = (152966, 6)
# 268.5196 	DATA 	Keypress: right
# 268.5330 	EXP 	coarse trial 10, hiding face, coarse_oculomatic_calib_values_candidate = [-1.3513215 -2.1745203] for face.pos = [-5.  5.], AI_data.shape = (153388, 6)
# 269.1728 	DATA 	Keypress: left
# 269.1864 	EXP 	coarse trial 10, showing face,face.pos = [-5.  5.], AI_data.shape = (153769, 6)
# 270.1392 	DATA 	Keypress: right
# 270.1526 	EXP 	coarse trial 10, hiding face, coarse_oculomatic_calib_values_candidate = [-2.1677294  0.8058794] for face.pos = [-5.  5.], AI_data.shape = (154315, 6)
# 270.5797 	DATA 	Keypress: right
# 270.5842 	EXP 	coarse trial 10, candidate values accepted,coarse_oculomatic_calib_values[10] = [-2.16772938  0.80587941], AI_data.shape = (154572, 6)
# 270.5976 	EXP 	coarse trial 11, showing face,face.pos = [-5. -5.], AI_data.shape = (154576, 6)
# 271.2165 	DATA 	Keypress: right
# 271.2300 	EXP 	coarse trial 11, hiding face, coarse_oculomatic_calib_values_candidate = [-2.7706397 -2.2990272] for face.pos = [-5. -5.], AI_data.shape = (154931, 6)
# 271.6612 	DATA 	Keypress: left
# 271.6748 	EXP 	coarse trial 11, showing face,face.pos = [-5. -5.], AI_data.shape = (155193, 6)
# 274.8519 	DATA 	Keypress: right
# 274.8656 	EXP 	coarse trial 11, hiding face, coarse_oculomatic_calib_values_candidate = [-1.466981  -1.6323916] for face.pos = [-5. -5.], AI_data.shape = (157017, 6)
# 275.2342 	DATA 	Keypress: right
# 275.2345 	EXP 	coarse trial 11, candidate values accepted,coarse_oculomatic_calib_values[11] = [-1.46698105 -1.63239157], AI_data.shape = (157241, 6)
# 275.2479 	EXP 	coarse trial 12, showing face,face.pos = [5. 5.], AI_data.shape = (157245, 6)
# 275.8946 	DATA 	Keypress: right
# 275.9083 	EXP 	coarse trial 12, hiding face, coarse_oculomatic_calib_values_candidate = [-3.4193316   0.31815547] for face.pos = [5. 5.], AI_data.shape = (157617, 6)
# 276.7429 	DATA 	Keypress: left
# 276.7564 	EXP 	coarse trial 12, showing face,face.pos = [5. 5.], AI_data.shape = (158113, 6)
# 277.4865 	DATA 	Keypress: right
# 277.5002 	EXP 	coarse trial 12, hiding face, coarse_oculomatic_calib_values_candidate = [1.0798211 1.2261193] for face.pos = [5. 5.], AI_data.shape = (158532, 6)
# 277.9037 	DATA 	Keypress: right
# 277.9041 	EXP 	coarse trial 12, candidate values accepted,coarse_oculomatic_calib_values[12] = [1.07982111 1.22611928], AI_data.shape = (158775, 6)
# 277.9172 	EXP 	coarse trial 13, showing face,face.pos = [-5. -5.], AI_data.shape = (158780, 6)
# 279.9263 	DATA 	Keypress: left
# 279.9400 	EXP 	coarse trial 13, hiding face, AI_data.shape = (159938, 6)
# 280.2254 	DATA 	Keypress: right
# 280.2392 	EXP 	coarse trial 13, showing face,face.pos = [-5. -5.], AI_data.shape = (160109, 6)
# 281.2680 	DATA 	Keypress: left
# 281.2816 	EXP 	coarse trial 13, hiding face, AI_data.shape = (160705, 6)
# 281.7411 	DATA 	Keypress: right
# 281.7544 	EXP 	coarse trial 13, showing face,face.pos = [-5. -5.], AI_data.shape = (160977, 6)
# 284.6671 	DATA 	Keypress: left
# 284.6808 	EXP 	coarse trial 13, hiding face, AI_data.shape = (162661, 6)
# 285.0396 	DATA 	Keypress: right
# 285.0563 	EXP 	coarse trial 13, showing face,face.pos = [-5. -5.], AI_data.shape = (162877, 6)
# 287.4692 	DATA 	Keypress: left
# 287.4823 	EXP 	coarse trial 13, hiding face, AI_data.shape = (164271, 6)
# 287.8879 	DATA 	Keypress: right
# 287.9062 	EXP 	coarse trial 13, showing face,face.pos = [-5. -5.], AI_data.shape = (164514, 6)
# 288.4833 	DATA 	Keypress: right
# 288.4971 	EXP 	coarse trial 13, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.9138386 -0.9331798] for face.pos = [-5. -5.], AI_data.shape = (164845, 6)
# 289.3412 	DATA 	Keypress: right
# 289.3458 	EXP 	coarse trial 13, candidate values accepted,coarse_oculomatic_calib_values[13] = [ 0.91383862 -0.9331798 ], AI_data.shape = (165344, 6)
# 289.3591 	EXP 	coarse trial 14, showing face,face.pos = [ 5. -5.], AI_data.shape = (165349, 6)
# 290.2560 	DATA 	Keypress: right
# 290.2696 	EXP 	coarse trial 14, hiding face, coarse_oculomatic_calib_values_candidate = [ 1.7659917 -1.6876354] for face.pos = [ 5. -5.], AI_data.shape = (165863, 6)
# 290.6175 	DATA 	Keypress: right
# 290.6179 	EXP 	coarse trial 14, candidate values accepted,coarse_oculomatic_calib_values[14] = [ 1.76599169 -1.68763542], AI_data.shape = (166075, 6)
# 290.6312 	EXP 	coarse trial 15, showing face,face.pos = [0. 5.], AI_data.shape = (166080, 6)
# 291.2223 	DATA 	Keypress: right
# 291.2359 	EXP 	coarse trial 15, hiding face, coarse_oculomatic_calib_values_candidate = [-0.4509764  0.9100183] for face.pos = [0. 5.], AI_data.shape = (166417, 6)
# 291.4934 	DATA 	Keypress: right
# 291.4939 	EXP 	coarse trial 15, candidate values accepted,coarse_oculomatic_calib_values[15] = [-0.4509764   0.91001832], AI_data.shape = (166576, 6)
# 291.5071 	EXP 	coarse trial 16, showing face,face.pos = [0. 5.], AI_data.shape = (166581, 6)
# 291.9877 	DATA 	Keypress: right
# 292.0006 	EXP 	coarse trial 16, hiding face, coarse_oculomatic_calib_values_candidate = [0.04890592 1.2413023 ] for face.pos = [0. 5.], AI_data.shape = (166856, 6)
# 292.2790 	DATA 	Keypress: right
# 292.2795 	EXP 	coarse trial 16, candidate values accepted,coarse_oculomatic_calib_values[16] = [0.04890592 1.24130225], AI_data.shape = (167025, 6)
# 292.2926 	EXP 	coarse trial 17, showing face,face.pos = [ 0. -5.], AI_data.shape = (167030, 6)
# 293.4818 	DATA 	Keypress: right
# 293.4951 	EXP 	coarse trial 17, hiding face, coarse_oculomatic_calib_values_candidate = [-0.10208441 -1.7719047 ] for face.pos = [ 0. -5.], AI_data.shape = (167715, 6)
# 293.7873 	DATA 	Keypress: right
# 293.7878 	EXP 	coarse trial 17, candidate values accepted,coarse_oculomatic_calib_values[17] = [-0.10208441 -1.77190471], AI_data.shape = (167895, 6)
# 293.8009 	EXP 	coarse trial 18, showing face,face.pos = [ 0. -5.], AI_data.shape = (167900, 6)
# 295.2543 	DATA 	Keypress: right
# 295.2677 	EXP 	coarse trial 18, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.29843286 -2.2534485 ] for face.pos = [ 0. -5.], AI_data.shape = (168734, 6)
# 295.6562 	DATA 	Keypress: right
# 295.6585 	EXP 	coarse trial 18, candidate values accepted,coarse_oculomatic_calib_values[18] = [ 0.29843286 -2.25344849], AI_data.shape = (168967, 6)
# 295.6709 	EXP 	coarse trial 19, showing face,face.pos = [0. 0.], AI_data.shape = (168972, 6)
# 298.4725 	DATA 	Keypress: right
# 298.4861 	EXP 	coarse trial 19, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.09800802 -0.5614969 ] for face.pos = [0. 0.], AI_data.shape = (170580, 6)
# 298.7704 	DATA 	Keypress: right
# 298.7724 	EXP 	coarse trial 19, candidate values accepted,coarse_oculomatic_calib_values[19] = [ 0.09800802 -0.56149691], AI_data.shape = (170752, 6)
# 298.7851 	EXP 	coarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (170757, 6)
# 300.9193 	DATA 	Keypress: left
# 300.9328 	EXP 	coarse trial 20, hiding face, AI_data.shape = (171990, 6)
# 301.2807 	DATA 	Keypress: right
# 301.2946 	EXP 	coarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (172196, 6)
# 302.0249 	DATA 	Keypress: left
# 302.0383 	EXP 	coarse trial 20, hiding face, AI_data.shape = (172622, 6)
# 302.6155 	DATA 	Keypress: right
# 302.6291 	EXP 	coarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (172960, 6)
# 303.2619 	DATA 	Keypress: left
# 303.2756 	EXP 	coarse trial 20, hiding face, AI_data.shape = (173332, 6)
# 304.0076 	DATA 	Keypress: right
# 304.0263 	EXP 	coarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (173761, 6)
# 305.1456 	DATA 	Keypress: right
# 305.1594 	EXP 	coarse trial 20, hiding face, coarse_oculomatic_calib_values_candidate = [-3.3869004   0.05144549] for face.pos = [-5.  5.], AI_data.shape = (174405, 6)
# 305.4259 	DATA 	Keypress: right
# 305.4310 	EXP 	coarse trial 20, candidate values accepted,coarse_oculomatic_calib_values[20] = [-3.38690042  0.05144549], AI_data.shape = (174570, 6)
# 305.4444 	EXP 	coarse trial 21, showing face,face.pos = [0. 5.], AI_data.shape = (174574, 6)
# 305.9659 	DATA 	Keypress: right
# 305.9796 	EXP 	coarse trial 21, hiding face, coarse_oculomatic_calib_values_candidate = [0.20977661 1.2132251 ] for face.pos = [0. 5.], AI_data.shape = (174874, 6)
# 306.1745 	DATA 	Keypress: right
# 306.1749 	EXP 	coarse trial 21, candidate values accepted,coarse_oculomatic_calib_values[21] = [0.20977661 1.21322513], AI_data.shape = (174998, 6)
# 306.1882 	EXP 	coarse trial 22, showing face,face.pos = [0. 0.], AI_data.shape = (175002, 6)
# 306.7932 	DATA 	Keypress: right
# 306.8068 	EXP 	coarse trial 22, hiding face, coarse_oculomatic_calib_values_candidate = [-3.0744348   0.30259103] for face.pos = [0. 0.], AI_data.shape = (175348, 6)
# 307.4090 	DATA 	Keypress: right
# 307.4127 	EXP 	coarse trial 22, candidate values accepted,coarse_oculomatic_calib_values[22] = [-3.07443476  0.30259103], AI_data.shape = (175707, 6)
# 307.4256 	EXP 	coarse trial 23, showing face,face.pos = [5. 5.], AI_data.shape = (175712, 6)
# 308.1416 	DATA 	Keypress: right
# 308.1553 	EXP 	coarse trial 23, hiding face, coarse_oculomatic_calib_values_candidate = [2.6770077 0.6244469] for face.pos = [5. 5.], AI_data.shape = (176123, 6)
# 308.5193 	DATA 	Keypress: right
# 308.5243 	EXP 	coarse trial 23, candidate values accepted,coarse_oculomatic_calib_values[23] = [2.67700768 0.62444693], AI_data.shape = (176347, 6)
# 308.5378 	EXP 	coarse trial 24, showing face,face.pos = [0. 0.], AI_data.shape = (176351, 6)
# 310.4635 	DATA 	Keypress: right
# 310.4773 	EXP 	coarse trial 24, hiding face, coarse_oculomatic_calib_values_candidate = [-0.49066135 -0.73969233] for face.pos = [0. 0.], AI_data.shape = (177455, 6)
# 310.7693 	DATA 	Keypress: right
# 310.7697 	EXP 	coarse trial 24, candidate values accepted,coarse_oculomatic_calib_values[24] = [-0.49066135 -0.73969233], AI_data.shape = (177632, 6)
# 310.7830 	EXP 	coarse trial 25, showing face,face.pos = [5. 0.], AI_data.shape = (177636, 6)
# 311.4714 	DATA 	Keypress: right
# 311.4850 	EXP 	coarse trial 25, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.9200567  -0.52850425] for face.pos = [5. 0.], AI_data.shape = (178031, 6)
# 311.8122 	DATA 	Keypress: right
# 311.8127 	EXP 	coarse trial 25, candidate values accepted,coarse_oculomatic_calib_values[25] = [ 0.9200567  -0.52850425], AI_data.shape = (178229, 6)
# 311.8258 	EXP 	coarse trial 26, showing face,face.pos = [5. 5.], AI_data.shape = (178234, 6)
# 313.9599 	DATA 	Keypress: left
# 313.9736 	EXP 	coarse trial 26, hiding face, AI_data.shape = (179469, 6)
# 314.2728 	DATA 	Keypress: right
# 314.2865 	EXP 	coarse trial 26, showing face,face.pos = [5. 5.], AI_data.shape = (179649, 6)
# 314.9817 	DATA 	Keypress: left
# 314.9954 	EXP 	coarse trial 26, hiding face, AI_data.shape = (180054, 6)
# 315.6067 	DATA 	Keypress: right
# 315.6212 	EXP 	coarse trial 26, showing face,face.pos = [5. 5.], AI_data.shape = (180413, 6)
# 315.9758 	DATA 	Keypress: right
# 315.9895 	EXP 	coarse trial 26, hiding face, coarse_oculomatic_calib_values_candidate = [1.5333806 1.3557628] for face.pos = [5. 5.], AI_data.shape = (180617, 6)
# 316.3789 	DATA 	Keypress: right
# 316.3792 	EXP 	coarse trial 26, candidate values accepted,coarse_oculomatic_calib_values[26] = [1.53338063 1.35576284], AI_data.shape = (180852, 6)
# 316.3800 	EXP 	coarse eye-tracking calibration end, coarse_stims_pos = [[ 5 -5]

# 316.3821 	EXP 	circular trajectory calibration start, AI_data.shape = (180857, 6)
# 316.3927 	EXP 	circular trajectory trial 0 start, faceID = 9, AI_data.shape = (180857, 6)
# 316.3927 	EXP 	circular trajectory trial 0, turn 0 start, AI_data.shape = (180857, 6)
# [...]
# 328.4045 	EXP 	circular trajectory trial 0, turn 3 end, AI_data.shape = (187767, 6)
# 328.4045 	EXP 	circular trajectory trial 0 end, AI_data.shape = (187767, 6)
# 328.4115 	EXP 	circular trajectory trial 1 start, faceID = 12, AI_data.shape = (187775, 6)
# [...]
# 388.5193 	EXP 	circular trajectory trial 5 end, AI_data.shape = (222356, 6)
# 388.5193 	EXP 	circular trajectory calibration end, AI_data.shape = (222356, 6)

#
# 388.5270 	EXP 	grid faces calibration start, AI_data.shape = (222368, 6)
# 388.5401 	EXP 	grid face trial 0, ISI start, AI_data.shape = (222368, 6)
# 389.0546 	EXP 	grid face trial 0, ISI end, AI_data.shape = see next entry
# 389.0546 	EXP 	grid face trial 0, face start, face.pos = [ 0. -5.], AI_data.shape = (222662, 6)
# 392.0783 	EXP 	grid face trial 0, face end, AI_data.shape = see next entry
# 392.0783 	EXP 	grid face trial 1, ISI start, AI_data.shape = (224406, 6)
# 392.5928 	EXP 	grid face trial 1, ISI end, AI_data.shape = see next entry
# 392.5928 	EXP 	grid face trial 1, face start, face.pos = [ 5. -5.], AI_data.shape = (224699, 6)
# 395.6166 	EXP 	grid face trial 1, face end, AI_data.shape = see next entry
# 395.6166 	EXP 	grid face trial 2, ISI start, AI_data.shape = (226445, 6)
# 396.1311 	EXP 	grid face trial 2, ISI end, AI_data.shape = see next entry
# 396.1311 	EXP 	grid face trial 2, face start, face.pos = [-5. -5.], AI_data.shape = (226736, 6)
# 399.1548 	EXP 	grid face trial 2, face end, AI_data.shape = see next entry
# 399.1548 	EXP 	grid face trial 3, ISI start, AI_data.shape = (228477, 6)
# 399.6694 	EXP 	grid face trial 3, ISI end, AI_data.shape = see next entry
# 399.6694 	EXP 	grid face trial 3, face start, face.pos = [5. 0.], AI_data.shape = (228773, 6)
# 402.6931 	EXP 	grid face trial 3, face end, AI_data.shape = see next entry
# 402.6931 	EXP 	grid face trial 4, ISI start, AI_data.shape = (230515, 6)
# 403.2077 	EXP 	grid face trial 4, ISI end, AI_data.shape = see next entry
# 403.2077 	EXP 	grid face trial 4, face start, face.pos = [5. 0.], AI_data.shape = (230810, 6)
# 406.2312 	EXP 	grid face trial 4, face end, AI_data.shape = see next entry
# 406.2312 	EXP 	grid face trial 5, ISI start, AI_data.shape = (232550, 6)
# 406.7458 	EXP 	grid face trial 5, ISI end, AI_data.shape = see next entry
# 406.7458 	EXP 	grid face trial 5, face start, face.pos = [5. 0.], AI_data.shape = (232841, 6)
# 409.7696 	EXP 	grid face trial 5, face end, AI_data.shape = see next entry
# 409.7696 	EXP 	grid face trial 6, ISI start, AI_data.shape = (234581, 6)
# 410.2840 	EXP 	grid face trial 6, ISI end, AI_data.shape = see next entry
# 410.2840 	EXP 	grid face trial 6, face start, face.pos = [-5.  5.], AI_data.shape = (234876, 6)
# 413.3077 	EXP 	grid face trial 6, face end, AI_data.shape = see next entry
# 413.3077 	EXP 	grid face trial 7, ISI start, AI_data.shape = (236617, 6)
# 413.8221 	EXP 	grid face trial 7, ISI end, AI_data.shape = see next entry
# 413.8221 	EXP 	grid face trial 7, face start, face.pos = [ 5. -5.], AI_data.shape = (236912, 6)
# 416.8461 	EXP 	grid face trial 7, face end, AI_data.shape = see next entry
# 416.8461 	EXP 	grid face trial 8, ISI start, AI_data.shape = (238652, 6)
# 417.3604 	EXP 	grid face trial 8, ISI end, AI_data.shape = see next entry
# 417.3604 	EXP 	grid face trial 8, face start, face.pos = [-5.  0.], AI_data.shape = (238948, 6)
# 420.3842 	EXP 	grid face trial 8, face end, AI_data.shape = see next entry
# 420.3842 	EXP 	grid face trial 9, ISI start, AI_data.shape = (240689, 6)
# 420.8985 	EXP 	grid face trial 9, ISI end, AI_data.shape = see next entry
# 420.8985 	EXP 	grid face trial 9, face start, face.pos = [ 5. -5.], AI_data.shape = (240983, 6)
# 423.9224 	EXP 	grid face trial 9, face end, AI_data.shape = see next entry
# 423.9224 	EXP 	grid face trial 10, ISI start, AI_data.shape = (242724, 6)
# 424.4368 	EXP 	grid face trial 10, ISI end, AI_data.shape = see next entry
# 424.4368 	EXP 	grid face trial 10, face start, face.pos = [ 0. -5.], AI_data.shape = (243018, 6)
# 427.4606 	EXP 	grid face trial 10, face end, AI_data.shape = see next entry
# 427.4606 	EXP 	grid face trial 11, ISI start, AI_data.shape = (244758, 6)
# 427.9749 	EXP 	grid face trial 11, ISI end, AI_data.shape = see next entry
# 427.9749 	EXP 	grid face trial 11, face start, face.pos = [5. 5.], AI_data.shape = (245054, 6)
# 430.9988 	EXP 	grid face trial 11, face end, AI_data.shape = see next entry
# 430.9988 	EXP 	grid face trial 12, ISI start, AI_data.shape = (246794, 6)
# 431.5133 	EXP 	grid face trial 12, ISI end, AI_data.shape = see next entry
# 431.5133 	EXP 	grid face trial 12, face start, face.pos = [-5.  0.], AI_data.shape = (247087, 6)
# 434.5301 	EXP 	grid face trial 12, face end, AI_data.shape = see next entry
# 434.5301 	EXP 	grid face trial 13, ISI start, AI_data.shape = (248824, 6)
# 435.0445 	EXP 	grid face trial 13, ISI end, AI_data.shape = see next entry
# 435.0445 	EXP 	grid face trial 13, face start, face.pos = [-5.  5.], AI_data.shape = (249118, 6)
# 438.0683 	EXP 	grid face trial 13, face end, AI_data.shape = see next entry
# 438.0683 	EXP 	grid face trial 14, ISI start, AI_data.shape = (250861, 6)
# 438.5828 	EXP 	grid face trial 14, ISI end, AI_data.shape = see next entry
# 438.5828 	EXP 	grid face trial 14, face start, face.pos = [5. 0.], AI_data.shape = (251157, 6)
# 441.6065 	EXP 	grid face trial 14, face end, AI_data.shape = see next entry
# 441.6065 	EXP 	grid face trial 15, ISI start, AI_data.shape = (252897, 6)
# 442.1209 	EXP 	grid face trial 15, ISI end, AI_data.shape = see next entry
# 442.1209 	EXP 	grid face trial 15, face start, face.pos = [5. 0.], AI_data.shape = (253193, 6)
# 445.1447 	EXP 	grid face trial 15, face end, AI_data.shape = see next entry
# 445.1447 	EXP 	grid face trial 16, ISI start, AI_data.shape = (254930, 6)
# 445.6592 	EXP 	grid face trial 16, ISI end, AI_data.shape = see next entry
# 445.6592 	EXP 	grid face trial 16, face start, face.pos = [0. 5.], AI_data.shape = (255224, 6)
# 448.6760 	EXP 	grid face trial 16, face end, AI_data.shape = see next entry
# 448.6760 	EXP 	grid face trial 17, ISI start, AI_data.shape = (256960, 6)
# 449.1904 	EXP 	grid face trial 17, ISI end, AI_data.shape = see next entry
# 449.1904 	EXP 	grid face trial 17, face start, face.pos = [5. 5.], AI_data.shape = (257253, 6)
# 452.2143 	EXP 	grid face trial 17, face end, AI_data.shape = see next entry
# 452.2143 	EXP 	grid face trial 18, ISI start, AI_data.shape = (258993, 6)
# 452.7286 	EXP 	grid face trial 18, ISI end, AI_data.shape = see next entry
# 452.7286 	EXP 	grid face trial 18, face start, face.pos = [0. 5.], AI_data.shape = (259287, 6)
# 455.7524 	EXP 	grid face trial 18, face end, AI_data.shape = see next entry
# 455.7524 	EXP 	grid face trial 19, ISI start, AI_data.shape = (261027, 6)
# 456.2669 	EXP 	grid face trial 19, ISI end, AI_data.shape = see next entry
# 456.2669 	EXP 	grid face trial 19, face start, face.pos = [5. 5.], AI_data.shape = (261322, 6)
# 459.2906 	EXP 	grid face trial 19, face end, AI_data.shape = see next entry
# 459.2906 	EXP 	grid face trial 20, ISI start, AI_data.shape = (263062, 6)
# 459.8051 	EXP 	grid face trial 20, ISI end, AI_data.shape = see next entry
# 459.8051 	EXP 	grid face trial 20, face start, face.pos = [ 0. -5.], AI_data.shape = (263356, 6)
# 462.8288 	EXP 	grid face trial 20, face end, AI_data.shape = see next entry
# 462.8288 	EXP 	grid face trial 21, ISI start, AI_data.shape = (265098, 6)
# 463.3434 	EXP 	grid face trial 21, ISI end, AI_data.shape = see next entry
# 463.3434 	EXP 	grid face trial 21, face start, face.pos = [ 5. -5.], AI_data.shape = (265392, 6)
# 466.3600 	EXP 	grid face trial 21, face end, AI_data.shape = see next entry
# 466.3600 	EXP 	grid face trial 22, ISI start, AI_data.shape = (267128, 6)
# 466.8745 	EXP 	grid face trial 22, ISI end, AI_data.shape = see next entry
# 466.8745 	EXP 	grid face trial 22, face start, face.pos = [5. 0.], AI_data.shape = (267424, 6)
# 469.8982 	EXP 	grid face trial 22, face end, AI_data.shape = see next entry
# 469.8982 	EXP 	grid face trial 23, ISI start, AI_data.shape = (269163, 6)
# 470.4128 	EXP 	grid face trial 23, ISI end, AI_data.shape = see next entry
# 470.4128 	EXP 	grid face trial 23, face start, face.pos = [-5. -5.], AI_data.shape = (269457, 6)
# 473.4296 	EXP 	grid face trial 23, face end, AI_data.shape = see next entry
# 473.4296 	EXP 	grid face trial 24, ISI start, AI_data.shape = (271193, 6)
# 473.9441 	EXP 	grid face trial 24, ISI end, AI_data.shape = see next entry
# 473.9441 	EXP 	grid face trial 24, face start, face.pos = [0. 0.], AI_data.shape = (271488, 6)
# 476.9678 	EXP 	grid face trial 24, face end, AI_data.shape = see next entry
# 476.9678 	EXP 	grid face trial 25, ISI start, AI_data.shape = (273228, 6)
# 477.4822 	EXP 	grid face trial 25, ISI end, AI_data.shape = see next entry
# 477.4822 	EXP 	grid face trial 25, face start, face.pos = [0. 0.], AI_data.shape = (273521, 6)
# 480.5060 	EXP 	grid face trial 25, face end, AI_data.shape = see next entry
# 480.5060 	EXP 	grid face trial 26, ISI start, AI_data.shape = (275261, 6)
# 481.0205 	EXP 	grid face trial 26, ISI end, AI_data.shape = see next entry
# 481.0205 	EXP 	grid face trial 26, face start, face.pos = [-5.  5.], AI_data.shape = (275554, 6)
# 484.0441 	EXP 	grid face trial 26, face end, AI_data.shape = see next entry
# 484.0441 	EXP 	grid face trial 27, ISI start, AI_data.shape = (277294, 6)
# 484.5586 	EXP 	grid face trial 27, ISI end, AI_data.shape = see next entry
# 484.5586 	EXP 	grid face trial 27, face start, face.pos = [0. 5.], AI_data.shape = (277589, 6)
# 487.5823 	EXP 	grid face trial 27, face end, AI_data.shape = see next entry
# 487.5823 	EXP 	grid face trial 28, ISI start, AI_data.shape = (279329, 6)
# 488.0967 	EXP 	grid face trial 28, ISI end, AI_data.shape = see next entry
# 488.0967 	EXP 	grid face trial 28, face start, face.pos = [0. 0.], AI_data.shape = (279624, 6)
# 491.1206 	EXP 	grid face trial 28, face end, AI_data.shape = see next entry
# 491.1206 	EXP 	grid face trial 29, ISI start, AI_data.shape = (281365, 6)
# 491.6351 	EXP 	grid face trial 29, ISI end, AI_data.shape = see next entry
# 491.6351 	EXP 	grid face trial 29, face start, face.pos = [0. 0.], AI_data.shape = (281661, 6)
# 494.6590 	EXP 	grid face trial 29, face end, AI_data.shape = see next entry
# 494.6590 	EXP 	grid face trial 30, ISI start, AI_data.shape = (283401, 6)
# 495.1732 	EXP 	grid face trial 30, ISI end, AI_data.shape = see next entry
# 495.1732 	EXP 	grid face trial 30, face start, face.pos = [-5.  0.], AI_data.shape = (283696, 6)
# 498.1970 	EXP 	grid face trial 30, face end, AI_data.shape = see next entry
# 498.1970 	EXP 	grid face trial 31, ISI start, AI_data.shape = (285437, 6)
# 498.7114 	EXP 	grid face trial 31, ISI end, AI_data.shape = see next entry
# 498.7114 	EXP 	grid face trial 31, face start, face.pos = [-5. -5.], AI_data.shape = (285729, 6)
# 501.7351 	EXP 	grid face trial 31, face end, AI_data.shape = see next entry
# 501.7351 	EXP 	grid face trial 32, ISI start, AI_data.shape = (287470, 6)
# 502.2496 	EXP 	grid face trial 32, ISI end, AI_data.shape = see next entry
# 502.2496 	EXP 	grid face trial 32, face start, face.pos = [-5. -5.], AI_data.shape = (287766, 6)
# 505.2736 	EXP 	grid face trial 32, face end, AI_data.shape = see next entry
# 505.2736 	EXP 	grid face trial 33, ISI start, AI_data.shape = (289504, 6)
# 505.7878 	EXP 	grid face trial 33, ISI end, AI_data.shape = see next entry
# 505.7878 	EXP 	grid face trial 33, face start, face.pos = [ 0. -5.], AI_data.shape = (289800, 6)
# 508.8116 	EXP 	grid face trial 33, face end, AI_data.shape = see next entry
# 508.8116 	EXP 	grid face trial 34, ISI start, AI_data.shape = (291539, 6)
# 509.3260 	EXP 	grid face trial 34, ISI end, AI_data.shape = see next entry
# 509.3260 	EXP 	grid face trial 34, face start, face.pos = [0. 5.], AI_data.shape = (291834, 6)
# 512.3498 	EXP 	grid face trial 34, face end, AI_data.shape = see next entry
# 512.3498 	EXP 	grid face trial 35, ISI start, AI_data.shape = (293574, 6)
# 512.8643 	EXP 	grid face trial 35, ISI end, AI_data.shape = see next entry
# 512.8643 	EXP 	grid face trial 35, face start, face.pos = [0. 0.], AI_data.shape = (293870, 6)
# 515.8811 	EXP 	grid face trial 35, face end, AI_data.shape = see next entry
# 515.8811 	EXP 	grid face trial 36, ISI start, AI_data.shape = (295606, 6)
# 516.3955 	EXP 	grid face trial 36, ISI end, AI_data.shape = see next entry
# 516.3955 	EXP 	grid face trial 36, face start, face.pos = [-5.  5.], AI_data.shape = (295899, 6)
# 519.4193 	EXP 	grid face trial 36, face end, AI_data.shape = see next entry
# 519.4193 	EXP 	grid face trial 37, ISI start, AI_data.shape = (297639, 6)
# 519.9337 	EXP 	grid face trial 37, ISI end, AI_data.shape = see next entry
# 519.9337 	EXP 	grid face trial 37, face start, face.pos = [5. 5.], AI_data.shape = (297934, 6)
# 522.9578 	EXP 	grid face trial 37, face end, AI_data.shape = see next entry
# 522.9578 	EXP 	grid face trial 38, ISI start, AI_data.shape = (299675, 6)
# 523.4722 	EXP 	grid face trial 38, ISI end, AI_data.shape = see next entry
# 523.4722 	EXP 	grid face trial 38, face start, face.pos = [-5.  0.], AI_data.shape = (299971, 6)
# 526.4957 	EXP 	grid face trial 38, face end, AI_data.shape = see next entry
# 526.4957 	EXP 	grid face trial 39, ISI start, AI_data.shape = (301709, 6)
# 527.0102 	EXP 	grid face trial 39, ISI end, AI_data.shape = see next entry
# 527.0102 	EXP 	grid face trial 39, face start, face.pos = [0. 5.], AI_data.shape = (302003, 6)
# 530.0343 	EXP 	grid face trial 39, face end, AI_data.shape = see next entry
# 530.0343 	EXP 	grid face trial 40, ISI start, AI_data.shape = (303744, 6)
# 530.5484 	EXP 	grid face trial 40, ISI end, AI_data.shape = see next entry
# 530.5484 	EXP 	grid face trial 40, face start, face.pos = [0. 5.], AI_data.shape = (304039, 6)
# 533.5652 	EXP 	grid face trial 40, face end, AI_data.shape = see next entry
# 533.5652 	EXP 	grid face trial 41, ISI start, AI_data.shape = (305778, 6)
# 534.0796 	EXP 	grid face trial 41, ISI end, AI_data.shape = see next entry
# 534.0796 	EXP 	grid face trial 41, face start, face.pos = [ 5. -5.], AI_data.shape = (306070, 6)
# 537.1034 	EXP 	grid face trial 41, face end, AI_data.shape = see next entry
# 537.1034 	EXP 	grid face trial 42, ISI start, AI_data.shape = (307810, 6)
# 537.6178 	EXP 	grid face trial 42, ISI end, AI_data.shape = see next entry
# 537.6178 	EXP 	grid face trial 42, face start, face.pos = [5. 5.], AI_data.shape = (308104, 6)
# 540.6416 	EXP 	grid face trial 42, face end, AI_data.shape = see next entry
# 540.6416 	EXP 	grid face trial 43, ISI start, AI_data.shape = (309844, 6)
# 541.1560 	EXP 	grid face trial 43, ISI end, AI_data.shape = see next entry
# 541.1560 	EXP 	grid face trial 43, face start, face.pos = [5. 5.], AI_data.shape = (310139, 6)
# 544.1798 	EXP 	grid face trial 43, face end, AI_data.shape = see next entry
# 544.1798 	EXP 	grid face trial 44, ISI start, AI_data.shape = (311879, 6)
# 544.6943 	EXP 	grid face trial 44, ISI end, AI_data.shape = see next entry
# 544.6943 	EXP 	grid face trial 44, face start, face.pos = [-5.  0.], AI_data.shape = (312172, 6)
# 547.7111 	EXP 	grid face trial 44, face end, AI_data.shape = see next entry
# 547.7111 	EXP 	grid face trial 45, ISI start, AI_data.shape = (313909, 6)
# 548.2255 	EXP 	grid face trial 45, ISI end, AI_data.shape = see next entry
# 548.2255 	EXP 	grid face trial 45, face start, face.pos = [-5.  0.], AI_data.shape = (314205, 6)
# 551.2493 	EXP 	grid face trial 45, face end, AI_data.shape = see next entry
# 551.2493 	EXP 	grid face trial 46, ISI start, AI_data.shape = (315945, 6)
# 551.7637 	EXP 	grid face trial 46, ISI end, AI_data.shape = see next entry
# 551.7637 	EXP 	grid face trial 46, face start, face.pos = [ 0. -5.], AI_data.shape = (316240, 6)
# 554.7875 	EXP 	grid face trial 46, face end, AI_data.shape = see next entry
# 554.7875 	EXP 	grid face trial 47, ISI start, AI_data.shape = (317984, 6)
# 555.3019 	EXP 	grid face trial 47, ISI end, AI_data.shape = see next entry
# 555.3019 	EXP 	grid face trial 47, face start, face.pos = [-5. -5.], AI_data.shape = (318278, 6)
# 558.3257 	EXP 	grid face trial 47, face end, AI_data.shape = see next entry
# 558.3257 	EXP 	grid face trial 48, ISI start, AI_data.shape = (320018, 6)
# 558.8402 	EXP 	grid face trial 48, ISI end, AI_data.shape = see next entry
# 558.8402 	EXP 	grid face trial 48, face start, face.pos = [ 5. -5.], AI_data.shape = (320313, 6)
# 561.8640 	EXP 	grid face trial 48, face end, AI_data.shape = see next entry
# 561.8640 	EXP 	grid face trial 49, ISI start, AI_data.shape = (322057, 6)
# 562.3784 	EXP 	grid face trial 49, ISI end, AI_data.shape = see next entry
# 562.3784 	EXP 	grid face trial 49, face start, face.pos = [0. 0.], AI_data.shape = (322347, 6)
# 565.4023 	EXP 	grid face trial 49, face end, AI_data.shape = see next entry
# 565.4023 	EXP 	grid face trial 50, ISI start, AI_data.shape = (324088, 6)
# 565.9167 	EXP 	grid face trial 50, ISI end, AI_data.shape = see next entry
# 565.9167 	EXP 	grid face trial 50, face start, face.pos = [ 0. -5.], AI_data.shape = (324384, 6)
# 568.9404 	EXP 	grid face trial 50, face end, AI_data.shape = see next entry
# 568.9404 	EXP 	grid face trial 51, ISI start, AI_data.shape = (326125, 6)
# 569.4548 	EXP 	grid face trial 51, ISI end, AI_data.shape = see next entry
# 569.4548 	EXP 	grid face trial 51, face start, face.pos = [-5. -5.], AI_data.shape = (326416, 6)
# 572.4787 	EXP 	grid face trial 51, face end, AI_data.shape = see next entry
# 572.4787 	EXP 	grid face trial 52, ISI start, AI_data.shape = (328156, 6)
# 572.9931 	EXP 	grid face trial 52, ISI end, AI_data.shape = see next entry
# 572.9931 	EXP 	grid face trial 52, face start, face.pos = [-5.  5.], AI_data.shape = (328450, 6)
# 576.0170 	EXP 	grid face trial 52, face end, AI_data.shape = see next entry
# 576.0170 	EXP 	grid face trial 53, ISI start, AI_data.shape = (330190, 6)
# 576.5314 	EXP 	grid face trial 53, ISI end, AI_data.shape = see next entry
# 576.5314 	EXP 	grid face trial 53, face start, face.pos = [-5.  5.], AI_data.shape = (330486, 6)
# 579.5482 	EXP 	grid face trial 53, face end, AI_data.shape = see next entry
# 579.5484 	EXP 	grid faces calibration end, AI_data.shape = (332223, 6)
# 579.5631 	EXP 	grid target eye-tracking calibration start, AI_data.shape = (332239, 6)
# 579.5829 	EXP 	grid target trial 0, ISI start, AI_data.shape = (332239, 6)
# 580.5769 	EXP 	grid target trial 0, ISI end, AI_data.shape = see next entry
# 580.5769 	EXP 	grid target trial 0, central target start, AI_data.shape = (332820, 6)
# 580.9177 	EXP 	grid target trial 0, central target fixation start, AI_data.shape = (333014, 6)
# 580.9247 	EXP 	grid target trial 0, central target fixation interrupted, AI_data.shape = (333018, 6)
# 581.5363 	EXP 	grid target trial 0, central target fixation start, AI_data.shape = (333368, 6)
# 581.5433 	EXP 	grid target trial 0, central target fixation interrupted, AI_data.shape = (333372, 6)
# 581.5919 	EXP 	grid target trial 0, central target fixation start, AI_data.shape = (333400, 6)
# 581.6406 	EXP 	grid target trial 0, central target fixation interrupted, AI_data.shape = (333428, 6)
# 581.9812 	EXP 	grid target trial 0, central target fixation start, AI_data.shape = (333622, 6)
# 582.0229 	EXP 	grid target trial 0, central target fixation interrupted, AI_data.shape = (333646, 6)
# 582.5928 	EXP 	grid target trial 0, central target end, fixation fail, AI_data.shape = see next entry
# 582.6137 	EXP 	grid target trial 1, ISI start, AI_data.shape = (333976, 6)
# 583.6077 	EXP 	grid target trial 1, ISI end, AI_data.shape = see next entry
# 583.6077 	EXP 	grid target trial 1, central target start, AI_data.shape = (334556, 6)
# 585.2830 	EXP 	grid target trial 1, central target fixation start, AI_data.shape = (335516, 6)
# 585.3872 	EXP 	grid target trial 1, central target fixation completed, AI_data.shape = (335576, 6)
# 585.4010 	EXP 	grid target trial 1, central target end, fixation success, AI_data.shape = see next entry
# 585.4149 	EXP 	grid target trial 1, grid target start, grid_target.pos = [5. 0.], AI_data.shape = (335589, 6)
# 585.4177 	EXP 	grid target trial 1, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (335596, 6)
# 585.5225 	EXP 	grid target trial 1, grid target fixation completed, grid_target.pos = [5. 0.], AI_data.shape = (335652, 6)
# 585.5332 	EXP 	grid target trial 1, grid target end, fixation success, AI_data.shape = see next entry
# 585.5332 	EXP 	grid target trial 1, face reward start, face.pos = [5. 0.], AI_data.shape = (335657, 6)
# 586.0616 	EXP 	grid target trial 1, face reward end, AI_data.shape = see next entry
# 586.0616 	EXP 	grid target trial 2, ISI start, AI_data.shape = (335950, 6)
# 587.0556 	EXP 	grid target trial 2, ISI end, AI_data.shape = see next entry
# 587.0556 	EXP 	grid target trial 2, central target start, AI_data.shape = (336530, 6)
# 588.6545 	EXP 	grid target trial 2, central target fixation start, AI_data.shape = (337445, 6)
# 588.7588 	EXP 	grid target trial 2, central target fixation completed, AI_data.shape = (337505, 6)
# 588.7726 	EXP 	grid target trial 2, central target end, fixation success, AI_data.shape = see next entry
# 588.7865 	EXP 	grid target trial 2, grid target start, grid_target.pos = [0. 0.], AI_data.shape = (337518, 6)
# 588.7895 	EXP 	grid target trial 2, grid target fixation start, grid_target.pos = [0. 0.], AI_data.shape = (337525, 6)
# 588.8910 	EXP 	grid target trial 2, grid target fixation completed, grid_target.pos = [0. 0.], AI_data.shape = (337580, 6)
# 588.9046 	EXP 	grid target trial 2, grid target end, fixation success, AI_data.shape = see next entry
# 588.9046 	EXP 	grid target trial 2, face reward start, face.pos = [0. 0.], AI_data.shape = (337584, 6)
# 589.4260 	EXP 	grid target trial 2, face reward end, AI_data.shape = see next entry
# 589.4260 	EXP 	grid target trial 3, ISI start, AI_data.shape = (337877, 6)
# 590.4200 	EXP 	grid target trial 3, ISI end, AI_data.shape = see next entry
# 590.4200 	EXP 	grid target trial 3, central target start, AI_data.shape = (338457, 6)
# 590.9205 	EXP 	grid target trial 3, central target fixation start, AI_data.shape = (338745, 6)
# 591.0248 	EXP 	grid target trial 3, central target fixation completed, AI_data.shape = (338805, 6)
# 591.0387 	EXP 	grid target trial 3, central target end, fixation success, AI_data.shape = see next entry
# 591.0525 	EXP 	grid target trial 3, grid target start, grid_target.pos = [0. 0.], AI_data.shape = (338818, 6)
# 593.0755 	EXP 	grid target trial 3, grid target end, fixation fail, AI_data.shape = see next entry
# 593.0755 	EXP 	grid target trial 4, ISI start, AI_data.shape = (339966, 6)
# 594.0694 	EXP 	grid target trial 4, ISI end, AI_data.shape = see next entry
# 594.0694 	EXP 	grid target trial 4, central target start, AI_data.shape = (340545, 6)
# 595.8421 	EXP 	grid target trial 4, central target fixation start, AI_data.shape = (341563, 6)
# 595.8491 	EXP 	grid target trial 4, central target fixation interrupted, AI_data.shape = (341567, 6)
# 595.8561 	EXP 	grid target trial 4, central target fixation start, AI_data.shape = (341571, 6)
# 595.9603 	EXP 	grid target trial 4, central target fixation completed, AI_data.shape = (341631, 6)
# 595.9742 	EXP 	grid target trial 4, central target end, fixation success, AI_data.shape = see next entry
# 595.9882 	EXP 	grid target trial 4, grid target start, grid_target.pos = [ 0. -5.], AI_data.shape = (341645, 6)
# 597.7679 	EXP 	grid target trial 4, grid target fixation start, grid_target.pos = [ 0. -5.], AI_data.shape = (342663, 6)
# 597.8722 	EXP 	grid target trial 4, grid target fixation completed, grid_target.pos = [ 0. -5.], AI_data.shape = (342721, 6)
# 597.8857 	EXP 	grid target trial 4, grid target end, fixation success, AI_data.shape = see next entry
# 597.8857 	EXP 	grid target trial 4, face reward start, face.pos = [ 0. -5.], AI_data.shape = (342725, 6)
# 598.4071 	EXP 	grid target trial 4, face reward end, AI_data.shape = see next entry
# 598.4071 	EXP 	grid target trial 5, ISI start, AI_data.shape = (343019, 6)
# 599.4009 	EXP 	grid target trial 5, ISI end, AI_data.shape = see next entry
# 599.4009 	EXP 	grid target trial 5, central target start, AI_data.shape = (343599, 6)
# 601.4170 	EXP 	grid target trial 5, central target end, fixation fail, AI_data.shape = see next entry
# 601.4379 	EXP 	grid target trial 6, ISI start, AI_data.shape = (344760, 6)
# 602.4319 	EXP 	grid target trial 6, ISI end, AI_data.shape = see next entry
# 602.4319 	EXP 	grid target trial 6, central target start, AI_data.shape = (345340, 6)
# 602.8491 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (345579, 6)
# 602.8560 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (345583, 6)
# 602.8630 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (345587, 6)
# 602.9603 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (345642, 6)
# 602.9672 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (345646, 6)
# 602.9881 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (345658, 6)
# 603.7040 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (346069, 6)
# 603.7944 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (346121, 6)
# 603.8013 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (346125, 6)
# 603.8153 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (346133, 6)
# 603.8223 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (346137, 6)
# 603.8431 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (346149, 6)
# 603.8570 	EXP 	grid target trial 6, central target fixation start, AI_data.shape = (346157, 6)
# 603.8640 	EXP 	grid target trial 6, central target fixation interrupted, AI_data.shape = (346161, 6)
# 604.4478 	EXP 	grid target trial 6, central target end, fixation fail, AI_data.shape = see next entry
# 604.4686 	EXP 	grid target trial 7, ISI start, AI_data.shape = (346499, 6)
# 605.4696 	EXP 	grid target trial 7, ISI end, AI_data.shape = see next entry
# 605.4696 	EXP 	grid target trial 7, central target start, AI_data.shape = (347083, 6)
# 606.1510 	EXP 	grid target trial 7, central target fixation start, AI_data.shape = (347474, 6)
# 606.2552 	EXP 	grid target trial 7, central target fixation completed, AI_data.shape = (347534, 6)
# 606.2691 	EXP 	grid target trial 7, central target end, fixation success, AI_data.shape = see next entry
# 606.2830 	EXP 	grid target trial 7, grid target start, grid_target.pos = [5. 0.], AI_data.shape = (347547, 6)
# 606.5346 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347693, 6)
# 606.5482 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347702, 6)
# 606.5552 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347705, 6)
# 606.5615 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347709, 6)
# 606.5821 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347719, 6)
# 606.5890 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347723, 6)
# 606.5961 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347727, 6)
# 606.6028 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347731, 6)
# 606.7212 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347799, 6)
# 606.7419 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347811, 6)
# 606.7560 	EXP 	grid target trial 7, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (347819, 6)
# 606.7697 	EXP 	grid target trial 7, grid target fixation interrupted, AI_data.shape = (347827, 6)
# 608.3058 	EXP 	grid target trial 7, grid target end, fixation fail, AI_data.shape = see next entry
# 608.3058 	EXP 	grid target trial 8, ISI start, AI_data.shape = (348702, 6)
# 609.2998 	EXP 	grid target trial 8, ISI end, AI_data.shape = see next entry
# 609.2998 	EXP 	grid target trial 8, central target start, AI_data.shape = (349281, 6)
# 610.8573 	EXP 	grid target trial 8, central target fixation start, AI_data.shape = (350176, 6)
# 610.8640 	EXP 	grid target trial 8, central target fixation interrupted, AI_data.shape = (350180, 6)
# 610.8709 	EXP 	grid target trial 8, central target fixation start, AI_data.shape = (350183, 6)
# 610.9265 	EXP 	grid target trial 8, central target fixation interrupted, AI_data.shape = (350215, 6)
# 611.3157 	EXP 	grid target trial 8, central target end, fixation fail, AI_data.shape = see next entry
# 611.3365 	EXP 	grid target trial 9, ISI start, AI_data.shape = (350443, 6)
# 612.3305 	EXP 	grid target trial 9, ISI end, AI_data.shape = see next entry
# 612.3305 	EXP 	grid target trial 9, central target start, AI_data.shape = (351023, 6)
# 612.3322 	EXP 	grid target trial 9, central target fixation start, AI_data.shape = (351027, 6)
# 612.4349 	EXP 	grid target trial 9, central target fixation completed, AI_data.shape = (351083, 6)
# 612.4487 	EXP 	grid target trial 9, central target end, fixation success, AI_data.shape = see next entry
# 612.4627 	EXP 	grid target trial 9, grid target start, grid_target.pos = [-5. -5.], AI_data.shape = (351097, 6)
# 612.5115 	EXP 	grid target trial 9, grid target fixation start, grid_target.pos = [-5. -5.], AI_data.shape = (351127, 6)
# 612.6168 	EXP 	grid target trial 9, grid target fixation completed, grid_target.pos = [-5. -5.], AI_data.shape = (351187, 6)
# 612.6297 	EXP 	grid target trial 9, grid target end, fixation success, AI_data.shape = see next entry
# 612.6297 	EXP 	grid target trial 9, face reward start, face.pos = [-5. -5.], AI_data.shape = (351192, 6)
# 613.1579 	EXP 	grid target trial 9, face reward end, AI_data.shape = see next entry
# 613.1579 	EXP 	grid target trial 10, ISI start, AI_data.shape = (351488, 6)
# 614.1586 	EXP 	grid target trial 10, ISI end, AI_data.shape = see next entry
# 614.1586 	EXP 	grid target trial 10, central target start, AI_data.shape = (352072, 6)
# 614.5273 	EXP 	grid target trial 10, central target fixation start, AI_data.shape = (352284, 6)
# 614.6314 	EXP 	grid target trial 10, central target fixation completed, AI_data.shape = (352344, 6)
# 614.6453 	EXP 	grid target trial 10, central target end, fixation success, AI_data.shape = see next entry
# 614.6594 	EXP 	grid target trial 10, grid target start, grid_target.pos = [5. 5.], AI_data.shape = (352357, 6)
# 615.0281 	EXP 	grid target trial 10, grid target fixation start, grid_target.pos = [5. 5.], AI_data.shape = (352570, 6)
# 615.1324 	EXP 	grid target trial 10, grid target fixation completed, grid_target.pos = [5. 5.], AI_data.shape = (352629, 6)
# 615.1458 	EXP 	grid target trial 10, grid target end, fixation success, AI_data.shape = see next entry
# 615.1458 	EXP 	grid target trial 10, face reward start, face.pos = [5. 5.], AI_data.shape = (352633, 6)
# 615.6672 	EXP 	grid target trial 10, face reward end, AI_data.shape = see next entry
# 615.6672 	EXP 	grid target trial 11, ISI start, AI_data.shape = (352927, 6)
# 616.6612 	EXP 	grid target trial 11, ISI end, AI_data.shape = see next entry
# 616.6612 	EXP 	grid target trial 11, central target start, AI_data.shape = (353508, 6)
# 617.7596 	EXP 	grid target trial 11, central target fixation start, AI_data.shape = (354136, 6)
# 617.8639 	EXP 	grid target trial 11, central target fixation completed, AI_data.shape = (354196, 6)
# 617.8777 	EXP 	grid target trial 11, central target end, fixation success, AI_data.shape = see next entry
# 617.8916 	EXP 	grid target trial 11, grid target start, grid_target.pos = [0. 0.], AI_data.shape = (354209, 6)
# 617.8952 	EXP 	grid target trial 11, grid target fixation start, grid_target.pos = [0. 0.], AI_data.shape = (354216, 6)
# 617.9961 	EXP 	grid target trial 11, grid target fixation completed, grid_target.pos = [0. 0.], AI_data.shape = (354272, 6)
# 618.0098 	EXP 	grid target trial 11, grid target end, fixation success, AI_data.shape = see next entry
# 618.0098 	EXP 	grid target trial 11, face reward start, face.pos = [0. 0.], AI_data.shape = (354276, 6)
# 618.5313 	EXP 	grid target trial 11, face reward end, AI_data.shape = see next entry
# 618.5313 	EXP 	grid target trial 12, ISI start, AI_data.shape = (354571, 6)
# 619.5390 	EXP 	grid target trial 12, ISI end, AI_data.shape = see next entry
# 619.5390 	EXP 	grid target trial 12, central target start, AI_data.shape = (355159, 6)
# 620.3669 	EXP 	grid target trial 12, central target fixation start, AI_data.shape = (355633, 6)
# 620.3811 	EXP 	grid target trial 12, central target fixation interrupted, AI_data.shape = (355642, 6)
# 620.3874 	EXP 	grid target trial 12, central target fixation start, AI_data.shape = (355646, 6)
# 620.4011 	EXP 	grid target trial 12, central target fixation interrupted, AI_data.shape = (355653, 6)
# 620.4497 	EXP 	grid target trial 12, central target fixation start, AI_data.shape = (355681, 6)
# 620.5540 	EXP 	grid target trial 12, central target fixation completed, AI_data.shape = (355741, 6)
# 620.5678 	EXP 	grid target trial 12, central target end, fixation success, AI_data.shape = see next entry
# 620.5817 	EXP 	grid target trial 12, grid target start, grid_target.pos = [ 5. -5.], AI_data.shape = (355754, 6)
# 620.5849 	EXP 	grid target trial 12, grid target fixation start, grid_target.pos = [ 5. -5.], AI_data.shape = (355761, 6)
# 620.6375 	EXP 	grid target trial 12, grid target fixation interrupted, AI_data.shape = (355788, 6)
# 622.6047 	EXP 	grid target trial 12, grid target end, fixation fail, AI_data.shape = see next entry
# 622.6047 	EXP 	grid target trial 13, ISI start, AI_data.shape = (356902, 6)
# 623.5986 	EXP 	grid target trial 13, ISI end, AI_data.shape = see next entry
# 623.5986 	EXP 	grid target trial 13, central target start, AI_data.shape = (357481, 6)
# 623.8072 	EXP 	grid target trial 13, central target fixation start, AI_data.shape = (357601, 6)
# 623.9115 	EXP 	grid target trial 13, central target fixation completed, AI_data.shape = (357661, 6)
# 623.9255 	EXP 	grid target trial 13, central target end, fixation success, AI_data.shape = see next entry
# 623.9393 	EXP 	grid target trial 13, grid target start, grid_target.pos = [-5.  0.], AI_data.shape = (357675, 6)
# 623.9951 	EXP 	grid target trial 13, grid target fixation start, grid_target.pos = [-5.  0.], AI_data.shape = (357708, 6)
# 624.0714 	EXP 	grid target trial 13, grid target fixation interrupted, AI_data.shape = (357751, 6)
# 624.1064 	EXP 	grid target trial 13, grid target fixation start, grid_target.pos = [-5.  0.], AI_data.shape = (357771, 6)
# 624.1132 	EXP 	grid target trial 13, grid target fixation interrupted, AI_data.shape = (357775, 6)
# 624.5443 	EXP 	grid target trial 13, grid target fixation start, grid_target.pos = [-5.  0.], AI_data.shape = (358021, 6)
# 624.6486 	EXP 	grid target trial 13, grid target fixation completed, grid_target.pos = [-5.  0.], AI_data.shape = (358080, 6)
# 624.6622 	EXP 	grid target trial 13, grid target end, fixation success, AI_data.shape = see next entry
# 624.6622 	EXP 	grid target trial 13, face reward start, face.pos = [-5.  0.], AI_data.shape = (358084, 6)
# 625.1837 	EXP 	grid target trial 13, face reward end, AI_data.shape = see next entry
# 625.1837 	EXP 	grid target trial 14, ISI start, AI_data.shape = (358375, 6)
# 626.1984 	EXP 	grid target trial 14, ISI end, AI_data.shape = see next entry
# 626.1984 	EXP 	grid target trial 14, central target start, AI_data.shape = (358968, 6)
# 626.7337 	EXP 	grid target trial 14, central target fixation start, AI_data.shape = (359272, 6)
# 626.7476 	EXP 	grid target trial 14, central target fixation interrupted, AI_data.shape = (359280, 6)
# 627.8947 	EXP 	grid target trial 14, central target fixation start, AI_data.shape = (359937, 6)
# 627.9017 	EXP 	grid target trial 14, central target fixation interrupted, AI_data.shape = (359941, 6)
# 628.0266 	EXP 	grid target trial 14, central target fixation start, AI_data.shape = (360013, 6)
# 628.0335 	EXP 	grid target trial 14, central target fixation interrupted, AI_data.shape = (360017, 6)
# 628.2141 	EXP 	grid target trial 14, central target end, fixation fail, AI_data.shape = see next entry
# 628.2350 	EXP 	grid target trial 15, ISI start, AI_data.shape = (360122, 6)
# 629.2361 	EXP 	grid target trial 15, ISI end, AI_data.shape = see next entry
# 629.2361 	EXP 	grid target trial 15, central target start, AI_data.shape = (360706, 6)
# 631.1825 	EXP 	grid target trial 15, central target fixation start, AI_data.shape = (361823, 6)
# 631.2867 	EXP 	grid target trial 15, central target fixation completed, AI_data.shape = (361883, 6)
# 631.3005 	EXP 	grid target trial 15, central target end, fixation success, AI_data.shape = see next entry
# 631.3144 	EXP 	grid target trial 15, grid target start, grid_target.pos = [5. 0.], AI_data.shape = (361896, 6)
# 631.6832 	EXP 	grid target trial 15, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (362107, 6)
# 631.7874 	EXP 	grid target trial 15, grid target fixation completed, grid_target.pos = [5. 0.], AI_data.shape = (362166, 6)
# 631.8010 	EXP 	grid target trial 15, grid target end, fixation success, AI_data.shape = see next entry
# 631.8010 	EXP 	grid target trial 15, face reward start, face.pos = [5. 0.], AI_data.shape = (362170, 6)
# 632.3224 	EXP 	grid target trial 15, face reward end, AI_data.shape = see next entry
# 632.3224 	EXP 	grid target trial 16, ISI start, AI_data.shape = (362465, 6)
# 633.3166 	EXP 	grid target trial 16, ISI end, AI_data.shape = see next entry
# 633.3166 	EXP 	grid target trial 16, central target start, AI_data.shape = (363045, 6)
# 634.8390 	EXP 	grid target trial 16, central target fixation start, AI_data.shape = (363918, 6)
# 634.9432 	EXP 	grid target trial 16, central target fixation completed, AI_data.shape = (363978, 6)
# 634.9571 	EXP 	grid target trial 16, central target end, fixation success, AI_data.shape = see next entry
# 634.9709 	EXP 	grid target trial 16, grid target start, grid_target.pos = [ 5. -5.], AI_data.shape = (363991, 6)
# 635.1797 	EXP 	grid target trial 16, grid target fixation start, grid_target.pos = [ 5. -5.], AI_data.shape = (364114, 6)
# 635.2840 	EXP 	grid target trial 16, grid target fixation completed, grid_target.pos = [ 5. -5.], AI_data.shape = (364173, 6)
# 635.2975 	EXP 	grid target trial 16, grid target end, fixation success, AI_data.shape = see next entry
# 635.2975 	EXP 	grid target trial 16, face reward start, face.pos = [ 5. -5.], AI_data.shape = (364177, 6)
# 635.8191 	EXP 	grid target trial 16, face reward end, AI_data.shape = see next entry
# 635.8191 	EXP 	grid target trial 17, ISI start, AI_data.shape = (364471, 6)
# 636.8130 	EXP 	grid target trial 17, ISI end, AI_data.shape = see next entry
# 636.8130 	EXP 	grid target trial 17, central target start, AI_data.shape = (365051, 6)
# 637.1124 	EXP 	grid target trial 17, central target fixation start, AI_data.shape = (365221, 6)
# 637.2169 	EXP 	grid target trial 17, central target fixation completed, AI_data.shape = (365280, 6)
# 637.2302 	EXP 	grid target trial 17, central target end, fixation success, AI_data.shape = see next entry
# 637.2440 	EXP 	grid target trial 17, grid target start, grid_target.pos = [-5.  0.], AI_data.shape = (365293, 6)
# 637.2471 	EXP 	grid target trial 17, grid target fixation start, grid_target.pos = [-5.  0.], AI_data.shape = (365299, 6)
# 637.3415 	EXP 	grid target trial 17, grid target fixation interrupted, AI_data.shape = (365351, 6)
# 639.2669 	EXP 	grid target trial 17, grid target end, fixation fail, AI_data.shape = see next entry
# 639.2669 	EXP 	grid target trial 18, ISI start, AI_data.shape = (366443, 6)
# 640.2608 	EXP 	grid target trial 18, ISI end, AI_data.shape = see next entry
# 640.2608 	EXP 	grid target trial 18, central target start, AI_data.shape = (367023, 6)
# 640.2627 	EXP 	grid target trial 18, central target fixation start, AI_data.shape = (367027, 6)
# 640.2886 	EXP 	grid target trial 18, central target fixation interrupted, AI_data.shape = (367039, 6)
# 641.1366 	EXP 	grid target trial 18, central target fixation start, AI_data.shape = (367522, 6)
# 641.1436 	EXP 	grid target trial 18, central target fixation interrupted, AI_data.shape = (367526, 6)
# 641.1505 	EXP 	grid target trial 18, central target fixation start, AI_data.shape = (367530, 6)
# 641.1714 	EXP 	grid target trial 18, central target fixation interrupted, AI_data.shape = (367542, 6)
# 641.1922 	EXP 	grid target trial 18, central target fixation start, AI_data.shape = (367554, 6)
# 641.2061 	EXP 	grid target trial 18, central target fixation interrupted, AI_data.shape = (367562, 6)
# 642.1863 	EXP 	grid target trial 18, central target fixation start, AI_data.shape = (368125, 6)
# 642.2907 	EXP 	grid target trial 18, central target fixation completed, AI_data.shape = (368185, 6)
# 642.3044 	EXP 	grid target trial 18, central target end, fixation success, AI_data.shape = see next entry
# 642.3183 	EXP 	grid target trial 18, grid target start, grid_target.pos = [-5.  5.], AI_data.shape = (368198, 6)
# 642.7012 	EXP 	grid target trial 18, grid target fixation start, grid_target.pos = [-5.  5.], AI_data.shape = (368420, 6)
# 642.7079 	EXP 	grid target trial 18, grid target fixation interrupted, AI_data.shape = (368424, 6)
# 643.1947 	EXP 	grid target trial 18, grid target fixation start, grid_target.pos = [-5.  5.], AI_data.shape = (368703, 6)
# 643.2083 	EXP 	grid target trial 18, grid target fixation interrupted, AI_data.shape = (368710, 6)
# 644.3412 	EXP 	grid target trial 18, grid target end, fixation fail, AI_data.shape = see next entry
# 644.3412 	EXP 	grid target trial 19, ISI start, AI_data.shape = (369352, 6)
# 645.3352 	EXP 	grid target trial 19, ISI end, AI_data.shape = see next entry
# 645.3352 	EXP 	grid target trial 19, central target start, AI_data.shape = (369932, 6)
# 647.3511 	EXP 	grid target trial 19, central target end, fixation fail, AI_data.shape = see next entry
# 647.3719 	EXP 	grid target trial 20, ISI start, AI_data.shape = (371094, 6)
# 648.3660 	EXP 	grid target trial 20, ISI end, AI_data.shape = see next entry
# 648.3660 	EXP 	grid target trial 20, central target start, AI_data.shape = (371673, 6)
# 649.0751 	EXP 	grid target trial 20, central target fixation start, AI_data.shape = (372079, 6)
# 649.1794 	EXP 	grid target trial 20, central target fixation completed, AI_data.shape = (372139, 6)
# 649.1933 	EXP 	grid target trial 20, central target end, fixation success, AI_data.shape = see next entry
# 649.2073 	EXP 	grid target trial 20, grid target start, grid_target.pos = [5. 5.], AI_data.shape = (372152, 6)
# 651.2299 	EXP 	grid target trial 20, grid target end, fixation fail, AI_data.shape = see next entry
# 651.2299 	EXP 	grid target trial 21, ISI start, AI_data.shape = (373303, 6)
# 652.2239 	EXP 	grid target trial 21, ISI end, AI_data.shape = see next entry
# 652.2239 	EXP 	grid target trial 21, central target start, AI_data.shape = (373883, 6)
# 652.6064 	EXP 	grid target trial 21, central target fixation start, AI_data.shape = (374101, 6)
# 652.6133 	EXP 	grid target trial 21, central target fixation interrupted, AI_data.shape = (374105, 6)
# 653.8506 	EXP 	grid target trial 21, central target fixation start, AI_data.shape = (374815, 6)
# 653.8576 	EXP 	grid target trial 21, central target fixation interrupted, AI_data.shape = (374819, 6)
# 654.2398 	EXP 	grid target trial 21, central target end, fixation fail, AI_data.shape = see next entry
# 654.2606 	EXP 	grid target trial 22, ISI start, AI_data.shape = (375041, 6)
# 655.2547 	EXP 	grid target trial 22, ISI end, AI_data.shape = see next entry
# 655.2547 	EXP 	grid target trial 22, central target start, AI_data.shape = (375619, 6)
# 657.2706 	EXP 	grid target trial 22, central target end, fixation fail, AI_data.shape = see next entry
# 657.2914 	EXP 	grid target trial 23, ISI start, AI_data.shape = (376777, 6)
# 658.3063 	EXP 	grid target trial 23, ISI end, AI_data.shape = see next entry
# 658.3063 	EXP 	grid target trial 23, central target start, AI_data.shape = (377369, 6)
# 658.7791 	EXP 	grid target trial 23, central target fixation start, AI_data.shape = (377638, 6)
# 658.8000 	EXP 	grid target trial 23, central target fixation interrupted, AI_data.shape = (377650, 6)
# 659.0016 	EXP 	grid target trial 23, central target fixation start, AI_data.shape = (377766, 6)
# 659.0155 	EXP 	grid target trial 23, central target fixation interrupted, AI_data.shape = (377774, 6)
# 659.0363 	EXP 	grid target trial 23, central target fixation start, AI_data.shape = (377786, 6)
# 659.0433 	EXP 	grid target trial 23, central target fixation interrupted, AI_data.shape = (377790, 6)
# 660.3222 	EXP 	grid target trial 23, central target end, fixation fail, AI_data.shape = see next entry
# 660.3430 	EXP 	grid target trial 24, ISI start, AI_data.shape = (378525, 6)
# 661.3371 	EXP 	grid target trial 24, ISI end, AI_data.shape = see next entry
# 661.3371 	EXP 	grid target trial 24, central target start, AI_data.shape = (379105, 6)
# 663.3530 	EXP 	grid target trial 24, central target end, fixation fail, AI_data.shape = see next entry
# 663.3738 	EXP 	grid target trial 25, ISI start, AI_data.shape = (380258, 6)
# 664.3748 	EXP 	grid target trial 25, ISI end, AI_data.shape = see next entry
# 664.3748 	EXP 	grid target trial 25, central target start, AI_data.shape = (380842, 6)
# 665.5150 	EXP 	grid target trial 25, central target fixation start, AI_data.shape = (381494, 6)
# 665.5983 	EXP 	grid target trial 25, central target fixation interrupted, AI_data.shape = (381542, 6)
# 666.3907 	EXP 	grid target trial 25, central target end, fixation fail, AI_data.shape = see next entry
# 666.4116 	EXP 	grid target trial 26, ISI start, AI_data.shape = (381999, 6)
# 667.4055 	EXP 	grid target trial 26, ISI end, AI_data.shape = see next entry
# 667.4055 	EXP 	grid target trial 26, central target start, AI_data.shape = (382578, 6)
# 669.4214 	EXP 	grid target trial 26, central target end, fixation fail, AI_data.shape = see next entry
# 669.4426 	EXP 	grid target trial 27, ISI start, AI_data.shape = (383731, 6)
# 670.4433 	EXP 	grid target trial 27, ISI end, AI_data.shape = see next entry
# 670.4433 	EXP 	grid target trial 27, central target start, AI_data.shape = (384314, 6)
# 672.4664 	EXP 	grid target trial 27, central target end, fixation fail, AI_data.shape = see next entry
# 672.4870 	EXP 	grid target trial 28, ISI start, AI_data.shape = (385471, 6)
# 673.4881 	EXP 	grid target trial 28, ISI end, AI_data.shape = see next entry
# 673.4881 	EXP 	grid target trial 28, central target start, AI_data.shape = (386054, 6)
# 675.5038 	EXP 	grid target trial 28, central target end, fixation fail, AI_data.shape = see next entry
# 675.5247 	EXP 	grid target trial 29, ISI start, AI_data.shape = (387208, 6)
# 676.5186 	EXP 	grid target trial 29, ISI end, AI_data.shape = see next entry
# 676.5186 	EXP 	grid target trial 29, central target start, AI_data.shape = (387788, 6)
# 678.5346 	EXP 	grid target trial 29, central target end, fixation fail, AI_data.shape = see next entry
# 678.5555 	EXP 	grid target trial 30, ISI start, AI_data.shape = (388940, 6)
# 679.5634 	EXP 	grid target trial 30, ISI end, AI_data.shape = see next entry
# 679.5634 	EXP 	grid target trial 30, central target start, AI_data.shape = (389528, 6)
# 681.5793 	EXP 	grid target trial 30, central target end, fixation fail, AI_data.shape = see next entry
# 681.6001 	EXP 	grid target trial 31, ISI start, AI_data.shape = (390678, 6)
# 682.5940 	EXP 	grid target trial 31, ISI end, AI_data.shape = see next entry
# 682.5940 	EXP 	grid target trial 31, central target start, AI_data.shape = (391258, 6)
# 683.1929 	EXP 	grid target trial 31, central target fixation start, AI_data.shape = (391592, 6)
# 683.2686 	EXP 	grid target trial 31, central target fixation interrupted, AI_data.shape = (391639, 6)
# 683.5605 	EXP 	grid target trial 31, central target fixation start, AI_data.shape = (391805, 6)
# 683.5744 	EXP 	grid target trial 31, central target fixation interrupted, AI_data.shape = (391813, 6)
# 683.9498 	EXP 	grid target trial 31, central target fixation start, AI_data.shape = (392025, 6)
# 683.9776 	EXP 	grid target trial 31, central target fixation interrupted, AI_data.shape = (392041, 6)
# 684.6102 	EXP 	grid target trial 31, central target end, fixation fail, AI_data.shape = see next entry
# 684.6309 	EXP 	grid target trial 32, ISI start, AI_data.shape = (392404, 6)
# 685.6458 	EXP 	grid target trial 32, ISI end, AI_data.shape = see next entry
# 685.6458 	EXP 	grid target trial 32, central target start, AI_data.shape = (392996, 6)
# 686.9597 	EXP 	grid target trial 32, central target fixation start, AI_data.shape = (393744, 6)
# 686.9667 	EXP 	grid target trial 32, central target fixation interrupted, AI_data.shape = (393748, 6)
# 687.4810 	EXP 	grid target trial 32, central target fixation start, AI_data.shape = (394040, 6)
# 687.4880 	EXP 	grid target trial 32, central target fixation interrupted, AI_data.shape = (394043, 6)
# 687.6617 	EXP 	grid target trial 32, central target end, fixation fail, AI_data.shape = see next entry
# 687.6825 	EXP 	grid target trial 33, ISI start, AI_data.shape = (394147, 6)
# 688.6767 	EXP 	grid target trial 33, ISI end, AI_data.shape = see next entry
# 688.6767 	EXP 	grid target trial 33, central target start, AI_data.shape = (394726, 6)
# 689.6647 	EXP 	grid target trial 33, central target fixation start, AI_data.shape = (395291, 6)
# 689.7680 	EXP 	grid target trial 33, central target fixation completed, AI_data.shape = (395345, 6)
# 689.7818 	EXP 	grid target trial 33, central target end, fixation success, AI_data.shape = see next entry
# 689.7957 	EXP 	grid target trial 33, grid target start, grid_target.pos = [-5.  5.], AI_data.shape = (395358, 6)
# 691.2636 	EXP 	grid target trial 33, grid target fixation start, grid_target.pos = [-5.  5.], AI_data.shape = (396195, 6)
# 691.3670 	EXP 	grid target trial 33, grid target fixation completed, grid_target.pos = [-5.  5.], AI_data.shape = (396255, 6)
# 691.3806 	EXP 	grid target trial 33, grid target end, fixation success, AI_data.shape = see next entry
# 691.3806 	EXP 	grid target trial 33, face reward start, face.pos = [-5.  5.], AI_data.shape = (396259, 6)
# 691.9020 	EXP 	grid target trial 33, face reward end, AI_data.shape = see next entry
# 691.9020 	EXP 	grid target trial 34, ISI start, AI_data.shape = (396551, 6)
# 692.9170 	EXP 	grid target trial 34, ISI end, AI_data.shape = see next entry
# 692.9170 	EXP 	grid target trial 34, central target start, AI_data.shape = (397143, 6)
# 693.9180 	EXP 	grid target trial 34, central target fixation start, AI_data.shape = (397707, 6)
# 694.0223 	EXP 	grid target trial 34, central target fixation completed, AI_data.shape = (397766, 6)
# 694.0360 	EXP 	grid target trial 34, central target end, fixation success, AI_data.shape = see next entry
# 694.0499 	EXP 	grid target trial 34, grid target start, grid_target.pos = [ 5. -5.], AI_data.shape = (397778, 6)
# 694.0521 	EXP 	grid target trial 34, grid target fixation start, grid_target.pos = [ 5. -5.], AI_data.shape = (397785, 6)
# 694.1545 	EXP 	grid target trial 34, grid target fixation completed, grid_target.pos = [ 5. -5.], AI_data.shape = (397841, 6)
# 694.1684 	EXP 	grid target trial 34, grid target end, fixation success, AI_data.shape = see next entry
# 694.1684 	EXP 	grid target trial 34, face reward start, face.pos = [ 5. -5.], AI_data.shape = (397845, 6)
# 694.6894 	EXP 	grid target trial 34, face reward end, AI_data.shape = see next entry
# 694.6894 	EXP 	grid target trial 35, ISI start, AI_data.shape = (398134, 6)
# 695.6836 	EXP 	grid target trial 35, ISI end, AI_data.shape = see next entry
# 695.6836 	EXP 	grid target trial 35, central target start, AI_data.shape = (398713, 6)
# 697.2129 	EXP 	grid target trial 35, central target fixation start, AI_data.shape = (399585, 6)
# 697.2268 	EXP 	grid target trial 35, central target fixation interrupted, AI_data.shape = (399593, 6)
# 697.6994 	EXP 	grid target trial 35, central target end, fixation fail, AI_data.shape = see next entry
# 697.7133 	EXP 	grid target eye-tracking calibration end, AI_data.shape = (399857, 6)
# 697.7146 	EXP 	calibration complete, duration 695.61sec, dropped frames = 40522
#

# ecdata = {}
# ims = {}
# impaths = {}
# lf_categories = {}
# tmp_image = None
# tmp_imagepath = None
# tmp_units = None
# tmp_pos = None
# tmp_size = None
# tmp_ori = None
# tmp_category = None
# tmp_catid = None
# tmp_cond = None
# tmp_acqfr = None
# tmp_stimtimestr = ''
# stimtime_mode = False
# tmp_isitimestr = ''
# isitime_mode = False
# for line in eclines:
#     if 'EXP \tstim_times:' in line:
#         stimtime_mode = True
#     if 'EXP \tinterstim_times:' in line:
#         isitime_mode = True
#     if stimtime_mode:
#         tmp_stimtimestr = tmp_stimtimestr + line
#         if ']' in line:
#             stimtime_mode = False
#     if isitime_mode:
#         tmp_isitimestr = tmp_isitimestr + line
#         if ']' in line:
#             isitime_mode = False
#
#     if 'stim start' not in line or 'image' not in line:
#         continue
#
#     col = line.split('trial')
#     if not col:
#         continue
#     subcol = [sc.strip() for sc in col[1].split(',')]
#     tmp_trial = int(subcol[0].split('/')[0].strip())
#     if 'cond' in subcol[3]:
#         tmp_cond = int(subcol[3].split('=')[1].strip())
#     else:
#         print('could not get cond from log entry')
#     if 'image' in subcol[4]:
#         tmp_image = subcol[4].split(':')[1].strip()
#         if tmp_image not in ims:
#             ims[tmp_image] = tmp_cond
#         tmp_category = tmp_image[0]
#         if tmp_category not in lf_categories:
#             lf_categories[tmp_category] = len(lf_categories)
#         tmp_catid = lf_categories[tmp_category]
#     else:
#         print('could not get image name from log entry')
#     if 'path' in subcol[5]:
#         tmp_imagepath = subcol[5].split('=')[1].strip()
#         if tmp_imagepath not in impaths:
#             impaths[tmp_imagepath] = tmp_cond
#     else:
#         print('could not get image name from log entry')
#     if 'units' in subcol[6]:
#         tmp_units = subcol[6].split('=')[1].strip()
#     else:
#         print('could not get units from log entry')
#     if 'pos' in subcol[7]:
#         tmp_pos = np.fromstring(subcol[7].split('=')[1].strip('[]'), sep=' ')
#     else:
#         print('could not get pos from log entry')
#     if 'size' in subcol[8]:
#         tmp_size = np.fromstring(subcol[8].split('=')[1].strip('[]'), sep=' ')
#     else:
#         print('could not get size from log entry')
#     if 'ori' in subcol[9]:
#         tmp_ori = float(subcol[9].split('=')[1].strip())
#     else:
#         print('could not get ori from log entry')
#     if 'acqfr' in subcol[15]:
#         tmp_acqfr = int(subcol[15].split('=')[1].strip())
#     else:
#         print('could not get acqfr from log entry')
#     trialdata[tmp_trial] = {'cond': tmp_cond,  # effectively image_id
#                             'image': tmp_image,
#                             'imagepath': tmp_imagepath,
#                             'units': tmp_units,
#                             'pos': tmp_pos,
#                             'size': tmp_size,
#                             'ori': tmp_ori,
#                             'category': tmp_category,
#                             'catid': tmp_catid,
#                             'acqfr': tmp_acqfr}


# % Extract stimulus information from log file

# *** TODO load from a pickle file or pandas frame instead of a text log

if log is not None:
    lines = log.splitlines()
else:
    raise RuntimeError('Could not find log file.')

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
tmp_units = None
tmp_pos = None
tmp_size = None
tmp_ori = None
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
    if 'units' in subcol[6]:
        tmp_units = subcol[6].split('=')[1].strip()
    else:
        print('could not get units from log entry')
    if 'pos' in subcol[7]:
        tmp_pos = np.fromstring(subcol[7].split('=')[1].strip('[]'), sep=' ')
    else:
        print('could not get pos from log entry')
    if 'size' in subcol[8]:
        tmp_size = np.fromstring(subcol[8].split('=')[1].strip('[]'), sep=' ')
    else:
        print('could not get size from log entry')
    if 'ori' in subcol[9]:
        tmp_ori = float(subcol[9].split('=')[1].strip())
    else:
        print('could not get ori from log entry')
    if 'acqfr' in subcol[15]:
        tmp_acqfr = int(subcol[15].split('=')[1].strip())
    else:
        print('could not get acqfr from log entry')
    trialdata[tmp_trial] = {'cond': tmp_cond,  # effectively image_id
                            'image': tmp_image,
                            'imagepath': tmp_imagepath,
                            'units': tmp_units,
                            'pos': tmp_pos,
                            'size': tmp_size,
                            'ori': tmp_ori,
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

dur_trial = dur_isi + dur_stim + dur_isi
n_samp_stim = int(np.ceil(dur_stim * md['framerate']))
n_samp_isi = int(np.round(dur_isi * md['framerate']))
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

lf_categories = {v: k for k, v in lf_categories.items()}
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

# trialdata_arr[trial_idx] = [cond/imageid, category_id, acqfr]
trialdata_arr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdata_arr[td] = [trialdata[td]['cond'], trialdata[td]['catid'], trialdata[td]['acqfr']]
units_list = [v for k, v in enumerate(np.unique([trialdata[td]['units'] for td in trialdata]))]
pos_arr = np.array([trialdata[td]['pos'] for td in trialdata])
size_arr = np.array([trialdata[td]['size'] for td in trialdata])
ori_arr = np.array([trialdata[td]['ori'] for td in trialdata])
trialdata_arr = trialdata_arr.astype(int)
all_stim_start_frames = trialdata_arr[:, 2]

stim = {}
if len(units_list) == 1:
    stim['units'] = units_list[0]
else:
    print('Not all stimulus units were the same.')
    stim['units'] = None
if np.all(np.isclose(pos_arr[0, :], pos_arr[:, :])):
    stim['pos'] = pos_arr[0]
else:
    print('Not all stimulus positions were the same.')
    stim['pos'] = None
if np.all(np.isclose(size_arr[0, :], size_arr[:, :])):
    stim['size'] = size_arr[0]
    print('All stimuli were the same size: {} {}'.format(stim['size'],
                                                         stim['units'] if stim['units'] is not None else ''))
else:
    print('Not all stimulus sizes were the same.')
    stim['size'] = None
if np.all(np.isclose(ori_arr[0], ori_arr[:])):
    stim['ori'] = ori_arr[0]
else:
    print('Not all stimulus orientations were the same.')
    stim['ori'] = None


if len(np.unique(all_stim_start_frames)) != len(all_stim_start_frames):
    raise RuntimeError('Imaging was interrupted or stopped before stimulus. ' +
                       'Handling this is not yet implemented.')                                

# condinds = [cond, trial_idx]
conds = np.unique(trialdata_arr[:, 0])
n_conds = len(conds)
n_trials = int(len(trialdata) / n_conds)
cats = np.unique(trialdata_arr[:, 1])
n_cats = len(cats)
conds_per_cat = int(n_conds / n_cats)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdata_arr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdata_arr[condinds[:], 2]
fridx = acqfr_by_conds

# Define stimuli


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
    tmp_pitch = -32768
    tmp_yaw = -32768
    tmp_roll = -32768
    imn = image_names[c]
    tmp_imagename = image_filenames[c]
    tmp_ip = os.path.join(stimimage_path, tmp_imagename)
    tmp_imagepath = tmp_ip if os.path.isfile(tmp_ip) else None
    if image_set == 'FOBmin' or image_set == 'FOBmany':
        # pattern_fn = r'^([0-9]{6}tUTC).*$'
        # FreiwaldFOB2018_Marm_Body_Spring_7_erode3px
        # FreiwaldFOB2012_Human_Head_10_erode3px
        # FreiwaldFOB2018_Marm_Head_Hunter_8_erode3px
        # FreiwaldFOB2018_Objects_Manmade1_8_erode3px
        # FreiwaldMarmosetCartoon_0_erode3px
        # FreiwaldFOB2018_Marm_Head_Lollipop_1_erode3px_inverted
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

if n_conds != conditions.shape[0]:
    u, c = np.unique(data[:]['cond'], return_counts=True)
    mult = u[c > 1]
    warn('Some different image files were combined into the same condition. ' +
         '{}'.format(data[data['cond'] == mult]['imagename']))
    del u, c, mult


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


# TODO: check for NaN values rather than using nanmean?
# if np.any(np.isnan(volume)):
#     raise Exception('NaNs found in preprocessed volume.')

idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)

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

FdFF_allfaces_meanRstimall = np.nanmean(data[(data['cat'] == b'face_mrm') & (data['yaw'] == 0) & (data['roll'] == 0)]['FdFF_meant'][:, :, idx_stim],
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

# FSIs(_by_roi) = [roi, fsi]
FSIs_dFF = (FdFF_allfaces_meanRstimall - FdFF_allobjs_meanRstimall) / \
           (FdFF_allfaces_meanRstimall + FdFF_allobjs_meanRstimall)
FSIs_zsc = (Fzsc_allfaces_meanRstimall - Fzsc_allobjs_meanRstimall) / \
           (Fzsc_allfaces_meanRstimall + Fzsc_allobjs_meanRstimall)


# Face selectivity d′
# based on Vinken et al Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
# Face selectivity was quantified by computing the d′ sensitivity index comparing trial-averaged responses to faces
# and non-faces:
# d′ = (μ_F - μ_NF) / sqrt((σ_F^2 + σ_NF^2) / 2)
# where μ_F and μ_NF are the across-stimulus averages of the trial-averaged responses to faces and non-faces, and
# σ_F and σ_NF are the across-stimulus SDs. This face d′ value quantifies how much higher (positive d′) or lower
# (negative d′) the response to a face is expected to be compared to a non-face, in SD units.

bool_F = (data['cat'] == b'face_mrm')
bool_NF = (data['cat'] != b'face_mrm')
mu_F = np.mean(data[bool_F]['FdFF'][:, :, :, idx_stim], axis=(0, 2, 3))
mu_NF = np.mean(data[bool_NF]['FdFF'][:, :, :, idx_stim], axis=(0, 2, 3))
sigma_F = np.std(np.mean(data[bool_F]['FdFF'], axis=2)[:, :, idx_stim], axis=(0, 2))
sigma_NF = np.std(np.mean(data[bool_NF]['FdFF'], axis=2)[:, :, idx_stim], axis=(0, 2))
dprime = (mu_F - mu_NF) / np.sqrt((sigma_F**2 + sigma_NF**2) / 2)
sort_dp = np.argsort(dprime)[::-1]
del bool_F, bool_NF, mu_F, mu_NF, sigma_F, sigma_NF


# % Define ROIs as tuned or untuned using the FSI

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For
# |face-selectivity index| = 1/3, that is, if the response to faces was at
# least twice (or at most half) that of nonface objects, a cell was classed
# as being face selective45–47.

print('|FSI| threshold: {}'.format(threshold_fsi))
tunidx_fsi = FSIs_zsc
tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).squeeze()
n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > threshold_fsi).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


# TODO regorganize

# Plot histograms
sp = os.path.join(save_path, save_pfix + '_Histogram_FSIs_fromFdFF' + save_ext) if saving else ''
plots.plot_hist_fsi(FSIs_dFF, threshold=threshold_fsi, title='FSIs calculated from FdFF values', save_path=sp)
sp = os.path.join(save_path, save_pfix + '_Histogram_FSIs_fromZscr' + save_ext) if saving else ''
plots.plot_hist_fsi(FSIs_zsc, threshold=threshold_fsi, title='FSIs calculated from z-scored values', save_path=sp)

sp = os.path.join(save_path, save_pfix + '_Histogram_dprimes_fromFdFF' + save_ext) if saving else ''
plots.plot_hist_dprime(dprime, threshold=threshold_dprime, title='dprimes calculated from FdFF values', save_path=sp)


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

above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
# TODO THIS SHOULD NOT BE HARD CODED
at_sortidx = (-np.mean(data[cond_idx[0:19]]['Fzsc_meant'][:, :, idx_stim], axis=(0,-1))[above_threshold]).argsort()


# Establish ordering for heatmaps

tmpl = np.array([b'blank', b'scram_s', b'scram_p',
                 b'face_mrm', b'face_rhe', b'face_hum', b'face_ctn',
                 b'obj', b'food',
                 b'body_mrm', b'animal'], dtype='|S8')

tmpl_labels = {b'blank': 'Blank',
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


lambda_sort = lambda x: (np.where(tmpl == x[1].category)[0][0]
                         if np.where(tmpl == x[1].category)[0].size > 0
                         else np.iinfo(np.where(tmpl == x[1].category)[0].dtype).max,
                         np.abs(x[1].roll),
                         x[1].roll,
                         x[1].yaw,
                         x[1].condition.decode().lower())
stimarr = data[:]['stimulus']
stimcond = [i for i, x in sorted(enumerate(stimarr), key=lambda_sort)]
stimsort = stimarr[stimcond]

tickinfo = {t.decode(): {} for t in tmpl}
for t in tmpl:
    ts = t.decode()
    wheret = np.where(data[stimcond]['cat'] == t)[0]
    if wheret.size > 0:
        tickinfo[ts]['start'] = wheret[0]
        tickinfo[ts]['end'] = wheret[-1]
        tickinfo[ts]['labelpos'] = (tickinfo[ts]['start'] + tickinfo[ts]['end']) / 2
        tickinfo[ts]['label'] = tmpl_labels[t]
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




# %% Plot the data


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

metrics = ['FdFF', 'Fzsc']
metric_labels = {'FdFF': 'dF/F',
                 'Fzsc': 'Z-score'}
n_metrics = len(metrics)

# ... by category, including the average for each condition within that category.
fr = md['framerate']
for r in range(n_plot_ROIs):
    ridx = sort_dp[plot_ROI_subset[r]]
    fig = plt.figure()
    fig.suptitle('ROI {}: mean response by category (each cond mean plotted)'.format(ridx), fontsize=10)
    axes = fig.subplots(nrows=n_metrics, ncols=n_cats)
    for m, met in enumerate(metrics):
        ymin = np.min(np.mean(data[met][:, ridx, :, :], axis=1))
        ymax = np.max(np.mean(data[met][:, ridx, :, :], axis=1))
        for cat in range(n_cats):
            ax = axes[m, cat]
            if m == 0:
                ax.set_title(tmpl_labels[categories[cat]], fontsize=10)
            if cat == 0:
                ax.set_ylabel(metric_labels[met], fontsize=10)
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
                ax.plot(range(n_samp_trial), np.mean(data[data['cat'] == categories[cat]][met][cnd, ridx, :, :], axis=0),
                        linewidth=0.5, markersize=0.5, color=str(np.linspace(0.4, 0.7, n_cnd_in_cat)[cnd]), zorder=1)
            Fmean = np.mean(data[data['cat'] == categories[cat]][met][:, ridx, :, :], axis=(0, 1))
            Fsem = np.std(data[data['cat'] == categories[cat]][met][:, ridx, :, :], axis=(0, 1)) / np.sqrt(n_cnd_in_cat)
            ax.plot(range(n_samp_trial), Fmean, color='0.0', zorder=3)
            ax.fill_between(range(n_samp_trial), Fmean - Fsem, Fmean + Fsem, facecolor='0.2', alpha=0.6, zorder=2)
    plt.show()
del fr, xticks, xticklabels


# ... by conditions (selected subset), including the average for each trial within that condition
fr = md['framerate']
fig = plt.figure()
fig.suptitle('mean response by condition (each trial plotted)', fontsize=8)
met = 'Fzsc'
focus_cat = b'face_mrm'
bool_focuscat = data['cat'] == focus_cat
stims = data[bool_focuscat]['stimulus']
stimconds = [i for i, x in sorted(enumerate(stims), key=lambda_sort)]
sortedstims = stims[stimconds]
conds_in_fcat = [i for i, x in sorted(enumerate(data[bool_focuscat]['stimulus']), key=lambda_sort)]
n_cnd_in_fcat = len(sortedstims)
axes = fig.subplots(nrows=(n_plot_ROIs + 1), ncols=(n_cnd_in_fcat + 1), sharey='row')
for r in range(n_plot_ROIs):
    ridx = sort_dp[plot_ROI_subset[r]]
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
    ymin = np.min(np.mean(data[met][:, ridx, :, :], axis=1))
    ymax = np.max(np.mean(data[met][:, ridx, :, :], axis=1))
    # if m == 0:
    #     ax.set_title(tmpl_labels[categories[cat]], fontsize=10)
    ax = axes[pr, 0]
    ax.axis('off')
    # if r == 0:
    #     ax.set_ylabel(metric_labels[met], fontsize=8)
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
        Fmean = np.mean(data[bool_cat][met][:, ridx, :, :], axis=(0, 1))
        Fsem = np.std(data[bool_cat][met][:, ridx, :, :], axis=(0, 1)) / np.sqrt(n_cnd_in_cat)
        ax.plot(range(n_samp_trial), Fmean, color=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), linewidth=1, zorder=3)
        ax.fill_between(range(n_samp_trial), Fmean - Fsem, Fmean + Fsem, facecolor=colorsys.hsv_to_rgb(cat / n_cats, 1.0, 1.0), alpha=0.6, zorder=2)

    # Plot each cond
    for cnd in range(n_cnd_in_fcat):
        bool_cnd = data['cond'] == sortedstims[cnd].condition
        ax = axes[pr, cnd + 1]
        ax.axis('off')
        ax.axvspan(dur_isi * fr, (dur_isi + dur_stim) * fr, color='0.9', zorder=0)
        ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
        for t in range(n_trials):
            ax.plot(range(n_samp_trial), data[bool_cnd][met][0, ridx, t, :],
                    color=str(np.linspace(0.4, 0.7, n_trials)[t]),
                    linewidth=0.1)
        Fmean = np.mean(data[bool_cnd][met][0, ridx, :, :], axis=0)
        Fsem = np.std(data[bool_cnd][met][0, ridx, :, :], axis=0) / np.sqrt(n_cnd_in_cat)
        ax.plot(range(n_samp_trial), Fmean, color='0.0', linewidth=1, zorder=3)
        ax.fill_between(range(n_samp_trial), Fmean - Fsem, Fmean + Fsem, facecolor='0.0', alpha=0.6, zorder=2)
del bool_cat, bool_cnd
plt.show()



# ... by conditions (selected subset), as a trial-averaged heatmap
fr = md['framerate']
fig = plt.figure()
fig.suptitle('trial-averaged heat maps by condition', fontsize=8)
met = 'Fzsc'
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
ax_dp.barh(range(0, n_ROIs), dprime[sort_dp], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(linestyle='--', linewidth=0.5, color='0.75')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)

for cnd in range(n_cnd_in_fcat):
    bool_cnd = data['cond'] == sortedstims[cnd].condition
    ax = axes[pr, cnd + 1]
    ax.axis('off')
    img_hm = ax.imshow(np.mean(data[bool_cnd][met][0, :, :, :], axis=1)[sort_dp],
                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
    xlines = [dur_isi * fr, (dur_isi + dur_stim) * fr]
    for xl in xlines:
        ax.axvline(x=xl, linestyle='--', linewidth=0.5, color='0.6')

del bool_cnd
plt.show()


# Plot heatmap of mean responses to all presented conditions (images) for ROIs
# with at least one stimulus period z-score > 0.5
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
img_hm = ax_hm.imshow(np.mean(data[stimcond]['Fzsc_meant'][:, :, idx_stim], axis=-1).swapaxes(0, 1)[sort_dp],
                      vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
# ax_hm.invert_yaxis()
# ax_hm.axvline(x=20)

cbar = plt.colorbar(img_hm, ax=ax_hm, shrink=0.6)  # , location='bottom', shrink=0.6)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0','0.5','1'])
cbar.set_label('mean Zscore during stimulus')

ax_dp.set_xlabel('Face d′')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[sort_dp], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(linestyle='--', linewidth=0.5, color='0.75')  # axis='x'
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
    
# ax_fsi.set_xlabel('FSI')
# ax_fsi.set_axisbelow(True)
# ax_fsi.set_xlim([-1, 1])
# ax_fsi.barh(range(0, n_ROIs), FSIs_zsc[sort_dp], height=1.0, color='0.5')
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
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)

# Plot for each ROI the category of the condition (image) that elicited the largest response  
above_threshold = np.where(np.abs(FSIs_zsc) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)

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


# Plot for each ROI the category elicited the largest average response
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)

# Plot for each ROI the category elicited the largest average response
# for only ROIs with FSI > threshold
above_threshold = np.where(np.abs(FSIs_zsc) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)

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
sn = save_pfix + '_ROIplot_ColorByRelativeResponseStrength_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)

above_threshold = np.where(FSIs_zsc > threshold_fsi)[0]
sn = save_pfix + '_ROIplot_ColorByRelativeResponseStrength_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], Fzsc_fob_norm[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90, save_path=sp)


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

