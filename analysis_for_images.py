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

    # y: observed fluorescence
    # c: calcium concentration
    # s: neural activity / spike train
    # b: baseline
    # "To produce calcium trace c, spike train s is filtered with the inverse filter of g, an infinite impulse response h, c = s * h."
    # decay factor γ, regularization parameter λ, data y, sigma noise

    oasisL0_c, oasisL0_s, oasisL0_b, oasisL0_g, oasisL0_lam = oasis.functions.deconvolve(Fr, penalty=0)
    Fr_oasisL0 = oasisL0_c + oasisL0_b
    Fr_dFF_oasisL0 = (Fr_oasisL0 - oasisL0_b) / oasisL0_b

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
else:
    eclines = ''

ecdata = {'zero': None,
          'crse': {'data': {}},
          'circ': {'data': {}},
          'grdf': {'data': {}},
          'grdt': {'data': {}}}

pattern_AIidx = r'.*AI_data\.shape\ *=\ *(\((\d+),\ *([0-9]+)\)|see\ *next\ *entry).*'
pattern_pos = r'.*\.pos\ *=\ *\[([^\[\]]*)\],?.*'
pattern_cvals = r'.*calib_values_candidate\ *=\ *\[([^\[\]]*)\].*'

lmode = ''
for idx_line, line in enumerate(eclines):
    # First pass through eye-calibration log file
    if 'EXP \toculomatic zeroing start' in line:
        # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
        lmode = 'zero'
        # '234.7010 \tEXP \toculomatic zeroing end, AI_data.shape = (133928, 6)'
        if 'zeroing end' in line:
            lmode = ''
    elif 'EXP \tcoarse eye-tracking calibration start' in line:
        # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
        lmode = 'crse'
        # '316.3800 \tEXP \tcoarse eye-tracking calibration end, coarse_stims_pos = [[ 5 -5]'
        if 'calibration end' in line:
            lmode = ''
    elif 'EXP \tcircular trajectory calibration start' in line:
        # '316.3821 \tEXP \tcircular trajectory calibration start, AI_data.shape = (180857, 6)'
        lmode = 'circ'
        # '388.5193 \tEXP \tcircular trajectory calibration end, AI_data.shape = (222356, 6)'
        if 'calibration end' in line:
            lmode = ''
    elif 'EXP \tgrid faces calibration start' in line:
        # '388.5270 \tEXP \tgrid faces calibration start, AI_data.shape = (222368, 6)'
        lmode = 'grdf'
        # '579.5484 \tEXP \tgrid faces calibration end, AI_data.shape = (332223, 6)'
        if 'calibration end' in line:
            lmode = ''
    elif 'EXP \tgrid target eye-tracking calibration start' in line:
        # '579.5631 \tEXP \tgrid target eye-tracking calibration start, AI_data.shape = (332239, 6)'
        lmode = 'grdt'
        # '697.7133 \tEXP \tgrid target eye-tracking calibration end, AI_data.shape = (399857, 6)'
        if 'calibration end' in line:
            lmode = ''
    if lmode == '':
        continue

    if re.match(pattern_AIidx, line) is not None:
        g_AIidx = re.match(pattern_AIidx, line).groups()
        if g_AIidx[0].replace(' ', '') != 'seenextentry':
            tmp_AIidx = int(g_AIidx[1])
        else:
            # Use 'sep' character as a placeholder for subsequent AI range value.
            tmp_AIidx = chr(31)
    else:
        g_AIidx = None
        tmp_AIidx = None
    if re.match(pattern_pos, line) is not None:
        g_pos = re.match(pattern_pos, line).groups()
        tmp_pos = np.fromstring(g_pos[0].strip(), dtype=float, sep=' ')
    else:
        g_pos = None
        tmp_pos = None
    if re.match(pattern_cvals, line) is not None:
        g_cvals = re.match(pattern_cvals, line).groups()
        tmp_cvals = np.fromstring(re.match(pattern_cvals, line).groups()[0].strip(), dtype=float, sep=' ')
    else:
        g_cvals = None
        tmp_cvals = None

    match lmode:
        case 'zero':
            pattern_zero = r'.*oculomatic\ *zeroing,\ *(presenting|hiding)\ *face,?.*'
            # '6.3156 \tEXP \toculomatic zeroing, presenting face, AI_data.shape = (2529, 6)'
            # '234.7007 \tEXP \toculomatic zeroing, hiding face, AI_data.shape = (133928, 6)'
            if re.match(pattern_zero, line) is not None:
                g = re.match(pattern_zero, line).groups()
                if g[0] == 'presenting':
                    ecdata['zero'] = {'AIrng': [None, None]}
                    ecdata['zero']['AIrng'][0] = tmp_AIidx
                elif g[0] == 'hiding':
                    ecdata['zero']['AIrng'][1] = tmp_AIidx

        case 'crse':
            pattern_coarse = r'.*coarse\ *(eye-tracking\ *calibration|trial)\ *(start|end|\d+)?,?\ *' + \
                             r'(showing\ *face|hiding\ *face|candidate)?,?.*'
                             # r'([^\[\]]*values_candidate\ *=\ *\[(.*)\]\ *[^\[\]]*\ *)?'
            # '234.8201 \tEXP \tcoarse eye-tracking calibration start, AI_data.shape = (134078, 6)'
            # '234.8261 \tEXP \tcoarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (134078, 6)'
            # '243.3274 \tEXP \tcoarse trial 0, hiding face, AI_data.shape = (138942, 6)'
            # '243.8488 \tEXP \tcoarse trial 0, showing face,face.pos = [ 5. -5.], AI_data.shape = (139240, 6)',
            # '247.9988 \tEXP \tcoarse trial 0, hiding face, coarse_oculomatic_calib_values_candidate = [ 0.825335  -2.4251535] for face.pos = [ 5. -5.], AI_data.shape = (141609, 6)'
            # '304.0263 \tEXP \tcoarse trial 20, showing face,face.pos = [-5.  5.], AI_data.shape = (173761, 6)'
            # '305.1594 \tEXP \tcoarse trial 20, hiding face, coarse_oculomatic_calib_values_candidate = ' + \
            #   '[-3.3869004   0.05144549] for face.pos = [-5.  5.], AI_data.shape = (174405, 6)'
            # '316.3800 \tEXP \tcoarse eye-tracking calibration end, coarse_stims_pos = [[ 5 -5]'
            if re.match(pattern_coarse, line) is not None:
                g = re.match(pattern_coarse, line).groups()

                if g[0].replace(' ', '') == 'eye-trackingcalibration':
                    if g[1] == 'start':
                        if 'n_trials' not in ecdata['crse']:
                            ecdata['crse']['n_trials'] = -1
                            ecdata['crse']['AIrng'] = [None, None]
                        ecdata['crse']['AIrng'][0] = tmp_AIidx
                    elif g[1] == 'end':
                        if tmp_AIidx is None:
                            tmp_AIidx = max(max([ecdata['crse']['data'][i]['AIrng']
                                                 for i, d in enumerate(ecdata['crse']['data'])]))
                        ecdata['crse']['AIrng'][1] = tmp_AIidx
                elif g[0] == 'trial':
                    trl = int(g[1])
                    if ecdata['crse']['n_trials'] - 1 < trl:
                        ecdata['crse']['n_trials'] = trl + 1
                        ecdata['crse']['data'][trl] = {'AIrng': [None, None], 'pos': None, 'cvals': None}
                    if g[2].replace(' ', '') == 'showingface':
                        ecdata['crse']['data'][trl]['AIrng'][0] = tmp_AIidx
                        ecdata['crse']['data'][trl]['pos'] = tmp_pos
                    elif g[2].replace(' ', '') == 'hidingface':
                        ecdata['crse']['data'][trl]['AIrng'][1] = tmp_AIidx
                        ecdata['crse']['data'][trl]['cvals'] = tmp_cvals
                    elif g[2] == 'candidate':
                        continue

        case 'circ':
            pattern_circ = r'.*circular\ *trajectory\ *(calibration|trial)\ *(start|end|\d+),?\ *(start|turn|end)?' + \
                           r',?\ *(faceID\ *=\ *)?(\d+)?\ *(start|end)?,*.*'
            # '316.3927 \tEXP \tcircular trajectory trial 0 start, faceID = 9, AI_data.shape = (180857, 6)'
            # '325.4014 \tEXP \tcircular trajectory trial 0, turn 3 start, AI_data.shape = (186039, 6)'
            # '328.4045 \tEXP \tcircular trajectory trial 0, turn 3 end, AI_data.shape = (187767, 6)'
            # '328.4045 \tEXP \tcircular trajectory trial 0 end, AI_data.shape = (187767, 6)'
            if re.match(pattern_circ, line) is not None:
                g = re.match(pattern_circ, line).groups()
                if g[0] == 'calibration':
                    if g[1] == 'start':
                        if 'n_trials' not in ecdata['circ']:
                            ecdata['circ']['n_trials'] = -1
                            ecdata['circ']['AIrng'] = [None, None]
                        ecdata['circ']['AIrng'][0] = tmp_AIidx
                    elif g[1] == 'end':
                        ecdata['circ']['AIrng'][1] = tmp_AIidx
                elif g[0] == 'trial':
                    trl = int(g[1])
                    if ecdata['circ']['n_trials'] - 1 < trl:
                        ecdata['circ']['n_trials'] = trl + 1
                        ecdata['circ']['data'][trl] = {'n_turns': -1, 'AIrng': [None, None], 'stim': None}
                    if g[2] == 'start':
                        ecdata['circ']['data'][trl]['AIrng'][0] = tmp_AIidx
                        ecdata['circ']['data'][trl]['stim'] = g[3].replace(' ', '').replace('=', '').strip() + g[4]
                    elif g[2] == 'turn':
                        trn = int(g[4])
                        if ecdata['circ']['data'][trl]['n_turns'] - 1 < trn:
                            if g[5] == 'start':
                                ecdata['circ']['data'][trl][trn] = {'AIrng': [None, None]}
                                ecdata['circ']['data'][trl][trn]['AIrng'][0] = tmp_AIidx
                            elif g[5] == 'end':
                                ecdata['circ']['data'][trl][trn]['AIrng'][1] = tmp_AIidx
                                ecdata['circ']['data'][trl]['n_turns'] = trn + 1
                    elif g[2] == 'end':
                        ecdata['circ']['data'][trl]['AIrng'][1] = tmp_AIidx

        case 'grdf':
            pattern_gridf = r'.*grid\ *faces?\ *(calibration|trial)\ *(start|end|\d+),?\ *(face|ISI)?\ *' + \
                            r'(start|end)?,?.*'
            # '388.5401 \tEXP \tgrid face trial 0, ISI start, AI_data.shape = (222368, 6)',
            # '389.0546 \tEXP \tgrid face trial 0, ISI end, AI_data.shape = see next entry',
            # '389.0546 \tEXP \tgrid face trial 0, face start, face.pos = [ 0. -5.], AI_data.shape = (222662, 6)'
            # '392.0783 \tEXP \tgrid face trial 0, face end, AI_data.shape = see next entry'
            # '392.0783 \tEXP \tgrid face trial 1, ISI start, AI_data.shape = (224406, 6)'
            if re.match(pattern_gridf, line) is not None:
                g = re.match(pattern_gridf, line).groups()

                # Replace 'sep' character placeholder with AI range value.
                typs = ['isi', 'face']
                for typ in typs:
                    rngs = [ecdata['grdf']['data'][i][typ]['AIrng'] if ecdata['grdf']['data'][i][typ] else [None, None]
                            for i, d in enumerate(ecdata['grdf']['data'])]
                    if np.any(np.array(rngs) == chr(31)):
                        septs = np.where(np.array(rngs) == chr(31))[0]
                        for sti in septs:
                            sepi = np.argwhere(np.array(ecdata['grdf']['data'][sti][typ]['AIrng']) == chr(31))[0][0]
                            ecdata['grdf']['data'][sti][typ]['AIrng'][sepi] = tmp_AIidx

                if g[0] == 'calibration':
                    if g[1] == 'start':
                        if 'n_trials' not in ecdata['grdf']:
                            ecdata['grdf']['n_trials'] = -1
                            ecdata['grdf']['AIrng'] = [None, None]
                        ecdata['grdf']['AIrng'][0] = tmp_AIidx
                    elif g[1] == 'end':
                        ecdata['grdf']['AIrng'][1] = tmp_AIidx
                elif g[0] == 'trial':
                    trl = int(g[1])
                    if ecdata['grdf']['n_trials'] - 1 < trl:
                        ecdata['grdf']['n_trials'] = trl + 1
                        ecdata['grdf']['data'][trl] = {'isi': {'AIrng': [None, None]},
                                                       'face': {'AIrng': [None, None], 'pos': None}}
                    if g[2] == 'face':
                        if g[3] == 'start':
                            ecdata['grdf']['data'][trl]['face']['AIrng'][0] = tmp_AIidx
                            ecdata['grdf']['data'][trl]['face']['pos'] = tmp_pos
                        elif g[3] == 'end':
                            ecdata['grdf']['data'][trl]['face']['AIrng'][1] = tmp_AIidx
                    elif g[2] == 'ISI':
                        if g[3] == 'start':
                            ecdata['grdf']['data'][trl]['isi']['AIrng'][0] = tmp_AIidx
                        elif g[3] == 'end':
                            ecdata['grdf']['data'][trl]['isi']['AIrng'][1] = tmp_AIidx

        case 'grdt':
            pattern_gridt = r'.*grid\ *target\ *(eye-tracking\ *calibration|trial)\ *(start|end|\d+),?\ *' + \
                            r'(ISI|central\ *target|grid\ *target|face\ *reward)?\ *(start|fixation|end)?,?\ *' + \
                            r'(start|interrupted|completed|fixation\ *success|fixation\ *fail)?,?.*'
            # '579.5631 \tEXP \tgrid target eye-tracking calibration start, AI_data.shape = (332239, 6)'
            # '579.5829 \tEXP \tgrid target trial 0, ISI start, AI_data.shape = (332239, 6)',
            # '580.5769 \tEXP \tgrid target trial 0, ISI end, AI_data.shape = see next entry',
            # '580.5769 \tEXP \tgrid target trial 0, central target start, AI_data.shape = (332820, 6)',
            # '580.9177 \tEXP \tgrid target trial 0, central target fixation start, AI_data.shape = (333014, 6)',
            # '580.9247 \tEXP \tgrid target trial 0, central target fixation interrupted, AI_data.shape = (333018, 6)',
            # '583.6077 \tEXP \tgrid target trial 1, central target start, AI_data.shape = (334556, 6)',
            # '585.4010 \tEXP \tgrid target trial 1, central target end, fixation success, AI_data.shape = see next entry',
            # '585.4149 \tEXP \tgrid target trial 1, grid target start, grid_target.pos = [5. 0.], AI_data.shape = (335589, 6)',
            # '585.4177 \tEXP \tgrid target trial 1, grid target fixation start, grid_target.pos = [5. 0.], AI_data.shape = (335596, 6)',
            # '585.5225 \tEXP \tgrid target trial 1, grid target fixation completed, grid_target.pos = [5. 0.], AI_data.shape = (335652, 6)',
            # '585.5332 \tEXP \tgrid target trial 1, grid target end, fixation success, AI_data.shape = see next entry',
            # '585.5332 \tEXP \tgrid target trial 1, face reward start, face.pos = [5. 0.], AI_data.shape = (335657, 6)',
            # '586.0616 \tEXP \tgrid target trial 1, face reward end, AI_data.shape = see next entry',
            if re.match(pattern_gridt, line) is not None:
                g = re.match(pattern_gridt, line).groups()

                # Replace 'sep' character placeholder with AI range value.
                typs = ['isi', 'ctr', 'targ', 'rwrd']
                for typ in typs:
                    rngs = [ecdata['grdt']['data'][i][typ]['AIrng'] if ecdata['grdt']['data'][i][typ] else [None, None]
                            for i, d in enumerate(ecdata['grdt']['data'])]
                    if np.any(np.array(rngs) == chr(31)):
                        septs = np.where(np.array(rngs) == chr(31))[0]
                        for sti in septs:
                            sepi = np.argwhere(np.array(ecdata['grdt']['data'][sti][typ]['AIrng']) == chr(31))[0][0]
                            ecdata['grdt']['data'][sti][typ]['AIrng'][sepi] = tmp_AIidx

                if g[0].replace(' ', '') == 'eye-trackingcalibration':
                    if g[1] == 'start':
                        if 'n_trials' not in ecdata['grdt']:
                            ecdata['grdt']['n_trials'] = -1
                            ecdata['grdt']['AIrng'] = [None, None]
                        ecdata['grdt']['AIrng'][0] = tmp_AIidx
                    elif g[1] == 'end':
                        ecdata['grdt']['AIrng'][1] = tmp_AIidx
                elif g[0] == 'trial':
                    trl = int(g[1])
                    if ecdata['grdt']['n_trials'] - 1 < trl:
                        ecdata['grdt']['n_trials'] = trl + 1
                        ecdata['grdt']['data'][trl] = {'isi': {'AIrng': [None, None]},
                                                       'ctr': None,
                                                       'targ': None,
                                                       'rwrd': None}
                    if g[2].replace(' ', '') == 'centraltarget':
                        if g[3] == 'start':
                            ecdata['grdt']['data'][trl]['ctr'] = {'AIrng': [None, None], 'success': None}
                        elif g[3] == 'fixation':
                            if g[4] == 'start':
                                ecdata['grdt']['data'][trl]['ctr']['AIrng'][0] = tmp_AIidx
                            elif g[4] == 'interrupted':
                                ecdata['grdt']['data'][trl]['ctr']['AIrng'][1] = tmp_AIidx
                            elif g[4] == 'completed':
                                ecdata['grdt']['data'][trl]['ctr']['AIrng'][1] = tmp_AIidx
                        elif g[3] == 'end':
                            if g[4].replace(' ', '') == 'fixationsuccess':
                                ecdata['grdt']['data'][trl]['ctr']['success'] = True
                            elif g[4].replace(' ', '') == 'fixationfail':
                                ecdata['grdt']['data'][trl]['ctr']['success'] = False
                    elif g[2].replace(' ', '') == 'gridtarget':
                        if g[3] == 'start':
                            ecdata['grdt']['data'][trl]['targ'] = {'AIrng': [None, None], 'pos': None, 'success': None}
                            ecdata['grdt']['data'][trl]['targ']['pos'] = tmp_pos
                        elif g[3] == 'fixation':
                            if g[4] == 'start':
                                ecdata['grdt']['data'][trl]['targ']['AIrng'][0] = tmp_AIidx
                            elif g[4] == 'interrupted':
                                ecdata['grdt']['data'][trl]['targ']['AIrng'][1] = tmp_AIidx
                            elif g[4] == 'completed':
                                ecdata['grdt']['data'][trl]['targ']['AIrng'][1] = tmp_AIidx
                        elif g[3] == 'end':
                            if g[4].replace(' ', '') == 'fixationsuccess':
                                ecdata['grdt']['data'][trl]['targ']['success'] = True
                            elif g[4].replace(' ', '') == 'fixationfail':
                                ecdata['grdt']['data'][trl]['targ']['success'] = False
                    # '588.9046 \tEXP \tgrid target trial 2, face reward start, face.pos = [0. 0.], AI_data.shape = (337584, 6)',
                    # '589.4260 \tEXP \tgrid target trial 2, face reward end, AI_data.shape = see next entry',
                    elif g[2].replace(' ', '') == 'facereward':
                        if g[3] == 'start':
                            ecdata['grdt']['data'][trl]['rwrd'] = {'AIrng': [None, None], 'pos': None}
                            ecdata['grdt']['data'][trl]['rwrd']['AIrng'][0] = tmp_AIidx
                            ecdata['grdt']['data'][trl]['rwrd']['pos'] = tmp_pos
                        elif g[3] == 'end':
                            ecdata['grdt']['data'][trl]['rwrd']['AIrng'][1] = tmp_AIidx
                    elif g[2] == 'ISI':
                        if g[3] == 'start':
                            ecdata['grdt']['data'][trl]['isi']['AIrng'][0] = tmp_AIidx
                        elif g[3] == 'end':
                            ecdata['grdt']['data'][trl]['isi']['AIrng'][1] = tmp_AIidx
        case _:
            continue
    del g_AIidx, g_pos, g_cvals, tmp_AIidx, tmp_pos, tmp_cvals


# Add analog eye tracking data.
def populate_eyetrack_data(logdict, targ, eyedata):
    for key, val in logdict.items():
        if isinstance(val, dict):
            populate_eyetrack_data(val, targ, eyedata)
        elif key == targ:
            if None not in val:
                s, e = val
                logdict['AIdata'] = eyedata[s:e]


ecdf


# % Extract stimulus information from log file

# *** TODO load from a pandas dataframe instead of a text log

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
    dp = dprime[ridx]
    fig = plt.figure()
    fig.suptitle('ROI {} dprime={:0.2f}: mean response by category (each cond mean plotted)'.format(ridx, dp))
    axes = fig.subplots(nrows=n_metrics, ncols=n_cats)
    for m, met in enumerate(metrics):
        ymin = np.min(np.mean(data[met][:, ridx, :, :], axis=1))
        ymax = np.max(np.mean(data[met][:, ridx, :, :], axis=1))
        for cat in range(n_cats):
            ax = axes[m, cat]
            if m == 0:
                ax.set_title(tmpl_labels[categories[cat]])
            if cat == 0:
                ax.set_ylabel(metric_labels[met])
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
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[sort_dp] < -threshold_dprime)[0].min(), color='0.2', linestyle='dotted', linewidth=1)
        ax_dp.axhline(np.where(dprime[sort_dp] > threshold_dprime)[0].max(), color='0.2', linestyle='dotted', linewidth=1)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[sort_dp], threshold_dprime), atol=0.05), color='0.2', linestyle='dotted', linewidth=1)

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
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[sort_dp] < -threshold_dprime)[0].min(), color='0.2', linestyle='dotted', linewidth=1)
        ax_dp.axhline(np.where(dprime[sort_dp] > threshold_dprime)[0].max(), color='0.2', linestyle='dotted', linewidth=1)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[sort_dp], threshold_dprime, atol=0.05)), color='0.2', linestyle='dotted', linewidth=1)
    
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

# Plot for each ROI the category of the condition (image) eliciting the largest response
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response, z > 0.5', save_path=sp)

# Plot for each ROI the category of the condition (image) eliciting the largest response
above_threshold = np.where(np.abs(FSIs_zsc) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingConditionImage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response, '
                              + 'FSI > {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

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


# Plot for each ROI the category eliciting the largest average response
above_threshold = np.where(ROIinfo[:]['top_cond_Fzsc'] > 0.5)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclZgt0p5' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category eliciting the largest average response, z > 0.5', save_path=sp)

# Plot for each ROI the category eliciting the largest average response
# for only ROIs with FSI > threshold
above_threshold = np.where(np.abs(FSIs_zsc) > threshold_fsi)[0]
ROI_colors = np.array([colorsys.hsv_to_rgb(tci, 1.0, 1.0) for tci in top_cat_mean_idn])
sn = save_pfix + '_ROIplot_ColorByCategoryOfMostActivatingCategoryOnAverage_inclFSIthrs' + save_ext
sp = os.path.join(save_path, sn) if saving else ''
plots.plot_roi_overlays(ROIs[above_threshold], ROI_colors[above_threshold],
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='category eliciting the largest average response, '
                              + 'FSI > {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

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
                        image=plots.auto_level_s2p_image(fov_image), flip='lr', rotate=-90,
                        title='relative response strength, z > 0.5', save_path=sp)

above_threshold = np.where(FSIs_zsc > threshold_fsi)[0]
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

