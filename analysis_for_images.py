#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from glob import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.stats import binned_statistic, kendalltau, spearmanr
import socket
from warnings import warn

import filters
from metadata import default_metadata
import parsers
import plots


# TODO: add consistent pyplot theme handling for plots https://github.com/raybuhr/pyplot-themes
#       including colorblind palette options https://personal.sron.nl/~pault/
# TODO: exclude based on eye tracking, at least when eyes are not open


# %% Settings

# dprime_F threshold
# based on Shi et al Tsao bioRxiv (Fig 1g, https://doi.org/10.1101/2023.12.06.570341):
# "The dotted vertical line marks d’ = 0.2, which we used as our threshold for identifying face-selective units."
threshold_dprime = 0.2

# Face-Selectivity Index (FSI) threshold
# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci (https://doi.org/10.1038/nn.2363):
# """
# For |face-selectivity index| = 1/3, that is, if the response to faces was at least twice (or 
# at most half) that of nonface objects, a cell was classed as being face selective.
# [...] neurons (94%) were face selective (that is, face-selectivity index larger than 1/3 or 
# smaller than -1/3).
# """
threshold_fsi = 1 / 3

threshold_cellprob = 0.0
threshold_Zscore = 0.5

# Exclusion criteria
exclude_by_movement = True

# Plotting parameters
plot_eyecal = False
plt.rcParams['figure.dpi'] = 300

# Metrics to consider for plots and calculations
metrics = ['FdFF', 'Fzsc']
metric_labels = {'FdFF': 'dF/F',
                 'Fzsc': 'Z-score'}

# Template establishing the ordering and labeling for plots
template = np.array([b'blank', b'scram_s', b'scram_p',
                     b'face_mrm', b'face_rhe', b'face_hum', b'face_ctn',
                     b'obj', b'food',
                     b'body_mrm', b'animal'], dtype='|S8')
template_labels = {b'blank': 'B',  # 'Blank',
                   b'scram_s': 'S',  # 'Scrambles (Spatial)',
                   b'scram_p': 'P',  # 'Scrambles (Phase)',
                   b'face_mrm': 'mrmFaces',  # 'Faces (Marmo)',
                   b'face_rhe': 'R',  # 'Faces (Rhesus)',
                   b'face_hum': 'H',  # 'Faces (Human)',
                   b'face_ctn': 'Ctn',  # 'Faces (Cartoon)',
                   b'obj': 'Objects',
                   b'food': 'Foods',
                   b'body_mrm': 'mrmBods',  # 'Bodies (Marmo)',
                   b'animal': 'Animals'}

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

# # 20220909d
# date_str = '20220909d'
# -- PD 200um (acquisition ended before stimulus)
# session_str = '174325tUTC_SP_depth200um_fov0600x0600um_res1p04x1p00umpx_fr09p608Hz_pow050p2mW_stimImagesSongFOBonly_AbortedAfterManyTrials'
# md['framerate'] = 9.608
# md['fov'] = dict()
# md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
# md['fov']['w_px'] = 600
# md['fov']['h_px'] = 600
# dirstr_suite2p = 'suite2p_cellpose2_d14px_pt-3p5_ft1p5*'

# # 20221016d
# date_str = '20221016d_olds2p'
date_str = '20221016d'
# -- PD (good | fixation spot on before every stimulus, not sure about background stability)
session_str = '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'
md = dict()
md['framerate'] = 6.364
md['fov'] = dict()
md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
md['fov']['w_px'] = 730
md['fov']['h_px'] = 730
# dirstr_suite2p = 'suite2p_old*'
dirstr_suite2p = 'suite2p_cellpose2_d14px_pt-3p5_ft1p5*'

# # 20230510d (300-400um fluid)
# date_str = '20230510d'
# # -- PD (bad | z drift, two stims only, imaging ended before stimulus)
# session_str = '141055tUTC_SP_depth200um_fov2190x2000um_res3p00x3p02umpx_fr06p993Hz_pow059p9mW_stimImagesSong2imTest'
# # -- OBJ? (okay? | half FOV dim) *some interesting object responses, look again*
# session_str = '155713tUTC_SP_depth200um_fov2190x2000um_res3p00x3p02umpx_fr06p993Hz_pow059p9mW_stimImagesSong230509dSel'

# # 20230809d (negligible fluid)
# date_str = '20230809d'
# # -- OBJ (okay? | movement exclusions needed, half FOV dim)
# session_str = '173936tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'

# # 20231001d (~430um fluid)
# date_str = '20231001d'
# # -- PD (not very bright, movement exclusions needed, duplicate cond issue) ... but maybe look more closely later
# session_str = '190608tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p3mW_stimImagesFOBmany'
# # -- PD (not very bright, movement exclusions needed) ... but maybe look more closely later
# session_str = '200422tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow060p3mW_stimImagesFOBmin'

# # 20231003d (~140um fluid)
# date_str = '20231003d'
# # -- PD 200um ( movement exclusions needed)
# session_str = '142836tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# # -- PD 150um ( )
# session_str = '145031tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p0mW_stimImagesFOBmin'
# # -- PD 200um ( )
# session_str = '153340tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# # -- PD 250um ( )
# session_str = '154955tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow089p8mW_stimImagesFOBmin'
# # -- MD 150um ( )
# session_str = '162025tUTC_SP_depth150um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p3mW_stimImagesFOBmin'
# # -- MD 200um ( )
# session_str = '163738tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow079p9mW_stimImagesFOBmin'
# # -- MD 200um ( )
# session_str = '165634tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow079p9mW_stimImagesFOBmin'
# # -- OBJ 200um ( )
# session_str = '173850tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow070p3mW_stimImagesFOBmin'

# # 20231007d (~120um fluid)
# date_str = '20231007d'
# # -- OBJ 200um (decent/good? | )
# session_str = '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'
# # -- OBJ 150um ( )
# session_str = '162705tUTC_SP_depth150um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow049p8mW_stimImagesFOBmin'
# # -- OBJ 250um ( )
# session_str = '164258tUTC_SP_depth250um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow069p8mW_stimImagesFOBmin'
# # -- OBJ 300um ( )
# session_str = '170046tUTC_SP_depth300um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow099p7mW_stimImagesFOBmin'
# # -- PD 200um ( )
# session_str = '174147tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow080p2mW_stimImagesFOBmin'
# # -- PD 200um ( | eyetrack crashed)
# session_str = '180407tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p359Hz_pow080p2mW_stimMultimodal'

# # 20231018d
# date_str = '20231018d'
# # -- OBJ 200um ( )
# session_str = '185135tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# # -- OBJ ( )
# session_str = '190745tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow075p8mW_stimImagesFOBmin'
# # -- OBJ ( )
# session_str = '192426tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Cashew
# animal_str = 'Cashew'

# # 20230728d (negligible fluid)
# date_str = '20230728d'
# # -- PD 200um ( | aniso, headpost loose)
# session_str = '135645tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p9mW_stimImagesFOBminAniso'
# # -- PD 200um ( | aniso, headpost loose, acquisition ended before stimulus)
# session_str = '141430tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow059p9mW_stimImagesFOBminAniso'
# # -- PD 200um ( | )
# session_str = '144334tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p9mW_stimImagesFOBmin'
# # -- PD 200um ( | drowsy, YouTube audio)
# session_str = '150131tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow059p9mW_stimImagesFOBmin'

# # 20230804d
# date_str = '20230804d'
# # -- PD 200um (aborted | juice and eyetracking issues)
# session_str = '150809tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p7mW_stimImagesFOBmany_Aborted
# # -- PD+OBJ 200um ( | )
# session_str = '154547tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p7mW_stimImagesFOBmany'
# # -- PD 200um ( | )
# session_str = '164834tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow059p7mW_stimImagesFOBmin'

# # 20230809d (~372um fluid PD, ~538um fluid OBJ)
# date_str = '20230809d'
# # -- PD 200um ( | )
# session_str = '143713tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p6mW_stimImagesFOBmin'
# # -- PD 200um ( | )
# session_str = '150042tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'
# # -- OBJ 200um ( | )
# session_str = '154139tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'
# # -- MTish 200um ( | )
# session_str = '161136tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow049p8mW_stimImagesFOBmin'

# # 20230810d (~400um fluid PD)
# date_str = '20230810d'
# # -- PD 200um ( | )
# session_str = '165302tUTC_SP_depth200um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmin'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Curly
# animal_str = 'Curly'
# # 20231103d
# date_str = '20231103d'
# # -- MTish 200um ( | )
# session_str = '162355tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow049p9mW_stimImagesFOBmin'
# # -- PDish 200um ( | )
# session_str = '165202tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow049p9mW_stimImagesFOBmin'

# # 20231110d
# date_str = '20231110d'
# # -- MTadjacent 200um ( | )
# session_str = '170315tUTC_SP_depth200um_fov2555x2500um_res2p50x2p50umpx_fr03p349Hz_pow059p9mW_stimMultimodal'

# # 20231113d
# date_str = '20231113d'
# # -- MTish 200um ( | )
# session_str = '185712tUTC_SP_depth200um_fov2190x2000um_res2p50x2p50umpx_fr04p853Hz_pow059p8mW_stimMultimodal'

# # 20231117d
# date_str = '20231117d'
# # -- OBJ 200um ( | )
# session_str = '165535tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow059p9mW_stimMultimodal'
# # -- PVandOBJ 200um ( | )
# session_str = '181016tUTC_SP_depth200um_fov2190x2920um_res2p50x2p50umpx_fr03p355Hz_pow065p1mW_stimMultimodal'


# # 20240130d
# date_str = '20240130d'
# # -- ? 200um ( | )
# session_str = '172813tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow060p2mW_stimMultimodal'
# # -- ? 200um ( | )
# session_str = '192116tUTC_SP_depth200um_fov2555x2500um_res2p50x2p50umpx_fr03p349Hz_pow060p2mW_stimMultimodal

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Dali
# animal_str = 'Dali'

# # 20230509d Right
# date_str = '20230509d'
# # -- PD-OBJ 250um ( | inattentive)
# session_str = '181700tUTC_SP_depth250um_fov2190x2000um_res3p02x3p00umpx_fr06p952Hz_pow050p0mW_stimImagesSong230509dSel'

# # 20230511d Right (~158um dura to glass)
# date_str = '20230511d'
# # -- PD 200um ( | headpost loose)
# session_str = '134800tUTC_SP_depth200um_fov2190x2000um_res3p02x3p02umpx_fr06p993Hz_pow050p0mW_stimImagesSong230509dSel'
# # -- PD 300um ( | )
# session_str = '150200tUTC_SP_depth300um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow065p0mW_stimImagesSong230509dSel'

# # 20230515d Right
# date_str = '20230515d'
# # -- PDish 200um ( | )
# session_str = '135100tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'
# # -- PDish 200um ( | )
# session_str = '144500tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesSong230509dSel'

# # 20230517d Right
# date_str = '20230517d'
# # -- PDish 200um ( | aniso stim)
# session_str = '135600tUTC_SP_depth200um_fov2628x2600um_res3p02x3p00umpx_fr04p484Hz_pow050p0mW_stimImagesFOBsel230517dAniso'

# # 20230522d Right
# date_str = '20230522d'
# # -- PD-OBJ ( | aniso stim, acquisition ended before stimulus)
# session_str = '153415tUTC_SP_depth200um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow051p8mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ ( | ) *CHECK*
# session_str = '170053tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow051p8mW_stimImagesSong230509dSel'

# # 20230525d Right
# date_str = '20230525d'
# # -- PD-OBJ 150um ( | aniso stim)
# session_str = '155856tUTC_SP_depth150um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow039p8mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ 200um ( | aniso stim)
# session_str = '164300tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow052p0mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ 250um ( | aniso stim)
# session_str = '164300tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow052p0mW_stimImagesFOBsel230517dAniso'

# # 20230606d Right
# date_str = '20230606d'
# # -- PD-OBJ 300um ( | aniso stim)
# session_str = '135543tUTC_SP_depth300um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow091p0mW_stimImagesFOBsel230517dAniso'

# # 20230608d Right
# date_str = '20230608d'
# # -- PD-OBJ 350um ( | aniso stim)
# session_str = '124502tUTC_SP_depth350um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow092p7mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ 400um ( | aniso stim)
# session_str = '134220tUTC_SP_depth400um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow122p3mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ 450um ( | aniso stim)
# session_str = '142713tUTC_SP_depth450um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow140p0mW_stimImagesFOBsel230517dAniso'

# # 20230611d Right
# date_str = '20230611d'
# # -- PD-OBJ 500um ( | aniso stim)
# session_str = '155917tUTC_SP_depth500um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow149p9mW_stimImagesFOBsel230517dAniso'

# # 20230615d Right
# date_str = '20230615d'
# # -- PD-OBJ 200um ( | aniso stim)
# session_str = '154218tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow040p2mW_stimImagesFOBsel230517dAniso'

# # 20230618d Right
# date_str = '20230618d'
# # -- PD-OBJ 300um ( | aniso stim)
# session_str = '150314tUTC_SP_depth300um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow069p9mW_stimImagesFOBsel230517dAniso'
# # -- PD-OBJ 400um ( | aniso stim)
# session_str = '154357tUTC_SP_depth400um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow121p8mW_stimImagesFOBsel230517dAniso'

# # 20230704d Right
# date_str = '20230704d'
# # -- PD-OBJ 200um ( | aniso stim)
# session_str = '153542tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p0mW_stimImagesFOBminMarmAniso'

# # 20230804d Right
# date_str = '20230804d'
# # -- PD-OBJ 200um ( | )
# session_str = '180205tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow059p7mW_stimImagesFOBmin'

# # 20230810d Right
# date_str = '20230810d'
# # -- PD 200um ( | )
# session_str = '140728tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesFOBmany'
# # -- PD 200um ( | )
# session_str = '150108tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow060p1mW_stimImagesSongFOBonly'
# # -- PD 300um ( | )
# session_str = '152105tUTC_SP_depth300um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow070p3mW_stimImagesFOBmin'

# # 20230831d Right
# date_str = '20230831d'
# # -- PD 200um ( | )
# session_str = '180939tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p485Hz_pow059p7mW_stimImagesFOBmin'
# # -- PD-OBJ 200um ( | )
# session_str = '184618tUTC_SP_depth200um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow050p2mW_stimImagesFOBmin'
# # -- PD-OBJ 300um ( | )
# session_str = '190748tUTC_SP_depth300um_fov1825x1825um_res2p50x2p50umpx_fr06p363Hz_pow080p0mW_stimImagesFOBmin'

# # 20230906d Right
# date_str = '20230906d'
# # -- PD 150um ( | )
# session_str = '173359tUTC_SP_depth150um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow030p2mW_stimImagesFOBmin'
# # -- PD 250um ( | )
# session_str = '181552tUTC_SP_depth250um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow080p4mW_stimImagesFOBmin'

# # 20230910d Right
# date_str = '20230910d'
# # -- PD-OBJ 200um ( | acquisition ended before stimulus)
# session_str = '170833tUTC_SP_depth200um_fov2628x2600um_res3p00x3p00umpx_fr04p481Hz_pow060p4mW_stimImagesFOBmany'
# # -- PD-OBJ 300um ( | )
# session_str = '183115tUTC_SP_depth300um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow079p5mW_stimImagesFOBmin'
# # -- PD-OBJ 150um ( | )
# session_str = '185714tUTC_SP_depth150um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow039p8mW_stimImagesFOBmin'

# # 20240919d Left
# date_str = '20240919d'
# # -- PD-MT? 200um ( | acquisition ended before stimulus)
# session_str = '170917tUTC_SP_depth200um_fov1272x1800um_res2p00x2p00umpx_fr04p330Hz_pow040p0mW_stimImagesSongFOBonly_withContinuation'
# # -- PD? 200um ( | )
# session_str = '180330tUTC_SP_depth200um_fov0742x0700um_res1p00x1p00umpx_fr04p734Hz_pow040p0mW_stimImagesSongFOBonly'

# # 20240923d Left
# date_str = '20240923d'
# # -- PD? 200um ( | CHECK Z-drift?)
# session_str = '164457tUTC_SP_depth200um_fov0742x0700um_res1p00x1p00umpx_fr04p734Hz_pow049p4mW_stimImagesSongFOBonly'

# # 20240925d Left
# date_str = '20240925d'
# # -- PD-MT 200um ( | CHECK Z-drift?)
# session_str = '171829tUTC_SP_depth200um_fov1272x1800um_res2p00x2p00umpx_fr04p330Hz_pow040p1mW_stimMultimodalImagesTones'
# # -- PD? 200um ( | )
# session_str = '181617tUTC_SP_depth200um_fov0742x0700um_res1p00x1p00umpx_fr04p734Hz_pow040p1mW_stimImagesSongFOBonly'
# # -- PD? 200um ( | )
# session_str = '184703tUTC_SP_depth200um_fov0742x0700um_res1p00x1p00umpx_fr04p734Hz_pow040p1mW_stimImagesFOBmin'

# # 20240926d Left
# date_str = '20240926d'
# # -- PD? 200um ( | some struggles)
# session_str = '152716tUTC_SP_depth200um_fov1060x1000um_res2p00x2p00umpx_fr09p147Hz_pow040p1mW_stimImagesFOBmin'
# # -- PD? 200um ( | some struggles)
# session_str = '160947tUTC_SP_depth200um_fov1060x1000um_res2p00x2p00umpx_fr09p147Hz_pow040p1mW_stimImagesSongFOBonly'
# # -- PD-MT? 200um ( | check Z-drift?)
# session_str = '171222tUTC_SP_depth260um_fov1060x1800um_res2p00x2p00umpx_fr05p196Hz_pow050p4mW_stimMultimodalImagesDotsGratingsTonesVocalizations'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Larry
# animal_str = 'Larry'

# # 20231007d
# date_str = '20231007d'
# # -- MD 200um ( | movement exclusions needed)
# session_str = '203704tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow049p8mW_stimImagesFOBmin'
# # -- PD 200um ( | )
# session_str = '205532tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow049p8mW_stimImagesFOBmin'
# # -- PV 200um ( | movement exclusions needed)
# session_str = '215640tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow060p1mW_stimImagesFOBmin'
# # -- PV 200um ( | )
# session_str = '221343tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow060p1mW_stimImagesFOBmin'

# # 20231008d
# date_str = '20231008d'
# # -- MTish 200um ( | )
# session_str = '203515tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p349Hz_pow079p8mW_stimMultimodal'
# # -- MDish 200um ( | )
# session_str = '211620tUTC_SP_depth200um_fov3066x3000um_res3p00x3p00umpx_fr03p347Hz_pow079p8mW_stimMultimodal'

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
dirstr_suite2p_plane = 'plane0'


# %% Define functions and classes


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


# %% Find files and load data

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

if 'stimImagesSongFOBonly' in session_str:
    dirstr_stimset = 'Song_etal_Wang_2022_NatCommun|480288_equalized_RGBA_FOBonly'.replace('|', os.path.sep)
elif 'stimImagesSong2imTest' in session_str:
    dirstr_stimset = 'Song_etal_Wang_2022_NatCommun|480288_equalized_RGBA_selected20230509d'.replace('|', os.path.sep)
elif 'stimImagesSong230509dSel' in session_str:
    dirstr_stimset = 'Song_etal_Wang_2022_NatCommun|480288_equalized_RGBA_selected20230509d'.replace('|', os.path.sep)
elif 'stimImagesFOBmin' in session_str:
    dirstr_stimset = 'FOBmin|Images|20230728d'.replace('|', os.path.sep)
elif 'stimImagesFOBmany' in session_str:
    dirstr_stimset = 'FOBmany|Images|20230728d'.replace('|', os.path.sep)
elif 'stimImagesFOBsel230517dAniso' in session_str:
    dirstr_stimset = 'MarmosetFOB2018|20230517d|Scaled512x512_MaskEroded3_SHINEdLum_RGBA_reorg_subset_aniso'.replace('|', os.path.sep)
elif 'stimImagesFOBsel230517d' in session_str:
    dirstr_stimset = 'MarmosetFOB2018|20230517d|Scaled512x512_MaskEroded3_SHINEdLum_RGBA_reorg_subset'.replace('|', os.path.sep)
elif 'stimMultimodal' in session_str:
    warn('Displaying multimodal stimulus images not supported yet.')
    dirstr_stimset = None
else:
    warn('Could not determine stimulus set from session name.')
    dirstr_stimset = None

stimimage_path = os.path.join(stim_path, dirstr_stimset)
if not os.path.isdir(stimimage_path):
    warn('Could not find stimulus image source path.')
    stimimage_path = None

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
        stimlog = parsers.parse_log_stim_image(session_log)
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
        sl = parsers.parse_log_stim_image(session_log)
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
    if len(dirlist_suite2p) > 1:
        warn('Found multiple suite2p folders, using the first one: {}'.format(os.path.basename(dirlist_suite2p[0])))
    s2p_path = dirlist_suite2p[0]
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

# Fix off-by-one issue caused by acqusition frame counter ('acqfr') 
# indexed starting at 1 rather than 0.
acqfr_keys = [c for c in stimlog.columns if 'acqfr' in c]
for ak in acqfr_keys:
    stimlog[ak] = stimlog[ak] - 1

# Determine basic stimulus presentation information
#   For sessions using older stimulus code, the exact number of stimulus or ISI frames could
#   vary slightly because the stim start was not locked to an acqusition frame increment.
dur_stim = np.round(np.mean(stimlog['dur_stim'].values), 2)
dur_isi = np.round(np.min(stimlog['dur_isi_pre'].values), 2)
dur_trial = dur_isi + dur_stim + dur_isi
n_samp_stim = np.bincount(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i']).argmax()
if np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][0] != 0:
    n_samp_isi = np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][0]
else:
    n_samp_isi = np.bincount(stimlog['acqfr_isi_f'] - stimlog['acqfr_isi_i']).nonzero()[0][1]
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

# Calculate the timing mismatch (contraction) introduced by rounding stim and/or isi frame samples down
acqfr_dilation_factor = (dur_trial * md['framerate']) / (n_samp_trial - 1)

# *** TODO: take only conds with image presentations
if not stimlog[:]['stim_mode'].isnull().any():
    if np.unique(stimlog[:]['stim_mode'].values).size != 1:
        warn('More than one stimulus mode was presented and is not yet fully supported.')
if not stimlog[:]['stim_class'].isnull().any():
    if np.unique(stimlog[:]['stim_class'].values).size != 1:
        warn('More than one stimulus class was presented and is not yet fully supported.')
if not stimlog[:]['stim_subclass'].isnull().any():
    if np.unique(stimlog[:]['stim_subclass'].values).size != 1:
        warn('More than one stimulus subclass was presented and is not yet fully supported.')

n_metrics = len(metrics)
n_conds = len(np.unique(stimlog['cond'].values))
n_trials = len(stimlog)
n_reps = int(len(stimlog) / n_conds)

if len(np.unique(stimlog['acqfr_stim_i'])) != len(stimlog['acqfr_stim_i']):
    warn('Acquisition was started after stimulus, interrupted, or stopped before stimulus.')

# Identify stimulus image set from file paths
# *** TODO: make this work on a per-cond/image level
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


# %% Organize fluorescence signals into a structured array data table

dlist = [('cond', 'S8'),
         ('stimulus', object),
         ('cat', 'S8'),
         ('id', 'S8'),
         ('pitch', 'i2'),
         ('yaw', 'i2'),
         ('roll', 'i2'),
         ('imagename', np.unicode_, 256)]

for m in metrics:
    dlist.append((m, 'f8', (n_ROIs, n_reps, n_samp_trial)))
del m

data = np.zeros(n_conds, dtype=dlist)
del dlist

for m in metrics:
    data[m] = np.nan
del m


# 'cond': None,
# 'stim_mode': None,
# 'stim_class': None,
# 'stim_subclass': None,

# 'image': None,
# 'image_path': None,

# 'units': None,
# 'pos': None,
# 'size': None,
# 'ori': None,
# 'color': None,
# 'colorSpace': None,
# 'contrast': None,
# 'opacity': None,
# 'texRes': None,

# Currently supported image sets:
# 'FOBmin_MarmOnly', 'FOBmin', 'FOBmany', 'Song_etal_Wang_2022_FOBonly'

trials_movement = []
for c in range(n_conds):
    # stimlog[stimlog['cond'] == c]['stim_class']
    tmp_cond = None
    tmp_cat = None
    tmp_id = None
    tmp_pitch = np.iinfo(np.int16).min
    tmp_yaw = np.iinfo(np.int16).min
    tmp_roll = np.iinfo(np.int16).min
    if np.unique(stimlog[stimlog['cond'] == c]['image'].values).size == 1:
        tmp_imagename = np.unique(stimlog[stimlog['cond'] == c]['image'].values)[0]
        imn = os.path.splitext(tmp_imagename)[0]
    else:
        warn('Not all images were the same for condition {}.'.format(c))
        tmp_imagename = ''
        imn = ''
    tmp_imagepath = os.path.join(stimimage_path, tmp_imagename) \
        if os.path.isfile(os.path.join(stimimage_path, tmp_imagename)) else None
    if image_set == 'FOBmin' or image_set == 'FOBmany':
        pattern_imn = r'^(Freiwald(FOB)?([0-9]*)?)?_?([^_]+)_([^_]+)_?([^_]+)?_([0-9]+)_?' + \
                      r'([^_]*erode[^_]*)?_?(inverted)?$'
        if re.match(pattern_imn, imn) is not None:
            sp = re.match(pattern_imn, imn).group(4)
            ct = re.match(pattern_imn, imn).group(5)
            di = re.match(pattern_imn, imn).group(6)
            nm = re.match(pattern_imn, imn).group(7)
            ed = 'e' if re.match(pattern_imn, imn).group(8) is not None else ''
            if nm.isnumeric():
                nm = float(nm)
                if nm.is_integer():
                    nm = int(nm)
                else:
                    warn('View index in filename incorrect ({}). '.format(tmp_imagename) +
                         'Expected integer, not float.')
            iv = re.match(pattern_imn, imn).group(9) is not None
            match sp:
                case 'Human':
                    if ct == 'Head':
                        tmp_cond = bytes('fh{:02}{}'.format(nm, ed), 'ascii')
                        tmp_cat = b'face_hum'
                        tmp_id = bytes('Hum{:02}{}'.format(nm, ed), 'ascii')
                        tmp_pitch = 0
                        tmp_roll = 0
                        tmp_yaw = 0
                case 'MacaqueRhesus':
                    if ct == 'Head':
                        tmp_cond = bytes('fr{:02}{}'.format(nm, ed), 'ascii')
                        tmp_cat = b'face_rhe'
                        tmp_id = bytes('Rhe{:02}{}'.format(nm, ed), 'ascii')
                        tmp_pitch = 0
                        tmp_roll = 0
                        tmp_yaw = 0
                case 'Marm':
                    if ct == 'Head':
                        if iv is True:
                            nm = 9
                        tmp_cond = bytes('fm{}{:02}{}'.format(di[0:3], nm, ed), 'ascii')
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
                        tmp_cond = bytes('bm{}{:02}{}'.format(di[0:3], nm, ed), 'ascii')
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
                        tmp_cond = bytes('om{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
                        tmp_cat = b'obj'
                    elif 'FruitVeg' in ct:
                        tmp_cond = bytes('vf{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
                        tmp_cat = b'food'
                    elif 'MultipartGeon' in ct:
                        tmp_cond = bytes('og{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
                        tmp_id = bytes('Geon{:01}'.format(ct_p2), 'ascii')
                        tmp_cat = b'obj'
                    elif 'Pairwise' in ct:
                        tmp_cond = bytes('op{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
                        tmp_cat = b'obj'
                    elif 'String' in ct:
                        tmp_cond = bytes('os{:01}{:03}{}'.format(ct_p2, nm, ed), 'ascii')
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
del tmp_cond, tmp_cat, tmp_id, tmp_pitch, tmp_yaw, tmp_roll, tmp_imagename, tmp_imagepath

# Sanity check for NaN values after loading data
for m in metrics:
    if np.any(np.isnan(data[m])):
        raise Exception('Found NaNs after loading {} data.'.format(m))
del m


# %% Sort data table according to template, define categories and conditions, and perform exclusions

sort_by_cond = lambda x: (np.where(template == x[1].category)[0][0]
                          if np.where(template == x[1].category)[0].size > 0
                          else np.iinfo(np.where(template == x[1].category)[0].dtype).max,
                          np.abs(x[1].roll),
                          x[1].roll,
                          x[1].yaw,
                          x[1].condition.decode().lower())
sort_by_cat = lambda x: (np.where(template == x[1])[0][0]
                         if np.where(template == x[1])[0].size > 0
                         else np.iinfo(np.where(template == x[1])[0].dtype).max)

data = data[[i for i, _ in sorted(enumerate(data['stimulus']), key=sort_by_cond)]]

categories = pd.unique(data['cat'])  # Use pandas instead of numpy to avoid automatic sorting
# categories = categories[[i for i, _ in sorted(enumerate(categories), key=sort_by_cat)]]
cat_to_catidx = {k: i for i, k in enumerate(categories)}
n_cats = len(categories)
conditions = pd.unique(data['cond'])  # Use pandas instead of numpy to avoid automatic sorting
# conditions = conditions[[i for i, _ in sorted(enumerate(conditions), key=sort_by_cond)]]
cond_to_condidx = {k: i for i, k in enumerate(conditions)}
cat_to_cond = {cat: [cnd for cnd in data[data['cat'] == cat]['cond']] 
               for cat in categories}
cat_to_condidx = {cat: [cond_to_condidx[cnd] for cnd in data[data['cat'] == cat]['cond']] 
                  for cat in categories}
cond_to_cat = {cnd: cat for cat, cndlist in cat_to_cond.items() for cnd in cndlist}
condidx_to_cat = {icnd: cat for cat, icndlist in cat_to_condidx.items() for icnd in icndlist}

if n_conds != conditions.shape[0]:
    u, c = np.unique(data['cond'], return_counts=True)
    mult = u[c > 1]
    warn('Some different image files were combined into the same condition. ' +
         '{}'.format(data[data['cond'] == mult]['imagename']))
    warn('May not be able to handle that yet...')
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
    if np.where(stimlog['acqfr_stim_f'] - stimlog['acqfr_stim_i'] < n_samp_stim):
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
        if acqfr_diff.is_integer():
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


# %% Plot across-stimulus population responses (similar to PSTH)
#    across-stimulus mean of trial-averaged population (ROI-averaged) responses

fig_psth = plt.figure()
fig_psth.suptitle('across-stimulus mean of trial-averaged population responses')
axes = fig_psth.subplots(nrows=n_metrics, ncols=1)
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * md['framerate'])
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
for mi, m in enumerate(metrics):
    ymean = np.mean(np.array([np.mean(np.nanmean(data[cat_to_condidx[c]][m], axis=(1, 2)), axis=0)
                              for c in categories]))
    ymin = ymean
    ymax = ymean
    ax = axes[mi]
    ax.set_ylabel(metric_labels[m])
    ax.tick_params(axis='both', which='major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xticks = [x * md['framerate'] for x in range(np.ceil(dur_trial).astype('int') + 1)]
    xticklabels = ['' if not np.isclose(xt, dur_isi * md['framerate']) 
                   and not np.isclose(xt, (dur_isi + dur_stim) * md['framerate'])
                   else '{}'.format(np.round(xt / md['framerate']).astype('int')) for xt in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.axvspan(dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate'], color='0.9', zorder=0)
    ax.set_xlim((0, np.ceil(dur_trial) * md['framerate']))
    ax.plot(xs,
            np.mean(np.nanmean(data[m], axis=(1, 2)), axis=0), 
            label='All', 
            color='0', linestyle='dotted', linewidth=1, zorder=4)
    for cati, cat in enumerate(categories):
        # if cat == b'blank':
        #     continue
        n_cnd_in_cat = (data[cat_to_condidx[cat]]['cond'].shape[0])
        Fmean = np.mean(np.nanmean(data[cat_to_condidx[cat]][m], axis=(1, 2)), axis=0)
        Fsem = np.std(np.nanmean(data[cat_to_condidx[cat]][m], axis=(1, 2)), axis=0) / np.sqrt(n_cnd_in_cat)
        ymin = np.min([ymin, np.min(Fmean - Fsem)])
        ymax = np.max([ymax, np.max(Fmean + Fsem)])
        ax.plot(xs, 
                Fmean,  # 'o-', markersize=2,
                label=template_labels[cat], 
                color=colorsys.hsv_to_rgb(cati / n_cats, 1.0, 1.0), zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, 
                        color=colorsys.hsv_to_rgb(cati / n_cats, 1.0, 1.0), alpha=0.1, zorder=2)
    ax.set_ylim((ymin - 0.05 * np.abs(ymax - ymin), ymax + 0.05 * np.abs(ymax - ymin)))
    del cati, cat
    # ax.legend(frameon=False, loc=(0.02, 0.7), fontsize=6)
fig_psth.show()

fig_psth_leg, ax_psth_leg = plt.subplots()
ax_psth_leg.legend(*ax.get_legend_handles_labels(), frameon=False, loc='center')
ax_psth_leg.axis('off')
fig_psth_leg.show()

if saving:
    sn = save_pfix + '_00_ResponsePlot_byCategory_PopulationPSTH'
    for se in save_ext:
        fig_psth.savefig(os.path.join(save_path, sn + se),
                         dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_psth_leg.savefig(os.path.join(save_path, sn + '_Legend' + se),
                         dpi=plt.rcParams['figure.dpi'], transparent=True)

del mi, m, xs, xticks, xticklabels


# %% Define booleans for face and non-face 'super categories'

#   *** TODO: Also consider yaw and roll...
bool_F = np.logical_or.reduce([data['cat'] == c for c in categories
                               if 'face' in c.decode() 
                               and 'blank' not in c.decode() and 'scram' not in c.decode()
                               and 'ctn' not in c.decode()])
bool_NF = np.logical_or.reduce([data['cat'] == c for c in categories 
                                if 'face' not in c.decode() 
                                and 'blank' not in c.decode() and 'scram' not in c.decode()])
bool_NFobj = np.logical_or.reduce([data['cat'] == c for c in categories 
                                   if 'face' not in c.decode() 
                                   and 'blank' not in c.decode() and 'scram' not in c.decode()
                                   and 'body' not in c.decode()])
bool_O = bool_NFobj
bool_B = np.logical_or.reduce([data['cat'] == c for c in categories 
                               if 'body' in c.decode() 
                               and 'blank' not in c.decode() and 'scram' not in c.decode()])

supercategories = ['F', 'O', 'B']
n_supcats = len(supercategories)
supcat_to_bool = {'F': bool_F, 'O': bool_O, 'B': bool_B}
supcat_to_supcatidx = {k: i for i, k in enumerate(supercategories)}
supcat_to_cat = {scat: pd.unique(np.array([cat for cat in data[supcat_to_bool[scat]]['cat']]))
                 for scat in supercategories}
supcat_to_catidx = {scat: pd.unique(np.array([cat_to_catidx[cat] for cat in data[supcat_to_bool[scat]]['cat']]))
                    for scat in supercategories}
supcat_to_cond = {scat: pd.unique(np.array([cnd for cnd in data[supcat_to_bool[scat]]['cond']]))
                  for scat in supercategories}
supcat_to_condidx = {scat: pd.unique(np.array([cond_to_condidx[cnd] for cnd in data[supcat_to_bool[scat]]['cond']]))
                     for scat in supercategories}
cat_to_supcat = {cat: scat for scat, catlist in supcat_to_cat.items() for cat in catlist}
catidx_to_supcat = {icat: scat for scat, icatlist in supcat_to_catidx.items() for icat in icatlist}
cond_to_supcat = {cnd: scat for scat, cndlist in supcat_to_cond.items() for cnd in cndlist}
condidx_to_supcat = {icnd: scat for scat, icndlist in supcat_to_condidx.items() for icnd in icndlist}


# %% Compute statistics for each ROI

idx_stim = range(n_samp_isi, n_samp_isi + n_samp_stim)

# Calculate across-stimulus, trial-averaged response means and standard deviations for each ROI
muR_F = {}
muR_NF = {}
muR_NFobj = {}
muR_O = {}
muR_B = {}
sigma_F = {}
sigma_NF = {}
sigma_NFobj = {}
sigma_O = {}
sigma_B = {}
sort_idx_muR_F = {}
for m in metrics:
    muR_F[m] = np.full(n_ROIs, np.nan)
    muR_NF[m] = np.full(n_ROIs, np.nan)
    muR_NFobj[m] = np.full(n_ROIs, np.nan)
    muR_B[m] = np.full(n_ROIs, np.nan)
    sigma_F[m] = np.full(n_ROIs, np.nan)
    sigma_NF[m] = np.full(n_ROIs, np.nan)
    sigma_NFobj[m] = np.full(n_ROIs, np.nan)
    sigma_B[m] = np.full(n_ROIs, np.nan)

    # Calculate across-stimulus, trial-averaged, frame-averaged mean responses
    #   Ordering of mean calculations matters here because the mean of a set is only the
    #   same as the mean of the mean of subsets if the subsets share the same sample size.
    muR_F[m] = np.mean(np.nanmean(data[bool_F][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_NF[m] = np.mean(np.nanmean(data[bool_NF][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_NFobj[m] = np.mean(np.nanmean(data[bool_NFobj][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    muR_O[m] = muR_NFobj[m]
    muR_B[m] = np.mean(np.nanmean(data[bool_B][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)

    # Calculate across-stimulus, trial-averaged, frame-averaged standard deviations
    #   Take the mean across trials and frames (okay because same n_samp_stim in each), 
    #   then take std across stimulus conditions.
    sigma_F[m] = np.std(np.nanmean(data[bool_F][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_NF[m] = np.std(np.nanmean(data[bool_NF][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_NFobj[m] = np.std(np.nanmean(data[bool_NFobj][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
    sigma_O[m] = sigma_NFobj[m]
    sigma_B[m] = np.std(np.nanmean(data[bool_B][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)

    # Create sorting index based on across-stimulus, trial-averaged, frame-averaged mean responses
    sort_idx_muR_F[m] = np.argsort(muR_F[m])[::-1]
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

    bool_same = (np.sign(muR_F[m]) == np.sign(muR_O[m]))
    bool_FposNFneg = (np.sign(muR_F[m]) > np.sign(muR_O[m]))
    bool_FnegNFpos = (np.sign(muR_F[m]) < np.sign(muR_O[m]))
    
    FSI[m][bool_same] = (muR_F[m][bool_same] - muR_O[m][bool_same]) / \
        (muR_F[m][bool_same] + muR_O[m][bool_same])
    FSI[m][bool_FposNFneg] = 1.0
    FSI[m][bool_FnegNFpos] = -1.0
del m, bool_same, bool_FposNFneg, bool_FnegNFpos

# Response reliability
# based on Vinken et al Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
# """
# The firing-rate reliability was determined per neural site. First, for each image, the number of repeated
# presentations (trials) was randomly split in half. Next, the responses were trial averaged to create two response
# vectors, one per half of the trials. These two split-half response vectors were then correlated, and the procedure
# was repeated for 100 random splits to compute an average correlation r. The reliability ρ was computed by applying
# the Spearman-Brown correction as follows:
# ρ = 2r / (1 + r)
# """


# Dynamic range
# based on Vinken et al Livingstone 2023 Sci Adv https://doi.org/10.1126/sciadv.adg1736
# """
# The dynamic range for faces was quantified by first identifying the “best” and “worst” face (highest and lowest
# response, respectively) using even trials, and then computing the normalized difference in response using the
# held-out odd trials:
# DR_F = (R_bestF - R_worstF) / (Rmax - Rmin)
# where Rbest F and Rworst F are the odd-trial–averaged responses to the best and worst face,
# and Rmax and Rmin are the maximum and minimum odd-trial–averaged responses. The dynamic range for non-faces was
# computed analogously.
# """


# Breadth of tuning
# based on Baylis et al Leonard 1985 Brain Res https://doi.org/10.1016/0006-8993(85)91356-3
# H = -k Σ_i=1_to_n p_i * log(p_i)
# k = scaling constant (set so that H = 1.0 when neuron responds equally well to all stimuli in the set of size n)
# p_i = the response to stimulus i expressed as a proportion of the total response to all stimuli in the set


# Calculate 'stimulus response vectors' for each ROI
resp_vect_cond = {}
resp_vect_cat = {}
for m in metrics:
    resp_vect_cond[m] = np.nanmean(data[m][:, :, :, idx_stim], axis=(2, 3)).T
    resp_vect_cat[m] = np.array([np.mean(np.nanmean(data[cat_to_condidx[c]][m][:, :, :, idx_stim], axis=(2, 3)), axis=0)
                                 for c in categories]).T
del m


# Store response values and tuning metrics for each ROI
stats = {}
for m in metrics:
    stats[m] = {}
    for r in range(n_ROIs):
        stats[m][r] = {}
    
        stats[m][r]['mask'] = np.concatenate((ROIs[r]['xpix'][:, np.newaxis], ROIs[r]['ypix'][:, np.newaxis]), axis=1)
        stats[m][r]['centroid_px'] = np.average(stats[m][r]['mask'], axis=0)
        stats[m][r]['centroid_um'] = md['fov']['resolution_umpx'] * stats[m][r]['centroid_px']
        
        stats[m][r]['resp_vect_cond'] = resp_vect_cond[m][r]
        stats[m][r]['peak_cond_idx'] = stats[m][r]['resp_vect_cond'].argmax()
        stats[m][r]['peak_cond'] = conditions[stats[m][r]['peak_cond_idx']]
        stats[m][r]['peak_cond_val'] = stats[m][r]['resp_vect_cond'].max()
        # stats[m][r]['cat_of_peak_cond'] = (data[data['cond'] == stats[m][r]['peak_cond']]['cat'])
        stats[m][r]['cat_of_peak_cond'] = cond_to_cat[stats[m][r]['peak_cond']]
    
        stats[m][r]['resp_vect_cat'] = resp_vect_cat[m][r]
        stats[m][r]['peak_cat_idx'] = stats[m][r]['resp_vect_cat'].argmax()
        stats[m][r]['peak_cat'] = categories[stats[m][r]['peak_cat_idx']]
        stats[m][r]['peak_cat_val'] = stats[m][r]['resp_vect_cat'].max()
        
        stats[m][r]['dprime_f'] = dprime[m][r]
        stats[m][r]['fsi'] = FSI[m][r]
    del r
del m

stats_df = {}
for m in metrics:
    stats_df[m] = pd.DataFrame({'roi': range(n_ROIs),
                                'mask': None,
                                'centroid_px': None,
                                'centroid_um': None,
                                'peak_cond': None,
                                'peak_cond_idx': None,
                                'peak_cond_val': None,
                                'cat_of_peak_cond': None,
                                'peak_cat': None,
                                'peak_cat_idx': None,
                                'peak_cat_val': None,
                                'dprime_f': None,
                                'fsi': None,
                                'resp_vect_cond': None,
                                'resp_vect_cat': None})
    stats_df[m].set_index(['roi'])

    for r in range(n_ROIs):
        stats_df[m].at[r, 'mask'] = np.concatenate((ROIs[r]['xpix'][:, np.newaxis], ROIs[r]['ypix'][:, np.newaxis]), axis=1)
        stats_df[m].at[r, 'centroid_px'] = np.average(stats_df[m].at[r, 'mask'], axis=0)
        stats_df[m].at[r, 'centroid_um'] = md['fov']['resolution_umpx'] * stats_df[m].at[r, 'centroid_px']
        
        stats_df[m].at[r, 'resp_vect_cond'] = resp_vect_cond[m][r]  # np.nanmean(data[m][:, r, :, :][:, :, idx_stim], axis=(1, 2))
        stats_df[m].at[r, 'peak_cond_idx'] = stats_df[m].loc[r]['resp_vect_cond'].argmax()
        stats_df[m].at[r, 'peak_cond'] = conditions[stats_df[m].at[r, 'peak_cond_idx']]
        stats_df[m].at[r, 'peak_cond_val'] = stats_df[m].loc[r]['resp_vect_cond'].max()
        # stats_df[m].at[r, 'cat_of_peak_cond'] = (data[data['cond'] == conditions[stats_df[m].at[r, 'peak_cond_idx']]]['cat'])
        stats_df[m].at[r, 'cat_of_peak_cond'] = cond_to_cat[stats_df[m].at[r, 'peak_cond']]

        stats_df[m].at[r, 'resp_vect_cat'] = resp_vect_cat[m][r]
        stats_df[m].at[r, 'peak_cat_idx'] = stats_df[m].loc[r]['resp_vect_cat'].argmax()
        stats_df[m].at[r, 'peak_cat'] = categories[stats_df[m].at[r, 'peak_cat_idx']]
        stats_df[m].at[r, 'peak_cat_val'] = stats_df[m].loc[r]['resp_vect_cat'].max()
        
        stats_df[m].at[r, 'dprime_f'] = dprime[m][r]
        stats_df[m].at[r, 'fsi'] = FSI[m][r]
    del r
del m

if n_metrics > 1:
    for mi, m in enumerate(metrics):
        if mi + 1 < n_metrics:
            bool_cnd = (stats_df[m]['peak_cond'].values != stats_df[metrics[mi+1]]['peak_cond'].values)
            ROIs_diff_peak_cnd = np.where(bool_cnd)[0]
            if ROIs_diff_peak_cnd.size > 0:
                warn('peak_cond mismatch ({} vs {}) for ROIs: {}'.format(m, metrics[mi+1], ROIs_diff_peak_cnd))
            bool_cat = (stats_df[m]['peak_cat'].values != stats_df[metrics[mi+1]]['peak_cat'].values)
            ROIs_diff_peak_cat = np.where(bool_cat)[0]
            if ROIs_diff_peak_cat.size > 0:
                warn('peak_cat mismatch ({} vs {}) for ROIs: {}'.format(m, metrics[mi+1], ROIs_diff_peak_cat))
    del mi, m, bool_cnd, ROIs_diff_peak_cnd, bool_cat, ROIs_diff_peak_cat


# %% Output information about ROI tuning based on defined thresholds

print('|FSI| threshold: {}'.format(threshold_fsi))
ROIs_tuned_idx = np.argwhere(np.abs(FSI['Fzsc'][np.argsort(FSI['Fzsc'])[::-1]]) > threshold_fsi).squeeze()
n_ROIs_tuned = np.argwhere(np.abs(FSI['Fzsc'][np.argsort(FSI['Fzsc'])[::-1]]) > threshold_fsi).shape[0]
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(round(((100 * n_ROIs_tuned) / n_ROIs), 2)))
del ROIs_tuned_idx, n_ROIs_tuned

print('|dprime| threshold: {}'.format(threshold_dprime))
ROIs_tuned_idx = np.argwhere(np.abs(dprime['Fzsc'][np.argsort(dprime['Fzsc'])[::-1]]) > threshold_fsi).squeeze()
n_ROIs_tuned = np.argwhere(np.abs(dprime['Fzsc'][np.argsort(dprime['Fzsc'])[::-1]]) > threshold_fsi).shape[0]
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(round(((100 * n_ROIs_tuned) / n_ROIs), 2)))
del ROIs_tuned_idx, n_ROIs_tuned


# %% Plot histograms of ROI tuning

sn = save_pfix + '_01_Histogram_FSI_fromFdFF'
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_hist_fsi(FSI['FdFF'], threshold=threshold_fsi,
                    title='FSIs calculated from FdFF values', save_path=sp)
sn = save_pfix + '_01_Histogram_FSI_fromZscr'
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_hist_fsi(FSI['Fzsc'], threshold=threshold_fsi,
                    title='FSIs calculated from z-scored values', save_path=sp)

sn = save_pfix + '_01_Histogram_Dprime_fromFzsc'
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_hist_dprime(dprime['Fzsc'], threshold=threshold_dprime,
                       title=r'$d^\prime_F$ calculated from Fzsc values', save_path=sp)
sn = save_pfix + '_01_Histogram_Dprime_fromFdFF'
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_hist_dprime(dprime['FdFF'], threshold=threshold_dprime,
                       title=r'$d^\prime_F$ calculated from FdFF values', save_path=sp)


# %% Plot across-stimulus mean of trial-averaged responses for example ROIs...
#    ... by category
#    ... with a separate figure for each example ROI

m = 'Fzsc'

# Select ROI subset
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
del n_plot_ROIs_div

dpm = 'Fzsc'
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * md['framerate'])
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
for r in range(n_plot_ROIs):
    ridx = sort_idx_dprime[dpm][plot_ROI_subset[r]]
    fig = plt.figure()
    fig.suptitle(r'ROI {} ($d^\prime_F$ {:0.2f}) across-stimulus mean responses'.format(ridx, dprime[dpm][ridx]))
    axes = fig.subplots(nrows=n_metrics, ncols=n_cats)
    for mi, m in enumerate(metrics):
        ymean = np.mean(np.nanmean(data[m][:, ridx, :, :], axis=1))
        ymin = ymean
        ymax = ymean
        for cati, cat in enumerate(categories):
            ax = axes[mi, cati]
            if mi == 0:
                ax.set_title(template_labels[cat], fontsize=6)
            if cati == 0:
                ax.set_ylabel(metric_labels[m])
                ax.tick_params(axis='both', which='major')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                xticks = [x * md['framerate'] for x in range(np.ceil(dur_trial).astype('int') + 1)]
                xticklabels = ['' if not np.isclose(xt, dur_isi * md['framerate'])
                               and not np.isclose(xt, (dur_isi + dur_stim) * md['framerate'])
                               else '{}'.format(np.round(xt / md['framerate']).astype('int')) for xt in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.axis('off')
            ax.axvspan(dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate'], color='0.9', zorder=0)
            n_cnd_in_cat = data[cat_to_condidx[cat]]['cond'].shape[0]
            for cnd in range(n_cnd_in_cat):
                ax.plot(xs,
                        np.nanmean(data[cat_to_condidx[cat]][m][cnd, ridx, :, :], axis=0),
                        linewidth=0.5, markersize=0.5,
                        color=str(np.linspace(0.4, 0.7, n_cnd_in_cat)[cnd]), zorder=1)
            Fmean = np.mean(np.nanmean(data[cat_to_condidx[cat]][m][:, ridx, :, :], axis=1), axis=0)
            Fsem = np.std(np.nanmean(data[cat_to_condidx[cat]][m][:, ridx, :, :], axis=1), axis=0) / np.sqrt(n_cnd_in_cat)
            ymin = np.min([ymin, np.min(Fmean - Fsem)])
            ymax = np.max([ymax, np.max(Fmean + Fsem)])
            ax.plot(xs, Fmean, color='0.0', zorder=3)
            ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, facecolor='0.2', alpha=0.6, zorder=2)
            ax.set_ylim((ymin - 0.05 * np.abs(ymax - ymin), ymax + 0.05 * np.abs(ymax - ymin)))            
    fig.show()

    if saving:
        sn = save_pfix + '_02_ResponsePlot_dprime{:0.2f}_'.format(dprime[dpm][ridx]).replace('.', 'p') + \
            'ROI{}'.format(ridx)
        for se in save_ext:
            fig.savefig(os.path.join(save_path, sn + se),
                        dpi=plt.rcParams['figure.dpi'], transparent=True)

del mi, m, xs, xticks, xticklabels
del dpm


# %% Plot trial-averaged mean responses...
#    ... for selected ROIs
#    ... for a subset of conditions (e.g., faces)

m = 'Fzsc'
# bool_focus = (data['cat'] == b'face_mrm')
bool_focus = bool_F
conds_focus = np.where(bool_focus)[0]
conds_focus = conds_focus[[i for i, _ in sorted(enumerate(data[bool_focus]['stimulus']), key=sort_by_cond)]]
n_conds_focus = len(conds_focus)

fig = plt.figure()
fig.suptitle('trial-averaged mean responses, selected ROIs')
if md['stim_locked_to_acqfr'] is True:
    xs = acqfr_dilation_factor * (np.arange(n_samp_trial) - n_samp_isi) + (dur_isi * md['framerate'])
else:
    xs = acqfr_dilation_factor * np.arange(n_samp_trial)
axes = fig.subplots(nrows=(n_plot_ROIs + 1), ncols=(n_conds_focus + 1), sharey='row')
for r in range(n_plot_ROIs):
    ridx = sort_idx_dprime[m][plot_ROI_subset[r]]
    if r == 0:
        ax = axes[0, 0]
        ax.axis('off')
        for cndi, cnd in enumerate(conds_focus):
            ax = axes[0, cndi + 1]
            ax.axis('off')
            ax.imshow(plt.imread(data[cnd]['stimulus'].filepath))
    pr = r + 1
    
    # Summary plots of category averages
    ymean = np.mean(np.nanmean(data[m][:, ridx, :, :], axis=1))
    ymin = ymean
    ymax = ymean
    ax = axes[pr, 0]
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r'ROI {}' '\n' '$d^\prime_F$ {:0.2f}'.format(ridx, dprime[m][ridx]), 
                  horizontalalignment='right', rotation=0, fontsize=4)
    ax.axvspan(dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate'], color='0.9', zorder=0)
    for cati, cat in enumerate(categories):
        Fmean = np.mean(np.nanmean(data[cat_to_condidx[cat]][m][:, ridx, :, :], axis=1), axis=0)
        Fsem = np.std(np.nanmean(data[cat_to_condidx[cat]][m][:, ridx, :, :], axis=1), axis=0) / np.sqrt(len(cat_to_condidx[cat]))
        ymin = np.min([ymin, np.min(Fmean - Fsem)])
        ymax = np.max([ymax, np.max(Fmean + Fsem)])
        ax.plot(xs, Fmean, color=colorsys.hsv_to_rgb(cati / n_cats, 1.0, 1.0), linewidth=1, zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem,
                        facecolor=colorsys.hsv_to_rgb(cati / n_cats, 1.0, 1.0), alpha=0.6, zorder=2)

    # Plots for each cond
    for cndi, cnd in enumerate(conds_focus):
        ax = axes[pr, cndi + 1]
        ax.axis('off')
        ax.axvspan(dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate'], color='0.9', zorder=0)
        for t in range(n_reps):
            ax.plot(xs, 
                    data[cnd][m][ridx, t, :],
                    color=str(np.linspace(0.4, 0.7, n_reps)[t]), linewidth=0.1)
        Fmean = np.nanmean(data[cnd][m][ridx, :, :], axis=0)
        Fsem = (np.nanstd(data[cnd][m][ridx, :, :], axis=0) / 
                np.sqrt(n_reps - np.isnan(data[cnd][m][ridx, :, :]).any(axis=1).sum()))  # remove conditions with NaNs
        ymin = np.min([ymin, np.min(Fmean - Fsem)])
        ymax = np.max([ymax, np.max(Fmean + Fsem)])
        ax.plot(xs, Fmean, color='0.0', linewidth=1, zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, facecolor='0.0', alpha=0.6, zorder=2)
    ax.set_ylim((ymin - 0.05 * np.abs(ymax - ymin), ymax + 0.05 * np.abs(ymax - ymin)))
fig.show()

if saving:
    sn = save_pfix + '_03_ResponsePlot_TrialAveraged_ROIsSelectedSubset'
    for se in save_ext:
        fig.savefig(os.path.join(save_path, sn + se),
                    dpi=plt.rcParams['figure.dpi'], transparent=True)

del m, xs, r, pr, cat, cnd, cndi, t 
del bool_focus, conds_focus, n_conds_focus


# %% Heatmap trial-averaged responses... for a subset of conditions (e.g., faces)...
#    ... for all ROIs
#    ... sorted by dprime_F

m = 'Fzsc'
# bool_focus = (data['cat'] == b'face_mrm')
bool_focus = bool_F
conds_focus = np.where(bool_focus)[0]
conds_focus = conds_focus[[i for i, _ in sorted(enumerate(data[bool_focus]['stimulus']), key=sort_by_cond)]]
n_conds_focus = len(conds_focus)

fig_hm = plt.figure()
axes = fig_hm.subplots(nrows=2, ncols=(n_conds_focus + 1), height_ratios=[40, 790], sharey='row')
fig_hm.subplots_adjust(hspace=0)
fig_hm.suptitle('trial-averaged mean responses, all ROIs')

pr = 0
ax = axes[pr, 0]
ax.axis('off')
for cndi, cnd in enumerate(conds_focus):
    ax = axes[pr, cndi + 1]
    ax.axis('off')
    img_st = ax.imshow(plt.imread(data[cnd]['stimulus'].filepath))

pr = 1
ax_dp = axes[pr, 0]
ax_dp.set_xlabel('$d^\prime_F$')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_dp.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                      color='0.2', linestyle='dotted', linewidth=0.5)

for cndi, cnd in enumerate(conds_focus):
    ax = axes[pr, cndi + 1]
    ax.axis('off')
    img_hm = ax.imshow(np.nanmean(data[cnd][m], axis=1)[sort_idx_dprime[m]],
                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
    xlines = [dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate']]
    for xl in xlines:
        ax.axvline(x=xl, linestyle='dashed', linewidth=0.5, color='0.4')
    if threshold_dprime is not None:
        if threshold_dprime != 0:
            ax.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
            ax.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
        else:
            ax.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                       color='0.2', linestyle='dotted', linewidth=0.5)
plots.set_plot_text_settings()
fig_hm.show()

fig_cb, ax_cb = plt.subplots()
cbar = plt.colorbar(img_hm, ax=ax_cb)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('Z-score')
ax_cb.remove()
plots.set_plot_text_settings()
fig_cb.show()

if saving:
    sn = save_pfix + '_04_Heatmap_TrialMeanAveragedSubset_sortDprime'
    #    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
    for se in save_ext:
        fig_hm.savefig(os.path.join(save_path, sn + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_cb.savefig(os.path.join(save_path, sn + '_Colorbar' + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)

del m, pr, cndi, cnd
del ax, xl, xlines


# %% Heatmap trial-averaged (median) responses... for a subset of conditions (e.g., faces)...
#    ... for all ROIs
#    ... sorted by dprime_F

m = 'Fzsc'
# bool_focus = (data['cat'] == b'face_mrm')
bool_focus = bool_F
conds_focus = np.where(bool_focus)[0]
conds_focus = conds_focus[[i for i, _ in sorted(enumerate(data[bool_focus]['stimulus']), key=sort_by_cond)]]
n_conds_focus = len(conds_focus)

fig_hm = plt.figure()
axes = fig_hm.subplots(nrows=2, ncols=(n_conds_focus + 1), height_ratios=[40, 790], sharey='row')
fig_hm.subplots_adjust(hspace=0)
fig_hm.suptitle('trial-averaged median responses, all ROIs')

pr = 0
ax = axes[pr, 0]
ax.axis('off')
for cndi, cnd in enumerate(conds_focus):
    ax = axes[pr, cndi + 1]
    ax.axis('off')
    img_st = ax.imshow(plt.imread(data[cnd]['stimulus'].filepath))

pr = 1
ax_dp = axes[pr, 0]
ax_dp.set_xlabel('$d^\prime_F$')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_dp.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                      color='0.2', linestyle='dotted', linewidth=0.5)

for cndi, cnd in enumerate(conds_focus):
    ax = axes[pr, cndi + 1]
    ax.axis('off')
    img_hm = ax.imshow(np.nanmedian(data[cnd][m], axis=1)[sort_idx_dprime[m]],
                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
    xlines = [dur_isi * md['framerate'], (dur_isi + dur_stim) * md['framerate']]
    for xl in xlines:
        ax.axvline(x=xl, linestyle='dashed', linewidth=0.5, color='0.4')
    if threshold_dprime is not None:
        if threshold_dprime != 0:
            ax.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
            ax.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
        else:
            ax.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                       color='0.2', linestyle='dotted', linewidth=0.5)
plots.set_plot_text_settings()
fig_hm.show()

fig_cb, ax_cb = plt.subplots()
cbar = plt.colorbar(img_hm, ax=ax_cb)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('Z-score')
ax_cb.remove()
plots.set_plot_text_settings()
fig_cb.show()

if saving:
    sn = save_pfix + '_04_Heatmap_TrialMedianAveragedSubset_sortDprime'
    #    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
    for se in save_ext:
        fig_hm.savefig(os.path.join(save_path, sn + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_cb.savefig(os.path.join(save_path, sn + '_Colorbar' + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)

del m, pr, cndi, cnd
del ax, xl, xlines


# %% Heatmap trial-averaged stimulus-epoch responses... for all conditions...
#    ... for all ROIs
#    ... sorted by dprime_F

m = 'Fzsc'

# Define category-dividing ticks
tickinfo = {t.decode(): {} for t in template}
for t in template:
    ts = t.decode()
    wheret = np.where(data[[i for i, _ in sorted(enumerate(data['stimulus']), key=sort_by_cond)]]['cat'] == t)[0]
    if wheret.size > 0:
        tickinfo[ts]['start'] = wheret[0]
        tickinfo[ts]['end'] = wheret[-1]
        tickinfo[ts]['labelpos'] = (tickinfo[ts]['start'] + tickinfo[ts]['end']) / 2
        tickinfo[ts]['label'] = template_labels[t]
    else:
        tickinfo.pop(ts)
del t, ts, wheret

sort_idx_cond = [i for i, _ in sorted(enumerate(data['stimulus']), key=sort_by_cond)]
fig_hm, (ax_hm, ax_dp, ax_fsi) = plt.subplots(1, 3, width_ratios=[7.5, 0.75, 0.75], sharey=True)
# fig_hm, (ax_hm, ax_dp) = plt.subplots(1, 2, width_ratios=[7.5, 0.75], sharey=True)
fig_hm.subplots_adjust(wspace=0.05)
fig_hm.suptitle('trial-averaged stimulus-epoch mean responses, all ROIs')
ax_hm.set_xlabel('Stimulus Image')
ax_hm.set_ylabel('ROI')
xtick_majors = []
xtick_majorlabels = []
xtick_minors = []
xtick_minorlabels = []
for i, t in enumerate(tickinfo):
    ti = tickinfo[t]
    if ti['start'] == ti['end']:
        # xtick_majors.append(ti['start'] + 0.5)
        # xtick_majorlabels.append(ti['label'])
        xtick_minors.append(ti['labelpos'])
        xtick_minorlabels.append(ti['label'])
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
img_hm = ax_hm.imshow(np.nanmean(data[sort_idx_cond][m][:, :, :, idx_stim], axis=(2, 3)).swapaxes(0, 1)[sort_idx_dprime[m]],
                      vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_hm.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
        ax_hm.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_hm.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                       color='0.2', linestyle='dotted', linewidth=0.5)

ax_dp.set_xlabel('$d^\prime_F$')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_dp.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    
ax_fsi.set_xlabel('FSI')
ax_fsi.set_axisbelow(True)
ax_fsi.set_xlim([-1, 1])
ax_fsi.barh(range(0, n_ROIs), FSI[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_fsi.axvline(x=0, color='0.0', linewidth=0.5)
ax_fsi.spines['right'].set_visible(False)
ax_fsi.spines['left'].set_visible(False)
ax_fsi.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_fsi.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_fsi.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)

plots.set_plot_text_settings()
fig_hm.show()

fig_cb, ax_cb = plt.subplots()
cbar = plt.colorbar(img_hm, ax=ax_cb)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('Z-score')
ax_cb.remove()
plots.set_plot_text_settings()
fig_cb.show()

if saving:
    sn = save_pfix + '_05_Heatmap_byCondition_sortDprime'
    #    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
    for se in save_ext:
        fig_hm.savefig(os.path.join(save_path, sn + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_cb.savefig(os.path.join(save_path, sn + '_Colorbar' + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)

del i, t, tick
del m


# %% Heatmap across-stimulus mean of trial-averaged stimulus-epoch responses... sorted by dprime_F...
#    ... for all ROIs
#    ... for all categories

m = 'Fzsc'

# Define category ticks
tickinfo = {t.decode(): {} for t in template}
for t in template:
    ts = t.decode()
    wheret = np.where(categories == t)[0]
    if wheret.size > 0:
        tickinfo[ts]['start'] = wheret[0]
        tickinfo[ts]['end'] = wheret[0]
        tickinfo[ts]['labelpos'] = wheret[0]
        tickinfo[ts]['label'] = template_labels[t]
    else:
        tickinfo.pop(ts)
del t, ts, wheret

fig_hm, (ax_hm, ax_dp, ax_fsi) = plt.subplots(1, 3, width_ratios=[7.5, 0.75, 0.75], sharey=True)
# fig_hm, (ax_hm, ax_dp) = plt.subplots(1, 2, width_ratios=[7.5, 0.75], sharey=True)
fig_hm.subplots_adjust(wspace=0.05)
fig_hm.suptitle(r'across-stimulus mean of trial-averaged responses, all ROIs, sorted by $d^\prime_F$')
ax_hm.set_xlabel('Stimulus Category')
ax_hm.set_ylabel('ROI')
xtick_majors = []
xtick_majorlabels = []
xtick_minors = []
xtick_minorlabels = []
for i, t in enumerate(tickinfo):
    ti = tickinfo[t]
    if ti['start'] == ti['end']:
        if i == 0:
            xtick_majors.append(ti['start'] - 0.5)
            xtick_majorlabels.append(None)
        xtick_majors.append(ti['start'] + 0.5)
        xtick_majorlabels.append(None)
        xtick_minors.append(ti['labelpos'])
        xtick_minorlabels.append(ti['label'])
    elif ti['end'] > ti['start']:
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
# img_hm = ax_hm.imshow(np.nanmean(data[sort_idx_cond][m][:, :, :, idx_stim], axis=(2, 3)).swapaxes(0, 1)[sort_idx_dprime[m]],
#                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
img_hm = ax_hm.imshow(np.vstack(stats_df[m]['resp_vect_cat'].values)[sort_idx_dprime[m]],
                      vmin=-0.5, vmax=0.5, aspect='auto', cmap='bwr', interpolation='none')
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_hm.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
        ax_hm.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                       color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_hm.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                       color='0.2', linestyle='dotted', linewidth=0.5)

ax_dp.set_xlabel(r'$d^\prime_F$')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_dp.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
if threshold_dprime is not None:
    if threshold_dprime != 0:
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] < -threshold_dprime)[0].min(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
        ax_dp.axhline(np.where(dprime[m][sort_idx_dprime[m]] > threshold_dprime)[0].max(),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    else:
        ax_dp.axhline(np.where(np.isclose(dprime[m][sort_idx_dprime[m]], threshold_dprime, atol=0.05)),
                      color='0.2', linestyle='dotted', linewidth=0.5)
    
ax_fsi.set_xlabel('FSI')
ax_fsi.set_axisbelow(True)
ax_fsi.set_xlim([-1, 1])
ax_fsi.barh(range(0, n_ROIs), FSI[m][sort_idx_dprime[m]], height=1.0, color='0.5')
ax_fsi.axvline(x=0, color='0.0', linewidth=0.5)
ax_fsi.spines['right'].set_visible(False)
ax_fsi.spines['left'].set_visible(False)
ax_fsi.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_fsi.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_fsi.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
    
plots.set_plot_text_settings()
fig_hm.show()

fig_cb, ax_cb = plt.subplots()
cbar = plt.colorbar(img_hm, ax=ax_cb)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('Z-score')
ax_cb.remove()
plots.set_plot_text_settings()
fig_cb.show()

if saving:
    sn = save_pfix + '_06_Heatmap_byCategory_sortDprime'
    #    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
    for se in save_ext:
        fig_hm.savefig(os.path.join(save_path, sn + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_cb.savefig(os.path.join(save_path, sn + '_Colorbar' + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)

del i, t, tick


# %% Heatmap across-stimulus mean of trial-averaged stimulus-epoch responses... sorted by across-stimulus mean...
#    ... for all ROIs
#    ... for all categories

m = 'Fzsc'

# Define category ticks
tickinfo = {t.decode(): {} for t in template}
for t in template:
    ts = t.decode()
    wheret = np.where(categories == t)[0]
    if wheret.size > 0:
        tickinfo[ts]['start'] = wheret[0]
        tickinfo[ts]['end'] = wheret[0]
        tickinfo[ts]['labelpos'] = wheret[0]
        tickinfo[ts]['label'] = template_labels[t]
    else:
        tickinfo.pop(ts)
del t, ts, wheret

fig_hm, (ax_hm, ax_dp, ax_fsi) = plt.subplots(1, 3, width_ratios=[7.5, 0.75, 0.75], sharey=True)
# fig_hm, (ax_hm, ax_dp) = plt.subplots(1, 2, width_ratios=[7.5, 0.75], sharey=True)
fig_hm.subplots_adjust(wspace=0.05)
fig_hm.suptitle('across-stimulus mean of trial-averaged responses, all ROIs, sorted by F mean')
ax_hm.set_xlabel('Stimulus Category')
ax_hm.set_ylabel('ROI')
xtick_majors = []
xtick_majorlabels = []
xtick_minors = []
xtick_minorlabels = []
for i, t in enumerate(tickinfo):
    ti = tickinfo[t]
    if ti['start'] == ti['end']:
        if i == 0:
            xtick_majors.append(ti['start'] - 0.5)
            xtick_majorlabels.append(None)
        xtick_majors.append(ti['start'] + 0.5)
        xtick_majorlabels.append(None)
        xtick_minors.append(ti['labelpos'])
        xtick_minorlabels.append(ti['label'])
    elif ti['end'] > ti['start']:
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
# img_hm = ax_hm.imshow(np.nanmean(data[sort_idx_cond][m][:, :, :, idx_stim], axis=(2, 3)).swapaxes(0, 1)[sort_idx_muR_F[m]],
#                       vmin=-1.0, vmax=1.0, aspect='auto', cmap='bwr', interpolation='none')
img_hm = ax_hm.imshow(np.vstack(stats_df[m]['resp_vect_cat'].values)[sort_idx_muR_F[m]],
                      vmin=-0.5, vmax=0.5, aspect='auto', cmap='bwr', interpolation='none')

ax_dp.set_xlabel('$d^\prime_F$')
ax_dp.set_axisbelow(True)
ax_dp.barh(range(0, n_ROIs), dprime[m][sort_idx_muR_F[m]], height=1.0, color='0.5')
ax_dp.axvline(x=0, color='0.0', linewidth=0.5)
ax_dp.spines['right'].set_visible(False)
ax_dp.spines['left'].set_visible(False)
ax_dp.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_dp.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_dp.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)
    
ax_fsi.set_xlabel('FSI')
ax_fsi.set_axisbelow(True)
ax_fsi.set_xlim([-1, 1])
ax_fsi.barh(range(0, n_ROIs), FSI[m][sort_idx_muR_F[m]], height=1.0, color='0.5')
ax_fsi.axvline(x=0, color='0.0', linewidth=0.5)
ax_fsi.spines['right'].set_visible(False)
ax_fsi.spines['left'].set_visible(False)
ax_fsi.grid(axis='x', linestyle='dashed', linewidth=0.5, color='0.8')
for tick in ax_fsi.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax_fsi.xaxis.get_major_ticks():
    tick.label1.set_fontsize(4)

plots.set_plot_text_settings()
fig_hm.show()

fig_cb, ax_cb = plt.subplots()
cbar = plt.colorbar(img_hm, ax=ax_cb)
cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
cbar.ax.set_yticklabels(['-1.0', '-0.5', '0', '0.5', '1'])
cbar.set_label('Z-score')
ax_cb.remove()
plots.set_plot_text_settings()
fig_cb.show()

if saving:
    sn = save_pfix + '_07_Heatmap_byCategory_sortMeanFace'
    #     '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
    for se in save_ext:
        fig_hm.savefig(os.path.join(save_path, sn + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)
        fig_cb.savefig(os.path.join(save_path, sn + '_Colorbar' + se),
                       dpi=plt.rcParams['figure.dpi'], transparent=True)

del i, t, tick


# %% Overlay ROI masks over imaging data... pseudocolored by the supercategory of the peak response condition...

m = 'Fzsc'

ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) if not np.isnan(h) else (0, 0, 0)
                       for h in np.divide([supcat_to_supcatidx[condidx_to_supcat[icnd]] if icnd in condidx_to_supcat else np.nan
                                           for icnd in stats_df[m]['peak_cond_idx'].values.astype(int)],
                                          n_supcats)])
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                        for h in np.divide([supcat_to_supcatidx[condidx_to_supcat[icnd]]
#                                            for icnd in stats_df[m]['peak_cond_idx'].values.astype(int)],
#                                           n_supcats)])
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                         for h in np.divide([cat_to_catidx[cat]
#                                             for cat in data[stats_df[m]['peak_cond_idx'].values.astype(int)]['cat']],
#                                           n_cats)])

# ...only for ROIs with |dprime_F| >= threshold
above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]
sn = save_pfix + '_08_ROIplot_ColorByCategoryOfPeakCondition' + \
    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response,\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dprime),
                        save_path=sp)

# ...only for ROIs with |FSI| >= threshold
above_threshold = np.where(np.abs(FSI[m]) >= threshold_fsi)[0]
sn = save_pfix + '_08_ROIplot_ColorByCategoryOfPeakCondition' + \
    '_threshFSI{:0.2f}'.format(threshold_fsi).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response,\n' +
                              r'FSI $\geq$ {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# ...only for ROIs with |mean Z-score| >= threshold
above_threshold = np.where(stats_df[m]['peak_cond_val'] >= threshold_Zscore)[0]
sn = save_pfix + '_08_ROIplot_ColorByCategoryOfPeakCondition' + \
    '_threshZ{:0.2f}'.format(threshold_Zscore).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category of the condition (image) eliciting the largest response,\n' +
                              r'z $\geq$ {:0.2f}'.format(threshold_Zscore),
                        save_path=sp)

del above_threshold, ROI_colors


# %% Overlay ROI masks over imaging data... pseudocolored by the peak across-stimulus average response supercategory...

m = 'Fzsc'

ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) if not np.isnan(h) else (0, 0, 0)
                       for h in np.divide([supcat_to_supcatidx[catidx_to_supcat[icat]]
                                           if icat in catidx_to_supcat else np.nan
                                           for icat in stats_df[m]['peak_cat_idx'].values.astype(int)],
                                          n_supcats)])
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                        for h in np.divide([supcat_to_supcatidx[catidx_to_supcat[icat]]
#                                            for icat in stats_df[m]['peak_cat_idx'].values.astype(int)],
#                                           n_supcats)])
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                         for h in np.divide(stats_df[m]['peak_cat_idx'].values.astype(int), n_cats)])

# ...only for ROIs with |dprime_F| >= threshold
above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]
sn = save_pfix + '_09_ROIplot_ColorByPeakCategory' + \
    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category eliciting the largest average response,\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dprime),
                        save_path=sp)

# ...only for ROIs with |FSI| >= threshold
above_threshold = np.where(np.abs(FSI[m]) >= threshold_fsi)[0]
sn = save_pfix + '_09_ROIplot_ColorByPeakCategory' + \
    '_threshFSI{:0.2f}'.format(threshold_fsi).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category eliciting the largest average response,\n' +
                              r'FSI $\geq$ {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# ...only for ROIs with |mean Z-score| >= threshold
above_threshold = np.where(stats_df[m]['peak_cat_val'] > threshold_Zscore)[0]
sn = save_pfix + '_09_ROIplot_ColorByPeakCategory' + \
    '_threshZ{:0.2f}'.format(threshold_Zscore).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='category eliciting the largest average response,\n' +
                              r'z $\geq$ {:0.2f}'.format(threshold_Zscore),
                        save_path=sp)

del above_threshold, ROI_colors


# %% Overlay ROI masks over mean frame image... pseudocolored by relative response strength...

resp_vect_FOB = np.array([muR_F[m], muR_O[m], muR_B[m]]).swapaxes(0, 1)
# Subtract the value corresponding to the least responsive category to make it relative
# (otherwise, an ROI that responds to all categories would show up as white)
resp_vect_FOB_rel = np.subtract(resp_vect_FOB.T, np.min(resp_vect_FOB, axis=1)).T

# Saturate colors at specified value (corresponding to the across-stimulus mean of trial-averaged responses)
ROI_colors_saturateval = 0.2
ROI_colors = resp_vect_FOB_rel / ROI_colors_saturateval
ROI_colors[ROI_colors > 1] = 1

# ...only for ROIs with |dprime_F| >= threshold
above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]
sn = save_pfix + '_10_ROIplot_ColorByRelativeResponseStrength' + \
    '_max{}{:0.2f}'.format(m, ROI_colors_saturateval).replace('.', 'p') + \
    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold],
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='relative response strength,\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dprime),
                        save_path=sp)

# ...only for ROIs with |FSI| >= threshold
above_threshold = np.where(np.abs(FSI[m]) >= threshold_fsi)[0]
sn = save_pfix + '_10_ROIplot_ColorByRelativeResponseStrength' + \
    '_max{}{:0.2f}'.format(m, ROI_colors_saturateval).replace('.', 'p') + \
    '_threshFSI{:0.2f}'.format(threshold_fsi).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold],
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='relative response strength,\n' +
                              r'FSI $\geq$ {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# ...only for ROIs with |mean Z-score| >= threshold
above_threshold = np.where(stats_df[m]['peak_cat_val'] > threshold_Zscore)[0]
sn = save_pfix + '_10_ROIplot_ColorByRelativeResponseStrength' + \
    '_max{}{:0.2f}'.format(m, ROI_colors_saturateval).replace('.', 'p') + \
    '_threshZ{:0.2f}'.format(threshold_Zscore).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold],
                        ROI_colors[above_threshold],
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='relative response strength,\n' +
                              r'z $\geq$ {:0.2f}'.format(threshold_Zscore), 
                        save_path=sp)

del above_threshold, ROI_colors


# %% Overlay peak response-eliciting stimulus images over imaging data... pseudocolored by the peak across-stimulus
#    average response supercategory...

m = 'Fzsc'

# resp_vect_FOB = np.array([muR_F[m], muR_O[m], muR_B[m]]).swapaxes(0, 1)
# ROI_colors_idx = np.argmax(resp_vect_FOB, axis=1).astype(int)
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                        for h in np.divide(ROI_colors_idx, resp_vect_FOB.shape[1])])
ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) if not np.isnan(h) else (0, 0, 0)
                       for h in np.divide([supcat_to_supcatidx[catidx_to_supcat[icat]] if icat in catidx_to_supcat else np.nan
                                           for icat in stats_df[m]['peak_cat_idx'].values.astype(int)],
                                          n_supcats)])
# ROI_colors = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0)
#                        for h in np.divide([supcat_to_supcatidx[catidx_to_supcat[icat]]
#                                            for icat in stats_df[m]['peak_cat_idx'].values.astype(int)],
#                                           n_supcats)])
ROI_images = np.array([data[stats_df[m]['peak_cond_idx'][ir]]['stimulus'].filepath for ir in range(n_ROIs)])

# ...only for ROIs with |dprime_F| >= threshold
above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]
sn = save_pfix + '_11_ROIplot_OverlayImageColorByPeakCategory' + \
    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_img(ROIs[above_threshold], 
                        images=ROI_images[above_threshold],
                        colors=ROI_colors[above_threshold], alpha=0.6,
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title='image eliciting peak response\nover category eliciting peak average response,\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dprime),
                        save_path=sp)

# ...only for ROIs with |FSI| >= threshold
above_threshold = np.where(np.abs(FSI[m]) >= threshold_fsi)[0]
sn = save_pfix + '_11_ROIplot_OverlayImageColorByPeakCategory' + \
    '_threshFSI{:0.2f}'.format(threshold_fsi).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_img(ROIs[above_threshold],
                        images=ROI_images[above_threshold],
                        colors=ROI_colors[above_threshold], alpha=0.6,
                        bgimage=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        title='image eliciting peak response\nover category eliciting peak average response,\n' +
                              r'FSI $\geq$ {:0.2f}'.format(threshold_fsi),
                        save_path=sp)

# ...only for ROIs with |mean Z-score| >= threshold
above_threshold = np.where(stats_df[m]['peak_cat_val'] > threshold_Zscore)[0]
sn = save_pfix + '_11_ROIplot_OverlayImageColorByPeakCategory' + \
    '_threshZ{:0.2f}'.format(threshold_Zscore).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_img(ROIs[above_threshold],
                        images=ROI_images[above_threshold],
                        colors=ROI_colors[above_threshold], alpha=0.6,
                        bgimage=plots.auto_level_s2p_image(fov_image),
                        flip='lr', rotate=-90,
                        title='image eliciting peak response\nover category eliciting peak average response,\n' +
                              r'z $\geq$ {:0.2f}'.format(threshold_Zscore),
                        save_path=sp)

del above_threshold, ROI_colors, ROI_images


# %% Overlay ROI masks over imaging data... pseudocolored by dprime_F value...

m = 'Fzsc'

ROI_colors = dprime[m]

# ...for all ROIs
sn = save_pfix + '_12_ROIplot_ColorByDprimeVal_AllROIs'
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs, 
                        ROI_colors, alpha=1.0, colormap='bwr', colorlim=1.0, 
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title=r'$d^\prime_F$ value',
                        save_path=sp)

# ...only for ROIs with |dprime_F| >= threshold
above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]
sn = save_pfix + '_12_ROIplot_ColorByDprimeVal' + \
    '_threshDprime{:0.2f}'.format(threshold_dprime).replace('.', 'p')
sp = [os.path.join(save_path, sn + se) for se in save_ext] if saving else []
plots.plot_overlays_roi(ROIs[above_threshold], 
                        ROI_colors[above_threshold], alpha=1.0, colormap='bwr', colorlim=1.0, 
                        bgimage=plots.auto_level_s2p_image(fov_image), 
                        flip='lr', rotate=-90,
                        title=r'$d^\prime_F$ value,' + '\n' +
                              r'$d^\prime_F$ $\geq$ {:0.2f}'.format(threshold_dprime),
                        save_path=sp)

del above_threshold, ROI_colors


# %% Compare ROI dprime_F values with respect to distance

m = 'Fzsc'

# centroid_px = np.vstack(stats_df[m]['centroid_px'].values)
centroid_um = np.vstack(stats_df[m]['centroid_um'].values)
roipair_dist_um = np.array([np.linalg.norm(centroid_um[r0] - centroid_um[r1]) 
                            for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])

if np.any(roipair_dist_um > np.sqrt(md['fov']['w_um']**2 + md['fov']['h_um']**2)):
    warn('Distance between some ROIs exceeds expected FOV diagonal.')

roipair_dprimediff = np.array([np.abs(dprime[m][r0] - dprime[m][r1])
                               for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])

w_bin_um = 25
n_bins = int(np.ceil(roipair_dist_um.max() / w_bin_um))
bin_edges = np.linspace(0, n_bins * w_bin_um, n_bins + 1)
bin_centers = np.linspace(w_bin_um / 2, (n_bins * w_bin_um) - (w_bin_um / 2), n_bins)
bin_medians, _, _ = binned_statistic(roipair_dist_um, roipair_dprimediff, statistic='median', bins=bin_edges)
bin_stds, _, _ = binned_statistic(roipair_dist_um, roipair_dprimediff, statistic='std', bins=bin_edges)
bin_ns, _, _ = binned_statistic(roipair_dist_um, roipair_dprimediff, statistic='count', bins=bin_edges)

# Exclude any bins with less than 100 pairs
n_pairs_required = 100
bin_centers = bin_centers[bin_ns > n_pairs_required]
bin_medians = bin_medians[bin_ns > n_pairs_required]
bin_stds = bin_stds[bin_ns > n_pairs_required]

fig_dprimediff = plt.figure()
ax = fig_dprimediff.subplots(1, 1)
ax.set_ylabel(r'$|\Delta d^\prime_F|$')
ax.set_xlabel('Distance (µm)')
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major')
ax.set_xlim((0, roipair_dist_um.max() + 1))
ax.set_ylim((roipair_dprimediff.min() - np.abs(0.1 * roipair_dprimediff.min()), 
             roipair_dprimediff.max() + np.abs(0.1 * roipair_dprimediff.max())))
ax.scatter(roipair_dist_um, roipair_dprimediff, marker='.', s=0.5, color='k', edgecolor='None')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='r', markerfacecolor='w', markersize=3, capsize=0,
            fmt='o', elinewidth=1, ecolor='r')
fig_dprimediff.show()

if saving:
    sn = save_pfix + '_13_RelationshipPlot_DprimeDiff_by_Distance'
    for se in save_ext:
        fig_dprimediff.savefig(os.path.join(save_path, sn + se),
                               dpi=plt.rcParams['figure.dpi'], transparent=True)


# %% Compare ROI value-based correlations between stimulus response vectors with respect to distance

m = 'Fzsc'

roipair_corr_respvect = np.array([np.corrcoef(resp_vect_cond[m][r0], resp_vect_cond[m][r1])[0, 1]
                                  for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])
# roipair_corr_respvect_sp = np.array([scipy.stats.pearsonr(resp_vect_cond[m][r0], resp_vect_cond[m][r1]).statistic 
#                                      for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])

w_bin_um = 25
n_bins = int(np.ceil(roipair_dist_um.max() / w_bin_um))
bin_edges = np.linspace(0, n_bins * w_bin_um, n_bins + 1)
bin_centers = np.linspace(w_bin_um / 2, (n_bins * w_bin_um) - (w_bin_um / 2), n_bins)

bin_medians, _, _ = binned_statistic(roipair_dist_um, roipair_corr_respvect, statistic='median', bins=bin_edges)
bin_stds, _, _ = binned_statistic(roipair_dist_um, roipair_corr_respvect, statistic='std', bins=bin_edges)

# Exclude any bins with less than 100 pairs
n_pairs_required = 100
bin_centers = bin_centers[bin_ns > n_pairs_required]
bin_medians = bin_medians[bin_ns > n_pairs_required]
bin_stds = bin_stds[bin_ns > n_pairs_required]

fig_r = plt.figure()
ax = fig_r.subplots(1, 1)
ax.set_ylabel(r'Stimulus response Pearson correlation ($\it{r}$)')
ax.set_xlabel('Distance (µm)')
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major')
ax.set_xlim((0, roipair_dist_um.max() + 1))
ax.set_ylim((roipair_corr_respvect.min() - np.abs(0.1 * roipair_corr_respvect.min()), 1))

ax.scatter(roipair_dist_um, roipair_corr_respvect, marker='.', s=0.5, color='k', edgecolor='None')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='r', markerfacecolor='w', markersize=3, capsize=0,
            fmt='o', elinewidth=1, ecolor='r')
plots.set_plot_text_settings()
fig_r.show()

if saving:
    sn = save_pfix + '_14_RelationshipPlot_StimulusResponsePearsonCorr_by_Distance'
    for se in save_ext:
        fig_r.savefig(os.path.join(save_path, sn + se),
                      dpi=plt.rcParams['figure.dpi'], transparent=True)


# %% Compare ROI rank-based correlations between stimulus response vectors with respect to distance

m = 'Fzsc'

response_rank_rho = np.array([spearmanr(resp_vect_cond[m][r0], resp_vect_cond[m][r1]).statistic 
                              for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])
response_rank_tau = np.array([kendalltau(resp_vect_cond[m][r0], resp_vect_cond[m][r1]).statistic 
                              for r0, r1 in list(itertools.combinations(range(n_ROIs), 2))])

w_bin_um = 25
n_bins = int(np.ceil(roipair_dist_um.max() / w_bin_um))
bin_edges = np.linspace(0, n_bins * w_bin_um, n_bins + 1)
bin_centers = np.linspace(w_bin_um / 2, (n_bins * w_bin_um) - (w_bin_um / 2), n_bins)

bin_medians, _, _ = binned_statistic(roipair_dist_um, response_rank_rho, statistic='median', bins=bin_edges)
bin_stds, _, _ = binned_statistic(roipair_dist_um, response_rank_rho, statistic='std', bins=bin_edges)

# Exclude any bins with less than 100 pairs
n_pairs_required = 100
bin_centers = bin_centers[bin_ns > n_pairs_required]
bin_medians = bin_medians[bin_ns > n_pairs_required]
bin_stds = bin_stds[bin_ns > n_pairs_required]

fig_rho = plt.figure()
ax = fig_rho.subplots(1, 1)
ax.set_ylabel(r'Stimulus response Spearman rank correlation ($\rho$)')
ax.set_xlabel('Distance (µm)')
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major')
ax.set_xlim((0, roipair_dist_um.max() + 1))
ax.set_ylim((response_rank_rho.min() - np.abs(0.1 * response_rank_rho.min()), 1))

ax.scatter(roipair_dist_um, response_rank_rho, marker='.', s=0.5, color='k', edgecolor='None')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='r', markerfacecolor='w', markersize=3, capsize=0,
            fmt='o', elinewidth=1, ecolor='r')
plots.set_plot_text_settings()
fig_rho.show()

if saving:
    sn = save_pfix + '_15_RelationshipPlot_StimulusResponseSpearmanRankCorr_by_Distance'
    for se in save_ext:
        fig_rho.savefig(os.path.join(save_path, sn + se),
                        dpi=plt.rcParams['figure.dpi'], transparent=True)


bin_medians, _, _ = binned_statistic(roipair_dist_um, response_rank_tau, statistic='median', bins=bin_edges)
bin_stds, _, _ = binned_statistic(roipair_dist_um, response_rank_tau, statistic='std', bins=bin_edges)

# Exclude any bins with less than 100 pairs
bin_medians = bin_medians[bin_ns > n_pairs_required]
bin_stds = bin_stds[bin_ns > n_pairs_required]

fig_tau = plt.figure()
ax = fig_tau.subplots(1, 1)
ax.set_ylabel(r'Stimulus response Kendall rank correlation ($\tau$)')
ax.set_xlabel('Distance (µm)')
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major')
ax.set_xlim((0, roipair_dist_um.max() + 1))
ax.set_ylim((response_rank_tau.min() - np.abs(0.1 * response_rank_tau.min()), 1))

ax.scatter(roipair_dist_um, response_rank_tau, marker='.', s=0.5, color='k', edgecolor='None')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='r', markerfacecolor='w', markersize=3, capsize=0,
            fmt='o', elinewidth=1, ecolor='r')
# ax.plot(ddxs, ddys)
plots.set_plot_text_settings()
fig_tau.show()

if saving:
    sn = save_pfix + '_15_RelationshipPlot_StimulusResponseKendallRankCorr_by_Distance'
    for se in save_ext:
        fig_tau.savefig(os.path.join(save_path, sn + se),
                        dpi=plt.rcParams['figure.dpi'], transparent=True)


# %% Playground for ANOVA testing and ZETA testing


# for given ROI, bins containing all stim values across all trials

# conds, rois, reps, frames

r = 0

reshape_test = data[m][:, r, :, idx_stim].swapaxes(0, 1).reshape(n_conds, n_samp_stim * n_reps)
test = np.full((n_conds, n_samp_stim * n_reps), np.nan)
for c in range(n_conds):
    test[c,:] = np.ravel(data[m][c, r, :, idx_stim])

np.allclose(np.nanmean(reshape_test, axis=1), np.nanmean(test, axis=1))

    
reshape_test2 = data[m][:, :, :, idx_stim].swapaxes(0, 1).reshape(n_ROIs, n_conds, n_samp_stim * n_reps)  # FAIL
reshape_test2 = data[m][:, :, :, idx_stim].reshape(n_ROIs, n_conds, n_reps * n_samp_stim)  # FAIL
reshape_test2 = np.vstack(np.moveaxis(data[m][:, :, :, idx_stim], [0, 1], [2, 3])).swapaxes(0, 2)
test2 = np.full((n_ROIs, n_conds, n_samp_stim * n_reps), np.nan)
for r in range(n_ROIs):
    for c in range(n_conds):
        # test2[r, c,:] = np.ravel(data[m][c, r, :, idx_stim])
        test2[r, c,:] = data[m][c, r, :, idx_stim].flatten()


np.allclose(np.nanmean(reshape_test2, axis=(0,1)), np.nanmean(test2, axis=(0,1)))

above_threshold = np.where(np.abs(dprime[m]) >= threshold_dprime)[0]



# %% Plot stimulus images

# n_subconds = len(cond_subset)

# if n_conds % 20 == 0:
#     n_cols = 20
#     n_rows = int(n_subconds / 20)
# else:
#     n_cols = 20
#     n_rows = round(n_subconds / 20)

# fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows,
#                         figsize=(n_cols * 512 / plt.rcParams['figure.dpi'], n_rows * 512 / plt.rcParams['figure.dpi']),
#                         layout='constrained')
# for row in range(n_rows):
#     for col in range(n_cols):
#         i = col + (row * n_cols)
#         ax = axs[row, col]
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.axis('off')
#         imp = os.path.join(stimimage_path, data[cond_idx[i]]['imagename'])
#         ax.imshow(plt.imread(imp))
# if saving:
#     sn = save_pfix + '_StimulusImages'
#     for se in save_ext:
#         fig.savefig(os.path.join(save_path, sn + se),
#                     dpi=plt.rcParams['figure.dpi'], transparent=True)


# %% Other approaches for measuring/approximating tuning

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
