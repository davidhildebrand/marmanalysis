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
# from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import least_squares as scipy_leastsquares
from scipy.stats import binned_statistic as scipy_binned_statistic
# from scipy.signal import find_peaks as find_peaks
from skimage import exposure, util
import socket
from warnings import warn

import filters
from metadata import default_metadata
import parsers
# import plots


# %% Settings

# Orientation-Selectivity Index (OSI) threshold
# based on ...
#   https://doi.org/ ...
# "..."
threshold_osi = 0.15

threshold_cellprob = 0.0
threshold_Zscore = 0.5

# Plotting parameters
plot_eyecal = False
plt.rcParams['figure.dpi'] = 600

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
# save_path = r'F:\Data\analysis'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # Louwho
animal_str = 'Louwho'

# # 20230906 MT chamber ribo-jGCaMP8s
# date_str = '20230906'
# session_str = '202924tUTC_SP_depth300um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow070p8mW_stimDriftGratings8dirFF'

# # 20240604d V1/V2 chamber cytosolic jGCaMP8s and soma-jGCaMP8s
# date_str = '20240604d'
# session_str = '174148tUTC_SP_depth200um_fov1060x1000um_res2p00x2p00umpx_fr09p147Hz_pow020p5mW_stimMultimodalGratings'
# session_str = '182603tUTC_SP_depth300um_fov2226x2400um_res3p00x3p00umpx_fr04p161Hz_pow029p7mW_stimMultimodalGratings'

# # 20240613d V1/V2 chamber cytosolic jGCaMP8s
# date_str = '20240613d'
# session_str = '202140tUTC_SP_depth200um_fov2412x2500um_res2p48x2p50umpx_fr02p605Hz_pow024p5mW_stimMultimodalGratings'
# session_str = '220609tUTC_SP_depth200um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow019p7mW_stimMultimodalGratings'
# session_str = '222646tUTC_SP_depth200um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow019p7mW_stimMultimodalGratings'

# # 20240615d V1/V2 chamber cytosolic jGCaMP8s
date_str = '20240615d'
# session_str = '204601tUTC_SP_depth050um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow010p5mW_stimMultimodalGratings'
# session_str = '210108tUTC_SP_depth060um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow010p5mW_stimMultimodalGratings'
# session_str = '211615tUTC_SP_depth060um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow010p5mW_stimMultimodalGratings'
session_str = '214336tUTC_SP_depth070um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow010p5mW_stimMultimodalGratings'
# session_str = '215423tUTC_SP_depth080um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow011p5mW_stimMultimodalGratings'
# session_str = '220821tUTC_SP_depth090um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow011p5mW_stimMultimodalGratings'
# session_str = '222114tUTC_SP_depth100um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow012p5mW_stimMultimodalGratings'

# # 20240616d V1/V2 chamber cytosolic jGCaMP8s
# date_str = '20240616d'
# session_str = '131801tUTC_SP_depth110um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow013p2mW_stimMultimodalGratings'
# session_str = '132914tUTC_SP_depth120um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow014p4mW_stimMultimodalGratings'
# session_str = '133951tUTC_SP_depth130um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow014p4mW_stimMultimodalGratings'
# session_str = '135134tUTC_SP_depth140um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow015p2mW_stimMultimodalGratings'
# session_str = '140205tUTC_SP_depth150um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow016p6mW_stimMultimodalGratings'
# session_str = '143448tUTC_SP_depth160um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow017p2mW_stimMultimodalGratings'
# session_str = '145005tUTC_SP_depth170um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow018p1mW_stimMultimodalGratings'
# session_str = '150902tUTC_SP_depth180um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow019p1mW_stimMultimodalGratings'
# session_str = '152906tUTC_SP_depth190um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow020p6mW_stimMultimodalGratings'
# session_str = '155356tUTC_SP_depth200um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow021p6mW_stimMultimodalGratings'
# session_str = '162824tUTC_SP_depth210um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow023p1mW_stimMultimodalGratings'
# session_str = '163813tUTC_SP_depth220um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow024p4mW_stimMultimodalGratings'
# session_str = '165418tUTC_SP_depth230um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow025p6mW_stimMultimodalGratings'
# session_str = '171634tUTC_SP_depth240um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow027p1mW_stimMultimodalGratings'
# session_str = '172850tUTC_SP_depth250um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow028p4mW_stimMultimodalGratings'
# session_str = '181333tUTC_SP_depth260um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow029p9mW_stimMultimodalGratings'
# session_str = '182423tUTC_SP_depth270um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow031p5mW_stimMultimodalGratings'
# session_str = '184918tUTC_SP_depth280um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow047p5mW_stimMultimodalGratings'

# # 20240617d V1/V2 chamber cytosolic jGCaMP8s
# date_str = '20240617d'
# session_str = '133252tUTC_SP_depth290um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow050p4mW_stimMultimodalGratings'
# session_str = '134309tUTC_SP_depth290um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow050p4mW_stimMultimodalGratings'
# session_str = '135719tUTC_SP_depth300um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow052p6mW_stimMultimodalGratings'
# session_str = '142921tUTC_SP_depth200um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow030p5mW_stimMultimodalGratings'
# session_str = '144307tUTC_SP_depth310um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow055p8mW_stimMultimodalGratings'
# session_str = '145400tUTC_SP_depth320um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow058p4mW_stimMultimodalGratings'
# session_str = '150615tUTC_SP_depth340um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow065p5mW_stimMultimodalGratings'
# session_str = '151619tUTC_SP_depth360um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow073p5mW_stimMultimodalGratings'
# session_str = '152626tUTC_SP_depth380um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow082p2mW_stimMultimodalGratings'
# session_str = '153610tUTC_SP_depth400um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow091p6mW_stimMultimodalGratings'
# session_str = '155205tUTC_SP_depth420um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow103p1mW_stimMultimodalGratings'
# session_str = '162035tUTC_SP_depth440um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow114p5mW_stimMultimodalGratings'
# session_str = '163849tUTC_SP_depth390um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow086p8mW_stimMultimodalGratings'
# session_str = '165312tUTC_SP_depth370um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow077p4mW_stimMultimodalGratings'
# session_str = '172227tUTC_SP_depth350um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow069p2mW_stimMultimodalGratings'
# session_str = '174040tUTC_SP_depth350um_fov0636x1500um_res2p00x2p00umpx_fr10p334Hz_pow069p2mW_stimMultimodalGratings'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if 'save_path' not in locals():
    save_path = ''
if 'savepath_str' not in locals():
    savepath_str = ''
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
filestr_eyecal_aidata = '*_EyeTrackingCalibration_AIdata.p'
if 'dirstr_suite2p' not in locals():
    dirstr_suite2p = 'suite2p*'
dirstr_suite2p_plane = 'plane0'


# %% Define functions and classes


deg_symbol = u'\N{DEGREE SIGN}'


def Rf(params, T):
    # Based on Pattadkal et al Priebe 2022 bioRxiv
    #     https://doi.org/10.1101/2022.06.23.497220
    Tpref = params[0]  # rad, preferred direction
    beta = params[1]  # 'tuning width factor'
    c = params[2]  # baseline
    a1 = params[3]  # peak 1 maxiumum amplitude
    a2 = params[4]  # peak 2 maxiumum amplitude
    R = (a1 * np.exp(beta * np.cos(T - Tpref))) + \
        (a2 * np.exp(beta * np.cos(np.pi + T - Tpref))) + \
        c
    return R


def gf(params, T):
    # Based on Fahey et al Tolias 2019 bioRxiv.
    #   https://doi.org/10.1101/745323
    Tpref = params[0]  # rad, preferred direction
    w = params[1]  # peak concentration or 'tuning width factor'
    g = np.exp(-w * (1 - np.cos(T - Tpref)))
    return g


def vf(params, T):
    a0 = params[2]  # baseline
    a1 = params[3]  # peak 1 maxiumum amplitude
    a2 = params[4]  # peak 2 maxiumum amplitude
    v = a0 + (a1 * gf(params, T)) + (a2 * gf(params, T - np.pi))
    return v


def dsi_model(params, T):
    r = Rf(params, T)  # Pattadkal et al Priebe 2022
    # r = vf(params, T)  # Fahey et al Tolias 2019
    return r


def dsi_objective(params, T, measRs):
    predRs = dsi_model(params, T)
    mse = np.square(np.subtract(predRs, measRs)).mean()
    return mse


def calculate_dsi(xs, ys, unit='deg', plotting=False, debugging=False):
    if unit == 'deg':
        thetas = np.radians(xs)
    elif unit == 'rad':
        thetas = xs
    measRs = ys

    # params = [Tpref, w/beta, a0, a1, a2]
    guess_baseline = np.mean(measRs)
    guess_amplitude = np.percentile(measRs, 90)
    guess = [(np.pi / 2),
             np.radians(360 / 8),
             guess_baseline,
             guess_amplitude,
             guess_amplitude]

    # TODO: check fit success?
    result = scipy_minimize(dsi_objective, guess, args=(thetas, measRs), method='L-BFGS-B')
    fit = result['x']
    Tpref_f = fit[0]
    w_f = fit[1]
    a0_f = fit[2]
    a1_f = fit[3]
    a2_f = fit[4]

    # based on Pattadkal etal Priebe 2022 bioRxiv
    # "The location of the peak of this fitted tuning curve is used as the
    # preferred direction of each cell."
    fit_curve = dsi_model(fit, np.radians(np.arange(0, 360)))
    max_peak_arg = fit_curve.argmax()
    # peak_locs = find_peaks(fit_curve, height=0)[0]
    # peaks = fit_curve[peak_locs]
    # max_peak_loc = peak_locs[peaks.argmax()]
    Tpref = np.radians(max_peak_arg)

    # TODO: separate to separate function
    # TODO: calculate and plot mean +/- stderr
    if plotting:
        plt.figure()
        plt.scatter(np.degrees(thetas), measRs, s=4, marker='.', facecolors='none', edgecolors='k')
        # plt.scatter(np.degrees(thetas), measRs, s=4, marker='.', facecolors='none', edgecolors='k')
        plt.plot(dsi_model(fit, np.radians(np.arange(0, 360))))
        plt.axvline(np.degrees(Tpref), color='m')
        ax = plt.gca()
        ax.set_xlabel('Direction (' + deg_symbol + ')', fontsize=12)
        ax.set_ylabel('dF/F', fontsize=12)
        # ax.set_xlim((0,360))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([d for d in range(0, 360, np.diff(np.degrees(thetas)).max().astype('int'))])
        # ax.set_xticklabels(['', 0, '', 2, ''])
        # plt.scatter(thetas, measRs, s=4, facecolors='none', edgecolors='k')
        # plt.plot(dsi_model(fit, np.arange(0, 2*np.pi))) #np.radians(np.arange(0, 360))))
        # plt.axvline(Tpref, color='m')
        plt.show()

    dT = thetas
    Rn = measRs + np.abs(np.min(measRs))
    dR = Rn
    # DSI based on Pattadkal etal Priebe 2022 bioRxiv
    dsi = np.sqrt(np.sum(dR * np.sin(dT))**2 + np.sum(dR * np.cos(dT))**2) / np.sum(dR)
    if unit == 'deg':
        Tpref = np.degrees(Tpref)
        Tpref_f = np.degrees(Tpref_f)

    if debugging:
        print('calculate_dsi dsi={:.2f} Tpref={} '.format(dsi, Tpref) +
              'Tpref_f={:.2f} w={:.2f} a0={:.2f} '.format(Tpref_f, w_f, a0_f) +
              'a1={:.2f} a2={:.2f} max_peak_arg={:.2f}'.format(a1_f, a2_f, max_peak_arg))

    return dsi, Tpref

# tex = "sin",
# mask = "none",
# units = None,
# anchor = "center",
# pos = (0.0, 0.0),
# size = None,
# sf = None,
# ori = 0.0,
# phase = (0.0, 0.0),
# texRes = 128,
# rgb = None,
# dkl = None,
# lms = None,
# color = (1.0, 1.0, 1.0),
# colorSpace = 'rgb',
# contrast = 1.0,
# opacity = None,
# depth = 0,
# rgbPedestal = (0.0, 0.0, 0.0),
# interpolate = False,
# draggable = False,
# blendmode = 'avg',
# name = None,
# autoLog = None,
# autoDraw = False,
# maskParams = None

# 'grating_tex': None,
# 'grating_mask': None,
# 'grating_units': None,
# 'grating_anchor': None,
# 'grating_pos': None,
# 'grating_size': None,
# 'grating_sf': None,
# 'grating_ori': None,
# 'grating_phase': None,
# 'grating_texRes': None,
# 'grating_color': None,
# 'grating_colorSpace': None,
# 'grating_contrast': None,
# 'grating_opacity': None,
# 'grating_interpolate': None,

# Load stimulus information

# # *** TODO load from a pickle file or pandas frame instead of a text log

if session_log is not None:
    lines = session_log.splitlines()
    trialdata = parsers.parse_log_stim_dots_orig(session_log)
else:
    raise RuntimeError('Could not find session log file.')

trialdata = {}
tmp_cond = None
tmp_f = None
tmp_acqfr = None
tmp_stimtimestr = ''
stimtime_mode = False
tmp_isitimestr = ''
isitime_mode = False
# 41.9371         EXP     trial 0, stim start, grating, full field, drifting, cond=5, ori=225.0, tex=sin, size=[75.67137421 75.67137421], sf=[1.2 0. ], tf=4, mask=None, contrast=1.0, acqfr=222
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

    if 'stim start' not in line:
        continue
    # print(line)
    col = line.split('trial')
    if not col:
        continue
    subcol = [sc.strip() for sc in col[1].split(',')]
    tmp_trial = int(subcol[0].strip())
    tmp_cond = int(subcol[5].split('=')[1].strip())

    tmp_dir = float(subcol[13].split('=')[1].strip())
    tmp_acqfr = int(subcol[20].split('=')[1].strip())
    trialdata[tmp_trial] = {'cond': tmp_cond,
                            'dir': tmp_dir,
                            'acqfr': tmp_acqfr}

if tmp_stimtimestr != '':
    s_si = tmp_stimtimestr.find('[')
    s_ei = tmp_stimtimestr.find(']')
    stimtimes = np.fromstring(tmp_stimtimestr[s_si+1:s_ei], sep=' ')
    dur_stim = np.round(np.mean(stimtimes), 2)
else:
    warn('Could not automatically detect stimulus duration, assuming default value of 1.0 sec.')
    dur_stim = 1.0

if tmp_isitimestr != '':
    s_si = tmp_isitimestr.find('[')
    s_ei = tmp_isitimestr.find(']')
    isitimes = np.fromstring(tmp_isitimestr[s_si+1:s_ei], sep=' ')
    dur_isi = np.round(np.min(isitimes), 2)
else:
    warn('Could not automatically detect interstimulus duration, assuming default value of 1.0 sec.')
    dur_isi = 1.0

# *** TODO automatically identify stimulus duration
# stimframes = int(np.ceil(2 * md['framerate']))
# isiframes = round(1 * md['framerate'])
# trialframes = n_samp_isi + n_samp_stim + n_samp_isi
# # *** TODO automatically identify stimulus duration
# dur_stim = 2.0
# dur_isi = 1.0
n_samp_stim = int(np.ceil(dur_stim * md['framerate']))
n_samp_isi = int(np.round(dur_isi * md['framerate']))
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

# trialdataarr = [trial_idx, cond, ori, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['dir'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)
all_stim_start_frames = trialdataarr[:, 2]

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,1])
n_conds = len(conds)
n_trials = int(len(trialdata) / n_conds)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdataarr[:,0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]


#%% Organize and average fluorescence traces

# F__by_cond = [roi, cond, t, F]
FdFF_by_cond = np.full([n_ROIs, n_conds, n_trials, n_samp_isi+n_samp_stim+n_samp_isi], np.nan)
FdFF_by_cond_top_decile = np.full([FdFF.shape[0], n_conds], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            fr_start = acqfr_by_conds[c][t] - n_samp_isi
            fr_end = acqfr_by_conds[c][t] + n_samp_stim + n_samp_isi
            if fr_start < 0 and t == 0:
                warn('Period before first trial was shorter than inter-stimulus interval.' + \
                     'Copied first present value to prevent error. ' + \
                     'But in the future this trial should be excluded.')
                n_missing = abs(fr_start)
                FdFF_by_cond[r, c, t, 0:n_missing] = np.array([FdFF[r, 0],] * n_missing).transpose()
                fr_start = 0
                FdFF_by_cond[r, c, t, n_missing:n_samp_trial] = FdFF[r, fr_start:fr_end]
                continue
            FdFF_by_cond[r,c,t,:] = FdFF[r, fr_start:fr_end]
FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,n_samp_isi:(n_samp_isi+n_samp_stim)]
FdFF_by_cond_meanR = np.mean(FdFF_by_cond_Rstim, axis=2) #mean across trials and selecting stimulus window
#FFdFF_by_cond_meanR = FdFF_by_cond_Rstim.reshape([FdFF_by_cond_Rstim.shape[0], FdFF_by_cond_Rstim.shape[1], -1]) # susceptible to noise

# F__by_cond = [roi, cond, t, F]
Fzsc_by_cond = np.full([n_ROIs, n_conds, n_trials, n_samp_isi+n_samp_stim+n_samp_isi], np.nan)
Fzsc_by_cond_top_decile = np.full([n_ROIs, n_conds], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            fr_start = acqfr_by_conds[c][t] - n_samp_isi
            fr_end = acqfr_by_conds[c][t] + n_samp_stim + n_samp_isi
            if fr_start < 0 and t == 0:
                warn('Period before first trial was shorter than inter-stimulus interval.' +
                     'Copied first present value to prevent error. ' +
                     'But in the future this trial should be excluded.')
                n_missing = abs(fr_start)
                Fzsc_by_cond[r, c, t, 0:n_missing] = Fzsc[r, 0]
                fr_start = 0
                Fzsc_by_cond[r, c, t, n_missing:n_samp_trial] = Fzsc[r, fr_start:fr_end]
                continue
            Fzsc_by_cond[r,c,t,:] = Fzsc[r, fr_start:fr_end]
            Fzsc_by_cond[r,c,t,:] = Fzsc[r,(acqfr_by_conds[c][t]-n_samp_isi):(acqfr_by_conds[c][t]+n_samp_stim+n_samp_isi)]
Fzsc_by_cond_Rstim = Fzsc_by_cond[:,:,:,n_samp_isi:(n_samp_isi+n_samp_stim)]
Fzsc_by_cond_meanR = np.mean(Fzsc_by_cond_Rstim, axis=2) #mean across trials and selecting stimulus window


#%% Compute direction selectivity index and find preferred direction tuning angle

# dsiT(__by_roi) = [roi, [dsi, T]]
Ts = np.repeat(conds, n_trials)
# distxs = np.repeat(conds, n_trials)
Rs = np.full([n_ROIs, (n_conds * n_trials)], np.nan)
dsiT = np.full([n_ROIs, 2], np.nan)
# for r in range(10):
#     Rs[r] = np.ravel(np.mean(FdFF_by_cond_Rstim[r], axis=2))
#     dsiT[r] = calculate_dsi(Ts, Rs[r], plotting=True)
for r in range(n_ROIs):
    Rs[r] = np.ravel(np.mean(FdFF_by_cond_Rstim[r], axis=2))
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
print('DSI tuning threshold: {}' .format(dsi_tuning_thresh))
tunidx_dsi = dsiT[:, 0]
DSI = dsiT[:, 0]
Tprefs = dsiT[:, 1]
Tprefs_norm = Tprefs / 360  # normalized to [0,1] range
tunidx_dsi_argsrt = np.argsort(tunidx_dsi)[::-1]
n_ROIs_tuned = np.argwhere(tunidx_dsi[tunidx_dsi_argsrt] >= dsi_tuning_thresh).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


# %% Plot histogram for the number of ROIs with corresponding tuning values

f_hist0 = plt.figure()
plt.hist(tunidx_dsi, bins=100)
plt.xlabel('Direction-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([0, 1])
plt.axvline(dsi_tuning_thresh, color='m')
# plt.axvline(-dsi_tuning_thresh, color='m')
f_hist0.show()
if saving:
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(dsi_tuning_thresh).replace('.', 'p') + '.svg'
    f_hist0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(dsi_tuning_thresh).replace('.', 'p') + '.png'
    f_hist0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)

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

plot_map(ROIs, Tprefs_norm, DSI, tuning_thresh=dsi_tuning_thresh, title=title_str,
         fov_size=fov_size, circular=True, ref_image=fov_image, save_path=save_path)


# TODO *** note that the >= might not be general (e.g. with FSI)
tuned_index = DSI >= dsi_tuning_thresh
ROIs_tuned = ROIs[tuned_index]
tuning_tuned = Tprefs[tuned_index]
tuning_mag_tuned = DSI[tuned_index]

if Tprefs_norm.max() > 1:
    warn(UserWarning('Provided tuning index has out-of-range values > 1.'))

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
# import scipy
# mport itertools
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

def dirdist_objective(params, xs, measured_dirdiff):
    predicted_dirdiff = dir_dist_dep_exp_equation(params, xs)
    mse = np.square(np.subtract(predicted_dirdiff, measured_dirdiff)).mean()
    return mse


# Calculate median values for 25 um distance bins
w_bin_um = 25
n_bins = int(np.ceil(distprefs[:, 0].max() / w_bin_um))
bin_edges = np.linspace(0, n_bins * w_bin_um, n_bins + 1)
bin_centers = np.linspace(w_bin_um / 2, (n_bins * w_bin_um) - (w_bin_um / 2), n_bins)
bin_medians, _, _ = scipy_binned_statistic(distprefs[:, 0], distprefs[:, 1], statistic='median', bins=bin_edges)
bin_stds, _, _ = scipy_binned_statistic(distprefs[:, 0], distprefs[:, 1], statistic='std', bins=bin_edges)

# TODO convert all this to take radians like other functions, then convert to deg

# params = [C, A, k]
guess = [70, 80, 0.018]
# result = scipy_minimize(dirdist_objective, guess, args=(distprefs[:, 0], distprefs[:, 1]), method='L-BFGS-B')
# result = scipy_minimize(dirdist_objective, guess, args=(bin_centers, bin_medians), method='Nelder-Mead')  # , method='L-BFGS-B')
result = scipy_leastsquares(dirdist_objective, guess, args=(distprefs[:, 0], distprefs[:, 1]), max_nfev=10000)
fit = result['x']
C_f = fit[0]
A_f = fit[1]
k_f = fit[2]
ddxs = np.linspace(0, np.round(np.max(bin_edges)))
ddys = C_f - (A_f * np.exp(-k_f * ddxs))

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
ax.set_xlim((0, 500))
ax.set_ylim((0, 180))

# Plot all pairs of direction difference and distance difference
ax.scatter(distprefs[:, 0], distprefs[:, 1], marker='.', s=1, edgecolor='none')

# Plot median values for 25um distance bins
# ax.scatter(bin_centers, bin_medians, marker='o', s=5, edgecolor='k', facecolor='w')
ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
            markeredgecolor='k', markerfacecolor='w', markersize=5, capsize=0,
            fmt='o', elinewidth=1, ecolor='k')
# TODO investigate whether it is an issue that direction difference is only accurate to 1º

ax.plot(ddxs, ddys)

plt.show()



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
    fig.suptitle('roi {} ({})'.format(r, ridx), fontsize=12)
    axes = fig.subplots(nrows=2, ncols=n_conds)
    for c in range(n_conds):
        ax = axes[0, c]
        ax.set_title(str(conds[c]) + deg_symbol, fontsize=10)
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
        ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
                     np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
        for t in range(n_trials):
            ax.plot(range(n_samp_trial), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(n_samp_trial), np.mean(Fzsc_by_cond[ridx,c,:,:], axis=0), color='tab:green')
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
        ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.1,
                     np.max(FdFF_by_cond[ridx,:,:,:]) + 0.1))
        for t in range(n_trials):
            ax.plot(range(n_samp_trial), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(n_samp_trial), np.mean(FdFF_by_cond[ridx,c,:,:], axis=0), color='tab:blue')    
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

# calculate_dsi(Ts, Rs[ridx], plotting=True, debugging=True)

# %%
# Frois_peak_frame = np.argmax(Frois, axis=1)
# y_height = 0
# plt.figure()
# for n in range(20):
#     r = np.random.randint(Frois.shape[0])
#     if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
#         plt.plot(np.linspace(1,18+36,18+36)-19, (Frois[r, Frois_peak_frame[r]-18:Frois_peak_frame[r]+36]) + y_height)
#         y_height += 0.5

# plt.xlabel('Frame # (@'+str(md['framerate'])+'Hz)')
# plt.yticks(range(int(np.ceil(y_height))))
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.axvline(x=0)
# plt.title('Peak ' + normalize + ' for 20 neurons')


# y_height = 0
# plt.figure()
# for n in range(20):
#     plt.plot( np.linspace(1,Frois.shape[1],Frois.shape[1]), ( Frois[ np.random.randint(Frois.shape[0]) ,:] ) + y_height, linewidth=0.2)
#     y_height += 3

# plt.xlabel('Frame # (@'+str(md['framerate'])+'Hz)')
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.title( normalize + ' for 20 neurons')
