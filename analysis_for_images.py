#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
import copy  # TODO not sure if this is necessary
from datetime import datetime
from glob import glob
# import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
# import pandas as pd
import re
# from scipy.optimize import minimize as scipy_minimize
# from scipy.signal import find_peaks as find_peaks
from skimage import exposure, util
import socket
from warnings import warn


# TODO: exclude suite2p badframes
# TODO: exclude when eyes are not open
# TODO: set path for images to include in figures during processing
# TODO: plot neuron traces over all conds or cats


# Remove stale metadata
if 'md' in locals():
    md = dict()
    del md


# Read suite2p outputs

# OLD PD Cadbury 20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx
# filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/Cadbury/20221016d'
# filename = '20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx.log'
# pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/suite2p/plane0/'
# filepath = r'F:\Sync\Freiwald\MarmoScope\Analysis\Data\Cadbury\20221016d\SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW'
# filename = '20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx.log'
# pf = r'F:\Sync\Freiwald\MarmoScope\Analysis\Data\Cadbury\20221016d\SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW\suite2p\plane0'

# --  GOOD OLD Cadbury PD  20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx
animal_str = 'Cadbury'
date_str = '20221016d_olds2p'
date_str = '20221016d'
session_str = '152643tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p364Hz_pow059p0mW_stimImagesSongFOBonly'
md = dict()
md['framerate'] = 6.364
md['fov'] = dict()
md['fov']['resolution_umpx'] = np.array([1.0, 1.0])
md['fov']['w_px'] = 730
md['fov']['h_px'] = 730


# -- MAYBE Cadbury MD?BOD? 
#  - not sure what to make of this, area tuning looks very jumbled now
# animal_str = 'Cadbury'
# date_str = '20231003d'
# session_str = '165634tUTC_SP_depth200um_fov0730x0730um_res1p00x1p00umpx_fr06p365Hz_pow079p9mW_stimImagesFOBmin'

# -- DECENT Cadbury OBJ
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '153335tUTC_SP_depth200um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow060p1mW_stimImagesFOBmany'
# -- OKAY Cadbury OBJ 
# animal_str = 'Cadbury'
# date_str = '20231007d'
# session_str = '164258tUTC_SP_depth250um_fov2000x2000um_res2p74x2p74umpx_fr06p363Hz_pow069p8mW_stimImagesFOBmin'

# -- OKAY Cadbury OBJ
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '185135tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'
# -- OKAY Cadbury  OBJ
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '190745tUTC_SP_depth250um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow075p8mW_stimImagesFOBmin'
# -- MAYBE Cadbury OBJ?PV? 
# animal_str = 'Cadbury'
# date_str = '20231018d'
# session_str = '192426tUTC_SP_depth200um_fov1460x1460um_res2p00x2p00umpx_fr06p364Hz_pow061p2mW_stimImagesFOBmin'

# -- MAYBE Dali PD mixed with anisoStim
# animal_str = 'Dali'
# date_str = '20230606d'
# session_str = '135543tUTC_SP_depth300um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow091p0mW_stimImagesFOBsel230517dAniso'

# -- MAYBE Dali PD mixed with anisoStim
# animal_str = 'Dali'
# date_str = '20230608d'
# session_str = '124502tUTC_SP_depth350um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow092p7mW_stimImagesFOBsel230517dAniso'
# -- MAYBE Dali PD mixed with anisoStim
# animal_str = 'Dali'
# date_str = '20230608d'
# session_str = '134220tUTC_SP_depth400um_fov2628x2600um_res3p00x3p00umpx_fr04p484Hz_pow122p3mW_stimImagesFOBsel230517dAniso'


title_str = animal_str + '_' + date_str + '_' + session_str
mdfile_str = '*_metadata.pickle'
datafile_str = '*_00001.tif'
logfile_str = '*.log'
logfile_re = r'.*^((?!disptimes).)*$'  # Exclude log files whose names contain 'disptimes'
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
save_path = ''

session_path = os.path.join(base_path, animal_str, date_str, session_str)
mdfile_list = glob(os.path.join(session_path, mdfile_str))
datafile_list = [f for f in glob(os.path.join(session_path, datafile_str))
                 if not os.path.isdir(f)]
logfile_list = [f for f in glob(os.path.join(session_path, logfile_str))
                if re.search(logfile_re, f) and not os.path.isdir(f)]
suite2p_list = [d for d in glob(os.path.join(session_path, suite2p_str))
                if not os.path.isfile(d)]

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
    if len(logfile_list) > 1:
        warn('Found multiple log files, using the first one: {}'.format(logfile_list[0]))
    lf_path = logfile_list[0]
    if os.path.isfile(lf_path):
        lf = open(lf_path, 'r')
        log = lf.read()
        lf.close()
    else:
        raise RuntimeError('Could not find log file.')

if len(suite2p_list) > 0:
    if len(suite2p_list) > 1:
        warn('Found multiple suite2p folders, using the first one: {}'.format(os.path.basename(suite2p_list[0])))
    s2p_path = suite2p_list[0]
    s2p_plane_path = os.path.join(s2p_path, suite2p_plane_str)
    if not os.path.isdir(s2p_plane_path):
        raise RuntimeError('Could not load suite2p plane0 folder.')
else:
    raise RuntimeError('Could not find suite2p folder.')


# %
# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# [...] neurons (94%) were face selective (that is, face-selectivity index
# larger than 1/3 or smaller than -1/3, dotted lines).
fsi_tuning_thresh = 1 / 4
osi_tuning_thresh = 1 / 4

cell_probability_thresh = 0.00

plt.rcParams['figure.dpi'] = 600
dpi = plt.rcParams['figure.dpi']


s2p_iscell = np.load(os.path.join(s2p_plane_path, 'iscell.npy'))
s2p_F = np.load(os.path.join(s2p_plane_path, 'F.npy'))
s2p_stat = np.load(os.path.join(s2p_plane_path, 'stat.npy'), allow_pickle=True)
s2p_ops = np.load(os.path.join(s2p_plane_path, 'ops.npy'), allow_pickle=True).item()
fov_image = s2p_ops['meanImg']

# cellinds = np.where(s2p_iscell[:,0] == 1.0)[0]
cellinds = np.where(s2p_iscell[:, 1] >= cell_probability_thresh)[0]
tmpROIs = s2p_stat[cellinds]
Frois = s2p_F[cellinds]
ROIidx_excl = np.where(np.std(Frois, axis=1) == 0)
ROIidx_incl = np.delete(np.arange(Frois.shape[0]), ROIidx_excl)
ROIs = tmpROIs[ROIidx_incl]
Frois = Frois[ROIidx_incl]
fov_h = s2p_ops['Ly']
fov_w = s2p_ops['Lx']
fov_size = (fov_h, fov_w)  # rows/height/y, columns/width/x
n_frames = Frois.shape[1]

###
# Alternative approach to computing FdFF, likely from David Fitzpatrick's lab:
# Baseline fluorescence (F0) was calculated by applying a rank-order filter to
# the raw fluorescence trace (10th percentile) with a rolling time window of 60s.
n_ROIs = Frois.shape[0]
FdFF_raw = (Frois - np.mean(Frois, axis=1)[:, np.newaxis]) / np.mean(Frois, axis=1)[:, np.newaxis]
Fzsc_raw = (Frois - np.mean(Frois, axis=1)[:, np.newaxis]) / np.std(Frois, axis=1)[:, np.newaxis]

FdFF_raw_20pct = np.percentile(FdFF_raw, 0.2)
Fzsc_raw_20pct = np.percentile(Fzsc_raw, 0.2)
# FdFF_raw_norm = FdFF_raw + np.abs(np.nanmin(FdFF_raw))
# Fzsc_raw_norm = Fzsc_raw + np.abs(np.nanmin(Fzsc_raw))
FdFF_norm = FdFF_raw.copy()
FdFF_norm[FdFF_norm < FdFF_raw_20pct] = FdFF_raw_20pct
FdFF_norm = FdFF_norm + np.abs(np.nanmin(FdFF_norm))
Fzsc_norm = Fzsc_raw.copy()
Fzsc_norm[Fzsc_norm < Fzsc_raw_20pct] = Fzsc_raw_20pct
Fzsc_norm = Fzsc_norm + np.abs(np.nanmin(Fzsc_norm))

if save_path == '':
    saving = False
else:
    saving = True


# % Define functions

def plot_map(rois, tuning, tuning_mag, tuning_thresh=0, size=(512, 512),
             circular=False, image=None, scale_bar=False, um_per_px=None,
             n_neighbors=None, save_path: str = ''):
    # The values tuning and tuning_mag must be within [0,1].
    # 'circular' determines whether tuning has the same color for 0 and 1
    # (True for MT, False for auditory)
    # TODO **** implement scale bar?

    dpi = plt.rcParams['figure.dpi'] / 2
    h, w = size  # rows/height/y, columns/width/x
    fsize = w / float(dpi), h / float(dpi)

    # ##### TODO *** THIS DOES NOT GENERALIZE
    n_rois = len(rois)
    tuned = np.abs(tuning_mag) > tuning_thresh
    rois_tuned = rois[tuned]
    tuning_tuned = tuning[tuned]
    tuning_mag_tuned = tuning_mag[tuned]

    if tuning.max() > 1:
        warn(UserWarning('provided tuning index has values > 1 (out of range)'))

    assert len(rois_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
    n_rois_tuned = len(rois_tuned)

    f0 = plt.figure(figsize=fsize)
    ax = f0.add_axes((0, 0, 1, 1))
    plt.set_cmap('hsv')
    # plt.axis('off')
    ax.axis('off')
    ax.set_frame_on(False)
    if image is not None:
        ilow, ihigh = np.percentile(image, (1.0, 99.98))
        ref_f64 = util.img_as_float64(image)
        ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
        ref = ref_rescale
        canvas = np.stack((ref,) * 3, axis=-1)  # copy single channel to form RGB image
    else:
        canvas = np.zeros([h, w, 3], dtype=np.float64)  # create a color canvas with frame size

    for r in range(n_rois_tuned):
        roi = rois_tuned[r]
        ry = roi['ypix']
        rx = roi['xpix']
        if circular is True:
            canvas[ry, rx, :] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
        else:
            for rgb in range(3):
                canvas[ry, rx, rgb] = abs(1 - 2 * abs(tuning_tuned[r] / 1.5 - rgb * 1 / 3))  # * tuning_mag[r]
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    # plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))  #, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot_FSIzsc_thresh' + \
                    '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
                    '_tuned{}of{}'.format(n_rois_tuned, n_rois) + \
                    '.png'
        f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)

    # Plot colorbar or colorwheel
    if circular is True:
        f1 = plt.figure()
        plt.set_cmap('hsv')
        # see https://stackoverflow.com/questions/62531754/how-to-draw-a-hsv-color-wheel-using-matplotlib
        # for color wheel options including saturation
        ax0 = f1.add_axes((0, 0, 1, 1), polar=True, frameon=False)
        ax0.set_axis_on()
        ax0.set_rticks([])
        ax0.set_xticks([0, np.pi / 2])
        ax0.set_xticklabels(['0', '90'])
        ax0.grid(False)
        ax1 = f1.add_axes(ax0.get_position(), projection='polar')
        ax1._direction = 2 * np.pi  # This is a nasty hack - using the hidden field to
        #                           # multiply the values such that 1 become 2*pi
        #                           # this field is supposed to take values 1 or -1 only!!
        # Plot the colorbar onto the polar axis
        # note - use orientation horizontal so that the gradient goes around
        # the wheel rather than centre out
        norm = mpl.colors.Normalize(0.0, (2 * np.pi))
        cb = mpl.colorbar.ColorbarBase(ax1,
                                       cmap=mpl.cm.get_cmap('hsv'),
                                       norm=norm,
                                       orientation='horizontal')
        # aesthetics - get rid of border and axis labels
        cb.outline.set_visible(False)
        ax1.set_axis_off()
        ax1.set_rlim([-1, 1])
        f1.show()
        if save_path != '':
            now = datetime.now()
            dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
            save_name = dt + '_ROIplot_FSIzsc_thresh' + \
                        '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
                        '_tuned{}of{}'.format(n_rois_tuned, n_rois) + \
                        '_legend.png'
            f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)

    # Santi original
    # # create colormap for reference
    # f3 = plt.figure()
    # x = np.linspace(0, 1, len(np.unique(tuning)) + 1)
    # y = np.linspace(1, 0, 101)
    # xx, yy = np.meshgrid(x, y)
    # canvas_colormap = np.ones([101, len(np.unique(tuning)) + 1, 3])
    # for rgb in range(3):
    #     if circular:
    #         canvas_colormap[:,:,rgb] = abs(1 - 2 * abs(xx - rgb * 1/3)) * yy
    #     else:
    #         canvas_colormap[:,:,rgb] = abs(1 - 2 * abs(xx/1.5 - rgb * 1/3)) * yy
    # plt.imshow(canvas_colormap, extent=[0,1,0,1], interpolation='none')
    # plt.xlabel('tuning')
    # plt.ylabel('tuning mag')
    # f2.show()

    if n_neighbors is not None:
        import seaborn as sns
        from sklearn import neighbors
        from sklearn.inspection import DecisionBoundaryDisplay

        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        X = np.empty((n_ROIs_tuned, 2))
        y = tuning_tuned * 8
        y = y.astype(int) + 1

        # tuned_logic = index_strength > strength_thresh
        # X = X[tuned_logic]
        # y = y[tuned_logic]
        for i in range(len(X)):
            X[i] = np.mean(rois_tuned[i]['xpix'][0]), h - np.mean(rois_tuned[i]['ypix'][0])

        # Create color maps
        for weights in ['uniform', 'distance']:
            # Create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, y)
            _, ax = plt.subplots()
            DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                cmap='Spectral',
                ax=ax,
                response_method='predict',
                plot_method='pcolormesh',
                shading='auto'
            )
            # Plot also the training points
            sns.scatterplot(
                x=X[:, 0],
                y=X[:, 1],
                hue=y,
                palette='Spectral',
                alpha=1.0,
                edgecolor='black',
                # legend='full',
                s=10
            )
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            plt.axis('square')
            # plt.title('(k = {}, weights = {})'.format(n_neighbors, weights))
        plt.show()


# % Extract stimulus information from log file

# *** TODO load from a pickle file or pandas frame instead of a text log
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
categories = {}
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
        if tmp_category not in categories:
            categories[tmp_category] = len(categories)
        tmp_catid = categories[tmp_category]
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

# # *** TODO automatically identify stimulus duration
# dur_stim = 2.0
# dur_isi = 1.0
n_samp_stim = int(np.ceil(dur_stim * md['framerate']))
n_samp_isi = int(np.round(dur_isi * md['framerate']))
n_samp_trial = n_samp_isi + n_samp_stim + n_samp_isi

categories = {v: k for k, v in categories.items()}
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
conditions = image_names

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

# FdFF = np.zeros(n_conds * n_trials, dtype=[('cond', 'S8'),
data = np.zeros(n_conds, dtype=[('cond', 'S8'),
                                ('cat', 'S8'),
                                ('id', 'S8'),
                                ('view', 'i2'),
                                ('roll', 'i2'),
                                ('FdFF', 'f4', (n_ROIs,
                                                n_trials,
                                                n_samp_trial)),
                                ('FdFFn', 'f4', (n_ROIs,
                                                 n_trials,
                                                 n_samp_trial)),
                                ('Fzsc', 'f4', (n_ROIs,
                                                n_trials,
                                                n_samp_trial)),
                                ('Fzscn', 'f4', (n_ROIs,
                                                 n_trials,
                                                 n_samp_trial)),
                                ('FdFF_meant', 'f4', (n_ROIs,
                                                      n_samp_trial)),
                                ('FdFFn_meant', 'f4', (n_ROIs,
                                                       n_samp_trial)),
                                ('Fzsc_meant', 'f4', (n_ROIs,
                                                      n_samp_trial)),
                                ('Fzscn_meant', 'f4', (n_ROIs,
                                                       n_samp_trial))
                                ])
data[:]['FdFF'] = np.nan
data[:]['FdFFn'] = np.nan
data[:]['Fzsc'] = np.nan
data[:]['Fzscn'] = np.nan
data[:]['FdFF_meant'] = np.nan
data[:]['FdFFn_meant'] = np.nan
data[:]['Fzsc_meant'] = np.nan
data[:]['Fzscn_meant'] = np.nan

#
# Fzsc = np.zeros(n_conds * n_trials, dtype=[('cond', 'S8'),
# Fzsc = np.zeros(n_conds, dtype=[('cond', 'S8'),
#                                ('cat', 'S8'),
#                                #('trial', int),
#                                ('R', 'f8', (n_ROIs,
#                                             n_trials,
#                                             n_samp_trial)),
#                                ('Rn', 'f8', (n_ROIs,
#                                              n_trials,
#                                              n_samp_trial))])

# Currently supports image sets:
# 'FOBmin_MarmOnly', 'FOBmin', 'FOBmany', 'Song_etal_Wang_2022_FOBonly'
for c in range(n_conds):
    tmp_cond = None
    tmp_cat = None
    tmp_id = None
    tmp_view = -32768
    tmp_roll = -32768
    imn = image_names[c]
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
                tmp_cond = bytes('fc{:02}'.format(nm), 'ascii')
                tmp_cat = b'face_ctn'
                tmp_id = bytes(nm, 'ascii')
                tmp_view = 0
                tmp_roll = 0
        else:
            warn('Could not recognize category or condition of image from filename.')
            tmp_cond = None
            tmp_cat = None
    elif image_set == 'Song_etal_Wang_2022_FOBonly':
        tmp_cond = bytes(conditions[c], 'ascii')
        match conditions[c][0]:
            case 'a':
                tmp_cat = b'animal'
            case 'o':
                tmp_cat = b'obj'
            case 'b':
                if conditions[c] == 'blank':
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
    # TODO: check this to see if truncated
    data[c]['cond'] = tmp_cond
    data[c]['cat'] = tmp_cat
    data[c]['id'] = tmp_id
    data[c]['view'] = tmp_view
    data[c]['roll'] = tmp_roll
    for t in range(n_trials):
        fr_start = fridx[c, t] - n_samp_isi
        fr_end = fridx[c, t] + n_samp_stim + n_samp_isi
        # if fr_start == -1 and t == 0:
        #     warn('Period before first trial was one shorter than inter-stimulus interval.' +
        #          'Copied first present value to prevent error. ' +
        #          'But in the future this trial should be excluded.')
        #     n_missing = np.abs(fr_start)
        #     data[c]['FdFF'][:, t, 0] = FdFF_raw[:, -1]
        #     data[c]['FdFFn'][:, t, 0] = FdFF_norm[:, -1]
        #     data[c]['Fzsc'][:, t, 0] = Fzsc_raw[:, -1]
        #     data[c]['Fzscn'][:, t, 0] = Fzsc_norm[:, -1]
        #     fr_start = 0
        #     data[c]['FdFF'][:, t, 1:n_samp_trial] = FdFF_raw[:, fr_start:fr_end]
        #     data[c]['FdFFn'][:, t, 1:n_samp_trial] = FdFF_norm[:, fr_start:fr_end]
        #     data[c]['Fzsc'][:, t, 1:n_samp_trial] = Fzsc_raw[:, fr_start:fr_end]
        #     data[c]['Fzscn'][:, t, 1:n_samp_trial] = Fzsc_norm[:, fr_start:fr_end]
        #     continue
        if fr_start < 0 and t == 0:
            warn('Period before first trial was shorter than inter-stimulus interval. ' +
                 'Copied first present value to prevent error. ' +
                 'But in the future this trial should be excluded.')
            n_missing = abs(fr_start)
            data[c]['FdFF'][:, t, 0:n_missing] = np.array([FdFF_raw[:, 0],] * n_missing).transpose()
            data[c]['FdFFn'][:, t, 0:n_missing] = np.array([FdFF_norm[:, 0],] * n_missing).transpose()
            data[c]['Fzsc'][:, t, 0:n_missing] = np.array([Fzsc_raw[:, 0],] * n_missing).transpose()
            data[c]['Fzscn'][:, t, 0:n_missing] = np.array([Fzsc_norm[:, 0],] * n_missing).transpose()
            fr_start = 0
            data[c]['FdFF'][:, t, n_missing:n_samp_trial] = FdFF_raw[:, fr_start:fr_end]
            data[c]['FdFFn'][:, t, n_missing:n_samp_trial] = FdFF_norm[:, fr_start:fr_end]
            data[c]['Fzsc'][:, t, n_missing:n_samp_trial] = Fzsc_raw[:, fr_start:fr_end]
            data[c]['Fzscn'][:, t, n_missing:n_samp_trial] = Fzsc_norm[:, fr_start:fr_end]
            continue
        if fr_end > n_frames:
            # TODO: support throwing away trials after imaging stops
            raise RuntimeError('Imaging was stopped before stimulus. ' +
                               'Handling this is not yet implemented.')
        data[c]['FdFF'][:, t, :] = FdFF_raw[:, fr_start:fr_end]
        data[c]['FdFFn'][:, t, :] = FdFF_norm[:, fr_start:fr_end]
        data[c]['Fzsc'][:, t, :] = Fzsc_raw[:, fr_start:fr_end]
        data[c]['Fzscn'][:, t, :] = Fzsc_norm[:, fr_start:fr_end]
        # if fr_start >= 0:
        #     data[c]['FdFF'][:, t, :] = FdFF_raw[:, fr_start:fr_end]
        #     data[c]['FdFFn'][:, t, :] = FdFF_norm[:, fr_start:fr_end]
        #     data[c]['Fzsc'][:, t, :] = Fzsc_raw[:, fr_start:fr_end]
        #     data[c]['Fzscn'][:, t, :] = Fzsc_norm[:, fr_start:fr_end]
        # elif fr_start < 0 and t == 0:
        #     # TODO: support throwing away trials with ISI before imaging
        #     warn('Period before first trial was shorter than inter-stimulus interval. ' +
        #          'Copied first present value to prevent error. ' +
        #          'But in the future this trial should be excluded.')
        #     n_missing = abs(fr_start)
        #     data[c]['FdFF'][:, t, 0:n_missing] = np.array([FdFF_raw[:, 0],] * n_missing).transpose()
        #     data[c]['FdFFn'][:, t, 0:n_missing] = np.array([FdFF_norm[:, 0],] * n_missing).transpose()
        #     data[c]['Fzsc'][:, t, 0:n_missing] = np.array([Fzsc_raw[:, 0],] * n_missing).transpose()
        #     data[c]['Fzscn'][:, t, 0:n_missing] = np.array([Fzsc_norm[:, 0],] * n_missing).transpose()
        #     fr_start = 0
        #     data[c]['FdFF'][:, t, n_missing:n_samp_trial] = FdFF_raw[:, fr_start:fr_end]
        #     data[c]['FdFFn'][:, t, n_missing:n_samp_trial] = FdFF_norm[:, fr_start:fr_end]
        #     data[c]['Fzsc'][:, t, n_missing:n_samp_trial] = Fzsc_raw[:, fr_start:fr_end]
        #     data[c]['Fzscn'][:, t, n_missing:n_samp_trial] = Fzsc_norm[:, fr_start:fr_end]
        # elif fr_end > n_frames:
        #     # TODO: support throwing away trials after imaging stops
        #     raise RuntimeError('Imaging was stopped before stimulus. ' +
        #                        'Handling this is not yet implemented.')
        # else:
        #     raise RuntimeError('Something went wrong when organizing the data.')
    data[c]['FdFF_meant'] = np.nanmean(data[c]['FdFF'], axis=1)
    data[c]['FdFFn_meant'] = np.nanmean(data[c]['FdFFn'], axis=1)
    data[c]['Fzsc_meant'] = np.nanmean(data[c]['Fzsc'], axis=1)
    data[c]['Fzscn_meant'] = np.nanmean(data[c]['Fzscn'], axis=1)


# del FdFF_raw, FdFF_norm, Fzsc_raw, Fzsc_norm


#
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

Fzsc_for_plot_bfo = np.array([Fzsc_allbodies_meanRstimall,
                              Fzsc_allfaces_meanRstimall,
                              Fzsc_allobjs_meanRstimall]).swapaxes(0, 1)

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
           
# OSIs(_by_roi) = [roi, osi]
OSIs_dFF = (FdFF_allobjs_meanRstimall - FdFF_allfaces_meanRstimall) / \
           (FdFF_allobjs_meanRstimall + FdFF_allfaces_meanRstimall)
OSIs_zsc = (Fzsc_allobjs_meanRstimall - Fzsc_allfaces_meanRstimall) / \
           (Fzsc_allobjs_meanRstimall + Fzsc_allfaces_meanRstimall)


# |FSI| threshold: 0.25
# Tuned ROIs: 894. Total ROIs: 6020.
# Percentage of tuned ROIs: 14.85%
# o_fsi_tuning_thresh = fsi_tuning_thresh
# o_FSIs_zsc = FSIs_zsc
# o_tunidx_fsi = tunidx_fsi

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


# % Define plotting function for face-body-object selective cells

def plot_ROIs_RGB(ROIs, RGB_ROIs, size=(512, 512), image=None, scale_bar=False, um_per_px=None,
                  n_neighbors=None, title:str='', save_path:str='', imn:str=''):
    dpi = plt.rcParams['figure.dpi'] / 2
    h, w = size  # rows/height/y, columns/width/x
    figsize = w / float(dpi), h / float(dpi)

    f0 = plt.figure(figsize=figsize)
    ax = f0.add_axes((0, 0, 1, 1))
    plt.set_cmap('hsv')
    # plt.axis('off')
    ax.axis('off')
    ax.set_frame_on(False)
    if image is not None:
        ilow, ihigh = np.percentile(image, (1.0, 99.98))
        ref_f64 = util.img_as_float64(image)
        ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
        ref = ref_rescale
        canvas = np.stack((ref,) * 3, axis=-1)  # copy single channel to form RGB image
    else:
        canvas = np.zeros([h, w, 3], dtype=np.float64)  # create a color canvas with frame size

    for r in range(len(ROIs)):
        ROI = ROIs[r]
        ry = ROI['ypix']
        rx = ROI['xpix']
        canvas[ry, rx, :] = RGB_ROIs[r]

    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    # plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))#, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    if title != '':
        ax.set_title(title, fontsize=2)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot_FSIzsc' + \
                    '_tuned{}_{}'.format(len(RGB_ROIs), imn) + \
                    '.png'
        f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)


# % Plot tuned cells with continuous tuning-wheel

# parameters
RGB_multiplier = 2.5
plotting_threshold_continuous = 0.25

# Fzsc_for_plot_continuous = copy.deepcopy(Fzsc_by_cat_meanRstimallnorm)
Fzsc_for_plot_continuous = copy.deepcopy(Fzsc_for_plot_bfo)

# Subtract the response to the least-tuned category (otherwise, an ROI that responds to all categories would show up as white)
Fzsc_least_tuned = np.min(Fzsc_for_plot_continuous, axis=1)
for col_i in range(3):
    Fzsc_for_plot_continuous[:, col_i] = Fzsc_for_plot_continuous[:, col_i] - Fzsc_least_tuned

Fzsc_for_plot_continuous[Fzsc_for_plot_continuous > 1] = 1  # Cap the RGB values to 1
Fzsc_for_plot_continuous = Fzsc_for_plot_continuous * RGB_multiplier  # This highlights ROIs with less tuning, at the expense of dynamic range
# Fzsc_for_plot_continuous[:, [0, 1, 2]] = Fzsc_for_plot_continuous[:, [key_bodies,
#                                                                       key_faces,
#                                                                       key_objs]]  # Swap orders to change colors.

# TODO : fix this because it gives more than the number stated as 
thresholding_logical_vector_continuous = np.max(Fzsc_for_plot_continuous,
                                                axis=1) > plotting_threshold_continuous  # Threshold so we don't plot un-tuned neurons (particularly important if using RGB_multiplier > 1)
# thresholding_logical_vector_continuous = ROIs_tuned_idx

ROIs_for_plot_continuous = ROIs[thresholding_logical_vector_continuous]
Fzsc_for_plot_continuous = Fzsc_for_plot_continuous[thresholding_logical_vector_continuous]


plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous,
              size=fov_size, image=fov_image, save_path=save_path)

# plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous,
#               size=fov_size, image=fov_image, title=title_str, save_path=save_path)

# plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous,
#               size=fov_size, image=fov_image, save_path=r'F:\Sync\Transient\Science\Conferences\20231111d-20231115d_SocietyForNeuroscience_AnnualMeeting_WashingtonDC\media')










# %% Plot tuned cells with discreete tuning-wheel

# parameters
plotting_threshold_discrete = 0.15
subtract_responses_to_other_stim = True
subtract_least_or_secondLeast_preferred_stim_responses = 0  # If set to 0, will subtract the responses to the least-preferred stim. If set to 1, will subtract the responses to the second-least (in this case, second-most) preferred stim


Fzsc_for_plot_discrete = copy.deepcopy(Fzsc_for_plot_bfo)

# Subtract to all responses the responses to the second-preferred stimulus
if subtract_responses_to_other_stim:
    for row_i in range(len(Fzsc_for_plot_discrete)):
        this_row = Fzsc_for_plot_discrete[row_i]
        response_to_non_preferred_stim = sorted(set(this_row))[
            subtract_least_or_secondLeast_preferred_stim_responses]  # This sorts the responses and selects the lowest(0) or second-lowest(1) response
        Fzsc_for_plot_discrete[row_i] = this_row - response_to_non_preferred_stim

# TODO : fix this it is thresholding zsc not FSI!
# thresholding_logical_vector_discrete = np.max(Fzsc_for_plot_discrete,
#                                               axis=1) > plotting_threshold_discrete  # Threshold
thresholding_logical_vector_discrete = ROIs_tuned_idx  # FSIs_zsc > fsi_tuning_thresh

ROIs_for_plot_discrete = ROIs[thresholding_logical_vector_discrete]
Fzsc_for_plot_discrete = Fzsc_for_plot_discrete[thresholding_logical_vector_discrete]

Fzsc_for_plot_preferredKey = np.argmax(Fzsc_for_plot_discrete, axis=1)
Fzsc_for_plot_discrete[:] = 0  # We will re-fill the preferredkeys with 1s in the folowwing for loop
for roi_i in range(len(Fzsc_for_plot_discrete)):
    Fzsc_for_plot_discrete[roi_i, Fzsc_for_plot_preferredKey[roi_i]] = 1

# Fzsc_for_plot_discrete[:, [0, 1, 2]] = Fzsc_for_plot_discrete[:, [key_bodies, key_faces,
#                                                                   key_objs]]  # Swap face indexes to be on the first column, making face-cells be red
# Fzsc_for_plot_discrete[:,[1,2]] = Fzsc_for_plot_discrete[:,[2,1]] #Swap face indexes to be on the first column, making face-cells be red

plot_ROIs_RGB(ROIs_for_plot_discrete, Fzsc_for_plot_discrete,
              size=fov_size, image=fov_image, save_path=save_path)


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

f0 = plt.figure()
plt.hist(FSIs_dFF, bins=100)
plt.xlabel('Face-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([-1, 1])
plt.axvline(fsi_tuning_thresh, color='m')
plt.axvline(-fsi_tuning_thresh, color='m')
f0.show()
if save_path != '':
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_FSIdFF_thresh' + \
                '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.svg'
    f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)
    save_name = dt + '_histogram_FSIdFF_thresh' + \
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
if save_path != '':
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_FSIzsc_thresh' + \
                '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.svg'
    f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)
    save_name = dt + '_histogram_FSIzsc_thresh' + \
                '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)


# %% 

# Non-zero hack
FdFF_absmin = -np.inf
for cd in np.unique(data['cond']):
    absmintmp = np.abs(np.min(np.nanmean(data[data['cond'] == cd]['FdFF_meant'][:, :, idx_stim], axis=(0, -1))))
    if absmintmp > FdFF_absmin:
        FdFF_absmin = absmintmp
Fzsc_absmin = -np.inf
for cd in np.unique(data['cond']):
    absmintmp = np.abs(np.min(np.nanmean(data[data['cond'] == cd]['Fzsc_meant'][:, :, idx_stim], axis=(0, -1))))
    if absmintmp > Fzsc_absmin:
        Fzsc_absmin = absmintmp

ImM_zsc = np.empty([n_ROIs_tuned, len(data['cond'][(data['cat'] == b'face_mrm')])])
ImSIs_zsc = np.empty([n_ROIs_tuned, len(data['cond'][(data['cat'] == b'face_mrm')])])
for cid, cim in enumerate(data['cond'][(data['cat'] == b'face_mrm')]):
    Fzsc_nowface_meanRstimall = np.nanmean(data[data['cond'] == cim]['Fzsc_meant'][:, :, idx_stim],
                                           axis=(0, -1)) + Fzsc_absmin
    Fzsc_otherfaces_meanRstimall = np.nanmean(data[data['cond'] != cim]['Fzsc_meant'][:, :, idx_stim],
                                              axis=(0, -1)) + Fzsc_absmin
    for r in range(n_ROIs_tuned):
        rti = ROIs_tuned_idx[r]
        ImM_zsc[r, cid] = Fzsc_nowface_meanRstimall[rti]
        ImSIs_zsc[r, cid] = (Fzsc_nowface_meanRstimall[rti] - Fzsc_otherfaces_meanRstimall[rti]) / \
                            (Fzsc_nowface_meanRstimall[rti] + Fzsc_otherfaces_meanRstimall[rti])

# idx_trial = range(n_samp_trial)
# Fzsc_allfaces_meanRstim = np.nanmean(data[(data['cat'] == b'face_mrm') & (data['view'] == 0) & (data['roll'] == 0)]['Fzsc_meant'][:, :, idx_trial],
#                                      axis=0) + Fzsc_absmin
# sorting_ind = ROIs_tuned_idx
# Fzsc_allfaces_meanRstim_sorted = Fzsc_allfaces_meanRstim[sorting_ind]
# # cats = categories_sorted

# plt.figure(dpi=1000)
# plt.imshow(Fzsc_allfaces_meanRstim_sorted, cmap='bwr')

# plt.clim(-1,1)
# #plt.title('2-D Heat Map in Matplotlib')
# #plt.colorbar()
# plt.tick_params(left=False)
# ax = plt.gca()
# ax.tick_params(left=False, right=False, labelleft=False)
# ax.set_xticks([c for c in range(0,60+1,20)])#, categories_sorted)
# ax.set_xticklabels([], fontsize=3, rotation=90)
# plt.show()


plt.figure(dpi=1000)
plt.imshow(ImM_zsc[range(20)], cmap='bwr')
#plt.clim(-1,1)
#plt.title('2-D Heat Map in Matplotlib')
plt.colorbar()
plt.tick_params(left=False, bottom=False)
ax = plt.gca()
ax.tick_params(left=False, right=False, bottom=False, labelleft=False)
ax.set_xticks([])  # c for c in range(0,20,20)])#, categories_sorted)
ax.set_xticklabels([], fontsize=3, rotation=90)
plt.xlabel('Face Image')
plt.ylabel('ROI')
plt.show()


# tunidx_fsi = ImSIs_zsc
# tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
# ROIs_tuned_idx = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).squeeze()
# n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).shape[0]
# pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
# print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
# print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


maxim = np.unravel_index(ImSIs_zsc[:,:].argmax(), ImSIs_zsc.shape)[1]

for it in range(20):
    ImSIs_zsc_argsort_imt = ImSIs_zsc[:, it].argsort()
    rois_sel = np.argwhere(np.abs(ImSIs_zsc[ImSIs_zsc_argsort_imt, it]) > fsi_tuning_thresh).squeeze()
    
    plot_ROIs_RGB(ROIs_for_plot_discrete[rois_sel], Fzsc_for_plot_discrete[rois_sel],
                  size=fov_size, image=fov_image, save_path=save_path, 
                  imn=data['cond'][(data['cat'] == b'face_mrm')][it].decode())


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
#     plot_map(ROIs, c * np.ones(FSIs_zsc.shape), FSIs_zsc, tuning_thresh=fsi_tuning_thresh,
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
