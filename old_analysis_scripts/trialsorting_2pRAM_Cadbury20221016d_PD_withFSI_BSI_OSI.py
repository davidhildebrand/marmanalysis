#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
#from scipy.optimize import minimize as scipy_minimize
#from scipy.signal import find_peaks as find_peaks
from skimage import exposure, util
from warnings import warn

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')


#%% Read suite2p outputs

filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/'
filename = '20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx.log'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/suite2p/plane0/'
save_path = ''
# save_path = '/Users/davidh/Data/Freiwald/Analysis/Cadbury/20221016d_2pRAM/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/'

acq_framerate = 6.36

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# [...] neurons (94%) were face selective (that is, face-selectivity index 
# larger than 1/3 or smaller than -1/3, dotted lines).
fsi_tuning_thresh = 1/4

cell_probability_thresh = 0 #.005

plt.rcParams['figure.dpi'] = 600
dpi = plt.rcParams['figure.dpi']

# *** TODO automatically identify stimulus duration
stimframes = int(np.ceil(2 * acq_framerate))
isiframes = round(1 * acq_framerate)
totalframes = isiframes + stimframes + isiframes

s2p_iscell = np.load(pf + 'iscell.npy')
s2p_F = np.load(pf + 'F.npy')
s2p_stat = np.load(pf + 'stat.npy', allow_pickle=True)
s2p_ops = np.load(pf + 'ops.npy', allow_pickle = True).item()
#s2p_ops['filelist']
ref_image = s2p_ops['meanImg']
#s2p_ops['refImg']

#cellinds = np.where(s2p_iscell[:,0] == 1.0)[0]
cellinds = np.where(s2p_iscell[:,1] >= cell_probability_thresh)[0]
tmpROIs = s2p_stat[cellinds]
Frois = s2p_F[cellinds]
ROIidx_excl = np.where(np.std(Frois, axis=1) == 0)
ROIidx_incl = np.delete(np.arange(Frois.shape[0]), ROIidx_excl)
ROIs = tmpROIs[ROIidx_incl]
Frois = Frois[ROIidx_incl]
fov_h = s2p_ops['Ly']
fov_w = s2p_ops['Lx']
fov_size = (fov_h, fov_w) # rows/height/y, columns/width/x

###
# Alternative approach to computing FdFF, likely from David Fitzpatrick's lab:
# Baseline fluorescence (F0) was calculated by applying a rank-order filter to 
# the raw fluorescence trace (10th percentile) with a rolling time window of 60s.
n_ROIs = Frois.shape[0]
FdFF = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.mean(Frois, axis=1)[:,np.newaxis]
Fzsc = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.std(Frois, axis=1)[:,np.newaxis]

if save_path == '':
    saving = False
else:
    saving = True


#%% Define functions

def plot_map(ROIs, tuning, tuning_mag, tuning_thresh=0, fov_size=(512,512), 
             circular=False, ref_image=None, scale_bar=False, um_per_px=None, 
             n_neighbors=None, save_path:str=''):
    # The values tuning and tuning_mag must be within [0,1].
    # 'circular' determines whether tuning has the same color for 0 and 1 
    # (True for MT, False for auditory)
    # TODO **** implement scale bar?
    
    dpi = plt.rcParams['figure.dpi'] / 2
    h, w = fov_size # rows/height/y, columns/width/x
    figsize = w / float(dpi), h / float(dpi)
    
    ###### TODO *** THIS DOES NOT GENERALIZE
    r_ROIs = len(ROIs)
    tuned = np.abs(tuning_mag) > tuning_thresh
    ROIs_tuned = ROIs[tuned]
    tuning_tuned = tuning[tuned]
    tuning_mag_tuned = tuning_mag[tuned]
    
    if tuning.max() > 1:
        warn(UserWarning('provided tuning index has values > 1 (out of range)'))
    
    assert len(ROIs_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
    n_ROIs_tuned = len(ROIs_tuned)
    
    f0 = plt.figure(figsize=figsize)
    ax = f0.add_axes([0, 0, 1, 1])
    plt.set_cmap('hsv')
    #plt.axis('off')
    ax.axis('off')
    ax.set_frame_on(False)
    if ref_image is not None:
        ilow, ihigh = np.percentile(ref_image, (1.0, 99.98))
        ref_f64 = util.img_as_float64(ref_image)
        ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
        ref = ref_rescale
        canvas = np.stack((ref,)*3, axis=-1) # copy single channel to form RGB image
    else:
        canvas = np.zeros([h, w, 3], dtype=np.float64) # create a color canvas with frame size

    for r in range(n_ROIs_tuned):
        ROI = ROIs_tuned[r]
        ry = ROI['ypix']
        rx = ROI['xpix']
        if circular is True:
            canvas[ry,rx,:] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
        else:
            for rgb in range(3):
                canvas[ry,rx,rgb] = abs(1 - 2 * abs(tuning_tuned[r] / 1.5 - rgb * 1/3)) #* tuning_mag[r]
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    #plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))#, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot_FSIzsc_thresh' + \
            '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
            '_tuned{}of{}'.format(n_ROIs_tuned, n_ROIs) + \
            '.png'
        f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    
    # Plot colorbar or colorwheel
    if circular is True:
        f1 = plt.figure()
        plt.set_cmap('hsv')
        # see https://stackoverflow.com/questions/62531754/how-to-draw-a-hsv-color-wheel-using-matplotlib
        # for color wheel options including saturation
        ax0 = f1.add_axes([0,0,1,1], polar=True, frameon=False)
        ax0.set_axis_on()
        ax0.set_rticks([])
        ax0.set_xticks([0, np.pi/2])
        ax0.set_xticklabels(['0', '90'])
        ax0.grid(False)
        ax1 = f1.add_axes(ax0.get_position(), projection='polar')
        ax1._direction = 2 * np.pi ## This is a nasty hack - using the hidden field to 
        #                                    ## multiply the values such that 1 become 2*pi
        #                                    ## this field is supposed to take values 1 or -1 only!!                 
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
        ax1.set_rlim([-1,1])
        f1.show()
        if save_path != '':
            now = datetime.now()
            dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
            save_name = dt + '_ROIplot_FSIzsc_thresh' + \
                '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
                '_tuned{}of{}'.format(n_ROIs_tuned, n_ROIs) + \
                '_legend.png'
            f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
       
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
            X[i] = np.mean(ROIs_tuned[i]['xpix'][0]), h - np.mean(ROIs_tuned[i]['ypix'][0])
            
        # Create color maps
        
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
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
                shading='auto')
        
            # #Plot also the training points
            sns.scatterplot(
                x=X[:,0],
                y=X[:,1],
                hue=y,
                palette='Spectral',
                alpha=1.0,
                edgecolor='black',
                #legend='full',
                s=10)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            plt.axis('square')
            #plt.title('(k = {}, weights = {})'.format(n_neighbors, weights))
        plt.show()
        

#%% Extract stimulus information from log file

# *** TODO load from a pickle file or pandas frame instead of a text log
file = open(filepath + os.path.sep + filename, 'r')
lines = file.read().splitlines()
file.close()

#37.1533         EXP     trial 0/240, stim start, image, cond=7, name=image7:b16.png, 
#path=/FreiwaldSync/MarmoScope/Stimulus/Images/Song_etal_Wang_2020_NatCommun/480288_equalized_RGBA_FOBonly/b16.png, 
#units=deg, pos=[0. 0.], size=[12.   7.2], ori=0.0, color=[1. 1. 1.], colorSpace=rgb, contrast=1.0, 
#opacity=1.0, texRes=512, acqfr=23, AI_data.shape=(1336, 5)
trialdata = {}
images = {}
categories = {}
for line in lines:
    if 'stim start' not in line or 'image' not in line:
        continue
    #print(line)
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
        if tmp_image not in images:
            images[tmp_image] = tmp_cond
        tmp_category = tmp_image[0]
        if tmp_category not in categories:
            categories[tmp_category] = len(categories)
        tmp_catid = categories[tmp_category]
    else:
        print('could not get image name from log entry')
    if 'acqfr' in subcol[15]:
        tmp_acqfr = int(subcol[15].split('=')[1].strip())
    else:
        print('could not get acqfr from log entry')
    trialdata[tmp_trial] = {'cond' : tmp_cond, #effectively image_id
                            'image' : tmp_image,
                            'category' : tmp_category, 
                            'catid' : tmp_catid,
                            'acqfr' : tmp_acqfr}
categories = {v: k for k, v in categories.items()}
images_filename = {v: k for k, v in images.items()}
images = {v: k.split('.')[0] for k, v in images.items()}

# trialdataarr[trial_idx] = [cond/imageid, category_id, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['catid'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)
all_stim_start_frames = trialdataarr[:,2]

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,0])
n_conds = len(conds)
n_trials = int(len(trialdata) / n_conds)
cats = np.unique(trialdataarr[:,1])
n_cats = len(cats)
conds_per_cat = int(n_conds / n_cats)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdataarr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]


#%% Organize and average fluorescence traces

# F__by_cond = [roi, cond, trial, frame]
FdFF_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            FdFF_by_cond[r,c,t,:] = FdFF[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cond, trial, frame]
FdFF_by_cond_meanRstim = np.mean(FdFF_by_cond_Rstim, axis=2) # [roi, cond, frame]
Fzsc_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            Fzsc_by_cond[r,c,t,:] = Fzsc[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
Fzsc_by_cond_Rstim = Fzsc_by_cond[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cond, trial, frame]
Fzsc_by_cond_meanRstim = np.mean(Fzsc_by_cond_Rstim, axis=2) # [roi, cond, frame]

catinds = np.full([len(cats), (conds_per_cat * n_trials)], np.nan)
for c in range(n_cats):
    catinds[c] = np.argwhere(trialdataarr[:, 1] == c).transpose()[0]
catinds = catinds.astype(int)
acqfr_by_cat = trialdataarr[catinds[:], 2]

# F__by_cat = [roi, cat, trial, frame]
FdFF_by_cat = np.full([n_ROIs, n_cats, conds_per_cat * n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_cats):
    for r in range(n_ROIs):
        for t in range(conds_per_cat * n_trials):
            FdFF_by_cat[r,c,t,:] = FdFF[r,(acqfr_by_cat[c][t]-isiframes):(acqfr_by_cat[c][t]+stimframes+isiframes)]
FdFF_by_cat_Rstim = FdFF_by_cat[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cat, trial, frame]
FdFF_by_cat_meanRstim = np.mean(FdFF_by_cat_Rstim, axis=2) # [roi, cat, frame]
#
Fzsc_by_cat = np.full([n_ROIs, n_cats, conds_per_cat * n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_cats):
    for r in range(n_ROIs):
        for t in range(conds_per_cat * n_trials):
            Fzsc_by_cat[r,c,t,:] = Fzsc[r,(acqfr_by_cat[c][t]-isiframes):(acqfr_by_cat[c][t]+stimframes+isiframes)]
Fzsc_by_cat_Rstim = Fzsc_by_cat[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cat, trial, frame]
Fzsc_by_cat_meanRstim = np.mean(Fzsc_by_cat_Rstim, axis=2) # [roi, cat, frame]


#%% Calculate tuning properties for each ROI (i.e. compute face selectivity index)
# ? ? ? and find preferred face(s)?

### FSI = (meanR_faces – meanR_nonfaceobj) / (meanR_faces + meanR_nonfaceobj)
# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For 
# |face-selectivity index| = 1/3, that is, if the response to faces was at 
# least twice (or at most half) that of nonface objects, a cell was classed 
# as being face selective45–47.
FdFF_by_cat_meanRstimall = np.mean(FdFF_by_cat_meanRstim, axis=-1)
Fzsc_by_cat_meanRstimall = np.mean(Fzsc_by_cat_meanRstim, axis=-1)
FdFF_by_cat_meanRstimallnorm = FdFF_by_cat_meanRstimall + np.abs(np.min(FdFF_by_cat_meanRstimall))
Fzsc_by_cat_meanRstimallnorm = Fzsc_by_cat_meanRstimall + np.abs(np.min(Fzsc_by_cat_meanRstimall))
key_faces = [c for c in categories if categories[c] =='m'][0]
key_objs = [c for c in categories if categories[c] =='u'][0]
key_bodies = [c for c in categories if categories[c] =='b'][0]
FdFF_allfaces_meanRstimall = FdFF_by_cat_meanRstimallnorm[:,key_faces]
Fzsc_allfaces_meanRstimall = Fzsc_by_cat_meanRstimallnorm[:,key_faces]
FdFF_allobjs_meanRstimall = FdFF_by_cat_meanRstimallnorm[:,key_objs]
Fzsc_allobjs_meanRstimall = Fzsc_by_cat_meanRstimallnorm[:,key_objs]
FdFF_allbodies_meanRstimall = FdFF_by_cat_meanRstimallnorm[:,key_bodies]
Fzsc_allbodies_meanRstimall = Fzsc_by_cat_meanRstimallnorm[:,key_bodies]
# FSIs(_by_roi) = [roi, fsi]

FSIs_dFF = (FdFF_allfaces_meanRstimall - FdFF_allobjs_meanRstimall) / (FdFF_allfaces_meanRstimall + FdFF_allobjs_meanRstimall)
FSIs_zsc = (Fzsc_allfaces_meanRstimall - Fzsc_allobjs_meanRstimall) / (Fzsc_allfaces_meanRstimall + Fzsc_allobjs_meanRstimall)


#%% Plotting face-body-object selective cells

import copy

def plot_ROIs_RGB(ROIs, RGB_ROIs, fov_size=(512,512), ref_image=None, scale_bar=False, um_per_px=None, 
             n_neighbors=None, save_path:str=''):
    
    dpi = plt.rcParams['figure.dpi'] / 2
    h, w = fov_size # rows/height/y, columns/width/x
    figsize = w / float(dpi), h / float(dpi)
    
    f0 = plt.figure(figsize=figsize)
    ax = f0.add_axes([0, 0, 1, 1])
    plt.set_cmap('hsv')
    #plt.axis('off')
    ax.axis('off')
    ax.set_frame_on(False)
    if ref_image is not None:
        ilow, ihigh = np.percentile(ref_image, (1.0, 99.98))
        ref_f64 = util.img_as_float64(ref_image)
        ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
        ref = ref_rescale
        canvas = np.stack((ref,)*3, axis=-1) # copy single channel to form RGB image
    else:
        canvas = np.zeros([h, w, 3], dtype=np.float64) # create a color canvas with frame size

    for r in range(len(ROIs)):
        ROI = ROIs[r]
        ry = ROI['ypix']
        rx = ROI['xpix']
        canvas[ry,rx,:] = RGB_ROIs[r]
        
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    #plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))#, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot_FSIzsc' + \
            '_tuned{}'.format(len(RGB_ROIs)) + \
            '.png'
        f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
        
        
#%% Plot tuned cells with continuous tuning-wheel

#parameters
RGB_multiplier = 2.5
plotting_threshold_continuous = 0.3

Fzsc_for_plot_continuous = copy.deepcopy(Fzsc_by_cat_meanRstimallnorm)

#Subtract the response to the least-tuned category (otherwise, an ROI that responds to all categories would show up as white)
Fzsc_least_tuned = np.min(Fzsc_for_plot_continuous, axis=1)
for col_i in range(3): 
    Fzsc_for_plot_continuous[:,col_i] = Fzsc_for_plot_continuous[:,col_i] - Fzsc_least_tuned

Fzsc_for_plot_continuous[Fzsc_for_plot_continuous > 1] = 1 #Cap the RGB values to 1
Fzsc_for_plot_continuous = Fzsc_for_plot_continuous * RGB_multiplier #This highlights ROIs with less tuning, at the expense of dynamic range
Fzsc_for_plot_continuous[:,[0,1,2]] = Fzsc_for_plot_continuous[:,[key_bodies,key_faces,key_objs]] #Swap face indexes to be on the first column, making face-cells be red


thresholding_logical_vector_continuous = np.max(Fzsc_for_plot_continuous, axis=1) > plotting_threshold_continuous #Threshold so we don't plot un-tuned neurons (particularly important if using RGB_multiplier > 1)
ROIs_for_plot_continuous = ROIs[thresholding_logical_vector_continuous]
Fzsc_for_plot_continuous = Fzsc_for_plot_continuous[thresholding_logical_vector_continuous]

plot_ROIs_RGB(ROIs_for_plot_continuous, Fzsc_for_plot_continuous, 
          fov_size=fov_size, ref_image=ref_image, save_path=save_path)


#%% Plot tuned cells with discreete tuning-wheel

#parameters
plotting_threshold_discrete = 0.15
subtract_responses_to_other_stim = True
subtract_least_or_secondLeast_preferred_stim_responses = 0 #If set to 0, will subtract the responses to the least-preferred stim. If set to 1, will subtract the responses to the second-least (in this case, second-most) preferred stim

Fzsc_for_plot_discrete = copy.deepcopy(Fzsc_by_cat_meanRstimallnorm) 

# Subtract to all responses the responses to the second-preferred stimulus
if subtract_responses_to_other_stim:
    for row_i in range(len(Fzsc_for_plot_discrete)):
        this_row = Fzsc_for_plot_discrete[row_i]
        response_to_non_preferred_stim = sorted(set(this_row))[subtract_least_or_secondLeast_preferred_stim_responses] # This sorts the responses and selects the lowest(0) or second-lowest(1) response
        Fzsc_for_plot_discrete[row_i] = this_row - response_to_non_preferred_stim

thresholding_logical_vector_discrete = np.max(Fzsc_for_plot_discrete, axis = 1) > plotting_threshold_discrete #Threshold 
ROIs_for_plot_discrete = ROIs[thresholding_logical_vector_discrete]
Fzsc_for_plot_discrete = Fzsc_for_plot_discrete[thresholding_logical_vector_discrete]

Fzsc_for_plot_preferredKey = np.argmax(Fzsc_for_plot_discrete,axis =1)
Fzsc_for_plot_discrete[:] = 0 # We will re-fill the preferredkeys with 1s in the folowwing for loop
for roi_i in range(len(Fzsc_for_plot_discrete)):
    Fzsc_for_plot_discrete[roi_i,Fzsc_for_plot_preferredKey[roi_i]] = 1

Fzsc_for_plot_discrete[:,[0,1,2]] = Fzsc_for_plot_discrete[:,[key_bodies,key_faces,key_objs]] #Swap face indexes to be on the first column, making face-cells be red
# Fzsc_for_plot_discrete[:,[1,2]] = Fzsc_for_plot_discrete[:,[2,1]] #Swap face indexes to be on the first column, making face-cells be red

plot_ROIs_RGB(ROIs_for_plot_discrete, Fzsc_for_plot_discrete, 
          fov_size=fov_size, ref_image=ref_image, save_path=save_path)


#%%



### TODO *** Could also calculate d’
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


#%% Define ROIs as tuned or untuned using the FSI

# based on Freiwald, Tsao and Livingstone 2009 Nat Neurosci https://doi.org/10.1038/nn.2363
# A face selectivity index was then computed as the ratio between difference
# and sum of face- and object-related responses. For 
# |face-selectivity index| = 1/3, that is, if the response to faces was at 
# least twice (or at most half) that of nonface objects, a cell was classed 
# as being face selective45–47.

print('|FSI| threshold: {}' .format(fsi_tuning_thresh))
tunidx_fsi = FSIs_zsc
tunidx_fsi_argsrt = np.argsort(tunidx_fsi)[::-1]
#n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) <= fsi_tuning_thresh)[0][0]
n_ROIs_tuned = np.argwhere(np.abs(tunidx_fsi[tunidx_fsi_argsrt]) > fsi_tuning_thresh).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

# tuning_index_cond = tunidx_fsi
# tuning_index_cat = tunidx_fsi
    

#%%
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
plt.xlim([-1,1])
plt.axvline(fsi_tuning_thresh, color='m')
plt.axvline(-fsi_tuning_thresh, color='m')
f0.show()
if save_path != '':
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_FSIdFF_thresh' + \
        '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.svg'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    save_name = dt + '_histogram_FSIdFF_thresh' + \
        '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)

f0 = plt.figure()
plt.hist(FSIs_zsc, bins=100)
plt.xlabel('Face-selectivity Index')
plt.ylabel('ROIs')
plt.xlim([-1,1])
plt.axvline(fsi_tuning_thresh, color='m')
plt.axvline(-fsi_tuning_thresh, color='m')
f0.show()
if save_path != '':
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_FSIzsc_thresh' + \
        '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.svg'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    save_name = dt + '_histogram_FSIzsc_thresh' + \
        '{:.2f}'.format(fsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)


#%% Compatibility with old variable names
# tuning_index_cat_tuned_neurons = tuning_index_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# #Frois_by_cat_tuned = Frois_by_cat[tuning_index_cat > tuning_index_thresh]
# FdFF_by_cat_tuned = FdFF_by_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]
# Fzsc_by_cat_tuned = Fzsc_by_cat[np.abs(tuning_index_cond) > fsi_tuning_thresh]  
# #Frois_by_cat_tuned = Frois_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
# FdFF_by_cat_tuned = FdFF_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]
# Fzsc_by_cat_tuned = Fzsc_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]


#%% Plot tuning map 

# plot_map(ROIs, np.ones(FSIs_zsc.shape), FSIs_zsc, tuning_thresh=fsi_tuning_thresh, 
#          fov_size=fov_size, ref_image=ref_image, circular=False)

# plot_map(ROIs, np.ones(FSIs_zsc.shape), FSIs_zsc, tuning_thresh=fsi_tuning_thresh, 
#          fov_size=fov_size, ref_image=ref_image, circular=False, save_path=save_path)

import time
for c in range(0,11):
    c = 0.1 * c
    plot_map(ROIs, c*np.ones(FSIs_zsc.shape), FSIs_zsc, tuning_thresh=fsi_tuning_thresh, 
              fov_size=fov_size, ref_image=ref_image, circular=False, save_path=save_path)
    time.sleep(1.2)

#%%

# print('Tuning index threshold: {}' .format(tuning_index_thresh))
# n_ROIs_tuned = Frois_by_cond_tuned.shape[0]
# n_neurons = Frois.shape[0]
# pct_tuned_neurons = round((100 * n_ROIs_tuned) / n_neurons, 2)
# print('Tuned neurons: {}. Total neurons: {}.'.format(n_ROIs_tuned,n_neurons))
# print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))

# for r in range(Frois_by_cat_tuned.shape[0]):
#     print(r)
#     plt.pause(0.05)
#     plt.subplots(1, 3, constrained_layout=True)
#     #if plot_rando_neurons == True:
#     #    r = np.random.randint(Frois_by_cat_tuned.shape[0])
#     for c in range(n_cats):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
#         #     continue
#         plt.subplot(1, 3, c+1)
#         plt.title('Stim: ' + str(categories[c]), fontsize=7)
#         plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         plt.ylim((-1,5))
#         plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize=6)
#         plt.ylabel(normalize, fontsize=6)
#         plt.tick_params(axis='both', which='major', labelsize=5)
#         for t in range(conds_per_cat * n_trials):
#             plt.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
#         plt.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0))
#         plt.suptitle('roi {} '.format(r), fontsize=10)
#         #plt.waitforbuttonpress()
#         print(np.std(np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0)))

#for r in range(Frois_by_cat_tuned.shape[0]):
for r in range(n_ROIs_tuned):
    ridx = tunidx_fsi_argsrt[r]
    print(r)
    ipd = 1 / dpi
    fig = plt.figure(figsize=(2*300*ipd,3*300*ipd))
    fig.clf()
    fig.suptitle('roi {} '.format(ridx), fontsize=12)
    axes = fig.subplots(nrows=2, ncols=3)
    for c in range(n_cats):
        #plt.subplot(2, 3, c+1)
        ax = axes[0,c]
        ax.set_title(categories[cats[c]], fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        #ax.set_ylim((-3,10))
        ax.set_ylim((np.min(Fzsc_by_cat[ridx,:,:,:]) - 0.2,
                     np.max(Fzsc_by_cat[ridx,:,:,:]) + 0.2))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), Fzsc_by_cat[ridx,c,t,:], color=str((0.4)+0.4*t/Fzsc_by_cat.shape[2]))
        ax.plot(range(totalframes), np.mean(Fzsc_by_cat[ridx,c,:,:], axis=0), color='tab:green')
    for c in range(n_cats):
        ax = axes[1,c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_xlim((0,4*acq_framerate))
        #ax.set_ylim((-1,2))
        ax.set_ylim((np.min(FdFF_by_cat[ridx,:,:,:]) - 0.1,
                     np.max(FdFF_by_cat[ridx,:,:,:]) + 0.1))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), FdFF_by_cat[ridx,c,t,:], color=str((0.4)+0.4*t/FdFF_by_cat.shape[2]))
        ax.plot(range(totalframes), np.mean(FdFF_by_cat[ridx,c,:,:], axis=0), color='tab:blue') 
    plt.show()
    #fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_PD_FSI_roi{}.svg'.format(r), format='svg', dpi=1200)
    plt.pause(0.05)
        

#%%
for r in range(Frois_by_cat_tuned.shape[0]):
    print(r)
    ipd = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(2*300*ipd,4*300*ipd))
    fig.clf()
    fig.suptitle('roi {} '.format(r), fontsize=12)
    axes = fig.subplots(nrows=4, ncols=3)
    for c in range(n_cats):
        ax = axes[0,c]
        ax.set_title(str(cats[c]), fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_ylim((-3,10))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        ax.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0), color='tab:green')
    for c in range(n_cats):
        ax = axes[1,c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_xlim((0,4*acq_framerate))
        ax.set_ylim((-1,2))
        for t in range(conds_per_cat * n_trials):
            ax.plot(range(totalframes), FdFF_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        ax.plot(range(totalframes), np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0), color='tab:blue') 
    for c in range(n_cats):
        ax = axes[2,c]
        ax.set_title(str(cats[c]), fontsize=10)
        if c == 0:
            ax.set_ylabel('Z-score', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_ylim((-0.1,0.5))
        #for t in range(conds_per_cat * n_trials):
        #    ax.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        Fmean = np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0)
        Fsem = np.std(FdFF_by_cat_tuned[r,c,:,:], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
        ax.plot(range(totalframes), Fmean, color='tab:green')
        ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:green', alpha=0.25)
    for c in range(n_cats):
        ax = axes[3,c]
        if c == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_xlim((0,4*acq_framerate))
        ax.set_ylim((-0.1,0.5))
        #for t in range(conds_per_cat * n_trials):
        #    ax.plot(range(totalframes), FdFF_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        Fmean = np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0)
        Fsem = np.std(FdFF_by_cat_tuned[r,c,:,:], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
        ax.plot(range(totalframes), Fmean, color='tab:blue')
        ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:blue', alpha=0.25)
    plt.show()
    fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_PD_FSI_roi{}.svg'.format(r), format='svg', dpi=1200)
    plt.pause(0.05)



#%% backup before moving to 4 rows

# for r in range(Frois_by_cat_tuned.shape[0]):
#     print(r)
#     ipd = 1 / plt.rcParams['figure.dpi']
#     fig = plt.figure(figsize=(2*300*ipd,3*300*ipd))
#     fig.clf()
#     fig.suptitle('roi {} '.format(r), fontsize=12)
#     axes = fig.subplots(nrows=2, ncols=3)
#     for c in range(n_cats):
#         #plt.subplot(2, 3, c+1)
#         ax = axes[0,c]
#         ax.set_title(str(cats[c]), fontsize=10)
#         if c == 0:
#             ax.set_ylabel('Z-score', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=8)
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.set_xticks([x * acq_framerate for x in range(5)])
#             ax.set_xticklabels(['', 0, '', 2, ''])
#         else:
#             ax.set_yticklabels([])
#             ax.set_xticklabels([])
#             ax.axis('off')
#         ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         ax.set_ylim((-0.1,0.5))
#         #for t in range(conds_per_cat * n_trials):
#         #    ax.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
#         Fmean = np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0)
#         Fsem = np.std(FdFF_by_cat_tuned[r,c,:,:], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
#         ax.plot(range(totalframes), Fmean, color='tab:green')
#         ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:green', alpha=0.25)
#     for c in range(n_cats):
#         ax = axes[1,c]
#         if c == 0:
#             ax.set_xlabel('Time (sec)', fontsize=8)
#             ax.set_ylabel('dF/F', fontsize=10)
#             ax.tick_params(axis='both', which='major', labelsize=8)
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.set_xticks([x * acq_framerate for x in range(5)])
#             ax.set_xticklabels(['', 0, '', 2, ''])
#         else:
#             ax.set_yticklabels([])
#             ax.set_xticklabels([])
#             ax.axis('off')
#         ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         ax.set_xlim((0,4*acq_framerate))
#         ax.set_ylim((-0.1,0.5))
#         #for t in range(conds_per_cat * n_trials):
#         #    ax.plot(range(totalframes), FdFF_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
#         Fmean = np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0)
#         Fsem = np.std(FdFF_by_cat_tuned[r,c,:,:], axis=0) / np.sqrt(FdFF_by_cat_tuned.shape[0])
#         ax.plot(range(totalframes), Fmean, color='tab:blue')
#         ax.fill_between(range(totalframes), Fmean - Fsem, Fmean + Fsem, facecolor='tab:blue', alpha=0.25)
#     plt.show()
#     #fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_PD_FSI_roi{}.svg'.format(r), format='svg', dpi=1200)
#     plt.pause(0.05)

# for r in range(Frois_by_cat_tuned.shape[0]):
#     print(r)
#     plt.pause(0.05)
#     plt.subplots(1, 3, constrained_layout=True)
#     for c in range(n_cats):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
#         #     continue
#         plt.subplot(1, 3, c+1)
#         plt.title('Stim: ' + str(categories[c]), fontsize=7)
#         plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         plt.ylim((-1,5))
#         plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize=6)
#         plt.ylabel(normalize, fontsize=6)
#         plt.tick_params(axis='both', which='major', labelsize=5)
#         for t in range(conds_per_cat * n_trials):
#             plt.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
#         plt.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0))
#         plt.suptitle('roi {} '.format(r), fontsize=10)
#         print(np.std(np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0)))

#%%

# #print(sum(Frois_by_cond_meanRstim[0,5]))
# # a = scipy.stats.ttest_1samp(Frois_by_cond_meanRstim[0,5], 0)
# # a[1]
# # print(sum(Ftest_cond[0,5]))
# # a = scipy.stats.ttest_1samp(Ftest_cond[0,5], 0)
# # a[1]
# print('Tuning index threshold: {}' .format(tuning_index_thresh))
# # n_ROIs_tuned = Frois_by_cond_tuned.shape[0]
# n_ROIs_tuned = FdFF_by_cond_tuned.shape[0]
# n_neurons = n_ROIs
# pct_tuned_neurons = round((100 * n_ROIs_tuned) / n_neurons, 2)
# print('Tuned neurons: {}. Total neurons: {}.'.format(n_ROIs_tuned,n_neurons))
# print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))

# # for r in range(Frois_by_cat_tuned.shape[0]):
# for r in range(FdFF_by_cat_tuned.shape[0]):
#     print(r)
#     plt.pause(0.05)
#     plt.subplots(1, 3, constrained_layout=True)
#     #if plot_rando_neurons == True:
#     #    r = np.random.randint(FdFF_by_cat_tuned.shape[0])
#     for c in range(n_cats):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
#         #     continue
#         plt.subplot(1, 3, c+1)
#         plt.title('Stim: ' + str(categories[c]), fontsize=7)
#         plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         plt.ylim((-1,5))
#         plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize=6)
#         plt.ylabel(normalize, fontsize=6)
#         plt.tick_params(axis='both', which='major', labelsize=5)
#         for t in range(conds_per_cat * n_trials):
#             plt.plot(range(totalframes), FdFF_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/FdFF_by_cat_tuned.shape[2]))
#         plt.plot(range(totalframes), np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0))
#         plt.suptitle('roi {} '.format(r), fontsize=10)
#         #plt.waitforbuttonpress()
#         print(np.std(np.mean(FdFF_by_cat_tuned[r,c,:,:], axis=0)))
        
        
# #%%

# Frois_peak_frame = np.argmax(Frois, axis=1)
# y_height = 0
# plt.figure()
# for n in range(20):
#     r = np.random.randint(n_ROIs)
#     if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
#         plt.plot( np.linspace(1,18+36,18+36)-19, (Frois[r,Frois_peak_frame[r]-18:Frois_peak_frame[r]+36]) + y_height)
#         y_height += 0.5
# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.yticks(range(int(np.ceil(y_height))))
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.axvline(x=0)
# plt.title('Peak ' + normalize + ' for 20 neurons')

# y_height = 0
# plt.figure(dpi = 900)
# for n in range(20):
#     plt.plot(np.linspace(1,Frois.shape[1],Frois.shape[1]), (Frois[np.random.randint(n_ROIs),:]) + y_height, linewidth=0.2)
#     y_height += 3

# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.title(normalize + ' for 20 neurons')
