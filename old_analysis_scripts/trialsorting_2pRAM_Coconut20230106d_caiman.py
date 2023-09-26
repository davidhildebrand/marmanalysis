#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#from scipy.signal import find_peaks as find_peaks
#from scipy.optimize import minimize as scipy_minimize
import scipy.stats as stats
from skimage import exposure, util
from warnings import warn

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')


#%% read CaImAn outputs


filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data'
# filename = '20230106d170634tUTC_Coconut_Auditory_fov1p2x1p2_res3umperpx.log'
filename = '20230106d170634tUTC_Coconut_Auditory_stimlog.csv'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Coconut/20230106d/SP_200umdeep_1p2mm_3umppix_18p82Hz_40mW'
pf = pf + os.path.sep + 'caiman/'
save_path = '' 
save_path = '/Users/davidh/Data/Freiwald/Analysis/Coconut/20230106d/SP_200umdeep_1p2mm_3umppix_18p82Hz_40mW/'

# SOC paths
# filepath = 'C:/FreiwaldSync/MarmoScope/Stimulus/Data/'
# filename = '20230106d170634tUTC_Coconut_Auditory_stimlog.csv'
# pf = r'C:\FreiwaldSync\MarmoScope\Analysis\Data\Coconut\20230106d\SP_200umdeep_1p2mm_3umppix_18p82Hz_40mW/'


acq_framerate = 18.82 # Hz

# based on Pattadkal etal Priebe 2022 bioRxiv
#   https://doi.org/10.1101/2022.06.23.497220
# For the cell to be included in the analysis, the response at the maximum 
# responsive direction had to be significantly different from baseline as 
# measured using t-test (p: 0.05). In addition, the response at this 
# maximally responsive direction also had to exceed a threshold, which 
# varied between sessions from 0.1 to 0.25 dF/F.
dFF_thresh = 0.5

shift_acqfr_bcDeepInt = 0 #5
n_neighbors = 10
anova_alpha = 0.05
ignore_wn_stim = True
ignore_vocalization_stim = True

dynamic_range_hsv = 0.75

plt.rcParams['figure.dpi'] = 600
dpi = plt.rcParams['figure.dpi']

# *** TODO automatically identify stimulus duration
stimframes = int(np.ceil(2 * acq_framerate))
isiframes = round(1 * acq_framerate)
totalframes = isiframes + stimframes + isiframes

# tuning = 'average' #average, percentile, t-test, or max
# normalize = 'Z-score' #Z-score #dF/F
# tuning_index_thresh = 0.5
# plot_least_tuned_neurons_first = True
# percentile = 90

Frois = np.load(pf + os.path.sep + 'temporal_good_batch32.npy')
ROIs = np.load(pf + os.path.sep + 'contours_good_batch32.npy', allow_pickle=True)
ref_image = np.load(pf + os.path.sep + '13f922ba-04b5-4e49-9400-caa40c4de474_mean_projection.npy', allow_pickle=True)
fov_size = ref_image.shape

n_ROIs = ROIs.shape[0]
FdFF = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.mean(Frois, axis=1)[:,np.newaxis]
Fzsc = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.std(Frois, axis=1)[:,np.newaxis]

if save_path == '':
    saving = False
else:
    saving = True    


#%% Define functions

def plot_map_caiman(ROIs, tuning, tuning_mag, tuning_thresh=0.5, fov_size=(512,512), 
             circular=False, ref_image=None, scale_bar=False, um_per_px=None, 
             n_neighbors=None, save_path:str=''):
    # The values tuning and tuning_mag must be within [0,1].
    # 'circular' determines whether tuning has the same color for 0 and 1 
    # (True for MT, False for auditory)
    # TODO **** implement scale bar?
    
    #circular=True
    dpi = plt.rcParams['figure.dpi'] / 2
    h, w = fov_size # rows/height/y, columns/width/x
    figsize = w / float(dpi), h / float(dpi)
    
    ###### TODO *** THIS DOES NOT GENERALIZE
    n_ROIs = len(ROIs)
    tuned = np.abs(tuning_mag) > tuning_thresh
    ROIs_tuned = ROIs[tuned]
    tuning_tuned = tuning[tuned]
    tuning_mag_tuned = tuning_mag[tuned]
    
    if tuning.max() > 1:
        warn(UserWarning('provided tuning index has values > 1 (out of range)'))
        tuning = tuning / tuning.max()
        
    if tuning_tuned.max() > 1:
        warn(UserWarning('provided tuning_tuned index has values > 1 (out of range)'))
        tuning_tuned = tuning_tuned / tuning_tuned.max()
    
    assert len(ROIs_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
    n_ROIs_tuned = len(ROIs_tuned)
    
    f0 = plt.figure(figsize=figsize)
    ax = f0.add_axes([0, 0, 1, 1])
    plt.set_cmap('hsv')
    #plt.axis('off')
    ax.axis('off')
    ax.set_frame_on(False)
    if ref_image is not None:
        ilow, ihigh = np.percentile(ref_image, (1.0, 99))
        ref_f64 = util.img_as_float64(ref_image)
        ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
        ref = ref_rescale
        canvas = np.stack((ref,)*3, axis=-1) # copy single channel to form RGB image
        canvas = (255* canvas).astype(np.uint8)
    else:
        canvas = np.zeros([h, w, 3], dtype=np.uint8) # create a color canvas with frame size
        
    import cv2
    for r in range(n_ROIs_tuned):
        ROI = ROIs_tuned[r].astype(int)
        ry = ROI[:,1]
        rx = ROI[:,0]
   
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        color_code = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1) 
        color_code = tuple([255*x for x in color_code])
        canvas = cv2.fillPoly(np.array(canvas[:,:,:]),[ROI], color =  color_code) 
        
        # for rgb in range(3):
        #     if circular:
                
        #         #canvas[:,:, rgb] = cv2.fillPoly(np.array(canvas[:,:, rgb]),[ROI], color = (255 * abs(1 - 2 * abs(tuning_tuned[r] - rgb * 1/3)))) 
        #     else:
        #         canvas[:,:, rgb] = abs(1 - 2 * abs(index_stim[i] / 1.5 - rgb * 1/3)) #* index_strength[i] #effectively removing red and purple for values close to 1  
    
    
        # if circular is True:
        #     canvas = cv2.fillPoly(canvas,[ROI], color = color_code)
        #     #canvas[ry,rx,:] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
        # else:
        #     canvas = cv2.fillPoly(canvas,[ROI], color = (255,0,0))
        #     #for rgb in range(3):
        #         #canvas[ry,rx,rgb] = abs(1 - 2 * abs(tuning_tuned[r] / 1.5 - rgb * 1/3)) #* tuning_mag[r]
                
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    #plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))#, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot' + \
            '_nplotted{}'.format(n_ROIs_tuned) + \
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
        deg_symbol = 'deg'
        ax0.set_xticklabels(['0' + deg_symbol, '90' + deg_symbol])
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
    n_ROIs = len(ROIs)
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
            '_nplotted'.format(n_ROIs_tuned) + \
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
                '_nplotted{}'.format(n_ROIs_tuned) + \
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

df = pd.read_csv(filepath + os.path.sep + filename)
df = df[~pd.isna(df["f"])]

trialdata = {}

# 41.9371         EXP     trial 0, stim start, grating, full field, drifting, cond=5, ori=225.0, tex=sin, size=[75.67137421 75.67137421], sf=[1.2 0. ], tf=4, mask=None, contrast=1.0, acqfr=222
for i_df in range(len(df)):
    tmp_cond = df.iloc[i_df]["cond"]
    tmp_f = df.iloc[i_df]["f"]
    tmp_acqft = int(df.iloc[i_df]["acqfr_stim_i"])
    
    if ('Vocal' in tmp_cond) and ignore_vocalization_stim:
        continue
    if ('wn' in tmp_f) and ignore_wn_stim:
        continue
    trialdata[i_df] = {'cond' : tmp_cond,
                       'f' : tmp_f,
                       'acqfr' : tmp_acqft}

# trialdataarr = [trial_idx, cond, ori, acqfr]
trialdataarr = np.array([0,0,0],dtype=object)

#trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr = np.vstack((trialdataarr,([trialdata[td]['cond'], trialdata[td]['f'], trialdata[td]['acqfr']])))

trialdataarr = trialdataarr[1:]

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,0])
conds_noblnk = conds[np.argwhere(conds != 'blank')].T[0].astype(int)
conds_noblnk_sort = np.argsort(conds_noblnk)
conds_noblnk = conds_noblnk[conds_noblnk_sort]
conds[0:conds_noblnk.shape[0]] = conds[conds_noblnk_sort]
n_conds = len(conds)
n_conds_noblnk = conds_noblnk.shape[0]
n_trials = int(len(trialdata) / n_conds)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdataarr[:,0] == conds[c]).T[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:],2].astype(int) - shift_acqfr_bcDeepInt

condfreqs = np.full([len(conds)], np.nan, dtype=float)
for c in range(n_conds):
    temp_freqs = trialdataarr[condinds[c],1]
    if np.all(temp_freqs == temp_freqs[0]):
        condfreqs[c] = float(temp_freqs[0])
    else:
        warn('something went wrong assigning frequencies to conds')
condfreqs_noblnk = condfreqs[~np.isnan(condfreqs)]
freqs = np.unique(condfreqs)
freqs_noblnk = freqs[~np.isnan(freqs)]
n_freqs = freqs.shape[0]
n_freqs_noblnk = freqs_noblnk.shape[0]

conds_per_freq_noblnk = int(n_conds_noblnk / n_freqs_noblnk)
freqconds_noblnk = np.full([n_freqs_noblnk, conds_per_freq_noblnk], np.nan)
for f in range(n_freqs_noblnk):
    freqconds_noblnk[f] = conds_noblnk[np.argwhere(condfreqs == freqs_noblnk[f]).squeeze()]
freqconds_noblnk = freqconds_noblnk.astype(int)
    
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:],2].astype(int) - shift_acqfr_bcDeepInt


#%% Organize and average fluorescence traces
                
# F__by_cond = [roi, cond, trial, frame]
FdFF_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            FdFF_by_cond[r,c,t,:] = FdFF[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
FdFF_by_cond_Rpre = FdFF_by_cond[:,:,:,0:isiframes] # [roi, cond, trial, frame]
FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cond, trial, frame]
FdFF_by_cond_meanRpre = np.mean(FdFF_by_cond_Rpre, axis=2) # [roi, cond, frame]
FdFF_by_cond_meanRstim = np.mean(FdFF_by_cond_Rstim, axis=2) # [roi, cond, frame]
FdFF_by_cond_meanmeanRpre = np.mean(FdFF_by_cond_Rpre, axis=(2,3)) # [roi, cond]
FdFF_by_cond_meanmeanRstim = np.mean(FdFF_by_cond_Rstim, axis=(2,3)) # [roi, cond]
#FdFF_by_cond_meanmeanRratio = FdFF_by_cond_meanmeanRstim / FdFF_by_cond_meanmeanRpre

Fzsc_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            Fzsc_by_cond[r,c,t,:] = Fzsc[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
Fzsc_by_cond_Rpre = Fzsc_by_cond[:,:,:,0:isiframes] # [roi, cond, trial, frame]
Fzsc_by_cond_Rstim = Fzsc_by_cond[:,:,:,isiframes:(isiframes+stimframes)] # [roi, cond, trial, frame]
Fzsc_by_cond_meanRpre = np.mean(Fzsc_by_cond_Rpre, axis=2) # [roi, cond, frame]
Fzsc_by_cond_meanRstim = np.mean(Fzsc_by_cond_Rstim, axis=2) # [roi, cond, frame]
Fzsc_by_cond_meanmeanRpre = np.mean(Fzsc_by_cond_Rpre, axis=(2,3)) # [roi, cond]
Fzsc_by_cond_meanmeanRstim = np.mean(Fzsc_by_cond_Rstim, axis=(2,3)) # [roi, cond]
#Fzsc_by_cond_meanmeanRratio = Fzsc_by_cond_meanmeanRstim / Fzsc_by_cond_meanmeanRpre

# # Frois_by_cond = [ROIno, cond, t, F]
# Frois_by_cond = np.full([Frois.shape[0], n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
# Frois_by_cond_top_decile = np.full([Frois.shape[0], n_conds], np.nan)
# for c in range(n_conds):
#     for r in range(Frois.shape[0]):
#         for t in range(n_trials):
#             Frois_by_cond[r,c,t,:] = Frois[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
#             if c == 7 and ignore_wn_stim:
#                 Frois_by_cond[r,c,t,:] = 0


#%% Calculate tuning properties for each ROI

# Define 'responsive cells' and 'tone-selective cells'
# based on Zeng et al Poo 2019 PNAS https://doi.org/10.1073/pnas.1816653116
# Cells showing significant differences in the fluorescence intensity signals 
# observed during baseline vs. stimulus-presentation periods (P<0.05, ANOVA) 
# were defined as “responsive cells”. 
# Of these, tone-selective cells were defined by significant differences in 
# intensity responses signals across all frequencies (P<0.05, ANOVA). 
# The “best frequency (BF)” is defined as the frequency of sound stimulus 
# that evoked the highest response. In our data set, nearly all responsive 
# cells were selective to pure tones. In cellular tonotopic maps, only 
# tone-selective cells were color coded and analyzed.

### ANOVA_resp/tone_dFF/zsc = [roi, [F, p]]
### ANOVA groups: blank, each stimulus period as a separate group
# NOTE shouldn't matter whether it is using dFF or zsc (... confirmed!)
ANOVA_respons = np.full([n_ROIs, 2], np.nan)
for r in range(n_ROIs):
    al = [Fzsc_by_cond_meanRstim[r,c] for c in range(n_conds)]
    ANOVA_respons[r] = stats.f_oneway(*al)
respons_idx = np.argwhere(ANOVA_respons[:,1] <= anova_alpha).squeeze()
n_respons = respons_idx.shape[0]
pct_respons = round(((100 * n_respons) / n_ROIs), 2)

# # just to compare dFF to zsc to be sure
# ANOVA_resp_dFF = np.full([n_ROIs, 2], np.nan)
# for r in range(n_ROIs):
#     al = [FdFF_by_cond_meanRstim[r,c] for c in range(n_conds)]
#     ANOVA_resp_dFF[r] = stats.f_oneway(*al)
# resp_dFF_idx = np.argwhere(ANOVA_resp_dFF[:,1] <= anova_alpha).squeeze()
# n_resp_dFF = resp_dFF_idx.shape[0]
# pct_resp_dFF = round(((100 * n_resp_dFF) / n_ROIs), 2)
# ANOVA_resp_zsc = np.full([n_ROIs, 2], np.nan)
# for r in range(n_ROIs):
#     al = [Fzsc_by_cond_meanRstim[r,c] for c in range(n_conds)]
#     ANOVA_resp_zsc[r] = stats.f_oneway(*al)
# resp_zsc_idx = np.argwhere(ANOVA_resp_zsc[:,1] <= anova_alpha).squeeze()
# n_resp_zsc = resp_zsc_idx.shape[0]
# pct_resp_zsc = round(((100 * n_resp_zsc) / n_ROIs), 2)

# # similar, but use groups consisting of all stim frames rather than means
# ANOVA_resp_dFF_c = np.full([n_ROIs, 2], np.nan)
# for r in range(n_ROIs):
#     al = [np.concatenate(FdFF_by_cond_Rstim[r,c,:]) for c in range(n_conds)]
#     ANOVA_resp_dFF_c[r] = stats.f_oneway(*al)
# resp_dFF_c_idx = np.argwhere(ANOVA_resp_dFF_c[:,1] <= anova_alpha)
# n_resp_dFF_c = resp_dFF_c_idx.shape[0]
#
# # include a group consisting of all the baseline frames concatenated
# ANOVA_resp_dFF_cc = np.full([n_ROIs, 2], np.nan)
# for r in range(n_ROIs):
#     al = [np.concatenate(FdFF_by_cond_Rstim[r,c,:]) for c in range(n_conds)]
#    al.append(np.ravel(FdFF_by_cond_Rpre[r,:,:]))
#     ANOVA_resp_dFF_cc[r] = stats.f_oneway(*al)
# resp_dFF_cc_idx = np.argwhere(ANOVA_resp_dFF_cc[:,1] <= anova_alpha)
# n_resp_dFF_cc = resp_dFF_cc_idx.shape[0]

ANOVA_tonesel = np.full([n_respons, 2], np.nan)
for r in range(n_respons):
    ri = respons_idx[r]
    al = [Fzsc_by_cond_meanRstim[ri,c] for c in range(n_conds) if conds[c] != 'blank']
    ANOVA_tonesel[r] = stats.f_oneway(*al)
tonesel_tmp = np.argwhere(ANOVA_tonesel[:,1] <= anova_alpha).squeeze()
tonesel_sort = np.argsort(ANOVA_tonesel[tonesel_tmp,1])
tonesel_idx = tonesel_tmp[tonesel_sort]
tonesel_ridx = respons_idx[tonesel_idx] # convert idx back to original ROI idx
n_tonesel = tonesel_idx.shape[0]
pct_toneselVall = round(((100 * n_tonesel) / n_ROIs), 2)
pct_toneselVrespons = round(((100 * n_tonesel) / n_respons), 2)


# based on Zeng et al Poo 2019 PNAS https://doi.org/10.1073/pnas.1816653116
# The “best frequency (BF)” is defined as the frequency of sound stimulus 
# that evoked the highest response. In our data set, nearly all responsive 
# cells were selective to pure tones. In cellular tonotopic maps, only 
# tone-selective cells were color coded and analyzed.
BC = np.argmax(np.abs(Fzsc_by_cond_meanmeanRstim[tonesel_ridx]), axis=-1) # 'best cond'
BF = condfreqs[BC] # 'best freq'
blnksel_idx = np.argwhere(conds[BC] == 'blank').squeeze()
blnksel_ridx = respons_idx[blnksel_idx] # convert idx back to original ROI idx
n_blnksel = blnksel_idx.shape[0]
pct_blnkselVall = round(((100 * n_blnksel) / n_ROIs), 2)
pct_blnkselVrespons = round(((100 * n_blnksel) / n_respons), 2)
pct_blnkselVtonesel = round(((100 * n_blnksel) / n_tonesel), 2)

tonesel_noblnk_idx = np.argwhere(conds[BC] != 'blank').squeeze()

# tonesel_noblnk_sort = np.argsort(ANOVA_tonesel[tonesel_noblnk_tmp,1])
# tonesel_noblnk_idx = tonesel_noblnk_tmp[tonesel_noblnk_sort]
tonesel_noblnk_ridx = respons_idx[tonesel_idx[tonesel_noblnk_idx]]
n_tonesel_noblnk = tonesel_noblnk_idx.shape[0]

BC_noblnk = BC[tonesel_noblnk_idx]
BF_noblnk = BF[tonesel_noblnk_idx]

# based on Pattadkal etal Priebe 2022 bioRxiv
#   https://doi.org/10.1101/2022.06.23.497220
# For the cell to be included in the analysis, the response at the maximum 
# responsive direction had to be significantly different from baseline as 
# measured using t-test (p: 0.05). In addition, the response at this 
# maximally responsive direction also had to exceed a threshold, which 
# varied between sessions from 0.1 to 0.25 dF/F.
RmaxBF = FdFF_by_cond_meanmeanRstim[tonesel_noblnk_ridx,BC_noblnk]
dFFpass_idx = np.argwhere(np.abs(RmaxBF) > dFF_thresh).squeeze()
dFFpass_ridx = tonesel_noblnk_ridx[dFFpass_idx]
n_dFFpass = dFFpass_idx.shape[0]

BC_noblnk_dFFpass = BC_noblnk[dFFpass_idx]
BF_noblnk_dFFpass = BF_noblnk[dFFpass_idx]


# # Define tuning indices using a t-tests
# # NOTE shouldn't matter whether it is using dFF or zsc
# # TODO *** check if second argument should be a calculated baseline
# tee_dFF = stats.ttest_1samp(FdFF_by_cond_meanRstim, 0, axis=2)
# Pvals_dFF = tee_dFF[1]
# Pvals_dFF_min_cond = np.min(Pvals_dFF, axis=1)
# tunidx_tee_dFF = 1 - Pvals_dFF_min_cond
# tunidx_tee_dFF_argsrt = np.argsort(tunidx_tee_dFF)[::-1]
# tee_zsc = stats.ttest_1samp(Fzsc_by_cond_meanRstim, 0, axis=2)
# Pvals_Fzsc = tee_zsc[1]
# Pvals_Fzsc_min_cond = np.min(Pvals_Fzsc, axis=1)
# tunidx_tee_zsc = 1 - Pvals_Fzsc_min_cond
# tunidx_tee_zsc_argsrt = np.argsort(tunidx_tee_zsc)[::-1]
#
# # Define tuning indices using quantiles
# qt1_dFF_all_conds = np.percentile(FdFF_by_cond_meanRstim, percentile, axis=2)
# qt1_dFF_max_cond = np.max(qt1_dFF_all_conds, axis=1)
# tunidx_qt1_dFF = qt1_dFF_max_cond
# tunidx_qt1_dFF_argsrt = np.argsort(tunidx_qt1_dFF)[::-1]
# qt1_zsc_all_conds = np.percentile(Fzsc_by_cond_meanRstim, percentile, axis=2)
# qt1_zsc_max_cond = np.max(qt1_zsc_all_conds, axis=1)
# tunidx_qt1_zsc = qt1_zsc_max_cond
# tunidx_qt1_zsc_argsrt = np.argsort(tunidx_qt1_zsc)[::-1]
#
# # Define tuning indices using average intensity
# avg_dFF = np.abs(np.mean(FdFF_by_cond_meanRstim, axis=-1))
# avg_dFF_max_cond = np.max(abs(avg_dFF), axis=1)
# tunidx_avg_dFF = avg_dFF_max_cond
# tunidx_avg_dFF_argsrt = np.argsort(tunidx_avg_dFF)[::-1]
# avg_zsc = np.abs(np.mean(Fzsc_by_cond_meanRstim, axis=-1))
# avg_zsc_max_cond = np.max(abs(avg_zsc), axis=1)
# tunidx_avg_zsc = avg_zsc_max_cond    
# tunidx_avg_zsc_argsrt = np.argsort(tunidx_avg_zsc)[::-1]

# if tuning == 't-test':
#     t_test = scipy.stats.ttest_1samp(Frois_by_cond_mean_response, 0, axis = 2)
#     p_vals = t_test[1]
#     p_vals_min_cond = np.min(p_vals, axis = 1)
#     tuning_index = 1 - p_vals_min_cond
# elif tuning == 'percentile':
#     first_quantile_all_conds = np.percentile(Frois_by_cond_mean_response, percentile, axis = 2)
#     preferred_stim = np.argmax(first_quantile_all_conds, axis = 1)
#     first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
#     tuning_index = first_quantile_max_cond
# elif tuning == 'max':
#     first_quantile_all_conds = np.max(Frois_by_cond_mean_response, axis = 2)
#     preferred_stim = np.argmax(first_quantile_all_conds, axis = 1)
#     first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
#     tuning_index = first_quantile_max_cond
# elif tuning == 'average':
#     average = np.abs(np.mean(Frois_by_cond_mean_response, axis = -1))
#     average_max_cond = np.max(abs(average), axis = 1)
#     tuning_index = average_max_cond
#     preferred_stim = np.argmax(average, axis = 1)


#%% Define ROIs as tuned or untuned using the DSI

# Frois_by_cond_tuned = Frois_by_cond[tuning_index > tuning_index_thresh]
# tuning_index_tuned_neurons = tuning_index[tuning_index > tuning_index_thresh]

# if plot_least_tuned_neurons_first:
#     Frois_by_cond_tuned = Frois_by_cond_tuned[( + tuning_index_tuned_neurons).argsort()]
# else:    
#     Frois_by_cond_tuned = Frois_by_cond_tuned[( - tuning_index_tuned_neurons).argsort()]
    

# # plot_map.plot_tuning_map(stat,iscell,ops, preferred_stim/7, tuning_index, circular = False, iscell_thresh = 0.01, strength_thresh = tuning_index_thresh, n_neighbors = n_neighbors)


# print('Tuning index threshold: {}' .format(tuning_index_thresh))
# n_tuned_neurons = Frois_by_cond_tuned.shape[0]
# n_neurons = Frois.shape[0]
# pct_tuned_neurons = round(100*n_tuned_neurons/n_neurons,2)
# print('Tuned neurons: {}. Total neurons: {}.'.format(n_tuned_neurons,n_neurons))
# print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))


#%% Plot histogram for the number of ROIs with corresponding tuning values

f0 = plt.figure()
logbins = np.logspace(start=np.log10(np.min(BF_noblnk_dFFpass)), stop=np.log10(np.max(BF_noblnk_dFFpass)), 
                      num=n_freqs_noblnk)
plt.hist(BF_noblnk_dFFpass, bins=logbins)#100)#n_freqs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('ROIs')
# print(plt.gca().get_xlim())
#plt.xlim([0,1])
#plt.axvline(dsi_tuning_thresh, color='m')
#plt.axvline(-dsi_tuning_thresh, color='m')
f0.show()
if saving:
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_caiman_freqs' + \
        '.svg'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    save_name = dt + '_histogram_caiman_freqs' + \
        '.png'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
f1 = plt.figure()
plt.hist(BF_noblnk_dFFpass, bins=n_freqs_noblnk)
plt.xlabel('Frequency (Hz)')
plt.ylabel('ROIs')
f1.show()

# f0 = plt.figure()
# logbins = np.logspace(start=np.log10(np.min(BF_noblnk)), stop=np.log10(np.max(BF_noblnk)), 
#                       num=n_freqs_noblnk)
# plt.hist(BF_noblnk, bins=logbins)#100)#n_freqs)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('ROIs')
# # print(plt.gca().get_xlim())
# #plt.xlim([0,1])
# #plt.axvline(dsi_tuning_thresh, color='m')
# #plt.axvline(-dsi_tuning_thresh, color='m')
# f0.show()
# if saving:
#     now = datetime.now()
#     dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
#     save_name = dt + '_histogram_freqs' + \
#         '.svg'
#     f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
#     save_name = dt + '_histogram_freqs' + \
#         '.png'
#     f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
# f1 = plt.figure()
# plt.hist(BF_noblnk, bins=n_freqs_noblnk)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('ROIs')
# f1.show()


#%% Plot tuning map 

# plot_tuning_map(s2p_stat, s2p_iscell, s2p_ops, Tprefs, DSI, strength_thresh=0.15, circular=True)

BF_norm = BF_noblnk_dFFpass / (np.max(BF_noblnk_dFFpass) + 1)

# fov_size = (399, 407)
plot_map_caiman(ROIs[dFFpass_ridx], BF_norm, np.ones(BF_norm.shape), tuning_thresh=0, 
         fov_size=fov_size, circular=False, ref_image=ref_image, save_path=save_path)

BF_noblnk_dFFpass_log2 = np.log2(BF_noblnk_dFFpass)
BF_noblnk_dFFpass_log2 -= np.log2(np.min(freqs)) # subtract the log-value of the lowest frequency
BF_noblnk_dFFpass_log2_norm = BF_noblnk_dFFpass_log2 / np.max(BF_noblnk_dFFpass_log2) * dynamic_range_hsv # normalize, and apply dynamic range (for non-circular stimulus conditions)

plot_map_caiman(ROIs[dFFpass_ridx], BF_noblnk_dFFpass_log2_norm, np.ones(BF_noblnk_dFFpass_log2_norm.shape), tuning_thresh=0, 
         fov_size=fov_size, circular=False, ref_image=ref_image, save_path=save_path)

# Plot a colorbar for the previous tuning map
V, H = np.mgrid[1:1:10j, 0:1:360j]
S = np.ones_like(V)
HSV = np.dstack((H,S,V))
RGB = mpl.colors.hsv_to_rgb(HSV)
fig, ax = plt.subplots()
plt.imshow(RGB[:,:round(360*dynamic_range_hsv)+1,:], origin="lower")
ticks = np.unique(BF_noblnk_dFFpass_log2_norm) * 360 
ax.set_xticks(ticks)
labels = []
for label_i in range(len(freqs)):
    labels.append(str(int(freqs[label_i])))
# ax.set_xticklabels(labels)
ax.set_xticklabels([])
plt.yticks([])
plt.show()
if save_path != '':
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_ROIplot_zscore' + \
        '_legend.png'
    fig.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)


#%% Plot responses across conditions for tuned ROIs

for r in range(0, n_dFFpass, 1):
    ridx = dFFpass_ridx[r]
    print('ROI {} with '.format(ridx) +
          'BF={} '.format(BF_noblnk_dFFpass[r].astype(int)) +
          'p={:.2}'.format(ANOVA_tonesel[tonesel_idx[tonesel_noblnk_idx[dFFpass_idx[r]]],1]))
    ipd = 1 / dpi
    fig = plt.figure(figsize=((8+2)*2*150*ipd,(4+1)*300*ipd))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.clf()
    fig.suptitle('roi {} ({})'.format(r, ridx), fontsize=10)
    # axes = fig.subplots(nrows=2, ncols=n_conds)
    # for c in range(n_conds):
    #     ax = axes[0,c]
    #     ax.set_title(str(condfreqs[c]), fontsize=4)
    #     if c == 0:
    #         ax.set_ylabel('Z-score', fontsize=8)
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.tick_params(axis='both', which='major', labelsize=8)
    #         ax.set_xticks([x * acq_framerate for x in range(5)])
    #         ax.set_xticklabels([])
    #     else:
    #         ax.set_yticklabels([])
    #         ax.set_xticklabels([])
    #         ax.axis('off')
    #     ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
    #     # ax.set_ylim((-2,8))
    #     ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
    #                   np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
    #     for t in range(n_trials):
    #         ax.plot(range(totalframes), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
    #     ax.plot(range(totalframes), np.mean(Fzsc_by_cond[ridx,c,:,:], axis=0), color='tab:green')
    axes = fig.subplots(nrows=2, ncols=n_freqs_noblnk)
    for f in range(n_freqs_noblnk):
        ax = axes[0,f]
        ax.set_title(str(int(freqs_noblnk[f])), fontsize=10)
        if f == 0:
            ax.set_ylabel('Z-score', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels([])
            #ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            #ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        # ax.set_ylim((-2,8))
        ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
                      np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
        cs = freqconds_noblnk[f]
        for c in cs:
            for t in range(n_trials):
                ax.plot(range(totalframes), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(Fzsc_by_cond[ridx,cs,:,:], axis=(0,1)), color='tab:green')
    # for c in range(n_conds):
    #     ax = axes[1,c]
    #     if c == 0:
    #         ax.set_xlabel('Time (sec)', fontsize=8)
    #         ax.set_ylabel('dF/F', fontsize=8)
    #         ax.tick_params(axis='both', which='major', labelsize=8)
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.set_xticks([x * acq_framerate for x in range(5)])
    #         ax.set_xticklabels(['', 0, '', 2, ''])
    #     else:
    #         ax.set_yticklabels([])
    #         ax.set_xticklabels([])
    #         ax.axis('off')
    #     ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
    #     ax.set_xlim((0,4*acq_framerate))
    #     # ax.set_ylim((-1,3))
    #     ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.2,
    #                   np.max(FdFF_by_cond[ridx,:,:,:]) + 0.2))
    #     for t in range(n_trials):
    #         ax.plot(range(totalframes), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
    #     ax.plot(range(totalframes), np.mean(FdFF_by_cond[ridx,c,:,:], axis=0), color='tab:blue')   
    for f in range(n_freqs_noblnk):
        ax = axes[1,f]
        if f == 0:
            ax.set_xlabel('Time (sec)', fontsize=8)
            ax.set_ylabel('dF/F', fontsize=8)
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
        ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.2,
                      np.max(FdFF_by_cond[ridx,:,:,:]) + 0.2))
        cs = freqconds_noblnk[f]
        for c in cs:
            for t in range(n_trials):
                ax.plot(range(totalframes), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(FdFF_by_cond[ridx,cs,:,:], axis=(0,1)), color='tab:blue')   
        ### Plot p_value of the mean
        #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
        # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
        # p_val = t_test[1]
        # plt.title('p value = ' + str(p_val.round(2)))
        
        #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
        #fig.waitforbuttonpress()
        #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
    plt.show()
    if saving:
        fig.savefig(save_path + os.path.sep + 'Coconut_20230106d_caiman_ridx{}_'.format(r) + \
                    'roi{}_bf{}'.format(ridx, BF_noblnk_dFFpass[r].astype(int)) + \
                    '.svg',
                    format='svg', dpi=1200)
    plt.pause(0.05)

# for r in range(0, n_tonesel_noblnk, 1):
#     ridx = tonesel_noblnk_ridx[r]
#     print('ROI {} with '.format(ridx) +
#           'BF={} '.format(BF_noblnk[r]) +
#           'p={:.2}'.format(ANOVA_tonesel[tonesel_idx[r],1]))
#     ipd = 1 / dpi
#     fig = plt.figure(figsize=((8+2)*2*150*ipd,(4+1)*300*ipd))
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     fig.clf()
#     fig.suptitle('roi {} ({})'.format(r, ridx), fontsize=10)
#     # axes = fig.subplots(nrows=2, ncols=n_conds)
#     # for c in range(n_conds):
#     #     ax = axes[0,c]
#     #     ax.set_title(str(condfreqs[c]), fontsize=4)
#     #     if c == 0:
#     #         ax.set_ylabel('Z-score', fontsize=8)
#     #         ax.spines['top'].set_visible(False)
#     #         ax.spines['right'].set_visible(False)
#     #         ax.tick_params(axis='both', which='major', labelsize=8)
#     #         ax.set_xticks([x * acq_framerate for x in range(5)])
#     #         ax.set_xticklabels([])
#     #     else:
#     #         ax.set_yticklabels([])
#     #         ax.set_xticklabels([])
#     #         ax.axis('off')
#     #     ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#     #     #ax.set_ylim((-2,8))
#     #     ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
#     #                  np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
#     #     for t in range(n_trials):
#     #         ax.plot(range(totalframes), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
#     #     ax.plot(range(totalframes), np.mean(Fzsc_by_cond[ridx,c,:,:], axis=0), color='tab:green')
#     axes = fig.subplots(nrows=2, ncols=n_freqs_noblnk)
#     for f in range(n_freqs_noblnk):
#         ax = axes[0,f]
#         ax.set_title(str(int(freqs_noblnk[f])), fontsize=10)
#         if f == 0:
#             #plt.xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
#             #ax.set_xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
#             ax.set_ylabel('Z-score', fontsize=8)
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.tick_params(axis='both', which='major', labelsize=8)
#             ax.set_xticks([x * acq_framerate for x in range(5)])
#             ax.set_xticklabels([])
#             #ax.set_xticklabels(['', 0, '', 2, ''])
#         else:
#             #ax.get_xaxis().set_visible(False)
#             ax.set_yticklabels([])
#             ax.set_xticklabels([])
#             ax.axis('off')
#         ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         #ax.set_ylim((-2,8))
#         ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
#                       np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
#         cs = freqconds_noblnk[f]
#         for c in cs:
#             for t in range(n_trials):
#                 ax.plot(range(totalframes), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
#         ax.plot(range(totalframes), np.mean(Fzsc_by_cond[ridx,cs,:,:], axis=(0,1)), color='tab:green')
#     # for c in range(n_conds):
#     #     ax = axes[1,c]
#     #     if c == 0:
#     #         ax.set_xlabel('Time (sec)', fontsize=8)
#     #         ax.set_ylabel('dF/F', fontsize=8)
#     #         ax.tick_params(axis='both', which='major', labelsize=8)
#     #         ax.spines['top'].set_visible(False)
#     #         ax.spines['right'].set_visible(False)
#     #         ax.set_xticks([x * acq_framerate for x in range(5)])
#     #         ax.set_xticklabels(['', 0, '', 2, ''])
#     #     else:
#     #         ax.set_yticklabels([])
#     #         ax.set_xticklabels([])
#     #         ax.axis('off')
#     #     ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#     #     ax.set_xlim((0,4*acq_framerate))
#     #     #ax.set_ylim((-1,3))
#     #     ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.2,
#     #                  np.max(FdFF_by_cond[ridx,:,:,:]) + 0.2))
#     #     for t in range(n_trials):
#     #         ax.plot(range(totalframes), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
#     #     ax.plot(range(totalframes), np.mean(FdFF_by_cond[ridx,c,:,:], axis=0), color='tab:blue')   
#     for f in range(n_freqs_noblnk):
#         ax = axes[1,f]
#         if f == 0:
#             ax.set_xlabel('Time (sec)', fontsize=8)
#             ax.set_ylabel('dF/F', fontsize=8)
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
#         #ax.set_ylim((-1,3))
#         ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.2,
#                       np.max(FdFF_by_cond[ridx,:,:,:]) + 0.2))
#         cs = freqconds_noblnk[f]
#         for c in cs:
#             for t in range(n_trials):
#                 ax.plot(range(totalframes), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
#         ax.plot(range(totalframes), np.mean(FdFF_by_cond[ridx,cs,:,:], axis=(0,1)), color='tab:blue')   
#         ### Plot p_value of the mean
#         #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
#         # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
#         # p_val = t_test[1]
#         # plt.title('p value = ' + str(p_val.round(2)))
        
#         #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
#         #fig.waitforbuttonpress()
#         #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
#     plt.show()
#     #if saving:
#     #    fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_roi{}.svg'.format(r), format='svg', dpi=1200)
#     plt.pause(0.05)


#%%


# #Frois_by_cond_tuned = Frois_by_cond_tuned[(Frois_by_cond_tuned_least_preferred_cond).argsort()]

# for r in range(Frois.shape[0]):
#     plt.pause(0.05)
#     #plt.figure(dpi=300)
#     plt.subplots(2, 4, constrained_layout=True, dpi = 600)
#     plt.tight_layout()
#     #if plot_random_neurons == True:
#     #    r = np.random.randint(Frois.shape[0])
#     #fig.clear()
#     for c in range(n_conds):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
#         #     continue
       
    
#         plt.subplot(2, 4, c+1)
#         plt.title('Stimulus: ' + str(conds[c]) + 'Hz', fontsize = 9)
            
#         plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         plt.ylim((-2,6))
#         if normalize == 'dF/F':
#             plt.ylim((-1,1))
#         plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize = 8)
#         plt.ylabel(normalize, fontsize = 8)
#         plt.tick_params(axis='both', which='major', labelsize=7)
#         for t in range(n_trials):
#             plt.plot(range(totalframes), Frois_by_cond_tuned[r, c, t, :], color=str((0.4)+0.4*t/15))
#         plt.plot(range(totalframes), np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0))
#         plt.suptitle('roi {} '.format(r), fontsize=10)
#         ### Plot p_value of the mean
        
#         #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
#         # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
#         # p_val = t_test[1]
#         # plt.title('p value = ' + str(p_val.round(2)))
        
        
        
#         #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
#         #fig.waitforbuttonpress()
#         #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
     

# Frois_peak_frame = np.argmax(Frois, axis = 1)
# y_height = 0
# plt.figure(dpi = 450)
# for n in range(20):
#     r = np.random.randint(Frois.shape[0])
#     if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
#         plt.plot( np.linspace(1,18+36,18+36)-19, ( Frois[r,Frois_peak_frame[r]-18:Frois_peak_frame[r]+36] ) + y_height)
#         y_height += 0.5

# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.yticks(range(int(np.ceil(y_height))))
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.axvline(x=0)
# plt.title('Peak ' + normalize + ' for 20 neurons')


# y_height = 0
# plt.figure(dpi = 900)
# for n in range(20):
#     plt.plot( np.linspace(1,Frois.shape[1],Frois.shape[1]), ( Frois[ np.random.randint(Frois.shape[0]) ,:] ) + y_height, linewidth=0.2)
#     y_height += 3

        
# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.title( normalize + ' for 20 neurons')

