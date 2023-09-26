#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')

#%% read suite2p output MATLAB file

filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/'
filename = '20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx.log'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/suite2p/plane0/'
save_path = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/roi_tuning/'

plot_rando_neurons = False
plot_least_tuned_first = False
tuning = 'fsi'#'percentile' #'t-test'
percentile = 90
normalize = 'Zscore' #'dF/F'#
tuning_index_thresh = 0.2
acq_framerate = 6.36 #9.61 #6.45
stimframes = int(np.ceil(2 * acq_framerate))
isiframes = round(1 * acq_framerate)

plt.rcParams['figure.dpi'] = 300

totalframes = isiframes + stimframes + isiframes

iscell = np.load(pf + 'iscell.npy')
F = np.load(pf + 'F.npy')
stat = np.load(pf + 'stat.npy', allow_pickle=True)

cellinds = np.where(iscell[:, 0] == 1.0)[0]
Frois = F[cellinds]
rois_to_exclude = np.where(np.std(Frois, axis=1) == 0)
rois_to_include = np.delete(np.arange(Frois.shape[0]), rois_to_exclude)
Frois = Frois[rois_to_include]

###
ROI_num = Frois.shape[0]
FdFF = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.mean(Frois, axis=1)[:,np.newaxis]
Fzscore = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.std(Frois, axis=1)[:,np.newaxis]


#% parse frame disp time log

# filepath = 'C:/Users/soterocoronel/Dropbox (Dropbox @RU)/Data/'
# filename = '20220823d192917tUTC_Coconut_Auditory_fov0p6x0p6.log'
# file = open(filepath + '/' + filename, 'r')
# lines = file.read().splitlines()
# file.close()

# for line in lines:
#     if not line:
#         continue
#     disptimes = [t.strip() for t in line.split(',') if t]
# disptimes = np.array(disptimes, dtype=float)
# disp_time_avg = np.mean(disptimes)
# disp_framerate_avg = 1 / disp_time_avg
# print('display average framerate was {:.3f} Hz'.format(disp_framerate_avg))

#%% parse log file

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


#%

# trialdataarr[trial_idx] = [cond/imageid, category_id, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['catid'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)
all_stim_start_frames = trialdataarr[:,2]

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,0])
cond_num = len(conds)
trial_num = int(len(trialdata) / cond_num)
cats = np.unique(trialdataarr[:,1])
cat_num = len(cats)
conds_per_cat = int(cond_num / cat_num)

condinds = np.full([len(conds), trial_num], np.nan)
for c in range(cond_num):
    condinds[c] = np.argwhere(trialdataarr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]


#%%

# F__by_cond = [roi, cond, t, F]
FdFF_by_cond = np.full([ROI_num, cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
FdFF_by_cond_top_decile = np.full([ROI_num, cond_num], np.nan)
for c in range(cond_num):
    for r in range(ROI_num):
        for t in range(trial_num):
            FdFF_by_cond[r,c,t,:] = FdFF[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
FdFF_by_cond_stimon = FdFF_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
FdFF_by_cond_mean_response = np.mean(FdFF_by_cond_stimon, axis=2) #mean across trials and selecting stimulus window
#FdFF_by_cond_mean_response = FdFF_by_cond_stimon.reshape([FdFF_by_cond_stimon.shape[0], FdFF_by_cond_stimon.shape[1], -1]) # susceptible to noise
            
# F__by_cond = [roi, cond, t, F]
Fzscore_by_cond = np.full([ROI_num, cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
Fzscore_by_cond_top_decile = np.full([ROI_num, cond_num], np.nan)
for c in range(cond_num):
    for r in range(ROI_num):
        for t in range(trial_num):
            Fzscore_by_cond[r,c,t,:] = Fzscore[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
Fzscore_by_cond_stimon = Fzscore_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
Fzscore_by_cond_mean_response = np.mean(Fzscore_by_cond_stimon, axis=2) #mean across trials and selecting stimulus window

# # Frois_by_cond = [ROIno, cond, trial, frame]
# Frois_by_cond = np.full([ROI_num, cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
# #Frois_by_cond_top_decile = np.full([ROI_num, cond_num], np.nan)
# for c in range(cond_num):
#     for r in range(ROI_num):
#         for t in range(trial_num):
#             Frois_by_cond[r,c,t,:] = Frois[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
# # Frois_by_cond_stimon = [ROIno, cond, trial, frame]
# Frois_by_cond_stimon = Frois_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
# # Frois_by_cond_mean_response = [ROIno, cond, frame]
# # mean across trials and selecting stimulus window
# Frois_by_cond_mean_response = np.mean(Frois_by_cond_stimon, axis=2)

catinds = np.full([len(cats), (conds_per_cat * trial_num)], np.nan)
for c in range(cat_num):
    catinds[c] = np.argwhere(trialdataarr[:, 1] == c).transpose()[0]
catinds = catinds.astype(int)
acqfr_by_cat = trialdataarr[catinds[:], 2]

# F__by_cat = [roi, cat, t, F]
FdFF_by_cat = np.full([ROI_num, cat_num, conds_per_cat * trial_num, isiframes+stimframes+isiframes], np.nan)
#FdFF_by_cat_top_decile = np.full([ROI_num, cat_num], np.nan)
for c in range(cat_num):
    for r in range(ROI_num):
        for t in range(conds_per_cat * trial_num):
            FdFF_by_cat[r,c,t,:] = FdFF[r,(acqfr_by_cat[c][t]-isiframes):(acqfr_by_cat[c][t]+stimframes+isiframes)]
# FdFF_by_cat_stimon = [roi, cat, trial, frame]
FdFF_by_cat_stimon = FdFF_by_cat[:,:,:,isiframes:(isiframes+stimframes)]
# Frois_by_cat_mean_response = [ROIno, cat, frame]
# mean across trials and selecting stimulus window
FdFF_by_cat_mean_response = np.mean(FdFF_by_cat_stimon, axis=2)

# F__by_cat = [roi, cat, t, F]
Fzscore_by_cat = np.full([ROI_num, cat_num, conds_per_cat * trial_num, isiframes+stimframes+isiframes], np.nan)
#Fzscore_by_cat_top_decile = np.full([ROI_num, cat_num], np.nan)
for c in range(cat_num):
    for r in range(ROI_num):
        for t in range(conds_per_cat * trial_num):
            Fzscore_by_cat[r,c,t,:] = FdFF[r,(acqfr_by_cat[c][t]-isiframes):(acqfr_by_cat[c][t]+stimframes+isiframes)]
# Frois_by_cat_stimon = [ROIno, cat, trial, frame]
Fzscore_by_cat_stimon = Fzscore_by_cat[:,:,:,isiframes:(isiframes+stimframes)]
# Frois_by_cat_mean_response = [ROIno, cat, frame]
# mean across trials and selecting stimulus window
Fzscore_by_cat_mean_response = np.mean(Fzscore_by_cat_stimon, axis=2)


#%%

if normalize == 'dF/F':
    Ftest_cond = FdFF_by_cond_mean_response
    Ftest_cat = FdFF_by_cat_mean_response
elif normalize == 'Zscore':
    Ftest_cond = Fzscore_by_cond_mean_response
    Ftest_cat = Fzscore_by_cat_mean_response

if tuning == 't-test':
    t_test_cond = scipy.stats.ttest_1samp(Ftest_cond, 0, axis=2)
    p_vals_cond = t_test_cond[1]
    p_vals_min_cond = np.min(p_vals_cond, axis=1)
    tuning_index_cond = 1 - p_vals_min_cond
    t_test_cat = scipy.stats.ttest_1samp(Ftest_cat, 0, axis=2)
    p_vals_cat = t_test_cat[1]
    p_vals_min_cat = np.min(p_vals_cat, axis=1)
    tuning_index_cat = 1 - p_vals_min_cat
elif tuning == 'percentile':
    first_quantile_all_conds = np.percentile(Ftest_cond, percentile, axis=2)
    first_quantile_max_cond = np.max(first_quantile_all_conds, axis=1)
    tuning_index_cond = first_quantile_max_cond
    first_quantile_all_cats = np.percentile(Ftest_cat, percentile, axis=2)
    first_quantile_max_cat = np.max(first_quantile_all_cats, axis=1)
    tuning_index_cat = first_quantile_max_cat
elif tuning == 'average':
    average_cond = np.abs(np.mean(Ftest_cond, axis=-1))
    average_max_cond = np.max(average_cond, axis=1)
    tuning_index_cond = average_max_cond
    average_cat = np.abs(np.mean(Ftest_cat, axis=-1))
    average_max_cat = np.max(average_cat, axis=1)
    tuning_index_cat = average_max_cat
elif tuning == 'fsi':
    #FSI = (mean responsefaces – mean responsenonface objects)/(mean responsefaces + mean responsenonface objects)
    Rcatframes = np.mean(Ftest_cat, axis=-1)
    Rcatnorm = Rcatframes + np.abs(np.min(Rcatframes))
    catidx_face = [c for c in categories if categories[c]=='m'][0]
    catidx_obj = [c for c in categories if categories[c]=='u'][0]
    Rfaces = Rcatframes[:,catidx_face]
    Robjs = Rcatnorm[:,catidx_obj]
    fsi = (Rfaces - Robjs) / (Rfaces + Robjs)
    tuning_index_cat = fsi
    tuning_index_cond = fsi
    

#%%
# plt.figure()
# plt.hist(tuning_index_cond, bins=1000)
# plt.xlabel('Tuning index (cond/perimage): {}'.format(tuning))
# plt.ylabel('Neurons')

# FdFF_by_cond_tuned = FdFF_by_cond[tuning_index_cond > tuning_index_thresh]
# Fzscore_by_cond_tuned = Fzscore_by_cond[tuning_index_cond > tuning_index_thresh]
# tuning_index_cond_tuned = tuning_index_cond[tuning_index_cond > tuning_index_thresh]
# if plot_least_tuned_first:
#     FdFF_by_cond_tuned_sorted = FdFF_by_cond_tuned[(+tuning_index_cond_tuned).argsort()]
#     Fzscore_by_cond_tuned_sorted = Fzscore_by_cond_tuned[(+tuning_index_cond_tuned).argsort()]
# else:    
#     FdFF_by_cond_tuned_sorted = FdFF_by_cond_tuned[(-tuning_index_cond_tuned).argsort()]
#     Fzscore_by_cond_tuned_sorted = Fzscore_by_cond_tuned[(-tuning_index_cond_tuned).argsort()]

# plt.figure()
# plt.hist(tuning_index_cat, bins=1000)
# plt.xlabel('Tuning index (category): {}'.format(tuning))
# plt.ylabel('Neurons')

tuned_index_cat = tuning_index_cat > tuning_index_thresh
FdFF_by_cat_tuned = FdFF_by_cat[tuned_index_cat]
Fzscore_by_cat_tuned = Fzscore_by_cat[tuned_index_cat]
tuning_index_cat_tuned = tuning_index_cat[tuned_index_cat]
plot_least_tuned_first = True
if plot_least_tuned_first:
    FdFF_by_cat_tuned_sorted = FdFF_by_cat_tuned[(+tuning_index_cat_tuned).argsort()]
    Fzscore_by_cat_tuned_sorted = Fzscore_by_cat_tuned[(+tuning_index_cat_tuned).argsort()]
else:    
    FdFF_by_cat_tuned_sorted = FdFF_by_cat_tuned[(-tuning_index_cat_tuned).argsort()]
    Fzscore_by_cat_tuned_sorted = Fzscore_by_cat_tuned[(-tuning_index_cat_tuned).argsort()]

# FdFF_tuned_by_cond = FdFF_by_cond[tuning_index_cond > tuning_index_thresh]
# Fzscore_tuned_by_cond = Fzscore_by_cond[tuning_index_cond > tuning_index_thresh]
# tuning_index_tuned_by_cond = tuning_index_cond[tuning_index_cond > tuning_index_thresh]
# FdFF_tuned_by_cat = FdFF_by_cat[tuning_index_cat > tuning_index_thresh]
# Fzscore_tuned_by_cat = Fzscore_by_cat[tuning_index_cat > tuning_index_thresh]
# tuning_index_tuned_by_cat = tuning_index_cat[tuning_index_cat > tuning_index_thresh]

# if plot_least_tuned_first:
#     FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(+tuning_index_tuned_by_cond).argsort()]
#     Fzscore_tuned_by_cond_sorted = Fzscore_tuned_by_cond[(+tuning_index_tuned_by_cond).argsort()]
#     FdFF_tuned_by_cat_sorted = FdFF_tuned_by_cat[(+tuning_index_tuned_by_cat).argsort()]
#     Fzscore_tuned_by_cat_sorted = Fzscore_tuned_by_cat[(+tuning_index_tuned_by_cat).argsort()]
# else:
#     FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(-tuning_index_tuned_by_cond).argsort()]
#     Fzscore_tuned_by_cond_sorted = Fzscore_tuned_by_cond[(-tuning_index_tuned_by_cond).argsort()]
#     FdFF_tuned_by_cat_sorted = FdFF_tuned_by_cat[(-tuning_index_tuned_by_cat).argsort()]
#     Fzscore_tuned_by_cat_sorted = Fzscore_tuned_by_cat[(-tuning_index_tuned_by_cat).argsort()]


#%%

# print(sum(Frois_by_cond_mean_response[0,5]))
# a = scipy.stats.ttest_1samp(Frois_by_cond_mean_response[0,5], 0)
# a[1]
print('Tuning index threshold: {}' .format(tuning_index_thresh))
n_tuned_neurons = FdFF_by_cat_tuned_sorted.shape[0]
n_neurons = ROI_num #Frois.shape[0]
pct_tuned_neurons = round((100 * n_tuned_neurons) / n_neurons, 2)
print('Tuned neurons: {}. Total neurons: {}.'.format(n_tuned_neurons,n_neurons))
print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))

#Frois_by_cond_tuned = Frois_by_cond_tuned[(Frois_by_cond_tuned_least_preferred_cond).argsort()]

# for r in range(Frois.shape[0]):
#     plt.pause(0.05)
#     plt.subplots(2, 4, constrained_layout=True,)
#     # plt.tight_layout()
#     if plot_rando_neurons == True:
#         r = np.random.randint(Frois.shape[0])
#     #fig.clear()
#     for c in range(cond_num):
#         # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
#         #     continue
#         plt.subplot(2, 4, c+1)
#         plt.title('Stim: ' + str(images[c]), fontsize=7)
#         plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
#         plt.ylim((-1,5))
#         plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize = 6)
#         plt.ylabel(normalize, fontsize = 6)
#         plt.tick_params(axis='both', which='major', labelsize=5)
#         for t in range(trial_num):
#             plt.plot(range(totalframes), Frois_by_cond_tuned[r, c, t, :], color=str((0.4)+0.4*t/15))
#         plt.plot(range(totalframes), np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0))
#         #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
#         #fig.waitforbuttonpress()
#         print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))

for r in range(Fzscore_by_cat_tuned_sorted.shape[0]):
    print(r)
    plt.pause(0.05)
    plt.subplots(1, 3, constrained_layout=True)
    for c in range(cat_num):
        plt.subplot(1, 3, c+1)
        plt.title('Stim: ' + str(categories[c]), fontsize=7)
        plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        plt.ylim((-1,5))
        plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize=6)
        plt.ylabel(normalize, fontsize=6)
        plt.tick_params(axis='both', which='major', labelsize=5)
        for t in range(conds_per_cat * trial_num):
            plt.plot(range(totalframes), Fzscore_by_cat_tuned_sorted[r,c,t,:], color=str((0.4)+0.4*t/Fzscore_by_cat_tuned_sorted.shape[2]))
        plt.plot(range(totalframes), np.mean(Fzscore_by_cat_tuned_sorted[r,c,:,:], axis=0))
        plt.suptitle('roi {} '.format(r), fontsize=10)
        print(np.std(np.mean(FdFF_by_cat_tuned_sorted[r,c,:,:], axis=0)))

#%%

for r in range(0, n_tuned_neurons, 1):
    ipd = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=((8+2)*2*150*ipd,(3+1)*300*ipd))
    #fig.subplots(nrows=2, ncols=8)
    fig.clf()
    fig.suptitle('roi {} '.format(r), fontsize=12)
    axes = fig.subplots(nrows=2, ncols=8)
    for c in range(cond_num):
        ax = axes[0,c]
        ax.set_title(str(conds[c]) + 'º', fontsize=10)
        if c == 0:
            #plt.xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
            #ax.set_xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
            ax.set_ylabel('Z-score', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([x * acq_framerate for x in range(5)])
            ax.set_xticklabels(['', 0, '', 2, ''])
        else:
            #ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.axis('off')
        ax.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        ax.set_ylim((-2,8))
        for t in range(trial_num):
            ax.plot(range(totalframes), Fzscore_tuned_by_cond_sorted[r,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(Fzscore_tuned_by_cond_sorted[r,c,:,:], axis=0), color='tab:green')
    for c in range(cond_num):
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
        for t in range(trial_num):
            ax.plot(range(totalframes), FdFF_tuned_by_cond_sorted[r,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(FdFF_tuned_by_cond_sorted[r,c,:,:], axis=0), color='tab:blue')    
        ### Plot p_value of the mean
        #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
        # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
        # p_val = t_test[1]
        # plt.title('p value = ' + str(p_val.round(2)))
        
        #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
        #fig.waitforbuttonpress()
        #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
    plt.show()
    fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_roi{}.svg'.format(r), format='svg', dpi=1200)
    plt.pause(0.05)







Frois_peak_frame = np.argmax(Frois, axis=1)
y_height = 0
plt.figure()
for n in range(20):
    r = np.random.randint(Frois.shape[0])
    if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
        plt.plot( np.linspace(1,18+36,18+36)-19, (Frois[r,Frois_peak_frame[r]-18:Frois_peak_frame[r]+36]) + y_height)
        y_height += 0.5
plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
plt.yticks(range(int(np.ceil(y_height))))
plt.ylabel(normalize + ' (arbitrary baseline)')
plt.axvline(x=0)
plt.title('Peak ' + normalize + ' for 20 neurons')

y_height = 0
plt.figure(dpi = 900)
for n in range(20):
    plt.plot(np.linspace(1,Frois.shape[1],Frois.shape[1]), (Frois[np.random.randint(Frois.shape[0]),:]) + y_height, linewidth=0.2)
    y_height += 3

plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
plt.ylabel(normalize + ' (arbitrary baseline)')
plt.title(normalize + ' for 20 neurons')
