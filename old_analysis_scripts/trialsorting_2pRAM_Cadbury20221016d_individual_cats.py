#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
from warnings import warn

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')

#%% read suite2p output MATLAB file

filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/'
filename = '20221016d152631tUTC_Cadbury_Images_2pRAMsp_fov0p73x0p73_res1umpx.log'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/suite2p/plane0/'
save_path = ''
#save_path = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteA_200umdeep_0p73by0p73mm_1umppix_6p36Hz_59mW/roi_tuning/'

plot_rando_neurons = False
tuning = 'fsi' #'percentile' #'t-test'
percentile = 90
normalize = 'Zscore' #'dF/F'#
tuning_index_thresh = 0
cell_probability_thresh = 0.0
plot_least_tuned_neurons_first = False
acq_framerate = 6.36 #9.61 #6.45

stimframes = int(np.ceil(2 * acq_framerate))
# TODO *** base this on actual trial data!
isiframes = round(1 * acq_framerate)




totalframes = isiframes + stimframes + isiframes

# iscell = np.array(mat['iscell'])
# F = np.array(mat['F'])
# cellinds = np.where(iscell[:, 0] == 1.0)[0]
s2p_iscell = np.load(pf + 'iscell.npy')
s2p_F = np.load(pf + 'F.npy')
s2p_stat = np.load(pf + 'stat.npy', allow_pickle = True)
cellinds = np.where(s2p_iscell[:,1] >= cell_probability_thresh)[0]
tmpROIs = s2p_stat[cellinds]
Frois = s2p_F[cellinds]
ROIidx_excl = np.where(np.std(Frois, axis=1) == 0)
ROIidx_incl = np.delete(np.arange(Frois.shape[0]), ROIidx_excl)
ROIs = tmpROIs[ROIidx_incl]
Frois = Frois[ROIidx_incl]

plt.rcParams['figure.dpi'] = 300

### 
if normalize == 'dF/F':
    Frois = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.mean(Frois, axis=1)[:,np.newaxis]
elif normalize == 'Zscore':
    Frois = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.std(Frois, axis=1)[:,np.newaxis]  
###

# FdFF = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.mean(Frois, axis=1)[:,np.newaxis]
# Fzscore = (Frois - np.mean(Frois, axis=1)[:,np.newaxis]) / np.std(Frois, axis=1)[:,np.newaxis]


#%% parse frame disp time log

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
        tmp_category = tmp_image[0:3]
        if tmp_category[2] == '.':
            tmp_category = tmp_category[0] + '0' + tmp_category[1]
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
images = {v: k for k, v in images.items()}

#%%

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

# Frois_by_cond = [ROIno, cond, trial, frame]
Frois_by_cond = np.full([Frois.shape[0], cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
Frois_by_cond_top_decile = np.full([Frois.shape[0], cond_num], np.nan)
for c in range(cond_num):
    for r in range(Frois.shape[0]):
        for t in range(trial_num):
            Frois_by_cond[r,c,t,:] = Frois[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]

# Frois_by_cond_stimon = [ROIno, cond, trial, frame]
Frois_by_cond_stimon = Frois_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
# Frois_by_cond_mean_response = [ROIno, cond, frame]
# mean across trials and selecting stimulus window
Frois_by_cond_mean_response = np.mean(Frois_by_cond_stimon, axis=2)

catinds = np.full([len(cats), (conds_per_cat * trial_num)], np.nan)
for c in range(cat_num):
    catinds[c] = np.argwhere(trialdataarr[:, 1] == c).transpose()[0]
catinds = catinds.astype(int)
acqfr_by_cat = trialdataarr[catinds[:], 2]

# Frois_by_cat = [ROIno, cat, t, F]
Frois_by_cat = np.full([Frois.shape[0], cat_num, conds_per_cat * trial_num, isiframes+stimframes+isiframes], np.nan)
Frois_by_cat_top_decile = np.full([Frois.shape[0], cat_num], np.nan)
for c in range(cat_num):
    for r in range(Frois.shape[0]):
        for t in range(conds_per_cat * trial_num):
            Frois_by_cat[r,c,t,:] = Frois[r,(acqfr_by_cat[c][t]-isiframes):(acqfr_by_cat[c][t]+stimframes+isiframes)]


Frois_by_cat_stimon = Frois_by_cat[:,:,:,isiframes:(isiframes+stimframes)]
#mean across trials and selecting stimulus window
Frois_by_cat_mean_response = np.mean(Frois_by_cat_stimon, axis=2)


#%%

if tuning == 't-test':
    t_test_cond = scipy.stats.ttest_1samp(Frois_by_cond_mean_response, 0, axis=2)
    p_vals_cond = t_test_cond[1]
    p_vals_min_cond = np.min(p_vals_cond, axis=1)
    tuning_index = 1 - p_vals_min_cond
    t_test_cat = scipy.stats.ttest_1samp(Frois_by_cat_mean_response, 0, axis=2)
    p_vals_cat = t_test_cat[1]
    p_vals_min_cat = np.min(p_vals_cat, axis=1)
    tuning_index_cat = 1 - p_vals_min_cat
elif tuning == 'percentile':
    first_quantile_all_conds = np.percentile(Frois_by_cond_mean_response, percentile, axis=2)
    first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
    tuning_index_cond = first_quantile_max_cond
    first_quantile_all_cats = np.percentile(Frois_by_cat_mean_response, percentile, axis=2)
    first_quantile_max_cat = np.max(first_quantile_all_cats, axis = 1)
    tuning_index_cat = first_quantile_max_cat
elif tuning == 'average':
    average_cond = np.abs(np.mean(Frois_by_cond_mean_response, axis=-1))
    average_max_cond = np.max(average_cond, axis=1)
    tuning_index_cond = average_max_cond
    average_cat = np.abs(np.mean(Frois_by_cat_mean_response, axis=-1))
    average_max_cat = np.max(average_cat, axis=1)
    tuning_index_cat = average_max_cat
elif tuning == 'fsi':
    #FSI = (mean responsefaces – mean responsenonface objects)/(mean responsefaces + mean responsenonface objects)
    #average_cond = np.abs(np.mean(Frois_by_cond_mean_response, axis=-1))
    Rcatframes = np.mean(Frois_by_cat_mean_response, axis=-1)
    Rcatnorm = Rcatframes + np.abs(np.min(Rcatframes))
    catidx_face = [c for c in categories if categories[c][0]=='m'][0]
    catidx_obj = [c for c in categories if categories[c][0]=='u'][0]
    catidx_body = [c for c in categories if categories[c][0]=='b'][0]
    Rfaces = Rcatnorm[:,catidx_face]
    Robjs = Rcatnorm[:,catidx_obj]
    fsi = (Rfaces - Robjs) / (Rfaces + Robjs)
    tuning_index_cat = fsi
    tuning_index_cond = fsi
else:
    warn('no tuning selected')



#%%
plt.figure()
plt.hist(tuning_index_cond, bins=1000)
plt.xlabel('Tuning index (cond/perimage): {}'.format(tuning))
plt.ylabel('Neurons')

if tuning == 'fsi':
    Frois_by_cond_tuned = Frois_by_cond[np.abs(tuning_index_cond) > tuning_index_thresh]
Frois_by_cond_tuned = Frois_by_cond[tuning_index_cond > tuning_index_thresh]
tuning_index_cond_tuned_neurons = tuning_index_cond[tuning_index_cond > tuning_index_thresh]
if plot_least_tuned_neurons_first:
    Frois_by_cond_tuned = Frois_by_cond_tuned[(+tuning_index_cond_tuned_neurons).argsort()]
else:    
    Frois_by_cond_tuned = Frois_by_cond_tuned[(-tuning_index_cond_tuned_neurons).argsort()]
tuning_index_cond_tuned_neurons = tuning_index_cond_tuned_neurons[(-tuning_index_cond_tuned_neurons).argsort()]

plt.figure()
plt.hist(tuning_index_cat, bins=1000)
plt.xlabel('Tuning index (category): {}'.format(tuning))
plt.ylabel('Neurons')
# plt.xlim([-1,1])

Frois_by_cat_tuned = Frois_by_cat[tuning_index_cat > tuning_index_thresh]
tuning_index_cat_tuned_neurons = tuning_index_cat[tuning_index_cat > tuning_index_thresh]
if plot_least_tuned_neurons_first:
    Frois_by_cat_tuned = Frois_by_cat_tuned[(+tuning_index_cat_tuned_neurons).argsort()]
else:    
    Frois_by_cat_tuned = Frois_by_cat_tuned[(-tuning_index_cat_tuned_neurons).argsort()]


#%%

print(sum(Frois_by_cond_mean_response[0,5]))
a = scipy.stats.ttest_1samp(Frois_by_cond_mean_response[0,5], 0)
a[1]
print('Tuning index threshold: {}' .format(tuning_index_thresh))
n_tuned_neurons = Frois_by_cond_tuned.shape[0]
n_neurons = Frois.shape[0]
pct_tuned_neurons = round((100 * n_tuned_neurons) / n_neurons, 2)
print('Tuned neurons: {}. Total neurons: {}.'.format(n_tuned_neurons,n_neurons))
print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))

#Frois_by_cond_tuned = Frois_by_cond_tuned[(Frois_by_cond_tuned_least_preferred_cond).argsort()]

# for r in range(Frois.shape[0]):
#     plt.pause(0.05)
#     plt.subplots(2, 4, constrained_layout=True)
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

# metric_cond = first_quantile_all_conds
# metric_max_cond = np.max(metric_cond, axis=-1)
# metric_cond = metric_cond[metric_max_cond > 1.25]

catarr = np.array(list(categories.values()))
sorting_ind = np.argsort(catarr)
tmp_ind = sorting_ind.copy()
tmp_ind[0:int(sorting_ind.shape[0]/3)] = sorting_ind[int(sorting_ind.shape[0]/3):int(2*sorting_ind.shape[0]/3)]
tmp_ind[int(sorting_ind.shape[0]/3):int(2*sorting_ind.shape[0]/3)] = sorting_ind[0:int(sorting_ind.shape[0]/3)]
sorting_ind = tmp_ind

categories_sorted = np.array(list(categories.values()))
categories_sorted = categories_sorted[sorting_ind]

# metric_cond = metric_cond[:,sorting_ind]
# metric_cond_mean_face = np.mean(metric_cond[:,20:40], axis = 1)
# metric_cond_mean_all = np.mean(metric_cond, axis = 1)
# metric_cond_diff_face = metric_cond_mean_face - metric_cond_mean_all
# metric_cond = metric_cond[np.argsort(metric_cond_mean_face - metric_cond_mean_all)]

# plt.rcParams['figure.dpi']= 1000

# plt.figure(dpi=1000)
# plt.imshow(metric_cond)
# plt.clim(0,1)
# plt.title('2-D Heat Map in Matplotlib')
# plt.colorbar()
# plt.xticks(range(60),categories_sorted)
# plt.xticks(fontsize=3, rotation=90)
# plt.show()

Frois_by_cat_tuned_sorted = Frois_by_cat_tuned[:,sorting_ind]
cats = categories_sorted

plt.figure(dpi=1000)
plt.imshow(np.mean(np.mean(Frois_by_cat_tuned_sorted[:,:,:,isiframes:isiframes+stimframes], axis=-1), axis=-1), cmap='bwr')
plt.clim(-1,1)
#plt.title('2-D Heat Map in Matplotlib')
#plt.colorbar()
plt.tick_params(left=False)
ax = plt.gca()
ax.tick_params(left=False, right=False, labelleft=False)
ax.set_xticks([c for c in range(0,60+1,20)])#, categories_sorted)
ax.set_xticklabels([], fontsize=3, rotation=90)
plt.show()

#%%

for r in range(Frois_by_cat_tuned.shape[0]):
    print(r)
    plt.pause(0.05)
    plt.subplots(6, 10, constrained_layout=True, dpi=1000)
    if plot_rando_neurons == True:
        r = np.random.randint(Frois_by_cat_tuned.shape[0])
    for c in range(cat_num):
        # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
        #     continue
        plt.subplot(6, 10, c+1)
        plt.title('Stim: ' + str(categories_sorted[c]), fontsize=7)
        plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        plt.ylim((-2.5,2.5))
        #plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize=6)
        #plt.ylabel(normalize, fontsize=6)
        #plt.tick_params(axis='both', which='major', labelsize=5)
        for t in range(conds_per_cat * trial_num):
            plt.plot(range(totalframes), Frois_by_cat_tuned[r,c,t,:], color=str((0.4)+0.4*t/Frois_by_cat_tuned.shape[2]))
        plt.plot(range(totalframes), np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0))
        if c > 0:
            plt.xticks([])
            plt.yticks([])
        #plt.suptitle('roi {} '.format(r), fontsize=10)
        #plt.waitforbuttonpress()
        print(np.std(np.mean(Frois_by_cat_tuned[r,c,:,:], axis=0)))

Frois_peak_frame = np.argmax(Frois, axis = 1)
y_height = 0
plt.figure()
for n in range(20):
    r = np.random.randint(Frois.shape[0])
    if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
        plt.plot( np.linspace(1,18+36,18+36)-19, ( Frois[r,Frois_peak_frame[r]-18:Frois_peak_frame[r]+36] ) + y_height)
        y_height += 0.5
plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
plt.yticks(range(int(np.ceil(y_height))))
plt.ylabel(normalize + ' (arbitrary baseline)')
plt.axvline(x=0)
plt.title('Peak ' + normalize + ' for 20 neurons')

y_height = 0
plt.figure(dpi = 900)
for n in range(20):
    plt.plot( np.linspace(1,Frois.shape[1],Frois.shape[1]), ( Frois[ np.random.randint(Frois.shape[0]) ,:] ) + y_height, linewidth=0.2)
    y_height += 3

        
plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
plt.ylabel(normalize + ' (arbitrary baseline)')
plt.title(normalize + ' for 20 neurons')
