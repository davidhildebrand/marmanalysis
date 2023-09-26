#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')

#%% read suite2p output MATLAB file

filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/'
filename = '20221016d163614tUTC_Cadbury_MovingDotsFullField_2pRAMsp_fov1p46x1p46_res2umpx.log'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteB_200umdeep_1p46by1p46mm_2umppix_6p36Hz_59mW/suite2p/plane0/'
save_path = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteB_200umdeep_1p46by1p46mm_2umppix_6p36Hz_59mW/'

acq_framerate = 6.36
# *** TODO automatically identify stimulus duration
stimframes = int(np.ceil(2 * acq_framerate))

plot_random_neurons = False
tuning = 'percentile' #average #percentile #t-test
normalize = 'Z-score' #Z-score #dF/F
percentile = 90
tuning_index_thresh = 1
plot_least_tuned_first = False

plt.rcParams['figure.dpi'] = 300

isiframes = round(1 * acq_framerate)
totalframes = isiframes + stimframes + isiframes

iscell = np.load(pf + 'iscell.npy')
F = np.load(pf + 'F.npy')
stat = np.load(pf + 'stat.npy', allow_pickle=True)

cellinds = np.where(iscell[:, 0] == 1.0)[0]
Frois = F[cellinds, :]

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

# *** TODO use a saved pickle file instead of a text log
file = open(filepath + os.path.sep + filename, 'r')
lines = file.read().splitlines()
file.close()
trialdata = {}

# 41.9371         EXP     trial 0, stim start, grating, full field, drifting, cond=5, ori=225.0, tex=sin, size=[75.67137421 75.67137421], sf=[1.2 0. ], tf=4, mask=None, contrast=1.0, acqfr=222
for line in lines:
    if 'stim start' not in line:
        continue
    # print(line)
    col = line.split('trial')
    if not col:
        continue
    subcol = [sc.strip() for sc in col[1].split(',')]
    tmp_trial = int(subcol[0].strip())
    tmp_cond = int(subcol[5].split('=')[1].strip())
 
    tmp_f = float(subcol[13].split('=')[1].strip())
    tmp_acqfr = int(subcol[20].split('=')[1].strip())
    trialdata[tmp_trial] = {'cond' : tmp_cond,
                            'f' : tmp_f,
                            'acqfr' : tmp_acqfr}

# trialdataarr = [trial_idx, cond, ori, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['f'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)
all_stim_start_frames = trialdataarr[:,2]

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,1])
cond_num = len(conds)
trial_num = int(len(trialdata) / cond_num)

condinds = np.full([len(conds), trial_num], np.nan)
for c in range(cond_num):
    condinds[c] = np.argwhere(trialdataarr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]


#%%

# F__by_cond = [roi, cond, t, F]
FdFF_by_cond = np.full([ROI_num, cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
FdFF_by_cond_top_decile = np.full([FdFF.shape[0], cond_num], np.nan)
for c in range(cond_num):
    for r in range(ROI_num):
        for t in range(trial_num):
            FdFF_by_cond[r,c,t,:] = FdFF[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
FdFF_by_cond_stimon = FdFF_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
FdFF_by_cond_mean_response = np.mean(FdFF_by_cond_stimon, axis=2) #mean across trials and selecting stimulus window
#FFdFF_by_cond_mean_response = FdFF_by_cond_stimon.reshape([FdFF_by_cond_stimon.shape[0], FdFF_by_cond_stimon.shape[1], -1]) # susceptible to noise
            
# F__by_cond = [roi, cond, t, F]
Fzscore_by_cond = np.full([ROI_num, cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
Fzscore_by_cond_top_decile = np.full([ROI_num, cond_num], np.nan)
for c in range(cond_num):
    for r in range(ROI_num):
        for t in range(trial_num):
            Fzscore_by_cond[r,c,t,:] = Fzscore[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
Fzscore_by_cond_stimon = Fzscore_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
Fzscore_by_cond_mean_response = np.mean(Fzscore_by_cond_stimon, axis=2) #mean across trials and selecting stimulus window



#%
"""
Plots to the right are direction tuning curves. The plotted points are mean 
response per direction and error bars show the standard error. The responses 
were fitted with a Von Mises direction tuning curve. Each cell’s direction
selectivity index (DSI) is indicated on top of the tuning curve
"""

distxs = np.repeat(conds, trial_num)
distys = np.ravel(np.mean(FdFF_by_cond_stimon[0], axis=2))
plt.figure()
# np.mean(FdFF_by_cond_stimon[0], axis=2)
# plt.scatter(np.repeat(conds, trial_num), 
#             np.mean(FdFF_by_cond_stimon[0], axis=2), # mean of stimon frames
#             s=4, 
#             facecolors='none', 
#             edgecolors='k')
plt.scatter(distxs, 
            distys, # mean of stimon frames
            s=4, 
            facecolors='none', 
            edgecolors='k')
#kappa, loc, scale = scipy.stats.vonmises.fit(np.repeat(conds, trial_num), 
#            np.mean(FdFF_by_cond_stimon[0], axis=2), fscale=1)
#plt.plot()





if normalize == 'dF/F':
    Ftest = FdFF_by_cond_mean_response
elif normalize == 'Z-score':
    Ftest = Fzscore_by_cond_mean_response

if tuning == 't-test':
    t_test = scipy.stats.ttest_1samp(Ftest, 0, axis=2)
    p_vals = t_test[1]
    p_vals_min_cond = np.min(p_vals, axis = 1)
    tuning_index = 1 - p_vals_min_cond
elif tuning == 'percentile':
    first_quantile_all_conds = np.percentile(Ftest, percentile, axis=2)
    first_quantile_max_cond = np.max(first_quantile_all_conds, axis=1)
    tuning_index = first_quantile_max_cond
elif tuning == 'average':
    average = np.abs(np.mean(Ftest, axis=-1))
    average_max_cond = np.max(abs(average), axis=1)
    tuning_index = average_max_cond



# Frois_by_cond_tuned = Frois_by_cond[tuning_index > tuning_index_thresh]
FdFF_tuned_by_cond = FdFF_by_cond[tuning_index > tuning_index_thresh]
Fzscore_tuned_by_cond = Fzscore_by_cond[tuning_index > tuning_index_thresh]
tuning_index_tuned = tuning_index[tuning_index > tuning_index_thresh]

if plot_least_tuned_first:
    #Frois_by_cond_tuned = Frois_by_cond_tuned[(+tuning_index_tuned_neurons).argsort()]
    FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(+tuning_index_tuned).argsort()]
    Fzscore_tuned_by_cond_sorted = Fzscore_tuned_by_cond[(+tuning_index_tuned).argsort()]
else:
    # Frois_by_cond_tuned = Frois_by_cond_tuned[(-tuning_index_tuned_neurons).argsort()]
    FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(-tuning_index_tuned).argsort()]
    Fzscore_tuned_by_cond_sorted = Fzscore_tuned_by_cond[(-tuning_index_tuned).argsort()]
#tuning_index_tuned_neurons = tuning_index_tuned_neurons[(-tuning_index_tuned_neurons).argsort()]

print('Tuning index threshold: {}' .format(tuning_index_thresh))
n_tuned = Fzscore_tuned_by_cond.shape[0]
n_total = Frois.shape[0]
pct_tuned = round(100 * n_tuned / n_total, 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_tuned, n_total))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


#%%

# Histogram
plt.figure()
plt.hist(tuning_index, bins=100)
plt.xlabel('Tuning index: {}'.format(tuning))
plt.ylabel('ROIs')

for r in range(0, 101, 1): # range(FdFF_tuned_by_cond.shape[0]):
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

#%%
# Frois_peak_frame = np.argmax(Frois, axis=1)
# y_height = 0
# plt.figure()
# for n in range(20):
#     r = np.random.randint(Frois.shape[0])
#     if Frois_peak_frame[r] > 18 and Frois_peak_frame[r] < Frois.shape[1]-36:
#         plt.plot(np.linspace(1,18+36,18+36)-19, (Frois[r, Frois_peak_frame[r]-18:Frois_peak_frame[r]+36]) + y_height)
#         y_height += 0.5

# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.yticks(range(int(np.ceil(y_height))))
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.axvline(x=0)
# plt.title('Peak ' + normalize + ' for 20 neurons')


# y_height = 0
# plt.figure()
# for n in range(20):
#     plt.plot( np.linspace(1,Frois.shape[1],Frois.shape[1]), ( Frois[ np.random.randint(Frois.shape[0]) ,:] ) + y_height, linewidth=0.2)
#     y_height += 3

# plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)')
# plt.ylabel(normalize + ' (arbitrary baseline)')
# plt.title( normalize + ' for 20 neurons')


#%%
