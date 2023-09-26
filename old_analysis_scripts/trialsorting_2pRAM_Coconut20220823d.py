#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
#from ScanImageTiffReader import ScanImageTiffReader
import scipy

import os
os.chdir("C:/FreiwaldSync/MarmoScope/Analysis/")
import plot_map 

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')

#%% read suite2p output MATLAB file
# mf = '/Users/davidh/Data/Freiwald/20220609d_Louwho_ExpressionCheck_MarmoScope/suite2p_test/test2/plane0/Fall.mat'
# mat = scipy.io.loadmat(mf)

# iscell = np.array(mat['iscell'])
# F = np.array(mat['F'])

filepath = 'C:/FreiwaldSync/MarmoScope/Stimulus/Data/'

### Max15
filename = '20220823d201909tUTC_Coconut_Auditory_fov2x2.log'
pf = r'F:\Santiago\Analysis\Coconut\20220823d\Max15_pl15at0umdeep_2by2mm_6p45Hz_2p75umppix_100pct_stim_00002\p5pad\suite2p\plane0/'
# pf = r'F:\Santiago\Analysis\Coconut\20220823d\Max15_pl15at0umdeep_2by2mm_6p45Hz_2p75umppix_100pct_stim_00002\p5pad\unet_single_1024_2022_11_22_00_35\out\suite2p\plane0/'

### SP
# pf = r'F:\Santiago\Analysis\Coconut\20220822d\SP_100umdeep_0p6by0p6mm_1umppix_9p61Hz_3pct_stim_00001\suite2p\plane0\reg_tif\unet_single_1024_2022_11_28_02_41\DIout\suite2p\plane0/'
# filename = '20220822d183322tUTC_Coconut_Auditory.log'

acq_framerate = 6.45
stimframes = int(np.ceil(2 * acq_framerate))

 
plot_random_neurons = False
tuning = 'average' #average #percentile #t-test
normalize = 'Z-score' #Z-score #dF/F
tuning_index_thresh = 0.5
plot_least_tuned_neurons_first = True
shift_acqfr_bcDeepInt = 0# 5
n_neighbors = 10
ignore_wn_stim = True

isiframes = round(1 * acq_framerate)

iscell = np.load(pf + 'iscell.npy')
F = np.load(pf + 'F.npy') - 1 * np.load(pf + 'Fneu.npy')
stat = np.load(pf + 'stat.npy', allow_pickle = True)
ops = np.load(pf + 'ops.npy', allow_pickle = True).item()

percentile = 90


totalframes = isiframes + stimframes + isiframes


cellinds = np.where(iscell[:, 0] == 1.0)[0]
Frois = F[cellinds, :]

### 
if normalize == 'dF/F':
    Frois = ( Frois - np.mean(Frois[:,:], axis = 1)[:,None] ) / np.mean(Frois[:,:], axis = 1)[:,None]
elif normalize == 'Z-score':
    Frois = ( Frois - np.mean(Frois[:,:], axis = 1)[:,None] ) / np.std(Frois[:,:], axis = 1)[:,None]  
###


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



file = open(filepath + '/' + filename, 'r')
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
    tmp_cond = int(subcol[4].split('=')[1].strip())
 
    tmp_f = subcol[5].split('=')[1].strip()
    if tmp_f == 'wn':
       tmp_f = 0 

    tmp_acqfr = int(subcol[9].split('=')[1].strip())
    trialdata[tmp_trial] = {'cond' : tmp_cond,
                            'f' : tmp_f,
                            'acqfr' : tmp_acqfr}

# trialdataarr = [trial_idx, cond, ori, acqfr]
trialdataarr = np.full([len(trialdata), 3], np.nan)
for td in trialdata:
    trialdataarr[td] = [trialdata[td]['cond'], trialdata[td]['f'], trialdata[td]['acqfr']]
trialdataarr = trialdataarr.astype(int)

# condinds = [cond, trial_idx]
conds = np.unique(trialdataarr[:,1])
cond_num = len(conds)
trial_num = int(len(trialdata) / cond_num)

condinds = np.full([len(conds), trial_num], np.nan)
for c in range(cond_num):
    condinds[c] = np.argwhere(trialdataarr[:, 0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]-shift_acqfr_bcDeepInt



# Frois_by_cond = [ROIno, cond, t, F]
Frois_by_cond = np.full([Frois.shape[0], cond_num, trial_num, isiframes+stimframes+isiframes], np.nan)
Frois_by_cond_top_decile = np.full([Frois.shape[0], cond_num], np.nan)
for c in range(cond_num):
    for r in range(Frois.shape[0]):
        for t in range(trial_num):
            Frois_by_cond[r,c,t,:] = Frois[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
            if c == 7 and ignore_wn_stim:
                Frois_by_cond[r,c,t,:] = 0
        
Frois_by_cond_stim_on = Frois_by_cond[:,:,:,isiframes:(isiframes+stimframes)]

Frois_by_cond_mean_response = np.mean(Frois_by_cond_stim_on,axis = 2) #mean across trials and selecting stimulus window
#Frois_by_cond_mean_response = Frois_by_cond_stim_on.reshape([Frois_by_cond_stim_on.shape[0], Frois_by_cond_stim_on.shape[1], -1]) # susceptible to noise


if tuning == 't-test':
    t_test = scipy.stats.ttest_1samp(Frois_by_cond_mean_response, 0, axis = 2)
    p_vals = t_test[1]
    p_vals_min_cond = np.min(p_vals, axis = 1)
    tuning_index = 1 - p_vals_min_cond
elif tuning == 'percentile':
    first_quantile_all_conds = np.percentile(Frois_by_cond_mean_response, percentile, axis = 2)
    preferred_stim = np.argmax(first_quantile_all_conds, axis = 1)
    first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
    tuning_index = first_quantile_max_cond
elif tuning == 'max':
    first_quantile_all_conds = np.max(Frois_by_cond_mean_response, axis = 2)
    preferred_stim = np.argmax(first_quantile_all_conds, axis = 1)
    first_quantile_max_cond = np.max(first_quantile_all_conds, axis = 1)
    tuning_index = first_quantile_max_cond
elif tuning == 'average':
    average = np.abs(np.mean(Frois_by_cond_mean_response, axis = -1))
    average_max_cond = np.max(abs(average), axis = 1)
    tuning_index = average_max_cond
    preferred_stim = np.argmax(average, axis = 1)

plot_map.plot_tuning_map(stat,iscell,ops, preferred_stim/7, tuning_index, circular = False, iscell_thresh = 0.01, strength_thresh = tuning_index_thresh, n_neighbors = n_neighbors)

plt.figure(dpi = 450)
plt.hist(tuning_index, bins = 100)
plt.xlabel('Tuning index: {}'.format(tuning))
plt.ylabel('Neurons')


Frois_by_cond_tuned = Frois_by_cond[tuning_index > tuning_index_thresh]
tuning_index_tuned_neurons = tuning_index[tuning_index > tuning_index_thresh]

if plot_least_tuned_neurons_first:
    Frois_by_cond_tuned = Frois_by_cond_tuned[( + tuning_index_tuned_neurons).argsort()]
else:    
    Frois_by_cond_tuned = Frois_by_cond_tuned[( - tuning_index_tuned_neurons).argsort()]
    


print('Tuning index threshold: {}' .format(tuning_index_thresh))
n_tuned_neurons = Frois_by_cond_tuned.shape[0]
n_neurons = Frois.shape[0]
pct_tuned_neurons = round(100*n_tuned_neurons/n_neurons,2)
print('Tuned neurons: {}. Total neurons: {}.'.format(n_tuned_neurons,n_neurons))
print('Percentage of tuned neurons: {}%'.format(pct_tuned_neurons))

#Frois_by_cond_tuned = Frois_by_cond_tuned[(Frois_by_cond_tuned_least_preferred_cond).argsort()]

#     continue
for r in range(Frois.shape[0]):
    
    plt.pause(0.05)
    #plt.figure(dpi =300)
    plt.subplots(2, 4, constrained_layout=True, dpi = 600)
   # plt.tight_layout()
    if plot_random_neurons == True:
        r = np.random.randint(Frois.shape[0])
    #fig.clear()
    for c in range(cond_num):
        # if np.mean(Frois_by_cond_tuned[r, c, :, isiframes:isiframes+stimframes]) < np.mean(Frois_by_cond_tuned[r, c, :, 0:isiframes]):
        #     continue
       
    
        plt.subplot(2, 4, c+1)
        plt.title('Stimulus: ' + str(conds[c]) + 'Hz', fontsize = 9)
            
        plt.axvspan(isiframes, (isiframes + stimframes), color='0.9')
        plt.ylim((-2,6))
        if normalize == 'dF/F':
            plt.ylim((-1,1))
        plt.xlabel('Frame # (@'+str(acq_framerate)+'Hz)', fontsize = 8)
        plt.ylabel(normalize, fontsize = 8)
        plt.tick_params(axis='both', which='major', labelsize=7)
        for t in range(trial_num):
            plt.plot(range(totalframes), Frois_by_cond_tuned[r, c, t, :], color=str((0.4)+0.4*t/15))
        plt.plot(range(totalframes), np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0))
        plt.suptitle('roi {} '.format(r), fontsize=10)
        ### Plot p_value of the mean
        
        #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
        # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
        # p_val = t_test[1]
        # plt.title('p value = ' + str(p_val.round(2)))
        
        
        
        #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
        #fig.waitforbuttonpress()
        #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
     

Frois_peak_frame = np.argmax(Frois, axis = 1)
y_height = 0
plt.figure(dpi = 450)
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
plt.title( normalize + ' for 20 neurons')



        