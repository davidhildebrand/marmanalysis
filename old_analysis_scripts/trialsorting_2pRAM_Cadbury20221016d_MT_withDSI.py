#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import minimize as scipy_minimize
#from scipy.signal import find_peaks as find_peaks
from skimage import exposure, util
from warnings import warn

#matplotlib.use('MacOSX')
#matplotlib.use('TkAgg')

#%% Read suite2p outputs

filepath = '/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Data/'
filename = '20221016d163614tUTC_Cadbury_MovingDotsFullField_2pRAMsp_fov1p46x1p46_res2umpx.log'
pf = '/Users/davidh/Sync/Freiwald/MarmoScope/Analysis/Data/Cadbury/20221016d/SP_SiteB_200umdeep_1p46by1p46mm_2umppix_6p36Hz_59mW/suite2p/plane0/'
save_path = ''
# save_path = '/Users/davidh/Data/Freiwald/Analysis/Cadbury/20221016d_2pRAM/SP_SiteB_200umdeep_1p46by1p46mm_2umppix_6p36Hz_59mW'

acq_framerate = 6.36

# based on Pattadkal etal Priebe 2022 bioRxiv
#   https://doi.org/10.1101/2022.06.23.497220
# "All cell pairs with DSI ≥ 0.15 are considered."
dsi_tuning_thresh = 0.15
cell_probability_thresh = 0 #.005

# plot_random_neurons = False
# tuning = 'percentile' #average #percentile #t-test
# normalize = 'dF/F' #'Z-score' #Z-score #dF/F
# percentile = 90
# tuning_index_thresh = 0
# plot_least_tuned_first = False

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
ROIs = s2p_stat[cellinds]
Frois = s2p_F[cellinds]
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

deg_symbol = u'\N{DEGREE SIGN}'

def Rf(params, T):
    # based on Pattadkal etal Priebe 2022 bioRxiv
    #   https://doi.org/10.1101/2022.06.23.497220
    Tpref = params[0] # rad, preferred direction
    beta = params[1] # 'tuning width factor'
    c = params[2] # baseline
    a1 = params[3] # peak 1 maxiumum amplitude
    a2 = params[4] # peak 2 maxiumum amplitude
    R = (a1 * np.exp(beta * np.cos(T - Tpref))) + \
        (a2 * np.exp(beta * np.cos(np.pi + T - Tpref))) + \
        c
    return R


def gf(params, T):
    # based on Fahey etal Tolias 2019 bioRxiv 
    #   https://doi.org/10.1101/745323
    Tpref = params[0] # rad, preferred direction
    w = params[1] # peak concentration or 'tuning width factor'
    g = np.exp(-w * (1 - np.cos(T - Tpref)))
    return g


def vf(params, T):
    a0 = params[2] # baseline
    a1 = params[3] # peak 1 maxiumum amplitude
    a2 = params[4] # peak 2 maxiumum amplitude
    v = a0 + (a1 * gf(params, T)) + (a2 * gf(params, T - np.pi))
    return v


def dsi_model(params, T):
    r = Rf(params, T) # use Pattadkal etal Priebe 2022 
    # r = vf(params, T) # use Fahey etal Tolias 2019
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
    #peak_locs = find_peaks(fit_curve, height=0)[0]
    #peaks = fit_curve[peak_locs]
    #max_peak_loc = peak_locs[peaks.argmax()]
    Tpref = np.radians(max_peak_arg)

    if plotting:
        plt.figure()
        plt.scatter(np.degrees(thetas), measRs, s=4, facecolors='none', edgecolors='k')
        plt.plot(dsi_model(fit, np.radians(np.arange(0, 360))))
        plt.axvline(np.degrees(Tpref), color='m')
        ax = plt.gca()
        ax.set_xlabel('Direction (' + deg_symbol + ')', fontsize=8)
        ax.set_ylabel('dF/F', fontsize=8)
        #ax.set_xlim((0,360))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([d for d in range(0, 360, np.diff(np.degrees(thetas)).max().astype('int'))])
        #ax.set_xticklabels(['', 0, '', 2, ''])
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
              'a1={:.2f} a2={:.2f}'.format(a1_f, a2_f))
    
    return dsi, Tpref

# from matplotlib.offsetbox import AnchoredOffsetbox
# class AnchoredScaleBar(AnchoredOffsetbox):
#     def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
#                  pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None, 
#                  **kwargs):
#         """
#         Draw a horizontal and/or vertical  bar with the size in data coordinate
#         of the give axes. A label will be drawn underneath (center-aligned).
#         - transform : the coordinate frame (typically axes.transData)
#         - sizex,sizey : width of x,y bar, in data units. 0 to omit
#         - labelx,labely : labels for x,y bars; None to omit
#         - loc : position in containing axes
#         - pad, borderpad : padding, in fraction of the legend font size (or prop)
#         - sep : separation between labels and bars in points.
#         - **kwargs : additional arguments passed to base class constructor
#         """
#         from matplotlib.patches import Rectangle
#         #from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
#         from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea
#         bars = AuxTransformBox(transform)
#         if sizex:
#             bars.add_artist(Rectangle((0,0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
#         if sizey:
#             bars.add_artist(Rectangle((0,0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))
#
#         if sizex and labelx:
#             self.xlabel = TextArea(labelx)
#             bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
#         if sizey and labely:
#             self.ylabel = TextArea(labely)
#             bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)
#
#         AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
#                                    child=bars, prop=prop, frameon=False, **kwargs)


def plot_map(ROIs, tuning, tuning_mag, tuning_thresh=0, fov_size=(512,512), 
             circular=False, ref_image=None, scale_bar=False, um_per_px=None, 
             n_neighbors=None, save_path:str=''):
    # The values tuning and tuning_mag must be within [0,1].
    # 'circular' determines whether tuning has the same color for 0 and 1 
    # (True for MT, False for auditory)
    # TODO **** implement scale bar?
    
    dpi = plt.rcParams['figure.dpi']
    h, w = fov_size # rows/height/y, columns/width/x
    figsize = w / float(dpi), h / float(dpi)
    
    n_ROIs = len(ROIs)
    # TODO *** note that the >= might not be general (e.g. with FSI)
    tuned = tuning_mag >= tuning_thresh
    ROIs_tuned = ROIs[tuned]
    tuning_tuned = tuning[tuned]
    tuning_mag_tuned = tuning_mag[tuned]
    
    if tuning.max() > 1:
        warn(UserWarning('provided tuning index has values > 1 (out of range)'))
    
    assert len(ROIs_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
    n_ROIs_tuned = len(ROIs_tuned)
    
    # if scale_bar is True:
    #     fig, ax = plt.subplots()
    #     y = np.random.rand(1000)
    #     x = np.arange(y.shape[0])
    #     ax.plot(x, y)
    #     scalebar = AnchoredScaleBar(ax.transData,
    #                                 100, '100 um', 'lower right', 
    #                                 pad=0.1,
    #                                 color='black',
    #                                 bbox_to_anchor=(0.5,0.5),
    #                                 bbox_transform=ax.transAxes,
    #                                 frameon=False,
    #                                 size_vertical=0.05)
    #     scalebar.set_clip_on(False)
    #     ax.add_artist(scalebar)
    
    #f0 = plt.figure()
    f0 = plt.figure(figsize=figsize)
    ax = f0.add_axes([0, 0, 1, 1])
    plt.set_cmap('hsv')
    #plt.axis('off')
    ax.axis('off')
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
            #for rgb in range(3):
            #canvas[ry,rx,:] = colorsys.hsv_to_rgb(tuning_tuned[r], tuning_mag[r] / tuning_mag.max(), 1.0) #  abs(1 - 2 * abs(tuning_tuned[r] - rgb * 1/3)) #* tuning_mag[r]
            canvas[ry,rx,:] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
        else:
            for rgb in range(3):
                canvas[ry,rx,rgb] = abs(1 - 2 * abs(tuning_tuned[r] / 1.5 - rgb * 1/3)) #* tuning_mag[r]
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    #plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))#, quant_steps))#, alpha=1.0)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    #ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    f0.show()
    if save_path != '':
        now = datetime.now()
        dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        save_name = dt + '_ROIplot_thresh' + \
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
        ax0.set_xticklabels(['', ''], fontsize=20)
        #ax0.set_xticklabels(['0' + deg_symbol, '90' + deg_symbol], fontsize=20)
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
        #ax1.set_rlim([-1,1])
        f1.show()
        if save_path != '':
            now = datetime.now()
            dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
            save_name = dt + '_ROIplot_thresh' + \
                '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
                '_tuned{}of{}'.format(n_ROIs_tuned, n_ROIs) + \
                '_legend.png'
            f1.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    
        # # working but slow
        # # also relevant https://stackoverflow.com/questions/62531754/how-to-draw-a-hsv-color-wheel-using-matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='polar')
        # rho = np.linspace(0,1,100) # Radius of 1, distance from center to outer edge
        # phi = np.linspace(0, np.pi*2., 1000) # in radians, one full circle
        # RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi
        # h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
        # h = np.flip(h)        
        # s = RHO               # saturation is set as a function of radius
        # v = np.ones_like(RHO) # value is constant
        # # convert the np arrays to lists. This actually speeds up the colorsys call
        # h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
        # c = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
        # c = np.array(c)
        # ax.scatter(PHI, RHO, c=c)
        # _ = ax.axis('off')
        # if save_path != '':
        #     now = datetime.now()
        #     dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
        #     save_name = dt + '_ROIplot_continuous_withsat_' + \
        #         'legend.png'
        #     fig.savefig(save_path + os.path.sep + dt, dpi=dpi, transparent=True)

        # # working if image creation or saturation alpha is preferred
        # # also relevant https://rosettacode.org/wiki/Color_wheel
        # f3 = plt.figure()
        # imw = 512
        # imh = 512
        # radius = 512 / 2.0
        # cy, cx = imh / 2, imw / 2
        # pix = np.ones([imh, imw, 3])
        # for x in range(imw):
        #     for y in range(imh):
        #         rx = x - cx
        #         ry = y - cy
        #         s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius
        #         if s <= 1.0:
        #             h = ((np.arctan2(ry, rx) / np.pi) + 1.0) / 2.0
        #             rgb = colorsys.hsv_to_rgb(h, s, 1.0)
        #             pix[x,y,:] = [c for c in rgb]
        #             #pix[x,y,:] = tuple([int(round(c*255.0)) for c in rgb])
        # plt.imshow(pix)
        # f3.show()
        
       
    # Santi original
    # create colormap for reference
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


#%% Parse log file

# *** TODO load from a pickle file or pandas frame instead of a text log
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
n_conds = len(conds)
n_trials = int(len(trialdata) / n_conds)

condinds = np.full([len(conds), n_trials], np.nan)
for c in range(n_conds):
    condinds[c] = np.argwhere(trialdataarr[:,0] == c).transpose()[0]
condinds = condinds.astype(int)
acqfr_by_conds = trialdataarr[condinds[:], 2]


#%% Organize and average fluorescence traces

# F__by_cond = [roi, cond, t, F]
FdFF_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
FdFF_by_cond_top_decile = np.full([FdFF.shape[0], n_conds], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            FdFF_by_cond[r,c,t,:] = FdFF[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
FdFF_by_cond_Rstim = FdFF_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
FdFF_by_cond_meanR = np.mean(FdFF_by_cond_Rstim, axis=2) #mean across trials and selecting stimulus window
#FFdFF_by_cond_meanR = FdFF_by_cond_Rstim.reshape([FdFF_by_cond_Rstim.shape[0], FdFF_by_cond_Rstim.shape[1], -1]) # susceptible to noise
            
# F__by_cond = [roi, cond, t, F]
Fzsc_by_cond = np.full([n_ROIs, n_conds, n_trials, isiframes+stimframes+isiframes], np.nan)
Fzsc_by_cond_top_decile = np.full([n_ROIs, n_conds], np.nan)
for c in range(n_conds):
    for r in range(n_ROIs):
        for t in range(n_trials):
            Fzsc_by_cond[r,c,t,:] = Fzsc[r,(acqfr_by_conds[c][t]-isiframes):(acqfr_by_conds[c][t]+stimframes+isiframes)]
Fzsc_by_cond_Rstim = Fzsc_by_cond[:,:,:,isiframes:(isiframes+stimframes)]
Fzsc_by_cond_meanR = np.mean(Fzsc_by_cond_Rstim, axis=2) #mean across trials and selecting stimulus window


#%% Compute direction selectivity index and find preferred direction tuning angle

# dsiT(__by_roi) = [roi, [dsi, T]]
Ts = np.repeat(conds, n_trials)
# distxs = np.repeat(conds, n_trials)
Rs = np.full([n_ROIs, (n_conds * n_trials)], np.nan)
dsiT = np.full([n_ROIs, 2], np.nan)
for r in range(n_ROIs):
    Rs[r] = np.ravel(np.mean(FdFF_by_cond_Rstim[r], axis=2))
    dsiT[r] = calculate_dsi(Ts, Rs[r]) #, plotting=True, debugging=True)

# FdFF_test = FdFF_by_cond_meanR
# Fzsc_test = Fzsc_by_cond_meanR
# 
# # Define tuning indices using a t-tests
# # TODO *** check if second argument should be a calculated baseline
# tee_dFF = scipy.stats.ttest_1samp(FdFF_test, 0, axis=2)
# Pvals_dFF = tee_dFF[1]
# Pvals_dFF_min_cond = np.min(Pvals_dFF, axis=1)
# tunidx_tee_dFF = 1 - Pvals_dFF_min_cond
# tunidx_tee_dFF_argsrt = np.argsort(tunidx_tee_dFF)[::-1]
# tee_zsc = scipy.stats.ttest_1samp(Fzsc_test, 0, axis=2)
# Pvals_zsc = tee_zsc[1]
# Pvals_zsc_min_cond = np.min(Pvals_zsc, axis=1)
# tunidx_tee_zsc = 1 - Pvals_zsc_min_cond
# tunidx_tee_zsc_argsrt = np.argsort(tunidx_tee_zsc)[::-1]
# 
# # Define tuning indices using quantiles
# qt1_dFF_all_conds = np.percentile(FdFF_test, percentile, axis=2)
# qt1_dFF_max_cond = np.max(qt1_dFF_all_conds, axis=1)
# tunidx_qt1_dFF = qt1_dFF_max_cond
# tunidx_qt1_dFF_argsrt = np.argsort(tunidx_qt1_dFF)[::-1]
# qt1_zsc_all_conds = np.percentile(Fzsc_test, percentile, axis=2)
# qt1_zsc_max_cond = np.max(qt1_zsc_all_conds, axis=1)
# tunidx_qt1_zsc = qt1_zsc_max_cond
# tunidx_qt1_zsc_argsrt = np.argsort(tunidx_qt1_zsc)[::-1]
# 
# # Define tuning indices using average intensity
# avg_dFF = np.abs(np.mean(FdFF_test, axis=-1))
# avg_dFF_max_cond = np.max(abs(avg_dFF), axis=1)
# tunidx_avg_dFF = avg_dFF_max_cond
# tunidx_avg_dFF_argsrt = np.argsort(tunidx_avg_dFF)[::-1]
# avg_zsc = np.abs(np.mean(Fzsc_test, axis=-1))
# avg_zsc_max_cond = np.max(abs(avg_zsc), axis=1)
# tunidx_avg_zsc = avg_zsc_max_cond    
# tunidx_avg_zsc_argsrt = np.argsort(tunidx_avg_zsc)[::-1]


#%% Plot one experimentally measured distribution used for DSI fitting

# plt.figure()
# r = np.random.randint(0, n_ROIs)
# plt.scatter(Ts, 
#             Rs[r], # mean of stimon frames
#             s=4, 
#             facecolors='none', 
#             edgecolors='k')
# plt.plot()


#%% Define ROIs as tuned or untuned

print('DSI tuning threshold: {}' .format(dsi_tuning_thresh))
tunidx_dsi = dsiT[:,0]
DSI = dsiT[:,0]
Tprefs = dsiT[:,1]
Tprefs_norm = Tprefs / 360 # normalized to [0,1] range
tunidx_dsi_argsrt = np.argsort(tunidx_dsi)[::-1]
#n_ROIs_tuned = np.argwhere(tunidx_dsi[tunidx_dsi_argsrt] <= dsi_tuning_thresh)[0][0]
n_ROIs_tuned = np.argwhere(tunidx_dsi[tunidx_dsi_argsrt] >= dsi_tuning_thresh).shape[0]
pct_tuned = round(((100 * n_ROIs_tuned) / n_ROIs), 2)
print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_ROIs))
print('Percentage of tuned ROIs: {}%'.format(pct_tuned))

#% Compute tuning indices
# if normalize == 'dF/F':
#     Ftest = FdFF_by_cond_meanR
# elif normalize == 'Z-score':
#     Ftest = Fzsc_by_cond_meanR
# 
# if tuning == 't-test':
#     t_test = scipy.stats.ttest_1samp(Ftest, 0, axis=2)
#     p_vals = t_test[1]
#     p_vals_min_cond = np.min(p_vals, axis=1)
#     tuning_index = 1 - p_vals_min_cond
# elif tuning == 'percentile':
#     first_quantile_all_conds = np.percentile(Ftest, percentile, axis=2)
#     first_quantile_max_cond = np.max(first_quantile_all_conds, axis=1)
#     tuning_index = first_quantile_max_cond
# elif tuning == 'average':
#     average = np.abs(np.mean(Ftest, axis=-1))
#     average_max_cond = np.max(abs(average), axis=1)
#     tuning_index = average_max_cond


# 
# FdFF_by_cond_tuned_dsi = FdFF_by_cond[tunidx_dsi > dsi_tuning_thresh]
# FdFF_by_cond_tuned_tee = FdFF_by_cond[tunidx_tee_dFF > tuning_index_thresh]
# Fzsc_by_cond_tuned_tee = Fzsc_by_cond[tunidx_tee_zsc > tuning_index_thresh]
# FdFF_by_cond_tuned_qt1 = FdFF_by_cond[tunidx_qt1_dFF > tuning_index_thresh]
# Fzsc_by_cond_tuned_qt1 = Fzsc_by_cond[tunidx_qt1_zsc > tuning_index_thresh]
# FdFF_by_cond_tuned_avg = FdFF_by_cond[tunidx_avg_dFF > tuning_index_thresh]
# Fzsc_by_cond_tuned_avg = Fzsc_by_cond[tunidx_avg_zsc > tuning_index_thresh]
# tuning_index_tuned = tunidx_tee_dFF[tunidx_tee_dFF > tuning_index_thresh]
# 
# if plot_least_tuned_first:
#     #Frois_by_cond_tuned = Frois_by_cond_tuned[(+tuning_index_tuned_neurons).argsort()]
#     FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(+tuning_index_tuned).argsort()]
#     Fzsc_tuned_by_cond_sorted = Fzsc_tuned_by_cond[(+tuning_index_tuned).argsort()]
# else:
#     # Frois_by_cond_tuned = Frois_by_cond_tuned[(-tuning_index_tuned_neurons).argsort()]
#     FdFF_tuned_by_cond_sorted = FdFF_tuned_by_cond[(-tuning_index_tuned).argsort()]
#     Fzsc_tuned_by_cond_sorted = Fzsc_tuned_by_cond[(-tuning_index_tuned).argsort()]
# #tuning_index_tuned_neurons = tuning_index_tuned_neurons[(-tuning_index_tuned_neurons).argsort()]
# 
# print('Tuning index threshold: {}' .format(tuning_index_thresh))
# n_ROIs_tuned = Fzsc_tuned_by_cond.shape[0]
# n_total = Frois.shape[0]
# pct_tuned = round(100 * n_ROIs_tuned / n_total, 2)
# print('Tuned ROIs: {}. Total ROIs: {}.'.format(n_ROIs_tuned, n_total))
# print('Percentage of tuned ROIs: {}%'.format(pct_tuned))


#%% Plot histogram for the number of ROIs with corresponding tuning values

f0 = plt.figure()
plt.hist(tunidx_dsi, bins=100)
plt.xlabel('Direction-Selectivity Index')
plt.ylabel('ROIs')
plt.xlim([0,1])
plt.axvline(dsi_tuning_thresh, color='m')
#plt.axvline(-dsi_tuning_thresh, color='m')
f0.show()
if saving:
    now = datetime.now()
    dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(dsi_tuning_thresh).replace('.', 'p') + '.svg'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)
    save_name = dt + '_histogram_DSI_thresh' + \
        '{:.2f}'.format(dsi_tuning_thresh).replace('.', 'p') + '.png'
    f0.savefig(save_path + os.path.sep + save_name, dpi=dpi, transparent=True)


#%% Plot tuning map 

# plot_tuning_map(s2p_stat, s2p_iscell, s2p_ops, Tprefs, DSI, strength_thresh=0.15, circular=True)

plot_map(ROIs, Tprefs_norm, DSI, tuning_thresh=dsi_tuning_thresh, 
         fov_size=fov_size, circular=True, ref_image=ref_image, save_path=save_path)


#%% Plot responses across conditions for direction-tuned ROIs

for r in range(0, n_ROIs_tuned, 1):
    ridx = tunidx_dsi_argsrt[r]
    #ridx = tunidx_dsi_argsrt[-2]
    print('ROI {} with '.format(ridx) +
          'DSI={:.2f} and '.format(DSI[ridx]) +
          'Tpref={:.2f}'.format(Tprefs[ridx]))
    ipd = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=((8+2)*2*150*ipd,(4+1)*300*ipd))
    #fig.subplots(nrows=2, ncols=8)
    fig.clf()
    fig.suptitle('roi {} ({})'.format(r, ridx), fontsize=12)
    axes = fig.subplots(nrows=2, ncols=8)
    for c in range(n_conds):
        ax = axes[0,c]
        ax.set_title(str(conds[c]) + deg_symbol, fontsize=10)
        if c == 0:
            #plt.xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
            #ax.set_xlabel('Frame (@'+str(acq_framerate)+'Hz)', fontsize=8)
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
        #ax.set_ylim((-2,8))
        ax.set_ylim((np.min(Fzsc_by_cond[ridx,:,:,:]) - 0.2,
                     np.max(Fzsc_by_cond[ridx,:,:,:]) + 0.2))
        for t in range(n_trials):
            ax.plot(range(totalframes), Fzsc_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(Fzsc_by_cond[ridx,c,:,:], axis=0), color='tab:green')
    for c in range(n_conds):
        ax = axes[1,c]
        if c == 0:
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
        #ax.set_ylim((-1,3))
        ax.set_ylim((np.min(FdFF_by_cond[ridx,:,:,:]) - 0.1,
                     np.max(FdFF_by_cond[ridx,:,:,:]) + 0.1))
        for t in range(n_trials):
            ax.plot(range(totalframes), FdFF_by_cond[ridx,c,t,:], color=str((0.4)+0.4*t/15))
        ax.plot(range(totalframes), np.mean(FdFF_by_cond[ridx,c,:,:], axis=0), color='tab:blue')    
        ### Plot p_value of the mean
        #t_test = scipy.stats.ttest_1samp(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0), 0) ## mean trace
        # t_test = scipy.stats.ttest_1samp(Frois_by_cond_tuned[r, c].flatten(), 0) ## individual traces 
        # p_val = t_test[1]
        # plt.title('p value = ' + str(p_val.round(2)))
        
        #fig.suptitle('roi {} cond {}'.format(r, c), fontsize=16)
        #fig.waitforbuttonpress()
        #print(np.std(np.mean(Frois_by_cond_tuned[r, c, :, :], axis=0)))
    plt.show()
    #if saving:
    #    fig.savefig(save_path + os.path.sep + 'CadBury_20221016d_roi{}.svg'.format(r), format='svg', dpi=1200)
    plt.pause(0.05)

# calculate_dsi(Ts, Rs[ridx], plotting=True, debugging=True)

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


# from suite2p.io import BinaryFile
# s2p_regfile = pf + os.path.sep + 'data.bin'
# bin_size = 1 #int(max(1, s2p_ops['nframes'] // s2p_ops['nbinned'], np.round(s2p_ops['tau'] * s2p_ops['fs'])))
# print('Binning movie in chunks of length %2.2d' % bin_size)
# with BinaryFile(read_filename=s2p_regfile, Ly=s2p_ops['Ly'], Lx=s2p_ops['Lx']) as f:
#     mov = f.bin_movie(bin_size=bin_size,
#                       bad_frames=s2p_ops.get('badframes'),
#                       y_range=s2p_ops['yrange'],
#                       x_range=s2p_ops['xrange'])

