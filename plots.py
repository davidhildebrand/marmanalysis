#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import colorsys
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import util
from skimage.exposure import rescale_intensity as ski_rescale_intensity
from skimage.transform import rotate as ski_rotate

from warnings import warn


def set_plot_text_settings():
    plt.rc('axes', titlesize=4, labelsize=6)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('legend', fontsize=6)
    plt.rc('figure', titlesize=4)


# % Define plotting function for histograms of selectivity metrics
def plot_hist_fsi(fsis, threshold=1/3, bins=41, title: str = '', save_path: str = ''):
    dpi = plt.rcParams['figure.dpi']

    f, ax1 = plt.subplots()
    plt.hist(fsis, bins=bins, range=(-1, 1), color='0.5')
    plt.xlim([-1, 1])
    ax1.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax1.set_xticks([-0.75, -0.25, 0.25, 0.75], minor=True)
    ax1.set_xticklabels([None, None, None, None], minor=True)
    plt.xlabel('Face-Selectivity Index')
    plt.ylabel('Number of ROIs')
    if title != '':
        ax1.set_title(title)

    ax2 = ax1.twinx()
    n_rois = len(fsis)
    weights = np.ones(n_rois) / n_rois
    plt.hist(fsis, weights=weights, bins=bins, range=(-1, 1), edgecolor='none', facecolor='none')
    plt.ylabel('Fraction of ROIs')

    if threshold is not None:
        if threshold != 0:
            plt.axvline(-threshold, color='0.2', linestyle='dashed', linewidth=1)
            plt.axvline(threshold, color='0.2', linestyle='dashed', linewidth=1)
        else:
            plt.axvline(threshold, color='0.2', linestyle='dashed', linewidth=1)

    set_plot_text_settings()
    f.tight_layout()
    f.show()
    if save_path != '':
        f.savefig(save_path, dpi=dpi, transparent=True)


def plot_hist_dprime(dprimes, threshold=0.2, bins=41, title: str = '', save_path: str = ''):
    dpi = plt.rcParams['figure.dpi']

    f, ax1 = plt.subplots()
    plt.hist(dprimes, bins=bins, color='0.5')
    # plt.xlim([-1, 1])
    # ax1.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    # ax1.set_xticks([-0.75, -0.25, 0.25, 0.75], minor=True)
    # ax1.set_xticklabels([None, None, None, None], minor=True)
    plt.xlabel('d′')
    plt.ylabel('Number of ROIs')
    if title != '':
        ax1.set_title(title)

    ax2 = ax1.twinx()
    n_rois = len(dprimes)
    weights = np.ones(n_rois) / n_rois
    plt.hist(dprimes, weights=weights, bins=bins, edgecolor='none', facecolor='none')
    plt.ylabel('Fraction of ROIs')

    if threshold is not None:
        if threshold != 0:
            plt.axvline(-threshold, color='0.2', linestyle='dashed', linewidth=1)
            plt.axvline(threshold, color='0.2', linestyle='dashed', linewidth=1)
        else:
            plt.axvline(threshold, color='0.2', linestyle='dashed', linewidth=1)

    set_plot_text_settings()
    f.tight_layout()
    f.show()
    if save_path != '':
        f.savefig(save_path, dpi=dpi, transparent=True)


def auto_level_s2p_image(image, target_median=5140):
    from skimage.util import img_as_uint, img_as_float64

    if image.ndim > 2:
        warn('auto_level_image may work slowly for image stacks')

    image = image - image.min()
    image = img_as_uint(image / 65535)
    
    high = 100.0
    while np.median(image) < target_median and high > 0:
        high = high - 0.5
        # print('Rescaling mean image. (median = {}, high = {})'.format(np.median(image), high))
        pl, ph = np.percentile(image, [0, high])
        image = img_as_uint(ski_rescale_intensity(image, in_range=(pl, ph)))
        
    return img_as_float64(image)


def plot_overlays_roi(rois, colors, size=None, 
                      bgimage=None, flip='lr', rotate=-90, scale_bar=False, um_per_px=None,
                      title: str = '', save_path: str = ''):
    n_rois = len(rois)
    n_colors = len(colors)
    assert n_rois == n_colors 

    if bgimage is not None:
        if size is not None and size != bgimage.shape:
            warn('input bgimage size does not match input size parameter, using bgimage size')
        ref = ski_rescale_intensity(util.img_as_float64(bgimage))
        if bgimage.ndim == 2:
            # Copy single channel bgimage to form an RGB image
            canvas = np.stack((ref,) * 3, axis=-1)
        elif bgimage.ndim == 3:
            if bgimage.shape[2] == 3:
                pass
            else:
                warn('unsupported input bgimage type (grayscale or RGB)')
        else:
            warn('unsupported input bgimage type (grayscale or RGB)')
        h, w, _ = canvas.shape  # rows/h/y, columns/w/x, channels
    else:
        if size is not None:
            h, w = size  # rows/h/y, columns/w/x 
        else:
            warn('no input bgimage or input size, estimating from ROI mask positions')
            w = np.array([rois[r]['xpix'].max() for r in range(n_rois)]).max()  # columns/w/x
            h = np.array([rois[r]['ypix'].max() for r in range(n_rois)]).max()  # rows/h/y
        canvas = np.zeros([h, w, 3], dtype=np.float64)

    for r, rt in enumerate(rois):
        ry = rt['ypix']
        rx = rt['xpix']
        canvas[ry, rx, :] = colors[r]

    match flip:
        case 'lr':
            canvas = np.fliplr(canvas)
        case 'ud':
            canvas = np.flipud(canvas)
        case None:
            pass
        case _:
            warn('unsupported flip parameter')
            
    if rotate is not None and rotate != 0:
        if rotate % 90 == 0:
            k = rotate / -90
            canvas = np.rot90(canvas, -k)
        else:
            canvas = ski_rotate(canvas, rotate)
        h, w, _ = canvas.shape  # rows/h/y, columns/w/x, channels

    f = plt.figure(figsize=(w / float(plt.rcParams['figure.dpi']), h / float(plt.rcParams['figure.dpi'])))  # (w, h), in
    ax = f.add_axes((0, 0, 1, 1))
    plt.set_cmap('hsv')
    ax.axis('off')
    ax.set_frame_on(False)
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    ax.imshow(canvas, interpolation='none', cmap='hsv')
    ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
    
    if title != '':
        ax.set_title(title)
    set_plot_text_settings()
    f.show()
    if save_path != '':
        f.savefig(save_path, dpi=dpi, transparent=True)
        f.savefig(save_path, dpi=plt.rcParams['figure.dpi'], transparent=True)


def plot_map(rois, tuning, tuning_mag, tuning_thresh=0, size=(512, 512),
             circular=False, image=None, scale_bar=False, um_per_px=None,
             n_neighbors=None, save_path: str = ''):
    # The values tuning and tuning_mag must be within [0,1].
    # 'circular' determines whether tuning has the same color for 0 and 1
    # (True for MT, False for auditory)
    # TODO **** implement scale bar?

    dpi = plt.rcParams['figure.dpi']
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
    ax.axis('off')
    ax.set_frame_on(False)
    if image is not None:
        ilow, ihigh = np.percentile(image, (1.0, 99.98))
        ref_f64 = util.img_as_float64(image)
        ref_rescale = ski_rescale_intensity(ref_f64, in_range=(ilow, ihigh))
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
    
    set_plot_text_settings()
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
        
        set_plot_text_settings()
        f1.show()
        if save_path != '':
            now = datetime.now()
            dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
            save_name = dt + '_ROIplot_FSIzsc_thresh' + \
                        '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
                        '_tuned{}of{}'.format(n_rois_tuned, n_rois) + \
                        '_legend.png'
            f1.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)


# def plot_map(rois, tuning, tuning_mag, tuning_thresh=0, size=(512, 512),
#              circular=False, image=None, scale_bar=False, um_per_px=None,
#              n_neighbors=None, save_path: str = ''):
#     # The values tuning and tuning_mag must be within [0,1].
#     # 'circular' determines whether tuning has the same color for 0 and 1
#     # (True for MT, False for auditory)
#     # TODO **** implement scale bar?
#
#     dpi = plt.rcParams['figure.dpi']
#     h, w = size  # rows/height/y, columns/width/x
#     fsize = w / float(dpi), h / float(dpi)
#
#     # ##### TODO *** THIS DOES NOT GENERALIZE
#     n_rois = len(rois)
#     tuned = np.abs(tuning_mag) > tuning_thresh
#     rois_tuned = rois[tuned]
#     tuning_tuned = tuning[tuned]
#     tuning_mag_tuned = tuning_mag[tuned]
#
#     if tuning.max() > 1:
#         warn(UserWarning('provided tuning index has values > 1 (out of range)'))
#
#     assert len(rois_tuned) == len(tuning_tuned) == len(tuning_mag_tuned)
#     n_rois_tuned = len(rois_tuned)
#
#     f0 = plt.figure(figsize=fsize)
#     ax = f0.add_axes((0, 0, 1, 1))
#     plt.set_cmap('hsv')
#     # plt.axis('off')
#     ax.axis('off')
#     ax.set_frame_on(False)
#     if image is not None:
#         ilow, ihigh = np.percentile(image, (1.0, 99.98))
#         ref_f64 = util.img_as_float64(image)
#         ref_rescale = exposure.rescale_intensity(ref_f64, in_range=(ilow, ihigh))
#         ref = ref_rescale
#         canvas = np.stack((ref,) * 3, axis=-1)  # copy single channel to form RGB image
#     else:
#         canvas = np.zeros([h, w, 3], dtype=np.float64)  # create a color canvas with frame size
#
#     for r in range(n_rois_tuned):
#         roi = rois_tuned[r]
#         ry = roi['ypix']
#         rx = roi['xpix']
#         if circular is True:
#             canvas[ry, rx, :] = colorsys.hsv_to_rgb(tuning_tuned[r], 1.0, 1.0)
#         else:
#             for rgb in range(3):
#                 canvas[ry, rx, rgb] = abs(1 - 2 * abs(tuning_tuned[r] / 1.5 - rgb * 1 / 3))  # * tuning_mag[r]
#     ax.tick_params(left=False, right=False, labelleft=False,
#                    labelbottom=False, bottom=False)
#     # plt.imshow(canvas, interpolation='none', cmap='hsv')#, cmap=mpl.cm.get_cmap('hsv'))  #, quant_steps))#, alpha=1.0)
#     ax.imshow(canvas, interpolation='none', cmap='hsv')
#     ax.set(xlim=[-0.5, w - 0.5], ylim=[h - 0.5, -0.5], aspect=1)
#     f0.show()
#     if save_path != '':
#         now = datetime.now()
#         dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
#         save_name = dt + '_ROIplot_FSIzsc_thresh' + \
#                     '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
#                     '_tuned{}of{}'.format(n_rois_tuned, n_rois) + \
#                     '.png'
#         f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)
#
#     # Plot colorbar or colorwheel
#     if circular is True:
#         f1 = plt.figure()
#         plt.set_cmap('hsv')
#         # see https://stackoverflow.com/questions/62531754/how-to-draw-a-hsv-color-wheel-using-matplotlib
#         # for color wheel options including saturation
#         ax0 = f1.add_axes((0, 0, 1, 1), polar=True, frameon=False)
#         ax0.set_axis_on()
#         ax0.set_rticks([])
#         ax0.set_xticks([0, np.pi / 2])
#         ax0.set_xticklabels(['0', '90'])
#         ax0.grid(False)
#         ax1 = f1.add_axes(ax0.get_position(), projection='polar')
#         ax1._direction = 2 * np.pi  # This is a nasty hack - using the hidden field to
#         #                           # multiply the values such that 1 become 2*pi
#         #                           # this field is supposed to take values 1 or -1 only!!
#         # Plot the colorbar onto the polar axis
#         # note - use orientation horizontal so that the gradient goes around
#         # the wheel rather than centre out
#         norm = mpl.colors.Normalize(0.0, (2 * np.pi))
#         cb = mpl.colorbar.ColorbarBase(ax1,
#                                        cmap=mpl.cm.get_cmap('hsv'),
#                                        norm=norm,
#                                        orientation='horizontal')
#         # aesthetics - get rid of border and axis labels
#         cb.outline.set_visible(False)
#         ax1.set_axis_off()
#         ax1.set_rlim([-1, 1])
#         f1.show()
#         if save_path != '':
#             now = datetime.now()
#             dt = now.strftime('%Y%m%d') + 'd' + now.strftime('%H%M%S') + 't'
#             save_name = dt + '_ROIplot_FSIzsc_thresh' + \
#                         '{:.2f}'.format(tuning_thresh).replace('.', 'p') + \
#                         '_tuned{}of{}'.format(n_rois_tuned, n_rois) + \
#                         '_legend.png'
#             f0.savefig(os.path.join(save_path, save_name), dpi=dpi, transparent=True)
#
#     # Santi original
#     # # create colormap for reference
#     # f3 = plt.figure()
#     # x = np.linspace(0, 1, len(np.unique(tuning)) + 1)
#     # y = np.linspace(1, 0, 101)
#     # xx, yy = np.meshgrid(x, y)
#     # canvas_colormap = np.ones([101, len(np.unique(tuning)) + 1, 3])
#     # for rgb in range(3):
#     #     if circular:
#     #         canvas_colormap[:,:,rgb] = abs(1 - 2 * abs(xx - rgb * 1/3)) * yy
#     #     else:
#     #         canvas_colormap[:,:,rgb] = abs(1 - 2 * abs(xx/1.5 - rgb * 1/3)) * yy
#     # plt.imshow(canvas_colormap, extent=[0,1,0,1], interpolation='none')
#     # plt.xlabel('tuning')
#     # plt.ylabel('tuning mag')
#     # f2.show()
#
#     if n_neighbors is not None:
#         import seaborn as sns
#         from sklearn import neighbors
#         from sklearn.inspection import DecisionBoundaryDisplay
#
#         # we only take the first two features. We could avoid this ugly
#         # slicing by using a two-dim dataset
#         X = np.empty((n_ROIs_tuned, 2))
#         y = tuning_tuned * 8
#         y = y.astype(int) + 1
#
#         # tuned_logic = index_strength > strength_thresh
#         # X = X[tuned_logic]
#         # y = y[tuned_logic]
#         for i in range(len(X)):
#             X[i] = np.mean(rois_tuned[i]['xpix'][0]), h - np.mean(rois_tuned[i]['ypix'][0])
#
#         # Create color maps
#         for weights in ['uniform', 'distance']:
#             # Create an instance of Neighbours Classifier and fit the data.
#             clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#             clf.fit(X, y)
#             _, ax = plt.subplots()
#             DecisionBoundaryDisplay.from_estimator(
#                 clf,
#                 X,
#                 cmap='Spectral',
#                 ax=ax,
#                 response_method='predict',
#                 plot_method='pcolormesh',
#                 shading='auto'
#             )
#             # Plot also the training points
#             sns.scatterplot(
#                 x=X[:, 0],
#                 y=X[:, 1],
#                 hue=y,
#                 palette='Spectral',
#                 alpha=1.0,
#                 edgecolor='black',
#                 # legend='full',
#                 s=10
#             )
#             plt.tick_params(left=False, right=False, labelleft=False,
#                             labelbottom=False, bottom=False)
#             plt.axis('square')
#             # plt.title('(k = {}, weights = {})'.format(n_neighbors, weights))
#         plt.show()
