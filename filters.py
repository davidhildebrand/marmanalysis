#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn


# The mpfi_* functions were ported from 2018 MPFI Neuroimaging Workshop code written in MATLAB by
# David Whitney <david.whitney@mpfi.org>:
#   https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop

def mpfi_prctile_alternative(x, k):
    """
    Calculate the k'th percentile of one-dimensional signal x.
    Described by the author as being "similar to, but generally much faster than", the MATLAB prctile function.

    This function was ported from a MATLAB function named 'percentile' included with function 'RankOrderFilter':
        Author: Arash Salarian <arash.salarian@ieee.org>
        Copyright: 2008 Arash Salarian
        URL: https://www.mathworks.com/matlabcentral/fileexchange/22111-rank-order-filter
    which was included in the 2018 MPFI Neuroimaging Workshop code by David Whitney <david.whitney@mpfi.org>:
         https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
    """
    x = np.sort(x)
    n = len(x)
    p = 1 + (n - 1) * k / 100

    if p == int(p):
        y = x[int(p) - 1]
    else:
        r1 = np.floor(p)
        r2 = r1 + 1
        y = x[int(r1) - 1] + (x[int(r2) - 1] - x[int(r1) - 1]) * k / 100

    return y


def mpfi_rank_order_filter(x, p, n):
    """
    Rank-order filter for one-dimensional signals.

    % y = rankOrderFilter(x, p, N) runs a rank-order filtering of order
    % N on x. y is the same size as x. To avoid edge effects, the x is expanded
    % by repeating the first and the last samples N/2 times. If x is a matrix,
    % RankOrderFilter operates along the columns of x.
    %
    % Rank-order filter calculates the p'th percentile of the data on an N-sized
    % window round each point of x. p can be a number between 0 and 100.
    %
    % When p is equal to 50, the output of this function will be the same as
    % MATLAB's PRCTILEFILT1(x,N); however, RankOrderFilter is almost always much
    % faster and needs less memory.
    %
    % When p is close to 0 (or to 100), a RankOrderFilter calculates an
    % approximate lower (or upper) envlope of the signal.

    This function was ported from a MATLAB function named 'RankOrderFilter':
        Author: Arash Salarian <arash.salarian@ieee.org>
        Copyright: 2008 Arash Salarian
        URL: https://www.mathworks.com/matlabcentral/fileexchange/22111-rank-order-filter
    which was included in the 2018 MPFI Neuroimaging Workshop code by David Whitney <david.whitney@mpfi.org>:
         https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
    """

    # if x.ndim == 1:
    #     x = np.expand_dims(x, axis=-1)
    # m, n = x.shape

    m = len(x)
    y = np.zeros(x.shape)
    k = n // 2

    # for i in range(n):
    # X = np.concatenate([np.full(k, x[0,0]), x[:,0], np.full(k, x[-1,0])])
    x_pad = np.concatenate([np.full(k, x[0]), x[:], np.full(k, x[-1])])
    for j in range(m):
        # y[j] = mpfi_prctile_alternative(x_pad[j:j+n], p)
        y[j] = np.percentile(x_pad[j:j+n], p, axis=0, method='midpoint')

    # return np.squeeze(y)
    return y


def mpfi_percentile_filter_1d(x, p, n=3, block_size=1000):
    """
    Percentile filter for one-dimensional signals.

    % Y = percentileFilt1(X, percentile, N) returns the output of the order N, one dimensional
    % percentile filtering of X.  Y is the same size as X; for the edge points,
    % zeros are assumed to the left and right of X.  If X is a matrix,
    % then percentileFilt1 operates along the columns of X.
    %
    % If you do not specify N, percentileFilt1 uses a default of N = 3.
    % For N odd, Y(k) is the percentile of X( k-(N-1)/2 : k+(N-1)/2 ).
    % For N even, Y(k) is the percentile of X( k-N/2 : k+N/2-1 ).
    %
    % Y = percentileFilt1(X,N,BLKSZ) uses a for-loop to compute BLKSZ ("block size")
    % output samples at a time.  Use this option with BLKSZ << LENGTH(X) if
    % you are low on memory (percentileFilt1 uses a working matrix of size
    % N x BLKSZ).  By default, BLKSZ == LENGTH(X); this is the fastest
    % execution if you have the memory for it.
    %
    % For matrices and N-D arrays, Y = percentileFilt1(X,N,[],DIM) or
    % Y = percentileFilt1(X,N,BLKSZ,DIM) operates along the dimension DIM.
    %
    % Example:
    %   Construct a noisy signal and apply a 10th order one-dimensional
    %   median filter to it.
    %
    % fs = 100;                               % Sampling rate
    % t = 0:1/fs:1;                           % Time vector
    % x = sin(2*pi*t*3)+.25*sin(2*pi*t*40);   % Noise Signal - Input
    % y = percentileFilt1(x,50,10);           % Median filtering - Output
    % plot(t,x,'k',t,y,'r'); grid;            % Plot
    % legend('Original Signal','Filtered Signal')

    This function was ported from a MATLAB function named 'percentileFilt1':
        Authors: L. Shure and T. Krauss (1993-08-03)
        Copyright: 1988-2004 The MathWorks, Inc.
        Revision: 1.8.4.6
        Date: 2012-10-29 19:31:41
    which was included in the 2018 MPFI Neuroimaging Workshop code by David Whitney <david.whitney@mpfi.org>:
         https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
    """

    nx = len(x)
    m = n // 2 if n % 2 == 0 else (n - 1) // 2
    # x_pad = np.concatenate([np.zeros(m), x, np.zeros(m)])
    x_pad = np.concatenate([x[m::-1], x, x[m::-1]])
    y = np.zeros(nx)

    # Work in chunks to save memory
    indr = np.arange(n)
    indc = np.arange(nx)

    for i in range(0, nx, block_size):
        end = min(i + block_size, nx)
        ind = np.add.outer(indc[i:end], indr)
        xx = x_pad[ind].transpose().reshape((n, end - i), order='F')
        y[i:end] = np.percentile(xx, p, axis=0, method='midpoint')

    return y


def mpfi_butterworth_filter(x, fs, cutoff, n=1):
    """
    Applies the low-pass butterworth filter from the SciPy package to a one-dimensional signal.
    Returned filtered signal is the same size as input signal, which is padded before filtering to reduce edge effects.

    x: signal
    fs: sampling frequency
    cutoff: cutoff period (sec)
    n: order of the butterworth filter

    This function is ported from a portion of a MATLAB function named 'baselinePercentileFilter':
        Authors: David Whitney (david.whitney@mpfi.org)
        URL: https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
    """

    from scipy.signal import butter, filtfilt

    pad = np.ceil(len(x) / 1).astype(int)
    x_pad = np.concatenate([x[pad::-1], x, x[:pad]])
    wn = (1 / cutoff) / (fs / 2)  # Cutoff frequency normalized to NyQuist frequency units
    b, a = butter(n, wn, btype='low')
    x_bw = filtfilt(b, a, x_pad)
    x_bw = x_bw[pad:pad+len(x)]

    return x_bw


def mpfi_baseline_filter(x, fs, p_rank=10, filtered_cutoff=120):
    """
    Uses a combination of one-dimensional median/butterworth high-pass filtered to compute a baseline
    for the input trace.

    x: signal
    fs: sampling frequency
    p_rank: percentile rank for the filter
    filtered_cutoff: cutoff for the filter

    This function is ported from a MATLAB function named 'baselinePercentileFilter':
        Authors: David Whitney (david.whitney@mpfi.org)
        URL: https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
    """

    from scipy.signal import butter, filtfilt

    # Compute a low-pass median filter
    pad = np.ceil(len(x) / 1).astype(int)
    x_pad = np.concatenate([x[pad::-1], x, x[pad::-1]])
    # x_lowpass = mpfi_percentile_filter_1d(x_pad, p_rank, round(filtered_cutoff * fs))
    x_lowpass = mpfi_rank_order_filter(x_pad, p_rank, round(filtered_cutoff * fs))
    x_lowpass = x_lowpass[pad:pad+len(x)]

    # Butterworth filter to smooth
    n_butter = 1
    wn = (1 / filtered_cutoff) / (fs / 2)  # Cutoff frequency normalized to NyQuist frequency units
    b, a = butter(n_butter, wn, btype='low')
    x_bw = filtfilt(b, a, np.concatenate([x_lowpass[pad::-1], x_lowpass, x_lowpass[:pad]]))
    x_bw = x_bw[pad:pad+len(x)]

    return x_bw


def calculate_baselines(f_rois, framerate=6.364, window=60, method='medianbw', **extras):
    """
    Calculate the baselines for all ROIs.

    f_rois: fluorescence trace signals for a set of ROIs, shape: (n_ROIs, n_frames)
    framerate: frame sampling frequency (Hz)
    window: period of filtering (sec)
    method: type of filter to use
    """
    if np.ndim(f_rois) == 1:
        f_rois = np.expand_dims(f_rois, axis=0)

    n_rois, n_frames = f_rois.shape
    rate = framerate  # Hz
    win = window  # sec
    win_frames = round(win * rate)  # frames

    # Butterworth filter critical frequency (normalized to NyQuist frequency units)
    wn = (1 / win) / (rate / 2)

    match method:
        case 'mean':
            from scipy.ndimage import convolve

            f0 = convolve(f_rois,
                          weights=[np.zeros(win_frames), np.ones(win_frames) / win_frames],
                          mode='reflect')

        case 'meanbw':
            from scipy.ndimage import convolve
            from scipy.signal import butter, filtfilt

            f0 = convolve(f_rois,
                          weights=[np.zeros(win_frames), np.ones(win_frames) / win_frames],
                          mode='reflect')
            b, a = butter(1, wn, btype='low')
            f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'median':
            from scipy.ndimage import median_filter

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = median_filter(f_rois[r], size=win_frames, mode='reflect')

        case 'medianbw':
            from scipy.ndimage import median_filter
            from scipy.signal import butter, filtfilt

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = median_filter(f_rois[r], size=win_frames, mode='reflect')
            b, a = butter(1, wn, btype='low')
            f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'pctile':
            # Similar to Wilson et al Fitzpatrick (https://doi.org/10.1038/s41586-018-0354-1) but uses built-in python
            # functions and different padding methods as a result:
            #   "ΔF/F0 was computed by defining F0 using a 60 sec percentile filter (typically 10th percentile), [...]"
            from scipy.ndimage import percentile_filter

            if not extras or 'percentile' not in extras:
                warn('No percentile value specified for percentile filter, using default of percentile=10.')
                pctl = 10
            else:
                pctl = extras['percentile']

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = percentile_filter(f_rois[r], percentile=pctl, size=win_frames, mode='reflect')

        case 'pctilebw':
            # Similar to Wilson et al Fitzpatrick (https://doi.org/10.1038/s41586-018-0354-1) but uses built-in python
            # functions and different padding methods as a result:
            #   "ΔF/F0 was computed by defining F0 using a 60 sec percentile filter (typically 10th percentile), which
            #   was then low-pass filtered at 0.01 Hz."
            from scipy.ndimage import percentile_filter
            from scipy.signal import butter, filtfilt

            if not extras or 'percentile' not in extras:
                warn('No percentile value specified for percentile filter, using default of percentile=10.')
                pctl = 10
            else:
                pctl = extras['percentile']

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = percentile_filter(f_rois[r], percentile=pctl, size=win_frames, mode='reflect')
            b, a = butter(1, wn, btype='low')
            f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'rank':
            from scipy.ndimage import rank_filter

            if not extras or 'rank' not in extras:
                warn('No rank value specified for rank filter, using default of rank=10.')
                rnk = 10
            else:
                rnk = extras['rank']

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = rank_filter(f_rois[r], rank=rnk, size=win_frames, mode='reflect')

        case 'rankbw':
            from scipy.ndimage import rank_filter
            from scipy.signal import butter, filtfilt

            if not extras or 'rank' not in extras:
                warn('No rank value specified for rank filter, using default of rank=10.')
                rnk = 10
            else:
                rnk = extras['rank']

            # Filter each ROI separately to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = rank_filter(f_rois[r], rank=rnk, size=win_frames, mode='reflect')
            b, a = butter(1, wn, btype='low')
            f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'maximin':
            # Same as suite2p dcnv (https://github.com/MouseLand/suite2p/blob/c88e1b/suite2p/extraction/dcnv.py#L127):
            #   "take the running max of the running min after smoothing with gaussian"
            #   sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
            #   win_baseline = 60.0 # in seconds, window in which to compute max/min filters
            from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
            from scipy.signal import butter, filtfilt

            if not extras or 'sigma' not in extras:
                warn('No sigma value specified for gaussian filter in maximin method, using default of sigma=10.')
                sgma = 10.
            else:
                sgma = extras['sigma']

            f0 = gaussian_filter(f_rois, sigma=[0., sgma], mode='reflect')
            f0 = minimum_filter1d(f0, size=win_frames, mode='reflect')
            f0 = maximum_filter1d(f0, size=win_frames, mode='reflect')

        case 'maximinbw':
            # Similar to suite2p (https://github.com/MouseLand/suite2p/blob/c88e1b/suite2p/extraction/dcnv.py#L127)
            # but with a butterworth filter applied to the result:
            #   "take the running max of the running min after smoothing with gaussian"
            #   sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
            #   win_baseline = 60.0 # in seconds, window in which to compute max/min filters
            from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
            from scipy.signal import butter, filtfilt

            if not extras or 'sigma' not in extras:
                warn('No sigma value specified for gaussian filter in maximin method, using default of sigma=10.')
                sgma = 10.
            else:
                sgma = extras['sigma']

            f0 = gaussian_filter(f_rois, sigma=[0., sgma], mode='reflect')
            f0 = minimum_filter1d(f0, size=win_frames, mode='reflect')
            f0 = maximum_filter1d(f0, size=win_frames, mode='reflect')
            b, a = butter(1, wn, btype='low')
            f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'mpfi_pctile':
            # Same as first part of Wilson et al Fitzpatrick (e.g., https://doi.org/10.1038/s41586-018-0354-1):
            #   "ΔF/F0 was computed by defining F0 using a 60 sec percentile filter (typically 10th percentile) [...]"

            if not extras or 'percentile' not in extras:
                warn('No percentile value specified for percentile filter, using default of percentile=10.')
                pctl = 10
            else:
                pctl = extras['percentile']

            # Filter each ROI separately to be compatible with function and to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = mpfi_percentile_filter_1d(f_rois[r], p=pctl, n=win_frames)

        case 'mpfi_pctilebw':
            # Same as Wilson et al Fitzpatrick (e.g., https://doi.org/10.1038/s41586-018-0354-1):
            #   "ΔF/F0 was computed by defining F0 using a 60 sec percentile filter (typically 10th percentile),
            #   which was then low-pass filtered at 0.01 Hz."

            if not extras or 'percentile' not in extras:
                warn('No percentile value specified for percentile filter, using default of percentile=10.')
                pctl = 10
            else:
                pctl = extras['percentile']

            # Filter each ROI separately to be compatible with function and to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = mpfi_percentile_filter_1d(f_rois[r], p=pctl, n=win_frames)
                f0[r] = mpfi_butterworth_filter(f0[r], fs=rate, cutoff=win)
            # b, a = butter(1, wn, btype='low')
            # f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case 'mpfi_rnkord':
            # Same as Mulholland et al Smith (https://doi.org/10.1016/j.jneumeth.2023.110051) but with settings similar
            # to (https://doi.org/10.1038/s41592-023-02098-1):
            #   "Baseline fluorescence (F0) was calculated by applying a rank-order filter to the raw fluorescence
            #   trace (tenth percentile) with a rolling time window of 60 sec."

            if not extras or 'rank' not in extras:
                warn('No rank value specified for rank-order filter, using default of rank=10.')
                rnk = 10
            else:
                rnk = extras['rank']

            # Filter each ROI separately to be compatible with function and to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = mpfi_rank_order_filter(f_rois[r], p=rnk, n=win_frames)

        case 'mpfi_rnkordbw':
            # Similar to Mulholland et al Smith (https://doi.org/10.1016/j.jneumeth.2023.110051) but with settings
            # similar to (https://doi.org/10.1038/s41592-023-02098-1) and a butterworth filter applied to the result:
            #   "Baseline fluorescence (F0) was calculated by applying a rank-order filter to the raw fluorescence
            #   trace (tenth percentile) with a rolling time window of 60 sec."

            if not extras or 'rank' not in extras:
                warn('No rank value specified for rank-order filter, using default of rank=10.')
                rnk = 10
            else:
                rnk = extras['rank']

            # Filter each ROI separately to be compatible with function and to avoid memory errors
            f0 = np.full([n_rois, n_frames], np.nan)
            for r in range(n_rois):
                f0[r] = mpfi_rank_order_filter(f_rois[r], p=rnk, n=win_frames)
                f0[r] = mpfi_butterworth_filter(f0[r], fs=rate, cutoff=win)
            # b, a = butter(1, wn, btype='low')
            # f0 = filtfilt(b, a, f0, method='pad', padtype='even')

        case _:
            raise ValueError('Unknown baseline filtering method: {}'.format(method))

    if f0.shape[0] == 1:
        f0 = np.squeeze(f0)
    return f0


def plot_example_baselines(f_rois, rois=2, frames=1000, framerate=6.364, window=60, include_mpfi=False, **extras):
    """
    Plot example baseline fluorescence traces for randomly sampled ROIs and frames.
    """
    import matplotlib.pyplot as plt

    if not extras or 'percentile' not in extras:
        warn('No percentile value specified for percentile filter, using default of percentile=10.')
        pctl = 10
    else:
        pctl = extras['percentile']
    if not extras or 'sigma' not in extras:
        warn('No sigma value specified for gaussian filter in maximin method, using default of sigma=10.')
        sgma = 10.
    else:
        sgma = extras['sigma']

    n_rois, n_frames = f_rois.shape
    n_plot_rois = rois
    n_samp_inspect = frames
    rate = framerate  # Hz
    win = window  # sec

    plot_rois = np.random.choice(n_rois, n_plot_rois)
    frame_start = np.random.choice(n_frames - n_samp_inspect, 1)[0]
    frame_end = frame_start + n_samp_inspect

    fig = plt.figure()
    fig.suptitle('Baseline fluorescence traces for a randomly sampled subset of ROIs and frames')
    axes = fig.subplots(nrows=n_plot_rois, ncols=1)
    for r in range(n_plot_rois):
        ridx = plot_rois[r]
        f_sub = f_rois[ridx, frame_start:frame_end]

        f0_mean = calculate_baselines(f_sub, framerate=rate, window=win, method='mean')
        f0_meanbw = calculate_baselines(f_sub, framerate=rate, window=win, method='meanbw')
        f0_med = calculate_baselines(f_sub, framerate=rate, window=win, method='median')
        f0_medbw = calculate_baselines(f_sub, framerate=rate, window=win, method='medianbw')

        f0_pctl = calculate_baselines(f_sub, framerate=rate, window=win, method='pctile', percentile=pctl)
        f0_pctlbw = calculate_baselines(f_sub, framerate=rate, window=win, method='pctilebw', percentile=pctl)

        # f0_rnk = calculate_baselines(f_sub, framerate=rate, window=win, method='rank', rank=rnk)
        # f0_rnkbw = calculate_baselines(f_sub, framerate=rate, window=win, method='rankbw', rank=rnk)

        f0_maximin = calculate_baselines(f_sub, framerate=rate, window=win, method='maximin', sigma=sgma)
        f0_maximinbw = calculate_baselines(f_sub, framerate=rate, window=win, method='maximinbw', sigma=sgma)

        if include_mpfi:
            f0_mpct = calculate_baselines(f_sub, framerate=rate, window=win, method='mpfi_pctile', percentile=pctl)
            f0_mpctbw = calculate_baselines(f_sub, framerate=rate, window=win, method='mpfi_pctilebw', percentile=pctl)

            if not extras or 'rank' not in extras:
                warn('No rank value specified for rank-order filter, using default of rank=10.')
                rnk = 10
            else:
                rnk = extras['rank']
            f0_mrnkord = calculate_baselines(f_sub, framerate=rate, window=win, method='mpfi_rnkord', rank=rnk)
            f0_mrnkordbw = calculate_baselines(f_sub, framerate=rate, window=win, method='mpfi_rnkordbw', rank=rnk)

        ymin = np.min(f_sub)
        ymax = np.max(f_sub)

        ax = axes[r]
        ax.set_ylabel('F')
        ax.set_xlabel('Frame')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim((ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax)))
        ax.set_xticks([0, n_samp_inspect])
        ax.set_xticklabels([frame_start, frame_end])

        xs = range(n_samp_inspect)

        ax.plot(xs, f_sub, label='F', linewidth=0.5, alpha=0.5, zorder=1)

        ax.plot(xs, f0_mean, label='mean', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(xs, f0_meanbw, label='mean', linestyle='dashed', linewidth=1, alpha=0.5, zorder=3)
        ax.plot(xs, f0_med, label='median', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(xs, f0_medbw, label='medianbw', linestyle='dashed', linewidth=1, alpha=0.5, zorder=3)

        ax.plot(xs, f0_pctl, label='pctile', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(xs, f0_pctlbw, label='pctilebw', linestyle='dashed', linewidth=1, alpha=0.5, zorder=3)

        # ax.plot(xs, f0_rnk, label='rnk', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
        # ax.plot(xs, f0_rnkbw, label='rnkbw', linestyle='dashed', linewidth=1, alpha=0.5, zorder=3)

        ax.plot(xs, f0_maximin, label='maximin', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(xs, f0_maximinbw, label='maximinbw', linestyle='dashed', linewidth=1, alpha=0.5, zorder=3)

        if include_mpfi:
            ax.plot(xs, f0_mpct, label='mpfi pct', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
            ax.plot(xs, f0_mpctbw, label='mpfi pctbw', linestyle='dotted', linewidth=1, alpha=0.5, zorder=3)

            ax.plot(xs, f0_mrnkord, label='mpfi rnkord', linestyle='solid', linewidth=1, alpha=0.5, zorder=2)
            ax.plot(xs, f0_mrnkordbw, label='mpfi rnkordbw', linestyle='dotted', linewidth=1, alpha=0.5, zorder=3)

        ax.legend(fontsize=4, ncol=len(ax.get_lines()), frameon=False, loc=(.02, .85))
    plt.show()
