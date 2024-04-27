#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def prctile_alternative(x, k):
    """
    Calculate the k'th percentile of one-dimensional signal x.
    Described by the author as being "similar to, but generally much faster than", the MATLAB prctile function.

    This function is derived from a MATLAB function named 'percentile' included with function 'RankOrderFilter'
        Author: Arash Salarian <arash.salarian@ieee.org>
        Copyright: 2008 Arash Salarian
        URL: https://www.mathworks.com/matlabcentral/fileexchange/22111-rank-order-filter
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


def rank_order_filter(x, p, n):
    """
    Rank order filter for one-dimensional signals.

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

    This function is derived from a MATLAB function named RankOrderFilter
        Author: Arash Salarian <arash.salarian@ieee.org>
        Copyright: 2008 Arash Salarian
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
        y[j] = prctile_alternative(x_pad[j:j+n], p)
        # y[j] = np.percentile(x_pad[j:j+n], p, axis=0, method='midpoint')

    # return np.squeeze(y)
    return y


def percentile_filter_1d(x, p, n=3, block_size=1000):
    """
    Median filter for one-dimensional signals.

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

    This function is derived from a MATLAB function named percentileFilt1
        Authors: L. Shure and T. Krauss (1993-08-03)
        Copyright: 1988-2004 The MathWorks, Inc.
        Revision: 1.8.4.6
        Date: 2012-10-29 19:31:41
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


def butterworth_filter(x, fs, n=1):
    """
    Applies the low-pass butterworth filter from the SciPy package to a one-dimensional signal.
    Returned filtered signal is the same size as input signal, which is padded with the edge
    values before filtering to reduce edge effects.

    x: signal
    fs: sampling frequency
    n: order of the butterworth filter
    """

    from scipy.signal import butter, filtfilt

    pad = np.ceil(len(x) / 1).astype(int)
    x_pad = np.concatenate([x[pad::-1], x, x[pad::-1]])
    wn = (1 / filtered_cutoff) / (fs / 2)  # Normalized cutoff frequency in NyQuist frequency units
    b, a = butter(n, wn, btype='low', output='ba')
    x_bw = filtfilt(b, a, x_pad)
    x_bw = x_bw[pad:pad+len(x)]

    return x_bw


def baseline_filter(x, fs, p_rank=10, filtered_cutoff=120):
    """
    Uses a combination of one-dimensional median/butterworth high-pass filtered to compute a baseline
    for the input trace.

    x: signal
    fs: sampling frequency
    p_rank: percentile rank for the filter
    filtered_cutoff: cutoff for the filter

    This function is derived from a MATLAB function named baselinePercentileFilter
        Authors: David Whitney (david.whitney@mpfi.org)
        Copyright: 2016 Max Planck Florida Institute
        URL: https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop/
    """

    from scipy.signal import butter, filtfilt

    # Compute a low-pass median filter
    pad = np.ceil(len(x) / 1).astype(int)
    x_pad = np.concatenate([x[pad::-1], x, x[pad::-1]])
    # x_lowpass = percentile_filter_1d(x_pad, p_rank, round(filtered_cutoff * fs))
    x_lowpass = rank_order_filter(x_pad, p_rank, round(filtered_cutoff * fs))
    x_lowpass = x_lowpass[pad:pad+len(x)]

    # Butterworth filter to smooth
    n_butter = 1
    wn = (1 / filtered_cutoff) / (fs / 2)  # Normalized cutoff frequency in NyQuist frequency units
    b, a = butter(n_butter, wn, btype='low', output='ba')
    x_bw = filtfilt(b, a, np.concatenate([x_lowpass[pad::-1], x_lowpass, x_lowpass[:pad]]))
    x_bw = x_bw[pad:pad+len(x)]

    return x_bw
