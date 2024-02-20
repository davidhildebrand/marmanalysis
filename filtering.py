import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# TODO test this against MATLAB version and compare results


# Functions ported from 2017 MPFI Imaging Course Workshop demo MATLAB code:
#   https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
#
# confirmed same output as MATLAB
def percentile_func(x, k):
    """
    Calculate the k'th percentile of x.
    This function is derived from a MATLAB function that is "similar to, but generally much faster than",
    the MATLAB prctile function.
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


# confirmed same output as MATLAB
def rankOrderFilter(x, p, N):
    """
    Rank order filter for 1D signals.
    %RankOrderFilter Rank order filter for 1D signals
    %  y = RankOrderFilter(x, window, thd) runs a rank-order filtering of order
    %  N on x. y is the same size as x. To avoid edge effects, the x is expanded
    %  by repeating the first and the last samples N/2 times. if x is a matrix,
    %  RankOrderFilter operates along the columns of x.
    %
    %  Rank-order filter calculates the p'th percentile of the data on a N
    %  sized window round each point of x. p can be a number between 0 and 100.
    %
    %  When p is equal to 50, the output of this function will be the same as
    %  MATLAB's PRCTILEFILT1(x,N); however, RankOrderFilter is almost always much
    %  faster and needs less memory.
    %
    %  When p is close to 0 (or to 100), a RankOrderFilter calculates an
    %  approximate lower (or upper) envlope of the signal.
    %
    %  Copyright 2008, Arash Salarian
    %  mailto://arash.salarian@ieee.org
    """

    # if x.ndim == 1:
    #     x = np.expand_dims(x, axis=-1)

    # m, n = x.shape
    m = len(x)
    y = np.zeros(x.shape)
    k = N // 2

    # for i in range(n):
    # X = np.concatenate([np.full(k, x[0,0]), x[:,0], np.full(k, x[-1,0])])
    X = np.concatenate([np.full(k, x[0]), x[:], np.full(k, x[-1])])
    for j in range(m):
        # y[j] = percentile_func(X[j:j + N], p)
        y[j] = np.percentile(X[j:j + N], p, axis=0, method='midpoint')

    return y  # np.squeeze(y)


# confirmed same output as MATLAB
def prctilefilt1d(x, percentile=50, n=3, block_size=1000):
    """
    One dimensional median filter.
    %percentileFilt1  One dimensional median filter.
    %   Y = percentileFilt1(X, percentile, N) returns the output of the order N, one dimensional
    %   percentile filtering of X.  Y is the same size as X; for the edge points,
    %   zeros are assumed to the left and right of X.  If X is a matrix,
    %   then percentileFilt1 operates along the columns of X.
    %
    %   If you do not specify N, percentileFilt1 uses a default of N = 3.
    %   For N odd, Y(k) is the percentile of X( k-(N-1)/2 : k+(N-1)/2 ).
    %   For N even, Y(k) is the percentile of X( k-N/2 : k+N/2-1 ).
    %
    %   Y = percentileFilt1(X,N,BLKSZ) uses a for-loop to compute BLKSZ ("block size")
    %   output samples at a time.  Use this option with BLKSZ << LENGTH(X) if
    %   you are low on memory (percentileFilt1 uses a working matrix of size
    %   N x BLKSZ).  By default, BLKSZ == LENGTH(X); this is the fastest
    %   execution if you have the memory for it.
    %
    %   For matrices and N-D arrays, Y = percentileFilt1(X,N,[],DIM) or
    %   Y = percentileFilt1(X,N,BLKSZ,DIM) operates along the dimension DIM.
    %
    %   % Example:
    %   %   Construct a noisy signal and apply a 10th order one-dimensional
    %   %   median filter to it.
    %
    %   fs = 100;                               % Sampling rate
    %   t = 0:1/fs:1;                           % Time vector
    %   x = sin(2*pi*t*3)+.25*sin(2*pi*t*40);   % Noise Signal - Input
    %   y = percentileFilt1(x,50,10);           % Median filtering - Output
    %   plot(t,x,'k',t,y,'r'); grid;            % Plot
    %   legend('Original Signal','Filtered Signal')
    %
    %   See also MEDIAN, FILTER, SGOLAYFILT, and MEDFILT1 in the Image
    %   Processing Toolbox.

    %   Author(s): L. Shure and T. Krauss, 8-3-93
    %   Copyright 1988-2004 The MathWorks, Inc.
    %   $Revision: 1.8.4.6 $  $Date: 2012/10/29 19:31:41 $
    """
    nx = len(x)
    m = n // 2 if n % 2 == 0 else (n - 1) // 2
    x_pad = np.concatenate([np.zeros(m), x, np.zeros(m)])
    y = np.zeros(nx)

    # Work in chunks to save memory
    indr = np.arange(n)
    indc = np.arange(nx)

    for i in range(0, nx, block_size):
        end = min(i + block_size, nx)  # min(i + blksz - 1, nx) - i + 1
        ind = np.add.outer(indc[i:end], indr)
        xx = x_pad[ind].transpose().reshape((n, end - i), order='F')
        y[i:end] = np.percentile(xx, percentile, axis=0, method='midpoint')
    return y


# confirmed same output as MATLAB
def baselinePercentileFilter(inputTrace, fps=15.1515, filteredCutOff=60, desiredPercentileRank=50):
    """
    Uses a combination of 1-d median/butterworth high-pass filtered to compute a baseline
    for the input trace.
    """
    isVerbose = False

    # Compute a low-pass median filter
    paddingLength = np.ceil(len(inputTrace) / 1).astype(int)
    paddedTrace = np.concatenate([inputTrace[paddingLength::-1], inputTrace, inputTrace[paddingLength::-1]])
    # filteredTrace = prctilefilt1d(paddedTrace, desiredPercentileRank, round(filteredCutOff * fps))
    filteredTrace = rankOrderFilter(paddedTrace, desiredPercentileRank, round(filteredCutOff * fps))
    filteredTrace = filteredTrace[paddingLength:paddingLength+len(inputTrace)]

    if isVerbose:
        plt.figure()
        plt.plot(filteredTrace)
        plt.show()

    # The low-pass filter the filtered trace to smooth it out
    butterWorthOrder = 1
    Wn = (1 / filteredCutOff) / (fps / 2)  # normalized cutoff frequency in unit of nyquist frequency
    b, a = butter(butterWorthOrder, Wn, 'low')
    highpassFilteredTrace = filtfilt(b, a, np.concatenate([filteredTrace[paddingLength::-1], filteredTrace, filteredTrace[:paddingLength]]))
    highpassFilteredTrace = highpassFilteredTrace[paddingLength:paddingLength+len(inputTrace)]

    if isVerbose:
        plt.figure()
        plt.plot(highpassFilteredTrace)
        plt.show()

    return highpassFilteredTrace