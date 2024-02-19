import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# TODO test this against MATLAB version and compare results


# Functions ported from 2017 MPFI Imaging Course Workshop demo MATLAB code:
#   https://github.com/dwhitneycmu/MPFI-Neuroimaging-Workshop
#
# def percentile(x, k):
#     """
#     Calculate the k'th percentile of x.
#     This function is derived from a MATLAB function that is "similar to, but generally much faster than",
#     the MATLAB prctile function.
#     """
#     x = np.sort(x)
#     n = len(x)
#
#     p = 1 + (n - 1) * k / 100
#
#     if p == int(p):
#         y = x[int(p) - 1]
#     else:
#         r1 = np.floor(p)
#         r2 = r1 + 1
#         y = x[int(r1) - 1] + (x[int(r2) - 1] - x[int(r1) - 1]) * k / 100
#
#     return y


# def rankOrderFilter(x, N, p):
#     """
#     Rank order filter for 1D signals.
#     """
#     transposedx = False
#     if x.ndim == 1:
#         x = np.expand_dims(x, axis=0)
#         transposedx = True
#
#     m, n = x.shape
#     y = np.zeros((m, n))
#     k = N // 2
#
#     for i in range(n):
#         X = np.concatenate([np.full(k, x[0, i]), x[:, i], np.full(k, x[-1, i])])
#         for j in range(m):
#             y[j, i] = percentile(X[j:j+N], p)
#
#     if transposedx:
#         y = np.transpose(y)
#
#     return y


def prctilefilt1d(x, n, block_size, percentile):
    """
    One dimensional median filter.

    Inputs:
    x          - vector
    n          - order of the filter
    block_size - block size
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
        y[i:end] = np.percentile(xx, percentile, axis=0)
    return y


def percentileFilt1(x, percentile=50, n=3, blksz=None):  # , DIM=None):
    """
    One dimensional median filter.
    """
    # Check if the input arguments are valid
    if DIM is not None and DIM > x.ndim:
        raise ValueError('Invalid Dimensions')

    # Reshape x into the right dimension.
    if DIM is None:
        # Work along the first non-singleton dimension
        x = np.moveaxis(x, np.argmin(x.shape), 0)
    else:
        # Put DIM in the first (row) dimension
        x = np.moveaxis(x, DIM, 0)

    # Verify that the block size is valid.
    if blksz is None:
        blksz = x.shape[0]  # number of rows of x (default)

    # Initialize y with the correct dimension
    y = np.zeros(x.shape)

    # Call prctilefilt1d (vector)
    # for i in range(np.prod(x.shape[1:]).astype(int)):
    #     y[:, i] = prctilefilt1d(x[:, i], n, blksz, percentile)
    y = prctilefilt1d(x, n, blksz, percentile)

    # # Convert y to the original shape of x
    # if DIM is None:
    #     y = np.moveaxis(y, 0, np.argmin(y.shape))
    # else:
    #     y = np.moveaxis(y, 0, DIM)
    #
    # return y



def baselinePercentileFilter(inputTrace, fps=15.1515, filteredCutOff=60, desiredPercentileRank=50):
    """
    Uses a combination of 1-d median/butterworth high-pass filtered to compute a baseline
    for the input trace.
    """
    isVerbose = False

    # Compute a low-pass median filter
    paddingLength = np.ceil(len(inputTrace) / 1).astype(int)
    paddedTrace = np.concatenate([inputTrace[paddingLength::-1], inputTrace, inputTrace[paddingLength::-1]])
    filteredTrace = percentileFilt1(paddedTrace, desiredPercentileRank, round(filteredCutOff * fps))
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