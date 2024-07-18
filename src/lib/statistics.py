import torch
import scipy
import numpy as np
import scipy.stats
import pandas as pd
from copy import copy
from scipy import signal
from scipy import fftpack
from scipy.stats import iqr
from numpy import linalg as LA, newaxis
from sklearn.decomposition import PCA
from src.tools import normalize
from scipy.stats import zscore as _zscore
from src.tools import set_k_diogonal, select_upper_triangular
from scipy.signal import filtfilt, butter, hilbert
from scipy.stats import moment, mode, skew, kurtosis
from src.calculateInfo import entropy
# from numba import jit
# import vbi.utility


def max_PSD_freq(x, dt, nperseg=512):
    '''!
    calculate the peak frequency of Power Specteral Density
    \param x array-like
        1-D array of data
    \param dt float
        sampling interval of the data in seconds
    \param nperseg integer
        number of samples in each segment
    \return float
        peak frequency of the PSD
    '''

    fs = 1.0 / dt
    freq, pxx = PSD(x, fs, nperseg=nperseg)
    index = np.argmax(pxx)

    return [freq[index]]  # output need to be list or array


def PSD(x, fs, **kwargs):
    '''!
    Estimate power spectral density using Welch's method.
    \param x array-like
        1-D array of data !TODO check if works with 2-D array
    \param fs float 
        sampling frequency of the data in Hz
    \param **kwargs
        keyword arguments to pass to scipy.signal.welch()
    \retval freq array-like
        frequency vector 
    \retual pxx
        power spectral density vector

    '''
    freq, Pxx_den = signal.welch(x, fs, **kwargs)  # nperseg=1024

    return freq, Pxx_den


def extract_FCD(data, wwidth=30, maxNwindows=100, olap=0.94,
                coldata=False, mode='corr', verbose=False):
    """!
    Functional Connectivity Dynamics from a collection of time series
    \param data array-like
        2-D array of data, with time series in rows (unless coldata is True)
    \param wwidth integer
        Length of data windows in which the series will be divided, in samples
    \param maxNwindows integer
        Maximum number of windows to be used. wwidth will be increased if necessary
    \param olap float between 0 and 1
        Overlap between neighboring data windows, in fraction of window length
    \param coldata Boolean
        if True, the time series are arranged in columns and rows represent time
    \param mode 'corr' | 'psync' | 'plock' | 'tdcorr'
        Measure to calculate the Functional Connectivity (FC) between nodes.
        - 'corr' : Pearson correlation. Uses the corrcoef function of numpy.
        - 'psync' : Pair-wise phase synchrony.
        - 'plock' : Pair-wise phase locking.
        - 'tdcorr' : Time-delayed correlation, looks for the maximum value in a cross-correlation of the data series


    \retval FCDmatrix numpy array
        Correlation matrix between all the windowed FCs.
    \retval CorrVectors numpy array
        Collection of FCs, linearized. Only the lower triangle values (excluding the diagonal) are returned
    \retval shift integer
        The distance between windows that was actually used (in samples)

    @author: jmaidana
    @author: porio

    """

    if olap >= 1:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data = copy(data.T)

    all_corr_matrix = []
    lenseries = len(data[0])

    try:
        Nwindows = min(((lenseries-wwidth*olap) //
                       (wwidth*(1-olap)), maxNwindows))
        shift = int((lenseries-wwidth)//(Nwindows-1))
        if Nwindows == maxNwindows:
            wwidth = int(shift//(1-olap))

        indx_start = range(0, (lenseries-wwidth+1), shift)
        indx_stop = range(wwidth, (1+lenseries), shift)

        nnodes = len(data)

        for j1, j2 in zip(indx_start, indx_stop):
            aux_s = data[:, j1:j2]
            if mode == 'corr':
                corr_mat = np.corrcoef(aux_s)
            elif mode == 'psync':
                corr_mat = np.zeros((nnodes, nnodes))
                for ii in range(nnodes):
                    for jj in range(ii):
                        corr_mat[ii, jj] = np.mean(
                            np.abs(np.mean(np.exp(1j*aux_s[[ii, jj], :]), 0)))
            elif mode == 'plock':
                corr_mat = np.zeros((nnodes, nnodes))
                for ii in range(nnodes):
                    for jj in range(ii):
                        corr_mat[ii, jj] = np.abs(
                            np.mean(np.exp(1j*np.diff(aux_s[[ii, jj], :], axis=0))))
            elif mode == 'tdcorr':
                corr_mat = np.zeros((nnodes, nnodes))
                for ii in range(nnodes):
                    for jj in range(ii):
                        maxCorr = np.max(np.correlate(aux_s[ii, :], aux_s[jj, :], mode='full')[
                            wwidth//2:wwidth+wwidth//2])
                        corr_mat[ii, jj] = maxCorr/np.sqrt(
                            np.dot(aux_s[ii, :], aux_s[ii, :])*np.dot(aux_s[jj, :], aux_s[jj, :]))
            all_corr_matrix.append(corr_mat)

        corr_vectors = np.array([allPm[np.tril_indices(nnodes, k=-1)]
                                for allPm in all_corr_matrix])
        CV_centered = corr_vectors - np.mean(corr_vectors, -1)[:, None]

        return np.corrcoef(CV_centered), corr_vectors, shift

    except Exception as e:
        if verbose:
            print(e)
        return [np.nan]


def moments(x, axis=1):
    '''!
    calculate the moments of a time series
    \param x array-like
        2-D array of data
    \param axis integer
        axis along which the moments are calculated
    \return array-like
        array of moments
    '''

    m, n = x.shape
    nn = m if (axis == 1) else n

    stats_vec = np.zeros(5 * nn)
    funcs = [np.mean, np.median, np.std, skew, kurtosis]
    for i in range(len(funcs)):
        stats_vec[(i*nn):((i+1)*nn)] = funcs[i](x, axis=axis)

    return stats_vec


def HighLowMu(y, axis=1):
    '''!
    Caculate the high-low-mean of a time series

    \param y array-like
        2-D array of data
    \param axis integer
        axis along which the moments are calculated
    \return array-like
        array of features
    '''

    n = y.shape[0] if (axis == 1) else y.shape[1]

    result = np.zeros(n)

    for i in range(n):

        x = y[i, :]
        mu = np.mean(x)
        mhi = np.mean(x[np.where(x < mu)])
        mu = np.mean(x)
        mhi = np.mean(x[x > mu])
        mlo = np.mean(x[x < mu])
        result[i] = (mhi - mu) / (mu - mlo)

    return result


def IQR(y, axis=1):
    '''!
    Compute the interquartile range of the data along the specified axis.
    \param y array-like
        2-D array of data
    \param axis integer
        axis along which the moments are calculated
    '''
    return iqr(y, axis=axis)


def abs_moments(y):
    '''!
    Compute the absolute moments of the data.
    '''

    sum_abs = np.sum(abs(y))
    mean_abs = np.mean(abs(y))
    median_abs = np.median(abs(y))
    peak_to_peak = abs(np.max(y)-np.min(y))

    return np.hstack([sum_abs, mean_abs, median_abs, peak_to_peak])


def higher_moments(x, axis=1):
    '''!
    Compute the moments of the data.

    \param x array-like
        2-D array of data
    \param axis integer
        axis along which the moments are calculated
    \return array-like
        array of moments
    '''

    m, n = x.shape
    nn = m if (axis == 1) else n

    stats_vec = np.zeros(9*nn)
    for i in range(9):
        stats_vec[(i*nn):((i+1)*nn)] = moment(x, moment=i+2, axis=axis)

    return stats_vec


def spectral_power(x, fs):
    '''!
    Calculate features from the spectral power of the given BOLD signal.

    \param x given BOLD signal [nnodes, ntime]
    \param fs sampling frequency [Hz]
    \return spectral power

    '''

    _stats = []
    f, Pxx_den = signal.periodogram(x, fs)
    # f = f[:num_freq_points]
    # Pxx_den = Pxx_den[:, :num_freq_points]

    funcs = [np.max, np.mean, np.median, np.std, skew, kurtosis]
    for i in range(len(funcs)):
        _stats.append(funcs[i](Pxx_den, axis=1).reshape(-1))

    #! TODO
    # for i in range(x.shape[0]):
    #     peaks = power_peaks(Pxx_den)
    #     _stats.append(peaks)
    # print(peaks)
    # _stats.append(np.diag(np.dot(
    #     Pxx_den, Pxx_den.transpose())).reshape(-1))

    return np.hstack(_stats)


def envelope(x, axis=1):
    '''!
    Calculate the features from envelope of a signal using hilbert transform.

    \param x 2d_array_like 
        The signal to be analyzed.
    \param axis 
        The axis along which to calculate the envelope.

    \returns 1d_array_like
        list of features.
    '''

    m, n = x.shape
    nn = m if (axis == 1) else n
    # stats_vec = np.zeros(4*nn)
    stats_vec = np.array([])

    analytic_signal = hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    funcs = [np.mean, np.std, np.mean, np.std]

    for i in range(2):
        stats_vec = np.append(stats_vec, funcs[i](
            amplitude_envelope, axis=axis))
        # stats_vec[(i*nn):((i+1)*nn)] = funcs[i](amplitude_envelope, axis=axis)
    for i in range(2, 4):
        stats_vec = np.append(stats_vec, funcs[i](
            instantaneous_phase, axis=axis))
        # stats_vec[(i*nn):((i+1)*nn)] = funcs[i](instantaneous_phase, axis=axis)

    return stats_vec


def fc_corr(x):
    '''!
    calculate the freatures from functional connectivity (FC)

    \param x np.ndarray (2d)
        input array
    \returns np.ndarray (1d)
    '''

    def funcs(x):
        vec = np.zeros(7)
        vec[0] = np.sum(x)
        vec[1] = np.max(x)
        vec[2] = np.min(x)
        vec[3] = np.mean(x)
        vec[4] = np.std(x)
        vec[5] = skew(x)
        vec[6] = kurtosis(x)
        return vec

    FCcorr = np.corrcoef(x)
    off_diag_sum_FC = np.sum(np.abs(FCcorr)) - np.trace(np.abs(FCcorr))

    FC_TRIU = np.triu(FCcorr, k=10)
    eigen_vals_FC, _ = LA.eig(FCcorr)
    pca = PCA(n_components=3)
    PCA_FC = pca.fit_transform(FCcorr)

    Upper_FC = []
    Lower_FC = []
    for i in range(0, len(FCcorr)):
        Upper_FC.extend(FCcorr[i][i+1:])
        Lower_FC.extend(FCcorr[i][0:i])

    q = np.quantile(FCcorr, [0.05, 0.25, 0.5, 0.75, 0.95])
    # eigen_vals_FC = eigen_vals_FC.reshape(-1).tolist()

    _stats = np.array([])
    _stats = np.append(_stats, q)
    _stats = np.append(_stats, funcs(Upper_FC))
    _stats = np.append(_stats, funcs(Lower_FC))
    _stats = np.append(_stats, funcs(PCA_FC.reshape(-1)))
    _stats = np.append(_stats, funcs(FC_TRIU.reshape(-1)))
    # last element produce core dump error in the training
    # _stats = np.append(_stats, funcs(np.real(eigen_vals_FC)))
    _stats = np.append(_stats, funcs(np.real(eigen_vals_FC[:-1])))

    # keep this the last element
    _stats = np.append(_stats, [off_diag_sum_FC])

    return _stats


def fc_corr_regions(x0, regions):
    '''!
    calculate the freatures from functional connectivity (FC) on given regions

    \param x np.ndarray (2d)
        input array
    \param regions  list[int]
        index of regions to use for FC
    \return np.ndarray (1d)
        list of feature values
    '''
    assert(len(regions) >=
           2), 'regions should be at least 2, use set_feature_properties to set regions'
    assert(isinstance(regions[0], (np.int64, int, np.int32)))

    def funcs(x):
        vec = np.zeros(7)
        vec[0] = np.sum(x)
        vec[1] = np.max(x)
        vec[2] = np.min(x)
        vec[3] = np.mean(x)
        vec[4] = np.std(x)
        vec[5] = skew(x)
        vec[6] = kurtosis(x)
        return vec

    x = x0[regions, :]

    rsFC = np.corrcoef(x)
    # rsFC = rsFC * (rsFC > 0.0)  #! Added based on Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017

    off_diag_sum_FC = np.sum(np.abs(rsFC)) - np.trace(np.abs(rsFC))

    FC_TRIU = np.triu(rsFC, k=10)
    eigen_vals_FC, _ = LA.eig(rsFC)
    pca = PCA(n_components=3)
    PCA_FC = pca.fit_transform(rsFC)

    Upper_FC = []
    Lower_FC = []
    for i in range(0, len(rsFC)):
        Upper_FC.extend(rsFC[i][i+1:])
        Lower_FC.extend(rsFC[i][0:i])

    q = np.quantile(rsFC, [0.25, 0.5, 0.75])
    # eigen_vals_FC = eigen_vals_FC.reshape(-1).tolist()

    _stats = np.array([])
    _stats = np.append(_stats, q)
    _stats = np.append(_stats, funcs(Upper_FC))
    _stats = np.append(_stats, funcs(Lower_FC))
    _stats = np.append(_stats, funcs(PCA_FC.reshape(-1)))
    _stats = np.append(_stats, funcs(FC_TRIU.reshape(-1)))
    # last element produce core dump error in the training
    _stats = np.append(_stats, funcs(np.real(eigen_vals_FC[:-1])))

    # keep this the last element
    _stats = np.append(_stats, [off_diag_sum_FC])

    return _stats


def fcd_regions(regions, x0, wwidth, maxNwindows, olap, mode, k_diagonal=20, verbose=False):
    '''!
    Calculate the freatures from functional connectivity dynamics (FCD) on given regions

    \param regions  list[int]
        index of regions to use for FCD
    \param x0 np.ndarray (2d)
        input BOLD signal [nnodes, ntime]
    \param wwidth int
        window width
    \param maxNwindows int
        maximum number of windows
    \param olap int
        overlap between windows
    \param mode str
        mode of FCD
    \param k_diagonal int  
        number of subdiagonal to be excluded from FCD matrix
    \param verbose bool
        verbose mode
    \returns np.ndarray (1d)
        list of feature values
    '''
    assert(len(regions) >=
           2), 'regions should be at least 2, use set_feature_properties to set regions'
    assert(isinstance(regions[0], (np.int64, int, np.int32)))
    x = x0[regions, :]
    result = extract_FCD(x, wwidth, maxNwindows, olap,
                         mode=mode, verbose=verbose)
    if len(result) == 1:  # find nan value
        return result
    else:
        FCDcorr, Pcorr, shift = result
        stats_vec = np.array([])
        stats_vec = np.append(stats_vec, np.mean(
            select_upper_triangular(FCDcorr, k_diagonal)))
        stats_vec = np.append(stats_vec, np.var(
            select_upper_triangular(FCDcorr, k_diagonal)))
    return stats_vec


def compute_fcd_filt(ts, mat_filt, win_len=30, win_sp=1, verbose=False):
    """Compute dynamic functional connectivity with SC filtering

    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape
    # returns the indices for upper triangle
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []
    speed_stack = []

    try:

        for t0 in range(0, ts.shape[0]-win_len, win_sp):
            t1 = t0+win_len
            fc = np.corrcoef(ts[t0:t1, :].T)
            fc = fc*(fc > 0)*(mat_filt)
            fc = fc[fc_triu_ids]
            fc_stack.append(fc)
            if t0 > 0:
                corr_fcd = np.corrcoef([fc, fc_prev])[0, 1]
                speed_fcd = 1-corr_fcd
                speed_stack.append(speed_fcd)
                fc_prev = fc
            else:
                fc_prev = fc

        fcs = np.array(fc_stack)
        speed_ts = np.array(speed_stack)
        FCD = np.corrcoef(fcs)
        return FCD, fcs, speed_ts
    except Exception as e:
        if verbose:
            print(e)
        return [np.nan]


def extract_fcd_filt(bold, regions_idx, win_len, win_sp, verbose=False):
    '''
    extract fcd of given regions from bold signal

    \param bold 2d array
        bold signal [nnodes, ntime]
    \param regions_idx list of int
        indices of regions
    \param win_len int
        sliding window length in samples
    \param win_sp int
        sliding window step in samples
    \param k int
        number of diagonal to set to zero
    \param verbose bool
        verbose flag
    '''
    nn = bold.shape[0]
    maskregions = np.zeros((nn, nn))
    maskregions[np.ix_(regions_idx, regions_idx)] = 1  # making a mask

    result = compute_fcd_filt(
        bold.T, maskregions, win_len=win_len, win_sp=win_sp, verbose=verbose)

    # check if return nan
    if len(result) == 1:
        if verbose:
            print(np.isnan(result).any())
        return result
    else:
        return result[0]


def fcd_filt(bold, regions_idx, win_len=20, win_sp=1, k=1, verbose=False):

    fcd_ = extract_fcd_filt(bold, regions_idx, win_len=win_len, win_sp=win_sp, verbose=verbose)
    if len(fcd_) != 1:
        return([np.mean(np.triu(fcd_, k=1)), np.var(np.triu(fcd_, k=k))])
    else:
        return fcd_


def fcd_corr_regions(regions, x0, wwidth, maxNwindows, olap, mode, k_diagonal=20, verbose=False):
    '''!
    Calculate the freatures from functional connectivity dynamics (FCD) on given regions

    \param regions  list[int]
        index of regions to use for FCD
    \param x0 np.ndarray (2d)
        input BOLD signal [nnodes, ntime]
    \param wwidth int
        window width
    \param maxNwindows int
        maximum number of windows
    \param olap int
        overlap between windows
    \param mode str
        mode of FCD
    \param k_diagonal int  
        number of subdiagonal to be excluded from FCD matrix
    \param verbose bool
        verbose mode
    \returns np.ndarray (1d)
        list of feature values
    '''
    assert(len(regions) >=
           2), 'regions should be at least 2, use set_feature_properties to set regions'
    assert(isinstance(regions[0], (np.int64, int, np.int32)))

    def funcs(x):
        vec = np.zeros(7)
        vec[0] = np.sum(x)
        vec[1] = np.max(x)
        vec[2] = np.min(x)
        vec[3] = np.mean(x)
        vec[4] = np.std(x)
        vec[5] = skew(x)
        vec[6] = kurtosis(x)
        return vec

    x = x0[regions, :]
    result = extract_FCD(x, wwidth, maxNwindows, olap,
                         mode=mode, verbose=verbose)
    if len(result) == 1:  # find nan value
        return result
    else:
        FCDcorr, Pcorr, shift = result
        off_diag_sum_FCD = np.sum(np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
        # !TODO check this line, probably wrong
        off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k_diagonal, 0.0))
        FCD_TRIU = np.triu(FCDcorr, k=k_diagonal)  # ! bug fix (was k=10)
        eigen_vals_FCD, _ = LA.eig(FCDcorr)
        pca = PCA(n_components=3)
        PCA_FCD = pca.fit_transform(FCDcorr)

        Upper_FCD = []
        Lower_FCD = []
        for i in range(0, len(FCDcorr)):
            Upper_FCD.extend(FCDcorr[i][i+1:])
            Lower_FCD.extend(FCDcorr[i][0:i])

        stats_vec = np.zeros(35)
        funcs = [np.sum, np.max, np.min, np.mean, np.std, skew, kurtosis]*5
        data = [eigen_vals_FCD.reshape(-1),
                PCA_FCD.reshape(-1),
                Upper_FCD,
                Lower_FCD,
                FCD_TRIU.reshape(-1)]
        n0 = 7
        for k in range(len(data)):
            for i in range(k*n0, (k+1)*n0):
                stats_vec[i] = funcs[i](data[k])

        stats_vec = np.append(stats_vec, np.quantile(
            FCDcorr, [0.25, 0.5, 0.75]))
        stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

        return stats_vec


def fcd_corr(FCD, k_diagonal=20):
    '''!
    Calculate the freatures from functional connectivity dynamics (FCD)

    \param FCD np.ndarray (2d)
        input FCD matrix
    \param k_diagonal int
        number of subdiagonal to be excluded from FCD matrix
    \returns np.ndarray (1d)
        list of feature values
    '''

    if len(FCD == 1):  # find nan value
        return FCD
    else:
        FCDcorr, Pcorr, shift = FCD
        off_diag_sum_FCD = np.sum(np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
        off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k_diagonal, 0.0))

        FCD_TRIU = np.triu(FCDcorr, k=k_diagonal)  # ! bug fix (was k=10)

        eigen_vals_FCD, _ = LA.eig(FCDcorr)
        pca = PCA(n_components=3)
        PCA_FCD = pca.fit_transform(FCDcorr)

        Upper_FCD = []
        Lower_FCD = []
        for i in range(0, len(FCDcorr)):
            Upper_FCD.extend(FCDcorr[i][i+1:])
            Lower_FCD.extend(FCDcorr[i][0:i])

        stats_vec = np.zeros(35)
        funcs = [np.sum, np.max, np.min, np.mean, np.std, skew, kurtosis]*5
        data = [eigen_vals_FCD.reshape(-1),
                PCA_FCD.reshape(-1),
                Upper_FCD,
                Lower_FCD,
                FCD_TRIU.reshape(-1)]
        n0 = 7
        for k in range(len(data)):
            for i in range(k*n0, (k+1)*n0):
                stats_vec[i] = funcs[i](data[k])

        stats_vec = np.append(stats_vec, np.quantile(
            FCDcorr, [0.05, 0.25, 0.5, 0.75, 0.95]))
        stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

        return stats_vec


def fc_sum(x):
    '''!
    Calculate the sum of functional connectivity (FC)

    \param x np.ndarray (2d)
        input BOLD signal [nnodes, ntime]
    \returns np.ndarray (1d)
        sum of functional connectivity
    '''
    FCcorr = np.corrcoef(x)
    off_diag_sum_FC = np.sum(np.abs(FCcorr)) - np.trace(np.abs(FCcorr))

    return off_diag_sum_FC


def fc_elements(x):
    '''!
    Calculate the  functional connectivity (FC)

    \param x np.ndarray (2d)
        input BOLD signal [nnodes, ntime]
    \returns np.ndarray (1d)
        functional connectivity elements 
    '''
    return np.corrcoef(x).reshape(-1)


def fluidity(FCD, k_diagonal=20):
    '''!
    Calculate the fluidity of the BOLD signal

    \param FCD np.ndarray (2d)
        input FCD matrix
    \param k_diagonal int
        number of subdiagonal to be excluded from FCD matrix
    \returns np.ndarray (1d)
        list of feature values
    '''

    if len(FCD) == 1:  # find nan value
        return FCD
    else:
        FCDcorr, _, _ = FCD
        # FCDcorr = select_upper_triangular(FCDcorr, k_diagonal) #! bug fix
        FCDcorr = np.triu(FCDcorr, k=0)
        return np.var(FCDcorr.reshape(-1))


def fcd_sum(FCD, k_diagonal=20):
    '''!
    Calculate the sum of functional connectivity dynamics (FCD)

    \param FCD np.ndarray (2d)
        input FCD matrix
    \param k_diagonal int   
        number of subdiagonal to be excluded from FCD matrix
    \returns np.ndarray (1d)
        sum of functional connectivity dynamics
    '''

    if len(FCD) == 1:  # find nan value
        return FCD
    else:
        FCDcorr, Pcorr, shift = FCD
        off_diag_sum_FCD = np.sum(np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
        off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k_diagonal, 0.0))

        return off_diag_sum_FCD


def fcd_elements(FCD, k_diagonal=20):
    '''!
    Calulate the functional connectivity dynamics (FCD)

    \param FCD np.ndarray (2d)    
        input FCD matrix
    \param k_diagonal int
        number of subdiagonal to be excluded from FCD matrix
    \returns np.ndarray (1d)
        functional connectivity dynamics elements

    '''

    if len(FCD) == 1:  # find nan value
        return FCD
    else:
        FCDcorr, Pcorr, shift = FCD
        FCDcorr = set_k_diogonal(FCDcorr, k_diagonal, 0.0)
        return FCDcorr.reshape(-1)


def under_area(y):
    '''!
    Calculate the under-area of the BOLD signal power spectrum
    '''

    # m, n = y.shape
    # nn = m if (axis == 1) else n
    y_area = np.trapz(y, dx=0.1, axis=1).reshape(-1)
    y_pwr = np.sum((y*y), axis=1).reshape(-1)
    y_pwr_n = (y_pwr / y_pwr.max()).reshape(-1)
    y_pwr_energy = y_pwr

    return np.hstack([y_area, y_pwr_n, y_pwr_energy])


def burstiness(y):
    '''!
    calculate the burstiness statistic from
    [from hctsa-py]

    \param y the input time series  
    \returns np.ndarray (1d)
        The burstiness statistic, B.

    - Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
    81, 48002 (2008).
    '''

    if y.mean() == 0:
        return np.nan

    r = np.std(y, axis=1) / np.mean(y, axis=1)
    B = (r - 1) / (r + 1)

    return B


def CustomSkewness(y, whatSkew='pearson', axis=1):
    '''!
    Calculate the skewness of the given 2d signal (BOLD)
    from hctsa-py

    \param y np.ndarray (2d)
        input BOLD signal [nnodes, ntime]

    \return np.ndarray (1d)
        skewness of the signal
    '''

    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y, axis=axis) - np.median(y, axis=axis)) / np.std(y, axis=axis)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y, [.25, .5, .75], axis=1)
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
        raise Exception('whatSkew must be either pearson or bowley.')


def fc_homotopic(bold):
    '''!
    Calculate the homotopic connectivity vector of a given brain activity

    \param bold : array_like [n_nodes, n_samples]
        The brain activity to be analyzed.
    \returns : array_like [n_nodes]
        The homotopic correlation vector.

    Negative correlations may be artificially induced when using global signal regression 
    in functional imaging pre-processing (Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017). 
    Therefore, results on negative weights should be interpreted with caution and should be understood 
    as complementary information underpinning the findings based on positive connections
    '''

    NHALF = int(bold.shape[0]//2)
    rsFC = np.corrcoef(bold)
    rsFC = rsFC * (rsFC > 0)
    rsFC = rsFC - np.diag(np.diag(rsFC))
    Homotopic_FC = np.diag(rsFC, k=NHALF)
    return Homotopic_FC


def available_features():
    '''!
    List the available features.

    \returns list str
        The list of available features.
    '''
    return [
        'moments',
        'higher_moments',
        'spectral_power',
        'envelope',
        'fcd_corr',
        'fcd_elements',
        'fcd_corr_regions',
        'fcd_regions',
        'fcd_filt',
        'fcd_sum',
        'fc_corr_regions',
        'fc_corr',
        'fc_elements',
        'fc_sum',
        'fc_homotopic',
        'fluidity',
        'fcd_sum',
        'fcd_edge',
        'raw_ts',
        'under_area',
        'abs_moments',
        'HighLowMu',
        'burstiness',
        'CustomSkewness',
        'IQR',
        'entropy',
        'fluidity'
    ]


def calculate_summary_statistics(obs,
                                 dt=0.1,
                                 features=["moments", "higher_moments",
                                           "spectral_power", "envelope",
                                           "fc_corr", "fcd_corr"],
                                 axis=1,
                                 wwidth=30,
                                 maxNwindows=200,
                                 olap=0.94,
                                 fcd_cor_mode="corr",
                                 fcd_k_diagonal=20,
                                 num_freq_points=200,
                                 whatSkew='pearson',
                                 export_zs_idx=False,  # index of cols to apply zscore
                                 export_info=False,     # columns of each features
                                 region_indices=[],
                                 verbose=False
                                 ):
    """! 
    Calculate summary statistics

    \param obs [array_like] [n_nodes, n_samples]
        The given activity signal to be analyzed.
    \param dt [float]
        The sampling interval of the activity signal.
    \param features [list of str]
        The features to be calculated.
    \param axis [int]
        The axis to be used for the calculation.
    \param wwidth [int]
        The window width to be used for the FCD calculation.
    \param maxNwindows [int]
        The maximum number of windows to be used for the FCD calculation.
    \param olap [float]
        The overlap to be used for the FCD calculation.
    \param fcd_cor_mode [str]
        The mode to be used for the FCD calculation.
    \param fcd_k_diagonal [int]
        The number of diagonals to be excluded for the FCD calculation.
    \param num_freq_points [int]
        The number of frequency points to be used spectral_power().
    \param whatSkew [str]
        The type of skewness to be used for CustomSkewness().
    \param export_zs_idx [bool]
        Whether to export the index of the columns to be zscored. ! deprecated
    \param export_info [bool]
        Whether to export the information of features to be usef for sclicing.
    \param region_indices [list of int]
        The indices of the regions to be used for the fc_corr_regions().
    \param verbose [bool]
        Whether to print the progress.
    \return [torch.Tensor]
        The summary statistics of the given activity signal.
        ! Will be changed to list in later versions.

    """

    assert(wwidth > 0)
    assert(maxNwindows > 0)
    assert(olap > 0)
    fs = 1.0 / dt

    # col_indices = np.array([])
    stats_vec = np.array([])
    zsl = []  # index of cols to apply zscore #!todo: remove
    features_info = {}

    def get_length(x):
        return (len(x)) if (len(x) > 0) else 0

    need_fcd = ["fcd_corr", "fcd_elements", "fcd_sum", "fluidity"]

    if any([x in features for x in need_fcd]):

        res = extract_FCD(obs,
                          wwidth=wwidth,
                          maxNwindows=maxNwindows,
                          olap=olap,
                          mode=fcd_cor_mode,
                          verbose=verbose)
        if len(res) > 1:
            FCD = res[0]
        else:
            print("FCD calculation failed. Returning empty array.")
            return np.array([])

    if "moments" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, moments(obs, axis=axis))
        features_info["moments"] = [ci, get_length(stats_vec)]

    if "higher_moments" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(
            stats_vec, higher_moments(obs, axis=axis))
        features_info["higher_moments"] = [ci, get_length(stats_vec)]

    if "spectral_power" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(
            stats_vec, spectral_power(obs, fs))
        features_info["spectral_power"] = [ci, get_length(stats_vec)]

    if "envelope" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, envelope(obs))
        features_info["envelope"] = [ci, get_length(stats_vec)]

    if "fc_corr" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fc_corr(obs))
        features_info["fc_corr"] = [ci, get_length(stats_vec)]

    if "fc_corr_regions" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fc_corr_regions(obs, region_indices))
        features_info["fc_corr_regions"] = [ci, get_length(stats_vec)]

    if "fcd_corr" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_corr(
            FCD, k_diagonal=fcd_k_diagonal))
        features_info["fcd_corr"] = [ci, get_length(stats_vec)]

    if "fcd_edge" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_edge(
            obs, k_diagonal=fcd_k_diagonal))
        features_info["fcd_edge"] = [ci, get_length(stats_vec)]

    if "fc_sum" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fc_sum(obs))
        features_info["fc_sum"] = [ci, get_length(stats_vec)]

    if "fcd_sum" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_sum(
            FCD, k_diagonal=fcd_k_diagonal))
        features_info["fcd_sum"] = [ci, get_length(stats_vec)]

    if "raw_ts" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, obs.flatten())
        features_info["raw_ts"] = [ci, get_length(stats_vec)]

    if "under_area" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, under_area(obs))
        features_info["under_area"] = [ci, get_length(stats_vec)]

    if "abs_moments" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, abs_moments(obs))
        features_info["abs_moments"] = [ci, get_length(stats_vec)]

    if "burstiness" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, burstiness(obs))
        features_info["burstiness"] = [ci, get_length(stats_vec)]

    if "CustomSkewness" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, CustomSkewness(obs,
                                                        whatSkew=whatSkew))
        features_info["CustomSkewness"] = [ci, get_length(stats_vec)]

    if "HighLowMu" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, HighLowMu(obs, axis=axis))
        features_info["HighLowMu"] = [ci, get_length(stats_vec)]

    if "IQR" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, IQR(obs, axis=axis))
        features_info["IQR"] = [ci, get_length(stats_vec)]

    if "entropy" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, entropy(obs))
        features_info["entropy"] = [ci, get_length(stats_vec)]

    if "fluidity" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fluidity(
            FCD, k_diagonal=fcd_k_diagonal))
        features_info["fluidity"] = [ci, get_length(stats_vec)]

    if 'fcd_elements' in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_elements(
            FCD, k_diagonal=fcd_k_diagonal))
        features_info['fcd_elements'] = [ci, get_length(stats_vec)]

    if 'fcd_corr_regions' in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_corr_regions(region_indices, obs, wwidth=wwidth,
                                                          maxNwindows=maxNwindows,
                                                          olap=olap,
                                                          mode=fcd_cor_mode,
                                                          k_diagonal=fcd_k_diagonal,
                                                          verbose=verbose))
        features_info['fcd_corr_regions'] = [ci, get_length(stats_vec)]

    if 'fcd_regions' in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_regions(region_indices, obs, wwidth=wwidth,
                                                     maxNwindows=maxNwindows,
                                                     olap=olap,
                                                     mode=fcd_cor_mode,
                                                     k_diagonal=fcd_k_diagonal,
                                                     verbose=verbose))
        features_info['fcd_regions'] = [ci, get_length(stats_vec)]

    if "fcd_filt" in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fcd_filt(obs,
                                                  regions_idx=region_indices,
                                                  win_sp=1,
                                                  win_len=wwidth,
                                                  verbose=verbose,
                                                  k=fcd_k_diagonal))
        features_info["fcd_filt"] = [ci, get_length(stats_vec)]

    if 'fc_elements' in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fc_elements(obs))
        features_info['fc_elements'] = [ci, get_length(stats_vec)]

    if 'fc_homotopic' in features:
        ci = get_length(stats_vec)
        stats_vec = np.append(stats_vec, fc_homotopic(obs))
        features_info['fc_homotopic'] = [ci, get_length(stats_vec)]

    if export_zs_idx or export_info:
        return torch.tensor(stats_vec).float(), zsl, features_info

    return torch.tensor(stats_vec).float()


def fft_1d_real(signal, fs):
    """!
    fft from 1 dimensional real signal

    @param signal: [np.array] real signal
    @param fs: [float] frequency sampling in Hz
    @return [np.array, np.array] frequency, normalized amplitude

    -  example:

    ```python
    B = 30.0  # max freqeuency to be measured.
    fs = 2 * B
    delta_f = 0.01
    N = int(fs / delta_f)
    T = N / fs
    t = np.linspace(0, T, N)
    nu0, nu1 = 1.5, 22.1
    amp0, amp1, ampNoise = 3.0, 1.0, 1.0
    signal = amp0 * np.sin(2 * np.pi * t * nu0) + amp1 * np.sin(2 * np.pi * t * nu1) +
            ampNoise * np.random.randn(*np.shape(t))
    freq, amp = fft_1d_real(signal, fs)
    pl.plot(freq, amp, lw=2)
    pl.show()
    ```

    """

    N = len(signal)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(N, 1.0 / fs)
    mask = np.where(f >= 0)

    freq = f[mask]
    amplitude = 2.0 * np.abs(F[mask] / N)

    return freq, amplitude


def compute_cofluctuation(ts):
    """!
    Compute co-fluctuation (functional connectivity edge) time series for
    each pair of nodes by element-wise multiplication of z-scored node time
    series.


    @param ts : array_like
        Time series of shape (time, nodes).ts

    \return array_like
        Co-fluctuation (edge time series) of shape (time, node_pairs).
    """
    T, N = ts.shape
    cf = np.zeros((T, int(N * (N-1)/2)))

    for k, (i, j) in enumerate(zip(*np.triu_indices(ts.shape[1], 1))):
        cf[:, k] = _zscore(ts[:, i]) * _zscore(ts[:, j])

    return cf


def cofluctuation_rss(cf, percentile=95.):
    """!
    Compute root sum square (RSS) from cofluctuation time series, and the
    value of the percentile.


    @param cf  array_like
        Cofluctuation time series of shape (time, node_pairs)
    @param percentile  float
        Percentile to compute.


    \retval RSS : array_like
        RSS time series of shape (time).
    \retval threshold : float
        Value of the given percentile.
    """
    RSS = np.sqrt(np.sum(cf * cf, axis=1))
    threshold = np.percentile(RSS, percentile)

    return RSS, threshold


# def compute_edge_fcd(cf):
#     """Compute the FCD from the co-fluctuation time series.

#     Parameters
#     ----------
#     cf : array_like
#         Cofluctuation time series of shape (time, node_pairs)

#     Returns
#     -------
#         eFCD: array_like
#             matrix of edge functional connectivity dynamics of shape (time, time)
#     """
#     return np.corrcoef(cf)

def compute_fcd_edge(bold):
    """!
    Compute the FCD from the BOLD time series.

    \param bold : array_like
        BOLD time series of shape (nnodes, ntime)

    \return array_like
            matrix of edge functional connectivity dynamics of shape (time, time)
    """
    try:
        cf = compute_cofluctuation(bold.T)  # (time, node_pairs)
        return np.corrcoef(cf)              # (time, time)
    except:
        return [np.nan]


def fcd_edge(bold, k_diagonal=20):
    """!
    Compute the FCD edge from the BOLD time series.
    it is a wrapper of compute_fcd_edge()

    \param bold : array_like
        BOLD time series of shape (nnodes, ntime)
    \param k_diagonal : int
        Number of subdiganal to be excluded from the FCD matrix.
    \return array_like
        list of statistics of FCD edge matrix.
    """

    result = compute_fcd_edge(bold)

    if len(result) == 1:  # find nan value
        return result
    else:
        FCDe = result
        off_diag_sum_FCD = np.sum(np.abs(FCDe)) - np.trace(np.abs(FCDe))
        off_diag_sum_FCD = np.sum(set_k_diogonal(FCDe, k_diagonal, 0.0))

        FCD_TRIU = np.triu(FCDe, k=k_diagonal)

        eigen_vals_FCD, _ = LA.eig(FCDe)
        pca = PCA(n_components=3)
        PCA_FCD = pca.fit_transform(FCDe)

        Upper_FCD = []
        Lower_FCD = []
        for i in range(0, len(FCDe)):
            Upper_FCD.extend(FCDe[i][i+1:])
            Lower_FCD.extend(FCDe[i][0:i])

        stats_vec = np.zeros(35)
        funcs = [np.sum, np.max, np.min, np.mean, np.std, skew, kurtosis]*5
        data = [eigen_vals_FCD.reshape(-1),
                PCA_FCD.reshape(-1),
                Upper_FCD,
                Lower_FCD,
                FCD_TRIU.reshape(-1)]
        n0 = 7
        for k in range(len(data)):
            for i in range(k*n0, (k+1)*n0):
                stats_vec[i] = funcs[i](data[k])

        stats_vec = np.append(stats_vec, np.quantile(FCDe,
                                                     [0.05, 0.25,
                                                      0.5, 0.75, 0.95]))
        stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

        return stats_vec
