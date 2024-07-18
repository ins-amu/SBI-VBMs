import torch
import numpy as np
# from numba import jit
from scipy.signal import filtfilt, butter, hilbert


def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


def build_matrix(linear_phfcd, size, k=-1):
    tri = np.zeros((size, size))
    i_lower = tril_indices_column(size, k=k)
    tri[i_lower] = linear_phfcd
    tri.T[i_lower] = tri[i_lower]  # make symmetric matrix
    return tri

def compute_plv(x):
    ''' 
    compute phase locking value (PLV) from time series

    Parameters
    ----------
    x: 2d array
        time series of shape (n_channels, n_time)

    '''
    n = x.shape[0]
    plv = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            phi1 = instant_phase(x[i, :])
            phi2 = instant_phase(x[j, :])
            plv[i, j] = plv[j, i] = phase_locking_value(phi1, phi2)
    return plv

def compute_pli(x):
    ''' 
    compute phase lag index (PLI) from time series

    Parameters
    ----------
    x: 2d array
        time series of shape (n_channels, n_time)

    '''
    n = x.shape[0]
    pli = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            phi1 = instant_phase(x[i, :])
            phi2 = instant_phase(x[j, :])
            pli[i, j] = pli[j, i] = phase_lag_index(phi1, phi2)

    return pli


def instant_phase(x):
    '''
    compute instantaneous phase from time series
    '''

    return np.unwrap(np.angle(hilbert(x)))

def phase_locking_value(theta1, theta2):
    '''
    compute phase locking value (PLV) from two time series

    Parameters
    ----------
    theta1: 1d array
        time series
    theta2: 1d array
        time series

    Returns
    -------
    plv: float
        phase locking value
    '''
    dtheta = theta1 - theta2
    complex_phase_diff = np.exp(1j*(dtheta))
    _plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return _plv

def filter_butter_bandpass(sig, fs, lowcut, highcut, order=5):
    """
    Butterworth filtering function

    :param sig: [np.array] Time series to be filtered
    :param fs: [float] Frequency sampling in Hz
    :param lowcut: [float] Lower value for frequency to be passed in Hz
    :param highcut: [float] Higher value for frequency to be passed in Hz
    :param order: [int] The order of the filter.
    :return: [np.array] filtered frequncy
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, sig)

def slice_x(x: torch.Tensor, features: list, info: dict):
    """
    Args:
    x (ndarray): input array
    features (list): list of features
    info (dict): dictionary with info about the features

    return (ndarray): sliced array
    """

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    assert(x.ndim == 2)
    x_ = torch.tensor([])

    if len(features) == 0:
        return x_

    for feature in features:
        if feature in info:
            coli, colf = info[feature][0], info[feature][1]
            x_ = torch.cat((x_, x[:, coli:colf]), dim=1)
        else:
            raise ValueError(f"{feature} not in info")

    return x_
