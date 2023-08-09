# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the sliding-window Functional Connectivity Dynamics (swFCD)
#
#  Translated to Python & refactoring by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
# from numba import jit
from scipy import stats
from src.lib import BOLDFilters

name = 'swFCD'


def calc_length(start, end, step):
    # This fails for a negative step e.g., range(10, 0, -1).
    # From https://stackoverflow.com/questions/31839032/python-how-to-calculate-the-length-of-a-range-without-creating-the-range
    return (end - start - 1) // step + 1


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x.flatten(), y.flatten())
    return corr_mat[0, 1]


def KolmogorovSmirnovStatistic(FCD1, FCD2):  # FCD similarity
    d, pvalue = stats.ks_2samp(FCD1.flatten(), FCD2.flatten())
    return d


def distance(FCD1, FCD2):  # FCD similarity, convenience function
    if not (np.isnan(FCD1).any() or np.isnan(FCD2).any()):  # No problems, go ahead!!!
        return KolmogorovSmirnovStatistic(FCD1, FCD2)
    else:
        return -1  # ERROR_VALUE

def calc_FCD(signal, windowSize=30, windowStep=3):
    N, Tmax = signal.shape
    lastWindow = Tmax - windowSize
    N_windows = calc_length(0, lastWindow, windowStep)
    
    windows = np.array([signal[:, t:t+windowSize+1].T for t in range(0, lastWindow, windowStep)])
    corr_matrices = np.array([np.corrcoef(win, rowvar=False) for win in windows])
    
    Isubdiag = np.tril_indices(N, k=-1)
    cotsampling = np.array([pearson_r(corr_matrices[ii][Isubdiag], corr_matrices[jj][Isubdiag])
                           for ii in range(N_windows)
                           for jj in range(ii+1, N_windows)])
    return cotsampling, N_windows


def init(S, N):
    return np.array([], dtype=np.float64)


def accumulate(FCDs, nsub, signal):
    FCDs = np.concatenate((FCDs, signal))  # Compute the FCD correlations
    return FCDs


def findMinMax(arrayValues):
    return np.min(arrayValues), np.argmin(arrayValues)
