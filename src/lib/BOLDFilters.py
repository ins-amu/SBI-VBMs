# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Applies filters to a BOLD signal
#
#  Translated to Python & refactoring by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from scipy.signal import butter, detrend, filtfilt
from scipy import signal
from numba import jit
# FILTER SETTINGS (from Gustavo Deco's FCD_LSD_model.m)
# -----------------------------------------------------


def BandPassFilter(boldSignal, low_cut=0.02, high_cut=0.1, TR=2.0, k=2):
    '''
    Convenience method to apply a filter (always the same one) to all areas in a BOLD signal. 

    Parameters
    ----------
    boldSignal : numpy.ndarray
        BOLD signal, with shape (N, Tmax), where N is the number of areas and Tmax is the number of time points.
    low_cut : float, optional
        Low cut frequency. The default is 0.02.
    high_cut : float, optional
        High cut frequency. The default is 0.1.
    TR : float, optional
        Sampling interval. The default is 2.0 second.
    removeStrongArtefacts : bool, optional
        If True, strong artefacts are removed. The default is True.

    returns
    -------
    signal_filt : numpy.ndarray
        Filtered BOLD signal, with shape (N, Tmax), where N is the number of areas and Tmax is the number of time points.


    '''

    assert (np.isnan(boldSignal).any() == False), 'NAN found in BOLD signal'

    (N, Tmax) = boldSignal.shape
    fnq = 1./(2.*TR)              # Nyquist frequency
    Wn = [low_cut/fnq, high_cut/fnq]
    bfilt, afilt = butter(k, Wn, btype='band', analog=False)

    signal_filt = np.zeros(boldSignal.shape)
    for seed in range(N):
        ts = boldSignal[seed, :]
        # Band pass filter. padlen modified to get the same result as in Matlab
        signal_filt[seed, :] = filtfilt(
            bfilt, afilt, ts, padlen=3*(max(len(bfilt), len(afilt))-1))
    return signal_filt
