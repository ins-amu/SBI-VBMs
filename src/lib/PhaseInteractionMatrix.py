# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  Computes the Phase-Interaction Matrix
#
#  Explained at
#  [Deco2019] Awakening: Predicting external stimulation to force transitions between different brain states
#       Gustavo Deco, Josephine Cruzat, Joana Cabral, Enzo Tagliazucchi, Helmut Laufs,
#       Nikos K. Logothetis, and Morten L. Kringelbach
#       PNAS September 3, 2019 116 (36) 18088-18097; https://doi.org/10.1073/pnas.1905534116
#  But defined as this at:
#  [Lopez-Gonzalez2020] Loss of consciousness reduces the stability of brain hubs and the heterogeneity of brain dynamics
#       Ane Lopez-Gonzalez, Rajanikant Panda, Adrian Ponce-Alvarez, Gorka Zamora-Lopez, Anira Escrichs,
#       Charlotte Martial, Aurore Thibaut, Olivia Gosseries, Morten L. Kringelbach, Jitka Annen,
#       Steven Laureys, and Gustavo Deco
#       bioRxiv preprint doi: https://doi.org/10.1101/2020.11.20.391482
#
#  Translated to Python by Xenia Kobeleva
#  Revised by Gustavo Patow
#  Refactored by Gustavo Patow
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import warnings
import numpy as np
from numba import jit
from scipy import signal, stats
from src.lib import BOLDFilters
import warnings
warnings.filterwarnings("ignore")

name = 'PhaseInteractionMatrix'


discardOffset = 0 #10  
# This was necessary in the old days when, after pre-processing, data had many errors/outliers at
# the beginning and at the end. Thus, the first (and last) 10 samples used to be discarded. Nowadays this filtering is
# done at the pre-processing stage itself, so this value is set to 0. Thus, depends on your data...

# BOLDFilters.flp = 0.008
# BOLDFilters.fhi = 0.08

@jit
def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


# def tril_indices_column(N, k=0):
#     row_i, col_i = np.nonzero(
#         np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
#     Isubdiag = (col_i,
#                 row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
#     return Isubdiag

@jit 
def _dfc(phases, t):
    N = phases.shape[0]
    dFC = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dFC[i, j] = np.cos(adif(phases[i, t - 1], phases[j, t - 1]))
    return dFC


def from_fMRI(ts, step=1):  
    ''' 
    Compute the Phase-Interaction Matrix of an input BOLD signal

    parameters
    ----------
    ts: 2D array
        BOLD signal (time series) of size N x Tmax
    low_cut: float
        Low cut frequency for the band-pass filter
    high_cut: float
        High cut frequency for the band-pass filter
    TR: float
        sampling interval (in seconds)
    
    returns
    -------
    PhIntMatr: 3D array
        Phase-Interaction Matrix of size Tmax x N x N

    '''

    (N, Tmax) = ts.shape
    npattmax = Tmax

    if not np.isnan(ts).any():
        phases = np.zeros((N, Tmax))
        PhIntMatr = np.zeros((npattmax, N, N))

        ts = ts - np.mean(ts, axis=1)[:, None]
        Xanalytic = signal.hilbert(ts, axis=1)

        for n in range(N):
            phases[n, :] = np.angle(Xanalytic[n, :])

        T = np.arange(0, Tmax, step)
        for t in T:
            PhIntMatr[t] = _dfc(phases, t)
    else:
        warnings.warn('Warning! PhaseInteractionMatrix.from_fMRI: NAN found')
        PhIntMatr = np.array([np.nan])
    return PhIntMatr
    

