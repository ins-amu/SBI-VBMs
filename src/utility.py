import torch
import numpy as np
from scipy import stats
from typing import Tuple
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from scipy.signal import detrend
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, mutual_info_regression

def band_pass_filter(boldSignal, low_cut=0.02, high_cut=0.1, TR=2.0, k=2):
    '''
    Convenience method to apply a filter to all areas in a BOLD signal. 

    Parameters
    ----------
    boldSignal : numpy.ndarray
        BOLD signal, with shape (N, Tmax), where N is the number of areas and Tmax 
        is the number of time points.
    low_cut : float, optional
        Low cut frequency. The default is 0.02.
    high_cut : float, optional
        High cut frequency. The default is 0.1.
    TR : float, optional
        Sampling interval. The default is 2.0 second.

    returns
    -------
    signal_filt : numpy.ndarray
        Filtered BOLD signal, with shape (N, Tmax), where N is the number of areas and 
        Tmax is the number of time points.


    '''

    assert (np.isnan(boldSignal).any() == False), 'NAN found in BOLD signal'

    (nn, nt) = boldSignal.shape
    fnq = 1.0/(2.0 * TR)              # Nyquist frequency
    Wn = [low_cut/fnq, high_cut/fnq]
    bfilt, afilt = butter(k, Wn, btype='band', analog=False)

    return filtfilt(bfilt, afilt, boldSignal,
                            padlen=3*(max(len(bfilt), len(afilt))-1))


def remove_strong_artefacts(x, threshold=3.0):

    nn = x.shape[0]
    for i in range(nn):
        ts = x[i, :]
        std_dev = threshold * np.std(ts)
        ts[ts > std_dev] = std_dev
        ts[ts < -std_dev] = -std_dev
        x[i, :] = ts
    return x

def hilbert_transform(x):
    '''
    Hilbert transform of a signal.

    Parameters
    ----------
    x : numpy.ndarray
        Signal to be transformed.

    Returns
    -------
    x_hilbert : numpy.ndarray
        Hilbert transform of the signal.

    '''
    x_hilbert = np.zeros_like(x, dtype=np.complex)
    for i in range(x.shape[0]):
        x_hilbert[i, :] = np.fft.ifft(np.fft.fft(x[i, :]) * 1j)
    return x_hilbert

def preprocessing_signal(boldSignal,
                         opts: dict = {}):
    _opts = {
        'bandpass': {'low': 0.01,   # bandpass filter
                     'high': 0.1,
                     'TR': None,
                     'order': 2},
        'detrend': False,               # detrend the signal
        'demean': False,                # demean the signal
        'zscore': False,                # zscore the signal
        'offset': 0,                    # remove the first offset time points
        'removeStrongArtefacts': False,
        'gsr': False,
    }
    if not "bandpass" in opts:
        opts["bandpass"] = None
    _opts.update(opts)

    offset = _opts['offset']
    if offset > 0:
        boldSignal = boldSignal[:, offset:]

    if _opts['zscore']:
        boldSignal = stats.zscore(boldSignal, axis=1)

    if _opts['demean']:
        boldSignal = boldSignal - np.mean(boldSignal, axis=1)[:, None]

    if _opts['detrend']:
        boldSignal = detrend(boldSignal, axis=1)

    if _opts['bandpass'] is not None:
        low_cut = _opts['bandpass']['low']
        high_cut = _opts['bandpass']['high']
        TR = _opts['bandpass']['TR']
        order = _opts['bandpass']['order']
        boldSignal = band_pass_filter(boldSignal,
                                      k=order,
                                      TR=TR,
                                      low_cut=low_cut,
                                      high_cut=high_cut
                                      )
    if _opts['removeStrongArtefacts']:
        boldSignal = remove_strong_artefacts(boldSignal)

    if _opts['gsr']:
        boldSignal = global_regress_signal(boldSignal)

    return boldSignal



def make_mask(nn, indices):
    '''
    make a 2D (nn x nn) mask for a given set of indices
    1 for the indices, 0 otherwise

    Parameters
    ----------
    nn : int
        number of nodes
    indices : array_like
        indices of nodes to be masked

    Returns
    -------
    m : numpy.ndarray
        mask
    '''

    m = np.zeros((nn,nn), dtype=int)
    m[np.ix_(indices, indices)] = 1

    return m

def moving_average(x, w, axis=1, mode="same"):
    return np.apply_along_axis(lambda x: np.convolve(x, np.ones(w), mode) / w, axis=axis, arr=x)


def brute_sample(prior, num_samples, nx=None, ny=None, num_ensebmles=1):
    '''
    Args:
        prior (sbi.utils.torchutils.BoxUniform): prior information
        num_samples (int): number of samples to generate
        nx (int): number of samples in x dimension
        ny (int): number of samples in y dimension
        num_ensebmles (int): number of ensembles to generate

    Returns:
        ndarray: samples
    '''
    try:
        low = prior.base_dist.low.tolist()
        high = prior.base_dist.high.tolist()
    except:
        if isinstance(prior, utils.MultipleIndependent):
            low = [prior.dists[i].low.item() for i in range(2)]
            high = [prior.dists[i].high.item() for i in range(2)]
        else:
            raise ValueError("prior not supported")

    assert(len(high) == len(low))
    if len(high) == 1:
        # theta = torch.linspace(low[0], high[0], steps=num_samples)[:, None].float()
        theta = torch.linspace(low[0], high[0], steps=num_samples).float()
        return theta.repeat(1, num_ensebmles).T

    elif len(high) == 2:

        assert(nx is not None)
        assert(ny is not None)

        interval = np.abs([high[0] - low[0], high[1] - low[1]])

        'true' if True else 'false'
        step_x = interval[0] / (nx - 1) if (nx > 1) else interval[0]
        step_y = interval[1] / (ny - 1) if (ny > 1) else interval[1]

        theta = []
        for i in range(nx):
            for j in range(ny):
                theta.append([low[0] + i * step_x, low[1] + j * step_y])

        theta = torch.tensor(theta).float()
        return theta.repeat(num_ensebmles, 1)

def timer(func):
    '''
    decorator to measure elapsed time

    Parameters
    -----------
    func: function
        function to be decorated
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end-start, message="{:s}".format(func.__name__))
        return result
    return wrapper


def is_sequence(x):
    if isinstance(x, collections.abc.Sized):
        return True
    else:
        return False

def flatten(t):
    """
    flatten a list of list

    Parameters
    ----------
    t : list of list

    Return:
        flattend list
    """
    return [item for sublist in t for item in sublist]


def set_k_diogonal(A, k, value=0.0):
    '''
    set k diagonals of the given matrix to given value.
    '''

    return _set_k_diogonal(A, k, value)

def get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits

def posterior_peaks(samples, return_dict=False, **kwargs):

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = get_limits(samples)
    samples = samples.numpy()
    n, dim = samples.shape

    try:
        labels = opts['labels']
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(
            samples[:, row],
            bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(
            limits[row, 0], limits[row, 1],
            opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())

def display_time(time, message=""):
    '''
    display elapsed time in hours, minutes, seconds

    Parameters
    -----------
    time: float
        elaspsed time in seconds
    '''

    hour = int(time/3600)
    minute = (int(time % 3600))//60
    second = time-(3600.*hour+60.*minute)
    print("{:s} Done in {:d} hours {:d} minutes {:09.6f} seconds".format(
        message, hour, minute, second))
