import src
import numpy as np
from numpy import linalg as LA
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis


def get_fc(bold):
    FC = np.corrcoef(bold)
    FC = FC * (FC > 0)
    FC = FC - np.diag(np.diagonal(FC))
    return FC

def fluidity(fcd, k=0):
        '''!
        Calculate the fluidity of the BOLD signal
        
        Parameters
        ----------
        fcd: np.ndarray (2d)
            input FCD matrix
        k: int
            number of subdiagonal to be excluded from FCD matrix
        Returns
        -------
        fluidity: float
        '''

        fcd = np.triu(fcd, k=k)
        return np.var(fcd.reshape(-1))


def fc_homotopic(bold, opt={}):
    '''!
    Calculate the homotopic connectivity vector of a given brain activity

    Parameters
    ----------
    bold: array_like [nn, nt]
        The brain activity to be analyzed.
    Returns
    -------
    Homotopic_FC_vector : array_like [n_nodes]
        The homotopic correlation vector.

    Negative correlations may be artificially induced when using global signal regression
    in functional imaging pre-processing (Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017).
    Therefore, results on negative weights should be interpreted with caution and should be understood
    as complementary information underpinning the findings based on positive connections
    '''

    _opt = dict(positive=True, avg=True)
    _opt.update(opt)

    positive = _opt['positive']
    avg = _opt['avg']

    NHALF = int(bold.shape[0]//2)
    rsFC = np.corrcoef(bold)
    if positive:
        rsFC = rsFC * (rsFC > 0)
    rsFC = rsFC - np.diag(np.diag(rsFC))
    hfc = np.diag(rsFC, k=NHALF)
    if avg:
        return np.mean(hfc)
    else:
        return hfc


def get_fcd(ts, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity.

    Arguments:
        ts:      time series of shape [nt, nn]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples

    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    ts = ts.T
    n_samples, n_nodes = ts.shape
    # returns the indices for upper triangle
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []
    speed_stack = []
    fc_prev = []

    for t0 in range(0, ts.shape[0]-win_len, win_sp):
        t1 = t0+win_len
        fc = np.corrcoef(ts[t0:t1, :].T)
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
    # speed_ts = np.array(speed_stack)
    fcd = np.corrcoef(fcs)
    return fcd

def fc_region(ts, indices):
    '''
    return the functional connectivity matrix of given indices
    '''
    nn = ts.shape[0]
    mask = src.utility.make_mask(nn, indices)
    fc_mask = get_fc(ts)

    return fc_mask*(mask)


def fcd_region(ts, indices, win_len=30, win_sp=1, verbose=False):
    '''
    calculate masked fcd

    Parameters
    ----------
    ts : array_like
        time series of shape [nn, nt]
    indices : array_like
        indices of regions to be masked
    win_len : int, optional
        sliding window length in samples, by default 30
    win_sp : int, optional
        sliding window step in samples, by default 1
    '''
    nn = ts.shape[0]
    mask = np.zeros((nn, nn))
    mask[np.ix_(indices, indices)] = 1

    ts = ts.T
    n_samples, n_nodes = ts.shape
    # returns the indices for upper triangle
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    fc_stack = []

    try:
        for t0 in range(0, ts.shape[0]-win_len, win_sp):
            t1 = t0+win_len
            fc = np.corrcoef(ts[t0:t1, :].T)
            fc = fc*(fc > 0)*(mask)
            fc = fc[fc_triu_ids]
            fc_stack.append(fc)

        fcs = np.array(fc_stack)
        FCD = np.corrcoef(fcs)
        return FCD
    except Exception as e:
        if verbose:
            print(e)
        return [np.nan]

def kuramoto_op(x, indices=None):
    '''
    calculate the kuramoto order parameter from given time series
    Parameters
    ----------
    x: 2d array
        time series, shape (nn, nt)
    indices: 1d array
        indices of regions to be masked
    Returns
    -------
    r: float
        average kuramoto order parameter
    '''
    if indices is not None:
        assert (len(indices) > 1), 'indices should be a list of length > 1'
        x = x[indices, :]
    
    # extract the phase time series
    x_h = hilbert(x, axis=1)   
    x_phase = np.angle(x_h)
    x_phase = np.unwrap(x_phase, axis=1)

    # calculate the order parameter
    r = np.abs(np.mean(np.exp(1j*x_phase), axis=0))
    return np.mean(r)



def matrix_stats(A, opts={}):
        '''!

                Parameters
        ----------
        x: np.ndarray (2d)
            input array
        opt: dict
            dictionary of parameters
        Returns
        -------
        stats: np.ndarray (1d)
            feature values
        '''
        _opts = dict(demean=False, k=1, PCA_n_components=3)
        _opts.update(opts)
        demean = _opts["demean"]

        def funcs(x, demean=False):
            if demean:
                vec = np.zeros(3)
                vec[0] = np.std(x)
                vec[1] = skew(x)
                vec[2] = kurtosis(x)

            else:
                vec = np.zeros(7)
                vec[0] = np.sum(x)
                vec[1] = np.max(x)
                vec[2] = np.min(x)
                vec[3] = np.mean(x)
                vec[4] = np.std(x)
                vec[5] = skew(x)
                vec[6] = kurtosis(x)
            return vec

        off_diag_sum_A = np.sum(np.abs(A)) - np.trace(np.abs(A))

        A_TRIU = np.triu(A, k=_opts["k"])
        eigen_vals_A, _ = LA.eig(A)
        pca = PCA(n_components=_opts["PCA_n_components"])
        PCA_A = pca.fit_transform(A)

        Upper_A = []
        Lower_A = []
        for i in range(0, len(A)):
            Upper_A.extend(A[i][i+1:])
            Lower_A.extend(A[i][0:i])

        q = np.quantile(A, [0.05, 0.25, 0.5, 0.75, 0.95])
        _stats = np.array([])
        _stats = np.append(_stats, q)
        _stats = np.append(_stats, funcs(Upper_A, demean))
        _stats = np.append(_stats, funcs(Lower_A, demean))
        _stats = np.append(_stats, funcs(PCA_A.reshape(-1), demean))
        _stats = np.append(_stats, funcs(A_TRIU.reshape(-1), demean))
        _stats = np.append(_stats, funcs(np.real(eigen_vals_A[:-1]), demean))
        _stats = np.append(_stats, [off_diag_sum_A])

        return _stats
