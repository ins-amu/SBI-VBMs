import numpy as np 
import collections.abc
from torch import linalg as tLA

def set_k_diogonal(A, k, value=0.0):
    '''
    set k diagonals of the given matrix to given value.
    '''

    assert(len(A.shape) == 2)
    n = A.shape[0]
    assert(k < n)
    for i in range(-k, k+1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value
    return A

def select_upper_triangular(A, k=0):
    '''
    select upper triangular elements of the given matrix
    '''
    assert(len(A.shape) == 2)
    n = A.shape[0]
    assert(k < n)
    idx = np.triu_indices(n, k=k)
    return A[idx]

def select_lower_triangular(A, k=0):
    '''
    select lower triangular elements of the given matrix
    '''
    assert(len(A.shape) == 2)
    n = A.shape[0]
    assert(k < n)
    idx = np.tril_indices(n, k=k)
    return A[idx]


def normalize(x, axis=0):
    xn = np.linalg.norm(x, axis=axis)
    return x/xn

def t_normalize(x, axis=0):

    xn = tLA.norm(x, axis=axis)
    return x/xn

def zscore(true_mean, post_mean, post_std):
    '''
    calculate z-score

    parameters
    ------------
    true_mean: float
        true value of the parameter
    post_mean: float
        mean [max] value of the posterior
    post_std: float
        standard deviation of postorior

    return
    --------

    z-score: float
        
    '''
    return np.abs((post_mean - true_mean) / post_std)

def shrinkage(prior_std, post_std):
    '''
    shrinkage = 1 -  \frac{sigma_{post}/sigma_{prior}} ^2

    parameters
    -----------
    prior_std: float
        standard deviation of prior
    post_std: float
        standard deviation of postorior

    return
    ----------
    shrinkage: float
    
    '''
    return 1 - (post_std / prior_std)**2


def RMSE(x1, x2):
    '''
    root mean square error
    '''
    return np.sqrt(((x1 - x2) ** 2).mean()) 


def is_sequence(x):
    if isinstance(x, collections.abc.Sized):
        return True
    else:
        return False
