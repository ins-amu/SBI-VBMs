import os
import sys
import json
import tqdm
import torch
import pickle
import warnings
import numpy as np
import seaborn as sns
from copy import copy
from os.path import join
from src.utility import *
from src.analysis import *
import sbi.utils as utils
import scipy.stats as stats
from typing import Optional
import multiprocessing as mp
from scipy.stats import kstest
from src.model.mpr import MPR
from sbi.inference import SNPE
import matplotlib.pyplot as plt
from scipy.special import kl_div
from sbi.analysis import pairplot
from src.utility import brute_sample
from src.utility import preprocessing_signal
from torch.distributions import Distribution
from sbi.utils.user_input_checks import process_prior
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------

LABELSIZE = 16
plt.rc('axes', labelsize=LABELSIZE)
plt.rc('axes', titlesize=LABELSIZE)
plt.rc('figure', titlesize=LABELSIZE)
plt.rc('legend', fontsize=LABELSIZE)
plt.rc('xtick', labelsize=LABELSIZE)
plt.rc('ytick', labelsize=LABELSIZE)

# --------------------------------------------------------------------

def train(prior: Optional[Distribution],
          theta:torch.Tensor, # shape: (n_samples, n_params)    torch.float32
          x:torch.Tensor,     # shape: (n_samples, n_features)  torch.float32
          n_threads:int=1):
    
    torch.set_num_threads(n_threads)
    inference = SNPE(prior=prior, 
                     density_estimator="maf", 
                     device="cpu")
    inference = inference.append_simulations(theta, x)
    posterior = inference.build_posterior(inference.train())

    return posterior


def get_features(bold: np.ndarray, offset=20):
    '''
    get features from given BOLD signal

    Parameters
    ----------
    bold : np.ndarray
        BOLD signal, with shape (nn, nt), where nn is the number of areas and nt
        is the number of time points.
    group : str, optional
        group name. The default is "".
    subj : str, optional
        subject name. The default is "".

    Returns
    -------
    Labels : list
        list of feature names.
    Stat_vec : list
        list of feature values.

    '''
    assert(bold.shape[1] > offset), "BOLD signal is too short"
    bold = bold[:, offset:]

    nn = 88
    Labels = []
    Stat_vec = []

    # FC_homotopic
    fch = fc_homotopic(bold)
    Labels += ['fch']
    Stat_vec += [fch]

    # FC
    fc = get_fc(bold)
    stat_vec = [np.sum(np.abs(fc)) - np.trace(np.abs(fc))]
    label = ['fc_sum']
    Labels += label
    Stat_vec += stat_vec

    # FCD
    fcd = get_fcd(bold)
    stat_vec = [np.sum(np.abs(fcd)) - np.trace(np.abs(fcd))]
    label = ['fcd_sum']
    Labels += label
    Stat_vec += stat_vec

    stat_vec = [fluidity(fcd)]
    label = ['fluidity']
    Labels += label
    Stat_vec += stat_vec

    return Labels, Stat_vec


def wrapper_simulate(counter, parameters, theta_i, to_file=True):

    if torch.is_tensor(theta_i):
        theta_i = np.float64(theta_i.numpy().tolist())
    elif isinstance(theta_i, np.ndarray):  # if is numpy array
        theta_i = copy(theta_i.tolist())
    try:
        _ = len(theta_i)
    except:
        theta_i = [theta_i]

    data_path = parameters['data_path']

    sol = MPR(parameters)
    bold = sol.simulate(theta_i)

    if bold.ndim == 2:
        assert(bold.shape[1] > 20), "BOLD signal is too short"
        labels, stat_vec = get_features(bold, offset=20)
        if to_file:
            np.savez(join(data_path, f"bold/bold_{counter:05d}.npz"), bold=bold, theta=theta_i)
            np.savez(join(data_path, f"stats/stats_{counter:05d}.npz"), stats=stat_vec, theta=theta_i)        
        return theta_i, stat_vec, bold
    else:
        return None, None, None

def batch_simulator(theta, parameters, n_workers=1, to_file=True):

    n_sim = theta.shape[0]
    data_path = parameters['data_path']
    os.makedirs(join(data_path, "bold"), exist_ok=True)
    os.makedirs(join(data_path, "stats"), exist_ok=True)

    def update_bar(_):
        pbar.update()

    with mp.Pool(processes=n_workers) as pool:
        with tqdm.tqdm(total=n_sim) as pbar:
            async_results = [pool.apply_async(wrapper_simulate, 
                                              args=(i, parameters, theta[i, :], to_file),
                                              callback=update_bar) for i in range(n_sim)]
            data = [async_result.get()[:2] for async_result in async_results]
        
    X, Theta = [], []
    for i in range(len(data)):
        if data[i][0] is not None:
            X.append(data[i][1])
            Theta.append(data[i][0])
    X = np.array(X, dtype=np.float32)
    Theta = np.array(Theta, dtype=np.float32)

    return torch.from_numpy(Theta), torch.from_numpy(X)

def load_SC_88(data_path):
    SC_path = join(data_path, "healthy_3610A")
    data = np.load(join(SC_path, "Structural_matrix_3610A.npz"))
    SC = data['weights']
    np.fill_diagonal(SC, 0.0)
    SC = SC/np.max(SC)
    return np.abs(SC)
# --------------------------------------------------------------------


def load_limbic_indices_88(data_path):
    SC_path = join(data_path, "healthy_3610A")
    region_labels = np.loadtxt(join(data_path,
                                    "healthy_3610A",
                                    "region_labels.txt"), dtype=str)
    limbc_labels = np.loadtxt(join(SC_path,
                                   "Australian_Labels.txt"),
                              dtype=str, usecols=(1,))
    ind = [np.where(region_labels == limbc_labels[i])[0][0]
           for i in range(len(limbc_labels))]
    ind.sort()
    return ind
# --------------------------------------------------------------------


def mask_M1(SC):
    """
    return interhemispheric links of SC
    """

    n, m = SC.shape
    assert(n == m)

    SC1 = np.zeros_like(SC)
    SC1[int(n/2):n, 0:int(n/2)] = SC[int(n/2):n, 0:int(n/2)]
    SC1[0:int(n/2), int(n/2):n] = SC[0:int(n/2), int(n/2):n]

    return SC1
# --------------------------------------------------------------------


def mask_M2(SC, region_indices):
    """
    return links of given regions of SC
    """

    n, m = SC.shape
    assert(n == m)

    SC1 = np.zeros_like(SC)

    left = []
    right = []

    for i in region_indices:
        if i < int(n/2):
            left.append(i)
        else:
            right.append(i)

    SC1[left, 0:int(n/2)] = SC[left, 0:int(n/2)]
    SC1[right, int(n/2):n] = SC[right, int(n/2):n]

    return SC1
# --------------------------------------------------------------------

def get_distance(x0, y0, metric='kl'):

    if isinstance(x0, torch.Tensor):
        x0 = x0.detach().numpy()
    if isinstance(y0, torch.Tensor):
        y0 = y0.detach().numpy()
    
    if x0.ndim == 2:
        x0 = x0.flatten()
    if y0.ndim == 2:
        y0 = y0.flatten()

    if metric == 'euclidean':
        return np.sqrt(np.sum((x0-y0)**2/len(x0)))

    elif metric == 'kl':  # Kullback-Leibler divergence
        return calculate_kl_divergence(x0, y0, 1.0)
    elif metric == 'ks':  # Kolmogorov-Smirnov test for goodness of fit
        return kstest(x0, y0)[0]

    elif metric == 'corr':
        return np.corrcoef(x0, y0)[0, 1]

    elif metric == 'cosine':
        return np.dot(x0, y0)/(np.linalg.norm(x0)*np.linalg.norm(y0))

    else:
        raise ValueError("Unknown metric")


def calculate_kl_divergence(x0, y0, shift=1.0):
    
    x0 = x0 + shift
    y0 = y0 + shift
    kl_values = kl_div(x0, y0)
    kl_sum = np.sum(kl_values) * 1.442695 # in bits
    return kl_sum


def fit_gaussian(data):
    # Estimate mean and standard deviation from the data
    mu, std = stats.norm.fit(data)
    return mu, std

def kl_divergence(mu1, std1, mu2, std2):
    # Create two normal distributions using the estimated means and standard deviations
    dist1 = stats.norm(mu1, std1)
    dist2 = stats.norm(mu2, std2)
    
    # Generate points within the range of the data
    x = np.linspace(min(mu1 - 3 * std1, mu2 - 3 * std2), max(mu1 + 3 * std1, mu2 + 3 * std2), 1000)
    
    # Calculate the probability density functions for the two distributions
    pdf1 = dist1.pdf(x)
    pdf2 = dist2.pdf(x)
    
    # Calculate the KL divergence between the two distributions
    kl_div = stats.entropy(pdf1, pdf2)
    
    return kl_div

def load_bold(idx, path):
    fname = join(path, "bold", f"bold_{idx:05d}.npz")
    if os.path.exists(fname):
        data = np.load(fname)
        bold = data['bold']
        theta = data['theta']
        return theta, bold
    print(f"File {fname} not found")
    return None, None
