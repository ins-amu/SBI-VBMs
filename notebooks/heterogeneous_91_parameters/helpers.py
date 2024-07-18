import os
import sys
import json
import tqdm
import torch
import pickle
import warnings
import numpy as np
from copy import copy
from os.path import join
import sbi.utils as utils
from typing import Optional
import scipy.stats as stats
import multiprocessing as mp
from src.model.mpr import MPR
from sbi.inference import SNPE
import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from src.lib.utility import slice_x
from scipy.stats import gaussian_kde
from src.utility import brute_sample
from torch.distributions import Distribution
from src.lib.feature_extraction import Features
from sbi.utils.user_input_checks import process_prior
from data_loader import AustralianDataset88_4gr
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------

LABELSIZE = 18
plt.rc('axes', labelsize=LABELSIZE)
plt.rc('axes', titlesize=LABELSIZE)
plt.rc('figure', titlesize=LABELSIZE)
plt.rc('legend', fontsize=LABELSIZE)
plt.rc('xtick', labelsize=LABELSIZE)
plt.rc('ytick', labelsize=LABELSIZE)

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


def wrapper_simulate(counter,
                     parameters,
                     theta_i,
                     features,
                     opts):

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
    stat_vec = [np.nan]

    if bold.ndim == 2:
        F = Features(features, opts)
        stat_vec, stat_info = F.calc_features(bold)
        num_features = len(stat_vec)
        stat_info['num_features'] = num_features

        fname = join(data_path, 'features_info.json')
        if not os.path.exists(fname):
            with open(fname, 'w') as f:
                json.dump(stat_info, f, indent=4)
    
        return theta_i, stat_vec, bold
    else:
        return theta_i, [np.nan], [np.nan]

# --------------------------------------------------------------------


def batch_simulator(i_NODE, theta, parameters, features, opts, mini_batch_size=125, n_workers=125):

    n_sim = theta.shape[0]
    data_path = parameters['data_path']
    mini_batch_size = min(mini_batch_size, n_sim)

    def update_bar(_):
        pbar.update()

    with mp.Pool(processes=n_workers) as pool:
        with tqdm.tqdm(total=n_sim) as pbar:
            async_results = [pool.apply_async(wrapper_simulate,
                                              args=(i,
                                                    parameters,
                                                    theta[i, :],
                                                    features,
                                                    opts),
                                              callback=update_bar)
                             for i in range(n_sim)]
            counter = 0
            for i in range(0, n_sim-1, mini_batch_size):
                try:
                    data_ = [async_results[j].get() for j in range(i, i+mini_batch_size)]
                    theta = [data_[j][0] for j in range(mini_batch_size)]
                    stat_vec = [data_[j][1] for j in range(mini_batch_size)]
                    bold = [data_[j][2] for j in range(mini_batch_size)]
                    
                    with open(join(data_path, f"bold/data_{i_NODE:05d}_{counter:05d}.pkl"), 'wb') as f:
                        pickle.dump({"bold": bold, "theta": theta}, f)
                    with open(join(data_path, f"stats/stats_{i_NODE:05d}_{counter:05d}.pkl"), 'wb') as f:
                        pickle.dump({"stats": stat_vec, "theta": theta}, f)
                except Exception as e:
                    print(e)
                counter +=1
            
            # [async_result.get() for async_result in async_results]

# --------------------------------------------------------------------

def single_extractor(bold, features, opts):

    if np.array(bold).ndim == 2:
        F = Features(features, opts)
        stat_vec, stat_info = F.calc_features(bold)
        return stat_vec
    else:
        return [np.nan]

def batch_extractor(data, opts, n_workers=1):
    '''
    load each batch of data and extract features
    '''

    features = list(opts.keys())
    Bold = data['bold']
    Theta = data['theta']
    stat_list = []
    theta_list = []
    n = len(Theta)

    def update_bar(_):
        pbar.update()

    with mp.Pool(processes=n_workers) as pool:
        with tqdm.tqdm(total=n) as pbar:
            async_results = [pool.apply_async(single_extractor,
                                              args=(Bold[i], 
                                                    features, 
                                                    opts),
                                              callback=update_bar)
                             for i in range(n)]
            results = [async_result.get() for async_result in async_results]
    
    for i in range(len(results)):
        if not np.isnan(results[i]).any():
            stat_list.append(results[i])
            theta_list.append(Theta[i])

    return np.array(theta_list), np.array(stat_list)


def sample_prior(num_simulations, prior_min, prior_max, data_path):
    prior_dist = utils.torchutils.BoxUniform(
        low=torch.as_tensor(prior_min),
        high=torch.as_tensor(prior_max))
    prior, _, _ = process_prior(prior_dist)

    theta = prior.sample((num_simulations,))
    torch.save(theta, join(data_path, "theta.pt"))
    torch.save(prior_dist, join(data_path, "prior.pt"))
    return theta



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
    ind = [int(i) for i in ind]
    
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
import pandas as pd

class Connectivity(object):

    def __init__(self, data_path, subdir="connectivity_84"):
        self.data_path = join(data_path, subdir)

    def load_SC(self):

        filename = join(self.data_path, "weights.txt")
        SC = np.loadtxt(filename)
        np.fill_diagonal(SC, 0.0)
        SC = SC/np.max(SC)
        return np.abs(SC)

    def load_limbic(self):

        roi_names = self.load_roi_names()
        lim_region = ['Hippocampus', 
                    'parahippocampal', 
                    'cingulate',
                    'Amygdala', 
                    'temporalpole',     
                    'middletemporal']   
        
        lim_vec = []
        lim_idxs = []

        for i in roi_names:
            for j in lim_region:
                if i.find(j) != -1:
                    lim_vec.append(i)
                    lim_idxs.append(roi_names.index(i))
        
        
        df = pd.DataFrame({'limbic': lim_vec, 'idx': lim_idxs})
        return df

    def load_roi_names(self):
        filename = join(self.data_path, "centers.txt")
        roi_names = []
        with open(filename, 'r') as file:
            for line in file:
                columns = line.strip().split()
                roi_names.append(columns[0])
        return roi_names


def get_num_features(fname):
    # read json file
    with open(fname) as f:
        data = json.load(f)
    return data['num_features']

def make_bash_script(i_NODE, 
                     mini_batch_size,
                     n_workers=1,
                     filename="script", 
                     partition="gpus", 
                     time="24:00:00",
                     account="icei-hbp-2023-0003", # "icei-hbp-2023-0003"
                     nodes=1, 
                     exe_file="one_batch.py"):
    """
    make bash script for slurm

    Parameters
    ----------
    i_batch : int
        batch number
    filename : str, optional
        name of the bash script, by default "script"
    partition : str, optional
        partition name, by default "gpus"
    time : str, optional
        time, by default "24:00:00"

    Returns
    -------
    str
        bash script
    
    """

    os.makedirs("log", exist_ok=True)

    # open file to write
    job_filename = join("log", f"{filename}_{i_NODE}.sh")
    with open(job_filename, "w") as f:
        f.write("#!/bin/bash \n")
        f.writelines(f"""
#SBATCH --account={account}
#SBATCH --time={time} 
#SBATCH --nodes={nodes} 
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --job-name={i_NODE}.job
#SBATCH --output=log/log_{i_NODE}.log
#SBATCH --error=log/err_{i_NODE}.err 
ml Python CUDA GCC
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
source /p/project/icei-hbp-2021-0002/prog/VE/vbi/bin/activate
python -W ignore {exe_file} {i_NODE} {mini_batch_size} {n_workers}

""")
    return job_filename
                     
def make_bash_script_local(i_NODE, 
                     mini_batch_size,
                     n_workers=1,
                     filename="script", 
                     partition="gpus", 
                     time="24:00:00",
                     account="icei-hbp-2023-0003", #"icei-hbp-2021-0002",
                     nodes=1):
    """
    make bash script for slurm

    Parameters
    ----------
    i_batch : int
        batch number
    filename : str, optional
        name of the bash script, by default "script"
    partition : str, optional
        partition name, by default "gpus"
    time : str, optional
        time, by default "24:00:00"

    Returns
    -------
    str
        bash script
    
    """

    os.makedirs("log", exist_ok=True)

    # open file to write
    job_filename = join("log", f"{filename}_{i_NODE}.sh")
    with open(job_filename, "w") as f:
        f.write("#!/bin/bash \n")
        f.writelines(f"""
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -W ignore one_batch.py {i_NODE} {mini_batch_size} {n_workers}
""")
    return job_filename

    

def load_region_indices(data_path, verbose=False):
    obj = AustralianDataset88_4gr(data_path)

    regions = ["frontal", "temporal", "limbic",
               "parietal", "occipital",
               "centralstructures", "cingulate"]

    reg_names = obj.load_subject_region()
    idx = []

    for region in regions:
        func = getattr(obj, "load_" + region)
        _idx = func(reg_names)[1]
        idx.append(_idx)

    idx_limbic = np.sort(idx[regions.index("limbic")]).tolist()
    idx_frontal = np.sort(idx[regions.index("frontal")]).tolist()
    idx_temporal = np.sort(idx[regions.index("temporal")]).tolist()
    idx_occipital = np.sort(idx[regions.index("occipital")]).tolist()
    idx_centralstructures = np.sort(
        idx[regions.index("centralstructures")]).tolist()
    idx_temporal = list(set(idx_temporal) - set(idx_limbic))
    idx_temporal.sort()
    idx[regions.index("temporal")] = idx_temporal
    idx_parietal = np.sort(idx[regions.index("parietal")]).tolist()

    # check how many regions have overlap
    if verbose:
        print("Overlap between regions")
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                tmp = set(idx[i]).intersection(set(idx[j]))
                if len(tmp) > 0:
                    print(regions[i], regions[j], len(tmp))
        print("=====================================")
            

    return {"limbic": idx_limbic,
            "frontal": idx_frontal,
            "temporal": idx_temporal,
            "occipital": idx_occipital,
            "centralstructures": idx_centralstructures,
            "parietal": idx_parietal}

def plot_matrix(A, ax, title='', ticks=None, cmap='RdBu_r'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im = ax.imshow(A, cmap=cmap)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=ticks)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.grid(False)


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

