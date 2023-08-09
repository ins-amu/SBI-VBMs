
import os
import tqdm
import torch
import pickle
import logging
import numpy as np
from os.path import join
from src.utility import timer
from copy import copy, deepcopy
from multiprocessing import Pool
from sbi.inference import SNPE, SNLE, SNRE
from src.model.mpr_damaged import MPR as _MPR


class MPR:
    valid_parameters = ['control', 'adj', 'eta', 'eta_distribution',
                        'selected_node_indices', 'fix_seed', 'J', "local_regions",
                        'seed', 'iapp', 'sti_indices', 
                        'square_wave_stimulation', 'dt', 'dt_bold', 'tau', 'delta',
                        'G', 'noise_amp', 'decimate', 'RECORD_AVG', 'alpha_sc', 'beta_sc',
                        't_initial', 't_transition', 't_final', 'record_step',
                        'data_path', 'adj_A', 'adj_B', "CLUSTER"]
    
    def __init__(self, par) -> None:

        self.check_parameters(par)
        _par = self.get_default_parameters()
        _par.update(par)
        for item in _par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        assert(len(self.adj)> 0)
        self.num_nodes = self.adj.shape[0]

        if not "eta_distribution" in _par.keys():
            self.eta_distribution = None

        if not "selected_node_indices" in _par.keys():
            self.selected_node_indices = None

        if (not "seed" in _par.keys()) or (_par["seed"] is None):
            self.seed = None
        else:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        y0 = np.random.rand(2*self.num_nodes)
        y0[:self.num_nodes] = y0[:self.num_nodes]*1.5
        y0[self.num_nodes:] = y0[self.num_nodes:]*4-2
        self.initial_condition = y0

        self.FEATURE_PROPERTIES_SET = False
        self.control_str = _par['control']

    def get_default_parameters(self):

        params = {
            "G": 0.5,          # global coupling strength
            "dt": 0.01,        # integration time step for mpr model [ms]
            "dt_bold": 0.001,  # integration time step for Balloon model [s]
            "J": 14.5,         # model parameter
            "eta": -4.6,       # model parameter
            "tau": 1.0,        # model parameter
            "delta": 0.7,      # model parameter
            "decimate": 500,      # sampling from mpr time series 
            "noise_amp": 0.037,   # amplitude of noise

            "fix_seed": 0,
            "alpha_sc": 0.0,
            "beta_sc": 0.0,
            "seed": None,

            # stimulation parameters
            "iapp": 0.0,                   # constant applyed current/stimulation amplitude
            "sti_indices": [],
            "square_wave_stimulation": 0,  # apply square wave stimulation
            "selected_node_indices": None,
            "eta_distribution": "equal",

            "t_initial": 0.0,              # initial time * 10[ms]
            "t_transition": 1000.0,        # transition time * 10 [ms]
            "t_final": 25000.0,            # end time * 10 [ms]

            "adj": [],                     # weighted connection matrix
            "adj_A": [],                   # interhemispheric links
            "adj_B": [],                   # specific region links
            "record_step": 10,             # sampling every n step from mpr time series

            "data_path": "output",
            "control": [],
            "RECORD_AVG": 0,
            "CLUSTER": ""
        }
        return params

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")
    # -------------------------------------------------------------------------
    def set_eta_distribution(self, par, subname):

        # check control to be a list
        self.control = self.control.tolist() if isinstance(
            self.control, np.ndarray) else self.control

        index = self.control.index('eta')
        # equal values for all eta --------------------------------------------
        if (self.eta_distribution is None) or self.eta_distribution == "equal":
            self.eta = par[index] * np.ones(self.num_nodes)
        # equal value for local nodes -----------------------------------------
        elif(self.eta_distribution == "equal_local"):
            assert(len(self.eta)==self.num_nodes)
            assert(self.selected_node_indices is not None)
            self.eta[self.selected_node_indices] = par[index]
        # uniform distribution ------------------------------------------------
        elif (self.eta_distribution == "uniform"):
            if self.selected_node_indices is None:
                assert(len(par) >= self.num_nodes)
                self.eta = par[index: index + self.num_nodes]
            else:
                nn = len(self.selected_node_indices)
                self.eta[self.selected_node_indices] = par[index:index+nn]

        elif (self.eta_distribution == "local_regions"):
            for ii in range(len(self.local_regions)):
                self.eta[self.local_regions[ii]] = par[index + ii]

        # normal distribution -------------------------------------------------
        elif (self.eta_distribution == "normal"):

            # check if eta is array or scalar
            if not isinstance(self.eta, (tuple, list, np.ndarray)):
                self.eta = np.ones(self.num_nodes) * self.eta

            idx = self.control_str.index("eta")
            mu_idx = idx
            sigma_idx = idx + 1

            assert(len(par) >= 2)  # par should include mu and sigma
            assert(len(self.selected_node_indices) > 0)
            eta0 = np.random.normal(loc=par[mu_idx],                # mu
                                    scale=par[sigma_idx],           # sigma
                                    size=len(self.selected_node_indices))
            self.eta[self.selected_node_indices] = eta0

        elif (self.eta_distribution == "bimodal"):

            if not isinstance(self.eta, (tuple, list, np.ndarray)):
                self.eta = np.ones(self.num_nodes) * self.eta

            idx = self.control_str.index("eta")
            mu_idx_loc = idx
            sigma_idx_loc = idx + 1
            mu_idx_glob = idx + 2
            sigma_idx_glob = idx + 3

            assert(len(par) >= 4)  # par should include 2 mu and 2 sigma
            assert(len(self.selected_node_indices) > 0)
            eta_loc = np.random.normal(loc=par[mu_idx_loc],
                                       scale=par[sigma_idx_loc],
                                       size=len(self.selected_node_indices))

            eta_glob = np.random.normal(loc=par[mu_idx_glob],
                                        scale=par[sigma_idx_glob],
                                        size=self.num_nodes)
            self.eta = eta_glob
            self.eta[self.selected_node_indices] = eta_loc

    # -------------------------------------------------------------------------

    def check_eta_distribution(self):
        if isinstance(self.eta, (tuple, list, np.ndarray)) and (len(self.eta) == self.num_nodes):
            pass
        else:
            self.eta *= np.ones(self.num_nodes)

    # -------------------------------------------------------------------------
    def check_control_parameters_combination(self):

        # alpha and beta are parameters
        if ("alpha" in self.control) and ("beta" in self.control):
            if ("eta" in self.control) or ("J" in self.control):
                print(
                    "[(alpha, beta) with (eta or J) can not be parameter simultaneously !!!")
                exit(0)
            else:
                self.eta = (self.alpha * self.delta) * np.ones(self.num_nodes)
                self.J = self.beta * np.sqrt(self.delta)
    # -------------------------------------------------------------------------    
    def check_adj_AB(self):
            if (abs(self.alpha_sc) > 0): 
                assert(len(self.adj_A) > 0), "adj_A is empty."
            if (abs(self.beta_sc) > 0):
                assert(len(self.adj_B) > 0), "adj_B is empty."

    # -------------------------------------------------------------------------
    def simulate(self, par=[], subname="0000000"):
        if torch.is_tensor(par):
            par = np.float64(par.numpy())
        else:
            par = copy(par)

        try:
            _ = len(par)
        except:
            par = [par]

        assert (len(par) >= len(self.control)
                ), "control vector parameter does not match."

        for item, value in enumerate(self.control):
            if value != "eta":
                setattr(self, value, par[item])

        if "eta" in self.control:  # tested only for control ["G", "eta"]

            self.set_eta_distribution(par, subname)
        else:
            self.check_eta_distribution()

        self.check_control_parameters_combination()
        self.check_adj_AB()
        

        obj = _MPR(self.dt,
                   self.dt_bold,
                   self.decimate,
                   self.initial_condition,
                   self.adj,
                   self.adj_A,
                   self.adj_B,
                   self.G,
                   self.eta,
                   self.alpha_sc,
                   self.beta_sc,
                   [int(i) for i in self.sti_indices],
                   self.J,
                   self.tau,
                   self.delta,
                   self.iapp,
                   int(self.square_wave_stimulation),
                   self.noise_amp,
                   self.record_step,
                   self.t_initial,
                   self.t_transition,
                   self.t_final,
                   self.RECORD_AVG,
                   int(self.fix_seed))

        sol = obj.heunStochasticIntegrate(self.data_path, subname)
        bold = np.asarray(sol).astype(np.float32).T

        return bold

# =============================================================================
# =============================================================================
# =============================================================================


class Inference():

    def __init__(self, parameters) -> None:
        super().__init__(parameters["data_path"])

        self.decimate = parameters['decimate']
        self.data_path = parameters['data_path']
        self.num_nodes = parameters['adj'].shape[0]
        try:
            self.CLUSTER = parameters['CLUSTER']
        except:
            self.CLUSTER = "local"


        if not os.path.exists(join(self.data_path, "bold")):
            os.makedirs(join(self.data_path, "bold"))
        self.features_in_x = None
    # -------------------------------------------------------------------------
    
    def sample_posterior(self,
                         obs_stats,
                         num_samples,
                         posterior
                         ):

        samples = posterior.sample((num_samples,), x=obs_stats)
        return samples
    # -------------------------------------------------------------------------

    @timer
    def train(self,
              num_simulations,
              prior,
              x,
              theta,
              num_threads=1,
              method="SNPE",
              device="cpu",
              density_estimator="maf"
              ):

        torch.set_num_threads(num_threads)

        if (len(x.shape) == 1):
            x = x[:, None]
        if (len(theta.shape) == 1):
            theta = theta[:, None]

        x = x[:num_simulations, :]
        theta = theta[:num_simulations, :]

        if method == "SNPE":
            inference = SNPE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNLE":
            inference = SNLE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNRE":
            inference = SNRE(
                prior=prior, density_estimator=density_estimator, device=device)
        else:
            print("unknown method, choose SNLE, SNRE or SNPE.")
            exit(0)

        # inference._device = self._device
        inference = inference.append_simulations(theta, x)
        _density_estimator = inference.train()
        posterior = inference.build_posterior(_density_estimator)

        return posterior
    # -------------------------------------------------------------------------
