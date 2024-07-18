from helpers import *

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)

SC = load_SC_88("../data/")
nn = num_nodes = SC.shape[0]
idx_regions_dict = load_region_indices("../data")
idx_labels = list(idx_regions_dict.keys())
idx_regions = list(idx_regions_dict.values())

SC_A = mask_M1(SC)
SC_B = mask_M2(SC, idx_regions_dict['limbic'])

SC = SC.T
SC_A = SC_A.T
SC_B = SC_B.T

parameters = {
    "G": 0.5,          # global coupling strength
    "dt": 0.01,        # integration time step for mpr model [ms]
    "dt_bold": 0.001,  # integration time step for Balloon model [s]
    "J": 14.5,         # model parameter
    "eta": -4.6 * np.ones(nn),       # model parameter
    "tau": 1.0,        # model parameter
    "delta": 0.7,      # model parameter
    "decimate": 500,      # sampling from mpr time series 
    "noise_amp": 0.037,   # amplitude of noise

    "fix_seed": 1,
    "alpha_sc": 0.0,
    "beta_sc": 0.0,
    "seed": SEED,

    # stimulation parameters
    "iapp": 0.0,                    # constant applyed current/stimulation amplitude
    "sti_indices": [],
    "square_wave_stimulation": 0,   # apply square wave stimulation

    "t_initial": 0.0,               # initial time * 10[ms]
    "t_transition": 5000.0,         # transition time * 10 [ms]
    "t_final": 25000.0,             # end time * 10 [ms]

    "adj": SC,                     # weighted connection matrix
    "adj_A": SC_A,                 # interhemispheric links
    "adj_B": SC_B,                 # specific region links
    "record_step": 10,             # sampling every n step from mpr time series

    # "data_path": "91_param_",
    "data_path": "/p/project/icei-hbp-2021-0002/Jun/91_param_",
    "control": ["G", "alpha_sc", "beta_sc", "eta"],
    "RECORD_AVG": 0,
    "CLUSTER": ""
}
parameters['eta_distribution'] = "uniform"

subfolder = "0"
data_path = parameters['data_path']
folders = [subfolder, "bold", "stats"]
for folder in folders:
    os.makedirs(join(data_path, folder), exist_ok=True)

opts = {
    "fc_corr": {'k': 10},
    "fcd_corr": {'k': 20, 'wwidth':30, 'olap':0.94, 'maxNwindows':200},
    "fc_sum": {},
    "fcd_sum": {},
    "fluidity": {},
    "moments": {},
    "higher_moments": {},
    "envelope": {},
    'fcd_edge_var': {},
    "fc_homotopic": {"positive": True},
    
    "fcd_filt_limbic": {"regions": idx_regions_dict['limbic'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_filt_frontal": {"regions": idx_regions_dict['frontal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_filt_occipital": {"regions": idx_regions_dict['occipital'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_filt_temporal": {"regions": idx_regions_dict['temporal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_filt_parietal": {"regions": idx_regions_dict['parietal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_filt_centralstructures": {"regions": idx_regions_dict['centralstructures'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_0": {"regions": idx_regions_dict['limbic'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_1": {"regions": idx_regions_dict['frontal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_2": {"regions": idx_regions_dict['occipital'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_3": {"regions": idx_regions_dict['temporal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_4": {"regions": idx_regions_dict['parietal'], "win_sp": 1, "k": 1, "win_len": 20},
    "fcd_corr_5": {"regions": idx_regions_dict['centralstructures'], "win_sp": 1, "k": 1, "win_len": 20},
    'fc_corr_0': {"regions": idx_regions_dict['limbic'], "k": 1},
    'fc_corr_1': {"regions": idx_regions_dict['frontal'], "k": 1},
    'fc_corr_2': {"regions": idx_regions_dict['occipital'], "k": 1},
    'fc_corr_3': {"regions": idx_regions_dict['temporal'], "k": 1},
    'fc_corr_4': {"regions": idx_regions_dict['parietal'], "k": 1},
    'fc_corr_5': {"regions": idx_regions_dict['centralstructures'], "k": 1},

    #"preprocessing": {
    #    "offset": 20,
        # 'demean': True,
        # 'zscore': True,
    #    }
}
features = list(opts.keys())

with open(join(data_path, "opts.json"), "w") as f:
    json.dump(opts, f, indent=4)

with open(join(data_path, "parameters.pkl"), "wb") as f:
    pickle.dump(parameters, f)

with open(join("data_path.json"), "w") as f:
    json.dump({"data_path": data_path}, f, indent=4)

num_simulations = 13 * 125000
mini_batch_size = 125
NUM_NODES = 50
n_workers = 125

prior_min_G_a_b = [0.2, 0.0, 0.0]
prior_max_G_a_b = [1.1, 1.0, 1.0]
selected_node_indices = list(range(num_nodes)) # for all nodes
parameters["selected_node_indices"] = selected_node_indices
prior_min_eta_mu = -6.0 * np.ones(len(selected_node_indices))
prior_max_eta_mu = -3.5 * np.ones(len(selected_node_indices))
prior_PAR_min = np.hstack([prior_min_G_a_b, prior_min_eta_mu])
prior_PAR_max = np.hstack([prior_max_G_a_b, prior_max_eta_mu])
theta = sample_prior(num_simulations, prior_PAR_min, prior_PAR_max, data_path)
theta_np = theta.numpy()
theta_list = np.array_split(theta_np, NUM_NODES, axis=0)

for i in range(NUM_NODES):
    np.savez(join(data_path, f"theta_{i}.npz"), theta=theta_list[i])

for i_NODE in range(NUM_NODES):
    job_file = make_bash_script(i_NODE, mini_batch_size, n_workers)
    os.system(f"sbatch {job_file}")

    #job_file = make_bash_script_local(i_NODE, mini_batch_size, n_workers)
    #os.system(f"bash {job_file}")
