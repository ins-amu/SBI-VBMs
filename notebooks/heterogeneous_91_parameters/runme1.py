from helpers import *

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)

SC = load_SC_88("../data/")
nn = num_nodes = SC.shape[0]
idx_regions_dict = load_region_indices("../data")
idx_labels = list(idx_regions_dict.keys())
idx_regions = list(idx_regions_dict.values())

parameters = {
    # "data_path": "../output/91_param_",
    "data_path": "/p/project/icei-hbp-2023-0003/Jun/91_param_",
}

subfolder = "offset"
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

    "preprocessing": {
       "offset": 20,
    #    'demean': True,
    #    'zscore': True,
    #    'detrend': True,
       }
}
features = list(opts.keys())

with open(join(data_path, subfolder, "opts.json"), "w") as f:
    json.dump(opts, f, indent=4)

with open(join("data_path.json"), "w") as f:
    json.dump({"data_path": data_path}, f, indent=4)

num_simulations = 12 * 125000
mini_batch_size = 125
NUM_NODES = 50
n_workers = 125

for i_NODE in range(NUM_NODES):
    job_file = make_bash_script(i_NODE, 
                                mini_batch_size, 
                                n_workers, 
                                exe_file="one_batch1.py",
                                account="icei-hbp-2023-0003")
    os.system(f"sbatch {job_file}")

    #job_file = make_bash_script_local(i_NODE, mini_batch_size, n_workers)
    #os.system(f"bash {job_file}")
