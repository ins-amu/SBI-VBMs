from helpers import *
warnings.filterwarnings("ignore")

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)

subfolder = "0"
path = "output"
n_sim = 200

os.makedirs(join(path, subfolder, "figs"), exist_ok=True)

SC = load_SC_88("data")
limb_idx = load_limbic_indices_88("data")
nn = num_nodes = SC.shape[0]
SC = SC.T

params = {
    "G": 0.5,                         # global coupling strength
    "fix_seed": 0,
    "seed": SEED,
    "t_initial": 0.0,               # initial time * 10[ms]
    "t_transition": 10.0,         # transition time * 10 [ms]
    "t_final": 15000.0,             # end time * 10 [ms]
    "adj": SC,                     # weighted connection matrix
    "data_path": "output",
    "control": ["G"],
}

prior_min = [0.0]
prior_max = [1.1]

prior = utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min),
    high=torch.as_tensor(prior_max))
prior, _, _ = process_prior(prior)

theta = brute_sample(prior, n_sim, n_sim, 1, 1)
torch.save(theta, join(path, "theta.pt"))
torch.save(prior, join(path, "prior.pt"))

theta, x = batch_simulator(theta, params, 10)
np.savez(join(path, "data.npz"), x=x, theta=theta)