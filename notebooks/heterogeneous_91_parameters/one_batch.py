from helpers import *

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)

i_NODE = int(sys.argv[1])
mini_batch_size = int(sys.argv[2])
n_workers = int(sys.argv[3])

data_path = json.load(open("data_path.json", "r"))['data_path']
parameters = pickle.load(open(join(data_path, "parameters.pkl"), "rb"))
theta = np.load(join(data_path, f"theta_{i_NODE}.npz"))['theta']
opts = json.load(open(join(data_path, "opts.json"), "r"))
features = list(opts.keys())
batch_simulator(i_NODE, theta, parameters, features, opts, mini_batch_size, n_workers)
