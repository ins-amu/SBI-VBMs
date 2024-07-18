from helpers import *

SEED = 2
torch.manual_seed(SEED)
np.random.seed(SEED)

i_NODE = int(sys.argv[1])
mini_batch_size = int(sys.argv[2])
n_workers = int(sys.argv[3])
n_batch = 260


subfolder = 'offset'
data_path = json.load(open("data_path.json", "r"))['data_path']
os.makedirs(join(data_path, subfolder, "stats"), exist_ok=True)

opts = json.load(open(join(data_path, subfolder, "opts.json"), "r"))
features = list(opts.keys())

for j in range(n_batch):
    data = pickle.load(open(f"{data_path}/bold/data_{i_NODE:05d}_{j:05d}.pkl", "rb"))
    theta, x = batch_extractor(data, opts, n_workers=n_workers)
    with open(join(data_path, subfolder, f"stats/stats_{i_NODE:05d}_{j:05d}.pkl"), 'wb') as f:
                        pickle.dump({"stats": x, "theta": theta}, f)

