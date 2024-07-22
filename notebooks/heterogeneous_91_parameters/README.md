# How to use this

These scripts are used to infer G,  α,  β  and 88 heterogeneous η .

- `runme.py` prepare the simulation BOLD signals using JSC cluster (50 Nodes, each 125 cores).
- `runme1.py` extract data features from BOLD signals
- `notebooko.ipynb` use the extracted data (functional and spatio-temporal) features to train the neural network. 
- `notebook_FC_FCD.ipynb` use only functional features to train the neural network for heterogeneous nodes with 91 parameters.

### Using the trained model

To use the trained neural network open `notes.ipynb`.
