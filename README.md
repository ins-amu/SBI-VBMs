# Simulation-based Inference on Virtual Brain Models of Disorders (SBI-VBMs)

## Installation

```sh
conda env create --file environment.yml --name vbm
conda activate vbm
cd SBI-VBMs
pip install -r requirements.txt
pip install -e .

# gpu support
# conda install -c conda-forge cupy cudatoolkit=11.3
# conda install -c conda-forge pytorch-gpu
```
