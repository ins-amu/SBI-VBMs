# Simulation-based Inference on Virtual Brain Models of Disorders (SBI-VBMs)

```
@article{SBI-VBMs,
	author={Hashemi, Meysam and Ziaeemehr, Abolfazl and Woodman, Marmaduke M. and Fousek, Jan and Petkoski, Spase and Jirsa, Viktor},
	title={Simulation-based Inference on Virtual Brain Models of Disorders},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ad6230},
	year={2024},
	abstract={Connectome-based models, also known as Virtual Brain Models (VBMs), have
		  been well established in network neuroscience to investigate pathophysiological causes
		  underlying a large range of brain diseases. The integration of an individual’s brain
		  imaging data in VBMs has improved patient-specific predictivity, although Bayesian
		  estimation of spatially distributed parameters remains challenging even with state-
		  of-the-art Monte Carlo sampling. VBMs imply latent nonlinear state space models
		  driven by noise and network input, necessitating advanced probabilistic machine
		  learning techniques for widely applicable Bayesian estimation. Here we present
		  Simulation-based Inference on Virtual Brain Models (SBI-VBMs), and demonstrate
		  that training deep neural networks on both spatio-temporal and functional fea-
		  tures allows for accurate estimation of generative parameters in brain disorders.
		  The systematic use of brain stimulation provides an effective remedy for the non-
		  identifiability issue in estimating the degradation limited to smaller subset of con-
		  nections. By prioritizing model structure over data, we show that the hierarchical
		  structure in SBI-VBMs renders the inference more effective, precise and biologically
		  plausible. This approach could broadly advance precision medicine by enabling fast
		  and reliable prediction of patient-specific brain disorders.}
}
```


This research has received funding from EU’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS 2.0 Project), No. 101137289 (Virtual Brain Twin Project), and government grant managed by the Agence Nationale de la Recherch reference ANR-22-PESN-0012 (France 2030 program).



## Installation

### Quick Install (Recommended)

The easiest way to install the package with all dependencies:

```sh
pip install -e .
```

For development (includes testing and formatting tools):
```sh
pip install -e ".[dev]"
```

For documentation building:
```sh
pip install -e ".[docs]"
```

To install everything:
```sh
pip install -e ".[all]"
```

### Building C++ Extensions

After installation, you need to compile the C++ model extensions:

```sh
# Install SWIG (if not already installed)
# Option 1: Using system package manager
sudo apt install swig

# Option 2: Using conda (recommended if using conda environment)
conda install -c conda-forge swig

# Build the extensions
cd src/model
make
```

### Verify Installation

After installation, verify that the package is correctly installed:

```sh
python -c "import src; print(f'Package version: {src.__version__}')"
```

This should print: `Package version: 0.2.0`

### Alternative: Conda Environment Installation

If you prefer using conda for environment management:

```sh
# Create conda environment with Python 3.9, 3.10, or 3.11
conda create -n sbi-vbms python=3.11
conda activate sbi-vbms

# Install SWIG for building C++ extensions
conda install -c conda-forge swig

# Install the package
pip install -e .

# Build C++ extensions
cd src/model
make

# Verify installation
python -c "import src; print(f'Package version: {src.__version__}')"
```
