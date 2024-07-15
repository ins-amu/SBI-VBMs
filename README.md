# Simulation-based Inference on Virtual Brain Models of Disorders (SBI-VBMs)

@article{SBI-VBMs,
	author={Hashemi, Meysam and Ziaeemehr, Abolfazl and Woodman, Marmaduke M. and Fousek, Jan and Petkoski, Spase and Jirsa, Viktor},
	title={Simulation-based Inference on Virtual Brain Models of Disorders},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ad6230},
	year={2024},
	abstract={Connectome-based models, also known as Virtual Brain Models (VBMs), have been well established in network neuroscience to investigate pathophysiological causes underlying a large range of brain diseases. The integration of an individual's brain imaging data in VBMs has improved patient-specific predictivity, although Bayesian estimation of spatially distributed parameters remains challenging even with state-of-the-art Monte Carlo sampling. VBMs imply latent nonlinear state space models driven by noise and network input, necessitating advanced probabilistic machine learning techniques for widely applicable Bayesian estimation. Here we present Simulation-based Inference on Virtual Brain Models (SBI-VBMs), and demonstrate that training deep neural networks on both spatio-temporal and functional features allows for accurate estimation of generative parameters in brain disorders. The systematic use of brain stimulation provides an effective remedy for the non-identifiability issue in estimating the degradation limited to smaller subset of connections. By prioritizing model structure over data, we show that the hierarchical structure in SBI-VBMs renders the inference more effective, precise and biologically plausible. This approach could broadly advance precision medicine by enabling fast and reliable prediction of patient-specific brain disorders.}
}


## Installation

```sh
conda env create --file environment.yml --name vbm
conda activate vbm
cd SBI-VBMs
pip install -r requirements.txt
pip install -e .

sudo apt install swig
cd src/model
make

# gpu support
# conda install -c conda-forge cupy cudatoolkit=11.3
# conda install -c conda-forge pytorch-gpu
```
