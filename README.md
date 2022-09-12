This repo aims to calibrate Visual Place Recognition network to a target environment and provide uncertainty estimation. For more details look at [our paper](http://arxiv.org/abs/2203.04446)

If you use the code in this repository please cite:
```
@article{lajoieCalibration2022,
	title = {Self-{Supervised} {Domain} {Calibration} and {Uncertainty} {Estimation} for {Place} {Recognition} via {Robust} {SLAM}},
	url = {http://arxiv.org/abs/2203.04446},
	doi = {10.48550/arXiv.2203.04446},
	year = {2022},
	author = {Lajoie, Pierre-Yves and Beltrame, Giovanni},
}
```

# Installation

With conda:
    `conda install pytorch=1.9 torchvision=0.10 cudatoolkit=11.3 numpy=1.20 -c pytorch`

Otherwise, look at `environment.yml` for all the python package versions used to produce the paper's results.

# Run the code

We provide various scripts to run our code in `experiments/`.
Take inspiration of those scripts to know how to call the code and integrate it in your own code base.

Corresponding configuration files are available in `config/`.
Change the values in the config files to match your system.

## Scripts

- `netvlad_training.py` : Calibration of a NetVLAD network.

- `uncertainty_training.py` : Train an uncertainty estimator.

- `uncertainty_evaluation.py` : Example of a evaluation only (no training).

Don't hesitate to extend this repository with different configurations.

# Acknowledgement

Partially inspired from [https://github.com/Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad), please use [my fork](https://github.com/lajoiepy/pytorch-NetVlad) if you want to train NetVLAD from scratch and produce PCA pkl.
