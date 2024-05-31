# Reproducibility code for dpMMD and dpHSIC (Differentially Private Permutation Test)

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[Differentially Private Permutation Tests: Applications to Kernel Methods](https://arxiv.org/pdf/2310.19043.pdf).

The code is written in [JAX](https://jax.readthedocs.io/) which can leverage the architecture of GPUs to provide considerable computational speedups.

## Dependencies
- `python 3.9`

The packages in [env_cpu.yml](env_cpu.yml)/[env_gpu.yml](env_gpu.yml) are required to run our tests and the ones we compare against.

The instructions to install only the dependencies required to run dpMMD and dpHSIC are available on the [dpkernel](https://github.com/antoninschrab/dpkernel) repository. 

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```bash
git clone git@github.com:antoninschrab/dpkernel-paper.git
cd dpkernel-paper
```
We then recommend creating a `conda` environment with the required dependencies:
- for GPU:
  ```bash
  conda env create -f env_gpu.yml
  conda activate dpkernel-env
  # can be deactivated by running:
  # conda deactivate
  ```
- or, for CPU:
  ```bash
  conda env create -f env_cpu.yml
  conda activate dpkernel-env
  # can be deactivated by running:
  # conda deactivate
  ```

## Downloading CelebA dataset 

The `img_align_celeba.zip` file containing the CelebA images can either, be downloaded manually from the official [website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), or be downloaded by executing
```bash
wget https://cseweb.ucsd.edu/\~weijian/static/datasets/celeba/img_align_celeba.zip
```

The file then needs to be unzipped in the main `dpkernel-paper` directory by running
```bash
unzip img_align_celeba.zip
```
The resulting `img_align_celeba` directory then contains all the CelebA images.

The list of attributes for the CelebA images is already provided as the [list_attr_celeba.txt](list_attr_celeba.txt) file.

## Reproducibility of the experiments

The code to reproduce the experiments of the paper can be found in the following notebooks:
- [experiments_perturbations.ipynb](experiments_perturbations.ipynb): perturbed uniform experiment,
- [experiments_celeba.ipynb](experiments_celeba.ipynb): CelebA experiment.

These rely on the samplers [sampler_perturbations.py](sampler_perturbations.py) and [sampler_celeba.py](sampler_celeba.py), and can, for example, be opened through Jupyter Lab by running
```bash
jupyter lab
```

The results of the experiments are saved in the [results](results) directory. 
Running the code of the [figures.ipynb](figures.ipynb) notebook generates the figures and saves them in the [figures](figures) directory.

## How to use dpMMD and dpHSIC in practice?

Our proposed dpMMD and dpHSIC tests are implemented in [dpkernel.py](dpkernel.py).

To use our tests in practice, we recommend using our `dpkernel` package which is available on the [dpkernel](https://github.com/antoninschrab/dpkernel) repository. 
It can be installed by running
```bash
pip install git+https://github.com/antoninschrab/dpkernel.git
```
Installation instructions and example code are available on the [dpkernel](https://github.com/antoninschrab/dpkernel) repository. 

We also illustrate how to use the dpMMD and dpHSIC tests in the [demo.ipynb](demo.ipynb) notebook.

## References

We implement two general methods for privatising the non-private MMD and HSIC tests, which we refer to as TOT ([tot.py](tot.py)) and SARRM ([sarrm.py](sarrm.py)):
- TOT: [The Test of Tests: A Framework For Differentially Private Hypothesis Testing](https://arxiv.org/abs/2302.04260), Zeki Kazan, Kaiyan Shi, Adam Groce, Andrew Bray. 
- SARRM: [Differentially Private Hypothesis Testing with the Subsampled and Aggregated Randomized Response Mechanism](https://arxiv.org/abs/2208.06803),
Victor Pena, Andres F. Barrientos.

We also compare dpMMD to the test of [A Differentially Private Kernel Two-Sample Test](https://arxiv.org/abs/1808.00380), Anant Raj, Ho Chung Leon Law, Dino Sejdinovic, Mijung Park, using their implementation available on the [private_tst](https://github.com/hcllaw/private_tst) repository corresponding to the cloned directory [private_me](private_me). 

## Contact

If you have any issues running our code, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@unpublished{kim2023differentially,
title={Differentially Private Permutation Tests: {A}pplications to Kernel Methods}, 
author={Ilmun Kim and Antonin Schrab},
year={2023},
url = {https://arxiv.org/abs/2310.19043},
eprint={2310.19043},
archivePrefix={arXiv},
primaryClass={math.ST}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dckernel](https://github.com/antoninschrab/dckernel/): Robust to Data Corruption dcMMD dcHSIC tests
