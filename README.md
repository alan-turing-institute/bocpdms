# BOCPDMS: Bayesian On-line Changepoint Detection with Model Selection

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/alan-turing-institute/bocpdms/master)

This repository contains code from the _Bayesian On-line Changepoint Detection with Model Selection_ project.

## Table of contents

* [About BOCPDMS](#about-bocpdms)
* [Citing this project](#citing-this-project)
* [Installation instructions](#installation-instructions)
* [Running the examples](#running-the-examples)
* [Reproducible research champions program](#reproducible-research-champions-program)
* [Contributors](#contributors)


## About BOCPDMS

Bayesian On-line Changepoint Detection (BOCPD) is a discrete-time inference framework introduced in the statistics and machine learning community independently by [Fearnhead & Liu (2007)](https://doi.org/10.1111/j.1467-9868.2007.00601.x) and [Adams & MacKay (2007)](https://arxiv.org/abs/0710.3742). Taken together, both papers have generated in excess of 500 citations and inspired more research in this area. The method is popular because it is efficient and runs in constant time per observation processed. We are working on extending the inference paradigm in several ways:

- [x] Unifiying Fearnhead & Liu (2007) and Adams & MacKay (2007)¹
- [x] Multivariate analysis¹
- [x] Robust analysis²
- [ ] Continuous-time models
- [ ] Point processes

### Papers

¹Jeremias Knoblauch and Theodoros Damoulas. [Spatio-temporal Bayesian On-line Changepoint Detection](https://arxiv.org/abs/1805.05383), _International Conference on Machine Learning_ (2018).

²Jeremias Knoblauch, Jack Jewson and Theodoros Damoulas. [Doubly Robust Bayesian Inference for Non-Stationary Streaming Data with β-Divergences](https://arxiv.org/abs/1806.02261), arXiv:1806.02261 (2018).

## Reproducible Research Champions

In May 2018, Theo Damoulas was selected as one of the Alan Turing Institute's Reproducible Research Champions - academics who encourage and promote reproducible research through their own work, and who want to take their latest project to the "next level" of reproducibility.

The Reproducible Research programme at the Turing is led by Kirstie Whitaker and Martin O'Reilly, with the Champions project also involving members of the Research Engineering Group.

Each of the Champions' projects will receive several weeks of support from the Research Engineering Group throughout Summer 2018; during this time, we will work on the project together with Jeremias and Theo and will track our efforts in this repository. Given our focus on reproducibility, we obviously won't be changing any of the code's functionality - but we will make it easier for you to install, use and test out your own ideas with the BOCPDMS methodology.

You can keep track of our progress through the Issues tab, and find out more about the Turing's Reproducible Research Champions project [here](https://github.com/alan-turing-institute/ReproducibleResearchResources).

## Installation Instructions

_Installation and usage instructions will be available shortly..._

Instructions w conda:
```
conda create -n bocpdms python=3.7
conda activate bocpdms
pip install -r requirements.txt
```
as a fun little side note, if you want to use jupyter lab with this new environment, you should also run the following command so you can see this new `bocpdms` kernel :sparkles:

```
conda install -c conda-forge jupyterlab
```
