# BOCPDMS: Bayesian On-line Changepoint Detection with Model Selection

This repository contains code from the _Bayesian On-line Changepoint Detection with Model Selection_ project.

## Introduction to BOCPDMS

Bayesian On-line Changepoint Detection (BOCPD) is a discrete-time inference framework introduced in the statistics and machine learning community independently by Fearnhead & Liu (2007) and Adams & MacKay (2007). Taken together, both papers have generated in excess of 500 citations and inspired more research in this area. The method is popular because it is efficient and runs in constant time per observation processed. We are working on extending the inference paradigm in several ways:

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

## Installation instructions

1. [Clone this repository](https://help.github.com/articles/cloning-a-repository/)
2. Change to the repository directory on your local machine
3. \[Optional] Create a new virtual environment for this project (see [*why use a virtual environment*](#why-use-a-virtual-environment) below)
4. Install the required packages using `pip install -r requirements.txt`
5. \[Optional] Verify that everything is working by running the tests (see [*run the tests*](#run-the-tests) below)

### Why use a virtual environment?

A virtual environment is an isolated instance of python that has it's own separately managed set of installed libraries (depandencies).
Creating a separate virtual environment for each project you are reproducing has two key advantages:
 1. It ensures you are using **only** the libraries specified by the authors. This verifies that
    they have provided **all** the information about the required libraries necessary to
    reproduce their work and that you are not accidentally relying on previously installed
    versions of common libraries.
  2. It ensure that you are using the **same versions** of the libraries specified by the
     authors. This ensures that a failure to reproduce is not caused by changes to libraries
     made between the authors publishing their project and you attempting to reproduce it.
  3. It ensures that none of the libraries required for the project interfere with the
    libraries installed in the standard python environment you use for your day to day work.

You can create a new virtual environment using python's built-in [venv](https://docs.python.org/3/library/venv.html) command.
Alternatively, if you use Anaconda to manage your python environment, you can [create a virtual environment
from within Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html).

#### Instructions w conda:

From inside the `bocpdms` folder on your computer:

```
conda create -n bocpdms python=3.7
conda activate bocpdms
pip install -r requirements.txt
```
as a fun little side note, if you want to use jupyter lab with this new environment, you should also run the following command so you can see this new `bocpdms` kernel :sparkles:
```
conda install -c conda-forge jupyterlab
conda install nb_conda_kernels
```

#### Instructions with virtualenv

**NOTE:**  The tests in this project will fail when run in a virtual environment created using
virtualenv.
This is due to a [known issue with matplotlib and virtualenv](https://matplotlib.org/faq/osx_framework.html).
If you use virtualenv to manage your python environment, please use python's built-in [venv](https://docs.python.org/3/library/venv.html) command to create your virtual environment for this project.

### Run the tests

From the repository directory run `python -m pytest`.

This will run all the tests in the `tests/` folder of the project.

You should see the following celebratory message :tada::sparkles::cake:

```bash
============================= test session starts =============================
platform win32 -- Python 3.7.0, pytest-3.7.1, py-1.7.0, pluggy-0.8.0
rootdir: \path\to\your\version\of\bocpdms, inifile:
collected 6 items

tests\test_Evaluation_tool.py .....                                      [ 83%]
tests\test_nile_example.py .                                             [100%]

========================== 6 passed in 17.83 seconds ==========================
```
