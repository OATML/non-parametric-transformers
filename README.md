# Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning

  **[Overview](#overview)**
| **[Abstract](#abstract)**
| **[Installation](#installation)**
| **[Examples](#examples)**
| **[Citation](#citation)**

[![arXiv](https://img.shields.io/badge/arXiv-2106.02584-b31b1b.svg)](https://arxiv.org/abs/2106.02584)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.7-red.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)


## Overview

Hi, good to see you here! 👋

Thanks for checking out the code for Non-Parametric Transformers (NPTs).

This codebase will allow you to reproduce experiments from the paper as well as use NPTs for your own research.

## Abstract

We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms. However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction. Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models. We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate npt
```

For now, we recommend installing CUDA <= 10.2:

See [issue with CUDA >= 11.0 here](https://github.com/pytorch/pytorch/issues/47908).
 
If you are running this on a system without a GPU, use the above with `environment_no_gpu.yml` instead.

## Examples

We now give some basic examples of running NPT.

NPT downloads all supported datasets automatically, so you don't need to worry about that.

We use [wandb](http://wandb.com/) to log experimental results.
Wandb allows us to conveniently track run progress online.
If you do not want wandb enabled, you can run `wandb off` in the shell where you execute NPT.

For example, run this to explore NPT with default configuration on Breast Cancer

```
python run.py --data_set breast-cancer
```

Another example: A run on the poker-hand dataset may look like this

```
python run.py --data_set poker-hand \
--exp_batch_size 4096 \
--exp_print_every_nth_forward 100
```

You can find all possible config arguments and descriptions in `NPT/configs.py` or using `python run.py --help`.

In `scripts/` we provide a list with the runs and correct hyperparameter configurations presented in the paper.

We hope you enjoy using the code and please feel free to reach out with any questions 😊


## Citation

If you find this code helpful for your work, please cite our paper
[Paper](https://arxiv.org/abs/2106.02584) as

```bibtex
@article{kossen2021self,
  title={Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning},
  author={Kossen, Jannik and Band, Neil and Gomez, Aidan N. and Lyle, Clare and Rainforth, Tom and Gal, Yarin},
  journal={arXiv:2106.02584},
  year={2021}
}
```
