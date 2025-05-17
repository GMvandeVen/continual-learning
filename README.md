# Continual Learning
[![DOI](https://zenodo.org/badge/150479999.svg)](https://zenodo.org/badge/latestdoi/150479999)

This is a PyTorch implementation of the continual learning experiments with deep neural networks described in the
following article:
* [Three types of incremental learning](https://www.nature.com/articles/s42256-022-00568-3) (2022, *Nature Machine Intelligence*)

This repository mainly supports experiments in the *academic continual learning setting*, whereby
a classification-based problem is split up into multiple, non-overlapping *contexts*
(or *tasks*, as they are often called) that must be learned sequentially.
Some support is also provided for running more flexible, "task-free" continual learning experiments
with gradual transitions between contexts.


### Earlier version
An earlier version of the code in this repository can be found 
[in this branch](https://github.com/GMvandeVen/continual-learning/tree/preprints).
This version of the code was used for the continual learning experiments described
in two preprints of the above article:
- Three scenarios for continual learning (<https://arxiv.org/abs/1904.07734>)
- Generative replay with feedback connections as a general strategy for continual learning
(<https://arxiv.org/abs/1809.10635>)


## Installation & requirements
The current version of the code has been tested with `Python 3.10.4` on a Fedora operating system
with the following versions of PyTorch and Torchvision:
* `pytorch 1.11.0`
* `torchvision 0.12.0`

Further Python-packages used are listed in `requirements.txt`.
Assuming Python and pip are set up, these packages can be installed using:
```bash
pip install -r requirements.txt
```

The code in this repository itself does not need to be installed, but a number of scripts should be made executable:
```bash
chmod +x main*.py compare*.py all_results.sh
```


## NeurIPS tutorial "Lifelong Learning Machines"
This code repository is used for the
[NeurIPS 2022 tutorial "Lifelong Learning Machines"](https://sites.google.com/view/neurips2022-llm-tutorial).
For details and instructions on how to re-run the experiments presented in this tutorial,
see the folder [NeurIPS-tutorial](NeurIPStutorial).


## ICLR blog post "On the computation of the Fisher Information in continual learning"
This repository is also used for the
[ICLR 2025 blog post "On the computation of the Fisher Information in continual learning"](https://iclr-blogposts.github.io/2025/blog/fisher/).
For details and instructions on how to re-run the experiments reported in this blog post,
see the folder [ICLR-blogpost](ICLRblogpost).


## Demos
##### Demo 1: Single continual learning experiment
```bash
./main.py --experiment=splitMNIST --scenario=task --si
```
This runs a single continual learning experiment:
the method Synaptic Intelligence on the task-incremental learning scenario of Split MNIST
using the academic continual learning setting.
Information about the data, the network, the training progress and the produced outputs is printed to the screen.
Expected run-time on a standard desktop computer is ~6 minutes, with a GPU it is expected to take ~3 minutes.

##### Demo 2: Comparison of continual learning methods
```bash
./compare.py --experiment=splitMNIST --scenario=task
```
This runs a series of continual learning experiments,
comparing the performance of various methods on the task-incremental learning scenario of Split MNIST.
Information about the different experiments, their progress and 
the produced outputs (e.g., a summary pdf) are printed to the screen.
Expected run-time on a standard desktop computer is ~100 minutes, with a GPU it is expected to take ~45 minutes.


## Re-running the comparisons from the article
The script `all_results.sh` provides step-by-step instructions for re-running the experiments and re-creating the
tables and figures reported in the article "Three types of incremental learning".

Although it is possible to run this script as it is, it will take very long and it is probably sensible to parallellize
the experiments.


## Running custom experiments
#### Academic continual learning setting
Custom individual experiments in the academic continual learning setting can be run with `main.py`.
The main options of this script are:
- `--experiment`: how to construct the context set? (`splitMNIST`|`permMNIST`|`CIFAR10`|`CIFAR100`)
- `--contexts`: how many contexts?
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)

To run specific methods, you can use the following:
- Separate Networks: `./main.py --separate-networks`
- Context-dependent-Gating (XdG): `./main.py --xdg`
- Elastic Weight Consolidation (EWC): `./main.py --ewc` (read first: [ICLR-blogpost](ICLRblogpost/README.md))
- Synaptic Intelligence (SI): `./main.py --si`
- Learning without Forgetting (LwF): `./main.py --lwf`
- Functional Regularization Of the Memorable Past (FROMP): `./main.py --fromp`
- Deep Generative Replay (DGR): `./main.py --replay=generative`
- Brain-Inspired Replay (BI-R): `./main.py --brain-inspired`
- Experience Replay (ER): `./main.py --replay=buffer`
- Averaged Gradient Episodic Memory (A-GEM): `./main.py --agem`
- Generative Classifier: `./main.py --gen-classifier`
- incremental Classifier and Representation Learning (iCaRL): `./main.py --icarl`

To run baseline models (see the article for details):
- None ("lower target"): `./main.py`
- Joint ("upper target"): `./main.py --joint`

For information on further options: `./main.py -h`.
The code supports combinations of several of the above methods.
It is also possible to create custom approaches by mixing components of different methods,
although not all possible combinations have been tested.

#### More flexible, "task-free" continual learning experiments
Custom individual experiments in a more flexible, "task-free" continual learning setting can be run with 
`main_task_free.py`. The main options of this script are:
- `--experiment`: how to construct the context set? (`splitMNIST`|`permMNIST`|`CIFAR10`|`CIFAR100`)
- `--contexts`: how many contexts?
- `--stream`: how to transition between contexts? (`fuzzy-boundaries`|`academic-setting`|`random`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)

For information on further options: `./main_task_free.py -h`. This script supports several of the above 
continual learning methods, but not (yet) all of them. Some methods have been slightly modified to 
make them suitable for the absence of (known) context boundaries.
In particular, methods that normally perform a certain consolidation operation at context boundaries, instead perform
this consolidation operation every `X` iterations, whereby `X` is set with the option `--update-every`. 

## On-the-fly plots during training
With this code progress during training can be tracked with on-the-fly plots. This feature requires `visdom`, 
which can be installed as follows:
```bash
pip install visdom
```
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main.py` or `./main_task_free.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.


### Citation
If you use this code in your research, please consider citing the main accompanying article:
```
@article{vandeven2022three,
  title={Three types of incremental learning},
  author={van de Ven, Gido M and Tuytelaars, Tinne and Tolias, Andreas S},
  journal={Nature Machine Intelligence},
  volume={4},
  pages={1185--1197},
  year={2022}
}
```

The BibTeX citations for the two preprints that were also produced using this code repository are given below.
Generally it is however preferred to cite the officially published version of the article,
but these preprints can be cited for aspects not featured in the published article.
```
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
```


### Acknowledgments
The research project from which this code originated has been supported by an IBRO-ISN Research Fellowship,
by the ERC-funded project *KeepOnLearning* (reference number 101021347),
by the National Institutes of Health (NIH) under awards R01MH109556 (NIH/NIMH) and P30EY002520 (NIH/NEI),
by the *Lifelong Learning Machines* (L2M) program of the Defence Advanced Research Projects Agency (DARPA)
via contract number HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA)
via Department of Interior/Interior Business Center (DoI/IBC) contract number D16PC00003.
Disclaimer: views and conclusions contained herein are those of the authors and should not be interpreted
as necessarily representing the official policies or endorsements, either expressed or implied,
of NIH, DARPA, IARPA, DoI/IBC, or the U.S. Government.
