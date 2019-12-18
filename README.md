# Continual Learning
This is a PyTorch implementation of the continual learning experiments described in the following papers:
* Three scenarios for continual learning ([link](https://arxiv.org/abs/1904.07734))
* Generative replay with feedback connections as a general strategy 
for continual learning ([link](https://arxiv.org/abs/1809.10635))


## Requirements
The current version of the code has been tested with:
* `pytorch 1.1.0`
* `torchvision 0.2.2`


## Running the experiments
Individual experiments can be run with `main.py`. Main options are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)
- `--tasks`: how many tasks?

To run specific methods, use the following:
- Context-dependent-Gating (XdG): `./main.py --xdg=0.8`
- Elastic weight consolidation (EWC): `./main.py --ewc --lambda=5000`
- Online EWC:  `./main.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic intelligenc (SI): `./main.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main.py --replay=current --distill`
- Deep Generative Replay (DGR): `./main.py --replay=generative`
- DGR with distillation: `./main.py --replay=generative --distill`
- Replay-trough-Feedback (RtF): `./main.py --replay=generative --distill --feedback`
- iCaRL: `./main.py --icarl --budget=2000`

For information on further options: `./main.py -h`.


## Running comparisons from the papers
#### "Three CL scenarios"-paper
[This paper](https://arxiv.org/abs/1904.07734) describes three scenarios for continual learning (Task-IL, Domain-IL &
Class-IL) and provides an extensive comparion of recently proposed continual learning methods. It uses the permuted and
split MNIST task protocols, with both performed according to all three scenarios.

A comparison of all methods included in this paper can be run with `_compare.py`. The
comparison in Appendix B can be run with `_compare_taskID.py`, and Figure C.1 can be recreated with `_compare_replay.py`.

#### "Replay-through-Feedback"-paper
The three continual learning scenarios were actually first identified in [this paper](https://arxiv.org/abs/1809.10635),
after which this paper introduces the Replay-through-Feedback framework as a more efficent implementation of generative
replay. 

A comparison of all methods included in this paper can be run with
`_compare_time.py`. This includes a comparison of the time these methods take to train (Figures 4 and 5).

We should note that the results reported in this paper were obtained with
[this earlier version](https://github.com/GMvandeVen/continual-learning/tree/9c0ca78f43c29594b376ca59516031fcdaa5d7ba)
of the code. 


## On-the-fly plots during training
With this code it is possible to track progress during training with on-the-fly plots. This feature requires `visdom`, 
which can be installed as follows:
```bash
pip install visdom
```
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.


### Citation
Please consider citing our papers if you use this code in your research:
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
The research projects from which this code originated have been supported by an IBRO-ISN Research Fellowship, by the 
Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA) via contract number 
HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of 
Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. Disclaimer: views and conclusions 
contained herein are those of the authors and should not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of DARPA, IARPA, DoI/IBC, or the U.S. Government.
