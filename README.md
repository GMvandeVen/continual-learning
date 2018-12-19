# Continual Learning & Replay-through-Feedback
This is a PyTorch implementation of the continual learning experiments described in the following paper:
* Gido M. van de Ven, Andreas S. Tolias (2018) Generative replay with feedback connections as a general strategy 
for continual learning, [arXiv preprint](https://arxiv.org/abs/1809.10635)

## Requirements
The current version of the code has been tested with:
* `pytorch 0.4.1`
* `torchvision 0.2.1`

## Running the experiments
Individual experiments can be run with `main.py`. Main options are:
- `--experiment`: which task protocol should be used? (`splitMNIST`|`permMNIST`)
- `--scenario`: according to which continual learning scenario should this protocol be performed? (`task`|`domain`|`class`)

To run specific methods, use the following:
- Elastic weight consolidation (EWC): `./main.py --ewc`
- Online EWC:  `./main.py --ewc --online`
- Synaptic intelligenc (SI): `./main.py --si`
- Context-dependent-Gating (XdG): `./main.py --XdG=0.8`
- Deep Generative Replay (DGR): `./main.py --replay=generative`
- DGR + distillation: `./main.py --replay=generative --distill`
- Replay-trough-Feedback (RtF): `./main.py --replay=generative --distill --feedback`

For information on further options: `./main.py -h`.

A comparison of all methods used in [our paper](https://arxiv.org/abs/1809.10635) can be run with 
`compare_MNIST.py`.

## On-the-fly plots during training
This code enables tracking of progress during training with on-the-fly plots. This feature requires `visdom`, 
which can be installed as follows:
```bash
pip install visdom
```
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
here). The flag `--visdom` should then be added when calling `./main.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.

### Citation
Please consider citing our paper if you use this code in your research:
```
@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
```

### Acknowledgments
The research project from which this code originated has been supported by an IBRO-ISN Research Fellowship, by the 
Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA) via contract number 
HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of 
Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. Disclaimer: views and conclusions 
contained herein are those of the authors and should not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of DARPA, IARPA, DoI/IBC, or the U.S. Government.
