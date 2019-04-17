import abc
from torch import nn


class Replayer(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract  module for a classifier/generator that can be trained with replay.

    Initiates ability to reset state of optimizer between tasks.'''

    def __init__(self):
        super().__init__()

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        self.optim_type = "adam"
        #--> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every task
        self.optim_list = []
        #--> self.[optim_list]   <list>, if optimizer should be reset after each task, provide list of required <dicts>

        # Replay: temperature for distillation loss (and whether it should be used)
        self.replay_targets = "hard"  # hard|soft
        self.KD_temp = 2.

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass
