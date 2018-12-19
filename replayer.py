import abc
from torch import nn


class Replayer(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract  module for a classifier/generator that can be trained with replay.

    Initiates ability to reset state of optimizer between tasks, and requires subclasses to implement a
    "train_a_batch"-function compatible with replay.'''

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


    #----------------- Replay-specifc functions -----------------#

    @abc.abstractmethod
    def train_a_batch(self, x, y, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <Variable> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <Variable> batch of corresponding labels
        [x_]              None or (<list> of) <Variable> batch of replayed inputs
        [y_]              None or (<list> of) <Variable> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <Variable> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes'''
        pass
