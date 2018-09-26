from torch import nn
from torch.nn import functional as F
from linear_nets import MLP
import excitability_modules as eM
from continual_learner import ContinualLearner
from replayer import Replayer
import utils


class Classifier(ContinualLearner, Replayer):
    '''Model for classifying images, "enriched" as "ContinualLearner"- and "Replayer"-object.'''

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu",
                 bias=True, excitability=False, excit_buffer=False):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        self.fcE = MLP(input_size=image_channels*image_size**2, output_size=fc_units, layers=fc_layers-1, hid_size=fc_units,
                       drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, final_nl=True,
                       bias=bias, excitability=excitability, excit_buffer=excit_buffer)
        mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2

        # classifier
        self.classifier = nn.Sequential(nn.Dropout(fc_drop),
                                        eM.LinearExcitability(mlp_output_size, classes, excit_buffer=True))

    def forward(self, x):
        final_features = self.fcE(self.flatten(x))
        return self.classifier(final_features)

    @property
    def name(self):
        return "{}_c{}".format(self.fcE.name, self.classes)


    def train_a_batch(self, x, y, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <Variable> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <Variable> batch of corresponding labels
        [x_]              None or (<list> of) <Variable> batch of replayed inputs
        [y_]              None or (<list> of) <Variable> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <Variable> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes'''

        # Set model to training-mode
        self.train()


        ##--(1)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            predL = None if y is None else F.cross_entropy(y_hat, y)

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().data[0] / x.size(0)
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [i.e., for offline with incremental task learning]


        ##--(2)-- REPLAYED DATA --##

        if x_ is not None:
            # If [x_] is a list, perform separate replay for each entry
            n_replays = len(x_) if type(x_)==list else 1
            if not type(x_)==list:
                x_ = [x_]
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Loop to perform each replay
            for replay_id in range(n_replays):
                # If requested, apply correct mask for each new replay
                if self.mask_dict is not None:
                    self.apply_XdGmask(task=replay_id+1)

                # Run model
                y_hat = self(x_[replay_id])
                # -if needed (e.g., incremental/multihead set-up), remove predictions for classes not in replayed task
                if active_classes is not None:
                    y_hat = y_hat[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id])
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    distilL_r[replay_id] = utils.loss_fn_kd(scores=y_hat,
                                                            target_scores=scores_[replay_id], T=self.KD_temp,
                                                            cuda=self._is_on_cuda())
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

        # Calculate total loss
        if x is None:
            loss_total = sum(loss_replay)/n_replays
        elif x_ is None:
            loss_total = predL
        else:
            loss_replay = sum(loss_replay)/n_replays
            loss_total = rnt*predL + (1-rnt)*loss_replay


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss(cuda=self._is_on_cuda())
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss(cuda=self._is_on_cuda())
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        # Reset optimizer
        self.optimizer.zero_grad()
        # Backpropagate errors
        loss_total.backward()
        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.data[0],
            'pred': predL.data[0] if predL is not None else 0,
            'pred_r': sum(predL_r).data[0]/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).data[0]/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.data[0], 'si_loss': surrogate_loss.data[0],
            'precision': precision if precision is not None else 0.,
        }

