import numpy as np
import torch
from torch.nn import functional as F
from models.fc.layers import fc_layer, fc_multihead_layer
from models.fc.nets import MLP, MLP_gates
from models.conv.nets import ConvLayers
from models.cl.memory_buffer_stream import MemoryBuffer
from models.cl.continual_learner import ContinualLearner
from models.utils import loss_functions as lf, modules


class Classifier(ContinualLearner, MemoryBuffer):
    '''Model for classifying images, "enriched" as ContinualLearner- and MemoryBuffer-object.'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False, phantom=False,
                 # -how to use context-ID?
                 xdg_prob=0., n_contexts=5, multihead=False, device='cpu'):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.stream_classifier = True
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop
        self.phantom = phantom

        # for using context information
        self.xdg_prob = xdg_prob
        self.n_contexts = n_contexts
        self.multihead = multihead

        # for consolidation-operations, how often to update the model relative to which stay close
        self.update_every = 1

        # settings for training
        self.binaryCE = False             #-> use binary (instead of multiclass) prediction error
        self.binaryCE_distill = False     #-> for classes from previous contexts, use the by the previous model
                                          #   predicted probs as binary targets (only in Class-IL with binaryCE)

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        self.flatten = modules.Flatten()  # flatten image to 2D-tensor
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        #------------------------------------------------------------------------------------------#
        #--> fully connected hidden layers
        if self.xdg_prob>0.:
            self.fcE = MLP_gates(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers-1,
                                 hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                                 excitability=excitability, excit_buffer=excit_buffer,
                                 gate_size=n_contexts, gating_prop=xdg_prob, final_gate=True, device=device)
        else:
            self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers-1,
                           hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                           excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated, phantom=phantom)
        mlp_output_size = fc_units if fc_layers>1 else self.conv_out_units

        #--> classifier
        if self.multihead:
            self.classifier = fc_multihead_layer(mlp_output_size, classes, n_contexts,
                                                 excit_buffer=True, nl='none', drop=fc_drop, device=device)
        else:
            self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop,
                                       phantom=phantom)

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.conv_out_units, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def forward(self, x, context=None):
        # -if needed, convert [context] to one-hot vector
        if (self.xdg_prob>0. or self.multihead) and (context is not None) and (type(context)==np.ndarray or context.dim()<2):
            context_one_hot = lf.to_one_hot(context, classes=self.n_contexts, device=self._device())

        hidden = self.convE(x)
        flatten_x = self.flatten(hidden)
        final_features = self.fcE(flatten_x, context_one_hot) if self.xdg_prob>0. else self.fcE(flatten_x)
        out = self.classifier(final_features, context_one_hot) if self.multihead else self.classifier(final_features)
        return out


    def feature_extractor(self, images, context=None):
        # -if needed, convert [context] to one-hot vector
        if (self.xdg_prob>0. or self.multihead) and (context is not None) and (type(context)==np.ndarray or context.dim()<2):
            context_one_hot = lf.to_one_hot(context, classes=self.n_contexts, device=self._device())

        hidden = self.convE(images)
        flatten_x = self.flatten(hidden)
        final_features = self.fcE(flatten_x, context_one_hot) if self.xdg_prob>0. else self.fcE(flatten_x)
        return final_features


    def classify(self, x, context=None, no_prototypes=False):
        '''For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes].'''
        if self.prototypes and not no_prototypes:
            return self.classify_with_prototypes(x, context=context)
        else:
            return self.forward(x, context=context)


    def train_a_batch(self, x, y, c=None, x_=None, y_=None, c_=None, scores_=None, rnt=0.5, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <1D-tensor> batch of corresponding labels
        [c]               <1D-tensor> or <np.ndarray>; for each batch-element in [x] its context-ID  --OR--
                          <2D-tensor>; for each batch-element in [x] a probability for every context-ID
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [c_]
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new context
        '''

        # Set model to training-mode
        self.train()
        # -however, if some layers are frozen, they should be set to eval() to prevent batch-norm layers from changing
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()

        # Reset optimizer
        self.optimizer.zero_grad()


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # Run model
            y_hat = self(x_, c_)

            # Calculate losses
            predL_r, distilL_r = None, None
            if (y_ is not None) and (y_ is not None):
                if self.binaryCE:
                    binary_targets_ = lf.to_one_hot(y_.cpu(), y_hat.size(1)).to(y_.device)
                    predL_r = F.binary_cross_entropy_with_logits(
                        input=y_hat, target=binary_targets_, reduction='none'
                    ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                else:
                    predL_r = F.cross_entropy(y_hat, y_, reduction='mean')
            if (scores_ is not None) and (scores_ is not None):
                kd_fn = lf.loss_fn_kd_binary if self.binaryCE else lf.loss_fn_kd
                distilL_r = kd_fn(scores=y_hat, target_scores=scores_, T=self.KD_temp)

            # Weigh losses
            if self.replay_targets == "hard":
                loss_replay = predL_r
            elif self.replay_targets == "soft":
                loss_replay = distilL_r

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else loss_replay

        # If using the replayed loss as an inequality constraint, calculate and store averaged gradient of replayed data
        if self.use_replay in ('inequality', 'both') and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if self.use_replay == 'both':
                loss_replay = (1-rnt) * loss_replay
            loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.data.view(-1))
            grad_rep = torch.cat(grad_rep)
            # If gradients are only used as inequality constraint, reset them
            if self.use_replay=='inequality':
                self.optimizer.zero_grad()


        ##--(2)-- CURRENT DATA --##

        if x is not None:
            # Run model
            y_hat = self(x, c)

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = lf.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                predL = None if y is None else F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()     #--> sum over classes, then average over batch
            else:
                # -multiclass prediction loss
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-accuracy
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        else:
            accuracy = predL = None
            # -> it's possible there is only "replay" [i.e., for offline with incremental context learning]


        # Combine loss from current and replayed batch
        if x_ is None or self.use_replay=='inequality':
            loss_total = loss_cur
        elif self.use_replay=='both':
            # -if the replayed loss is both added to the current loss and used as inequality constraint,
            #  the gradients of the replayed loss are already backpropagated and accumulated
            loss_total = rnt*loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay


        ##--(3)-- PARAMETER REGULARIZATION LOSSES --##

        # Add a parameter regularization penalty to the loss function
        weight_penalty_loss = None
        if self.weight_penalty:
            if self.importance_weighting=='si':
                weight_penalty_loss = self.surrogate_loss()
            elif self.importance_weighting=='fisher':
                if self.fisher_kfac:
                    weight_penalty_loss = self.ewc_kfac_loss()
                else:
                    weight_penalty_loss = self.ewc_loss()
            loss_total += self.reg_strength * weight_penalty_loss


        ##--(4)-- COMPUTE (AND MANIPULATE) GRADIENTS --##

        # Backpropagate errors (for the part of the loss that has not yet been backpropagated)
        loss_total.backward()

        # A-GEM: check whether gradients to be used align with gradients of replayed data, project them if needed
        if self.use_replay in ('inequality', 'both') and x_ is not None:
            # -reorganize the gradients to be used for the optimization step as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                # -if violated, project the current gradient onto the gradient of the replayed batch ...
                length_rep = (grad_rep * grad_rep).sum()
                grad_proj = grad_cur - (angle / (length_rep + self.eps_agem)) * grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param


        ##--(5)-- TAKE THE OPTIMIZATION STEP --##
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': predL_r.item() if (x_ is not None and predL_r is not None) else 0,
            'distil_r': distilL_r.item() if (scores_ is not None and distilL_r is not None) else 0,
            'param_reg': weight_penalty_loss.item() if weight_penalty_loss is not None else 0,
            'accuracy': accuracy if accuracy is not None else 0.,
        }

