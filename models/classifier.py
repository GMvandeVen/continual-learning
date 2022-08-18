import torch
from torch.nn import functional as F
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.conv.nets import ConvLayers
from models.cl.memory_buffer import MemoryBuffer
from models.cl.continual_learner import ContinualLearner
from models.utils import loss_functions as lf, modules
from models.utils.ncl import additive_nearest_kf


class Classifier(ContinualLearner, MemoryBuffer):
    '''Model for classifying images, "enriched" as ContinualLearner- and MemoryBuffer-object.'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False, phantom=False):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop
        self.phantom = phantom

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
        self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers-1,
                       hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated, phantom=phantom)
        mlp_output_size = fc_units if fc_layers>1 else self.conv_out_units
        #--> classifier
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


    def forward(self, x, return_intermediate=False):
        hidden = self.convE(x)
        flatten_x = self.flatten(hidden)
        if not return_intermediate:
            final_features = self.fcE(flatten_x)
        else:
            final_features, intermediate = self.fcE(flatten_x, return_intermediate=True)
            intermediate["classifier"] = final_features
        out = self.classifier(final_features)
        return (out, intermediate) if return_intermediate else out


    def feature_extractor(self, images):
        return self.fcE(self.flatten(self.convE(images)))

    def classify(self, x, allowed_classes=None, no_prototypes=False):
        '''For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes].'''
        if self.prototypes and not no_prototypes:
            return self.classify_with_prototypes(x, allowed_classes=allowed_classes)
        else:
            image_features = self.flatten(self.convE(x))
            hE = self.fcE(image_features)
            scores = self.classifier(hE)
            return scores if (allowed_classes is None) else scores[:, allowed_classes]


    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, context=1,
                      **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new context
        [active_classes]  None or (<list> of) <list> with "active" classes
        [context]         <int> context-ID, with first context labelled as '1' (e.g., for setting context-specific mask)
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

        # Should gradient be computed separately for each context? (needed when a context-mask is combined with replay)
        gradient_per_context = True if ((self.mask_dict is not None) and (x_ is not None)) else False


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # If there are different predictions per context, [y_] or [scores_] are lists and [x_] must be evaluated
            # separately on each of them (although [x_] could be a list as well!)
            PerContext = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not PerContext:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per context and there is no context-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous context
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per context, evaluate model on this context's replay
                if (type(x_)==list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(context=replay_id+1)
                    y_hat_all = self(x_temp_)

                # -if needed, remove predictions for classes not active in the replayed context
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    if self.binaryCE:
                        binary_targets_ = lf.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                        predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                            input=y_hat, target=binary_targets_, reduction='none'
                        ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                    else:
                        predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    kd_fn = lf.loss_fn_kd_binary if self.binaryCE else lf.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)

                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If needed, perform backward pass before next context-mask (gradients of all contexts will be accumulated)
                if gradient_per_context:
                    weight = 1. if self.use_replay=='inequality' else (1.-rnt)
                    weighted_replay_loss_this_context = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_context.backward()

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        if (x_ is not None) and self.lwf_weighting and (not self.scenario=='class'):
            loss_replay *= (context-1)

        # If using the replayed loss as an inequality constraint, calculate and store averaged gradient of replayed data
        if self.use_replay in ('inequality', 'both') and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_context:
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
            # If requested, apply correct context-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(context=context)

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not active in the current context
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = lf.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                if self.binaryCE_distill and (scores is not None):
                    # -replace targets for previously seen classes with predictions of previous model
                    binary_targets[:,:scores.size(1)] = torch.sigmoid(scores / self.KD_temp)
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
        elif gradient_per_context or self.use_replay=='both':
            # -if backward passes are performed per context (i.e., XdG combined with replay), or when the replayed loss
            #  is both added to the current loss and used as inequality constraint, the gradients of the replayed loss
            #  are already backpropagated and accumulated
            loss_total = rnt*loss_cur
        else:
            if self.lwf_weighting:
                loss_total = loss_replay if (x is None) else loss_cur+loss_replay
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
            # -check inequality constraint
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

        # Precondition gradient of current data using projection matrix constructed from parameter importance estimates
        if self.precondition:

            if self.importance_weighting=='fisher' and not self.fisher_kfac:
                #--> scale gradients by inverse diagonal Fisher
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            # Retrieve prior fisher matrix
                            n = n.replace(".", "__")
                            fisher = getattr(self, "{}_EWC_estimated_fisher{}".format(n, "" if self.online else context))
                            # Scale loss landscape by inverse prior fisher and divide learning rate by data size
                            scale = (fisher + self.alpha**2) ** (-1)
                            p.grad *= scale  # scale lr by inverse prior information
                            p.grad /= self.data_size  # scale lr by prior (necessary for stability in 1st context)

            elif self.importance_weighting=='fisher' and self.fisher_kfac:
                #--> scale gradients by inverse Fisher kronecker factors
                def scale_grad(label, layer):
                    assert isinstance(layer, fc_layer)
                    info = self.KFAC_FISHER_INFO[label]  # get previous KFAC fisher
                    A = info["A"].to(self._device())
                    G = info["G"].to(self._device())
                    linear = layer.linear
                    if linear.bias is not None:
                        g = torch.cat( (linear.weight.grad, linear.bias.grad[..., None]), -1).clone()
                    else:
                        g = layer.linear.weight.grad.clone()

                    assert g.shape[-1] == A.shape[-1]
                    assert g.shape[-2] == G.shape[-2]
                    iA = torch.eye(A.shape[0]).to(self._device()) * (self.alpha)
                    iG = torch.eye(G.shape[0]).to(self._device()) * (self.alpha)

                    As, Gs = additive_nearest_kf({"A": A, "G": G}, {"A": iA, "G": iG})  # kronecker sums
                    Ainv = torch.inverse(As)
                    Ginv = torch.inverse(Gs)

                    scaled_g = Ginv @ g @ Ainv
                    if linear.bias is not None:
                        linear.weight.grad = scaled_g[..., 0:-1].detach() / self.data_size
                        linear.bias.grad = scaled_g[..., -1].detach() / self.data_size
                    else:
                        linear.weight.grad = scaled_g[..., 0:-1, :] / self.data_size

                    # make sure to reset all phantom to have no zeros
                    if not hasattr(layer, 'phantom'):
                        raise ValueError(f"Layer {label} does not have phantom parameters")
                    # make sure phantom stays zero
                    layer.phantom.grad.zero_()
                    layer.phantom.data.zero_()

                scale_grad("classifier", self.classifier)
                for i in range(1, self.fcE.layers + 1):
                    label = f"fcLayer{i}"
                    scale_grad(label, getattr(self.fcE, label))

            elif self.importance_weighting=='owm' and context>1:
                def scale_grad(label, layer):
                    info = self.KFAC_FISHER_INFO[label]  # get previous KFAC fisher
                    A = info['A'].to(self._device())

                    linear = layer.linear
                    if linear.bias is not None:
                        g = torch.cat((linear.weight.grad, linear.bias.grad[..., None]), -1).clone()
                    else:
                        g = layer.linear.weight.grad.clone()

                    assert (g.shape[-1] == A.shape[-1])
                    iA = torch.eye(A.shape[0]).to(self._device())  # * (self.alpha)
                    As = A / self.alpha + iA
                    Ainv = torch.inverse(As)
                    scaled_g = g @ Ainv

                    if linear.bias is not None:
                        linear.weight.grad = scaled_g[..., 0:-1].detach()
                        linear.bias.grad = scaled_g[..., -1].detach()
                    else:
                        linear.weight.grad = scaled_g[..., 0:-1, :]

                scale_grad('classifier', self.classifier)
                for i in range(1, self.fcE.layers + 1):
                    label = f"fcLayer{i}"
                    scale_grad(label, getattr(self.fcE, label))


        ##--(5)-- TAKE THE OPTIMIZATION STEP --##
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'param_reg': weight_penalty_loss.item() if weight_penalty_loss is not None else 0,
            'accuracy': accuracy if accuracy is not None else 0.,
        }

