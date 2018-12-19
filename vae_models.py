import torch
from torch import nn
from torch.nn import functional as F
import excitability_modules as eM
import utils
import linear_nets
from replayer import Replayer


class AutoEncoder(Replayer):
    """Class for variational auto-encoder (VAE) models."""

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", z_dim=20):

        # Set configurations
        super().__init__()
        self.label = "VAE"
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.fc_units = fc_units

        # Training related components that should be set before training
        # -criterion for reconstruction
        self.recon_criterion = None
        # -weigths of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.
        self.lamda_pl = 0. # --> when used as "classifier with feedback-connections", this should be set to 1.

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")


        ######------SPECIFY MODEL------######

        # encoder: flatten image to 2D-tensor
        self.flatten = utils.Flatten()
        # encoder: fully connected hidden layers
        self.fcE = linear_nets.MLP(
            input_size=image_channels*image_size**2, output_size=fc_units, layers=fc_layers-1, hid_size=fc_units,
            drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, final_nl=True,
        )
        enc_mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2

        # classifier (from final hidden layer of encoder)
        self.classifier = nn.Sequential(nn.Dropout(fc_drop),
                                        eM.LinearExcitability(enc_mlp_output_size, classes))

        # reparametrization ("to Z and back")
        out_nl = True if fc_layers>1 else False
        dec_mlp_input_size = fc_units if fc_layers>1 else image_channels*image_size**2
        self.toZ = nn.Linear(enc_mlp_output_size, z_dim)       # estimating mean
        self.toZlogvar = nn.Linear(enc_mlp_output_size, z_dim) # estimating log(SD**2)
        self.fromZ = linear_nets.fc_layer(z_dim, dec_mlp_input_size, batch_norm=(out_nl and fc_bn),
                                         nl=fc_nl if out_nl else "none")

        # decoder: fully connected hidden layers (with no non-linearity or batchnorm in final layer!)
        self.fcD = linear_nets.MLP(
            input_size=fc_units, output_size=image_channels*image_size**2, layers=fc_layers-1, hid_size=fc_units,
            drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, final_nl=False,
        )
        # decoder: reshape to image
        self.reshapeD = utils.ToImage(image_channels=image_channels)

    @property
    def name(self):
        fc_label = "{}--".format(self.fcE.name) if self.fc_layers>1 else ""
        hid_label = "{}{}-".format("i", self.image_channels*self.image_size**2) if self.fc_layers==1 else ""
        z_label = "z{}".format(self.z_dim)
        return "{}({}{}{}-c{})".format(self.label, fc_label, hid_label, z_label, self.classes)


    def encode(self, x):
        '''Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar].'''
        # extract final hidden features (forward-pass)
        hE = self.fcE(self.flatten(x))
        # get parameters for reparametrization
        z_mean = self.toZ(hE)
        z_logvar = self.toZlogvar(hE)
        return z_mean, z_logvar, hE

    def classify(self, x):
        '''For input [x], return all predicted "scores"/"logits".'''
        hE = self.fcE(self.flatten(x))
        y_hat = self.classifier(hE)
        return y_hat

    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        '''Pass latent variable activations through feedback connections, to give reconstructed image [image_recon].'''
        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.reshapeD(image_features)
        return image_recon

    def forward(self, x, full=False):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input:  - [x]   <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x]
        - [y_hat]       <2D-tensor> with predicted logits for each class
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction
        If [full] is False, output is simply the predicted logits (i.e., [y_hat]).'''
        if full:
            # encode (forward), reparameterize and decode (backward)
            mu, logvar, hE = self.encode(x)
            z = self.reparameterize(mu, logvar) if self.training else mu
            x_recon = self.decode(z)
            # classify
            y_hat = self.classifier(hE)
            # return
            return (x_recon, y_hat, mu, logvar, z)
        else:
            return self.classify(x) # -> if [full]=False, only forward pass for prediction


    def sample(self, size, allowed_predictions=None, return_scores=False):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_predictions] <list> of [class_ids] which are allowed to be predicted
                - [return_scores]       <bool>; if True, [y_hat] is also returned

        OUTPUT: - [X]     <4D-tensor> generated images
                - [y]     <1D-tensor> predicted corresponding labels
                - [y_hat] <2D-tensor> predicted "logits"/"scores" for all [allowed_predictions]'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # sample z
        z = torch.randn(size, self.z_dim)
        z = z.to(self._device())

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z)

        # use forward model to predict scores of generated images
        with torch.no_grad():
            y_hat = self.classify(X)
        y_hat = y_hat[:, allowed_predictions] if (allowed_predictions is not None) else y_hat
        # get "predicted" class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus classes-labels
        return (X, y, y_hat) if return_scores else (X, y)


    def loss_function(self, recon_x, x, y_hat=None, y_target=None, scores=None, mu=None, logvar=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x_recon]         <4D-tensor> reconstructed image in same shape as [x]
                - [x]               <4D-tensor> original image
                - [y_hat]           <2D-tensor> with predicted "logits" for each class
                - [y_target]        <1D-tensor> with target-classes (as integers)
                - [scores]          <2D-tensor> with target "logits" for each class
                - [mu]              <2D-tensor> with either [z] or the estimated mean of [z]
                - [logvar]          None or <2D-tensor> with estimated log(SD^2) of [z]

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how normally distributed [z] is"
                - [predL]        prediction loss indicating how well targets [y] are predicted
                - [distilL]      knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                     match the target "logits" ([scores])'''

        batch_size = x.size(0)

        ###-----Reconstruction loss-----###
        reconL = self.recon_criterion(recon_x.view(batch_size, -1), x.view(batch_size, -1))

        ###-----Variational loss-----###
        if logvar is not None:
            #---- see Appendix B from: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            # -normalise by same number of elements as in reconstruction
            variatL /= (self.image_channels * self.image_size ** 2)
            # --> because self.recon_criterion averages over batch-size but also over all pixels/elements in recon!!
        else:
            variatL = torch.tensor(0., device=self._device())

        ###-----Prediction loss-----###
        if y_target is not None:
            predL = F.cross_entropy(y_hat, y_target, size_average=True)
        else:
            predL = torch.tensor(0., device=self._device())

        ###-----Distilliation loss-----###
        if scores is not None:
            n_classes_to_consider = y_hat.size(1)  #--> zeroes will be added to [scores] to make its size match [y_hat]
            distilL = utils.loss_fn_kd(scores=y_hat[:, :n_classes_to_consider], target_scores=scores, T=self.KD_temp)
        else:
            distilL = torch.tensor(0., device=self._device())

        # Return a tuple of the calculated losses
        return reconL, variatL, predL, distilL



    def train_a_batch(self, x, y, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes'''

        # Set model to training-mode
        self.train()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x is not None:
            # Run the model
            recon_batch, y_hat, mu, logvar, z = self(x, full=True)
            # If needed (e.g., incremental or multihead set-up), remove predictions for classes not in current task
            if active_classes is not None:
                y_hat = y_hat[:, active_classes[-1]] if type(active_classes[0])==list else y_hat[:, active_classes]
            # Calculate all losses
            reconL, variatL, predL, _ = self.loss_function(recon_x=recon_batch, x=x, y_hat=y_hat,
                                                           y_target=y, mu=mu, logvar=logvar)
            # Weigh losses as requested
            loss = self.lamda_rcl*reconL + self.lamda_vl*variatL + self.lamda_pl*predL

            # Calculate training-precision
            if y is not None:
                _, predicted = y_hat.max(1)
                precision = (y == predicted).sum().item() / x.size(0)

        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:
            # If [x_] is a list, perform separate replay for each entry
            n_replays = len(x_) if type(x_)==list else 1
            if not type(x_)==list:
                x_ = [x_]
                y_ = [y_]
                scores_ = [scores_]
                if active_classes is not None:
                    active_classes = [active_classes]

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            reconL_r = [None]*n_replays
            variatL_r = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Loop to perform each replay
            for replay_id in range(n_replays):
                # Run the model
                recon_batch, y_hat, mu, logvar, z = self(x_[replay_id], full=True)
                # If needed (e.g., incremental or multihead set-up), remove predictions for classes not in replayed task
                if active_classes is not None:
                    y_hat = y_hat[:, active_classes[replay_id]]
                # Calculate all losses
                reconL_r[replay_id], variatL_r[replay_id], predL_r[replay_id], distilL_r[replay_id] = self.loss_function(
                    recon_x=recon_batch, x=x_[replay_id], y_hat=y_hat,
                    y_target=y_[replay_id] if (y_ is not None) else None,
                    scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, logvar=logvar,
                )
                # Weigh losses as requested
                loss_replay[replay_id] = self.lamda_rcl*reconL_r[replay_id] + self.lamda_vl*variatL_r[replay_id]
                if self.replay_targets=="hard":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] += self.lamda_pl*distilL_r[replay_id]

            # Calculate total loss
            if x is not None:
                loss_total = rnt*loss + (1-rnt)*sum(loss_replay)/n_replays
            else:
                loss_total = sum(loss_replay)/n_replays
        else:
            loss_total = loss


        # Reset optimizer
        self.optimizer.zero_grad()
        # Backpropagate errors
        loss_total.backward()
        # Take optimization-step
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(), 'precision': precision,
            'recon': reconL.item() if x is not None else 0,
            'variat': variatL.item() if x is not None else 0,
            'pred': predL.item() if x is not None else 0,
            'recon_r': sum(reconL_r).item()/n_replays if x_ is not None else 0,
            'variat_r': sum(variatL_r).item()/n_replays if x_ is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
        }



