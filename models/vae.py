import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.fc.layers import fc_layer,fc_layer_split
from models.fc.nets import MLP
from models.conv.nets import ConvLayers,DeconvLayers
from models.cl.continual_learner import ContinualLearner
from models.utils import loss_functions as lf, modules


class VAE(ContinualLearner):
    """Class for variational auto-encoder (VAE) model."""

    def __init__(self, image_size, image_channels,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl="relu", fc_gated=False, excit_buffer=False,
                 # -prior
                 prior="standard", z_dim=20, n_modes=1,
                 # -decoder
                 recon_loss='BCE', network_output="sigmoid", deconv_type="standard", **kwargs):
        '''Class for variational auto-encoder (VAE) models.'''

        # Set configurations for setting up the model
        super().__init__()
        self.label = "VAE"
        self.image_size = image_size
        self.image_channels = image_channels
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth
        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss # options: BCE|MSE
        self.network_output = network_output

        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # Prior-related parameters
        self.prior = prior
        self.n_modes = n_modes

        # Weigths of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.

        self.average = True #--> makes that [reconL] and [variatL] are both divided by number of input-pixels

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")


        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = ConvLayers(conv_type=conv_type, block_type="basic", num_blocks=num_blocks,
                                image_channels=image_channels, depth=self.depth, start_channels=start_channels,
                                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
                                output="none" if no_fnl else "normal", global_pooling=global_pooling,
                                gated=conv_gated)
        # -flatten image to 2D-tensor
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        #------------------------------------------------------------------------------------------#
        # -fully connected hidden layers
        self.fcE = MLP(input_size=self.conv_out_units, output_size=fc_units, layers=fc_layers-1,
                       hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, gated=fc_gated,
                       excit_buffer=excit_buffer)
        mlp_output_size = fc_units if fc_layers > 1 else self.conv_out_units
        # -to z
        self.toZ = fc_layer_split(mlp_output_size, z_dim, nl_mean='none', nl_logvar='none')

        ##>----Decoder (= p[x|z])----<##
        out_nl = True if fc_layers > 1 else (True if (self.depth > 0 and not no_fnl) else False)
        real_h_dim_down = fc_units if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none")
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        self.fcD = MLP(input_size=fc_units, output_size=self.convE.out_units(image_size, ignore_gp=True),
                       layers=fc_layers-1, hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                       gated=fc_gated, output=self.network_output if self.depth==0 else 'normal')
        # to image-shape
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth>0 else image_channels)
        # through deconv-layers
        self.convD = DeconvLayers(
            image_channels=image_channels, final_channels=start_channels, depth=self.depth,
            reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated,
            output=self.network_output, deconv_type=deconv_type,
        )

        ##>----Prior----<##
        # -if using the GMM-prior, add its parameters
        if self.prior=="GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False



    ##------ NAMES --------##

    def get_name(self):
        convE_label = "{}--".format(self.convE.name) if self.depth>0 else ""
        fcE_label = "{}--".format(self.fcE.name) if self.fc_layers>1 else "{}{}-".format("h" if self.depth>0 else "i",
                                                                                          self.conv_out_units)
        z_label = "z{}{}".format(self.z_dim, "" if self.prior=="standard" else "-{}{}".format(self.prior, self.n_modes))
        decoder_label = "--{}".format(self.network_output)
        return "{}={}{}{}{}".format(self.label, convE_label, fcE_label, z_label, decoder_label)

    @property
    def name(self):
        return self.get_name()



    ##------ LAYERS --------##

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        '''Return list with shape of all hidden layers.'''
        # create list with hidden convolutional layers
        layer_list = self.convE.layer_info(image_size=self.image_size)
        # add output of final convolutional layer (if there was at least one conv-layer and there's fc-layers after)
        if (self.fc_layers>0 and self.depth>0):
            layer_list.append([self.conv_out_channels, self.conv_out_size, self.conv_out_size])
        # add layers of the MLP
        if self.fc_layers>1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list



    ##------ FORWARD FUNCTIONS --------##

    def encode(self, x):
        '''Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE].'''
        # Forward-pass through conv-layers
        hidden_x = self.convE(x)
        image_features = self.flatten(hidden_x)
        # Forward-pass through fc-layers
        hE = self.fcE(image_features)
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE, hidden_x

    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()#.requires_grad_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        '''Decode latent variable activations [z] (=<2D-tensor>) into [image_recon] (=<4D-tensor>).'''
        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, full=False, reparameterize=True, **kwargs):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction
        If [full] is False, output is the reconstructed image (i.e., [x_recon]).
        '''
        # -encode (forward), reparameterize and decode (backward)
        mu, logvar, hE, hidden_x = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        x_recon = self.decode(z)
        # -return
        return (x_recon, mu, logvar, z) if full else x_recon

    def feature_extractor(self, images):
        '''Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images.'''
        return self.fcE(self.flatten(self.convE(images)))



    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, sample_mode=None, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [sample_mode]   <int> to sample from specific mode of [z]-distribution

        OUTPUT: - [X]             <4D-tensor> generated images / image-features'''

        # set model to eval()-mode
        self.eval()

        # pick for each sample the prior-mode to be used
        if self.prior=="GMM":
            if sample_mode is None:
                # -randomly sample modes from all possible modes
                sampled_modes = np.random.randint(0, self.n_modes, size)
            else:
                # -always sample from the provided mode
                sampled_modes = np.repeat(sample_mode, size)

        # sample z
        if self.prior=="GMM":
            prior_means = self.z_class_means
            prior_logvars = self.z_class_logvars
            # -for each sample to be generated, select the previously sampled mode
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim).to(self._device())

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor
        return X



    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, average=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''

        batch_size = x.size(0)
        if self.recon_loss=="MSE":
            # reconL = F.mse_loss(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            # reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss=="BCE":
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                            reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError("Wrong choice for type of reconstruction-loss!")
        # --> if [average]=True, reconstruction loss is averaged over all pixels/elements (otherwise it is summed)
        #       (averaging over all elements in the batch will be done later)
        return reconL


    def calculate_log_p_z(self, z):
        '''Calculate log-likelihood of sampled [z] under the prior distribution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            log_p_z = lf.log_Normal_standard(z, average=False, dim=1)  # [batch_size]

        if self.prior == "GMM":
            ## Get [means] and [logvars] of all (possible) modes
            allowed_modes = list(range(self.n_modes))
            # -calculate/retireve the means and logvars for the selected modes
            prior_means = self.z_class_means[allowed_modes, :]
            prior_logvars = self.z_class_logvars[allowed_modes, :]
            # -rearrange / select for each batch prior-modes to be used
            z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
            means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
            logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            n_modes = len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            # --> for each element in batch, calculate log-likelihood for all pseudoinputs: [batch_size] x [n_modes]
            a_max, _ = torch.max(a, dim=1)  # [batch_size]
            # --> for each element in batch, take highest log-likelihood over all pseudoinputs
            #     this is calculated and used to avoid underflow in the below computation
            a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
            a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))  # -> sum over modes: [batch_size]
            log_p_z = a_logsum + a_max  # [batch_size]

        return log_p_z


    def calculate_variat_loss(self, z, mu, logvar):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            # --> calculate analytically
            # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        elif self.prior=="GMM":
            # --> calculate "by estimation"

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            log_p_z = self.calculate_log_p_z(z)
            # ----->  log_p_z: [batch_size]

            ## Calculate "log_q_z_x" (entropy of "reparameterized" [z] given [x])
            log_q_z_x = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
            # ----->  log_q_z_x: [batch_size]

            ## Combine
            variatL = -(log_p_z - log_q_z_x)

        return variatL


    def loss_function(self, x, x_recon, mu, z, logvar=None, batch_weights=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         None or <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
        '''

        ###-----Reconstruction loss-----###
        batch_size = x.size(0)
        x_recon = (x_recon[0].view(batch_size, -1), x_recon[1].view(batch_size, -1)) if self.network_output=='split' else x_recon.view(batch_size, -1)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True, x_recon=x_recon)#-average over pixels
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)                         #-average over batch

        ###-----Variational loss-----###
        if logvar is not None:
            variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar)
            variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)  # -> average over batch
            variatL /= (self.image_channels * self.image_size ** 2)               # -> divide by # of input-pixels
        else:
            variatL = torch.tensor(0., device=self._device())

        # Return a tuple of the calculated losses
        return reconL, variatL



    ##------ EVALUATION FUNCTIONS --------##

    def get_latent_lls(self, x):
        '''Encode [x] as [z!x] and return log-likelihood.

        Input:  - [x]              <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]

        Output: - [log_likelihood] <1D-tensor> of shape [batch]
        '''

        # Run forward pass of model to get [z_mu] and [z_logvar]
        z_mu, z_logvar, _, _ = self.encode(x)

        # Calculate log_p_z
        log_p_z = self.calculate_log_p_z(z_mu)

        ## NOTE: we could additionally use [z_logvar] and compute KL-divergence with prior?
        return log_p_z


    def estimate_lls(self, x, S='mean', importance=True):
        '''Estimate log-likelihood for [x] using [S] importance samples (or Monte Carlo samples, if [importance]=False).

        Input:  - [x]              <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]
                - [S]              <int> (= # importance samples) or 'mean' (= use [z_mu] as single importance sample)
                - [importance]     <bool> if True do importance sampling, otherwise do Monte Carlo sampling

        Output: - [log_likelihood] <1D-tensor> of shape [batch]
        '''
        # Run forward pass of model to get [z_mu] and [z_logvar]
        if importance:
            z_mu, z_logvar, _, _ = self.encode(x)

        if S=='mean':
            if importance:
                # -->  Use [z_mu] as a 'single importance sample'
                # Calculate log_p_z
                log_p_z = self.calculate_log_p_z(z_mu)
                # Calculate log_q_z_x
                z_mu_dummy = torch.zeros_like(z_mu)  # to avoid unnecessary gradient tracking in next computation
                log_q_z_x = lf.log_Normal_diag(z_mu_dummy, mean=z_mu_dummy, log_var=z_logvar, average=False, dim=1)
            else:
                # -->  Use the overall prior mean as a 'single Monte Carlo sample'
                if self.prior=="GMM":
                    sampled_modes = np.random.randint(0, self.n_modes, x.size(0))
                    z_mu = self.z_class_means[sampled_modes, :]
                    ## NOTE: if using a GMM-prior with multiple modes, this does not really make sense!!!
                else:
                    z_mu = torch.zeros(x.size(0), self.z_dim).to(self._device())
            # Calcuate p_x_z
            # -reconstruct input
            x_recon = self.decode(z_mu)
            # -calculate p_x_z (under Gaussian observation model with unit variance)
            log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)
            # Calculate log-likelihood
            log_likelihood = (log_p_x_z + log_p_z - log_q_z_x) if importance else log_p_x_z
        else:
            #--> Use [S] importance/Monte Carlo samples
            # Define tensor in which to store the log-likelihoods of each sample
            all_lls = torch.zeros([S, x.size(0)], dtype=torch.float32, device=self._device())
            # For each sample, calculate log_likelihood
            for s_id in range(S):
                if importance:
                    # Reparameterize (i.e., sample z_s)
                    z = self.reparameterize(z_mu, z_logvar)
                    # Calculate log_p_z
                    log_p_z = self.calculate_log_p_z(z)
                    # Calculate log_q_z_x
                    log_q_z_x = lf.log_Normal_diag(z, mean=z_mu, log_var=z_logvar, average=False, dim=1)
                else:
                    # Sample z_s
                    if self.prior == "GMM":
                        # -randomly pick for each sample the prior-mode to be used
                        sampled_modes = np.random.randint(0, self.n_modes, x.size(0))
                        # -for each sample to be generated, select the mean & logvar for the sampled mode
                        z_means = self.z_class_means[sampled_modes, :]
                        z_logvars = self.z_class_logvars[sampled_modes, :]
                        # -sample using the selected means & logvars
                        z = self.reparameterize(z_means, z_logvars)
                    else:
                        # -sample from standard normal distribution
                        z = torch.randn(x.size(0), self.z_dim).to(self._device())
                # Calcuate p_x_z
                # -reconstruct input
                x_recon = self.decode(z)
                # -calculate p_x_z (under Gaussian observation model with unit variance)
                log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)
                # Calculate log-likelihoods for this importance sample
                all_lls[s_id] = (log_p_x_z + log_p_z - log_q_z_x) if importance else log_p_x_z
            # Calculate average log-likelihood over all (importance) samples for this test sample
            #  (for this, convert log-likelihoods back to likelihoods before summing them!)
            log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)
        return log_likelihood



    ##------ TRAINING FUNCTIONS --------##

    def train_a_batch(self, x, x_=None, rnt=0.5, **kwargs):
        '''Train model for one batch ([x]), possibly supplemented with replayed data ([x_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
        [rnt]               <number> in [0,1], relative importance of new context
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

        ##--(1)-- CURRENT DATA --##
        if x is not None:
            # Run the model
            recon_batch, mu, logvar, z = self(x, full=True, reparameterize=True)

            # Calculate losses
            reconL, variatL = self.loss_function(x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

            # Weigh losses as requested
            loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL

        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:
            # Run the model
            recon_batch, mu, logvar, z = self(x_, full=True, reparameterize=True)

            # Calculate losses
            reconL_r, variatL_r = self.loss_function(x=x_, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

            # Weigh losses as requested
            loss_replay = self.lamda_rcl*reconL_r + self.lamda_vl*variatL_r

        # Calculate total loss
        loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)

        # Backpropagate errors
        loss_total.backward()
        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'recon': reconL.item() if x is not None else 0,
            'variat': variatL.item() if x is not None else 0,
            'recon_r': reconL_r.item() if x_ is not None else 0,
            'variat_r': variatL_r.item() if x_ is not None else 0,
        }
