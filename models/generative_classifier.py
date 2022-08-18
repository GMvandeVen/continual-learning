import numpy as np
import torch
from torch import nn
from models.vae import VAE



class GenerativeClassifier(nn.Module):
    """Class for generative classifier with separate VAE for each class to be learned."""

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=False, fc_gated=False,
                 # -prior
                 z_dim=20, prior="standard", n_modes=1,
                 # -decoder
                 recon_loss='BCE', network_output="sigmoid", deconv_type="standard"):

        # Set configurations for setting up the model
        super().__init__()
        self.classes = classes
        self.label = "GenClassifier"

        # Atributes defining how to do inference
        self.S = "mean"  # "mean": use [z_mu] as single (importance) sample; <int>: use this much (importance) samples
        self.importance = True
        self.from_latent = False

        # Define a VAE for each class to be learned
        for class_id in range(classes):
            new_vae = VAE(image_size, image_channels,
                          # -conv-layers
                          conv_type=conv_type, depth=depth, start_channels=start_channels,
                          reducing_layers=reducing_layers, conv_bn=conv_bn, conv_nl=conv_nl,
                          num_blocks=num_blocks, global_pooling=global_pooling, no_fnl=no_fnl,
                          conv_gated=conv_gated,
                          # -fc-layers
                          fc_layers=fc_layers, fc_units=fc_units, fc_drop=fc_drop, fc_bn=fc_bn,
                          fc_nl=fc_nl, excit_buffer=excit_buffer, fc_gated=fc_gated,
                          # -prior
                          z_dim=z_dim, prior=prior, n_modes=n_modes,
                          # -decoder
                          recon_loss=recon_loss, network_output=network_output, deconv_type=deconv_type)
            setattr(self, "vae{}".format(class_id), new_vae)


    ##------ NAMES --------##

    def get_name(self):
        return "x{}-{}".format(self.classes, self.vae0.get_name())

    @property
    def name(self):
        return self.get_name()


    ##------ UTILITIES --------##

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, only_x=True, class_id=None, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device.'''

        for sample_id in range(size):
            # sample from which class-specific VAE to sample
            selected_class_id = np.random.randint(0, self.classes, 1)[0] if class_id is None else class_id
            model_to_sample_from = getattr(self, 'vae{}'.format(selected_class_id))

            # sample from that VAE
            new_sample = model_to_sample_from.sample(1)

            # concatanate generated X (and y)
            X = torch.cat([X, new_sample], dim=0) if sample_id>0 else new_sample
            if not only_x:
                y = torch.cat([y, torch.LongTensor([selected_class_id]).to(self._device())]) if (
                    sample_id>0
                ) else torch.LongTensor([selected_class_id]).to(self._device())

        # return samples as [size]x[channels]x[image_size]x[image_size] tensor (and labels as [size] tensor)
        return X if only_x else (X, y)


    ##------ CLASSIFICATION FUNCTIONS --------##

    def classify(self, x, allowed_classes=None, **kwargs):
        '''Given an input [x], get the scores based on [self.S] importance samples (if self.S=='mean', use [z_mu]).

        Input:  - [x]        <4D-tensor> of shape [batch]x[channels]x[image_size]x[image_size]

        Output: - [scores]   <2D-tensor> of shape [batch]x[allowed_classes]
        '''
        # If not provided, set [allowed_classes] to all possible classes
        if allowed_classes is None:
            allowed_classes = list(range(self.classes))
        # For each possible class, compute its 'score' (i.e., likelihood of input under generative model of that class)
        scores = torch.zeros([x.size(0), len(allowed_classes)], dtype=torch.float32, device=self._device())
        for class_id in allowed_classes:
            if self.from_latent:
                scores[:,class_id] = getattr(self, 'vae{}'.format(class_id)).get_latent_lls(x)
            else:
                scores[:,class_id] = getattr(self, 'vae{}'.format(class_id)).estimate_lls(
                    x, S=self.S, importance=self.importance
                )
        return scores