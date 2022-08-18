import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from models.utils import modules
from models.conv.nets import ConvLayers
from models.fc.layers import fc_layer


class FeatureExtractor(torch.nn.Module):
    '''Model for encoding (i.e., feature extraction) and images.'''

    def __init__(self, image_size, image_channels,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False):

        # Model configurations
        super().__init__()
        self.label = "FeatureExtractor"
        self.depth = depth

        # Optimizer (needs to be set before training starts))
        self.optim_type = None
        self.optimizer = None
        self.optim_list = []

        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels

    @property
    def name(self):
        return self.convE.name

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.convE.list_init_layers()
        return list

    def forward(self, x):
        return self.convE(x)

    def train_discriminatively(self, train_loader, iters, classes, lr=0.001, optimizer='adam'):
        '''Train the feature extractor for [iters] iterations on data from [train_loader].

        [model]             model to optimize
        [train_loader]      <dataloader> for training [model] on
        [iters]             <int> (max) number of iterations (i.e., batches) to train for
        [classes]           <int> number of possible clasess (softmax layer with that many units will be added to model)
        '''

        # Create (temporary) classification output layer
        self.flatten = modules.Flatten()
        self.classifier = fc_layer(self.conv_out_units, classes, excit_buffer=True, nl='none').to(self._device())

        # Define optimizer
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': lr},]
        self.optimizer = optim.SGD(optim_list) if optimizer=="sgd" else optim.Adam(optim_list, betas=(0.9, 0.999))

        # Set model to training-mode
        self.train()

        # Create progress-bar (with manual control)
        bar = tqdm.tqdm(total=iters)

        iteration = epoch = 0
        while iteration < iters:
            epoch += 1

            # Loop over all batches of an epoch
            for batch_idx, (data, y) in enumerate(train_loader):
                iteration += 1

                # Reset optimizer
                self.optimizer.zero_grad()

                # Prepare data
                data, y = data.to(self._device()), y.to(self._device())

                # Run model
                features = self(data)
                y_hat = self.classifier(self.flatten(features))

                # Calculate loss
                loss = F.cross_entropy(input=y_hat, target=y, reduction='mean')

                # Calculate training-accuracy
                accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / data.size(0)

                # Backpropagate errors
                loss.backward()

                # Take optimization-step
                self.optimizer.step()

                # Update progress bar
                bar.set_description(
                    ' <FEAUTRE EXTRACTOR> | training loss: {loss:.3} | training accuracy: {prec:.3} |'.format(
                        loss=loss.cpu().item(), prec=accuracy
                    )
                )
                bar.update(1)

                # Break if max-number of iterations is reached
                if iteration == iters:
                    bar.close()
                    break
