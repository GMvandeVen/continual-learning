from torch import nn
from models.classifier import Classifier


class SeparateClassifiers(nn.Module):
    '''Model for classifying images with a separate network for each context.'''

    def __init__(self, image_size, image_channels, classes_per_context, contexts,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False):

        # Configurations
        super().__init__()
        self.classes_per_context = classes_per_context
        self.contexts = contexts
        self.label = "SeparateClassifiers"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        # Define a separate network for each context to be learned
        for context_id in range(self.contexts):
            new_network = Classifier(
                image_size, image_channels, classes_per_context,
                # -conv-layers
                conv_type=conv_type, depth=depth, start_channels=start_channels, reducing_layers=reducing_layers,
                conv_bn=conv_bn, conv_nl=conv_nl, num_blocks=num_blocks, global_pooling=global_pooling, no_fnl=no_fnl,
                conv_gated=conv_gated,
                # -fc-layers
                fc_layers=fc_layers, fc_units=fc_units, fc_drop=fc_drop, fc_bn=fc_bn, fc_nl=fc_nl, fc_gated=fc_gated,
                bias=bias, excitability=excitability, excit_buffer=excit_buffer
            )
            setattr(self, 'context{}'.format(context_id+1), new_network)


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for context_id in range(self.contexts):
            list += getattr(self, 'context{}'.format(context_id+1)).list_init_layers()
        return list

    @property
    def name(self):
        return "SepNets-{}".format(self.context1.name)


    def train_a_batch(self, x, y, c=None, context=None, **kwargs):
        '''Train model for one batch ([x],[y]) from the indicated context.

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [c]               <1D-tensor> or <np.ndarray>; for each batch-element in [x] its context-ID
        [context]         <int> the context, can be used if all elements in [x] are from same context
        '''

        # Train the sub-network of the indicated context on this batch
        if context is not None:
            loss_dict = getattr(self, 'context{}'.format(context)).train_a_batch(x, y)
        else:
            for context_id in range(self.contexts):
                if context_id in c:
                    x_to_use = x[c == context_id]
                    y_to_use = y[c == context_id]
                    loss_dict = getattr(self, 'context{}'.format(context_id+1)).train_a_batch(x_to_use, y_to_use)
                    # NOTE: this way, only the [lost_dict] of the last context in the batch is returned

        # Return the dictionary with different training-loss split in categories
        return loss_dict

