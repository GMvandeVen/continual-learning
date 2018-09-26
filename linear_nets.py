from torch import nn
import numpy as np
import utils
import excitability_modules as em


class fc_layer(nn.Module):
    '''Module for a fully connected layer.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size,
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, nl="relu"):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        self.bn = nn.BatchNorm1d(out_size) if batch_norm else utils.Identity()
        self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else utils.Identity())

    def forward(self, x):
        pre_activ = self.bn(self.linear(self.dropout(x)))
        output = self.nl(pre_activ)
        return output


class MLP(nn.Module):
    '''Modeule for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, size_per_layer=None,
                 drop=0, batch_norm=True, nl="relu", bias=True, excitability=False, excit_buffer=False, final_nl=True):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_size], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers] and [hid_size]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [final_nl]         <bool>; if False, final layer has no non-linearity and batchnorm'''

        super().__init__()

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{fnl}".format(
            drop="" if drop==0 else "-drop_{}".format(drop),
            bias="" if bias else "-noBias", exc="-exc" if excitability else "", bn="-bn" if batch_norm else "",
            nl="-lr" if nl=="leakyrelu" else "", fnl="-nfnl" if not final_nl else "",
        )
        # -set label
        self.label = "MLP({}{})".format(size_per_layer, nd_label) if self.layers>0 else ""

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            layer = fc_layer(
                in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop,
                batch_norm=False if (lay_id==self.layers and not final_nl) else batch_norm,
                nl="no" if (lay_id==self.layers and not final_nl) else nl
            )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = utils.Identity()

    def forward(self, x):
        for lay_id in range(1, self.layers+1):
            x = getattr(self, 'fcLayer{}'.format(lay_id))(x)
        return x

    @property
    def name(self):
        return self.label