from torch import nn
import numpy as np
import models.conv.layers as conv_layers
from models.utils import modules


class ConvLayers(nn.Module):
    '''Convolutional feature extractor model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [image_channels] x [image_size] x [image_size] tensor
    Output: [batch_size] x [out_channels] x [out_size] x [out_size] tensor
                - with [out_channels] = [start_channels] x 2**[reducing_layers] x [block.expansion]
                       [out_size] = [image_size] / 2**[reducing_layers]'''

    def __init__(self, conv_type="standard", block_type="basic", num_blocks=2,
                 image_channels=3, depth=5, start_channels=16, reducing_layers=None, batch_norm=True, nl="relu",
                 output="normal", global_pooling=False, gated=False):
        '''Initialize stacked convolutional layers (either "standard" or "res-net" ones--1st layer is always standard).

        [conv_type]         <str> type of conv-layers to be used: [standard|resnet]
        [block_type]        <str> block-type to be used: [basic|bottleneck] (only relevant if [type]=resNet)
        [num_blocks]        <int> or <list> (with len=[depth]-1) of # blocks in each layer
        [image_channels]    <int> # channels of input image to encode
        [depth]             <int> # layers
        [start_channels]    <int> # channels in 1st layer, doubled in every "rl" (=reducing layer)
        [reducing_layers]   <int> # layers in which image-size is halved & # channels doubled (default=[depth]-1)
                                      ("rl"'s are the last conv-layers; in 1st layer # channels cannot double)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> non-linearity to be used: [relu|leakyrelu]
        [output]            <str>  if - "normal", final layer is same as all others
                                      - "none", final layer has no batchnorm or non-linearity
        [global_pooling]    <bool> whether to include global average pooling layer at very end
        [gated]             <bool> whether conv-layers should be gated (not implemented for ResNet-layers)'''

        # Process type and number of blocks
        conv_type = "standard" if depth<2 else conv_type
        if conv_type=="resNet":
            num_blocks = [num_blocks]*(depth-1) if type(num_blocks)==int else num_blocks
            assert len(num_blocks)==(depth-1)
            block = conv_layers.Bottleneck if block_type == "bottleneck" else conv_layers.BasicBlock

        # Prepare label
        type_label = "C" if conv_type=="standard" else "R{}".format("b" if block_type=="bottleneck" else "")
        channel_label = "{}-{}x{}".format(image_channels, depth, start_channels)
        block_label = ""
        if conv_type=="resNet" and depth>1:
            block_label += "-"
            for block_num in num_blocks:
                block_label += "b{}".format(block_num)
        nd_label = "{bn}{nl}{gp}{gate}{out}".format(bn="b" if batch_norm else "", nl="l" if nl=="leakyrelu" else "",
                                                    gp="p" if global_pooling else "", gate="g" if gated else "",
                                                    out="n" if output=="none" else "")
        nd_label = "" if nd_label=="" else "-{}".format(nd_label)

        # Set configurations
        super().__init__()
        self.depth = depth
        self.rl = depth-1 if (reducing_layers is None) else (reducing_layers if (depth+1)>reducing_layers else depth)
        rl_label = "" if self.rl==(self.depth-1) else "-rl{}".format(self.rl)
        self.label = "{}{}{}{}{}".format(type_label, channel_label, block_label, rl_label, nd_label)
        self.block_expansion = block.expansion if conv_type=="resNet" else 1
        # -> constant by which # of output channels of each block is multiplied (if >1, it creates "bottleneck"-effect)
        double_factor = self.rl if self.rl<depth else depth-1 # -> how often # start-channels is doubled
        self.out_channels = (start_channels * 2**double_factor) * self.block_expansion if depth>0 else image_channels
        # -> number channels in last layer (as seen from image)
        self.start_channels = start_channels  # -> number channels in 1st layer (doubled in every "reducing layer")
        self.global_pooling = global_pooling  # -> whether or not average global pooling layer should be added at end

        # Conv-layers
        output_channels = start_channels
        for layer_id in range(1, depth+1):
            # should this layer down-sample? --> last [self.rl] layers should be down-sample layers
            reducing = True if (layer_id > (depth-self.rl)) else False
            # calculate number of this layer's input and output channels
            input_channels = image_channels if layer_id==1 else output_channels * self.block_expansion
            output_channels = output_channels*2 if (reducing and not layer_id==1) else output_channels
            # define and set the convolutional-layer
            if conv_type=="standard" or layer_id==1:
                conv_layer = conv_layers.conv_layer(input_channels, output_channels, stride=2 if reducing else 1,
                                                    drop=0, nl="no" if output=="none" and layer_id==depth else nl,
                                                    batch_norm=False if output=="none" and layer_id==depth else batch_norm,
                                                    gated= False if output=="none" and layer_id==depth else gated)
            else:
                conv_layer = conv_layers.res_layer(input_channels, output_channels, block=block,
                                                   num_blocks=num_blocks[layer_id-2], stride=2 if reducing else 1,
                                                   drop=0, batch_norm=batch_norm, nl=nl,
                                                   no_fnl=True if output=="none" and layer_id==depth else False)
            setattr(self, 'convLayer{}'.format(layer_id), conv_layer)
        # Perform pooling (if requested)
        self.pooling = nn.AdaptiveAvgPool2d((1,1)) if global_pooling else modules.Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all conv-layers
        for layer_id in range(skip_first+1, self.depth+1-skip_last):
            (x, pre_act) = getattr(self, 'convLayer{}'.format(layer_id))(x, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                if layer_id<(self.depth-skip_last):
                    hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Global average pooling (if requested)
        x = self.pooling(x)
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def out_size(self, image_size, ignore_gp=False):
        '''Given [image_size] of input, return the size of the "final" image that is outputted.'''
        out_size = int(np.ceil(image_size / 2**(self.rl))) if self.depth>0 else image_size
        return 1 if (self.global_pooling and not ignore_gp) else out_size

    def out_units(self, image_size, ignore_gp=False):
        '''Given [image_size] of input, return the total number of units in the output.'''
        return self.out_channels * self.out_size(image_size, ignore_gp=ignore_gp)**2

    def layer_info(self, image_size):
        '''Return list with shape of all hidden layers.'''
        layer_list = []
        reduce_number = 0  # keep track how often image-size has been halved
        double_number = 0  # keep track how often channel number has been doubled
        for layer_id in range(1, self.depth):
            reducing = True if (layer_id > (self.depth-self.rl)) else False
            if reducing:
                reduce_number += 1
            if reducing and layer_id>1:
                double_number += 1
            pooling = True if self.global_pooling and layer_id==(self.depth-1) else False
            expansion = 1 if layer_id==1 else self.block_expansion
            # add shape of this layer to list
            layer_list.append([(self.start_channels * 2**double_number) * expansion,
                               1 if pooling else int(np.ceil(image_size / 2**reduce_number)),
                               1 if pooling else int(np.ceil(image_size / 2**reduce_number))])
        return layer_list

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.depth+1):
            list += getattr(self, 'convLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label




class DeconvLayers(nn.Module):
    '''"Deconvolutional" feature decoder model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [in_channels] x [in_size] x [in_size] tensor
    Output: (tuple of) [batch_size] x [image_channels] x [final_size] x [final_size] tensor
                - with [final_size] = [in_size] x 2**[reducing_layers]
                       [in_channels] = [final_channels] x 2**min([reducing_layers], [depth]-1)'''

    def __init__(self, image_channels=3, final_channels=16, depth=5, reducing_layers=None, batch_norm=True, nl="relu",
                 gated=False, output="normal", smaller_kernel=False, deconv_type="standard"):
        '''[image_channels] # channels of image to decode
        [final_channels]    # channels in layer before output, was halved in every "rl" (=reducing layer) when moving
                                through model; corresponds to [start_channels] in "ConvLayers"-module
        [depth]             # layers (seen from the image, # channels is halved in each layer going to output image)
        [reducing_layers]   # of layers in which image-size is doubled & number of channels halved (default=[depth]-1)
                               ("rl"'s are the first conv-layers encountered--i.e., last conv-layers as seen from image)
                               (note that in the last layer # channels cannot be halved)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> what non-linearity to use -- choices: [relu, leakyrelu, sigmoid, none]
        [gated]             <bool> whether deconv-layers should be gated
        [output]            <str>; if - "normal", final layer is same as all others
                                      - "none", final layer has no non-linearity
                                      - "sigmoid", final layer has sigmoid non-linearity
        [smaller_kernel]    <bool> if True, use kernel-size of 2 (instead of 4) & without padding in reducing-layers'''

        # configurations
        super().__init__()
        self.depth = depth if depth>0 else 0
        self.rl = self.depth-1 if (reducing_layers is None) else min(self.depth, reducing_layers)
        type_label = "Deconv" if deconv_type=="standard" else "DeResNet"
        nd_label = "{bn}{nl}{gate}{out}".format(bn="-bn" if batch_norm else "", nl="-lr" if nl=="leakyrelu" else "",
                                                gate="-gated" if gated else "",
                                                out="" if output=="normal" else "-{}".format(output))
        self.label = "{}-ic{}-{}x{}-rl{}{}{}".format(type_label, image_channels, self.depth, final_channels, self.rl,
                                                     "s" if smaller_kernel else "", nd_label)
        if self.depth>0:
            self.in_channels = final_channels * 2**min(self.rl, self.depth-1) # -> input-channels for deconv
            self.final_channels = final_channels                              # -> channels in layer before output
        self.image_channels = image_channels                                  # -> output-channels for deconv

        # "Deconv"- / "transposed conv"-layers
        if self.depth>0:
            output_channels = self.in_channels
            for layer_id in range(1, self.depth+1):
                # should this layer down-sample? --> first [self.rl] layers should be down-sample layers
                reducing = True if (layer_id<(self.rl+1)) else False
                # update number of this layer's input and output channels
                input_channels = output_channels
                output_channels = int(output_channels/2) if reducing else output_channels
                # define and set the "deconvolutional"-layer
                if deconv_type=="standard":
                    new_layer = conv_layers.deconv_layer(
                        input_channels, output_channels if layer_id<self.depth else image_channels,
                        stride=2 if reducing else 1, batch_norm=batch_norm if layer_id<self.depth else False,
                        nl=nl if layer_id<self.depth or output=="normal" else (
                            "none" if output=="none" else nn.Sigmoid()
                        ), gated=gated, smaller_kernel=smaller_kernel
                    )
                else:
                    new_layer = conv_layers.deconv_res_layer(
                        input_channels, output_channels if layer_id < self.depth else image_channels,
                        stride=2 if reducing else 1, batch_norm=batch_norm if layer_id < self.depth else False,
                        nl=nl, smaller_kernel=smaller_kernel, output="normal" if layer_id<self.depth else output
                    )
                setattr(self, 'deconvLayer{}'.format(layer_id), new_layer)

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all "deconv"-layers
        if self.depth>0:
            for layer_id in range(skip_first+1, self.depth+1-skip_last):
                (x, pre_act) = getattr(self, 'deconvLayer{}'.format(layer_id))(x, return_pa=True)
                if return_lists:
                    pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                    if layer_id<(self.depth-skip_last):
                        hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        if self.depth>0:
            input_image_size = np.sqrt(in_units/self.in_channels) #-> size of image fed to last layer (seen from image)
            return input_image_size * 2**self.rl
        else:
            return np.sqrt(in_units/self.image_channels)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.depth+1):
            list += getattr(self, 'deconvLayer{}'.format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label
