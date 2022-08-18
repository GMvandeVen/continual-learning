import torch.nn as nn
from models.utils import modules


#-----------------------------------------------------------------------------------------------------------#

#####################
### ResNet-blocks ###
#####################

class BasicBlock(nn.Module):
    '''Standard building block for ResNets.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl="relu", no_fnl=False):
        super(BasicBlock, self).__init__()

        # normal block-layers
        self.block_layer1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False if batch_norm else True),
            nn.BatchNorm2d(planes) if batch_norm else modules.Identity(),
            nn.ReLU() if nl=="relu" else nn.LeakyReLU()
        )
        self.block_layer2 = nn.Sequential(
            nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1,
                      bias=False if batch_norm else True),
            nn.BatchNorm2d(self.expansion*planes) if batch_norm else modules.Identity()
        )

        # shortcut block-layer
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                          bias=False if batch_norm else True),
                nn.BatchNorm2d(self.expansion*planes) if batch_norm else modules.Identity()
            )

        # final non-linearity
        self.nl = (nn.ReLU() if nl=="relu" else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer2(self.block_layer1(x))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = [self.block_layer1[0], self.block_layer2[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


class Bottleneck(nn.Module):
    '''Building block (with "bottleneck") for ResNets.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl="relu", no_fnl=False):
        super(Bottleneck, self).__init__()

        # normal block-layers
        self.block_layer1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False if batch_norm else True),
            nn.BatchNorm2d(planes) if batch_norm else modules.Identity(),
            nn.ReLU() if nl == "relu" else nn.LeakyReLU()
        )
        self.block_layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False if batch_norm else True),
            nn.BatchNorm2d(planes) if batch_norm else modules.Identity(),
            nn.ReLU() if nl == "relu" else nn.LeakyReLU()
        )
        self.block_layer3 = nn.Sequential(
            nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False if batch_norm else True),
            nn.BatchNorm2d(self.expansion*planes) if batch_norm else modules.Identity()
        )

        # shortcut block-layer
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                          bias=False if batch_norm else True),
                nn.BatchNorm2d(self.expansion*planes) if batch_norm else True
            )

        # final non-linearity
        self.nl = (nn.ReLU() if nl == "relu" else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer3(self.block_layer2(self.block_layer1(x)))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = [self.block_layer1[0], self.block_layer2[0], self.block_layer3[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


#-----------------------------------------------------------------------------------------------------------#

###################
### Conv-layers ###
###################

class conv_layer(nn.Module):
    '''Standard convolutional layer. Possible to return pre-activations.'''

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 drop=0, batch_norm=False, nl=nn.ReLU(), bias=True, gated=False):
        super().__init__()
        if drop>0:
            self.dropout = nn.Dropout2d(drop)
        self.conv = nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_planes)
        if gated:
            self.gate = nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding,
                                  bias=False)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl=="relu" else (nn.LeakyReLU() if nl=="leakyrelu" else modules.Identity())

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.conv(input)) if hasattr(self, 'bn') else self.conv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.conv]


class res_layer(nn.Module):
    '''Convolutional res-net layer. Possible to return pre-activations.'''

    def __init__(self, in_planes, out_planes, block=BasicBlock, num_blocks=2, stride=1, drop=0, batch_norm=True,
                 nl="relu", no_fnl=False):

        ## NOTE: should [no_fnl] be changed so that also no batch_norm is applied?? ##

        # Set configurations
        super().__init__()
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes * block.expansion

        # Create layer
        self.dropout = nn.Dropout2d(drop)
        for block_id in range(num_blocks):
            # -first block has given stride, later blocks have stride 1
            new_block = block(in_planes, out_planes, stride=stride if block_id==0 else 1, batch_norm=batch_norm, nl=nl,
                              no_fnl=True if block_id==(num_blocks-1) else False)
            setattr(self, "block{}".format(block_id+1), new_block)
            in_planes = out_planes * block.expansion
        # self.bn = nn.BatchNorm2d(out_planes * block.expansion) if batch_norm else utils.Identity()
        self.nl = (nn.ReLU() if nl == "relu" else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x, return_pa=False):
        x = self.dropout(x)
        for block_id in range(self.num_blocks):
            x = getattr(self, "block{}".format(block_id+1))(x)
        output = self.nl(x)
        return (output, x) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for block_id in range(self.num_blocks):
            list += getattr(self, 'block{}'.format(block_id+1)).list_init_layers()
        return list


#-----------------------------------------------------------------------------------------------------------#

#####################
### Deconv-blocks ###
#####################

class DeconvBlock(nn.Module):
    '''Building block for deconv-layer with multiple blocks.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True, nl="relu", no_fnl=False, smaller_kernel=False):
        super(DeconvBlock, self).__init__()

        # normal block-layers
        self.block_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_planes, planes, stride=stride, bias=False if batch_norm else True,
                               kernel_size=(2 if smaller_kernel else 4) if stride==2 else 3,
                               padding=0 if (stride==2 and smaller_kernel) else 1),
            nn.BatchNorm2d(planes) if batch_norm else modules.Identity(),
            nn.ReLU() if nl=="relu" else nn.LeakyReLU()
        )
        self.block_layer2 = nn.Sequential(
            nn.ConvTranspose2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1,
                               bias=False if batch_norm else True),
            nn.BatchNorm2d(self.expansion*planes) if batch_norm else modules.Identity()
        )

        # shortcut block-layer
        self.shortcut = modules.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                                   output_padding=0 if stride==1 else 1, bias=False if batch_norm else True),
                nn.BatchNorm2d(self.expansion*planes) if batch_norm else modules.Identity()
            )

        # final non-linearity
        self.nl = (nn.ReLU() if nl=="relu" else nn.LeakyReLU()) if not no_fnl else modules.Identity()

    def forward(self, x):
        out = self.block_layer2(self.block_layer1(x))
        out += self.shortcut(x)
        return self.nl(out)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = [self.block_layer1[0], self.block_layer2[0]]
        if not type(self.shortcut) == modules.Identity:
            list.append(self.shortcut[0])
        return list


#-----------------------------------------------------------------------------------------------------------#

#####################
### Deconv-layers ###
#####################

class deconv_layer(nn.Module):
    '''Standard "deconvolutional" layer. Possible to return pre-activations.'''

    def __init__(self, input_channels, output_channels, stride=1, drop=0, batch_norm=True, nl="relu", bias=True,
                 gated=False, smaller_kernel=False):
        super().__init__()
        if drop>0:
            self.dropout = nn.Dropout2d(drop)
        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, bias=bias, stride=stride,
                                         kernel_size=(2 if smaller_kernel else 4) if stride==2 else 3,
                                         padding=0 if (stride==2 and smaller_kernel) else 1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_channels)
        if gated:
            self.gate = nn.ConvTranspose2d(input_channels, output_channels, bias=False, stride=stride,
                                           kernel_size=(2 if smaller_kernel else 4) if stride==2 else 3,
                                           padding=0 if (stride==2 and smaller_kernel) else 1)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif nl in ("sigmoid", "hardtanh"):
            self.nl = nn.Sigmoid() if nl=="sigmoid" else nn.Hardtanh(min_val=-4.5, max_val=0)
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else modules.Identity())

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.deconv(input)) if hasattr(self, 'bn') else self.deconv(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.deconv]


class deconv_layer_split(nn.Module):
    '''"Deconvolutional" layer outputing [mean] and [logvar] for each unit.'''

    def __init__(self, input_channels, output_channels, nl_mean="sigmoid", nl_logvar="hardtanh",
                 stride=1, drop=0, batch_norm=True, bias=True, gated=False, smaller_kernel=False):
        super().__init__()
        self.mean = deconv_layer(input_channels, output_channels, nl=nl_mean, smaller_kernel=smaller_kernel,
                                 stride=stride, drop=drop, batch_norm=batch_norm, bias=bias, gated=gated)
        self.logvar = deconv_layer(input_channels, output_channels, nl=nl_logvar, smaller_kernel=smaller_kernel,
                                   stride=stride, drop=drop, batch_norm=batch_norm, bias=False, gated=gated)

    def forward(self, x, return_pa=False):
        mean, pre_activ = self.mean(x, return_pa=True)
        logvar = self.logvar(x)
        return ((mean, logvar), pre_activ) if return_pa else (mean, logvar)

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


class deconv_res_layer(nn.Module):
    '''Deconvolutional res-net layer. Possible to return pre-activations.'''

    def __init__(self, in_planes, out_planes, block=DeconvBlock, num_blocks=2, stride=1, drop=0, batch_norm=True,
                 nl="relu", smaller_kernel=False, output="normal"):

        ## NOTE: should [output=="none"] be changed so that also no batch_norm is applied?? ##

        # Set configurations
        super().__init__()
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes * block.expansion

        # Create layer
        self.dropout = nn.Dropout2d(drop)
        for block_id in range(num_blocks):
            # -first block has given stride, later blocks have stride 1
            new_block = block(in_planes, out_planes, stride=stride if block_id==0 else 1, batch_norm=batch_norm, nl=nl,
                              no_fnl=True if block_id==(num_blocks-1) else False, smaller_kernel=smaller_kernel)
            setattr(self, "block{}".format(block_id+1), new_block)
            in_planes = out_planes * block.expansion
        # self.bn = nn.BatchNorm2d(out_planes * block.expansion) if batch_norm else utils.Identity()
        if output=="sigmoid":
            self.nl = nn.Sigmoid()
        elif output=="normal":
            self.nl = nn.ReLU() if nl=="relu" else nn.LeakyReLU()
        elif output=="none":
            self.nl = modules.Identity()
        else:
            raise NotImplementedError("Ouptut '{}' not implemented for deconvolutional ResNet layer.".format(output))

    def forward(self, x, return_pa=False):
        x = self.dropout(x)
        for block_id in range(self.num_blocks):
            x = getattr(self, "block{}".format(block_id+1))(x)
        output = self.nl(x)
        return (output, x) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for block_id in range(self.num_blocks):
            list += getattr(self, 'block{}'.format(block_id+1)).list_init_layers()
        return list

