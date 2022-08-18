import numpy as np
from torch import nn


##-------------------------------------------------------------------------------------------------------------------##

#################################
## Custom-written "nn-Modules" ##
#################################

class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Shape(nn.Module):
    '''A nn-module to shape a tensor of shape [shape].'''
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.dim = len(shape)

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(shape = {})'.format(self.shape)
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor(-tuple) to a 4-dim "image"-tensor(-tuple) with [image_channels] channels.'''
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        if type(x)==tuple:
            batch_size = x[0].size(0)   # first dimenstion should be batch-dimension.
            image_size = int(np.sqrt(x[0].nelement() / (batch_size*self.image_channels)))
            return (x_item.view(batch_size, self.image_channels, image_size, image_size) for x_item in x)
        else:
            batch_size = x.size(0)   # first dimenstion should be batch-dimension.
            image_size = int(np.sqrt(x.nelement() / (batch_size*self.image_channels)))
            return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


##-------------------------------------------------------------------------------------------------------------------##