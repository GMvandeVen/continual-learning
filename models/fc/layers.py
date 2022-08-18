import numpy as np
import torch
from torch import nn
from models.utils import modules
from models.fc import excitability_modules as em


class fc_layer(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(), drop=0., bias=True, batch_norm=False,
                 excitability=False, excit_buffer=False, gated=False, phantom=False):
        super().__init__()
        self.bias = False if batch_norm else bias
        if drop>0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if phantom:
            self.phantom = nn.Parameter(torch.zeros(out_size), requires_grad=True)
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else modules.Identity())

    def forward(self, x, return_pa=False, **kwargs):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        if hasattr(self, 'phantom'):
            gated_pre_activ = gated_pre_activ + self.phantom
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]



class fc_layer_split(nn.Module):
    '''Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors'''

    def __init__(self, in_size, out_size, nl_mean=nn.Sigmoid(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()

        self.mean = fc_layer(in_size, out_size, drop=drop, bias=bias, excitability=excitability,
                             excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_mean)
        self.logvar = fc_layer(in_size, out_size, drop=drop, bias=False, excitability=excitability,
                               excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_logvar)

    def forward(self, x):
        return (self.mean(x), self.logvar(x))

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list



class fc_layer_fixed_gates(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations". Has fixed gates (of specified dimension).

    Input:  [batch_size] x ... x [in_size] tensor         &        [batch_size] x ... x [gate_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False,
                 gate_size=0, gating_prop=0.8, device='cpu'):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gate_size>0:
            self.gate_mask = torch.tensor(
                np.random.choice([0., 1.], size=(gate_size, out_size), p=[gating_prop, 1.-gating_prop]),
                dtype=torch.float, device=device
            )
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == "none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else modules.Identity())

    def forward(self, x, gate_input=None, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = torch.matmul(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate_mask') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]



class fc_multihead_layer(nn.Module):
    '''Fully connected layer with a separate head for each context.

    Input:  [batch_size] x ... x [in_size] tensor         &        [batch_size] x ... x [n_contexts] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, classes, n_contexts, nl=nn.ReLU(),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, device='cpu'):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, classes, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(classes)
        if n_contexts > 0:
            self.gate_mask = torch.zeros(size=(n_contexts, classes), dtype=torch.float, device=device)
            classes_per_context = int(classes/n_contexts)
            for context_id in range(n_contexts):
                self.gate_mask[context_id, (context_id * classes_per_context):((context_id + 1) * classes_per_context)] = 1.
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == "none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else modules.Identity())

    def forward(self, x, gate_input=None, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        gate = torch.matmul(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate_mask') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]