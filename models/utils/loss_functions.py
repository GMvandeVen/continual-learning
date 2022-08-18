import numpy as np
import torch
from torch.nn import functional as F


##-------------------------------------------------------------------------------------------------------------------##

####################
## Loss functions ##
####################

def loss_fn_kd(scores, target_scores, T=2., weights=None, norm=False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be <2D-tensors>, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # If [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -targets_norm * log_scores_norm

    # Sum over the prob-scores of all classes (1) and then average over all elements in the batch (2)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                                 #-> sum over classes
    KD_loss_unnorm = weighted_average(KD_loss_unnorm, weights=weights, dim=0)  #-> average over batch

    # Normalize
    KD_loss = (KD_loss_unnorm * T**2) if norm else KD_loss_unnorm

    return KD_loss



def loss_fn_kd_binary(scores, target_scores, T=2., weights=None, norm=False):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)

    # If [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -( targets_norm * torch.log(scores_norm) + (1-targets_norm) * torch.log(1-scores_norm) )

    # Sum over the prob-scores of all classes (1) and then average over all elements in the batch (2)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                                 #-> sum over classes
    KD_loss_unnorm = weighted_average(KD_loss_unnorm, weights=weights, dim=0)  #-> average over batch

    # Normalize
    KD_loss = (KD_loss_unnorm * T**2) if norm else KD_loss_unnorm


    return KD_loss


##-------------------------------------------------------------------------------------------------------------------##

######################
## Helper functions ##
######################

def weighted_average(tensor, weights=None, dim=0):
    '''Computes weighted average of [tensor] over dimension [dim].'''
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size())>0 else 1
        assert len(weights)==batch_size
        #sum_weights = sum(weights)
        #norm_weights = torch.Tensor([weight/sum_weights for weight in weights]).to(tensor.device)
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights*tensor, dim=dim)
    return mean

def to_one_hot(y, classes, device=None):
    '''Convert <nd-array> or <tensor> with integers [y] to a 2D "one-hot" <tensor>.'''
    if type(y)==torch.Tensor:
        device=y.device
        y = y.cpu()
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c if device is None else c.to(device)


##-------------------------------------------------------------------------------------------------------------------##

########################################################
## Calculate log-likelihood for various distributions ##
########################################################

def log_Normal_standard(x, mean=0, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=I.
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    log_normal = -0.5 * torch.pow(x-mean, 2)
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=exp[log_var].
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    log_normal = -0.5 * (log_var + torch.pow(x-mean, 2) / torch.exp(log_var))
    if dim is not None and dim==-1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)

def log_Bernoulli(x, mean, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Bernoulli distribution(s) with mu=[mean].
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    probs = torch.clamp(mean, min=1e-5, max=1.-1e-5)
    log_bernoulli = x*torch.log(probs) + (1. - x)*torch.log(1. - probs)
    if dim is not None and dim==-1:
        log_bernoulli = log_bernoulli.view(log_bernoulli.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_bernoulli, dim) if dim is not None else torch.mean(log_bernoulli)
    else:
        return torch.sum(log_bernoulli, dim) if dim is not None else torch.sum(log_bernoulli)

##-------------------------------------------------------------------------------------------------------------------##