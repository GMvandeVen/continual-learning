import math
import numpy as np
import random
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F
from models.fc import excitability_modules as em


## This code has been based upon: https://github.com/team-approx-bayes/fromp (accessed 8 July 2021)


#--------------------------------------------------------------------------------------------#

############################
## COMPUTATION OF HESSIAN ##
############################

# Calculate the diagonal elements of the hessian
def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s

# Calculate the full softmax hessian
def full_softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    e = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return s[:, :, None]*e[None, :, :] - s[:, :, None]*s[:, None, :]


#--------------------------------------------------------------------------------------------#

######################
## HELPER FUNCTIONS ##
######################

def _update_input(self, input, output):
    self.input = input[0].data
    self.output = output

def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = (param.get_device() != old_param_device) if param.is_cuda else (old_param_device != -1)
        if warn:
            raise TypeError('Parameters are on different devices, not currently supported.')
    return old_param_device

def _parameters_to_matrix(parameters):
    param_device = None
    mat = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        m = param.shape[0]
        mat.append(param.view(m, -1))
    return torch.cat(mat, dim=-1)

def _parameters_grads_to_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            raise ValueError('Gradient is not available.')
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)


#--------------------------------------------------------------------------------------------#

#####################
## FROMP OPTIMIZER ##
#####################

class opt_fromp(Optimizer):
    '''Implements the FROMP algorithm (Pan et al., 2020 NeurIPS) as a PyTorch-optimizer, combined with Adam.

    Args:
        model (nn.Module): model whose parameters are to be trained
        lr (float, optional): learning rate (default: 0.001)
        betas (tuple, optional): coefs for computing running mean of gradient and its square (default: (0.9, 0.999))
        amsgrad (bool, optional): whether to use the AMSGrad-variant of the Adam algorithm (default: False)
        tau (float, optional): how strongly to weight the regularization term, FROMP's main hyperparameter (default: 1.)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        prior_prec (float, optional): ... (default: 1e-3)
        grad_clip_norm (float, optional): what value to clip the norm of the gradient to during training (default: 1.)
        per_context (bool, optional): ... (default: True)
    '''

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), amsgrad=False,
                 tau=1., eps=1e-8, prior_prec=1e-3, grad_clip_norm=1., per_context=True):

        # Check for invalid arguments
        if not 0.0 <= lr:
            raise ValueError("invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= prior_prec:
            raise ValueError("invalid prior precision: {}".format(prior_prec))
        if grad_clip_norm is not None and not 0.0 <= grad_clip_norm:
            raise ValueError("invalid gradient clip norm: {}".format(grad_clip_norm))
        if not 0.0 <= tau:
            raise ValueError("invalid tau: {}".format(tau))

        # Deal with arguments set per parameter group (ALTHOUGH PARAMETER GROUPS ARE NOT FUNCTIONAL WITH THIS OPTIMIZER)
        defaults = dict(lr=lr, betas=betas, eps=eps, prior_prec=prior_prec, grad_clip_norm=grad_clip_norm,
                        tau=tau, amsgrad=amsgrad)
        super(opt_fromp, self).__init__(model.parameters(), defaults)

        # Set the model and its trainable modules
        self.per_context = per_context
        self.model = model
        self.train_modules = []
        self.set_train_modules(model)
        for module in self.train_modules:
            module.register_forward_hook(_update_input)

        # Initialize the optimizer's state variables
        parameters = self.param_groups[0]['params']
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        self.state['mu_previous'] = p.clone().detach()
        self.state['fisher'] = torch.zeros_like(self.state['mu'])
        self.state['step'] = 0
        self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
        self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
        if amsgrad:
            self.state['max_exp_avg_sq'] = torch.zeros_like(self.state['mu'])

    # Set all trainable modules, required for calculating Jacobians in PyTorch
    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    #----------------------------------------------------------------------------------------------------------#

    # Calculate the gradient of the parameters [lc] with respect to the loss (required for calculating the Jacobian)
    def cac_grad(self, loss, lc, retain_graph=None):
        linear_grad = torch.autograd.grad(loss, lc, retain_graph=retain_graph)
        grad = []
        for i, module in enumerate(self.train_modules):
            # print("--> Starting module {} of {}".format(i, len(self.train_modules)))
            g = linear_grad[i].detach()
            a = module.input.clone().detach()
            m = a.shape[0]

            if isinstance(module, nn.Linear) or isinstance(module, em.LinearExcitability):
                with torch.no_grad():
                    grad.append(torch.einsum('ij,ik->ijk', g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    a = F.unfold(a, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding,
                                 stride=module.stride)
                    _, k, hw = a.shape
                    _, c, _, _ = g.shape
                    g = g.view(m, c, -1)
                    grad.append(torch.einsum('ijl,ikl->ijk', g, a))
                    if module.bias is not None:
                        a = torch.ones((m, 1, hw), device=a.device)
                        grad.append(torch.einsum('ijl,ikl->ijk', g, a))

            if isinstance(module, nn.BatchNorm1d):
                with torch.no_grad():
                    grad.append(torch.mul(g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.BatchNorm2d):
                with torch.no_grad():
                    grad.append(torch.einsum('ijkl->ij', torch.mul(g, a)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ijkl->ij', g))

        grad_m = _parameters_to_matrix(grad)
        return grad_m.detach()

    # Calculate the Jacobian matrix
    def cac_jacobian(self, output, lc):
        if output.dim() > 2:
            raise ValueError('the dimension of output must be smaller than 3.')
        elif output.dim() == 2:
            num_fun = output.shape[1]
        grad = []
        for i in range(num_fun):
            retain_graph = None if i == num_fun-1 else True
            loss = output[:, i].sum()
            g = self.cac_grad(loss, lc, retain_graph=retain_graph)
            grad.append(g)
        result = torch.zeros((grad[0].shape[0], grad[0].shape[1], num_fun),
                             dtype=grad[0].dtype, device=grad[0].device)
        for i in range(num_fun):
            result[:, :, i] = grad[i]
        return result

    #----------------------------------------------------------------------------------------------------------#

    # Calculate values (memorable_logits, hkh_l) for regularisation term (all but the first context)
    def init_context(self, context_id, eps=1e-6, reset=True, classes_per_context=2, label_sets=None):

        # If requested, reset the adam-optimizer
        if reset:
            self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
            self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
            self.state['step'] = 0

        # Initiliase objects to be stored as empty lists
        self.state['kernel_inv'] = []
        self.state['memorable_logits'] = []

        # Compute covariance (using the pre-computed Fisher matrix)
        fisher = self.state['fisher']
        prior_prec = self.param_groups[0]['prior_prec']
        covariance = 1. / (fisher + prior_prec)          #-> size: [n_params]

        # Get and store parameter values
        mu = self.state['mu']
        self.state['mu_previous'] = mu.clone().detach()
        parameters = self.param_groups[0]['params']
        vector_to_parameters(mu, parameters)

        # Loop over all contexts so far
        self.model.eval()
        for i in range(context_id if self.per_context else 1):

            # Collect all memorable points for this context from the memory buffer
            classes_in_context = range(classes_per_context*i, classes_per_context*(i+1)) if self.per_context else range(
                classes_per_context*context_id
            )
            mem_points_np = np.concatenate([self.model.memory_sets[id] for id in classes_in_context], axis=0)
            memorable_points_t = torch.from_numpy(mem_points_np).to(self.model._device())
            #-> size: [n_per_context]x[ch]x[length]x[width]

            # Compute and store the mean of their function values (i.e., the predicted logits)
            self.zero_grad()
            logits = self.model.forward(memorable_points_t)
            preds = logits if (label_sets[i] is None) else logits[:, label_sets[i]]
            preds = torch.softmax(preds, dim=-1)   #-> size: [n_per_context]x[classes_per_context]
            self.state['memorable_logits'].append(preds.detach())

            # Compute and store the kernel of their function values
            lc = []
            for module in self.train_modules:
                lc.append(module.output)
            kernel_inv = []
            num_classes = preds.shape[-1]

            for class_id in range(num_classes):
                loss = preds[:, class_id].sum()
                retain_graph = True if class_id < num_classes-1 else None
                grad = self.cac_grad(loss, lc, retain_graph=retain_graph) #-> size: [n_mem_points]x[n_params]
                with torch.no_grad():
                    kernel = torch.einsum('ij,j,pj->ip', grad, covariance, grad) + \
                             torch.eye(grad.shape[0], dtype=grad.dtype, device=grad.device)*eps
                    # -store inverse of kernel (size: [n_mem_points]x[n_mem_points]) for this class via Cholesky decomp
                    kernel_inv.append(torch.cholesky_inverse(torch.cholesky(kernel)))

            self.state['kernel_inv'].append(kernel_inv)

    # After training on a new context, update the fisher matrix estimate
    def update_fisher(self, dataloader, label_set=None):
        fisher = self.state['fisher']

        self.model.eval()
        for data,_ in dataloader:
            data = data.to(self.model._device())

            self.zero_grad()
            logits = self.model.forward(data)
            preds = logits if label_set is None else logits[:, label_set]

            lc = []
            for module in self.train_modules:
                lc.append(module.output)
            jac = self.cac_jacobian(preds, lc).detach()
            with torch.no_grad():
                hes = full_softmax_hessian(preds.detach())
                jhj = torch.einsum('ijd,idp,ijp->j', jac, hes, jac)
                fisher.add_(jhj)

    #----------------------------------------------------------------------------------------------------------#

    def step(self, x, y, label_sets, context_id, classes_per_context):
        '''Performs a single optimization step.'''

        defaults = self.defaults
        lr = self.param_groups[0]['lr']
        beta1, beta2 = self.param_groups[0]['betas']
        amsgrad = self.param_groups[0]['amsgrad']
        parameters = self.param_groups[0]['params']
        mu = self.state['mu']

        self.model.train()

        # Calculate normal loss term over current context's data, and compute its gradient
        vector_to_parameters(mu, parameters)
        self.zero_grad()
        logits = self.model.forward(x) if (
                label_sets[context_id] is None
        ) else self.model.forward(x)[:, label_sets[context_id]]
        loss_cur = F.cross_entropy(input=logits, target=y, reduction='mean')
        accuracy = (y == logits.max(1)[1]).sum().item() / x.size(0)
        loss_cur.backward(retain_graph=None)
        grad = _parameters_grads_to_vector(parameters).detach()

        # Calculate the loss term corresponding to the memorable points, and compute & add their gradients
        if context_id > 0:
            self.model.eval()
            kernel_inv = self.state['kernel_inv']
            memorable_logits = self.state['memorable_logits']
            grad_t_sum = torch.zeros_like(grad)
            for t in range(context_id if self.per_context else 1):

                # Select subset of memorable points to use in this batch
                batch_size_per_context = int(np.ceil(x.shape[0] / context_id)) if self.per_context else x.shape[0]
                if self.per_context:
                    memory_samples_per_context = (len(self.model.memory_sets[0])*classes_per_context)
                else:
                    memory_samples_per_context = (len(self.model.memory_sets[0])*classes_per_context*context_id)
                if batch_size_per_context<memory_samples_per_context:
                    indeces_in_this_batch = random.sample(range(memory_samples_per_context), batch_size_per_context)
                else:
                    indeces_in_this_batch = list(range(memory_samples_per_context))

                if self.per_context:
                    classes_in_context = range(classes_per_context * t, classes_per_context * (t + 1))
                else:
                    classes_in_context = range(classes_per_context*context_id)
                mem_points_np = np.concatenate([self.model.memory_sets[id] for id in classes_in_context], axis=0)
                memorable_data_t = torch.from_numpy(mem_points_np[indeces_in_this_batch]).to(self.model._device())
                label_set_t = label_sets[t]
                self.zero_grad()
                logits = self.model.forward(memorable_data_t)
                preds_t = logits if (label_set_t is None) else logits[:, label_set_t]

                num_fun = preds_t.shape[-1]
                preds_t = torch.softmax(preds_t, dim=-1)
                lc = []
                for module in self.train_modules:
                    lc.append(module.output)
                for fi in range(num_fun):
                    # \Lambda * Jacobian
                    loss_jac_t = preds_t[:, fi].sum()
                    retain_graph = True if fi < num_fun - 1 else None
                    jac_t = self.cac_grad(loss_jac_t, lc, retain_graph=retain_graph)

                    # m_t - m_{t-1}
                    logits_t = preds_t[:, fi].detach()
                    delta_logits = logits_t - memorable_logits[t][indeces_in_this_batch,fi]

                    # K_{t-1}^{-1}
                    kernel_inv_t = kernel_inv[t][fi][:, indeces_in_this_batch][indeces_in_this_batch]

                    # Calculate K_{t-1}^{-1} (m_t - m_{t-1})
                    with torch.no_grad():
                        kinvf_t = torch.squeeze(torch.matmul(kernel_inv_t, delta_logits[:,None]), dim=-1)

                        grad_t = torch.einsum('ij,i->j', jac_t, kinvf_t)

                    grad_t_sum.add_(grad_t)

            # Weight term corresponding to memorable points by [tau] and add to gradient
            with torch.no_grad():
                grad_t_sum.mul_(defaults['tau'])
                grad.add_(grad_t_sum)

        # Do gradient norm clipping
        clip_norm = self.defaults['grad_clip_norm']
        if clip_norm is not None:
            grad_norm = torch.norm(grad)
            grad_norm = 1.0 if grad_norm < clip_norm else grad_norm/clip_norm
            grad.div_(grad_norm)

        # Given the gradient computed above, prepare for the updated based on Adam algorithm
        exp_avg, exp_avg_sq = self.state['exp_avg'], self.state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = self.state['max_exp_avg_sq']
        self.state['step'] += 1
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(self.param_groups[0]['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(self.param_groups[0]['eps'])
        bias_correction1 = 1 - beta1 ** self.state['step']
        bias_correction2 = 1 - beta2 ** self.state['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        # Do the parameter update
        mu.addcdiv_(exp_avg, denom, value=-step_size)
        vector_to_parameters(mu, parameters)

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_cur.item(),
            'loss_current': loss_cur.item(),
            'pred': loss_cur.item(),
            'accuracy': accuracy if accuracy is not None else 0.,
        }

    #----------------------------------------------------------------------------------------------------------#
