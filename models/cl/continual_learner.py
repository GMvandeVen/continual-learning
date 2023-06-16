import abc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from utils import get_data_loader
from models import fc
from models.utils.ncl import additive_nearest_kf


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(self):
        super().__init__()

        # List with the methods to create generators that return the parameters on which to apply param regularization
        self.param_list = [self.named_parameters]  #-> lists the parameters to regularize with SI or diagonal Fisher
                                                   #   (default is to apply it to all parameters of the network)
        #-> with OWM or KFAC Fisher, only parameters in [self.fcE] and [self.classifier] are regularized

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        self.optim_type = "adam"
        #--> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every context
        self.optim_list = []
        #--> self.[optim_list]   <list>, if optimizer should be reset after each context, provide list of required <dicts>

        # Scenario, singlehead & negative samples
        self.scenario = 'task'       # which scenario will the model be trained on
        self.classes_per_context = 2 # number of classes per context
        self.singlehead = False      # if Task-IL, does the model have a single-headed output layer?
        self.neg_samples = 'all'     # if Class-IL, which output units should be set to 'active'?

        # LwF / Replay
        self.replay_mode = "none"    # should replay be used, and if so what kind? (none|current|buffer|all|generative)
        self.replay_targets = "hard" # should distillation loss be used? (hard|soft)
        self.KD_temp = 2.            # temperature for distillation loss
        self.use_replay = "normal"   # how to use the replayed data? (normal|inequality|both)
                                     # -inequality = use gradient of replayed data as inequality constraint for gradient
                                     #               of the current data (as in A-GEM; Chaudry et al., 2019; ICLR)
        self.eps_agem = 0.           # parameter that improves numerical stability of AGEM (if set slighly above 0)
        self.lwf_weighting = False   # LwF has different weighting of the 'stability' and 'plasticity' terms than replay

        # XdG:
        self.mask_dict = None        # -> <dict> with context-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # Parameter-regularization
        self.weight_penalty = False
        self.reg_strength = 0       #-> hyperparam: how strong to weigh the weight penalty ("regularisation strength")
        self.precondition = False
        self.alpha = 1e-10          #-> small constant to stabilize inversion of the Fisher Information Matrix
                                    #   (this is used as hyperparameter in OWM)
        self.importance_weighting = 'fisher'  #-> Options for estimation of parameter importance:
                                              #   - 'fisher':   Fisher Information matrix (e.g., as in EWC, NCL)
                                              #   - 'si':       ... diagonal, online importance estimation ...
                                              #   - 'owm':      ...
        self.fisher_kfac = False    #-> whether to use a block-diagonal KFAC approximation to the Fisher Information
                                    #   (alternative is a diagonal approximation)
        self.fisher_n = None        #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.fisher_labels = "all"  #-> what label(s) to use for any given sample when calculating the FI matrix?
                                    #   - 'all':    use all labels, weighted according to their predicted probabilities
                                    #   - 'sample': sample one label to use, using predicted probabilities for sampling
                                    #   - 'pred':   use the predicted label (i.e., the one with highest predicted prob)
                                    #   - 'true':   use the true label (NOTE: this is also called "empirical FI")
        self.fisher_batch = 1       #-> batch size for estimating FI-matrix (should be 1, for best results)
                                    #   (different from 1 only works if [fisher_labels]='pred' or 'true')
        self.context_count = 0      #-> counts 'contexts' (if a prior is used, this is counted as the first context)
        self.data_size = None       #-> inverse prior (can be set to # samples per context, or used as hyperparameter)
        self.epsilon = 0.1          #-> dampening parameter (SI): bounds 'omega' when squared parameter-change goes to 0
        self.offline = False        #-> use separate penalty term per context (as in original EWC paper)
        self.gamma = 1.             #-> decay-term for old contexts' contribution to cummulative FI (as in 'Online EWC')
        self.randomize_fisher = False

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, context):
        '''Apply context-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [context]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a context-specific mask has been specified
        for i,excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[context][i]] = 0.    # -> find context-specific mask
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this mask

    def reset_XdGmask(self):
        '''Remove context-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this unit mask


    #------------- "Synaptic Intelligence"-specifc functions -------------#

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())

    def prepare_importance_estimates_dicts(self):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        W = {}
        p_old = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old):
        '''Update the running parameter importance estimates in W.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad*(p.detach()-p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        '''After completing training on a context, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W[n]/(p_change**2 + epsilon)
                    try:
                        omega = getattr(self, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add

                    # Store these new values in the model
                    self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                    self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        # Calculate SI's surrogate loss, sum over all parameters
                        losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())


    #----------------- EWC-specifc functions -----------------#

    def initialize_fisher(self):
        '''Initialize diagonal fisher matrix with the prior precision (as in NCL).'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -take initial parameters as zero for regularization purposes
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone()*0)
                    # -precision (approximated by diagonal Fisher Information matrix)
                    self.register_buffer( '{}_EWC_estimated_fisher'.format(n), torch.ones(p.shape) / self.data_size)

    def estimate_fisher(self, dataset, allowed_classes=None):
        '''After completing training on a context, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1 (unless specifically asked to do otherwise)
        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch is None else self.fisher_batch,
                                      cuda=self._is_on_cuda())

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index > self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            # calculate FI-matrix (according to one of the four options)
            if self.fisher_labels=='all':
                # -use a weighted combination of all labels
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.cross_entropy(output, label)  #--> get neg log-likelihoods for this class
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                    # Square gradients and keep running sum (using the weights)
                    for gen_params in self.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
                                if self.randomize_fisher:
                                    idx = torch.randperm(est_fisher_info[n].nelement())
                                    est_fisher_info[n] = est_fisher_info[n].view(-1)[idx].view(
                                        est_fisher_info[n].size())
            else:
                # -only use one particular label for each datapoint
                if self.fisher_labels=='true':
                    # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                    label = torch.LongTensor([y]) if type(y)==int else y  #-> shape: [self.fisher_batch]
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label.to(self._device())
                elif self.fisher_labels=='pred':
                    # --> use predicted label to calculate loglikelihood:
                    label = output.max(1)[1]
                elif self.fisher_labels=='sample':
                    # --> sample one label from predicted probabilities
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)       #--> get predicted probabilities
                    weights_array = np.array(label_weights[0].cpu())   #--> change to np-array, avoiding rounding errors
                    label = np.random.choice(len(weights_array), 1, p=weights_array/weights_array.sum())
                    label = torch.LongTensor(label).to(self._device()) #--> change label to tensor on correct device
                # calculate negative log-likelihood
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                # calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward()
                # square gradients and keep running sum
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.detach() ** 2
                            if self.randomize_fisher:
                                idx = torch.randperm(est_fisher_info[n].nelement())
                                est_fisher_info[n] = est_fisher_info[n].view(-1)[idx].view(est_fisher_info[n].size())

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -mode (=MAP parameter estimate)
                    self.register_buffer('{}_EWC_prev_context{}'.format(n, self.context_count+1 if self.offline else ""),
                                         p.detach().clone())
                    # -precision (approximated by diagonal Fisher Information matrix)
                    if (not self.offline) and hasattr(self, '{}_EWC_estimated_fisher'.format(n)):
                        existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        est_fisher_info[n] += self.gamma * existing_values
                    self.register_buffer(
                        '{}_EWC_estimated_fisher{}'.format(n, self.context_count+1 if self.offline else ""), est_fisher_info[n]
                    )

        # Increase context-count
        self.context_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        try:
            losses = []
            # If "offline EWC", loop over all previous contexts as each context has separate penalty term
            num_penalty_terms = self.context_count if (self.offline and self.context_count>0) else 1
            for context in range(1, num_penalty_terms+1):
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                            n = n.replace('.', '__')
                            mean = getattr(self, '{}_EWC_prev_context{}'.format(n, context if self.offline else ""))
                            fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, context if self.offline else ""))
                            # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                            fisher = fisher if self.offline else self.gamma*fisher
                            # Calculate weight regularization loss
                            losses.append((fisher * (p-mean)**2).sum())
            # Sum the regularization loss from all parameters (and from all contexts, if "offline EWC")
            return (1./2)*sum(losses)
        except AttributeError:
            # Regularization loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())


    # ----------------- KFAC-specifc functions -----------------#

    def initialize_kfac_fisher(self):
        '''Initialize Kronecker-factored Fisher matrix with the prior precision (as in NCL).'''
        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.eye(abar_dim) / np.sqrt(self.data_size)
            G = torch.eye(g_dim) / np.sqrt(self.data_size)
            return {"A": A, "G": G, "weight": linear.weight.data * 0,
                    "bias": None if linear.bias is None else linear.bias.data * 0}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info["classifier"] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        self.KFAC_FISHER_INFO = initialize()

    def estimate_kfac_fisher(self, dataset, allowed_classes=None):
        """After completing training on a context, estimate KFAC Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes
        """

        print('computing kfac fisher')

        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            G = torch.zeros(g_dim, g_dim)
            if linear.bias is None:
                bias = None
            else:
                bias = linear.bias.data.clone()
            return {"A": A, "G": G, "weight": linear.weight.data.clone(), "bias": bias}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info["classifier"] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples, weight=1):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            if not hasattr(layer, 'phantom'):
                raise Exception(f"Layer {label} does not have phantom parameters")
            g = layer.phantom.grad.detach()
            G = g[..., None] @ g[..., None, :]
            _a = intermediate[label].detach()
            # Here we do one batch at a time (not ideal)
            assert _a.shape[0] == 1
            a = _a[0]

            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1).to(self._device())
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]["A"].to(self._device())
            Go = est_fisher_info[label]["G"].to(self._device())
            est_fisher_info[label]["A"] = Ao + weight * A / n_samples
            est_fisher_info[label]["G"] = Go + weight * G / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples, weight=1):
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples, weight=weight)
            update_fisher_info_layer(est_fisher_info, intermediate, "classifier", self.classifier, n_samples,
                                     weight=weight)

        # initialize estimated fisher info
        est_fisher_info = initialize()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1 (unless specifically asked to do otherwise)
        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch is None else self.fisher_batch,
                                      cuda=self._is_on_cuda())

        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for i, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if i > n_samples:
                break
            # run forward pass of model
            x = x.to(self._device())
            _output, intermediate = self(x, return_intermediate=True)
            output = _output if allowed_classes is None else _output[:, allowed_classes]
            # calculate FI-matrix (according to one of the four options)
            if self.fisher_labels=='all':
                # -use a weighted combination of all labels
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                    update_fisher_info(est_fisher_info, intermediate, n_samples, weight=label_weights[0][label_index])
            else:
                # -only use one particular label for each datapoint
                if self.fisher_labels == 'true':
                    # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                    label = torch.LongTensor([y]) if type(y) == int else y  # -> shape: [self.fisher_batch]
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label.to(self._device())
                elif self.fisher_labels == 'pred':
                    # --> use predicted label to calculate loglikelihood:
                    label = output.max(1)[1]
                elif self.fisher_labels == 'sample':
                    # --> sample one label from predicted probabilities
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)  # --> get predicted probabilities
                    weights_array = np.array(label_weights[0].cpu())  # --> change to np-array, avoiding rounding errors
                    label = np.random.choice(len(weights_array), 1, p=weights_array / weights_array.sum())
                    label = torch.LongTensor(label).to(self._device())  # --> change label to tensor on correct device

                # calculate negative log-likelihood
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

                # Calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward()
                update_fisher_info(est_fisher_info, intermediate, n_samples)


        for label in est_fisher_info:
            An = est_fisher_info[label]["A"].to(self._device())  # new kronecker factor
            Gn = est_fisher_info[label]["G"].to(self._device())  # new kronecker factor
            Ao = self.gamma * self.KFAC_FISHER_INFO[label]["A"].to(self._device())  # old kronecker factor
            Go = self.KFAC_FISHER_INFO[label]["G"].to(self._device())               # old kronecker factor

            As, Gs = additive_nearest_kf({"A": Ao, "G": Go}, {"A": An, "G": Gn})  # sum of kronecker factors
            self.KFAC_FISHER_INFO[label]["A"] = As
            self.KFAC_FISHER_INFO[label]["G"] = Gs

            for param_name in ["weight", "bias"]:
                p = est_fisher_info[label][param_name].to(self._device())
                self.KFAC_FISHER_INFO[label][param_name] = p

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_kfac_loss(self):
        fcE = self.fcE

        def loss_for_layer(label, layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            info = self.KFAC_FISHER_INFO[label]
            A = info["A"].detach().to(self._device())
            G = info["G"].detach().to(self._device())
            bias0 = info["bias"]
            weight0 = info["weight"]
            bias = layer.linear.bias
            weight = layer.linear.weight
            if bias0 is not None and bias is not None:
                p = torch.cat([weight, bias[..., None]], -1)
                p0 = torch.cat([weight0, bias0[..., None]], -1)
            else:
                p = weight
                p0 = weight0
            assert p.shape[-1] == A.shape[1]
            assert p0.shape[-1] == A.shape[1]
            dp = p.to(self._device()) - p0.to(self._device())
            return torch.sum(dp * (G @ dp @ A))

        classifier = self.classifier
        if self.context_count > 0:
            l = loss_for_layer("classifier", classifier)
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                nl = loss_for_layer(label, getattr(fcE, label))
                l += nl
            return 0.5 * l
        else:
            return torch.tensor(0.0, device=self._device())


    # ----------------- OWM-specifc functions -----------------#

    def estimate_owm_fisher(self, dataset, **kwargs):
        '''After completing training on a context, estimate OWM Fisher Information matrix based on [dataset].'''

        ## QUESTION: Should OWM not also be applied to the outputs??

        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            return {'A': A, 'weight': linear.weight.data.clone(),
                    'bias': None if linear.bias is None else linear.bias.data.clone()}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples):
            _a = intermediate[label].detach()
            # Here we do one batch at a time (not ideal)
            assert (_a.shape[0] == 1)
            a = _a[0]
            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1).to(self._device())
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]['A'].to(self._device())
            est_fisher_info[label]['A'] = Ao + A / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples):
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples)
            update_fisher_info_layer(est_fisher_info, intermediate, 'classifier', n_samples)

        # initialize estimated fisher info
        est_fisher_info = initialize()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for i, (x, _) in enumerate(data_loader):
            if i > n_samples:
                break
            # run forward pass of model
            x = x.to(self._device())
            output, intermediate = self(x, return_intermediate=True)
            # update OWM importance matrix
            self.zero_grad()
            update_fisher_info(est_fisher_info, intermediate, n_samples)

        if self.context_count == 0:
            self.KFAC_FISHER_INFO = {}

        for label in est_fisher_info:
            An = est_fisher_info[label]['A'].to(self._device())  # new kronecker factor
            if self.context_count == 0:
                self.KFAC_FISHER_INFO[label] = {}
                As = An
            else:
                Ao = self.gamma * self.KFAC_FISHER_INFO[label]['A'].to(self._device())  # old kronecker factor
                frac = 1 / (self.context_count + 1)
                As = (1 - frac) * Ao + frac * An

            self.KFAC_FISHER_INFO[label]['A'] = As

            for param_name in ['weight', 'bias']:
                p = est_fisher_info[label][param_name].to(self._device())
                self.KFAC_FISHER_INFO[label][param_name] = p

        self.context_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)