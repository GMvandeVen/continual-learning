import abc
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def reservoir_sampling(samples_so_far, budget):
    '''Reservoir sampling algorithm to decide whether an new sample should be stored in the buffer or not.'''

    # If buffer is not yet full, simply add the new sample at the first available index
    if samples_so_far < budget:
        return samples_so_far

    # If buffer is full, draw random number to decide whether new sample should replace old sample (and which one)
    rand = np.random.randint(0, samples_so_far + 1)
    if rand < budget:
        return rand #--> new sample should replace old sample at index [rand]
    else:
        return -1   #--> new sample should not be stored in the buffer


class MemoryBuffer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module for classifier for maintaining a memory buffer using (global-)class-based reservoir sampling."""

    def __init__(self):
        super().__init__()

        # Settings
        self.use_memory_buffer = False
        self.budget = 100          #-> this is the overall budget (there is not memory buffer per class)
        self.samples_so_far = 0
        self.contexts_so_far = []

        # Settings related to using the memory buffer for nearest-class-mean classification
        self.prototypes = False    #-> whether classification is performed by using prototypes as class templates
        self.compute_means = True  #-> whenever new data is added, class-means must be recomputed
        self.norm_exemplars = True


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images, **kwargs):
        pass

    def initialize_buffer(self, config, return_c=False):
        '''Initalize the memory buffer with tensors of correct shape filled with zeros.'''
        self.buffer_x = torch.zeros(self.budget, config['channels'], config['size'], config['size'],
                                    dtype=torch.float32, device=self._device())
        self.buffer_y = torch.zeros(self.budget, dtype=torch.int64, device=self._device())
        if return_c:
            self.buffer_c = torch.zeros(self.budget, dtype=torch.int64, device=self._device())
        pass

    def add_new_samples(self, x, y, c):
        '''Process the data, and based on reservoir sampling algorithm potentially add to the buffer.'''

        # Whenever new training data is observed, indicate that class-means of stored data should be recomputed
        self.compute_means = True

        # Loop through all the samples contained in [x]
        for index in range(x.shape[0]):
            # -check whether this sample should be added to the memory buffer
            reservoir_index = reservoir_sampling(self.samples_so_far, self.budget)
            # -increase count of number of encountered samples
            self.samples_so_far += 1
            # -if selected, add the sample to the memory buffer
            if reservoir_index >= 0:
                self.buffer_x[reservoir_index] = x[index].to(self._device())
                self.buffer_y[reservoir_index] = y[index].to(self._device())
                if hasattr(self, 'buffer_c'):
                    self.buffer_c[reservoir_index] = c[index].to(self._device())

    def sample_from_buffer(self, size):
        '''Randomly sample [size] samples from the memory buffer.'''

        # If more samples are requested than in the buffer, set [size] to number of samples currently in the buffer
        samples_in_buffer = min(self.samples_so_far, self.budget)
        if size>samples_in_buffer:
            size = samples_in_buffer

        # Randomly select samples from the buffer and return them
        selected_indeces = np.random.choice(samples_in_buffer, size=size, replace=False)
        x = self.buffer_x[selected_indeces]
        y = self.buffer_y[selected_indeces]
        c = self.buffer_c[selected_indeces] if hasattr(self, 'buffer_c') else None
        return (x, y, c)

    def keep_track_of_contexts_so_far(self, c):
        self.contexts_so_far += [item.item() for item in c]

    def sample_contexts(self, size):
        if len(self.contexts_so_far)==0:
            raise AssertionError('No contexts have been observed yet.')
        else:
            return torch.tensor(np.random.choice(self.contexts_so_far, size, replace=True))


    def classify_with_prototypes(self, x, context=None):
        """Classify images by nearest-prototype / nearest-mean-of-exemplars rule (after transform to feature space)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch

        OUTPUT:     scores = <tensor> of size (bsz,n_classes)
        """

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means (=prototypes) need to be recomputed?
        if self.compute_means:
            self.possible_classes = [] #--> list of classes present in the memory buffer
            memory_set_means = []      #--> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for y in range(self.classes):
                if y in self.buffer_y:
                    self.possible_classes.append(y)
                    # Collect all stored samples of [y]
                    x_this_y = self.buffer_x[self.buffer_y==y]
                    c_this_y = self.buffer_c[self.buffer_y==y] if hasattr(self, 'buffer_c') else None
                    # Extract their features
                    with torch.no_grad():
                        features = self.feature_extractor(x_this_y, context=c_this_y)
                    if self.norm_exemplars:
                        features = F.normalize(features, p=2, dim=1)
                    # Calculate their mean and add to list
                    mu_y = features.mean(dim=0, keepdim=True)
                    if self.norm_exemplars:
                        mu_y = F.normalize(mu_y, p=2, dim=1)
                    memory_set_means.append(mu_y.squeeze())       # -> squeeze removes all dimensions of size 1
                else:
                    memory_set_means.append(None) # to indicate that this class is not present in the memory buffer
            # Update model's attributes
            self.memory_set_means = memory_set_means
            self.compute_means = False

        # Reorganize the [memory_set_means]-<tensor>
        memory_set_means = [self.memory_set_means[i] for i in self.possible_classes]
        means = torch.stack(memory_set_means)      # (n_possible_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_possible_classes, feature_size)
        means = means.transpose(1, 2)              # (batch_size, feature_size, n_possible_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x, context=context)    # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)             # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)         # (batch_size, feature_size, n_possible_classes)

        # For each sample in [x], find the (negative) distance of its extracted features to exemplar-mean of each class
        scores = -(feature-means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_possible_classes)

        # For all classes not in the memory, return a score of [-inf]
        all_scores = torch.ones(batch_size, self.classes, device=self._device())*-np.inf
        all_scores[:, self.possible_classes] = scores          # (batch_size, n_classes)

        # Set mode of model back
        self.train(mode=mode)

        return scores
