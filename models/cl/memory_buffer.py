import abc
import torch
from torch import nn
from torch.nn import functional as F
from utils import get_data_loader
import copy
import numpy as np
from models.cl.fromp_optimizer import softmax_hessian


class MemoryBuffer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that enables it to maintain a memory buffer."""

    def __init__(self):
        super().__init__()

        # List with memory-sets
        self.memory_sets = [] #-> each entry of [self.memory_sets] is an <np.array> of N images with shape (N, Ch, H, W)
        self.memory_set_means = []
        self.compute_means = True

        # Settings
        self.use_memory_buffer = False
        self.budget_per_class = 100
        self.use_full_capacity = False
        self.sample_selection = 'random'
        self.norm_exemplars = True

        # Atributes defining how to use memory-buffer
        self.prototypes = False  #-> perform classification by using prototypes as class templates
        self.add_buffer = False  #-> add the memory buffer to the training set of the current task


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass


    ####----MANAGING THE MEMORY BUFFER----####

    def reduce_memory_sets(self, m):
        for y, P_y in enumerate(self.memory_sets):
            self.memory_sets[y] = P_y[:m]

    def construct_memory_set(self, dataset, n, label_set):
        '''Construct memory set of [n] examples from [dataset] using 'herding', 'random' or 'fromp' selection.

        Note that [dataset] should be from specific class; selected sets are added to [self.memory_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        memory_set = []

        if self.sample_selection=="fromp":
            first_entry = True

            # Loop over all samples in the dataset
            dataloader = get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for i, dt in enumerate(dataloader):
                # Compute for each sample its "importance score"
                data, _ = dt
                f = self.forward(data.to(self._device()))
                lamb = softmax_hessian(f if label_set is None else f[:,label_set])
                lamb = torch.sum(lamb.cpu(), dim=-1).detach()

                # Store both the samples and their computed scores
                if first_entry:
                    memorable_points = data
                    scores = lamb
                    first_entry = False
                else:
                    memorable_points = torch.cat([memorable_points, data], dim=0)
                    scores = torch.cat([scores, lamb], dim=0)

            # Select the samples with the best (or worst) scores, and store them in the memory buffer
            if len(memorable_points) > n:
                _, indices = scores.sort(descending=True)
                memorable_points = memorable_points[indices[:n]]
            # -add this [memory_set] as a [n]x[ich]x[isz]x[isz] to the list of [memory_sets]
            self.memory_sets.append(memorable_points.numpy())

        elif self.sample_selection=="herding":
            # Compute features for each example in [dataset]
            first_entry = True
            dataloader = get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # Calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # One by one, select samples so the mean of all selected samples is as close to [class_mean] as possible
            selected_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k>0:
                    selected_samples_sum = torch.sum(selected_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + selected_samples_sum)/(k+1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Samples in the memory buffer should not be repeated!!!!")
                list_of_selected.append(index_selected)

                memory_set.append(dataset[index_selected][0].numpy())
                selected_features[k] = copy.deepcopy(features[index_selected])
                # -make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 10000
            # -add this [memory_set] as a [n]x[ich]x[isz]x[isz] to the list of [memory_sets]
            self.memory_sets.append(np.array(memory_set))

        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                memory_set.append(dataset[k][0].numpy())
            # -add this [memory_set] as a [n]x[ich]x[isz]x[isz] to the list of [memory_sets]
            self.memory_sets.append(np.array(memory_set))

        # Set mode of model back
        self.train(mode=mode)


    ####----CLASSIFICATION----####

    def classify_with_prototypes(self, x, allowed_classes=None):
        """Classify images by nearest-prototype / nearest-mean-of-exemplars rule (after transform to feature space)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     scores = <tensor> of size (bsz,n_classes)
        """

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means (=prototypes) need to be recomputed?
        if self.compute_means:
            memory_set_means = []  #--> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.memory_sets:
                exemplars = []
                # Collect all 'exemplars' in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                memory_set_means.append(mu_y.squeeze())       # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.memory_set_means = memory_set_means
            self.compute_means = False

        # Reorganize the [memory_set_means]-<tensor>
        memory_set_means = self.memory_set_means if allowed_classes is None else [
            self.memory_set_means[i] for i in allowed_classes
        ]
        means = torch.stack(memory_set_means)      # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)              # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)    # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)             # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)         # (batch_size, feature_size, n_classes)

        # For each sample in [x], find the (negative) distance of its extracted features to exemplar-mean of each class
        scores = -(feature-means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)

        # Set mode of model back
        self.train(mode=mode)

        return scores

