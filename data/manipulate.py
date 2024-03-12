import torch
from torch.utils.data import Dataset


def permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image

#----------------------------------------------------------------------------------------------------------#

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class MemorySetDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [memory_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, memory_sets, target_transform=None):
        super().__init__()
        self.memory_sets = memory_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.memory_sets)):
            total += len(self.memory_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.memory_sets)):
            examples_in_this_class = len(self.memory_sets[class_id])
            if index < (total + examples_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                example_id = index - total
                break
            else:
                total += examples_in_this_class
        image = torch.from_numpy(self.memory_sets[class_id][example_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''To modify an existing dataset with a transform.
    This is useful for creating different permutations of MNIST without loading the data multiple times.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)

# ----------------------------------------------------------------------------------------------------------#

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size()) == 4)
        for t, m, s in zip(tensor.permute(1, 0, 2, 3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
