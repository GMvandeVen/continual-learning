from torchvision import datasets, transforms
from data.manipulate import UnNormalize


# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'MNIST': [
        transforms.ToTensor(),
    ],
    'MNIST32': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'CIFAR10': [
        transforms.ToTensor(),
    ],
    'CIFAR100': [
        transforms.ToTensor(),
    ],
    'CIFAR10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'CIFAR100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'CIFAR10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'CIFAR100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'MNIST': {'size': 28, 'channels': 1, 'classes': 10},
    'MNIST32': {'size': 32, 'channels': 1, 'classes': 10},
    'CIFAR10': {'size': 32, 'channels': 3, 'classes': 10},
    'CIFAR100': {'size': 32, 'channels': 3, 'classes': 100},
}
