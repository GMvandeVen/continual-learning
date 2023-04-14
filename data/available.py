from sklearn.calibration import LabelEncoder
from torchvision import datasets, transforms
from data.manipulate import UnNormalize
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

LABEL_COLUMN = 'Attack Type'

def clean_dataset(df: pd.DataFrame):
    # drop uneeded columns
    df.drop(columns=['Unnamed: 0', 'Attack Tool', 'Label'], inplace=True) # Seq
    # drop columns that contain a lot of null values
    d = {}
    for col in df.columns:
        if df[col].isnull().sum() > (len(df) * 0.7):
            df.drop(columns = [col], inplace=True)
    # drop rows that contain NaN values and duplicates
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    # reduce the classes percentage
    reduce_class(df, LABEL_COLUMN, 'Benign', 0.5)
    reduce_class(df, LABEL_COLUMN, 'UDPFlood', 0.6)
    reduce_class(df, LABEL_COLUMN, 'HTTPFlood', 0.43)
    reduce_class(df, LABEL_COLUMN, 'SlowrateDoS', 0.35)
    reduce_class(df, LABEL_COLUMN, 'ICMPFlood', 1)

    #  standard normalization
    
    # change type of True/False columns to bool
    for col in df.columns:
        vals = df[col].unique()
        if len(vals) in [1,2]:
            d[col] = bool
    df = df.astype(d)
    # normalize float and int columns
    s = df.select_dtypes(include=['float', 'int'])
    s = (s - s.mean())/s.std()
    for col in s: 
        df[col] = s[col]
    # rechange type of bool columns to int
    d = {} 
    for col in df.columns:
        vals = df[col].unique()
        if len(vals) in [1,2]:
            d[col] = int
    df = df.astype(d)

def reduce_class(df, col_name, class_val, p):
    indices = df.index[df[col_name] == class_val]
    df.drop(indices[:int(len(indices) * p)], inplace=True)

def split_dataset(dataframe):
    y_data = dataframe[LABEL_COLUMN].values
    x_data = dataframe.drop(columns=[LABEL_COLUMN]).values
    return x_data, y_data


class networkDataset(Dataset):

    #classes = ['Benign', 'SYNScan', 'TCPConnectScan', 'UDPScan', 'ICMPFlood', 'UDPFlood', 'SYNFlood', 'HTTPFlood', 'SlowrateDoS']

    @property
    def train_labels(self):
        return self.targets
    
    @property
    def test_labels(self):
        return self.targets
    
    @property
    def train_data(self):
        return self.data

    @property
    def train_data(self):
        return self.data

    def __init__(self, src_file, train=True, transform=None, target_transform=None, download=False, transforms=None):
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")
        
        if has_separate_transform:
            transforms = datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(src_file)
        clean_dataset(df)
        df[LABEL_COLUMN] = LabelEncoder().fit_transform(df[LABEL_COLUMN])
        x_data, y_data = split_dataset(df)
        # y_data_trans = list(map(lambda label: self.classes.index(label), y_data))

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

        if train:
            self.data, self.targets = x_train, y_train
        else:
            self.data, self.targets = x_test, y_test
        
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # preds = self.x_data[idx, :]
        # pol = self.y_data[idx]
        # sample = {'predictors': preds, 'political': pol}

        vector, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.reshape(vector, (9,9)), mode="L")

        #print(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# specify available data-sets.
AVAILABLE_DATASETS = {
    'MNIST': datasets.MNIST,
    'CIFAR100': datasets.CIFAR100,
    'CIFAR10': datasets.CIFAR10,
    '5GNIDD': networkDataset,
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
    '5GNIDD': [
        #transforms.Pad(1),
        #transforms.ToPILImage(),
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
    '5GNIDD': {'size':9, 'channels': 1, 'classes': 9},
}
