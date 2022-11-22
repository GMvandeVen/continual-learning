import os
import numpy as np
import pickle
import torch
from torchvision import transforms
import copy
import tqdm
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from models.fc import excitability_modules as em
from data.available import AVAILABLE_TRANSFORMS

##-------------------------------------------------------------------------------------------------------------------##

#######################
## General utilities ##
#######################

def checkattr(args, attr):
    '''Check whether attribute exists, whether it's a boolean and whether its value is True.'''
    return hasattr(args, attr) and type(getattr(args, attr))==bool and getattr(args, attr)

##-------------------------------------------------------------------------------------------------------------------##

#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, cuda=False, drop_last=False, augment=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=True, drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )

def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c

##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

##-------------------------------------------------------------------------------------------------------------------##

#########################################
## Model-saving and -loading functions ##
#########################################

def save_checkpoint(model, model_dir, verbose=True, name=None):
    '''Save state of [model] as dictionary to [model_dir] (if name is None, use "model.name").'''
    # -name/path to store the checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # -if required, create directory in which to save checkpoint
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # -create the dictionary containing the checkpoint
    checkpoint = {'state': model.state_dict()}
    if hasattr(model, 'mask_dict') and model.mask_dict is not None:
        checkpoint['mask_dict'] = model.mask_dict
    # -(try to) save the checkpoint
    try:
        torch.save(checkpoint, path)
        if verbose:
            print(' --> saved model {name} to {path}'.format(name=name, path=model_dir))
    except OSError:
        print(" --> saving model '{}' failed!!".format(name))

def load_checkpoint(model, model_dir, verbose=True, name=None, strict=True):
    '''Load saved state (in form of dictionary) at [model_dir] (if name is None, use "model.name") to [model].'''
    # -path from where to load checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # load parameters (i.e., [model] will now have the state of the loaded model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'], strict=strict)
    if 'mask_dict' in checkpoint:
        model.mask_dict = checkpoint['mask_dict']
    # notify that we succesfully loaded the checkpoint
    if verbose:
        print(' --> loaded checkpoint of {name} from {path}'.format(name=name, path=model_dir))

##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print( "--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("       of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                 - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params

def print_model_info(model, message=None):
    '''Print information on [model] onto the screen.'''
    print(55*"-" if message is None else ' {} '.format(message).center(55, '-'))
    print(model)
    print(55*"-")
    _ = count_parameters(model)

##-------------------------------------------------------------------------------------------------------------------##

########################################
## Parameter-initialization functions ##
########################################

def weight_reset(m):
    '''Reinitializes parameters of [m] according to default initialization scheme.'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, em.LinearExcitability):
        m.reset_parameters()

def weight_init(model, strategy="xavier_normal", std=0.01):
    '''Initialize weight-parameters of [model] according to [strategy].

    [xavier_normal]     "normalized initialization" (Glorot & Bengio, 2010) with Gaussian distribution
    [xavier_uniform]    "normalized initialization" (Glorot & Bengio, 2010) with uniform distribution
    [normal]            initialize with Gaussian(mean=0, std=[std])
    [...]               ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() >= 2:
            if strategy=="xavier_normal":
                nn.init.xavier_normal_(p)
            elif strategy=="xavier_uniform":
                nn.init.xavier_uniform_(p)
            elif strategy=="normal":
                nn.init.normal_(p, std=std)
            else:
                raise ValueError("Invalid weight-initialization strategy {}".format(strategy))

def bias_init(model, strategy="constant", value=0.01):
    '''Initialize bias-parameters of [model] according to [strategy].

    [zero]      set them all to zero
    [constant]  set them all to [value]
    [positive]  initialize with Uniform(a=0, b=[value])
    [any]       initialize with Uniform(a=-[value], b=[value])
    [...]       ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() == 1:
            ## NOTE: be careful if excitability-parameters are added to the model!!!!
            if strategy == "zero":
                nn.init.constant_(p, val=0)
            elif strategy == "constant":
                nn.init.constant_(p, val=value)
            elif strategy == "positive":
                nn.init.uniform_(p, a=0, b=value)
            elif strategy == "any":
                nn.init.uniform_(p, a=-value, b=value)
            else:
                raise ValueError("Invalid bias-initialization strategy {}".format(strategy))

##-------------------------------------------------------------------------------------------------------------------##

def preprocess(feature_extractor, dataset_list, config, batch=128, message='<PREPROCESS>'):
    '''Put a list of datasets through a feature-extractor, to return a new list of pre-processed datasets.'''
    device = feature_extractor._device()
    new_dataset_list = []
    progress_bar = tqdm.tqdm(total=len(dataset_list))
    progress_bar.set_description('{} | dataset {}/{} |'.format(message, 0, len(dataset_list)))
    for dataset_id in range(len(dataset_list)):
        loader = get_data_loader(dataset_list[dataset_id], batch_size=batch, drop_last=False,
                                 cuda=feature_extractor._is_on_cuda())
        # -pre-allocate tensors, which will be filled slice-by-slice
        all_features = torch.empty((len(loader.dataset), config['channels'], config['size'], config['size']))
        all_labels = torch.empty((len(loader.dataset)), dtype=torch.long)
        count = 0
        for x, y in loader:
            x = feature_extractor(x.to(device)).cpu()
            all_features[count:(count+x.shape[0])] = x
            all_labels[count:(count+x.shape[0])] = y
            count += x.shape[0]
        new_dataset_list.append(TensorDataset(all_features, all_labels))
        progress_bar.update(1)
        progress_bar.set_description('{} | dataset {}/{} |'.format(message, dataset_id + 1, len(dataset_list)))
    progress_bar.close()
    return new_dataset_list