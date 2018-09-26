import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
import copy
import data
from encoder import Classifier
from vae_models import AutoEncoder

###################
## Loss function ##
###################

def loss_fn_kd(scores, target_scores, T=2., cuda=False):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensor-wrapped Variables, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.cuda() if cuda else zeros_to_add
        targets_norm = Variable(torch.cat([targets_norm.data, zeros_to_add], dim=1))

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = (-targets_norm * log_scores_norm).mean()

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss



##-------------------------------------------------------------------------------------------------------------------##


#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=True,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )


def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()



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
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("Model-name: \"" + model.name + "\"")
    print(40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-" + "\n\n")


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''

    config = data.get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,
        only_config=True, verbose=False, exception=True if args.seed==0 else False,
    )
    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        )
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        )
    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        )

    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay=False if (args.replay=="none") else True,
                                  replay_model_name=replay_model_name)
    return param_stamp


def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    task_stamp = "{exp}{n}-{sce}".format(exp=args.experiment, n=args.tasks, sce=args.scenario)
    if verbose:
        print("\n"+" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "i{n}-lr{lr}-b{bsz}-{optim}".format(n=args.iters, lr=args.lr, bsz=args.batch, optim=args.optimizer)
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for replay
    if replay:
        replay_stamp = "{rep}{KD}{model}{gi}".format(
            rep=args.replay, KD="-KD{}".format(args.temp) if args.distill else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.g_iters) if (
                (replay_model_name is not None) and (not args.iters==args.g_iters)
            ) else ""
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for EWC / SI
    if (args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda,
            fi="{}{}".format("N" if args.fisher_n is None else args.fisher_n, "E" if args.emp_fi else ""),
            o="-O{}".format(args.gamma) if args.online else "",
        ) if (args.ewc_lambda>0 and args.ewc) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (args.si_c>0 and args.si) else ""
        both = "--" if (args.ewc_lambda>0 and args.ewc) and (args.si_c>0 and args.si) else ""
        if verbose and args.ewc_lambda>0 and args.ewc:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and args.si_c>0 and args.si:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
        (args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si)
    ) else ""

    # -XdG
    xdg_stamp = "--XdG{}".format(args.gating_prop) if (hasattr(args, "gating_prop") and args.gating_prop>0) else ""

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, replay_stamp, ewc_stamp, xdg_stamp, "-{}".format(args.seed),
    )

    ## Print param-stamp on screen and return
    print(param_stamp)
    return param_stamp


##-------------------------------------------------------------------------------------------------------------------##

#################################
## Custom-written "nn-Modules" ##
#################################


class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels.'''
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        image_size = int(np.sqrt(x.nelement() / (batch_size*self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class ToImage(nn.Module):
    '''Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor'''

    def __init__(self, image_channels=1):
        super().__init__()
        # reshape to 4D-tensor
        self.reshape = Reshape(image_channels=image_channels)
        # put through sigmoid-nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        image_size = np.sqrt(in_units/self.image_channels)
        return image_size


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


##-------------------------------------------------------------------------------------------------------------------##