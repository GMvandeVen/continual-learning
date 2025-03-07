#!/usr/bin/env python3
import numpy as np
import torch
# -custom-written libraries
import utils
from utils import checkattr
from data.load import get_singlecontext_datasets
from models import define_models as define
from train import train_standard
from params import options
from params.param_values import check_for_errors,set_default_values
from eval import callbacks as cb
from eval import evaluate


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'pretrain': True}
    # Define input options
    parser = options.define_args(filename="main_pretrain", description='Train classifier for pretraining conv-layers.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    # Parse, process and check chosen options
    args = parser.parse_args()
    set_default_values(args, also_hyper_params=False, single_context=True) # -set defaults based on chosen experiment
    check_for_errors(args, **kwargs)                                       # -check for incompatible options
    return args


## Function for running one experiment
def run(args, verbose=False):

    # Use cuda or mps (apple silicon)?
    cuda = torch.cuda.is_available() and args.gpu
    mps = torch.backends.mps.is_available() and args.gpu
    if cuda:
        device = torch.device("cuda")
    elif mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Report whether cuda or mps is used
    if verbose:
        if cuda:
            print("CUDA is used")
        elif mps:
            print("MPS is used (apple silicon GPU)")
        else:
            print("NO GPU is used!")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    elif mps:
        torch.mps.manual_seed(args.seed)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (trainset, testset), config = get_singlecontext_datasets(
        name=args.experiment, data_dir=args.d_dir, verbose=True,
        normalize = utils.checkattr(args, "normalize"), augment = utils.checkattr(args, "augment"),
    )

    # Specify "data-loader" (among others for easy random shuffling and 'batchifying')
    train_loader = utils.get_data_loader(trainset, batch_size=args.batch, cuda=cuda, drop_last=True)

    # Determine number of iterations:
    iters = args.iters if args.iters else args.epochs*len(train_loader)

    #-------------------------------------------------------------------------------------------------#

    #-----------------#
    #----- MODEL -----#
    #-----------------#

    # Specify model
    if verbose:
        print("\n\n " +' DEFINE MODEL '.center(70, '*'))
    cnn = define.define_standard_classifier(args=args, config=config, device=device, depth=args.depth)

    # Initialize (pre-trained) parameters
    define.init_params(cnn, args)

    # Set optimizer
    optim_list = [{'params': filter(lambda p: p.requires_grad, cnn.parameters()), 'lr': args.lr}]
    cnn.optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))

    # Print some model-characteristics on the screen
    if verbose:
        utils.print_model_info(cnn)

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Setting up Visdom environment
    if utils.checkattr(args, 'visdom'):
        if verbose:
            print('\n\n'+' VISDOM '.center(70, '*'))
        from visdom import Visdom
        env_name = args.experiment
        graph_name = cnn.name
        visdom = {'env': Visdom(env=env_name), 'graph': graph_name}
    else:
        visdom = None

    # Determine after how many iterations to evaluate the model (in visdom)
    loss_log = args.loss_log if (args.loss_log is not None) else len(train_loader)
    acc_log = args.acc_log if (args.acc_log is not None) else len(train_loader)

    # Define callback-functions to evaluate during training
    # -loss
    loss_cbs = [cb._classifier_loss_cb(log=loss_log, visdom=visdom)]
    # -accuracy
    eval_cbs = [cb._eval_cb(log=acc_log, test_datasets=[testset], visdom=visdom, test_size=args.acc_n)]

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- (PRE-)TRAINING -----#
    #--------------------------#

    # (Pre)train model
    if verbose:
        print("\n\n " +' TRAINING '.center(70, '*'))
    train_standard.train(cnn, train_loader, iters, loss_cbs=loss_cbs, eval_cbs=eval_cbs)

    # Save (pre)trained conv-layers and the full model
    if checkattr(args, 'save'):
        # -conv-layers
        save_name = cnn.convE.name if (
            not hasattr(args, 'convE_stag') or args.convE_stag=="none"
        ) else "{}-{}{}".format(cnn.convE.name, args.convE_stag,
                                "-s{}".format(args.seed) if checkattr(args, 'seed_to_stag') else "")
        utils.save_checkpoint(cnn.convE, args.m_dir, name=save_name)
        # -full model
        save_name = cnn.name if (
            not hasattr(args, 'full_stag') or args.full_stag=="none"
        ) else "{}-{}".format(cnn.name, args.full_stag)
        utils.save_checkpoint(cnn, args.m_dir, name=save_name)

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    # Evaluate accuracy of final model on full test-set
    if verbose:
        print("\n\n " +' EVALUATION '.center(70, '*'))
    train_acc = evaluate.test_acc(cnn, trainset, verbose=False, test_size=None)
    test_acc = evaluate.test_acc(cnn, testset, verbose=False, test_size=None)
    if verbose:
        print('=> ave accuracy (on training set):  {:.4f}'.format(train_acc))
        print('=> ave accuracy (on testing set):   {:.4f}\n'.format(test_acc))



if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)