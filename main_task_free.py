#!/usr/bin/env python3
import os
import numpy as np
import time
import torch
from torch import optim
# -custom-written libraries
import utils
from utils import checkattr
from data.load import get_context_set
from data.labelstream import SharpBoundaryStream, RandomStream, FuzzyBoundaryStream
from data.datastream import DataStream
from models import define_models as define
from models.cl.continual_learner import ContinualLearner
from models.cl.memory_buffer_stream import MemoryBuffer
from train.train_stream import train_on_stream, train_gen_classifier_on_stream
from params import options
from params.param_stamp import get_param_stamp, get_param_stamp_from_args, visdom_name
from params.param_values import set_method_options,check_for_errors,set_default_values
from eval import evaluate, callbacks as cb


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'main': True, 'no_boundaries': True}
    # Define input options
    parser = options.define_args(filename="main_task_free",
                                 description='Run a "task-free" continual learning experiment '
                                             '(i.e., no [known,] sharp boundaries between contexts).')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Parse, process and check chosen options
    args = parser.parse_args()
    set_method_options(args)                                             # -"convenience"-option used, select components
    set_default_values(args, also_hyper_params=True, no_boundaries=True) # -set defaults, some based on chosen options
    check_for_errors(args, **kwargs)                                     # -check for incompatible options
    return args


def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    if checkattr(args, 'get_stamp'):
        print(get_param_stamp_from_args(args=args, no_boundaries=True))
        exit()

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

    #-----------------------#
    #----- CONTEXT SET -----#
    #-----------------------#

    # Prepare the context set for the chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir,
        normalize=checkattr(args, "normalize"), verbose=verbose, exception=(args.seed==0),
        singlehead=checkattr(args, 'singlehead')
    )

    #-------------------------------------------------------------------------------------------------#

    #-----------------------------#
    #----- FEATURE EXTRACTOR -----#
    #-----------------------------#

    # Define the feature extractor
    depth = args.depth if hasattr(args, 'depth') else 0
    use_feature_extractor = checkattr(args, 'hidden') or (
            checkattr(args, 'freeze_convE') and (not args.replay=="generative") and (not checkattr(args, "add_buffer"))
            and (not checkattr(args, 'gen_classifier'))
    )
    #--> when the convolutional layers are frozen, it is faster to put the data through these layers only once at the
    #     beginning, but this currently does not work with iCaRL or pixel-level generative replay/classification
    if use_feature_extractor and depth>0:
        if verbose:
            print("\n\n " + ' DEFINE FEATURE EXTRACTOR '.center(70, '*'))
        feature_extractor = define.define_feature_extractor(args=args, config=config, device=device)
        # - initialize (pre-trained) parameters
        define.init_params(feature_extractor, args, verbose=verbose)
        # - freeze the parameters & set model to eval()-mode
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        # - print characteristics of feature extractor on the screen
        if verbose:
            utils.print_model_info(feature_extractor)
        # - reset size and # of channels to reflect the extracted features rather than the original images
        config = config.copy()  # -> make a copy to avoid overwriting info in the original config-file
        config['size'] = feature_extractor.conv_out_size
        config['channels'] = feature_extractor.conv_out_channels
        depth = 0
    else:
        feature_extractor = None

    # Convert original data to features (so this doesn't need to be done at run-time)
    if (feature_extractor is not None) and args.depth>0:
        if verbose:
            print("\n\n " + ' PUT DATA TRHOUGH FEATURE EXTRACTOR '.center(70, '*'))
        train_datasets = utils.preprocess(feature_extractor, train_datasets, config, batch=args.batch,
                                          message='<TRAINSET>')
        test_datasets = utils.preprocess(feature_extractor, test_datasets, config, batch=args.batch,
                                         message='<TESTSET> ')

    #-------------------------------------------------------------------------------------------------#

    #-----------------------#
    #----- DATA-STREAM -----#
    #-----------------------#

    # Set up the stream of context-labels to use
    if args.stream == "academic-setting":
        label_stream = SharpBoundaryStream(n_contexts=args.contexts, iters_per_context=args.iters)
    elif args.stream == "fuzzy-boundaries":
        label_stream = FuzzyBoundaryStream(
            n_contexts=args.contexts, iters_per_context=args.iters, fuzziness=args.fuzziness,
            batch_size=1 if checkattr(args, 'labels_per_batch') else args.batch
        )
    elif args.stream == "random":
        label_stream = RandomStream(n_contexts=args.contexts)
    else:
        raise NotImplementedError("Stream type '{}' not currently implemented.".format(args.stream))

    # Set up the data-stream to be presented to the network
    data_stream = DataStream(
        train_datasets, label_stream, batch_size=args.batch, return_context=(args.scenario=="task"),
        per_batch=True if (args.stream=="academic-setting") else checkattr(args, 'labels_per_batch'),
    )

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- CLASSIFIER -----#
    #----------------------#

    # Define the classifier
    if verbose:
        print("\n\n " + ' DEFINE THE CLASSIFIER '.center(70, '*'))
    model = define.define_classifier(args=args, config=config, device=device, depth=depth, stream=True)

    # Some type of classifiers consist of multiple networks
    n_networks = len(train_datasets) if checkattr(args, 'separate_networks') else (
        model.classes if checkattr(args, 'gen_classifier') else 1
    )

    # Go through all networks to ...
    for network_id in range(n_networks):
        model_to_set = getattr(model, 'context{}'.format(network_id+1)) if checkattr(args, 'separate_networks') else (
            getattr(model, 'vae{}'.format(network_id)) if checkattr(args, 'gen_classifier') else model
        )
        # ... initialize / use pre-trained / freeze model-parameters, and
        define.init_params(model_to_set, args)
        # ... define optimizer (only include parameters that "requires_grad")
        model_to_set.optim_list = [{'params': filter(lambda p: p.requires_grad, model_to_set.parameters()),
                                    'lr': args.lr}]
        model_to_set.optim_type = args.optimizer
        if model_to_set.optim_type=="adam":
            model_to_set.optimizer = optim.Adam(model_to_set.optim_list, betas=(0.9, 0.999))
        elif model_to_set.optim_type=="sgd":
            model_to_set.optimizer = optim.SGD(model_to_set.optim_list,
                                               momentum=args.momentum if hasattr(args, 'momentum') else 0.)

    # On what scenario will model be trained?
    model.scenario = args.scenario
    model.classes_per_context = config['classes_per_context']

    # Print some model-characteristics on the screen
    if verbose:
        if checkattr(args, 'gen_classifier') or checkattr(args, 'separate_networks'):
            message = '{} copies of:'.format(len(train_datasets))
            utils.print_model_info(model.vae0 if checkattr(args, 'gen_classifier') else model.context1, message=message)
        else:
            utils.print_model_info(model)

    # -------------------------------------------------------------------------------------------------#

    # For multiple continual learning methods: how often (after how many iters) to perform the consolidation operation?
    # (this can be interpreted as: how many iterations together should be considered a "context")
    model.update_every = args.update_every if hasattr(args, 'update_every') else 1

    # -------------------------------------------------------------------------------------------------#

    # ----------------------------------------------------#
    # ----- CL-STRATEGY: CONTEXT-SPECIFIC COMPONENTS -----#
    # ----------------------------------------------------#

    # XdG: already indicated when defining the classifier

    #-------------------------------------------------------------------------------------------------#

    #-------------------------------------------------#
    #----- CL-STRATEGY: PARAMETER REGULARIZATION -----#
    #-------------------------------------------------#

    # Parameter regularization by adding a weight penalty (e.g., SI)
    if isinstance(model, ContinualLearner) and checkattr(args, 'weight_penalty'):
        model.weight_penalty = True
        model.importance_weighting = args.importance_weighting
        model.reg_strength = args.reg_strength
        if model.importance_weighting=='si':
            model.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.1

    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------#
    #----- CL-STRATEGY: FUNCTIONAL REGULARIZATION -----#
    #--------------------------------------------------#

    # Should a distillation loss (i.e., soft targets) be used? (e.g., for LwF)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay'):
        model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
        model.KD_temp = args.temp if hasattr(args, 'temp') else 2.

    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Should the model be trained with replay?
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay'):
        model.replay_mode = args.replay

    # A-GEM: How should the gradient of the loss on replayed data be used? (added, as inequality constraint or both?)
    if isinstance(model, ContinualLearner) and hasattr(args, 'use_replay'):
        model.use_replay = args.use_replay
        model.eps_agem = args.eps_agem if hasattr(args, 'eps_agem') else 0.

    #-------------------------------------------------------------------------------------------------#

    #-------------------------#
    #----- MEMORY BUFFER -----#
    #-------------------------#

    # Should a memory buffer be maintained? (e.g., for experience replay or prototype-based classification)
    use_memory_buffer = checkattr(args, 'prototypes') or args.replay=="buffer"
    if isinstance(model, MemoryBuffer) and use_memory_buffer:
        model.use_memory_buffer = True
        model.budget = args.budget
        model.initialize_buffer(config, return_c=(args.scenario=='task'))

    # Should classification be done using prototypes as class templates?
    model.prototypes = checkattr(args, 'prototypes')

    # Relevant for "modified iCaRL": whether to use binary loss
    if model.label=="Classifier":
        model.binaryCE = checkattr(args, 'bce')

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- PARAMETER STAMP -----#
    #---------------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        if verbose:
            print('\n\n' + ' PARAMETER STAMP '.center(70, '*'))
    param_stamp = get_param_stamp(
        args, model.name, feature_extractor_name= feature_extractor.name if (feature_extractor is not None) else None,
        verbose=verbose, no_boundaries=True,
    )

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Setting up Visdom environment
    if utils.checkattr(args, 'visdom'):
        if verbose:
            print('\n\n'+' VISDOM '.center(70, '*'))
        from visdom import Visdom
        env_name = "{exp}{con}-{sce}".format(exp=args.experiment, con=args.contexts, sce=args.scenario)
        visdom = {'env': Visdom(env=env_name), 'graph': visdom_name(args)}
    else:
        visdom = None

    # Callbacks for reporting and visualizing loss
    loss_cbs = [
        cb._gen_classifier_loss_cb(
            log=args.loss_log, classes=None, visdom=None,
        ) if checkattr(args, 'gen_classifier') else cb._classifier_loss_cb(
            log=args.loss_log, visdom=visdom, model=model, contexts=None,
        )
    ]

    # Callbacks for reporting and visualizing accuracy
    eval_cbs = [
        cb._eval_cb(log=args.acc_log, test_datasets=test_datasets, visdom=visdom, iters_per_context=args.iters,
                    test_size=args.acc_n)
    ]

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    # Train model
    if args.train:
        if verbose:
            print('\n\n' + ' TRAINING '.center(70, '*'))
        # -keep track of training-time
        if args.time:
            start = time.time()
        # -select training function
        train_fn = train_gen_classifier_on_stream if checkattr(args, 'gen_classifier') else train_on_stream
        # -perform training
        train_fn(model, data_stream, iters=args.iters*args.contexts, eval_cbs=eval_cbs, loss_cbs=loss_cbs)
        # -get total training-time in seconds, write to file and print to screen
        if args.time:
            training_time = time.time() - start
            time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
            time_file.write('{}\n'.format(training_time))
            time_file.close()
            if verbose and args.time:
                print("Total training time = {:.1f} seconds\n".format(training_time))
        # -save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                    not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of previously trained model...")
        load_name = "mM-{}".format(param_stamp) if (
            not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose, strict=False)

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print('\n\n' + ' EVALUATION '.center(70, '*'))

    # Set attributes of model that define how to do classification
    if checkattr(args, 'gen_classifier'):
        model.S = args.eval_s

    # Evaluate accuracy of final model on full test-set
    if verbose:
        print("\n Accuracy of final model on test-set:")
    accs = []
    for context_id in range(args.contexts):
        acc = evaluate.test_acc(
            model, test_datasets[context_id], verbose=False, context_id=context_id, allowed_classes=list(
                range(config['classes_per_context'] * context_id, config['classes_per_context'] * (context_id+1))
            ) if (args.scenario == "task" and not checkattr(args, 'singlehead')) else None, test_size=None,
        )
        if verbose:
            print(" - Context {}: {:.4f}".format(context_id+1, acc))
        accs.append(acc)
    average_accs = sum(accs) / args.contexts
    if verbose:
        print('=> average accuracy over all {} contexts: {:.4f}\n\n'.format(args.contexts, average_accs))
    # -write out to text file
    file_name = "{}/acc-{}{}.txt".format(args.r_dir, param_stamp,
                                         "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
    output_file = open(file_name, 'w')
    output_file.write('{}\n'.format(average_accs))
    output_file.close()




if __name__ == '__main__':
    # -load input-arguments
    args = handle_inputs()
    # -run experiment
    run(args, verbose=True)