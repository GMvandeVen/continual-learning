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
from models import define_models as define
from models.cl.continual_learner import ContinualLearner
from models.cl.memory_buffer import MemoryBuffer
from models.cl import fromp_optimizer
from train.train_task_based import train_cl, train_fromp, train_gen_classifier
from params import options
from params.param_stamp import get_param_stamp, get_param_stamp_from_args, visdom_name
from params.param_values import set_method_options,check_for_errors,set_default_values
from eval import evaluate, callbacks as cb
from visual import visual_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'main': True}
    # Define input options
    parser = options.define_args(filename="main", description='Run an individual continual learning experiment '
                                                              'using the "academic continual learning setting".')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Parse, process and check chosen options
    args = parser.parse_args()
    set_method_options(args)                         # -if a method's "convenience"-option is chosen, select components
    set_default_values(args, also_hyper_params=True) # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                 # -check whether incompatible options are selected
    return args


def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    if checkattr(args, 'get_stamp'):
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir,
        normalize=checkattr(args, "normalize"), verbose=verbose, exception=(args.seed==0),
        singlehead=checkattr(args, 'singlehead'), train_set_per_class=checkattr(args, 'gen_classifier')
    )
    # The experiments in this script follow the academic continual learning setting,
    # the above lines of code therefore load both the 'context set' and the 'data stream'

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

    #----------------------#
    #----- CLASSIFIER -----#
    #----------------------#

    # Define the classifier
    if verbose:
        print("\n\n " + ' DEFINE THE CLASSIFIER '.center(70, '*'))
    model = define.define_classifier(args=args, config=config, device=device, depth=depth)

    # Some type of classifiers consist of multiple networks
    n_networks = len(train_datasets) if (checkattr(args, 'separate_networks') or
                                         checkattr(args, 'gen_classifier')) else 1

    # Go through all networks to ...
    for network_id in range(n_networks):
        model_to_set = getattr(model, 'context{}'.format(network_id+1)) if checkattr(args, 'separate_networks') else (
            getattr(model, 'vae{}'.format(network_id)) if checkattr(args, 'gen_classifier') else model
        )
        # ... initialize / use pre-trained / freeze model-parameters, and
        define.init_params(model_to_set, args)
        # ... define optimizer (only include parameters that "requires_grad")
        if not checkattr(args, 'fromp'):
            model_to_set.optim_list = [{'params': filter(lambda p: p.requires_grad, model_to_set.parameters()),
                                        'lr': args.lr}]
            model_to_set.optim_type = args.optimizer
            if model_to_set.optim_type in ("adam", "adam_reset"):
                model_to_set.optimizer = optim.Adam(model_to_set.optim_list, betas=(0.9, 0.999))
            elif model_to_set.optim_type=="sgd":
                model_to_set.optimizer = optim.SGD(model_to_set.optim_list,
                                                   momentum=args.momentum if hasattr(args, 'momentum') else 0.)

    # On what scenario will model be trained? If needed, indicate whether singlehead output / how to set active classes.
    model.scenario = args.scenario
    model.classes_per_context = config['classes_per_context']
    model.singlehead = checkattr(args, 'singlehead')
    model.neg_samples = args.neg_samples if hasattr(args, 'neg_samples') else "all"

    # Print some model-characteristics on the screen
    if verbose:
        if checkattr(args, 'gen_classifier') or checkattr(args, 'separate_networks'):
            message = '{} copies of:'.format(len(train_datasets))
            utils.print_model_info(model.vae0 if checkattr(args, 'gen_classifier') else model.context1, message=message)
        else:
            utils.print_model_info(model)

    # -------------------------------------------------------------------------------------------------#

    # ----------------------------------------------------#
    # ----- CL-STRATEGY: CONTEXT-SPECIFIC COMPONENTS -----#
    # ----------------------------------------------------#

    # XdG: create for every context a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and checkattr(args, 'xdg') and args.gating_prop > 0.:
        model.mask_dict = {}
        for context_id in range(args.contexts):
            model.mask_dict[context_id + 1] = {}
            for i in range(model.fcE.layers):
                layer = getattr(model.fcE, "fcLayer{}".format(i + 1)).linear
                if context_id == 0:
                    model.excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(args.gating_prop * n_units), replace=False)
                model.mask_dict[context_id + 1][i] = gated_units

    #-------------------------------------------------------------------------------------------------#

    #-------------------------------------------------#
    #----- CL-STRATEGY: PARAMETER REGULARIZATION -----#
    #-------------------------------------------------#

    # Options for computing the Fisher Information matrix (e.g., EWC, Online-EWC, KFAC-EWC, NCL)
    use_fisher = hasattr(args, 'importance_weighting') and args.importance_weighting=="fisher" and \
                 (checkattr(args, 'precondition') or checkattr(args, 'weight_penalty'))
    if isinstance(model, ContinualLearner) and use_fisher:
        # -how to estimate the Fisher Information
        model.fisher_n = args.fisher_n if hasattr(args, 'fisher_n') else None
        model.fisher_labels = args.fisher_labels if hasattr(args, 'fisher_labels') else 'all'
        model.fisher_batch = args.fisher_batch if hasattr(args, 'fisher_batch') else 1
        # -options relating to 'Offline EWC' (Kirkpatrick et al., 2017) and 'Online EWC' (Schwarz et al., 2018)
        model.offline = checkattr(args, 'offline')
        if not model.offline:
            model.gamma = args.gamma if hasattr(args, 'gamma') else 1.
        # -if requested, initialize Fisher with prior
        if checkattr(args, 'fisher_init'):
            model.data_size = args.data_size  #-> sets how strong the prior is
            model.context_count = 1           #-> makes that already on the first context regularization will happen
            if model.fisher_kfac:
                model.initialize_kfac_fisher()
            else:
                model.initialize_fisher()

    # Parameter regularization by adding a weight penalty (e.g., EWC, SI, NCL, EWC-KFAC)
    if isinstance(model, ContinualLearner) and checkattr(args, 'weight_penalty'):
        model.weight_penalty = True
        model.importance_weighting = args.importance_weighting
        model.reg_strength = args.reg_strength
        if model.importance_weighting=='si':
            model.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.1

    # Parameter regularization through pre-conditioning of the gradient (e.g., OWM, NCL)
    if isinstance(model, ContinualLearner) and checkattr(args, 'precondition'):
        model.precondition = True
        model.importance_weighting = args.importance_weighting
        model.alpha = args.alpha

    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------#
    #----- CL-STRATEGY: FUNCTIONAL REGULARIZATION -----#
    #--------------------------------------------------#

    # Should a distillation loss (i.e., soft targets) be used? (e.g., for LwF, but also for BI-R)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay'):
        model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
        model.KD_temp = args.temp if hasattr(args, 'temp') else 2.
        if args.replay=="current" and model.replay_targets=="soft":
            model.lwf_weighting = True

    # Should the FROMP-optimizer by used?
    if checkattr(args, 'fromp'):
        model.optimizer = fromp_optimizer.opt_fromp(model, lr=args.lr, tau=args.tau, betas=(0.9, 0.999))

    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # DGR: Should a separate generative model be trained to generate the data to be replayed?
    train_gen = True if (args.replay=="generative" and not checkattr(args, 'feedback')) else False
    if train_gen:
        if verbose:
            print("\n\n " + ' SEPARATE GENERATIVE MODEL '.center(70, '*'))
        # -specify architecture
        generator = define.define_vae(args=args, config=config, device=device, depth=depth)
        # -initialize parameters
        define.init_params(generator, args, verbose=verbose)
        # -set optimizer(s)
        generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()),
                                 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
        # -print architecture to screen
        if verbose:
            utils.print_model_info(generator)
    else:
        generator = None

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

    # Should a memory buffer be maintained? (e.g., for experience replay, FROMP or prototype-based classification)
    use_memory_buffer = checkattr(args, 'prototypes') or checkattr(args, 'add_buffer') \
                        or args.replay=="buffer" or checkattr(args, 'fromp')
    if isinstance(model, MemoryBuffer) and use_memory_buffer:
        model.use_memory_buffer = True
        model.budget_per_class = args.budget
        model.use_full_capacity = checkattr(args, 'use_full_capacity')
        model.sample_selection = args.sample_selection if hasattr(args, 'sample_selection') else 'random'
        model.norm_exemplars = (model.sample_selection=="herding")

    # Should the memory buffer be added to the training set of the current context?
    model.add_buffer = checkattr(args, 'add_buffer')

    # Should classification be done using prototypes as class templates?
    model.prototypes = checkattr(args, 'prototypes')

    # Relevant for iCaRL: whether to use binary distillation loss for previous classes
    if model.label=="Classifier":
        model.binaryCE = checkattr(args, 'bce')
        model.binaryCE_distill = checkattr(args, 'bce_distill')

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- PARAMETER STAMP -----#
    #---------------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        if verbose:
            print('\n\n' + ' PARAMETER STAMP '.center(70, '*'))
    param_stamp = get_param_stamp(
        args, model.name, replay_model_name=generator.name if train_gen else None,
        feature_extractor_name= feature_extractor.name if (feature_extractor is not None) else None, verbose=verbose,
    )

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Prepare for keeping track of performance during training for plotting in pdf
    plotting_dict = evaluate.initiate_plotting_dict(args.contexts) if (
            checkattr(args, 'pdf') or checkattr(args, 'results_dict')
    ) else None

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
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, replay=False if args.replay=="none" else True,
                        model=model if checkattr(args, 'feedback') else generator, contexts=args.contexts,
                        iters_per_context=args.iters if checkattr(args, 'feedback') else args.g_iters)
    ] if (train_gen or checkattr(args, 'feedback')) else [None]
    loss_cbs = [
        cb._gen_classifier_loss_cb(
            log=args.loss_log, classes=config['classes'], visdom=visdom if args.loss_log>args.iters else None,
        ) if checkattr(args, 'gen_classifier') else cb._classifier_loss_cb(
            log=args.loss_log, visdom=visdom, model=model, contexts=args.contexts, iters_per_context=args.iters,
        )
    ] if (not checkattr(args, 'feedback')) else generator_loss_cbs

    # Callbacks for evaluating and plotting generated / reconstructed samples
    no_samples = (checkattr(args, "no_samples") or feature_extractor is not None)
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=visdom, config=config, sample_size=args.sample_n,
                      test_datasets=None if checkattr(args, 'gen_classifier') else test_datasets)
    ] if (train_gen or checkattr(args, 'feedback') or checkattr(args, 'gen_classifier')) and not no_samples else [None]

    # Callbacks for reporting and visualizing accuracy
    # -after each [acc_log], for visdom
    eval_cbs = [
        cb._eval_cb(log=args.acc_log, test_datasets=test_datasets, visdom=visdom, iters_per_context=args.iters,
                    test_size=args.acc_n)
    ] if (not checkattr(args, 'prototypes')) and (not checkattr(args, 'gen_classifier')) else [None]
    # -after each context, for plotting in pdf (when using prototypes / generative classifier, this is also for visdom)
    context_cbs = [
        cb._eval_cb(log=args.iters, test_datasets=test_datasets, plotting_dict=plotting_dict,
                    visdom=visdom if checkattr(args, 'prototypes') or checkattr(args, 'gen_classifier') else None,
                    iters_per_context=args.iters, test_size=args.acc_n, S=args.eval_s if hasattr(args, 'eval_s') else 1)
    ]

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    # Should a baseline be used (i.e., 'joint training' or 'cummulative training')?
    baseline = 'joint' if checkattr(args, 'joint') else ('cummulative' if checkattr(args, 'cummulative') else 'none')

    # Train model
    if args.train:
        if verbose:
            print('\n\n' + ' TRAINING '.center(70, '*'))
        # -keep track of training-time
        if args.time:
            start = time.time()
        # -select correct training function
        train_fn = train_fromp if checkattr(args, 'fromp') else (
            train_gen_classifier if checkattr(args, 'gen_classifier') else train_cl
        )
        # -perform training
        train_fn(
            model, train_datasets, iters=args.iters, batch_size=args.batch, baseline=baseline,
            sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=loss_cbs, context_cbs=context_cbs,
            # -if using generative replay with a separate generative model:
            generator=generator, gen_iters=args.g_iters if hasattr(args, 'g_iters') else args.iters,
            gen_loss_cbs=generator_loss_cbs,
        )
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
    for i in range(args.contexts):
        acc = evaluate.test_acc(
            model, test_datasets[i], verbose=False, test_size=None, context_id=i, allowed_classes=list(
                range(config['classes_per_context']*i, config['classes_per_context']*(i+1))
            ) if (args.scenario=="task" and not checkattr(args, 'singlehead')) else None,
        )
        if verbose:
            print(" - Context {}: {:.4f}".format(i + 1, acc))
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
    # -if requested, also save the results-dict (with accuracy after each task)
    if checkattr(args, 'results_dict'):
        file_name = "{}/dict-{}--n{}{}".format(args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n,
                                               "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
        utils.save_object(plotting_dict, file_name)

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if checkattr(args, 'pdf'):
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = visual_plt.open_pdf(plot_name)
        # -show samples and reconstructions (either from main model or from separate generator)
        if checkattr(args, 'feedback') or args.replay=="generative" or checkattr(args, 'gen_classifier'):
            evaluate.show_samples(
                model if checkattr(args, 'feedback') or checkattr(args, 'gen_classifier') else generator, config,
                size=args.sample_n, pdf=pp
            )
            if not checkattr(args, 'gen_classifier'):
                for i in range(args.contexts):
                    evaluate.show_reconstruction(model if checkattr(args, 'feedback') else generator,
                                                 test_datasets[i], config, pdf=pp, context=i+1)
        figure_list = []  #-> create list to store all figures to be plotted
        # -generate all figures (and store them in [figure_list])
        plot_list = []
        for i in range(args.contexts):
            plot_list.append(plotting_dict["acc per context"]["context {}".format(i + 1)])
        figure = visual_plt.plot_lines(
            plot_list, x_axes=plotting_dict["x_context"],
            line_names=['context {}'.format(i + 1) for i in range(args.contexts)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [plotting_dict["average"]], x_axes=plotting_dict["x_context"],
            line_names=['average all contexts so far']
        )
        figure_list.append(figure)
        # -add figures to pdf
        for figure in figure_list:
            pp.savefig(figure)
        # -close pdf
        pp.close()
        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    # -load input-arguments
    args = handle_inputs()
    # -run experiment
    run(args, verbose=True)