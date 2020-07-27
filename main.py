#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import visual_plt
import utils
import pandas as pd
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_multitask_experiment
from encoder import Classifier
from vae_models import AutoEncoder
import callbacks as cb
from train import train_cl
from continual_learner import ContinualLearner
from exemplars import ExemplarHandler
from replayer import Replayer
from param_values import set_default_values


parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                    ' examples (only if --bce & --scenario="class")')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback', action="store_true", help="equip model with feedback connections")
replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']
replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--agem', action='store_true', help="use gradient of replay as inequality constraint")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', action='store_true', help="Use 'Context-dependent Gating' (Masse et al, 2018)")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# data storage ('exemplars') parameters
store_params = parser.add_argument_group('Data Storage Parameters')
store_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
store_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
store_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")
store_params.add_argument('--budget', type=int, default=1000, dest="budget", help="how many samples can be stored?")
store_params.add_argument('--herding', action='store_true', help="use herding to select stored data (instead of random)")
store_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="# iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="# iters after which to plot samples")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")



def run(args, verbose=False):

    # Set default arguments & check for incompatible options
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    # -if [iCaRL] is selected, select all accompanying options
    if hasattr(args, "icarl") and args.icarl:
        args.use_exemplars = True
        args.add_exemplars = True
        args.bce = True
        args.bce_distill = True
    # -if XdG is selected but not the Task-IL scenario, give error
    if (not args.scenario=="task") and args.xdg:
        raise ValueError("'XdG' is only compatible with the Task-IL scenario.")
    # -if EWC, SI, XdG, A-GEM or iCaRL is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.xdg or args.icarl or args.agem):
        raise NotImplementedError("EWC, SI, XdG, A-GEM and iCaRL are not supported with feedback connections.")
    # -if A-GEM is selected without any replay, give warning
    if args.agem and args.replay=="none":
        raise Warning("The '--agem' flag is selected, but without any type of replay. "
                      "For the original A-GEM method, also select --replay='exemplars'.")
    # -if EWC, SI, XdG, A-GEM or iCaRL is selected together with offline-replay, give error
    if args.replay=="offline" and (args.ewc or args.si or args.xdg or args.icarl or args.agem):
        raise NotImplementedError("Offline replay cannot be combined with EWC, SI, XdG, A-GEM or iCaRL.")
    # -if binary classification loss is selected together with 'feedback', give error
    if args.feedback and args.bce:
        raise NotImplementedError("Binary classification loss not supported with feedback connections.")
    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if (args.xdg and args.gating_prop>0) and (not args.replay=="none") and (args.ewc or args.si):
        raise NotImplementedError("XdG is not supported with both '{}' replay and EWC / SI.".format(args.replay))
        #--> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating backward step on EWC/SI-loss also for each mask separately)
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario=="class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario=="task":
        scenario="domain"

    # If only want param-stamp, get it printed to screen and exit
    if hasattr(args, "get_stamp") and args.get_stamp:
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
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
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
        verbose=verbose, exception=True if args.seed==0 else False,
    )


    #-------------------------------------------------------------------------------------------------#

    #------------------------------#
    #----- MODEL (CLASSIFIER) -----#
    #------------------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        model.lamda_pl = 1. #--> to make that this VAE is also trained to classify
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
            binaryCE=args.bce, binaryCE_distill=args.bce_distill, AGEM=args.agem,
        ).to(device)

    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type=="sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))


    #-------------------------------------------------------------------------------------------------#

    #----------------------------------#
    #----- CL-STRATEGY: EXEMPLARS -----#
    #----------------------------------#

    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.add_exemplars or args.replay=="exemplars"):
        model.memory_budget = args.budget
        model.norm_exemplars = args.norm_exemplars
        model.herding = args.herding


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- CL-STRATEGY: ALLOCATION -----#
    #-----------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        if args.ewc:
            model.fisher_n = args.fisher_n
            model.gamma = args.gamma
            model.online = args.online
            model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        if args.si:
            model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and (args.xdg and args.gating_prop>0):
        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(args.tasks):
            mask_dict[task_id+1] = {}
            for i in range(model.fcE.layers):
                layer = getattr(model.fcE, "fcLayer{}".format(i+1)).linear
                if task_id==0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(args.gating_prop*n_units), replace=False)
                mask_dict[task_id+1][i] = gated_units
        model.mask_dict = mask_dict
        model.excit_buffer_list = excit_buffer_list


    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, Replayer):
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        # -set optimizer(s)
        generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr_gen}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
    else:
        generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, model.name, verbose=verbose, replay=True if (not args.replay=="none") else False,
        replay_model_name=generator.name if (args.replay=="generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.pdf or args.metrics:
        # -define [metrics_dict] to keep track of performance during training for storing & for later plotting in pdf
        metrics_dict = evaluate.initiate_metrics_dict(n_tasks=args.tasks, scenario=args.scenario)
        # -evaluate randomly initiated model on all tasks & store accuracies in [metrics_dict] (for calculating metrics)
        if not args.use_exemplars:
            metrics_dict = evaluate.intial_accuracy(model, test_datasets, metrics_dict,
                                                    classes_per_task=classes_per_task, scenario=scenario,
                                                    test_size=None, no_task_mask=False)
    else:
        metrics_dict = None

    # Prepare for plotting in visdom
    # -visdom-settings
    if args.visdom:
        env_name = "{exp}{tasks}-{scenario}".format(exp=args.experiment, tasks=args.tasks, scenario=args.scenario)
        graph_name = "{fb}{replay}{syn}{ewc}{xdg}{icarl}{bud}".format(
            fb="1M-" if args.feedback else "",
            replay="{}{}{}".format(args.replay, "D" if args.distill else "", "-aGEM" if args.agem else ""),
            syn="-si{}".format(args.si_c) if args.si else "",
            ewc="-ewc{}{}".format(args.ewc_lambda,"-O{}".format(args.gamma) if args.online else "") if args.ewc else "",
            xdg="" if (not args.xdg) or args.gating_prop==0 else "-XdG{}".format(args.gating_prop),
            icarl="-iCaRL" if (args.use_exemplars and args.add_exemplars and args.bce and args.bce_distill) else "",
            bud="-bud{}".format(args.budget) if (
                    args.use_exemplars or args.add_exemplars or args.replay=="exemplars"
            ) else "",
        )
        visdom = {'env': env_name, 'graph': graph_name}
    else:
        visdom = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Callbacks for reporting on and visualizing loss
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom, model=model if args.feedback else generator, tasks=args.tasks,
                        iters_per_task=args.iters if args.feedback else args.g_iters,
                        replay=False if args.replay=="none" else True)
    ] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=args.tasks,
                           iters_per_task=args.iters, replay=False if args.replay=="none" else True)
    ] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=visdom, config=config, test_datasets=test_datasets,
                      sample_size=args.sample_n, iters_per_task=args.iters if args.feedback else args.g_iters)
    ] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
    eval_cbs = [
        cb._eval_cb(log=args.prec_log, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    scenario=scenario, with_exemplars=False)
    ] if (not args.use_exemplars) else [None]
    #--> during training on a task, evaluation cannot be with exemplars as those are only selected after training
    #    (instead, evaluation for visdom is only done after each task, by including callback-function into [metric_cbs])

    # Callbacks for calculating statists required for metrics
    # -pdf / reporting: summary plots (i.e, only after each task) (when using exemplars, also for visdom)
    metric_cbs = [
        cb._metric_cb(log=args.iters, test_datasets=test_datasets,
                      classes_per_task=classes_per_task, metrics_dict=metrics_dict, scenario=scenario,
                      iters_per_task=args.iters, with_exemplars=args.use_exemplars),
        cb._eval_cb(log=args.iters, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    scenario=scenario, with_exemplars=True) if args.use_exemplars else None
    ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if verbose:
        print("\nTraining...")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(
        model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
        iters=args.iters, batch_size=args.batch,
        generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
        sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
        metric_cbs=metric_cbs, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    )
    # Get total training-time in seconds, and write to file
    if args.time:
        training_time = time.time() - start
        time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
        time_file.write('{}\n'.format(training_time))
        time_file.close()


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
    ) for i in range(args.tasks)]
    average_precs = sum(precs) / args.tasks
    # -print on screen
    if verbose:
        print("\n Precision on test-set{}:".format(" (softmax classification)" if args.use_exemplars else ""))
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs))

    # -with exemplars
    if args.use_exemplars:
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=True,
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        ) for i in range(args.tasks)]
        average_precs_ex = sum(precs) / args.tasks
        # -print on screen
        if verbose:
            print(" Precision on test-set (classification using exemplars):")
            for i in range(args.tasks):
                print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
            print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs_ex))

    if args.metrics:
        # Accuracy matrix
        if args.scenario in ('task', 'domain'):
            R = pd.DataFrame(data=metrics_dict['acc per task'],
                             index=['after task {}'.format(i + 1) for i in range(args.tasks)])
            R.loc['at start'] = metrics_dict['initial acc per task'] if (not args.use_exemplars) else [
                'NA' for _ in range(args.tasks)
            ]
            R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
            BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
                     R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks - 1)]
            FWTs = [0. if args.use_exemplars else (
                R.loc['after task {}'.format(i+1), 'task {}'.format(i + 2)] - R.loc['at start', 'task {}'.format(i+2)]
            ) for i in range(args.tasks-1)]
            forgetting = []
            for i in range(args.tasks - 1):
                forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
            R.loc['FWT (per task)'] = ['NA'] + FWTs
            R.loc['BWT (per task)'] = BWTs + ['NA']
            R.loc['F (per task)'] = forgetting + ['NA']
            BWT = sum(BWTs) / (args.tasks - 1)
            F = sum(forgetting) / (args.tasks - 1)
            FWT = sum(FWTs) / (args.tasks - 1)
            metrics_dict['BWT'] = BWT
            metrics_dict['F'] = F
            metrics_dict['FWT'] = FWT
            # -print on screen
            if verbose:
                print("Accuracy matrix")
                print(R)
                print("\nFWT = {:.4f}".format(FWT))
                print("BWT = {:.4f}".format(BWT))
                print("  F = {:.4f}\n\n".format(F))
        else:
            if verbose:
                # Accuracy matrix based only on classes in that task (i.e., evaluation as if Task-IL scenario)
                R = pd.DataFrame(data=metrics_dict['acc per task (only classes in task)'],
                                 index=['after task {}'.format(i + 1) for i in range(args.tasks)])
                R.loc['at start'] = metrics_dict[
                    'initial acc per task (only classes in task)'
                ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
                R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
                print("Accuracy matrix, based on only classes in that task ('as if Task-IL scenario')")
                print(R)

                # Accuracy matrix, always based on all classes
                R = pd.DataFrame(data=metrics_dict['acc per task (all classes)'],
                                 index=['after task {}'.format(i + 1) for i in range(args.tasks)])
                R.loc['at start'] = metrics_dict[
                    'initial acc per task (only classes in task)'
                ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
                R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
                print("\nAccuracy matrix, always based on all classes")
                print(R)

                # Accuracy matrix, based on all classes thus far
                R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to trained task)'],
                                 index=['after task {}'.format(i + 1) for i in range(args.tasks)])
                print("\nAccuracy matrix, based on all classes up to the trained task")
                print(R)

            # Accuracy matrix, based on all classes up to the task being evaluated
            # (this is the accuracy-matrix used for calculating the metrics in the Class-IL scenario)
            R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to evaluated task)'],
                             index=['after task {}'.format(i + 1) for i in range(args.tasks)])
            R.loc['at start'] = metrics_dict[
                'initial acc per task (only classes in task)'
            ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
            R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
            BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
                     R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks-1)]
            FWTs = [0. if args.use_exemplars else (
                R.loc['after task {}'.format(i+1), 'task {}'.format(i+2)] - R.loc['at start', 'task {}'.format(i+2)]
            ) for i in range(args.tasks-1)]
            forgetting = []
            for i in range(args.tasks - 1):
                forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
            R.loc['FWT (per task)'] = ['NA'] + FWTs
            R.loc['BWT (per task)'] = BWTs + ['NA']
            R.loc['F (per task)'] = forgetting + ['NA']
            BWT = sum(BWTs) / (args.tasks-1)
            F = sum(forgetting) / (args.tasks-1)
            FWT = sum(FWTs) / (args.tasks-1)
            metrics_dict['BWT'] = BWT
            metrics_dict['F'] = F
            metrics_dict['FWT'] = FWT
            # -print on screen
            if verbose:
                print("\nAccuracy matrix, based on all classes up to the evaluated task")
                print(R)
                print("\n=> FWT = {:.4f}".format(FWT))
                print("=> BWT = {:.4f}".format(BWT))
                print("=>  F = {:.4f}\n".format(F))

    if verbose and args.time:
        print("=> Total training time = {:.1f} seconds\n".format(training_time))


    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average precision on full test set
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs_ex if args.use_exemplars else average_precs))
    output_file.close()
    # -metrics-dict
    if args.metrics:
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(metrics_dict, file_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
        pp = visual_plt.open_pdf(plot_name)

        # -show samples and reconstructions (either from main model or from separate generator)
        if args.feedback or args.replay=="generative":
            evaluate.show_samples(model if args.feedback else generator, config, size=args.sample_n, pdf=pp)
            for i in range(args.tasks):
                evaluate.show_reconstruction(model if args.feedback else generator, test_datasets[i], config, pdf=pp,
                                             task=i+1)

        # -show metrics reflecting progression during training
        figure_list = []  #-> create list to store all figures to be plotted

        # -generate all figures (and store them in [figure_list])
        key = "acc per task ({} task)".format("all classes up to trained" if scenario=='class' else "only classes in")
        plot_list = []
        for i in range(args.tasks):
            plot_list.append(metrics_dict[key]["task {}".format(i + 1)])
        figure = visual_plt.plot_lines(
            plot_list, x_axes=metrics_dict["x_task"],
            line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [metrics_dict["average"]], x_axes=metrics_dict["x_task"],
            line_names=['average all tasks so far']
        )
        figure_list.append(figure)

        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

        # -close pdf
        pp.close()

        # -print name of generated plot on screen
        if verbose:
            print("\nGenerated plot: {}\n".format(plot_name))



if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)