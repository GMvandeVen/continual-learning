#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import visual_plt
import utils
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

# exemplar parameters
icarl_params = parser.add_argument_group('Exemplar Parameters')
icarl_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
icarl_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
icarl_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task dataset")
icarl_params.add_argument('--budget', type=int, default=2000, dest="budget", help="how many exemplars can be stored?")
icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="# iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="# iters after which to plot samples")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")



def run(args):

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
    # -if EWC, SI or XdG is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.xdg or args.icarl):
        raise NotImplementedError("EWC, SI, XdG and iCaRL are not supported with feedback connections.")
    # -if binary classification loss is selected together with 'feedback', give error
    if args.feedback and args.bce:
        raise NotImplementedError("Binary classification loss not supported with feedback connections.")
    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if args.xdg and (not args.replay=="none") and (args.ewc or args.si):
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
        _ = get_param_stamp_from_args(args=args)
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

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
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed==0 else False,
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
            binaryCE=args.bce, binaryCE_distill=args.bce_distill,
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
    param_stamp = get_param_stamp(
        args, model.name, verbose=True, replay=True if (not args.replay=="none") else False,
        replay_model_name=generator.name if (args.replay=="generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    # -main model
    print("\n")
    utils.print_model_info(model, title="MAIN MODEL")
    # -generator
    if generator is not None:
        utils.print_model_info(generator, title="GENERATOR")

    # Prepare for plotting in visdom
    # -define [precision_dict] to keep track of performance during training for storing and for later plotting in pdf
    precision_dict = evaluate.initiate_precision_dict(args.tasks)
    precision_dict_exemplars = evaluate.initiate_precision_dict(args.tasks) if args.use_exemplars else None
    # -visdom-settings
    if args.visdom:
        env_name = "{exp}{tasks}-{scenario}".format(exp=args.experiment, tasks=args.tasks, scenario=args.scenario)
        graph_name = "{fb}{replay}{syn}{ewc}{xdg}{icarl}{bud}".format(
            fb="1M-" if args.feedback else "", replay="{}{}".format(args.replay, "D" if args.distill else ""),
            syn="-si{}".format(args.si_c) if args.si else "",
            ewc="-ewc{}{}".format(args.ewc_lambda,"-O{}".format(args.gamma) if args.online else "") if args.ewc else "",
            xdg="" if (not args.xdg) or args.gating_prop==0 else "-XdG{}".format(args.gating_prop),
            icarl="-iCaRL" if (args.use_exemplars and args.add_exemplars and args.bce and args.bce_distill) else "",
            bud="-bud{}".format(args.budget) if (
                    args.use_exemplars or args.add_exemplars or args.replay=="exemplars"
            ) else "",
        )
        visdom = {'env': env_name, 'graph': graph_name}
        if args.use_exemplars:
            visdom_exemplars = {'env': env_name, 'graph': "{}-EX".format(graph_name)}
    else:
        visdom = visdom_exemplars = None


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
    eval_cb = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=visdom, precision_dict=None, iters_per_task=args.iters,
        test_size=args.prec_n, classes_per_task=classes_per_task, scenario=scenario,
    )
    # -pdf / reporting: summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, precision_dict=precision_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task, scenario=scenario,
    )
    # -with exemplars (both for visdom & reporting / pdf)
    eval_cb_exemplars = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, visdom=visdom_exemplars, classes_per_task=classes_per_task,
        precision_dict=precision_dict_exemplars, scenario=scenario, iters_per_task=args.iters,
        with_exemplars=True,
    ) if args.use_exemplars else None
    # -collect them in <lists>
    eval_cbs = [eval_cb, eval_cb_full]
    eval_cbs_exemplars = [eval_cb_exemplars]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    print("--> Training:")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(
        model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
        iters=args.iters, batch_size=args.batch,
        generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
        sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
        eval_cbs_exemplars=eval_cbs_exemplars, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    )
    # Get total training-time in seconds, and write to file
    training_time = time.time() - start
    time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
    time_file.write('{}\n'.format(training_time))
    time_file.close()


    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    print("\n\n--> Evaluation ({}-incremental learning scenario):".format(args.scenario))

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
    ) for i in range(args.tasks)]
    print("\n Precision on test-set (softmax classification):")
    for i in range(args.tasks):
        print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
    average_precs = sum(precs) / args.tasks
    print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs))

    # -with exemplars
    if args.use_exemplars:
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=True,
            allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        ) for i in range(args.tasks)]
        print("\n Precision on test-set (classification using exemplars):")
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        average_precs_ex = sum(precs) / args.tasks
        print('=> average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs_ex))
    print("\n")


    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average precision on full test set
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs_ex if args.use_exemplars else average_precs))
    output_file.close()
    # -precision-dict
    file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
    utils.save_object(precision_dict_exemplars if args.use_exemplars else precision_dict, file_name)

    # Average precision on full test set not evaluated using exemplars (i.e., using softmax on final layer)
    if args.use_exemplars:
        output_file = open("{}/prec_noex-{}.txt".format(args.r_dir, param_stamp), 'w')
        output_file.write('{}\n'.format(average_precs))
        output_file.close()
        # -precision-dict:
        file_name = "{}/dict_noex-{}".format(args.r_dir, param_stamp)
        utils.save_object(precision_dict, file_name)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if args.pdf:
        # -open pdf
        pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, param_stamp))

        # -show samples and reconstructions (either from main model or from separate generator)
        if args.feedback or args.replay=="generative":
            evaluate.show_samples(model if args.feedback else generator, config, size=args.sample_n, pdf=pp)
            for i in range(args.tasks):
                evaluate.show_reconstruction(model if args.feedback else generator, test_datasets[i], config, pdf=pp,
                                             task=i+1)

        # -show metrics reflecting progression during training
        figure_list = []  #-> create list to store all figures to be plotted
        # -generate all figures (and store them in [figure_list])
        figure = visual_plt.plot_lines(
            precision_dict["all_tasks"], x_axes=precision_dict["x_task"],
            line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [precision_dict["average"]], x_axes=precision_dict["x_task"],
            line_names=['average all tasks so far']
        )
        figure_list.append(figure)
        if args.use_exemplars:
            figure = visual_plt.plot_lines(
                precision_dict_exemplars["all_tasks"], x_axes=precision_dict_exemplars["x_task"],
                line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
            )
            figure_list.append(figure)
        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

        # -close pdf
        pp.close()




if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args)