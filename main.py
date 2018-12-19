#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
from torch import nn
import visual_plt
import utils
import evaluate
from data import get_multitask_experiment
from encoder import Classifier
from vae_models import AutoEncoder
import callbacks as cb
from train import train_cl
from continual_learner import ContinualLearner


parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, default=5, help='number of tasks')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, default=400, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, default=2000, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback', action="store_true", help="equip model with feedback connections")
replay_params.add_argument('--replay', type=str, default='none', choices=['offline', 'exact', 'generative', 'none', 'current'])
replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
# -generative model parameters
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, default=5000.,dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
cl_params.add_argument('--gamma', type=float, default=1., help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, default=0.1, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--XdG', type=float, default=0., dest="gating_prop",help="XdG: prop neurons per layer to gate")

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

    # Set default arguments
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    # -if XdG is selected but not the incremental task learning scenario, give error
    if (not args.scenario=="task") and args.gating_prop>0:
        raise ValueError("'XdG' only works for the incremental task learning scenario.")
    # -if EWC, SI or XdG is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.gating_prop>0):
        raise NotImplementedError("EWC, SI and XdG are not supported with feedback connections.")
    # -if XdG is selected together with replay of any kind, give error
    if args.gating_prop>0 and (not args.replay=="none"):
        raise NotImplementedError("XdG is not supported with '{}' replay.".format(args.replay))
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

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
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,
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
            fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.gating_prop>0 else False,
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

    # Set loss-function for reconstruction
    if args.feedback:
        model.recon_criterion = nn.BCELoss(size_average=True)


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------#
    #----- CL-STRATEGY: ALLOCATION -----#
    #-----------------------------------#

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.gamma = args.gamma
        model.online = args.online
        model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and args.gating_prop>0:
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
    model.replay_targets = "soft" if args.distill else "hard"
    model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        # -specify architecture
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        ).to(device)
        # -set optimizer(s)
        generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()), 'lr': args.lr}]
        generator.optim_type = args.optimizer
        if generator.optim_type in ("adam", "adam_reset"):
            generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
        elif generator.optim_type == "sgd":
            generator.optimizer = optim.SGD(generator.optim_list)
        # -set reconstruction criterion
        generator.recon_criterion = nn.BCELoss(size_average=True)
    else:
        generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#

    # Get parameter-stamp (and print on screen)
    param_stamp = utils.get_param_stamp(
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

    # Prepare for plotting
    # -open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, param_stamp)) if args.pdf else None
    # -define [precision_dict] to keep track of performance during training for later plotting
    precision_dict = evaluate.initiate_precision_dict(args.tasks)
    # -visdom-settings
    if args.visdom:
        env_name = "{exp}{tasks}-{scenario}".format(exp=args.experiment, tasks=args.tasks, scenario=args.scenario)
        graph_name = "{fb}{mode}{syn}{ewc}{XdG}".format(
            fb="1M-" if args.feedback else "", mode=args.replay,
            syn="-si{}".format(args.si_c) if args.si else "",
            ewc="-ewc{}{}".format(args.ewc_lambda, "-O{}".format(args.gamma) if args.online else "") if args.ewc else "",
            XdG="" if args.gating_prop==0 else "-XdG{}".format(args.gating_prop)
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
                        iters_per_task=args.g_iters, replay=False if args.replay=="none" else True)
    ] if (train_gen or args.feedback) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=args.tasks,
                           iters_per_task=args.iters, replay=False if args.replay=="none" else True)
    ] if (not args.feedback) else [None]

    # Callbacks for evaluating and plotting generated / reconstructed samples
    sample_cbs = [
        cb._sample_cb(log=args.sample_log, visdom=visdom, config=config, test_datasets=test_datasets,
                      sample_size=args.sample_n, iters_per_task=args.g_iters)
    ] if (train_gen or args.feedback) else [None]

    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log])
    eval_cb = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=visdom, iters_per_task=args.iters,scenario=args.scenario,
        collate_fn=utils.label_squeezing_collate_fn, test_size=args.prec_n, classes_per_task=classes_per_task,
        task_mask=True if isinstance(model, ContinualLearner) and (args.gating_prop>0) else False
    )
    # -pdf: for summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, precision_dict=precision_dict, scenario=args.scenario,
        collate_fn=utils.label_squeezing_collate_fn, iters_per_task=args.iters, classes_per_task=classes_per_task,
        task_mask = True if isinstance(model, ContinualLearner) and (args.gating_prop > 0) else False
    )
    # -collect them in <lists>
    eval_cbs = [eval_cb, eval_cb_full]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    print("--> Training:")
    # Keep track of training-time
    start = time.time()
    # Train model
    train_cl(
        model, train_datasets, replay_mode=args.replay, scenario=args.scenario, classes_per_task=classes_per_task,
        iters=args.iters, batch_size=args.batch, collate_fn=utils.label_squeezing_collate_fn,
        visualize=True if args.visdom else False,
        generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
        sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
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

    print('\n\n--> Evaluation ("incremental {} learning scenario"):'.format(args.scenario))

    # Generation (plot in pdf)
    if (pp is not None) and train_gen:
        evaluate.show_samples(generator, config, size=args.sample_n, pdf=pp)
    if (pp is not None) and args.feedback:
        evaluate.show_samples(model, config, size=args.sample_n, pdf=pp)

    # Reconstruction (plot in pdf)
    if (pp is not None) and (train_gen or args.feedback):
        for i in range(args.tasks):
            if args.feedback:
                evaluate.show_reconstruction(model, test_datasets[i], config, pdf=pp, task=i+1)
            else:
                evaluate.show_reconstruction(generator, test_datasets[i], config, pdf=pp, task=i+1)

    # Classifier (print on screen & write to file)
    if args.scenario=="task":
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None,
            task_mask=True if isinstance(model, ContinualLearner) and args.gating_prop>0 else False,
            task=i+1, allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1)))
        ) for i in range(args.tasks)]
    else:
        precs = [evaluate.validate(
            model, test_datasets[i], verbose=False, test_size=None, task=i+1
        ) for i in range(args.tasks)]
    print("\n Precision on test-set:")
    for i in range(args.tasks):
        print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
    average_precs = sum(precs) / args.tasks
    print('=> average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs))


    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # Average precision on full test set (no restrictions on which nodes can be predicted: "incremental" / "singlehead")
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs))
    output_file.close()

    # Precision-dictionary
    file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
    utils.save_object(precision_dict, file_name)

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # If requested, generate pdf
    if pp is not None:
        # -create list to store all figures to be plotted.
        figure_list = []
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
        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

    # Close pdf
    if pp is not None:
        pp.close()



if __name__ == '__main__':
    args = parser.parse_args()
    run(args)