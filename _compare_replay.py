#!/usr/bin/env python3
import argparse
import os
from param_stamp import get_param_stamp_from_args
import visual_plt
import numpy as np
import main
from param_values import set_default_values


description = 'Evaluate variants of "exact replay" as function of available memory budget.'
parser = argparse.ArgumentParser('./_compare_replay.py', description=description)
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")

# model architecture parameters
model_params = parser.add_argument_group('Parameters Main Model')
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
train_params.add_argument('--lr', type=float,  help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# exemplar parameters
icarl_params = parser.add_argument_group('Exemplar Parameters')
icarl_params.add_argument('--budget', type=int, default=2000, dest="budget", help="how many exemplars can be stored?")
icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
# - generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='# of latent variables (def=100)')
# - hyper-parameters
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--gen-iters', type=int, dest="g_iters", help="# batches to optimize generator (def=[iters])")
gen_params.add_argument('--lr-gen', type=float, help="learning rate (separate) generator (default: lr)")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")


## Load input-arguments
args = parser.parse_args()


## Memory budget values to compare
budget_list_permMNIST = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
budget_list_splitMNIST = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]



def get_prec(args, ext=""):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if not os.path.isfile('{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)):
        print(" ...running: ... ")
        main.run(args)
    # -get average precision
    fileName = '{}/prec{}-{}.txt'.format(args.r_dir, ext, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision
    return ave


def collect_all(method_dict, seed_list, args, ext="", name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_prec(args, ext=ext)
    # -return updated dictionary with results
    return method_dict




if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args, also_hyper_params=False)
    # -set other default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## Select correct memory budget list
    budget_list = budget_list_permMNIST if args.experiment=="permMNIST" else budget_list_splitMNIST

    ## Add non-optional input argument that will be the same for all runs
    args.ewc = False
    args.ewc_lambda = 5000.
    args.si = False
    args.si_c = 0.1
    args.xdg = False
    args.gating_prop = 0.
    args.feedback = False
    args.log_per_task = True

    ## Add input arguments that will be different for different runs
    args.replay = "none"
    args.distill = False
    args.use_exemplars = False
    args.add_exemplars = False
    args.icarl = False
    args.bce_distill = False
    # args.seed could of course also vary!



    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###### BASELINE #########

    args.replay = "none"
    BASE = {}
    BASE = collect_all(BASE, seed_list, args, name="None")


    ###### GENERATIVE REPLAY #########

    args.replay = "generative"
    DGR = {}
    DGR = collect_all(DGR, seed_list, args, name="DGR")

    args.distill = True
    DGRD = {}
    DGRD = collect_all(DGRD, seed_list, args, name="DGR + distill")


    ###### EXACT REPLAY VARIANTS #########

    ## Replay during training
    args.replay = "exemplars"
    args.distill = False
    EXR = {}
    for budget in budget_list:
        args.budget = budget
        EXR[budget] = {}
        EXR[budget] = collect_all(EXR[budget], seed_list, args,
                                  name="Replay Stored Data - budget = {}".format(budget))

    ## Replay during execution
    args.replay = "none"
    args.use_exemplars = True
    EXU = {}
    for budget in budget_list:
        args.budget = budget
        EXU[budget] = {}
        EXU[budget] = collect_all(EXU[budget], seed_list, args,
                                  name="Classify with Exemplars - budget = {}".format(budget))

    ## Replay during training & during execution
    args.replay = "exemplars"
    args.use_exemplars = True
    EXRU = {}
    for budget in budget_list:
        args.budget = budget
        EXRU[budget] = {}
        EXRU[budget] = collect_all(EXRU[budget], seed_list, args,
                                   name="Replay Stored Data & Classify with Exemplars - budget = {}".format(budget))

    ## iCaRL (except not necessarily with "herding" and "norm_exemplars")!
    if args.scenario=="class":
        args.replay = "none"
        args.use_exemplars = True
        args.bce = True
        args.bce_distill = True
        args.add_exemplars = True
        ICARL = {}
        for budget in budget_list:
            args.budget = budget
            ICARL[budget] = {}
            ICARL[budget] = collect_all(ICARL[budget], seed_list, args,
                                        name="iCaRL - budget = {}".format(budget))


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summaryGenRep-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # set scale of y-axis
    y_lim = [0,1] if args.scenario=="class" else None

    # Methods for comparison
    h_lines = [np.mean([BASE[seed] for seed in seed_list])]
    h_lines.append(np.mean([DGR[seed] for seed in seed_list]))
    h_lines.append(np.mean([DGRD[seed] for seed in seed_list]))
    h_errors = [np.sqrt(np.var([BASE[seed] for seed in seed_list]) / (len(seed_list)-1))] if args.n_seeds>1 else None
    if args.n_seeds>1:
        h_errors.append(np.sqrt(np.var([DGR[seed] for seed in seed_list]) / (len(seed_list) - 1)))
        h_errors.append(np.sqrt(np.var([DGRD[seed] for seed in seed_list]) / (len(seed_list) - 1)))
    h_labels = ["None", "DGR", "DGR+distill"]
    h_colors = ["grey", "indianred", "red"]


    # Different variants of exact replay
    # -prepare
    ave_EXR = []
    sem_EXR = []
    ave_EXU = []
    sem_EXU = []
    ave_EXRU = []
    sem_EXRU = []
    if args.scenario=="class":
        ave_ICARL = []
        sem_ICARL = []

    for budget in budget_list:
        all_entries = [EXR[budget][seed] for seed in seed_list]
        ave_EXR.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_EXR.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        all_entries = [EXU[budget][seed] for seed in seed_list]
        ave_EXU.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_EXU.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        all_entries = [EXRU[budget][seed] for seed in seed_list]
        ave_EXRU.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_EXRU.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        if args.scenario=="class":
            all_entries = [ICARL[budget][seed] for seed in seed_list]
            ave_ICARL.append(np.mean(all_entries))
            if args.n_seeds > 1:
                sem_ICARL.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

    # -collect
    lines = [ave_EXR, ave_EXU, ave_EXRU]
    errors = [sem_EXR, sem_EXU, sem_EXRU] if args.n_seeds > 1 else None
    line_names = ["Replay exemplars", "Classify with exemplars", "Replay & classify with exemplars"]
    colors = ["black", "grey", "darkgrey"]
    if args.scenario=="class":
        lines.append(ave_ICARL)
        line_names.append("iCaRL")
        colors.append("brown")
        if args.n_seeds>1:
            errors.append(sem_ICARL)

    # -plot
    figure = visual_plt.plot_lines(
        lines, x_axes=budget_list, ylabel="average precision (after all tasks)", title=title, x_log=True, ylim=y_lim,
        line_names=line_names, xlabel="Total memory budget", with_dots=True, colors=colors, list_with_errors=errors,
        h_lines=h_lines, h_errors=h_errors, h_labels=h_labels, h_colors=h_colors,
    )
    figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))