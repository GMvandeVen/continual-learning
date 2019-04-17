#!/usr/bin/env python3
import argparse
import os
import numpy as np
import utils
from param_stamp import get_param_stamp_from_args
import visual_plt
import main


description = 'Compare performance & training time of various continual learning methods.'
parser = argparse.ArgumentParser('./compare_time.py', description=description)
parser.add_argument('--seed', type=int, default=1111, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
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
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, default=2000, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
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
cl_params.add_argument('--lambda', type=float, default=5000.,dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--o-lambda', type=float, default=5000., help="--> online EWC: regularisation strength")
cl_params.add_argument('--gamma', type=float, default=1., help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--c', type=float, default=0.1, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', type=float, default=0.8, dest="xdg",help="XdG: prop neurons per layer to gate")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if not os.path.isfile('{}/prec-{}.txt'.format(args.r_dir, param_stamp)):
        print(" ... running ... ")
        main.run(args)
    # -get results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir, param_stamp))
    # -get average precisions & trainig-times
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    fileName = '{}/time-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    training_time = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return tuple with the results
    return (dict, ave, training_time)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict


if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    # -set default arguments
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

    ## Add non-optional input argument that will be the same for all runs
    args.bce = False
    args.bce_distill = False
    args.icarl = False
    args.use_exemplars = False
    args.add_exemplars = False
    args.budget = 2000
    args.herding = False
    args.norm_exemplars = False
    args.log_per_task = True

    ## As this script runs the comparions in the "RtF-paper" (van de Ven & Tolias, 2018, arXiv),
    ## the empirical Fisher Matrix is used for EWC
    args.emp_fi = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.feedback = False
    args.ewc = False
    args.online = False
    args.si = False
    args.gating_prop = 0.
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Offline
    args.replay = "offline"
    SO = {}
    SO = collect_all(SO, seed_list, args, name="Offline")

    ## None
    args.replay = "none"
    SN = {}
    SN = collect_all(SN, seed_list, args, name="None")


    ###----"XdG"----####

    ## XdG
    if args.scenario=="task":
        args.gating_prop = args.xdg
        SXDG = {}
        SXDG = collect_all(SXDG, seed_list, args, name="XdG")
        args.gating_prop = 0


    ###----"EWC / SI"----####

    ## EWC
    args.ewc = True
    SEWC = {}
    SEWC = collect_all(SEWC, seed_list, args, name="EWC")

    ## online EWC
    args.online = True
    args.ewc_lambda = args.o_lambda
    SOEWC = {}
    SOEWC = collect_all(SOEWC, seed_list, args, name="Online EWC")
    args.ewc = False
    args.online = False

    ## SI
    args.si = True
    SSI = {}
    SSI = collect_all(SSI, seed_list, args, name="SI")
    args.si = False


    ###----"REPLAY"----###

    ## LwF
    args.replay = "current"
    args.distill = True
    SLWF = {}
    SLWF = collect_all(SLWF, seed_list, args, name="LwF")
    args.distill = False

    ## DGR
    args.replay = "generative"
    SRP = {}
    SRP = collect_all(SRP, seed_list, args, name="DGR")

    ## DGR+distill
    args.replay = "generative"
    args.distill = True
    SRKD = {}
    SRKD = collect_all(SRKD, seed_list, args, name="DGR+distill")

    ## RtF
    args.replay = "generative"
    args.distill = True
    args.feedback = True
    ORKD = {}
    ORKD = collect_all(ORKD, seed_list, args, name="RtF")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    prec = {}
    ave_prec = {}
    train_time = {}

    ## Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:
        i = 0
        prec[seed] = [
            SO[seed][i]["average"], SN[seed][i]["average"],
            SRKD[seed][i]["average"], SRP[seed][i]["average"], ORKD[seed][i]["average"], SLWF[seed][i]["average"],
            SEWC[seed][i]["average"], SOEWC[seed][i]["average"], SSI[seed][i]["average"],
        ]

        i = 1
        ave_prec[seed] = [
            SO[seed][i], SN[seed][i],
            SRKD[seed][i], SRP[seed][i], ORKD[seed][i], SLWF[seed][i],
            SEWC[seed][i], SOEWC[seed][i], SSI[seed][i],
        ]

        i = 2
        train_time[seed] = [
            SO[seed][i], SN[seed][i],
            SRKD[seed][i], SRP[seed][i], ORKD[seed][i], SLWF[seed][i],
            SEWC[seed][i], SOEWC[seed][i], SSI[seed][i],
        ]

        if args.scenario=="task":
            prec[seed].append(SXDG[seed][0]["average"])
            ave_prec[seed].append(SXDG[seed][1])
            train_time[seed].append(SXDG[seed][2])


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    x_axes = SRKD[args.seed][0]["x_task"]

    # select names / colors / ids
    names = ["None"]
    colors = ["grey"]
    ids = [1]
    if args.scenario=="task":
        names.append("XdG")
        colors.append("purple")
        ids.append(9)
    names += ["EWC", "o-EWC", "SI", "LwF", "DGR", "DGR+distil", "RtF", "Offline"]
    colors += ["deepskyblue", "blue", "yellowgreen", "goldenrod", "indianred", "red", "maroon", "black"]
    ids += [6,7,8,5,3,2,4,0]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                 title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:12s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:12s} {:.2f}".format(name, 100*means[i]))
    print("#"*60)

    # line-plot
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(prec[args.seed][id])):
            all_entries = [prec[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if len(seed_list) > 1:
                new_sem_line.append(1.96*np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    figure = visual_plt.plot_lines(ave_lines, x_axes=x_axes, line_names=names, colors=colors, title=title,
                                   xlabel="tasks", ylabel="average precision (on tasks seen so far)",
                                   list_with_errors=sem_lines if len(seed_list)>1 else None)
    figure_list.append(figure)

    # scatter-plot (accuracy vs training-time)
    accuracies = []
    times = []
    for id in ids[:-1]:
        accuracies.append([ave_prec[seed][id] for seed in seed_list])
        times.append([train_time[seed][id]/60 for seed in seed_list])
    xmax = np.max(times)
    ylim = (0,1.025)
    figure = visual_plt.plot_scatter_groups(x=times, y=accuracies, colors=colors[:-1], figsize=(12, 15), ylim=ylim,
                                            ylabel="average precision (after all tasks)", names=names[:-1],
                                            xlabel="training time (in min)", title=title, xlim=[0, xmax + 0.05 * xmax])
    figure_list.append(figure)


    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))