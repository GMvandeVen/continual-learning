#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main


description = 'Compare two ways of using task-ID info (with different CL strategies) on permuted / split MNIST.'
parser = argparse.ArgumentParser('./_compare_taskID.py', description=description)
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--tasks', type=int, default=5, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")

# model architecture parameters
model_params = parser.add_argument_group('Parameters Main Model')
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
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--gamma', type=float, default=1., help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--c', type=float, default=0.1, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', type=float, default=0.8, dest="xdg",help="XdG: prop neurons per layer to gate")

# exemplar parameters
exemplar_params = parser.add_argument_group('Exemplar Parameters')
exemplar_params.add_argument('--use-exemplars', action='store_true', help="use stored exemplars for classification?")
exemplar_params.add_argument('--budget', type=int, default=2000, dest="budget",help="how many exemplars can be stored?")
exemplar_params.add_argument('--herding',action='store_true',help="use herding to select exemplars (instead of random)")
exemplar_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")



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
    args.scenario = "task"
    args.feedback = False
    args.add_exemplars = False
    args.bce_distill= False
    args.icarl = False
    args.log_per_task = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.ewc = False
    args.online = False
    args.si = False
    args.gating_prop = 0.
    args.singlehead = False
    # args.seed could of course also vary!

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    #########---> Task-ID only in output layer (i.e., multi-headed softmax layer)

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")

    ## online EWC
    args.online = True
    args.ewc_lambda = args.o_lambda
    OEWC = {}
    OEWC = collect_all(OEWC, seed_list, args, name="Online EWC")
    args.ewc = False
    args.online = False

    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")

    ## DGR
    args.replay = "generative"
    args.distill = False
    RP = {}
    RP = collect_all(RP, seed_list, args, name="DGR")

    ## DGR+distill
    args.replay = "generative"
    args.distill = True
    RKD = {}
    RKD = collect_all(RKD, seed_list, args, name="DGR+distill")
    args.replay = "none"
    args.distill = False



    #########---> Task-ID only in hidden layers (i.e., XdG)
    args.singlehead = True
    args.gating_prop = args.xdg

    ## None
    args.replay = "none"
    SNONE = {}
    SNONE = collect_all(SNONE, seed_list, args, name="None")

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

    ## LwF
    args.replay = "current"
    args.distill = True
    SLWF = {}
    SLWF = collect_all(SLWF, seed_list, args, name="LwF")

    ## DGR
    args.replay = "generative"
    args.distill = False
    SRP = {}
    SRP = collect_all(SRP, seed_list, args, name="DGR")

    ## DGR+distill
    args.replay = "generative"
    args.distill = True
    SRKD = {}
    SRKD = collect_all(SRKD, seed_list, args, name="DGR+distill")
    args.replay = "none"
    args.distill = False


    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ave_prec = {}

    ## For each seed, create list with average precisions
    for seed in seed_list:
        ave_prec[seed] = [NONE[seed], EWC[seed], OEWC[seed], SI[seed], LWF[seed], RP[seed], RKD[seed],
                          SNONE[seed], SEWC[seed], SOEWC[seed], SSI[seed], SLWF[seed], SRP[seed], SRKD[seed]]



    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    title_list = ["Task-ID in output-layer", "Task-ID in hidden-layers"]

    # select names / colors / ids
    names = ["None", "EWC", "o-EWC", "SI", "LwF", "DGR", "DGR+distil"]
    colors = ["grey", "deepskyblue", "blue", "yellowgreen", "goldenrod", "indianred", "red"]
    base_ids = [0,1,2,3,4,5,6]
    ids = [
        base_ids,
        [i+len(base_ids) for i in base_ids]
    ]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # multiple bar-plot
    mean_list = []
    sems_list = []
    cis_list = []
    for id_list in ids:
        mean_list.append([np.mean([ave_prec[seed][id] for seed in seed_list]) for id in id_list])
        if args.n_seeds>1:
            sems_list.append([np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in id_list])
            cis_list.append([1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in id_list])
    figure = visual_plt.plot_bars(mean_list, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                  title_list=title_list, top_title=title, yerr=cis_list if args.n_seeds>1 else None,
                                  ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*70+"\n      SUMMARY RESULTS: {}\n".format(title)+"-"*70)
    print(" "*13+"Task-ID in output layer     Task-ID in hidden layers\n"+"-"*70)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:15s}   {:.2f} ({:.2f})                  {:.2f} ({:.2f})".format(
                name, 100*mean_list[0][i], 100*sems_list[0][i], 100*mean_list[1][i], 100*sems_list[1][i],
            ))
        else:
            print("{:16s}    {:.2f}                        {:.2f}".format(
                name, 100*mean_list[0][i], 100*mean_list[1][i]
            ))
    print("#"*70)

    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))