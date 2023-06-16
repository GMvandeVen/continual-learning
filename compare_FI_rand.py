#!/usr/bin/env python3
import os
import numpy as np
# -custom-written code
import main
import utils
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors,set_default_values
from params import options
from visual import visual_plt as my_plt
import torch


## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000., 10000000000000.]
#lamda_list = [1., 1000., 1000000., 1000000000., 1000000000000.]
lamda_list_permMNIST = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000.]
lamda_list_CIFAR = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
                    100000000000., 1000000000000., 10000000000000., 100000000000000.]
#lamda_list_CIFAR = [1., 100., 10000., 1000000., 100000000., 10000000000., 10000000000.]
#lamda_list_CIFAR = [1., 100., 10000., 1000000., 100000000.]#, 10000000000., 10000000000.]
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000., 10000000000000.]
lamda_list_permMNIST = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
                        100000000000.]
lamda_list_permMNIST = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
                        100000000000.]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_hyper': True}
    # Define input options
    parser = options.define_args(filename="compare_hyperParams",
                                 description='Compare performance EWC with differrent ways of computing FI matrix.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    #parser = options.add_cl_options(parser, **kwargs)
    parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    parser.add_argument('--online', action='store_true', help='use Online EWC rather than Offline EWC')
    parser.add_argument("--precondition", action='store_true', help="parameter regularization by gradient projection")
    parser.add_argument("--alpha", type=float, default=1e-10, help="small constant stabilizing inversion importance matrix")
    parser.add_argument('--no-all', action='store_true', help='do not run the time-consuming "ALL"')
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    args.log_per_context = True
    set_default_values(args, also_hyper_params=False)  # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                   # -check whether incompatible options are selected
    return args


## Function for running experiments and collecting results
def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/time-{}.txt'.format(args.r_dir, param_stamp)) and os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
        print(" already run: {}".format(param_stamp))
    else:
        args.train = True
        print("\n ...running: {} ...".format(param_stamp))
        main.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -get training time
    fileName = '{}/time-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    traintime = float(file.readline())
    file.close()
    # -return it
    return (traintime, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_result(args)
    # -return updated dictionary with results
    return method_dict


if __name__ == '__main__':

    ## Load input-arguments
    args = handle_inputs()
    args.time = True

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    ## Select parameter-lists based on chosen experiment
    lamda_list = lamda_list_permMNIST if args.experiment=="permMNIST" else (lamda_list_CIFAR if args.experiment=="CIFAR100" else lamda_list)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    precondition = args.precondition
    args.precondition = False

    ## Baselline
    args.replay = "none"
    BASE = {}
    BASE = collect_all(BASE, seed_list, args, name="None")

    args.precondition = precondition
    args.weight_penalty = True
    args.importance_weighting = 'fisher'
    args.offline = False if args.online else True
    args.fisher_batch = 1
    args.randomize_fisher = True

    ## EWC, "true"
    TRUE = {}
    args.fisher_labels = "true"
    args.fisher_n = None
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        TRUE[ewc_lambda] = {}
        TRUE[ewc_lambda] = collect_all(TRUE[ewc_lambda], seed_list, args,
                                       name="EWC -- FI-labels='true' (lambda={})".format(ewc_lambda))

    ## EWC, "true" - batch=128
    TRUE128 = {}
    args.fisher_labels = "true"
    args.fisher_n = None
    args.fisher_batch = 128
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        TRUE128[ewc_lambda] = {}
        TRUE128[ewc_lambda] = collect_all(TRUE128[ewc_lambda], seed_list, args,
                                          name="EWC -- FI-labels='true' - batch=128 (lambda={})".format(ewc_lambda))
    args.fisher_batch = 1

    ## EWC, "all" -- only 500 samples per task
    ALL500 = {}
    args.fisher_labels = "all"
    args.fisher_n = 500
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        ALL500[ewc_lambda] = {}
        ALL500[ewc_lambda] = collect_all(ALL500[ewc_lambda], seed_list, args,
                                         name="EWC -- FI-labels='all' - n=500 (lambda={})".format(ewc_lambda))

    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------------#
    #----- COLLECT DATA & PRINT ON SCREEN-----#
    #-----------------------------------------#

    ext_lambda_list = [0] + lamda_list
    ext_lambda_list = lamda_list
    print("\n")

    base_entries = [BASE[seed][1] for seed in seed_list]
    mean_base = np.mean(base_entries)
    sem_base = (np.sqrt(np.var(base_entries)) / (args.n_seeds-1)) if args.n_seeds>1 else None

    ###---EWC "all, with n=500" ---###
    mean_all500 = []
    if args.n_seeds>1:
        sem_all500 = []
    for ewc_lambda in lamda_list:
        new_entries = [ALL500[ewc_lambda][seed][1] for seed in seed_list]
        mean_all500.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_all500.append(np.sqrt(np.var(new_entries)) / (args.n_seeds-1))
    lambda_all500 = ext_lambda_list[np.argmax(mean_all500)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='all' - n=500")
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_all500))
    print("---> lambda = {}     --    {}".format(ext_lambda_list[np.argmax(mean_all500)], np.max(mean_all500)))

    ###---EWC "true" ---###
    mean_true = []
    if args.n_seeds>1:
        sem_true = []
    for ewc_lambda in lamda_list:
        new_entries = [TRUE[ewc_lambda][seed][1] for seed in seed_list]
        mean_true.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_true.append(np.sqrt(np.var(new_entries)) / (args.n_seeds-1))
    lambda_true = ext_lambda_list[np.argmax(mean_true)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='true'")
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_true))
    print("---> lambda = {}     --    {}".format(ext_lambda_list[np.argmax(mean_true)], np.max(mean_true)))

    ###---EWC "true" - batch=128 ---###
    mean_true128 = []
    if args.n_seeds>1:
        sem_true128 = []
    for ewc_lambda in lamda_list:
        new_entries = [TRUE128[ewc_lambda][seed][1] for seed in seed_list]
        mean_true128.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_true128.append(np.sqrt(np.var(new_entries)) / (args.n_seeds-1))
    lambda_true128 = ext_lambda_list[np.argmax(mean_true128)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='true' - batch=128")
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_true128))
    print("---> lambda = {}     --    {}".format(ext_lambda_list[np.argmax(mean_true128)], np.max(mean_true128)))

    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.contexts, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Test accuracy (after all contexts)"

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ########### ALL HYPERPARAM-VALUES ###########

    # - select lines, names and colors
    lines = [mean_all500, mean_true, mean_true128]
    errors = [sem_all500, sem_true, sem_true128] if args.n_seeds>1 else None
    names = ["All - n=500", "True", "True - batch=128"]
    colors = ["grey", "orange", "blue"]
    # - make plot (line plot - only average)
    figure = my_plt.plot_lines(lines, x_axes=ext_lambda_list, ylabel=ylabel, line_names=names, list_with_errors=errors,
                               title=title, x_log=True, xlabel="EWC: lambda log-scale)",
                               with_dots=True, colors=colors, h_lines=[mean_base],
                               h_errors=[sem_base] if args.n_seeds>1 else None, h_labels=["None"])
    figure_list.append(figure)


    ########### ACCURACY (AND TRAIN-TIMES) OF BEST HYPERPARAMS ###########

    # Collect the best accuracies (and training times)
    ave_prec = {}
    train_time = {}
    for seed in seed_list:
        ave_prec[seed] = [BASE[seed][1], ALL500[lambda_all500][seed][1],
                          TRUE[lambda_true][seed][1], TRUE128[lambda_true128][seed][1]]
        train_time[seed] = [BASE[seed][0], ALL500[lambda_all500][seed][0],
                            TRUE[lambda_true][seed][0], TRUE128[lambda_true128][seed][0]]
    names = ["None", "All - n=500", "True", "True - batch=128"]
    colors = ["green", "grey", "orange", "blue"]
    ids = [0, 1, 2, 3,]

    # Avearge accuracy
    # -bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = my_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                             title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)
    # -print results to screen
    print("\n\n"+"#"*49+"\n        AVERAGE TEST ACCURACY (in %)\n"+"-"*49)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:21s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:21s} {:.2f}".format(name, 100*means[i]))
        if i==0:
            print("-"*49)
    print("#"*49)

    # Training time
    # -bar-plot
    means = [np.mean([train_time[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems = [np.sqrt(np.var([train_time[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([train_time[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = my_plt.plot_bar(means, names=names, colors=colors, ylabel="Training Time (in Sec)",
                             title=title, yerr=cis if len(seed_list) > 1 else None)
    figure_list.append(figure)
    # -print results to screen
    print("\n\n" + "#" * 49 + "\n         TOTAL TRAINING TIME (in Sec)\n" + "-" * 49)
    for i, name in enumerate(names):
        if len(seed_list) > 1:
            print("{:21s} {:4.0f}  (+/- {:2.0f}),  n={}".format(name, means[i], sems[i], len(seed_list)))
        else:
            print("{:21s} {:4.0f}".format(name, means[i]))
        if i==0:
            print("-"*49)
    print("#" * 49)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
