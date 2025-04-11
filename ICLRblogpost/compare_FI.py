#!/usr/bin/env python3
import sys
import os
import numpy as np
# -expand module search path to parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -custom-written code
import main
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors,set_default_values
from params import options
from visual import visual_plt as my_plt


## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000., 10000000000000.]


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
    parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    # Add options specific for EWC
    param_reg = parser.add_argument_group('Parameter Regularization')
    param_reg.add_argument('--offline', action='store_true', help='use Offline EWC rather than Online EWC')
    param_reg.add_argument("--fisher-n-all", type=float, default=500, metavar='N',
                           help="how many samples to approximate FI in 'ALL-n=X'")
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

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ## Baselline
    args.replay = "none"
    BASE = {}
    BASE = collect_all(BASE, seed_list, args, name="None")

    # -set EWC-specific arguments
    args.weight_penalty = True
    args.importance_weighting = 'fisher'

    ## EWC, "sample"
    SAMPLE = {}
    args.fisher_labels = "sample"
    args.fisher_n = None
    args.fisher_batch = 1
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        SAMPLE[ewc_lambda] = {}
        SAMPLE[ewc_lambda] = collect_all(SAMPLE[ewc_lambda], seed_list, args,
                                         name="EWC -- FI-labels='sample' (lambda={})".format(ewc_lambda))

    ## EWC, "true"
    TRUE = {}
    args.fisher_labels = "true"
    args.fisher_n = None
    args.fisher_batch = 1
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

    ## EWC, "all" -- only [args.fisher_n_all] samples per task
    ALL500 = {}
    args.fisher_labels = "all"
    args.fisher_n = args.fisher_n_all
    args.fisher_batch = 1
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        ALL500[ewc_lambda] = {}
        ALL500[ewc_lambda] = collect_all(ALL500[ewc_lambda], seed_list, args,
                                         name="EWC -- FI-labels='all' - n={} (lambda={})".format(args.fisher_n_all, ewc_lambda))

    ## EWC, "all"
    ALL = {}
    args.fisher_labels = "all"
    args.fisher_n = None
    args.fisher_batch = 1
    for ewc_lambda in lamda_list:
        args.reg_strength=ewc_lambda
        ALL[ewc_lambda] = {}
        ALL[ewc_lambda] = collect_all(ALL[ewc_lambda], seed_list, args,
                                        name="EWC -- FI-labels='all' (lambda={})".format(ewc_lambda))


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------------#
    #----- COLLECT DATA & PRINT ON SCREEN-----#
    #-----------------------------------------#

    ext_lambda_list = [0] + lamda_list
    ext_lambda_list = lamda_list
    print("\n")

    base_entries = [BASE[seed][1] for seed in seed_list]
    mean_base = np.mean(base_entries)
    sem_base = (np.sqrt(np.var(base_entries)) / np.sqrt(args.n_seeds)) if args.n_seeds>1 else None

    ###---EWC "all" ---###
    mean_all = []
    sem_all = []
    for ewc_lambda in lamda_list:
        new_entries = [ALL[ewc_lambda][seed][1] for seed in seed_list]
        mean_all.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_all.append(np.sqrt(np.var(new_entries)) / np.sqrt((args.n_seeds)))
    lambda_all = ext_lambda_list[np.argmax(mean_all)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='all'")
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_all))
    print("---> lambda = {}     --    {}".format(lambda_all, np.max(mean_all)))

    ###---EWC "all, with n=500" ---###
    mean_all500 = []
    if args.n_seeds>1:
        sem_all500 = []
    for ewc_lambda in lamda_list:
        new_entries = [ALL500[ewc_lambda][seed][1] for seed in seed_list]
        mean_all500.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_all500.append(np.sqrt(np.var(new_entries)) / np.sqrt((args.n_seeds)))
    lambda_all500 = ext_lambda_list[np.argmax(mean_all500)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='all' - n={}".format(args.fisher_n_all))
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_all500))
    print("---> lambda = {}     --    {}".format(ext_lambda_list[np.argmax(mean_all500)], np.max(mean_all500)))

    ###---EWC "sample" ---###
    mean_sample = []
    if args.n_seeds>1:
        sem_sample = []
    for ewc_lambda in lamda_list:
        new_entries = [SAMPLE[ewc_lambda][seed][1] for seed in seed_list]
        mean_sample.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_sample.append(np.sqrt(np.var(new_entries)) / np.sqrt((args.n_seeds)))
    lambda_sample = ext_lambda_list[np.argmax(mean_sample)]
    # -print on screen
    print("\n\nEWC  --  FI-labels='sample'")
    print(" param list (lambda): {}".format(ext_lambda_list))
    print("  {}".format(mean_sample))
    print("---> lambda = {}     --    {}".format(ext_lambda_list[np.argmax(mean_sample)], np.max(mean_sample)))

    ###---EWC "true" ---###
    mean_true = []
    if args.n_seeds>1:
        sem_true = []
    for ewc_lambda in lamda_list:
        new_entries = [TRUE[ewc_lambda][seed][1] for seed in seed_list]
        mean_true.append(np.mean(new_entries))
        if args.n_seeds>1:
            sem_true.append(np.sqrt(np.var(new_entries)) / np.sqrt((args.n_seeds)))
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
            sem_true128.append(np.sqrt(np.var(new_entries)) / np.sqrt((args.n_seeds)))
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
    lines = [mean_all, mean_all500, mean_sample, mean_true, mean_true128]
    errors = [sem_all, sem_all500, sem_sample, sem_true, sem_true128] if args.n_seeds>1 else None
    names = ["All", "All - n={}".format(args.fisher_n_all), "Sample", "True", "True - batch=128"]
    colors = ["black", "grey", "red", "orange", "blue"]
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
        ave_prec[seed] = [BASE[seed][1], ALL[lambda_all][seed][1], ALL500[lambda_all500][seed][1], SAMPLE[lambda_sample][seed][1],
                          TRUE[lambda_true][seed][1], TRUE128[lambda_true128][seed][1]]
        train_time[seed] = [BASE[seed][0], ALL[lambda_all][seed][0], ALL500[lambda_all500][seed][0], SAMPLE[lambda_sample][seed][0],
                            TRUE[lambda_true][seed][0], TRUE128[lambda_true128][seed][0]]
    names = ["None", "All", "All - n={}".format(args.fisher_n_all), "Sample", "True", "True - batch=128"]
    colors = ["green", "black", "grey", "red", "orange", "blue"]
    ids = [0, 1, 2, 3, 4, 5]

    # Avearge accuracy
    # -bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list]))/np.sqrt(len(seed_list)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list]))/np.sqrt(len(seed_list)) for id in ids]
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
        sems = [np.sqrt(np.var([train_time[seed][id] for seed in seed_list])) / np.sqrt(len(seed_list)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([train_time[seed][id] for seed in seed_list])) / np.sqrt(len(seed_list)) for id in ids]
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
