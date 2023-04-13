#!/usr/bin/env python3
import os
import numpy as np
# -custom-written code
import main
from utils import checkattr
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors,set_default_values
from params import options
from visual import visual_plt


## Memory budget values to compare
budget_list_CIFAR100 = [1, 2, 5, 10, 20, 50, 100, 200, 500]
budget_list_splitMNIST = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_replay': True}
    # Define input options
    parser = options.define_args(filename="compare_replay",
                                 description='Evaluate CL methods storing data as function of available memory budget.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Should some methods not be included?
    parser.add_argument('--no-fromp', action='store_true', help="no FROMP")
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    set_default_values(args, also_hyper_params=False) # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                  # -check whether incompatible options are selected
    return args


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
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
    # -return it
    return ave


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

    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## Select correct memory budget list
    budget_list = budget_list_CIFAR100 if args.experiment=="CIFAR100" else budget_list_splitMNIST


    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    budget_limit_FROMP = 1000
    if checkattr(args, 'tau_per_budget'):
        if args.scenario=="task":
            tau_dict = {'1': 100000., '2': 1000., '5': 100000., '10': 0.001, '20': 10000., '50': 1000.,
                        '100': 0.01, '200': 0.01, '500': 0.1, '1000': 10.}
        elif args.scenario=="domain":
            tau_dict = {'1': 0.001, '2': 100000., '5': 100000., '10': 100000., '20': 100000., '50': 10000.,
                        '100': 10., '200': 1., '500': 10., '1000': 0.1}
        elif args.scenario=="class":
            tau_dict = {'1': 100000., '2': 0.01, '5': 10000., '10': 100000., '20': 10000., '50': 1000.,
                        '100': 1000., '200': 10., '500': 0.001, '1000': 1.}


    ###### BASELINES #########

    args.replay = "none"
    BASE = {}
    BASE = collect_all(BASE, seed_list, args, name="None")

    iters_temp = args.iters
    args.iters = args.contexts*iters_temp
    args.joint = True
    JOINT = {}
    JOINT = collect_all(JOINT, seed_list, args, name="Joint")
    args.joint = False
    args.iters = iters_temp


    ###### CL METHODS STORING DATA #########

    ## Experience Replay
    args.replay = "buffer"
    args.sample_selection = "random"
    args.distill = False
    ER = {}
    for budget in budget_list:
        args.budget = budget
        ER[budget] = {}
        ER[budget] = collect_all(ER[budget], seed_list, args, name="Experience Replay - budget = {}".format(budget))

    ## A-GEM
    args.replay = "buffer"
    args.distill = False
    args.sample_selection = "random"
    args.use_replay = "inequality"
    AGEM = {}
    for budget in budget_list:
        args.budget = budget
        AGEM[budget] = {}
        AGEM[budget] = collect_all(AGEM[budget], seed_list, args, name="A-GEM - budget = {}".format(budget))
    args.use_replay = "normal"

    ## FROMP
    if not checkattr(args, 'no_fromp'):
        args.replay = "none"
        args.fromp = True
        args.sample_selection = "fromp"
        FROMP = {}
        for budget in budget_list:
            if budget<=budget_limit_FROMP:
                args.budget = budget
                if checkattr(args, 'tau_per_budget'):
                    args.tau = tau_dict['{}'.format(budget)]
                FROMP[budget] = {}
                FROMP[budget] = collect_all(FROMP[budget], seed_list, args, name="FROMP - budget = {}".format(budget))
        args.fromp = False

    ## iCaRL
    if args.scenario=="class":
        args.replay = "none"
        args.prototypes = True
        args.bce = True
        args.bce_distill = True
        args.add_buffer = True
        args.sample_selection = 'herding'
        args.neg_samples = "all-so-far"
        ICARL = {}
        for budget in budget_list:
            args.budget = budget
            ICARL[budget] = {}
            ICARL[budget] = collect_all(ICARL[budget], seed_list, args, name="iCaRL - budget = {}".format(budget))


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summaryExactRep-{}{}-{}".format(args.experiment,args.contexts,args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # set scale of y-axis
    y_lim = [0,1] if args.scenario=="class" else None
    y_lim = None

    # Methods for comparison
    h_lines = [np.mean([BASE[seed] for seed in seed_list]), np.mean([JOINT[seed] for seed in seed_list])]
    h_errors = [np.sqrt(np.var([BASE[seed] for seed in seed_list]) / (len(seed_list)-1)),
                np.sqrt(np.var([JOINT[seed] for seed in seed_list]) / (len(seed_list)-1))] if args.n_seeds>1 else None
    h_labels = ["None", "Joint"]
    h_colors = ["grey", "black"]


    # Different variants of exact replay
    # -prepare
    ave_ER = []
    sem_ER = []
    ave_AGEM = []
    sem_AGEM = []
    if not checkattr(args, 'no_fromp'):
        ave_FROMP = []
        sem_FROMP = []
    if args.scenario=="class":
        ave_ICARL = []
        sem_ICARL = []

    for budget in budget_list:
        all_entries = [ER[budget][seed] for seed in seed_list]
        ave_ER.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_ER.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        all_entries = [AGEM[budget][seed] for seed in seed_list]
        ave_AGEM.append(np.mean(all_entries))
        if args.n_seeds > 1:
            sem_AGEM.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

        if not checkattr(args, 'no_fromp'):
            if budget<=budget_limit_FROMP:
                all_entries = [FROMP[budget][seed] for seed in seed_list]
                ave_FROMP.append(np.mean(all_entries))
                if args.n_seeds > 1:
                    sem_FROMP.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))
            else:
                ave_FROMP.append(np.nan)
                if args.n_seeds>1:
                    sem_FROMP.append(np.nan)

        if args.scenario=="class":
            all_entries = [ICARL[budget][seed] for seed in seed_list]
            ave_ICARL.append(np.mean(all_entries))
            if args.n_seeds > 1:
                sem_ICARL.append(np.sqrt(np.var(all_entries) / (len(all_entries) - 1)))

    # -collect
    lines = [ave_ER, ave_AGEM]
    errors = [sem_ER, sem_AGEM] if args.n_seeds > 1 else None
    line_names = ["ER", "A-GEM"]
    colors = ["red", "orangered"]
    if not checkattr(args, 'no_fromp'):
        lines.append(ave_FROMP)
        line_names.append("FROMP")
        colors.append("goldenrod")
        if args.n_seeds>1:
            errors.append(sem_FROMP)
    if args.scenario=="class":
        lines.append(ave_ICARL)
        line_names.append("iCaRL")
        colors.append("purple")
        if args.n_seeds>1:
            errors.append(sem_ICARL)

    # -plot
    figure = visual_plt.plot_lines(
        lines, x_axes=budget_list, ylabel="average accuracy (after all contexts)", title=title, x_log=True, ylim=y_lim,
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