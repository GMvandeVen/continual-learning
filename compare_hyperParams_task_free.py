#!/usr/bin/env python3
import os
import numpy as np
# -custom-written code
import main_task_free
import utils
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors,set_default_values
from params import options
from visual import visual_plt as my_plt


## Parameter-values to compare
c_list = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000., 100000., 1000000., 10000000.]
xdg_list = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_hyper': True, 'no_boundaries': True}
    # Define input options
    parser = options.define_args(filename="compare_hyperParams_task_free", description='Hyperparamer gridsearches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Should the gridsearch not be run for some methods?
    parser.add_argument('--no-xdg', action='store_true', help="no XdG")
    parser.add_argument('--no-si', action='store_true', help="no SI")
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    args.log_per_context = True
    set_default_values(args, also_hyper_params=False)  # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                   # -check whether incompatible options are selected
    return args


## Function for running experiments and collecting results
def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args, no_boundaries=True)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
        print(" already run: {}".format(param_stamp))
    else:
        args.train = True
        print("\n ...running: {} ...".format(param_stamp))
        main_task_free.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments
    args = handle_inputs()

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    ## Baselline
    args.replay = "none"
    BASE = get_result(args)

    ## XdG
    if args.scenario=="task" and not utils.checkattr(args, 'no_xdg'):
        XDG = {}
        always_xdg = utils.checkattr(args, 'xdg')
        if always_xdg:
            gating_prop_selected = args.gating_prop
        args.xdg = True
        for xdg in xdg_list:
            args.gating_prop = xdg
            XDG[xdg] = get_result(args)
        args.xdg = always_xdg
        if always_xdg:
            args.gating_prop = gating_prop_selected

    ## SI
    if not utils.checkattr(args, 'no_si'):
        SI = {}
        args.weight_penalty = True
        args.importance_weighting = 'si'
        for si_c in c_list:
            args.reg_strength = si_c
            SI[si_c] = get_result(args)
        args.weight_penalty = False


    #-------------------------------------------------------------------------------------------------#

    #-----------------------------------------#
    #----- COLLECT DATA & PRINT ON SCREEN-----#
    #-----------------------------------------#

    ext_c_list = [0] + c_list
    print("\n")


    ###---XdG---###

    if args.scenario == "task" and not utils.checkattr(args, 'no_xdg'):
        # -collect data
        ave_acc_xdg = [XDG[c] for c in xdg_list]
        # -print on screen
        print("\n\nCONTEXT-DEPENDENT GATING (XDG))")
        print(" param list (gating_prop): {}".format(xdg_list))
        print("  {}".format(ave_acc_xdg))
        print("---> gating_prop = {}     --    {}".format(xdg_list[np.argmax(ave_acc_xdg)], np.max(ave_acc_xdg)))


    ###---SI---###

    if not utils.checkattr(args, 'no_si'):
        # -collect data
        ave_acc_si = [BASE] + [SI[c] for c in c_list]
        # -print on screen
        print("\n\nSYNAPTIC INTELLIGENCE (SI)")
        print(" param list (si_c): {}".format(ext_c_list))
        print("  {}".format(ave_acc_si))
        print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_acc_si)], np.max(ave_acc_si)))


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.contexts, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Test accuracy (after all contexts)"

    # calculate limits y-axes (to have equal axes for all graphs)
    full_list = []
    if not utils.checkattr(args, 'no_si'):
        full_list += ave_acc_si
    if args.scenario=="task" and not utils.checkattr(args, 'no_xdg'):
        full_list += ave_acc_xdg
    miny = np.min(full_list)
    maxy = np.max(full_list)
    marginy = 0.1*(maxy-miny)
    ylim = (np.max([miny-2*marginy, 0]),
            np.min([maxy+marginy,1])) if not args.scenario=="class" else (0, np.min([maxy+marginy,1]))

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ###---XdG---###
    if args.scenario=="task" and not utils.checkattr(args, 'no_xdg'):
        figure = my_plt.plot_lines([ave_acc_xdg], x_axes=xdg_list, ylabel=ylabel,
                                line_names=["XdG"], colors=["deepskyblue"], ylim=ylim,
                                title=title, x_log=False, xlabel="XdG: % of nodes gated",
                                with_dots=True, h_line=BASE, h_label="None")
        figure_list.append(figure)

    ###---SI---###
    if not utils.checkattr(args, 'no_si'):
        figure = my_plt.plot_lines([ave_acc_si[1:]], x_axes=c_list, ylabel=ylabel, line_names=["SI"],
                                   colors=["yellowgreen"], title=title, x_log=True, xlabel="SI: c (log-scale)",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))