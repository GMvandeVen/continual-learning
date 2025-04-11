#!/usr/bin/env python3
import sys
import os
import numpy as np
# -expand module search path to parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -custom-written code
import main
import utils
from utils import checkattr
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors,set_default_values
from params import options
from visual import visual_plt


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_all': True}
    # Define input options
    parser = options.define_args(filename="compare", description='Compare and plot performance of CL strategies.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    args.log_per_context = True
    set_default_values(args, also_hyper_params=True)  # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)                  # -check whether incompatible options are selected
    return args


## Functions for running experiments and collecting results
def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether the 'results-dict' is already available; if not, run the experiment
    file_to_check = '{}/dict-{}--n{}{}.pkl'.format(
        args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n,
        "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else ""
    )
    if os.path.isfile(file_to_check):
        print(" already run: {}".format(param_stamp))
    else:
        args.train = True
        args.results_dict = True
        print(" ...running: {}".format(param_stamp))
        main.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}{}.txt'.format(args.r_dir, param_stamp,
                                        "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -get results-dict
    dict = utils.load_object(
        "{}/dict-{}--n{}{}".format(args.r_dir, param_stamp, "All" if args.acc_n is None else args.acc_n,
                                   "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
    )
    # -print average accuracy on screen
    print("--> average accuracy: {}".format(ave))
    # -return average accuracy
    return (dict, ave)

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

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    ## JOINT training, again for each context (only using number of iterations from that context)
    args.cummulative = True
    args.reinit = True
    JOINT = {}
    JOINT = collect_all(JOINT, seed_list, args, name="Joint")
    args.reinit = False
    args.cummulative = False


    ###----"CONTEXT-SPECIFIC"----####

    if args.scenario=="task":
        ## Separate network per context
        fc_units_temp = args.fc_units
        args.fc_units = args.fc_units_sep
        args.separate_networks = True
        SEP = {}
        SEP = collect_all(SEP, seed_list, args, name="Separate Networks")
        args.separate_networks = False
        args.fc_units = fc_units_temp

        ## XdG
        always_xdg =  checkattr(args, 'xdg')
        args.xdg = True
        XDG = {}
        XDG = collect_all(XDG, seed_list, args, name="XdG")
        args.xdg = always_xdg


    ###----"PARAMETER REGULARIZATION"----####

    ## EWC
    args.weight_penalty = True
    args.importance_weighting = 'fisher'
    args.offline = True
    args.reg_strength = args.ewc_lambda
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.weight_penalty = False
    args.offline = False

    ## SI
    args.weight_penalty = True
    args.importance_weighting = 'si'
    args.reg_strength = args.si_c
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.weight_penalty = False


    ###----"FUNCTIONAL REGULARIZATION"----####

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")
    args.replay = "none"
    args.distill = False

    ## FROMP
    args.fromp = True
    args.sample_selection = "fromp"
    FROMP = {}
    FROMP = collect_all(FROMP, seed_list, args, name="FROMP")
    args.fromp = False


    ###----"REPLAY"----###

    ## DGR
    args.replay = "generative"
    args.distill = False
    DGR = {}
    DGR = collect_all(DGR, seed_list, args, name="Deep Generative Replay")

    ## Experience Replay
    args.replay = "buffer"
    args.sample_selection = "random"
    ER = {}
    ER = collect_all(ER, seed_list, args, name="Experience Replay (budget = {})".format(args.budget))
    args.replay = "none"


    ###----"TEMPLATE-BASED CLASSIFICATION"----####

    if args.scenario=="class":
        ## iCaRL
        args.bce = True
        args.bce_distill = True
        args.prototypes = True
        args.add_buffer = True
        args.sample_selection = "herding"
        args.neg_samples = "all-so-far"
        ICARL = {}
        ICARL = collect_all(ICARL, seed_list, args, name="iCaRL (budget = {})".format(args.budget))
        args.bce = False
        args.bce_distill = False
        args.prototypes = False
        args.add_buffer = False

        ## Generative Classifier
        args.gen_classifier = True
        classes_per_context = 2 if args.experiment=="splitMNIST" else 10
        args.iters = int(args.iters / classes_per_context)
        args.fc_units = args.fc_units_gc
        args.fc_lay = args.fc_lay_gc
        args.z_dim = args.z_dim_gc
        args.hidden = True
        args.lr = 0.001
        GENCLASS = {}
        GENCLASS = collect_all(GENCLASS, seed_list, args, name="Generative Classifier")


    #-------------------------------------------------------------------------------------------------#

    #---------------------------------------------#
    #----- COLLECT RESULTS: AVERAGE ACCURACY -----#
    #---------------------------------------------#

    ## For each seed, create list with average test accuracy
    ave_acc = {}
    for seed in seed_list:
        ave_acc[seed] = [NONE[seed][1], JOINT[seed][1], EWC[seed][1], SI[seed][1], LWF[seed][1], FROMP[seed][1],
                         DGR[seed][1], ER[seed][1]]
        if args.scenario=="task":
            ave_acc[seed].append(XDG[seed][1])
            ave_acc[seed].append(SEP[seed][1])
        elif args.scenario=="class":
            ave_acc[seed].append(ICARL[seed][1])
            ave_acc[seed].append(GENCLASS[seed][1])

    ## For each seed, create lists with test accuracy throughout training
    prec = {}
    for seed in seed_list:
        # -for plot of average accuracy throughout training
        key = "average"
        prec[seed] = [NONE[seed][0][key], JOINT[seed][0][key], EWC[seed][0][key], SI[seed][0][key], LWF[seed][0][key],
                      FROMP[seed][0][key], DGR[seed][0][key], ER[seed][0][key]]
        if args.scenario=="task":
            prec[seed].append(XDG[seed][0][key])
            prec[seed].append(SEP[seed][0][key])
        elif args.scenario=="class":
            prec[seed].append(ICARL[seed][0][key])
            prec[seed].append(GENCLASS[seed][0][key])


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------#
    #----- REPORTING / PLOTTING: AVERAGE ACCURACY -----#
    #--------------------------------------------------#

    # name for plot
    plot_name = "tutorialplots-{}{}-{}".format(args.experiment, args.contexts, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # select names / colors / ids
    names = ["None", "Joint"]
    colors = ["grey", "black"]
    ids = [0, 1]
    if args.scenario=="task":
        names += ['Separate Networks', 'XdG']
        colors += ['dodgerblue', 'deepskyblue']
        ids += [9, 8]
    names += ['EWC', 'SI', 'LwF', 'FROMP (b={})'.format(args.budget), 'DGR', "ER (b={})".format(args.budget)]
    colors += ['darkgreen', 'yellowgreen', 'gold', 'goldenrod', 'indianred', 'red']
    ids += [2, 3, 4, 5, 6, 7]
    if args.scenario=="class":
        names += ['Generative Classifier', "iCaRL (b={})".format(args.budget)]
        colors += ['indigo', 'purple']
        ids += [9, 8]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="Average accuracy (after all contexts)",
                                 title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"#"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:27s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:27s} {:.2f}".format(name, 100*means[i]))
        if i==1:
            print("="*60)
    print("#"*60)

    # line-plot
    x_axes = NONE[args.seed][0]["x_context"]
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
                                   xlabel="# of contexts", ylabel="Average accuracy (on contexts so far)",
                                   list_with_errors=sem_lines if len(seed_list)>1 else None)
    figure_list.append(figure)

    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))