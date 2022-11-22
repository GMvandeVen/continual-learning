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


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_all': True}
    # Define input options
    parser = options.define_args(filename="compare", description='Compare performance of CL strategies.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Should some methods not be included in the comparison?
    parser.add_argument('--no-context-spec', action='store_true', help="no XdG or Separate Networks")
    parser.add_argument('--no-reg', action='store_true', help="no EWC or SI")
    parser.add_argument('--no-fromp', action='store_true', help="no FROMP")
    parser.add_argument('--no-bir', action='store_true', help="no BI-R")
    parser.add_argument('--no-agem', action='store_true', help="no A-GEM")
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
    # -check whether already run; if not do so
    file_to_check = '{}/acc-{}{}.txt'.format(args.r_dir, param_stamp,
                                             "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
    if os.path.isfile(file_to_check):
        print(" already run: {}".format(param_stamp))
    elif os.path.isfile("{}/mM-{}".format(args.m_dir, param_stamp)):
        args.train = False
        print(" ...testing: {}".format(param_stamp))
        main.run(args)
    else:
        args.train = True
        print(" ...running: {}".format(param_stamp))
        main.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}{}.txt'.format(args.r_dir, param_stamp,
                                        "--S{}".format(args.eval_s) if checkattr(args, 'gen_classifier') else "")
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -print average accuracy on screen
    print("--> average accuracy: {}".format(ave))
    # -return average accuracy
    return ave

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

    ## JOINT training (using total number of iterations from all contexts)
    iters_temp = args.iters
    args.iters = args.contexts*iters_temp
    args.joint = True
    JOINT = {}
    JOINT = collect_all(JOINT, seed_list, args, name="Joint")
    args.joint = False
    args.iters = iters_temp


    ###----"CONTEXT-SPECIFIC"----####

    if args.scenario=="task" and not checkattr(args, 'no_context_spec'):
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

    if not checkattr(args, 'no_reg'):
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
    else:
        EWC = SI = None


    ###----"FUNCTIONAL REGULARIZATION"----####

    ## LwF
    args.replay = "current"
    args.distill = True
    LWF = {}
    LWF = collect_all(LWF, seed_list, args, name="LwF")
    args.replay = "none"
    args.distill = False

    ## FROMP
    if not checkattr(args, 'no_fromp'):
        args.fromp = True
        args.sample_selection = "fromp"
        FROMP = {}
        FROMP = collect_all(FROMP, seed_list, args, name="FROMP")
        args.fromp = False
    else:
        FROMP = None


    ###----"REPLAY"----###

    ## DGR
    args.replay = "generative"
    args.distill = False
    DGR = {}
    DGR = collect_all(DGR, seed_list, args, name="Deep Generative Replay")

    ## BI-R
    if not checkattr(args, 'no_bir'):
        args.replay = "generative"
        args.feedback = True
        args.hidden = True
        args.dg_gates = True
        args.prior = "GMM"
        args.per_class = True
        args.distill = True
        BIR = {}
        BIR = collect_all(BIR, seed_list, args, name="Brain-Inspired Replay")
        args.feedback = False
        args.hidden = False
        args.dg_gates = False
        args.prior = "standard"
        args.per_class = False
        args.distill = False
    else:
        BIR = None

    ## Experience Replay
    args.replay = "buffer"
    args.sample_selection = "random"
    ER = {}
    ER = collect_all(ER, seed_list, args, name="Experience Replay (budget = {})".format(args.budget))
    args.replay = "none"

    ## A-GEM
    if not checkattr(args, 'no_agem'):
        args.replay = "buffer"
        args.sample_selection = "random"
        args.use_replay = "inequality"
        AGEM = {}
        AGEM = collect_all(AGEM, seed_list, args, name="A-GEM (budget = {})".format(args.budget))
        args.replay = "none"
        args.use_replay = "normal"
    else:
        AGEM = None


    ###----"TEMPLATE-BASED CLASSIFICATION"----####

    if args.scenario=="class" and not args.neg_samples=="current":
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
        ave_acc[seed] = [NONE[seed], JOINT[seed],
                         0 if EWC is None else EWC[seed], 0 if SI is None else SI[seed], LWF[seed],
                         0 if FROMP is None else FROMP[seed],
                         DGR[seed], 0 if BIR is None else BIR[seed], ER[seed], 0 if AGEM is None else AGEM[seed]]
        if args.scenario=="task" and not checkattr(args, 'no_context_spec'):
            ave_acc[seed].append(XDG[seed])
            ave_acc[seed].append(SEP[seed])
        elif args.scenario=="class" and not args.neg_samples=="current":
            ave_acc[seed].append(ICARL[seed])
            ave_acc[seed].append(GENCLASS[seed])


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------------#
    #----- REPORTING / PLOTTING: AVERAGE ACCURACY -----#
    #--------------------------------------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.contexts, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)

    # select names / colors / ids
    names = ["None", "Joint"]
    colors = ["grey", "black"]
    ids = [0, 1]
    if args.scenario=="task" and not checkattr(args, 'no_context_spec'):
        names += ['Separate Networks', 'XdG']
        colors += ['dodgerblue', 'deepskyblue']
        ids += [11, 10]
    if not checkattr(args, 'no_reg'):
        names += ['EWC', 'SI']
        colors += ['darkgreen', 'yellowgreen']
        ids += [2, 3]
    names.append('LwF')
    colors.append('gold')
    ids.append(4)
    if not checkattr(args, 'no_fromp'):
        names.append("FROMP (b={})".format(args.budget))
        colors.append('goldenrod')
        ids.append(5)
    names.append('DGR')
    colors.append('indianred')
    ids.append(6)
    if not checkattr(args, 'no_bir'):
        names.append('BI-R')
        colors.append('lightcoral')
        ids.append(7)
    names.append("ER (b={})".format(args.budget))
    colors.append('red')
    ids.append(8)
    if not checkattr(args, 'no_agem'):
        names.append("A-GEM (b={})".format(args.budget))
        colors.append('orangered')
        ids.append(9)
    if args.scenario=="class" and not args.neg_samples=="current":
        names += ['Generative Classifier', "iCaRL (b={})".format(args.budget)]
        colors += ['indigo', 'purple']
        ids += [11, 10]

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average accuracy (after all contexts)",
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

    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))