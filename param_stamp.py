import data


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from encoder import Classifier
    from vae_models import AutoEncoder

    scenario = args.scenario
    # If Task-IL scenario is chosen with single-headed output layer, set args.scenario to "domain"
    # (but note that when XdG is used, task-identity information is being used so the actual scenario is still Task-IL)
    if args.singlehead and args.scenario=="task":
        scenario="domain"

    config = data.get_multitask_experiment(
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        verbose=False,
    )

    if args.feedback:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, z_dim=args.z_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl,
        )
        model.lamda_pl = 1.
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
            fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
            binaryCE=args.bce, binaryCE_distill=args.bce_distill,
        )

    train_gen = True if (args.replay=="generative" and not args.feedback) else False
    if train_gen:
        generator = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'],
            fc_layers=args.g_fc_lay, fc_units=args.g_fc_uni, z_dim=args.g_z_dim, classes=config['classes'],
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl,
        )

    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay=False if (args.replay=="none") else True,
                                  replay_model_name=replay_model_name)
    return param_stamp



def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.experiment, multi_n=multi_n_stamp)
    if verbose:
        print("\n"+" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if hasattr(args, "lr_gen") else "",
        bsz=args.batch, optim=args.optimizer,
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if hasattr(args, 'ewc') and ((args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si)):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda,
            fi="{}{}".format("N" if args.fisher_n is None else args.fisher_n, "E" if args.emp_fi else ""),
            o="-O{}".format(args.gamma) if args.online else "",
        ) if (args.ewc_lambda>0 and args.ewc) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (args.si_c>0 and args.si) else ""
        both = "--" if (args.ewc_lambda>0 and args.ewc) and (args.si_c>0 and args.si) else ""
        if verbose and args.ewc_lambda>0 and args.ewc:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and args.si_c>0 and args.si:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
        hasattr(args, 'ewc') and ((args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si))
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (hasattr(args, 'xdg') and args.xdg) and (hasattr(args, "gating_prop") and args.gating_prop>0):
        xdg_stamp = "--XdG{}".format(args.gating_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.gating_prop))

    # -for replay
    if replay:
        replay_stamp = "{rep}{KD}{model}{gi}".format(
            rep=args.replay,
            KD="-KD{}".format(args.temp) if args.distill else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.gen_iters) if (
                hasattr(args, "gen_iters") and (replay_model_name is not None) and (not args.iters==args.gen_iters)
            ) else ""
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for exemplars / iCaRL
    exemplar_stamp = ""
    if hasattr(args, 'use_exemplars') and (args.add_exemplars or args.use_exemplars or args.replay=="exemplars"):
        exemplar_opts = "b{}{}{}".format(args.budget, "H" if args.herding else "", "N" if args.norm_exemplars else "")
        use = "{}{}".format("addEx-" if args.add_exemplars else "", "useEx-" if args.use_exemplars else "")
        exemplar_stamp = "--{}{}".format(use, exemplar_opts)
        if verbose:
            print(" --> Exemplars:     " + "{}{}".format(use, exemplar_opts))

    # -for binary classification loss
    binLoss_stamp = ""
    if hasattr(args, 'bce') and args.bce:
        binLoss_stamp = '--BCE_dist' if (args.bce_distill and args.scenario=="class") else '--BCE'

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp, exemplar_stamp, binLoss_stamp,
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    print(param_stamp)
    return param_stamp