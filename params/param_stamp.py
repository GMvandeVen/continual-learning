from data.load import get_context_set
from models import define_models as define
from utils import checkattr


def visdom_name(args):
    '''Get name for graph in visdom from [args].'''
    iCaRL = (checkattr(args, 'prototypes') and checkattr(args, 'add_buffer') and checkattr(args, 'bce')
             and checkattr(args, 'bce_distill'))
    name = "{fb}{replay}{param_reg}{xdg}{icarl}{fromp}{bud}".format(
        fb="1M-" if checkattr(args, 'feedback') else "",
        replay="{}{}{}".format(args.replay, "D" if checkattr(args, 'distill') else "",
                               "-aGEM" if hasattr(args, 'use_replay') and args.use_replay=='inequality' else ""),
        param_reg="-par{}-{}".format(args.reg_strength,
                                     args.importance_weighting) if checkattr(args, 'weight_penalty') else '',
        xdg="" if (not checkattr(args, 'xdg')) or args.gating_prop == 0 else "-XdG{}".format(args.gating_prop),
        icarl="-iCaRL" if iCaRL else "",
        fromp="-FROMP{}".format(args.tau) if checkattr(args, 'fromp') else "",
        bud="-bud{}".format(args.budget) if args.replay=='buffer' or iCaRL else "",
    )
    return name


def get_param_stamp_from_args(args, no_boundaries=False):
    '''To get param-stamp a bit quicker.'''

    config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir, only_config=True,
        normalize=checkattr(args, "normalize"), verbose=False, singlehead=checkattr(args, 'singlehead'),
    )

    # -get feature extractor architecture (if used)
    feature_extractor_name = None
    depth = args.depth if hasattr(args, 'depth') else 0
    use_feature_extractor = checkattr(args, 'hidden') or (
            checkattr(args, 'freeze_convE') and (not args.replay=="generative") and (not checkattr(args, "add_buffer"))
            and (not checkattr(args, "augment")) and (not checkattr(args, 'gen_classifier'))
    )
    if use_feature_extractor:
        feature_extractor = define.define_feature_extractor(args=args, config=config, device='cpu')
        feature_extractor_name = feature_extractor.name if depth > 0 else None
        config = config.copy()  # -> make a copy to avoid overwriting info in the original config-file
        config['size'] = feature_extractor.conv_out_size
        config['channels'] = feature_extractor.conv_out_channels
        depth = 0
    # -get classifier architecture
    model = define.define_classifier(args=args, config=config, device='cpu', depth=depth, stream=no_boundaries)
    # -get generator architecture (if used)
    train_gen = True if (args.replay=="generative" and not checkattr(args, 'feedback')) else False
    if train_gen:
        generator = define.define_vae(args=args, config=config, device='cpu', depth=depth)

    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay_model_name=replay_model_name,
                                  feature_extractor_name=feature_extractor_name, no_boundaries=no_boundaries)
    return param_stamp


def get_param_stamp(args, model_name, verbose=True, replay_model_name=None, feature_extractor_name=None,
                    no_boundaries=False):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for problem specification
    multi_n_stamp = "{n}{joint}{cum}-{sce}".format(n=args.contexts, joint="-Joint" if checkattr(args, 'joint') else "",
                                                   cum="-Cummulative" if checkattr(args, 'cummulative') else "",
                                                   sce=args.scenario) if hasattr(args, "contexts") else ""
    stream_stamp = "-{stream}{fuzz}".format(
        stream=args.stream, fuzz="{}-".format(args.fuzziness) if args.stream=="fuzzy-boundaries" else "-"
    ) if no_boundaries else ""
    problem_stamp = "{exp}{stream}{norm}{aug}{multi_n}".format(
        exp=args.experiment, stream=stream_stamp, norm="-N" if hasattr(args, 'normalize') and args.normalize else "",
        aug="+" if hasattr(args, "augment") and args.augment else "", multi_n=multi_n_stamp
    )
    if verbose:
        print(" --> problem:       "+problem_stamp)

    # -for model
    model_stamp = model_name if feature_extractor_name is None else "H{}--{}".format(feature_extractor_name, model_name)
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for training settings
    if checkattr(args, "pre_convE") and hasattr(args, 'depth') and args.depth>0:
        ltag = "" if ((not hasattr(args, "convE_ltag")) or args.convE_ltag=="none") else "-{}{}".format(
            args.convE_ltag, "-ps" if checkattr(args, 'seed_to_ltag') else ""
        )
        pre = "-pCvE{}".format(ltag)
    else:
        pre = ""
    freeze_conv = (checkattr(args, "freeze_convE") and hasattr(args, 'depth') and args.depth>0)
    freeze = "-fCvE" if (freeze_conv and (feature_extractor_name is None)) else ""
    train_stamp = "i{num}-lr{lr}-b{bsz}{pre}{freeze}-{optim}{mom}{neg}{recon}".format(
        num=args.iters, lr=args.lr, bsz=args.batch, pre=pre, freeze=freeze, optim=args.optimizer, mom="-m{}".format(
            args.momentum
        ) if args.optimizer=='sgd' and hasattr(args, 'momentum') and args.momentum>0 else "",
        neg="-{}".format(args.neg_samples) if (
                args.scenario=="class" and (not checkattr(args, 'gen_classifier')) and (not no_boundaries)
        ) else "",
        recon="-{}".format(args.recon_loss) if (
                checkattr(args, 'gen_classifier') or (hasattr(args, 'replay') and args.replay=="generative")
        ) else "",
    )
    if verbose:
        print(" --> train-params:  " + train_stamp)

    # -for parameter regularization
    param_reg_stamp = ""
    if checkattr(args, 'weight_penalty') or checkattr(args, 'precondition'):
        param_reg_stamp = "-"
        # -how is parameter regularization done (weight penalty and/or preconditioning)?
        if checkattr(args, 'weight_penalty'):
            param_reg_stamp += "-PReg{}".format(args.reg_strength)
        if checkattr(args, 'precondition'):
            param_reg_stamp += "-PreC{}".format(args.alpha)
        # -how is the parameter importance computed?
        if args.importance_weighting=='fisher':
            param_reg_stamp += "-FI{}{}{}{}{}{}{}".format(
                "kfac" if checkattr(args, 'fisher_kfac') else 'diag',
                "I{}".format(args.data_size) if checkattr(args, 'fisher_init') else "",
                "N" if args.fisher_n is None else args.fisher_n,
                "Emp" if args.fisher_labels=="true" else ("Pred" if args.fisher_labels=="pred" else (
                    "Sam" if args.fisher_labels=="sample" else "All"
                )),
                "B{}".format(args.fisher_batch) if (hasattr(args, 'fisher_batch') and args.fisher_batch>1) else "",
                # -use a separate term per task or a forgetting coefficient:
                "-offline" if checkattr(args, 'offline') else (
                    "-forg{}".format(args.gamma) if hasattr(args, 'gamma') and args.gamma < 1 else ""
                ),
                "-randFI" if checkattr(args, 'randomize_fisher') else "",
            )
        elif args.importance_weighting=='si':
            param_reg_stamp += "-SI{}".format(args.epsilon)
        elif args.importance_weighting=='owm':
            param_reg_stamp += "-OWM"

    # -for context-specific components
    xdg_stamp = ""
    if checkattr(args, 'xdg') and args.gating_prop>0:
        xdg_stamp = "--XdG{}".format(args.gating_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.gating_prop))

    # -for replay / functional regularization (except FROMP)
    replay_stamp = ""
    if hasattr(args, 'replay') and not args.replay=="none":
        replay_stamp = "{rep}{KD}{use}{model}{gi}{lrg}".format(
            rep=args.replay,
            KD="-KD{}".format(args.temp) if checkattr(args, 'distill') else "",
            use="-{}{}".format(
                "A-GEM" if args.use_replay=='inequality' else "both",
                "" if ((not hasattr(args, 'eps_agem')) or args.eps_agem==0) else args.eps_agem
            ) if hasattr(args, 'use_replay') and (not args.use_replay=='normal') else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.gen_iters) if (
                hasattr(args, "gen_iters") and (replay_model_name is not None) and (not args.iters==args.gen_iters)
            ) else "",
            lrg="-glr{}".format(args.lr_gen) if (
                hasattr(args, "lr_gen") and (replay_model_name is not None) and (not args.lr==args.lr_gen)
            ) else "",
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
        replay_stamp = "--{}".format(replay_stamp)

    # -for memory-buffer & its use (e.g., FROMP, iCaRL)
    memory_buffer_stamp = ""
    use_memory_buffer = checkattr(args, 'prototypes') or checkattr(args, 'add_buffer') or args.replay=="buffer" \
                        or checkattr(args, 'fromp')
    if use_memory_buffer:
        buffer_opts = "b{bud}{cap}{sel}".format(
            bud=args.budget, cap="-FC" if checkattr(args, 'use_full_capacity') else "",
            sel=args.sample_selection if hasattr(args, 'sample_selection') else 'random'
        )
        use = "{}{}{}".format("addB-" if checkattr(args, 'add_buffer') else "",
                              "useB-" if checkattr(args, 'prototypes') else "",
                              "fromp{}-".format(args.tau) if checkattr(args, 'fromp') else "")
        memory_buffer_stamp = "--{}{}".format(use, buffer_opts)
        if verbose:
            print(" --> memory buffer: " + "{}{}".format(use, buffer_opts))

    # -for binary classification loss (e.g., iCaRL)
    bin_stamp = ""
    if checkattr(args, 'bce'):
        bin_stamp = '--BCE_dist' if (checkattr(args, 'bce_distill') and args.scenario=="class") else '--BCE'

    # -specific to task-free protocol: how often to update the 'previous_model' relative to which to stay close
    stream_stamp = ""
    if no_boundaries and hasattr(args, 'update_every') and not args.update_every==1:
        if use_memory_buffer or replay_stamp or param_reg_stamp:
            stream_stamp = '--upEv{}'.format(args.update_every)

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}{}".format(
        problem_stamp, model_stamp, train_stamp, param_reg_stamp, xdg_stamp, replay_stamp, memory_buffer_stamp,
        bin_stamp, stream_stamp, "-s{}".format(args.seed) if not args.seed==0 else ""
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp