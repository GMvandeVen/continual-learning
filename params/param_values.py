from utils import checkattr


def set_method_options(args, **kwargs):
    # If the 'convenience' option for a specific method is selected, select the corresponding defaults
    if checkattr(args, 'ewc'):
        args.weight_penalty = True
        args.importance_weighting = 'fisher'
        args.offline = True
    if checkattr(args, 'si'):
        args.weight_penalty = True
        args.importance_weighting = 'si'
    if checkattr(args, 'ncl'):
        args.weight_penalty = True
        args.precondition = True
        args.importance_weighting = 'fisher'
        args.fisher_kfac = True
        args.fisher_init = True
    if checkattr(args, 'kfac_ewc'):
        args.weight_penalty = True
        args.importance_weighting = 'fisher'
        args.fisher_kfac = True
    if checkattr(args, 'owm'):
        args.precondition = True
        args.importance_weighting = 'owm'
    if checkattr(args, "lwf"):
        args.replay = "current"
        args.distill = True
    if checkattr(args, 'agem'):
        args.replay = "buffer"
        args.use_replay = "inequality"
    if checkattr(args, 'brain_inspired'):
        args.replay = "generative"
        args.feedback = True  # --> replay-through-feedback
        args.prior = 'GMM'  # --> conditional replay
        args.per_class = True  # --> conditional replay
        args.dg_gates = True  # --> gating based on internal context (has hyper-param 'dg_prop')
        args.hidden = True  # --> internal replay
        args.pre_convE = True  # --> internal replay
        args.distill = True  # --> distillation
    if checkattr(args, "icarl"):
        args.prototypes = True
        args.add_buffer = True
        args.bce = True
        args.bce_distill = True
        args.sample_selection = 'herding'


def set_default_values(args, also_hyper_params=True, single_context=False, no_boundaries=False):
    # -set default-values for certain arguments based on chosen experiment
    args.normalize = args.normalize if args.experiment in ('CIFAR10', 'CIFAR100') else False
    args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
    if not single_context:
        args.contexts = (
            5 if args.experiment in ('splitMNIST', 'CIFAR10') else 10
        ) if args.contexts is None else args.contexts
        args.iters = (2000 if args.experiment == 'splitMNIST' else 5000) if args.iters is None else args.iters
    args.lr = (0.001 if args.experiment == 'splitMNIST' else 0.0001) if args.lr is None else args.lr
    args.batch = (128 if args.experiment in ('splitMNIST', 'permMNIST') else 256) if args.batch is None else args.batch
    if checkattr(args, 'separate_networks'):
        args.fc_units = (100 if args.experiment == 'splitMNIST' else 400) if args.fc_units is None else args.fc_units
    else:
        args.fc_units = (400 if args.experiment == 'splitMNIST' else (
            1000 if args.experiment == 'permMNIST' else 2000
        )) if args.fc_units is None else args.fc_units
    if hasattr(args, 'fc_units_sep'):
        args.fc_units_sep = (
            100 if args.experiment == 'splitMNIST' else 400
        ) if args.fc_units_sep is None else args.fc_units_sep
    if hasattr(args, 'fc_units_gc'):
        args.fc_units_gc = 85 if args.fc_units_gc is None else args.fc_units_gc
        args.fc_lay_gc = (3 if args.experiment == 'splitMNIST' else 2) if args.fc_lay_gc is None else args.fc_lay_gc
        args.z_dim_gc = (5 if args.experiment == 'splitMNIST' else 20) if args.z_dim_gc is None else args.z_dim_gc
    if hasattr(args, 'recon_loss'):
        args.recon_loss = (
            "MSE" if args.experiment in ('CIFAR10', 'CIFAR100') else "BCE"
        ) if args.recon_loss is None else args.recon_loss
    if hasattr(args, "dg_type"):
        args.dg_type = ("context" if args.scenario == 'domain' else "class") if args.dg_type is None else args.dg_type
    if hasattr(args, 'budget'):
        args.budget = (10 if args.experiment == 'permMNIST' else 100) if args.budget is None else args.budget
        if hasattr(args, 'sample_selection'):
            args.sample_selection = ('fromp' if checkattr(args, 'fromp') else (
                'herding' if checkattr(args, 'icarl') else 'random'
            )) if args.sample_selection is None else args.sample_selection
    # -set other default arguments (if they were not selected)
    if hasattr(args, 'lr_gen'):
        args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
        args.g_iters = args.iters if args.g_iters is None else args.g_iters
        args.g_z_dim = args.z_dim if args.g_z_dim is None else args.g_z_dim
        args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
        args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -unless the number of iterations after which to log is explicitly set, set them equal to # of iters per context
    if not single_context:
        args.acc_log = args.iters if (not hasattr(args, 'acc_log')) or args.acc_log is None else args.acc_log
        args.loss_log = args.iters if (not hasattr(args, 'loss_log')) or args.loss_log is None else args.loss_log
        args.sample_log = args.iters if (not hasattr(args,'sample_log')) or args.sample_log is None else args.sample_log

    # -set default-values for certain arguments based on chosen scenario & experiment
    if hasattr(args, 'scenario') and args.scenario == 'task' and hasattr(args, 'gating_prop'):
        # -context-specific gating
        args.gating_prop = (
            0.85 if args.experiment == 'CIFAR100' else (0.9 if args.experiment == 'splitMNIST' else 0.6)
        ) if args.gating_prop is None else args.gating_prop
    if also_hyper_params:
        # -regularization strength
        if not hasattr(args, 'si_c'):
            args.si_c = None
        if not hasattr(args, 'ewc_lambda'):
            args.ewc_lambda = None
        if no_boundaries:
            args.si_c = 10. if args.si_c is None else args.si_c
        elif args.scenario == 'task':
            args.si_c = (
                10. if args.experiment == 'splitMNIST' else (100. if args.experiment == 'CIFAR100' else 10.)
            ) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                100000. if args.experiment == 'splitMNIST' else (1000. if args.experiment == 'CIFAR100' else 100.)
            ) if args.ewc_lambda is None else args.ewc_lambda
        elif args.scenario == 'domain':
            args.si_c = (
                50000. if args.experiment == 'splitMNIST' else (500. if args.experiment == 'CIFAR100' else 10.)
            ) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                10000000000. if args.experiment == 'splitMNIST' else (1000. if args.experiment == 'CIFAR100' else 100.)
            ) if args.ewc_lambda is None else args.ewc_lambda
        elif args.scenario == 'class':
            args.si_c = (5000. if args.experiment == 'splitMNIST' else 5.) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                1000000000. if args.experiment == 'splitMNIST' else 100.
            ) if args.ewc_lambda is None else args.ewc_lambda
        if hasattr(args, 'reg_strength'):
            args.reg_strength = (
                args.si_c if checkattr(args, 'si') else (args.ewc_lambda if checkattr(args, 'ewc') else 1.)
            ) if args.reg_strength is None else args.reg_strength
        # -use a prior for the Fisher (as in NCL)
        if hasattr(args, 'data_size'):
            args.data_size = (12000 if args.experiment == 'splitMNIST' else (
                60000 if args.experiment == 'permMNIST' else (5000 if args.experiment == 'CIFAR100' else 10000)
            )) if args.data_size is None else args.data_size
        # -gating based on internal context (brain-inspired replay)
        if args.scenario == 'task' and hasattr(args, 'dg_prop'):
            args.dg_prop = (0. if args.experiment == 'splitMNIST' else 0.) if args.dg_prop is None else args.dg_prop
        elif args.scenario == 'domain' and hasattr(args, 'dg_prop'):
            args.dg_prop = (0.1 if args.experiment == 'splitMNIST' else 0.5) if args.dg_prop is None else args.dg_prop
        elif args.scenario == 'class' and hasattr(args, 'dg_prop'):
            args.dg_prop = (0.1 if args.experiment == 'splitMNIST' else 0.7) if args.dg_prop is None else args.dg_prop
    if hasattr(args, 'tau'):
        # -fromp
        args.tau = ((0.01 if args.scenario == 'task' else (
            10. if args.scenario == 'domain' else 1000.
        )) if args.experiment == 'splitMNIST' else 1.) if args.tau is None else args.tau


def check_for_errors(args, pretrain=False, **kwargs):
    if pretrain:
        if checkattr(args, 'augment') and not args.experiment in ('CIFAR10', 'CIFAR100'):
            raise ValueError("Augmentation is only supported for 'CIFAR10' or 'CIFAR-100'.")
    if not pretrain:
        if (checkattr(args, 'separate_networks') or checkattr(args, 'xdg')) and (not args.scenario == "task"):
            raise ValueError("'XdG' or 'SeparateNetworks' can only be used with --scenario='task'.")
        # -Replay-through-Feedback model is not (yet) implemented with all possible options
        if checkattr(args, 'feedback') and (checkattr(args, 'precondition') or (
                hasattr(args, 'use_replay') and args.use_replay in ('inequality', 'both')
        )):
            raise NotImplementedError('Replay-through-Feedback currently does not support gradient projection.')
        if checkattr(args, 'feedback') and checkattr(args, 'xdg'):
            raise NotImplementedError('Replay-through-Feedback currently does not support XdG (in the encoder).')
        if checkattr(args, 'feedback') and args.importance_weighting=='fisher' and checkattr(args, 'fisher_kfac'):
            raise NotImplementedError('Replay-through-Feedback currently does not support using KFAC Fisher.')
        if checkattr(args, 'feedback') and checkattr(args, 'bce'):
            raise NotImplementedError('Replay-through-Feedback currently does not support binary classification loss.')
        # -if 'BCEdistill' is selected for other than scenario=="class", give error
        if checkattr(args, 'bce_distill') and not args.scenario=="class":
            raise ValueError("BCE-distill can only be used for class-incremental learning.")
        # -with parameter regularization, not (yet) all combinations are implemented
        if hasattr(args, 'importance_weighting') and args.importance_weighting=='owm' and \
                checkattr(args, 'weight_penalty'):
            raise NotImplementedError('OWM-based importance weighting not supported with parameter weight penalty.')
        if hasattr(args, 'importance_weighting') and args.importance_weighting=='si' and \
                checkattr(args, 'precondition'):
            raise NotImplementedError('SI-based importance weighting not supported with parameter pre-conditioning.')
        # -FROMP has a limited range of options it can be combined with
        if checkattr(args, 'fromp') and hasattr(args, 'optimizer') and args.optimizer=="sgd":
            raise NotImplementedError('FROMP is only supported with ADAM optimizer.')
        if checkattr(args, 'fromp') and hasattr(args, 'replay') and not args.replay=="none":
            raise NotImplementedError('FROMP is not supported combined with replay.')
        if checkattr(args, 'fromp') and (checkattr(args, 'weight_penalty') or checkattr(args, 'precondition')):
            raise NotImplementedError('FROMP is not supported combined with parameter regularization.')
        # -the Generative Classifier implemented here cannot be combined with other approaches
        if checkattr(args, 'gen_classifier') and hasattr(args, 'replay') and not args.replay == "none":
            raise NotImplementedError('The Generative Classifier is not supported with replay.')
        if checkattr(args, 'gen_classifier') and (checkattr(args, 'weight_penalty') or checkattr(args, 'precondition')):
            raise NotImplementedError('The Generative Classifier is not supported with parameter regularization.')
        if checkattr(args, 'gen_classifier') and checkattr(args, 'fromp'):
            raise NotImplementedError('The Generative Classifier is not supported with FROMP.')
        # -a conditional generative model for GR is only supported in combination with Replay-through-Feedback
        if (checkattr(args, 'per_class') or checkattr(args, 'dg_gates')) and not checkattr(args, 'feedback'):
            raise NotImplementedError('A VAE with separate mode per class or context-specific gates in the decoder is '
                                      'only supported in combination with the replay-through-feedback model.')
        # -warning about that XdG and FROMP and KFAC are only applied to fully connected layers?
        trainable_conv = hasattr(args, 'depth') and args.depth>0 and ((not checkattr(args, 'freeze_convE')) or
                                                                      checkattr(args, 'hidden'))
        if checkattr(args, 'xdg') and trainable_conv:
            print('Note that XdG is only applied to the fully connected layers of the network.')
        if checkattr(args, 'fromp') and trainable_conv:
            print('Note that FROMP is only applied to the fully connected layers of the network.')
        if checkattr(args, 'fisher_kfac') and trainable_conv:
            print('Note that parameter regularization based on KFAC Fisher is only applied to '
                  'the fully connected layers of the network.')
        if hasattr(args, 'importance_weighting') and args.importance_weighting=='owm' and trainable_conv:
            print('Note that OWM is only applied to the fully connected layers of the network.')

