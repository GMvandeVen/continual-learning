import utils
from utils import checkattr

##-------------------------------------------------------------------------------------------------------------------##

def define_classifier(args, config, device, depth=0, stream=False):
    if checkattr(args, 'separate_networks'):
        model = define_separate_classifiers(args=args, config=config, device=device, depth=depth)
    elif checkattr(args, 'feedback'):
        model = define_rtf_classifier(args=args, config=config, device=device, depth=depth)
    elif checkattr(args, 'gen_classifier'):
        model = define_generative_classifer(args=args, config=config, device=device, depth=depth)
    elif stream:
        model = define_stream_classifier(args=args, config=config, device=device, depth=depth)
    else:
        model = define_standard_classifier(args=args, config=config, device=device, depth=depth)
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining discriminative classifier model
def define_stream_classifier(args, config, device, depth=0):
    # Import required model
    from models.classifier_stream import Classifier
    # Specify model
    model = Classifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth > 0 else None,
        start_channels=args.channels if depth > 0 else None,
        reducing_layers=args.rl if depth > 0 else None,
        num_blocks=args.n_blocks if depth > 0 else None,
        conv_bn=(True if args.conv_bn == "yes" else False) if depth > 0 else None,
        conv_nl=args.conv_nl if depth > 0 else None,
        no_fnl=True if depth > 0 else None,
        global_pooling=checkattr(args, 'gp') if depth > 0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn == "yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac'),
        # -how to use context-ID
        xdg_prob=args.gating_prop if checkattr(args, 'xdg') else 0.,
        n_contexts=args.contexts,
        multihead=((args.scenario=='task') and not checkattr(args, 'singlehead')),
        device=device
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining discriminative classifier model
def define_standard_classifier(args, config, device, depth=0):
    # Import required model
    from models.classifier import Classifier
    # Specify model
    model = Classifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac')
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining 'replay-through-feedback' model
def define_rtf_classifier(args, config, device, depth=0):
    # Import required model
    from models.cond_vae import CondVAE
    # Specify model
    model = CondVAE(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['output_units'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth > 0 else None,
        start_channels=args.channels if depth > 0 else None,
        reducing_layers=args.rl if depth > 0 else None,
        num_blocks=args.n_blocks if depth > 0 else None,
        conv_bn=(True if args.conv_bn == "yes" else False) if depth > 0 else None,
        conv_nl=args.conv_nl if depth > 0 else None,
        global_pooling=checkattr(args, 'gp') if depth > 0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=(args.fc_bn=="yes"),
        fc_nl=args.fc_nl,
        excit_buffer=True,
        # -prior
        prior=args.prior if hasattr(args, "prior") else "standard",
        n_modes=args.n_modes if hasattr(args, "prior") else 1,
        per_class=args.per_class if hasattr(args, "prior") else False,
        z_dim=args.z_dim,
        # -decoder
        recon_loss=args.recon_loss,
        network_output="none" if checkattr(args, "normalize") else "sigmoid",
        deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
        dg_gates=checkattr(args, 'dg_gates'),
        dg_type=args.dg_type if hasattr(args, 'dg_type') else "context",
        dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
        contexts=args.contexts if hasattr(args, 'contexts') else None,
        scenario=args.scenario if hasattr(args, 'scenario') else None, device=device,
        # -classifier
        classifier=True,
    ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining classifier model with separate network per context
def define_separate_classifiers(args, config, device, depth=0):
    # Import required model
    from models.separate_classifiers import SeparateClassifiers
    # Specify model
    model = SeparateClassifiers(
        image_size=config['size'],
        image_channels=config['channels'],
        classes_per_context=config['classes_per_context'],
        contexts=args.contexts,
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=True if args.fc_bn=="yes" else False,
        fc_nl=args.fc_nl,
        excit_buffer=True,
    ).to(device)
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining generative classifier (with separate VAE per class)
def define_generative_classifer(args, config, device, depth=0):
    # Import required model
    from models.generative_classifier import GenerativeClassifier
    # Specify model
    model = GenerativeClassifier(
        image_size=config['size'],
        image_channels=config['channels'],
        classes=config['classes'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth>0 else None,
        start_channels=args.channels if depth>0 else None,
        reducing_layers=args.rl if depth>0 else None,
        num_blocks=args.n_blocks if depth>0 else None,
        conv_bn=(True if args.conv_bn=="yes" else False) if depth>0 else None,
        conv_nl=args.conv_nl if depth>0 else None,
        no_fnl=True if depth>0 else None,
        global_pooling=checkattr(args, 'gp') if depth>0 else None,
        # -fc-layers
        fc_layers=args.fc_lay,
        fc_units=args.fc_units,
        fc_drop=args.fc_drop,
        fc_bn=(args.fc_bn=="yes"),
        fc_nl=args.fc_nl,
        excit_buffer=True,
        # -prior
        prior=args.prior if hasattr(args, "prior") else "standard",
        n_modes=args.n_modes if hasattr(args, "prior") else 1,
        z_dim=args.z_dim,
        # -decoder
        recon_loss=args.recon_loss,
        network_output="none" if checkattr(args, "normalize") else "sigmoid",
        deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining feature extractor model
def define_feature_extractor(args, config, device):
    # -import required model
    from models.feature_extractor import FeatureExtractor
    # -create model
    model = FeatureExtractor(
        image_size=config['size'],
        image_channels=config['channels'],
        # -conv-layers
        conv_type=args.conv_type,
        depth=args.depth,
        start_channels=args.channels,
        reducing_layers=args.rl,
        num_blocks=args.n_blocks,
        conv_bn=True if args.conv_bn=="yes" else False,
        conv_nl=args.conv_nl,
        global_pooling=checkattr(args, 'gp'),
    ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining VAE model
def define_vae(args, config, device, depth=0):
    # Import required model
    from models.vae import VAE
    # Specify model
    model = VAE(
        image_size=config['size'],
        image_channels=config['channels'],
        # -conv-layers
        depth=depth,
        conv_type=args.conv_type if depth > 0 else None,
        start_channels=args.channels if depth > 0 else None,
        reducing_layers=args.rl if depth > 0 else None,
        num_blocks=args.n_blocks if depth > 0 else None,
        conv_bn=(True if args.conv_bn == "yes" else False) if depth > 0 else None,
        conv_nl=args.conv_nl if depth > 0 else None,
        global_pooling=False if depth > 0 else None,
        # -fc-layers
        fc_layers=args.g_fc_lay if hasattr(args, 'g_fc_lay') else args.fc_lay,
        fc_units=args.g_fc_uni if hasattr(args, 'g_fc_uni') else args.fc_units,
        fc_drop=0,
        fc_bn=(args.fc_bn=="yes"),
        fc_nl=args.fc_nl,
        excit_buffer=True,
        # -prior
        prior=args.prior if hasattr(args, "prior") else "standard",
        n_modes=args.n_modes if hasattr(args, "prior") else 1,
        z_dim=args.g_z_dim if hasattr(args, 'g_z_dim') else args.z_dim,
        # -decoder
        recon_loss=args.recon_loss,
        network_output="none" if checkattr(args, "normalize") else "sigmoid",
        deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
    ).to(device)
    # Return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args, verbose=False):

    ## Initialization
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)

    ## Use pre-training
    if checkattr(args, "pre_convE") and hasattr(model, 'depth') and model.depth>0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag=="none"
        ) else "{}-{}{}".format(model.convE.name, args.convE_ltag,
                                "-s{}".format(args.seed) if checkattr(args, 'seed_to_ltag') else "")
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name, verbose=verbose)

    ## Freeze some parameters?
    if checkattr(args, "freeze_convE") and hasattr(model, 'convE'):
        for param in model.convE.parameters():
            param.requires_grad = False
        model.convE.frozen = True #--> so they're set to .eval() duting trainng to ensure batchnorm-params do not change

##-------------------------------------------------------------------------------------------------------------------##
