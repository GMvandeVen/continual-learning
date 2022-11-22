from eval import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _sample_cb(log, config, visdom=None, test_datasets=None, sample_size=64):
    '''Initiates function for evaluating samples of generative model.

    [test_datasets]     None or <list> of <Datasets> (if provided, also reconstructions are shown)'''

    def sample_cb(generator, batch, context=1, class_id=None, **kwargs):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        if batch % log == 0:

            # Evaluate reconstruction-ability of model on [test_dataset]
            if test_datasets is not None:
                # Reconstruct samples from current context
                evaluate.show_reconstruction(generator, test_datasets[context-1], config, size=int(sample_size/2),
                                             visdom=visdom, context=context)

            # Generate samples
            evaluate.show_samples(
                generator, config, visdom=visdom, size=sample_size,
                visdom_title='Samples{}'.format(" VAE-{}".format(class_id) if class_id is not None else "")
            )

    # Return the callback-function (except if visdom is not selected!)
    return sample_cb if (visdom is not None) else None


def _eval_cb(log, test_datasets, visdom=None, plotting_dict=None, iters_per_context=None, test_size=None,
             summary_graph=True, S='mean'):
    '''Initiates function for evaluating performance of classifier (in terms of accuracy).

    [test_datasets]       <list> of <Datasets>; also if only 1 context, it should be presented as a list!
    '''

    def eval_cb(classifier, batch, context=1):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if (context is None or context==1) else (context-1)*iters_per_context + batch

        # Evaluate the classifier every [log] iterations
        if iteration % log == 0:

            # If needed, set the requested way of doing inference as attributes of the classifier
            if (S is not None) and hasattr(classifier, 'S'):
                classifier.S = S

            # Evaluate the classifier on multiple contexts (and log to visdom)
            evaluate.test_all_so_far(classifier, test_datasets, context, iteration, test_size=test_size,
                                     visdom=visdom, summary_graph=summary_graph, plotting_dict=plotting_dict)

    ## Return the callback-function (except if visdom is not selected!)
    return eval_cb if (visdom is not None) or (plotting_dict is not None) else None


##------------------------------------------------------------------------------------------------------------------##

########################################################################
## Callback-functions for keeping track of loss and training progress ##
########################################################################

def _classifier_loss_cb(log=1, visdom=None, model=None, contexts=None, iters_per_context=None, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the classifier's training.'''

    def cb(bar, iter, loss_dict, context=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        if visdom is not None:
            from visual import visual_visdom

        iteration = iter if context==1 else (context-1)*iters_per_context + iter

        # progress-bar
        if progress_bar and bar is not None:
            context_stm = "" if (contexts is None) else " Context: {}/{} |".format(context, contexts)
            bar.set_description(
                '<CLASSIFIER> |{t_stm} training loss: {loss:.3} | training accuracy: {prec:.3} |'
                    .format(t_stm=context_stm, loss=loss_dict['loss_total'], prec=loss_dict['accuracy'])
            )
            bar.update(1)

        # log the loss of the solver (to visdom)
        if (visdom is not None) and (iteration % log == 0):
            if contexts is None or contexts==1:
                plot_data = [loss_dict['pred']]
                names = ['prediction']
            else:
                plot_data = [loss_dict['pred']]
                names = ['current']
                if hasattr(model, 'replay') and not model.replay=='none':
                    if model.replay_targets == "hard":
                        plot_data += [loss_dict['pred_r']]
                        names += ['replay']
                    elif model.replay_targets == "soft":
                        plot_data += [loss_dict['distil_r']]
                        names += ['distill']
                if hasattr(model, 'reg_strength') and model.reg_strength>0:
                    plot_data += [loss_dict['param_reg']]
                    names += ['param reg']
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="CLASSIFIER: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb


def _VAE_loss_cb(log=1, visdom=None, model=None, contexts=None, iters_per_context=None, replay=False,
                 progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    if visdom is not None:
        from visual import visual_visdom

    def cb(bar, iter, loss_dict, context=1):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter if context==1 else (context-1)*iters_per_context + iter

        # progress-bar
        if progress_bar and bar is not None:
            context_stm = "" if (contexts is None) else " Context: {}/{} |".format(context, contexts)
            bar.set_description('  <VAE>      |{t_stm} training loss: {loss:.3} |{acc}'.format(
                t_stm=context_stm, loss=loss_dict['loss_total'], acc=' training accuracy: {:.3} |'.format(
                    loss_dict['accuracy']
                ) if model.label=='CondVAE' and model.lamda_pl>0 else ''
            ))
            bar.update(1)

        # log the loss of the solver (to visdom)
        if (visdom is not None) and (iteration % log == 0):
            if contexts is None or contexts==1:
                plot_data = [loss_dict['recon'], loss_dict['variat']]
                names = ['Recon', 'Variat']
                if model.lamda_pl > 0:
                    plot_data += [loss_dict['pred']]
                    names += ['Prediction']
            else:
                plot_data = [loss_dict['recon'], loss_dict['variat']]
                names = ['Recon', 'Variat']
                if model.label=='CondVAE' and model.lamda_pl > 0:
                    plot_data += [loss_dict['pred']]
                    names += ['Prediction']
                if replay:
                    plot_data += [loss_dict['recon_r'], loss_dict['variat_r']]
                    names += ['Recon - r', 'Variat - r']
                    if model.label=='CondVAE' and model.lamda_pl>0:
                        if model.replay_targets=="hard":
                            plot_data += [loss_dict['pred_r']]
                            names += ['Pred - r']
                        elif model.replay_targets=="soft":
                            plot_data += [loss_dict['distil_r']]
                            names += ['Distill - r']
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="VAE: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb


def _gen_classifier_loss_cb(log=1, classes=None, visdom=None, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    if visdom is not None:
        from visual import visual_visdom

    def cb(bar, iter, loss_dict, class_id=0):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        # progress-bar
        if progress_bar and bar is not None:
            class_stm = "" if (classes is None) else " Class: {}/{} |".format(class_id+1, classes)
            model_stm = "  <multiple VAEs>   " if (classes is None) else "  <VAE>      "
            bar.set_description('{m_stm}|{c_stm} training loss: {loss:.3} |'
                                .format(m_stm=model_stm, c_stm=class_stm, loss=loss_dict['loss_total']))
            bar.update(1)

        # plot training loss every [log]
        if (visdom is not None) and (iter % log == 0):
            plot_data = [loss_dict['recon'], loss_dict['variat']]
            names = ['Recon loss', 'Variat loss']

            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iter,
                title="VAE{}: loss ({})".format("" if classes is None else "-{}".format(class_id), visdom["graph"]),
                env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb