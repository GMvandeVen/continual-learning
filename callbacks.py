import visual_visdom
import evaluate


#########################################################
## Callback-functions for evaluating model-performance ##
#########################################################

def _sample_cb(log, config, visdom=None, test_datasets=None, sample_size=64, iters_per_task=None):
    '''Initiates function for evaluating samples of generative model.

    [test_datasets]     None or <list> of <Datasets> (if provided, also reconstructions are shown)'''

    def sample_cb(generator, batch, task=1):
        '''Callback-function, to evaluate sample (and reconstruction) ability of the model.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        if iteration % log == 0:

            # Evaluate reconstruction-ability of model on [test_dataset]
            if test_datasets is not None:
                # Reconstruct samples from current task
                evaluate.show_reconstruction(generator, test_datasets[task-1], config, size=int(sample_size/2),
                                             visdom=visdom, task=task)

            # Generate samples
            evaluate.show_samples(generator, config, visdom=visdom, size=sample_size,
                                  title="Generated images after {} iters in task {}".format(batch, task))

    # Return the callback-function (except if neither visdom or pdf is selected!)
    return sample_cb if (visdom is not None) else None



def _eval_cb(log, test_datasets, visdom=None, precision_dict=None, iters_per_task=None,
             test_size=None, classes_per_task=None, scenario="class", summary_graph=True, with_exemplars=False):
    '''Initiates function for evaluating performance of classifier (in terms of precision).

    [test_datasets]     <list> of <Datasets>; also if only 1 task, it should be presented as a list!
    [classes_per_task]  <int> number of "active" classes per task
    [scenario]          <str> how to decide which classes to include during evaluating precision'''

    def eval_cb(classifier, batch, task=1):
        '''Callback-function, to evaluate performance of classifier.'''

        iteration = batch if task==1 else (task-1)*iters_per_task + batch

        # evaluate the solver on multiple tasks (and log to visdom)
        if iteration % log == 0:
            evaluate.precision(classifier, test_datasets, task, iteration,
                               classes_per_task=classes_per_task, scenario=scenario, precision_dict=precision_dict,
                               test_size=test_size, visdom=visdom, summary_graph=summary_graph,
                               with_exemplars=with_exemplars)

    ## Return the callback-function (except if neither visdom or [precision_dict] is selected!)
    return eval_cb if ((visdom is not None) or (precision_dict is not None)) else None



##------------------------------------------------------------------------------------------------------------------##

###############################################################
## Callback-functions for keeping track of training-progress ##
###############################################################

def _solver_loss_cb(log, visdom, model=None, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates function for keeping track of, and reporting on, the progress of the solver's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to call on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            bar.set_description(
                '  <SOLVER>   |{t_stm} training loss: {loss:.3} | training precision: {prec:.3} |'
                    .format(t_stm=task_stm, loss=loss_dict['loss_total'], prec=loss_dict['precision'])
            )
            bar.update(1)

        # log the loss of the solver (to visdom)
        if (iteration % log == 0) and (visdom is not None):
            if tasks is None or tasks==1:
                plot_data = [loss_dict['pred']]
                names = ['prediction']
            else:
                weight_new_task = 1. / task if replay else 1.
                plot_data = [weight_new_task*loss_dict['pred']]
                names = ['pred']
                if replay:
                    if model.replay_targets=="hard":
                        plot_data += [(1-weight_new_task)*loss_dict['pred_r']]
                        names += ['pred - r']
                    elif model.replay_targets=="soft":
                        plot_data += [(1-weight_new_task)*loss_dict['distil_r']]
                        names += ['distill - r']
                if model.ewc_lambda>0:
                    plot_data += [model.ewc_lambda * loss_dict['ewc']]
                    names += ['EWC (lambda={})'.format(model.ewc_lambda)]
                if model.si_c>0:
                    plot_data += [model.si_c * loss_dict['si_loss']]
                    names += ['SI (c={})'.format(model.si_c)]
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="SOLVER: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function.
    return cb



def _VAE_loss_cb(log, visdom, model, tasks=None, iters_per_task=None, replay=False, progress_bar=True):
    '''Initiates functions for keeping track of, and reporting on, the progress of the generator's training.'''

    def cb(bar, iter, loss_dict, task=1):
        '''Callback-function, to perform on every iteration to keep track of training progress.'''

        iteration = iter if task==1 else (task-1)*iters_per_task + iter

        # progress-bar
        if progress_bar and bar is not None:
            task_stm = "" if (tasks is None) else " Task: {}/{} |".format(task, tasks)
            bar.set_description(
                '  <VAE>      |{t_stm} training loss: {loss:.3} | training precision: {prec:.3} |'
                    .format(t_stm=task_stm, loss=loss_dict['loss_total'], prec=loss_dict['precision'])
            )
            bar.update(1)

        # log the loss of the solver (to visdom)
        if (iteration % log == 0) and (visdom is not None):
            if tasks is None or tasks==1:
                plot_data = [loss_dict['recon'], loss_dict['variat']]
                names = ['Recon', 'Variat']
                if model.lamda_pl > 0:
                    plot_data += [loss_dict['pred']]
                    names += ['Prediction']
            else:
                weight_new_task = 1. / task if replay else 1.
                plot_data = [weight_new_task*loss_dict['recon'], weight_new_task*loss_dict['variat']]
                names = ['Recon', 'Variat']
                if model.lamda_pl > 0:
                    plot_data += [weight_new_task*loss_dict['pred']]
                    names += ['Prediction']
                if replay:
                    plot_data += [(1-weight_new_task)*loss_dict['recon_r'], (1-weight_new_task)*loss_dict['variat_r']]
                    names += ['Recon - r', 'Variat - r']
                    if model.lamda_pl>0:
                        if model.replay_targets=="hard":
                            plot_data += [(1-weight_new_task)*loss_dict['pred_r']]
                            names += ['pred - r']
                        elif model.replay_targets=="soft":
                            plot_data += [(1-weight_new_task)*loss_dict['distil_r']]
                            names += ['distill - r']
            visual_visdom.visualize_scalars(
                scalars=plot_data, names=names, iteration=iteration,
                title="VAE: loss ({})".format(visdom["graph"]), env=visdom["env"], ylabel="training loss"
            )

    # Return the callback-function
    return cb
