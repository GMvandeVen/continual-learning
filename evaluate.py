import numpy as np
import torch
import visual_visdom
import visual_plt
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             with_exemplars=False, no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(model._device()), labels.to(model._device())
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            if with_exemplars:
                predicted = model.classify_with_exemplars(data, allowed_classes=allowed_classes)
                # - in case of Domain-IL scenario, collapse all corresponding domains into same class
                if max(predicted).item() >= model.classes:
                    predicted = predicted % model.classes
            else:
                scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
                _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
              test_size=None, visdom=None, verbose=False, summary_graph=True, with_exemplars=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_tasks = len(datasets)

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    precs = []
    for i in range(n_tasks):
        if i+1 <= current_task:
            if scenario=='domain':
                allowed_classes = None
            elif scenario=='task':
                allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
            elif scenario=='class':
                allowed_classes = list(range(classes_per_task*current_task))
            precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, with_exemplars=with_exemplars,
                                  no_task_mask=no_task_mask, task=i+1))
        else:
            precs.append(0)
    average_precs = sum([precs[task_id] for task_id in range(current_task)]) / current_task

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            precs, names=names, title="precision ({})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylabel="test precision"
        )
        if n_tasks>1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_precs], names=["ave"], title="ave precision ({})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylabel="test precision"
            )



####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####


def initiate_metrics_dict(n_tasks, scenario):
    '''Initiate <dict> with all measures to keep track of.'''
    metrics_dict = {}
    metrics_dict["average"] = []     # ave acc over all tasks so far: Task-IL -> only classes in task
                                     #                                Class-IL-> all classes so far (up to trained task)
    metrics_dict["x_iteration"] = [] # total number of iterations so far
    metrics_dict["x_task"] = []      # number of tasks so far (indicating the task on which training just finished)
    # Accuracy matrix
    if not scenario=="class":
        # -in the domain-incremetnal learning scenario, each task has the same classes
        # -in the task-incremental learning scenario, only the classes within each task are considered
        metrics_dict["acc per task"] = {}
        for i in range(n_tasks):
            metrics_dict["acc per task"]["task {}".format(i+1)] = []
    else:
        # -in the class-incremental learning scenario, accuracy matrix can be defined in different ways
        metrics_dict["acc per task (only classes in task)"] = {}
        metrics_dict["acc per task (all classes up to trained task)"] = {}
        metrics_dict["acc per task (all classes up to evaluated task)"] = {}
        metrics_dict["acc per task (all classes)"] = {}
        for i in range(n_tasks):
            metrics_dict["acc per task (only classes in task)"]["task {}".format(i+1)] = []
            metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(i + 1)] = []
            metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(i + 1)] = []
            metrics_dict["acc per task (all classes)"]["task {}".format(i + 1)] = []
    return metrics_dict


def intial_accuracy(model, datasets, metrics_dict, classes_per_task=None, scenario="domain", test_size=None,
                    verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks using [datasets] before any learning.'''

    n_tasks = len(datasets)

    if not scenario=="class":
        precs = []
    else:
        precs_all_classes = []
        precs_only_classes_in_task = []
        precs_all_classes_upto_task = []

    for i in range(n_tasks):
        if not scenario=="class":
            precision = validate(
                model, datasets[i], test_size=test_size, verbose=verbose,
                allowed_classes=None if scenario=="domain" else list(range(classes_per_task*i, classes_per_task*(i+1))),
                no_task_mask=no_task_mask, task=i+1
            )
            precs.append(precision)
        else:
            # -all classes
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None,
                                 no_task_mask=no_task_mask, task=i + 1)
            precs_all_classes.append(precision)
            # -only classes in task
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
            precs_only_classes_in_task.append(precision)
            # -classes up to evaluated task
            allowed_classes = list(range(classes_per_task * (i + 1)))
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
            precs_all_classes_upto_task.append(precision)

    if not scenario=="class":
        metrics_dict["initial acc per task"] = precs
    else:
        metrics_dict["initial acc per task (all classes)"] = precs_all_classes
        metrics_dict["initial acc per task (only classes in task)"] = precs_only_classes_in_task
        metrics_dict["initial acc per task (all classes up to evaluated task)"] = precs_all_classes_upto_task
    return metrics_dict


def metric_statistics(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
                      metrics_dict=None, test_size=None, verbose=False, with_exemplars=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [metrics_dict]      None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision'''

    n_tasks = len(datasets)

    # Calculate accurcies per task, possibly in various ways (if Class-IL scenario)
    precs_all_classes = []
    precs_all_classes_so_far = []
    precs_only_classes_in_task = []
    precs_all_classes_upto_task = []
    for i in range(n_tasks):
        # -all classes
        if scenario in ('domain', 'class'):
            precision = validate(
                model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None,
                no_task_mask=no_task_mask, task=i + 1, with_exemplars=with_exemplars
            ) if (not with_exemplars) or (i<current_task) else 0.
            precs_all_classes.append(precision)
        # -all classes up to trained task
        if scenario in ('class'):
            allowed_classes = list(range(classes_per_task * current_task))
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (i<current_task) else 0.
            precs_all_classes_so_far.append(precision)
        # -all classes up to evaluated task
        if scenario in ('class'):
            allowed_classes = list(range(classes_per_task * (i+1)))
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
            precs_all_classes_upto_task.append(precision)
        # -only classes in that task
        if scenario in ('task', 'class'):
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
            precision = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
            precs_only_classes_in_task.append(precision)

    # Calcualte average accuracy over all tasks thus far
    if scenario=='task':
        average_precs = sum([precs_only_classes_in_task[task_id] for task_id in range(current_task)]) / current_task
    elif scenario=='domain':
        average_precs = sum([precs_all_classes[task_id] for task_id in range(current_task)]) / current_task
    elif scenario=='class':
        average_precs = sum([precs_all_classes_so_far[task_id] for task_id in range(current_task)]) / current_task

    # Append results to [metrics_dict]-dictionary
    for task_id in range(n_tasks):
        if scenario=="task":
            metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_only_classes_in_task[task_id])
        elif scenario=="domain":
            metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
        else:
            metrics_dict["acc per task (all classes)"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
            metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(task_id+1)].append(
                precs_all_classes_so_far[task_id]
            )
            metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(task_id+1)].append(
                precs_all_classes_upto_task[task_id]
            )
            metrics_dict["acc per task (only classes in task)"]["task {}".format(task_id+1)].append(
                precs_only_classes_in_task[task_id]
            )
    metrics_dict["average"].append(average_precs)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    return metrics_dict



####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----GENERATION EVALUATION----####
####-----------------------------####


def show_samples(model, config, pdf=None, visdom=None, size=32, title="Generated images"):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Generate samples from the model
    sample = model.sample(size)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        visual_plt.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='Generated samples ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)



####--------------------------------------------------------------------------------------------------------------####

####---------------------------------####
####----RECONSTRUCTION EVALUATION----####
####---------------------------------####


def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, task=None, collate_fn=None):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Get data
    data_loader = utils.get_data_loader(dataset, size, cuda=model._is_on_cuda(), collate_fn=collate_fn)
    (data, labels) = next(iter(data_loader))
    data, labels = data.to(model._device()), labels.to(model._device())

    # Evaluate model
    with torch.no_grad():
        recon_batch, y_hat, mu, logvar, z = model(data, full=True)

    # Plot original and reconstructed images
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.view(-1, config['channels'], config['size'], config['size'])
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size*2)))
    # -make plots
    if pdf is not None:
        task_stm = "" if task is None else " (task {})".format(task)
        visual_plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=nrow, title="Reconstructions" + task_stm
        )
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='Reconstructions ({})'.format(visdom["graph"]), env=visdom["env"], nrow=nrow,
        )

    # Set model back to initial mode
    model.train(mode=mode)