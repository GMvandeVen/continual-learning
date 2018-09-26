import torch
from torch.autograd import Variable
import visual_visdom
import visual_plt
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, collate_fn=None, allowed_classes=None,
             task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer
    if task_mask:
        model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda(), collate_fn=collate_fn)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data = Variable(data).cuda() if model._is_on_cuda() else Variable(data)
        labels = Variable(labels).cuda() if model._is_on_cuda() else Variable(labels)
        if (allowed_classes is not None):
            labels = labels - allowed_classes[0]
        scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
        _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().data[0]
        total_tested += len(data)
    precision = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    precision = {}
    precision["all_tasks"] = [[] for _ in range(n_tasks)]
    precision["average"] = []
    precision["x_iteration"] = []
    precision["x_task"] = []
    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="class",
              precision_dict=None, collate_fn=None, test_size=None, visdom=None, verbose=False, summary_graph=True,
              task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    n_tasks = len(datasets)
    precs = []
    for i in range(n_tasks):
        if i+1 <= current_task:
            if scenario=='domain':
                allowed_classes = None
            elif scenario=='task':
                allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
            elif scenario=='class':
                allowed_classes = list(range(classes_per_task*current_task))
            precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose, collate_fn=collate_fn,
                                  allowed_classes=allowed_classes, task_mask=task_mask, task=i+1))
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

    # Append results to [progress]-dictionary and return
    if precision_dict is not None:
        for task_id, _ in enumerate(names):
            precision_dict["all_tasks"][task_id].append(precs[task_id])
        precision_dict["average"].append(average_precs)
        precision_dict["x_iteration"].append(iteration)
        precision_dict["x_task"].append(current_task)
    return precision_dict



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
    sample, labels = model.sample(size)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()

    # Plot generated images in [pdf] and/or [visdom]
    if pdf is not None:
        visual_plt.plot_images_from_tensor(image_tensor, pdf, title=title)
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='generated samples ({})'.format(visdom["graph"]), env=visdom["env"]
        )

    # Set model back to initial mode
    model.train(mode=mode)



####--------------------------------------------------------------------------------------------------------------####

####---------------------------------####
####----RECONSTRUCTION EVALUATION----####
####---------------------------------####


def show_reconstruction(model, dataset, config, pdf=None, visdom=None, size=32, task=None, collate_fn=None):
    '''Plot reconstructed examples by an auto-encoder [model] on [dataset], in [pdf] and/or in [visdom].'''

    cuda = model._is_on_cuda()

    # Set model to evaluation-mode
    mode = model.training
    model.eval()

    # Get data
    data_loader = utils.get_data_loader(dataset, size, cuda=cuda, collate_fn=collate_fn)
    (data, labels) = next(iter(data_loader))

    # Evaluate model
    data = Variable(data, volatile=True).cuda() if cuda else Variable(data, volatile=True)
    recon_batch, y_hat, mu, logvar, z = model(data, full=True)

    # Plot original and reconstructed images
    comparison = torch.cat(
        [data.view(-1, config['channels'], config['size'], config['size'])[:size],
         recon_batch.view(-1, config['channels'], config['size'], config['size'])[:size]]
    ).cpu()
    image_tensor = comparison.data.view(-1, config['channels'], config['size'], config['size'])
    if pdf is not None:
        task_stm = "" if task is None else " (task {})".format(task)
        visual_plt.plot_images_from_tensor(
            image_tensor, pdf, nrow=8, title="Reconstructions" + task_stm
        )
    if visdom is not None:
        visual_visdom.visualize_images(
            tensor=image_tensor, name='reconstructed samples ({})'.format(visdom["graph"]), env=visdom["env"],
        )

    # Set model back to initial mode
    model.train(mode=mode)