from torch import optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
import utils
import dgr
from continual_learner import ContinualLearner



def train_cl(model, train_datasets, replay_mode="none", scenario="class", classes_per_task=None,
             iters=2000, batch_size=32, collate_fn=None, visualize=True,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list()):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [visualize]         <bool>, whether all losses should be calculated for plotting (even if not used)
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''


    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()

    # Initiate possible sources for replay (no replay for 1st task)
    previous_model = previous_scholar = previous_datasets = None
    exact_replay = generative_replay = current_replay = False

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0 or visualize):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):

        # Do not train if non-positive iterations
        if iters <= 0:
            return

        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            exact_replay = True


        ####################################### MAIN MODEL #######################################

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0 or visualize):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Reset state of optimizer for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task

        # Define a tqdm progress bar
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in progress:

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, collate_fn=collate_fn,
                                                         drop_last=True))
                iters_left = len(data_loader)
            if exact_replay:
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in incremental task learning scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                train_datasets[task_id], batch_size_replay, cuda=cuda,
                                collate_fn=collate_fn, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size, cuda=cuda,
                                                                          collate_fn=collate_fn, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = None
            else:
                # Sample training data of current task, wrap them in (cuda-)Variables
                x, y = next(data_loader)
                x = Variable(x).cuda() if cuda else Variable(x)
                if scenario == "task":
                    y = y - classes_per_task*(task-1) # -> incremental task learning: adjust y-targets to 'active range'
                y = Variable(y).cuda() if cuda else Variable(y)


            #####-----REPLAYED BATCH-----#####
            if not exact_replay and not generative_replay and not current_replay:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Current Replay <<--##
            if current_replay:
                scores_ = None
                if not scenario=="task":
                    # Use same as CURRENT BATCH to replay
                    x_ = x
                    y_ = y if ((model.replay_targets=="hard") or visualize) else None
                    # Get predicted "logits"/"scores" on replayed data (from previous model)
                    if (model.replay_targets=="soft") or visualize:
                        if scenario == "domain":
                            scores_ = Variable(previous_model(x_).data)
                        elif scenario == "class":
                            scores_ = Variable(previous_model(x_)[:, :(classes_per_task * (task - 1))].data)
                            # --> zero probabilities will be added in the [utils.loss_fn_kd]-function
                else:
                    if model.replay_targets=="hard":
                        raise NotImplementedError(
                            "'Current' replay with 'hard targets' not implemented for 'incremental task learning'."
                        )
                    # For each task to replay, use same [x] as in CURRENT BATCH
                    x_ = list()
                    for task_id in range(task-1):
                        x_.append(x)
                    # Get predicted "logits" on replayed data (from previous model)
                    if (model.replay_targets=="soft") or visualize:
                        scores_ = list()
                        for task_id in range(task-1):
                            scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(Variable(scores_temp.data))

            ##-->> Exact Replay <<--##
            if exact_replay:
                scores_ = None
                if not scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables
                    x_, y_ = next(data_loader_previous)
                    x_ = Variable(x_).cuda() if cuda else Variable(x_)
                    if (model.replay_targets=="hard") or visualize:
                        y_ = Variable(y_).cuda() if cuda else Variable(y_)
                    else:
                        y_ = None
                    # Get predicted "logits"/"scores" on replayed data (from previous model)
                    if (model.replay_targets=="soft") or visualize:
                        if scenario=="domain":
                            scores_ = Variable(previous_model(x_).data)
                        elif scenario=="class":
                            scores_ = Variable(previous_model(x_)[:, :(classes_per_task*(task-1))].data)
                            #--> zero probabilities will be added in the [utils.loss_fn_kd]-function
                else:
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(Variable(x_temp).cuda() if cuda else Variable(x_temp))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if (model.replay_targets == "hard") or visualize:
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(Variable(y_temp).cuda() if cuda else Variable(y_temp))
                        else:
                            y_.append(None)
                    # Get predicted "logits" on replayed data (from previous model)
                    if ((model.replay_targets=="soft") or visualize) and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(Variable(scores_temp.data))

            ##-->> Generative Replay <<--##
            if generative_replay:
                if not scenario=="task":
                    # Which classes could be predicted (=[allowed_predictions])?
                    allowed_predictions = None if scenario=="domain" else list(range(classes_per_task*(task-1)))
                    # Sample replayed data, along with their predicted "logits" (both from previous model / scholar)
                    sample_model = previous_model if generator is None else previous_scholar
                    x_, y_, scores_ = sample_model.sample(batch_size, allowed_predictions=allowed_predictions,
                                                          return_scores=True)
                    x_ = Variable(x_).cuda() if cuda else Variable(x_)
                    y_ = Variable(y_).cuda() if cuda else Variable(y_)
                    scores_ = Variable(scores_).cuda() if cuda else Variable(scores_)
                    # -only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                    y_ = y_ if ((model.replay_targets=="hard") or visualize) else None
                    scores_ = scores_ if ((model.replay_targets=="soft") or visualize) else None
                else:
                    x_ = list()
                    y_ = list()
                    scores_ = list()
                    # For each previous task, list which classes could be predicted
                    allowed_pred_list = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(task)]
                    for prev_task_id in range(1, task):
                        # Sample replayed data, along with their predicted "logits" (both from previous model / scholar)
                        sample_model = previous_model if generator is None else previous_scholar
                        batch_size_replay = int(np.ceil(batch_size / (task-1))) if (task > 2) else batch_size
                        x_temp, y_temp, scores_temp = sample_model.sample(
                            batch_size_replay, allowed_predictions=allowed_pred_list[prev_task_id-1],
                            return_scores=True,
                        )
                        x_.append(Variable(x_temp).cuda() if cuda else Variable(x_temp))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if (model.replay_targets == "hard") or visualize:
                            y_.append(Variable(y_temp).cuda() if cuda else Variable(y_temp))
                        else:
                            y_.append(None)
                        # -only keep [scores_] if required (as otherwise unnecessary computations will be done)
                        if (model.replay_targets=="soft") or visualize:
                            scores_.append(Variable(scores_temp).cuda() if cuda else Variable(scores_temp))
                        else:
                            scores_.append(None)


            # Find [active_classes]
            active_classes = None  #-> for "domain"-sce, always all classes are active
            if scenario=="task":
                # -for "task"-sce, create <list> with for all tasks so far a <list> with the active classes
                active_classes = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(task)]
            elif scenario=="class":
                # -for "class"-sce, create one <list> with active classes of all tasks so far
                active_classes = list(range(classes_per_task*task))

            # Train the model with this batch
            loss_dict = model.train_a_batch(
                x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes, task=task, rnt = 1./task,
            )

            # Update running parameter importance estimates in W
            if isinstance(model, ContinualLearner) and (model.si_c>0 or visualize):
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad.data*(p.data-p_old[n]))
                        p_old[n] = p.data.clone()

            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, task=task)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index, task=task)
            if model.label=="VAE":
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(model, batch_index, task=task)


        ####################################### GENERATOR #######################################

        if generator is not None:

            # Reset state of optimizer for every task (if requested)
            if generator.optim_type=="adam_reset":
                generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

            # Initialize number of iters left on current data-loader(s)
            iters_left = iters_left_previous = 1

            # Define a tqdm progress bar
            progress = tqdm.tqdm(range(1, gen_iters+1))

            # Loop over all iterations.
            for batch_index in progress:

                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if iters_left == 0:
                    data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                             collate_fn=collate_fn, drop_last=True))
                    iters_left = len(data_loader)
                if exact_replay:
                    iters_left_previous -= 1
                    if iters_left_previous == 0:
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size, cuda=cuda,
                                                                          collate_fn=collate_fn, drop_last=True))
                        iters_left_previous = len(data_loader_previous)

                # Sample training data of current task, wrap in (cuda-)Variables
                x, _ = next(data_loader)
                x = Variable(x).cuda() if cuda else Variable(x)

                # Sample replayed training data, wrap them (cuda-)Variables
                if exact_replay:
                    x_, _ = next(data_loader_previous)
                    x_ = Variable(x_).cuda() if cuda else Variable(x_)
                elif generative_replay:
                    x_, _ = previous_scholar.sample(batch_size)
                    x_ = Variable(x_).cuda() if cuda else Variable(x_)
                elif current_replay:
                    x_ = x
                else:
                    x_ = None

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y=None, x_=x_, y_=None, rnt=1./task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)


        ##----------> UPON FINISHING EACH TASK...

        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0 or visualize):
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario=="task" else (list(range(classes_per_task*task)) if scenario=="class" else None)
            model.estimate_fisher(train_dataset, allowed_classes=allowed_classes, collate_fn=collate_fn)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0 or visualize):
            model.update_omega(W, model.epsilon)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model)
        previous_model.eval()
        if generator is not None:
            scholar = dgr.Scholar(generator=generator, solver=model)
            previous_scholar = copy.deepcopy(scholar)
        if replay_mode=='generative':
            generative_replay = True
        elif replay_mode=='exact':
            previous_datasets = train_datasets[:task]
            exact_replay = True
        elif replay_mode=='current':
            current_replay = True
