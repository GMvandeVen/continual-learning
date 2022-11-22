import torch
import tqdm
import copy
from utils import checkattr
from models.cl.continual_learner import ContinualLearner


def train_on_stream(model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):
    '''Incrementally train a model on a ('task-free') stream of data.
    Args:
        model (Classifier): model to be trained, must have a built-in `train_a_batch`-method
        datastream (DataStream): iterator-object that returns for each iteration the training data
        iters (int, optional): max number of iterations, could be smaller if `datastream` runs out (default: ``2000``)
        *_cbs (list of callback-functions, optional): for evaluating training-progress (defaults: empty lists)
    '''

    # Define tqdm progress bar(s)
    progress = tqdm.tqdm(range(1, iters + 1))

    ##--> SI: Register starting parameter values
    if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
        start_new_W = True
        model.register_starting_param_values()

    previous_model = None

    for batch_id, (x,y,c) in enumerate(datastream, 1):

        if batch_id > iters:
            break

        ##--> SI: Prepare <dicts> to store running importance estimates and param-values before update
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            if start_new_W:
                W, p_old = model.prepare_importance_estimates_dicts()
                start_new_W = False

        # Move data to correct device
        x = x.to(model._device())
        y = y.to(model._device())
        if c is not None:
            c = c.to(model._device())

        # If using separate networks, the y-targets need to be adjusted
        if model.label == "SeparateClassifiers":
            for sample_id in range(x.shape[0]):
                y[sample_id] = y[sample_id] - model.classes_per_context * c[sample_id]

        # Add replay...
        (x_, y_, c_, scores_) = (None, None, None, None)
        if hasattr(model, 'replay_mode') and model.replay_mode=='buffer' and previous_model is not None:
            # ... from the memory buffer
            (x_, y_, c_) = previous_model.sample_from_buffer(x.shape[0])
            if model.replay_targets=='soft':
                with torch.no_grad():
                    scores_ = previous_model.classify(x_, c_, no_prototypes=True)
        elif hasattr(model, 'replay_mode') and model.replay_mode=='current' and previous_model is not None:
            # ... using the data from the current batch (as in LwF)
            x_ = x
            if c is not None:
                c_ = previous_model.sample_contexts(x_.shape[0]).to(model._device())
            with torch.no_grad():
                scores_ = previous_model.classify(x, c_, no_prototypes=True)
                _, y_ = torch.max(scores_, dim=1)
        # -only keep [y_] or [scores_], depending on whether replay is with 'hard' or 'soft' targets
        y_ = y_ if (hasattr(model, 'replay_targets') and model.replay_targets == "hard") else None
        scores_ = scores_ if (hasattr(model, 'replay_targets') and model.replay_targets == "soft") else None

        # Train the model on this batch
        loss_dict = model.train_a_batch(x, y, c, x_=x_, y_=y_, c_=c_, scores_=scores_, rnt=0.5)

        ##--> SI: Update running parameter importance estimates in W (needed for SI)
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            model.update_importance_estimates(W, p_old)

        # Add the observed data to the memory buffer (if selected by the algorithm that fills the memory buffer)
        if checkattr(model, 'use_memory_buffer'):
            model.add_new_samples(x, y, c)
        if hasattr(model, 'replay_mode') and model.replay_mode == 'current' and c is not None:
            model.keep_track_of_contexts_so_far(c)

        # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
        for loss_cb in loss_cbs:
            if loss_cb is not None:
                loss_cb(progress, batch_id, loss_dict)
        for eval_cb in eval_cbs:
            if eval_cb is not None:
                eval_cb(model, batch_id, context=None)

        ##--> SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si' and model.weight_penalty:
            if (batch_id % model.update_every)==0:
                model.update_omega(W, model.epsilon)
                start_new_W = True

        ##--> Replay: update source for replay
        if hasattr(model, 'replay_mode') and (not model.replay_mode=="none"):
            if (batch_id % model.update_every)==0:
                previous_model = copy.deepcopy(model).eval()

    # Close progres-bar(s)
    progress.close()

#------------------------------------------------------------------------------------------------------------#

def train_gen_classifier_on_stream(model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):
    '''Incrementally train a generative classifier model on a ('task-free') stream of data.
    Args:
        model (Classifier): generative classifier, each generative model must have a built-in `train_a_batch`-method
        datastream (DataStream): iterator-object that returns for each iteration the training data
        iters (int, optional): max number of iterations, could be smaller if `datastream` runs out (default: ``2000``)
        *_cbs (list of callback-functions, optional): for evaluating training-progress (defaults: empty lists)
    '''

    # Define tqdm progress bar(s)
    progress = tqdm.tqdm(range(1, iters + 1))

    for batch_id, (x,y,_) in enumerate(datastream, 1):

        if batch_id > iters:
            break

        # Move data to correct device
        x = x.to(model._device())
        y = y.to(model._device())

        # Cycle through all classes. For each class present, take training step on corresponding generative model
        for class_id in range(model.classes):
            if class_id in y:
                x_to_use = x[y==class_id]
                loss_dict = getattr(model, "vae{}".format(class_id)).train_a_batch(x_to_use)
                # NOTE: this way, only the [lost_dict] of the last class present in the batch enters into the [loss_cb]

        # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
        for loss_cb in loss_cbs:
            if loss_cb is not None:
                loss_cb(progress, batch_id, loss_dict)
        for eval_cb in eval_cbs:
            if eval_cb is not None:
                eval_cb(model, batch_id, context=None)

    # Close progres-bar(s)
    progress.close()