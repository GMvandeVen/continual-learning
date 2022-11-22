import tqdm


def train(model, train_loader, iters, loss_cbs=list(), eval_cbs=list()):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].

    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''

    device = model._device()

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break