import copy
import numpy as np
from torch import nn
from tqdm import tqdm
from data.manipulate import SubDataset
from federated.sampling import sample_iid

from federated.update import LocalUpdate
from federated.utils import average_weights
from train.train_task_based import train_cl


# Intended for 5GNIDD dataset only
def train_fl(
    global_model: nn.Module,
    train_datasets: list[SubDataset],
    local_train_fn=train_cl,
    sample_fn=sample_iid,
    global_iters=10,
    local_iters=2000,
    frac=0.1,
    num_clients=10,
    batch_size=32,
    baseline="none",
    loss_cbs=list(),
    eval_cbs=list(),
    sample_cbs=list(),
    context_cbs=list(),
    generator=None,
    gen_iters=0,
    gen_loss_cbs=list(),
    **kwargs,
):
    # Dictionary that contains a list of dataset indexes for each context
    user_groups = sample_fn(train_datasets, num_clients)

    # Set the model to train
    global_model.train()

    # Copy weights
    global_weights = global_model.state_dict()

    # Training

    for epoch in tqdm(range(global_iters), desc="Global Training progress:", leave=True):
        local_weights = []
        # local_losses = []

        global_model.train()
        m = max(int(frac * num_clients), 1)
        idxs_users = np.random.choice(range(num_clients), m, replace=False)

        for idx in tqdm(idxs_users, desc="Local Training progress: ", leave=False):
            print(f" Client ID: {idx} ".center(70, '*'))
            local_update = LocalUpdate(
                train_datasets=train_datasets,
                idxs=user_groups[idx],
                train_fn=local_train_fn,
                iters=local_iters,
                batch_size=batch_size,
                baseline=baseline,
                loss_cbs=loss_cbs,
                eval_cbs=eval_cbs,
                sample_cbs=sample_cbs,
                context_cbs=context_cbs,
                generator=generator,
                gen_iters=gen_iters,
                gen_loss_cbs=gen_loss_cbs,
                **kwargs,
            )
            # w, loss = local_update.update_weights(
            #     model=copy.deepcopy(global_model),
            #     global_round=epoch,
            # )
            weights = local_update.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
            )
            local_weights.append(copy.deepcopy(weights))
            # local_losses.append(copy.deepcopy(loss))

        # Calculate global weights
        global_weights = average_weights(local_weights)

        # Update global weights
        global_model.load_state_dict(global_weights)
