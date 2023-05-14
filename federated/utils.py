import copy
import torch

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def fl_exp_details(iid: bool, num_clients: int, frac: int, local_batch_size: int, local_iters: int, global_iters: int):
    print('    Federated parameters:')
    if iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of clients  : {num_clients}')
    print(f'    Fraction of clients  : {frac}')
    print(f'    Local Batch size   : {local_batch_size}')
    print(f'    Local Epochs       : {local_iters}\n')
    print(f'    Global Epochs       : {global_iters}\n')
    return
