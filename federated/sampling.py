import numpy as np
from data.manipulate import SubDataset

# Intended for 5GNIDD datasets only
# subdatasets -> dataset divided into contexts
def sample_iid(subdatasets: list[SubDataset], num_clients: int):
    def _sample_iid_subdataset(dataset: SubDataset, num_clients: int):
        """
        Sample I.I.D. client data from dataset
        :param dataset:
        :param num_clients:
        :return: dict of sample indexes
        """

        idxs_per_label = {}
        all_idxs = [ i for i in range(len(dataset)) ]
        labels = dataset.train_labels
        for idx, label in zip(all_idxs, labels):
            idxs_per_label.setdefault(label, set()).add(idx)

        dict_clients = {}
        for idxs in idxs_per_label.values():
            num_items = len(idxs) // num_clients
            for i in range(num_clients):
                subset = set(np.random.choice(list(idxs), num_items, replace=False))
                dict_clients.setdefault(i, set()).update(subset)
                idxs.difference_update(subset)
        return dict_clients

    all_dict_clients = [ _sample_iid_subdataset(subdataset, num_clients) for subdataset in subdatasets ]
    global_dict_clients = { i: [ dc[i] for dc in all_dict_clients ] for i in range(num_clients) }
    return global_dict_clients
