import itertools
import torch
from torch.utils.data import DataLoader


def repeater(data_loader):
    '''Function to enable looping through a data-loader indefinetely.'''
    for loader in itertools.repeat(data_loader):
        for data in loader:
            yield data


class DataStream:
    '''Iterator for setting up data-stream, with context for each observation or iteration given by `label_stream`.'''

    def __init__(self, datasets, label_stream, batch_size=1, per_batch=False, shuffle=True, return_context=False):
        '''Instantiate the DataStream-object.
        Args:
            datasets (list): list of Datasets, each on representing a context
            label_stream (LabelStream): iterator dictating from which context (task, domain or class) to sample
            batch_size (int, optional): # of samples per mini-batch (default: ``1``)
            per_batch (bool, optional): if ``True``, each label from `label_stream` specifies entire mini-batch;
                if ``False``, there is separate context-label for each sample in a mini-batch (default: ``False``)
            shuffle (bool, optional): whether the DataLoader should shuffle the Datasets (default: ``True``)
            return_context (bool, optional): whether identity of the context should be returned (default: ``False``)
        '''

        self.datasets = datasets
        self.label_stream = label_stream
        self.n_contexts = label_stream.n_contexts
        self.batch_size = batch_size
        self.per_batch = per_batch
        self.return_context = return_context

        # To keep track of the actual label-sequence being used
        self.sequence = []

        # Create separate data-loader for each context (using 'repeater' to enable looping through them indefinitely)
        self.dataloaders = []
        for context_label in range(self.n_contexts):
            self.dataloaders.append(repeater(
                DataLoader(datasets[context_label], batch_size=batch_size if per_batch else 1, shuffle=shuffle,
                           drop_last=True)
            ))

    def __iter__(self):
        return self

    def __next__(self):
        '''Function to return the next batch (x,y,c).'''
        if self.per_batch or self.batch_size == 1:
            # All samples in the mini-batch come from same context.
            context_label = next(self.label_stream)
            self.sequence.append(context_label)
            (x, y) = next(self.dataloaders[context_label])
            c = torch.tensor([context_label]*self.batch_size) if self.return_context else None
        else:
            # Multiple samples per mini-batch that might come from different contexts.
            x = []
            y = []
            c = [] if self.return_context else None
            for _ in range(self.batch_size):
                context_label = next(self.label_stream)
                self.sequence.append(context_label)
                (xi, yi) = next(self.dataloaders[context_label])
                x.append(xi)
                y.append(yi)
                if self.return_context:
                    c.append(context_label)
            x = torch.cat(x)
            y = torch.cat(y)
            c = torch.tensor(c) if self.return_context else None
        return (x, y, c)