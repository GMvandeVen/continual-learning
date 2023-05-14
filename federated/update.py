from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(self, train_datasets, idxs, train_fn, iters, batch_size, baseline, loss_cbs, eval_cbs, sample_cbs, context_cbs, generator, gen_iters, gen_loss_cbs, **kwargs):
        self.traindata = [ DatasetSplit(train_dataset, idxs[i]) for i, train_dataset in enumerate(train_datasets)]
        self.train_fn = train_fn
        self.iters = iters
        self.batch_size = batch_size
        self.baseline = baseline
        self.loss_cbs = loss_cbs
        self.eval_cbs = eval_cbs
        self.sample_cbs = sample_cbs
        self.context_cbs = context_cbs
        self.generator = generator
        self.gen_iters = gen_iters
        self.gen_loss_cbs = gen_loss_cbs
        self.kwargs = kwargs

    def update_weights(self, model, global_round):
        self.train_fn(
            model,
            self.traindata,
            iters=self.iters,
            batch_size=self.batch_size,
            baseline=self.baseline,
            loss_cbs=self.loss_cbs,
            eval_cbs=self.eval_cbs,
            sample_cbs=self.sample_cbs,
            context_cbs=self.context_cbs,
            generator=self.generator,
            gen_iters=self.gen_iters,
            gen_loss_cbs=self.gen_loss_cbs,
            **self.kwargs,
        )
        return model.state_dict()