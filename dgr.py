import torch
from torch import nn


class Scholar(nn.Module):
    '''Scholar module for Deep Generative Replay (with two separate models).'''

    def __init__(self, generator, solver):
        '''Instantiate a new Scholar-object.

        [generator]:   <Generator> for generating images from previous tasks
        [solver]:      <Solver> for classifying images'''

        super().__init__()
        self.generator = generator
        self.solver = solver


    def sample(self, size, allowed_predictions=None, return_scores=False):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_predictions] <list> of [class_ids] which are allowed to be predicted
                - [return_scores]       <bool>; if True, [y_hat] is also returned

        OUTPUT: - [X]     <4D-tensor> generated images
                - [y]     <1D-tensor> predicted corresponding labels
                - [y_hat] <2D-tensor> predicted "logits"/"scores" for all [allowed_predictions]'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # sample images
        x, _ = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver(x)
        y_hat = y_hat[:, allowed_predictions] if (allowed_predictions is not None) else y_hat

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        # set model back to its initial mode
        self.train(mode=mode)

        return (x, y, y_hat) if return_scores else (x, y)
