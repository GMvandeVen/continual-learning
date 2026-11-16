#!/usr/bin/env python3
'''Checks for the IEWC importance estimate. Run from the repository root with: python3 -m unittest tests/test_iewc.py'''
import copy
import unittest
from unittest import mock
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from models.classifier import Classifier
from models.cl import continual_learner
from models.utils.iewc import ResidualDiversityWeights


def get_model_and_data(n_samples=20, seed=0):
    torch.manual_seed(seed)
    model = Classifier(image_size=4, image_channels=1, classes=3, fc_layers=2, fc_units=6, fc_bn=False).double()
    x = torch.randn(n_samples, 1, 4, 4, dtype=torch.float64)
    y = torch.randint(0, 3, (n_samples,))
    return model, TensorDataset(x, y)


def get_importance(model, dataset):
    model = copy.deepcopy(model)
    model.estimate_iewc_fisher(dataset)
    return {n: getattr(model, '{}_EWC_estimated_fisher'.format(n.replace('.', '__'))).clone()
            for n, p in model.named_parameters()}


def manual_importance(model, dataset, kappa=1e-2, weights=None):
    '''Direct implementation of the IEWC formula, used as reference.'''
    x, y = dataset.tensors
    terms, scales = [], []
    for index in range(len(y)):
        output = model(x[index:index+1])
        output_grad = torch.autograd.grad(F.cross_entropy(output, y[index:index+1]), output, retain_graph=True)[0]
        scales.append(output_grad.pow(2).sum())
        model.zero_grad()
        F.cross_entropy(output, y[index:index+1]).backward()
        terms.append({n: p.grad.detach()**2 for n, p in model.named_parameters()})
    scales = torch.stack(scales)
    tau = kappa * torch.quantile(scales[scales > 0], 0.5)
    weights = torch.ones(len(y), dtype=torch.float64) if weights is None else weights
    return {n: sum(weights[i] * terms[i][n] / (scales[i] + tau) for i in range(len(y))) / len(y) for n in terms[0]}


class TestIEWC(unittest.TestCase):

    def assertAllClose(self, a, b):
        for n in a:
            self.assertTrue(torch.allclose(a[n], b[n], rtol=1e-8, atol=1e-12), msg=n)

    def test_matches_formula(self):
        model, dataset = get_model_and_data()
        model.eval()
        self.assertAllClose(get_importance(model, dataset), manual_importance(model, dataset))

    def test_reduces_to_empirical_fisher(self):
        # With unit denominators (s_n=1 and kappa=0), IEWC gives the empirical Fisher of EWC ('--fisher-labels=true')
        model, dataset = get_model_and_data()
        model.iewc_kappa = 0.
        model.iewc_output_norm = lambda r: torch.ones((), dtype=r.dtype)
        iewc = get_importance(model, dataset)
        model.fisher_labels = 'true'
        model.estimate_fisher(dataset)
        # (with a full pass over the data, estimate_fisher() divides the sum over all samples by N-1)
        n = len(dataset)
        ewc = {name: getattr(model, '{}_EWC_estimated_fisher'.format(name.replace('.', '__'))) * (n-1) / n
               for name in iewc}
        self.assertAllClose(iewc, ewc)

    def test_invariant_to_loss_scale(self):
        model, dataset = get_model_and_data()
        reference = get_importance(model, dataset)
        model.iewc_sample_losses = lambda output, label: 1000. * F.cross_entropy(output, label, reduction='none')
        self.assertAllClose(get_importance(model, dataset), reference)

    def test_all_zero(self):
        # If all output-gradients are zero (saturated predictions of the true class), the importance is zero
        model, dataset = get_model_and_data()
        dataset.tensors[1].zero_()
        with torch.no_grad():
            model.classifier.linear.weight.zero_()
            model.classifier.linear.bias.copy_(torch.tensor([1000., 0., 0.]))
        for value in get_importance(model, dataset).values():
            self.assertTrue(torch.all(value == 0))

    def test_rdw(self):
        # Residual-diversity weights of a few known directions
        rdw = ResidualDiversityWeights(max_basis=2)
        e = torch.eye(3, dtype=torch.float64)
        weights = [rdw(g) for g in (2*e[0], e[0], e[0]+e[1], -e[1], e[2], torch.zeros(3))]
        self.assertTrue(torch.allclose(torch.tensor(weights), torch.tensor([1., 0., 0.5, 0., 1., 0.]), atol=1e-6))
        # In the estimate, the weights are normalized to have mean one
        model, dataset = get_model_and_data()
        model.eval()
        model.iewc_weights = 'rdw'
        rdw = ResidualDiversityWeights(max_basis=model.rdw_cap)
        x, y = dataset.tensors
        weights = []
        for index in range(len(y)):
            model.zero_grad()
            F.cross_entropy(model(x[index:index+1]), y[index:index+1]).backward()
            weights.append(rdw(torch.cat([p.grad.flatten() for p in model.parameters()])))
        weights = torch.tensor(weights, dtype=torch.float64)
        weights = weights / weights.mean()
        self.assertAlmostEqual(weights.mean().item(), 1.)
        # (RDW depends on the order of the samples, so here they are not shuffled)
        unshuffled_loader = lambda dataset, batch_size, **kwargs: torch.utils.data.DataLoader(dataset, batch_size)
        with mock.patch.object(continual_learner, 'get_data_loader', unshuffled_loader):
            importance = get_importance(model, dataset)
        self.assertAllClose(importance, manual_importance(model, dataset, weights=weights))


if __name__ == '__main__':
    unittest.main()
