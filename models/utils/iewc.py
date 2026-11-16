"""IEWC utility functions"""
import torch


class ResidualDiversityWeights:
    '''Streaming 'residual-diversity weights' (RDW) for IEWC (Wiest, 2026; https://arxiv.org/abs/2611.16039).

    In a fixed sample order, the weight of a new (flattened) gradient [g] is h = ||(I - Q^T Q) u||^2, with u = g/||g||
    and Q an orthonormal basis of the residual directions of at most [max_basis] earlier gradients. The weights still
    need to be normalized to have mean one. If the basis is large, it is kept on the CPU.'''

    def __init__(self, max_basis=64, eps=1e-12, max_gpu_size=2**26):
        self.max_basis = max_basis
        self.eps = eps
        self.max_gpu_size = max_gpu_size  #-> max # of elements of the basis to keep it on the device of the gradients
        self.basis = None   #-> <tensor> [max_basis]x[n_params], of which the first [self.rank] rows are filled
        self.rank = 0

    def __call__(self, g):
        u = g.detach().flatten().float()
        if self.max_basis * u.numel() > self.max_gpu_size:
            u = u.cpu()
        norm = u.norm()
        if norm == 0:
            return 0.
        u = u / norm
        if self.rank == 0:
            residual = u
        else:
            Q = self.basis[:self.rank]
            residual = u - Q.t() @ (Q @ u)
        h = residual.pow(2).sum().item()
        if h > self.eps and self.rank < self.max_basis:
            if self.basis is None:
                self.basis = torch.zeros(self.max_basis, u.numel(), device=u.device)
            self.basis[self.rank] = residual / h ** 0.5
            self.rank += 1
        return h
