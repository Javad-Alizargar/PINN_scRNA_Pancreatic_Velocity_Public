"""Neural components and exact per-gene derivatives for the public PINN."""

import torch
import torch.nn as nn


class TrajectoryNet(nn.Module):
    def __init__(self, n_genes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_genes), nn.Softplus(),
        )

    def forward(self, t):
        return self.net(t)


class RegulatoryNet(nn.Module):
    def __init__(self, n_genes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_genes), nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def per_gene_derivatives(x_hat, t, create_graph=False):
    """Return the cellwise derivative of each output gene with respect to t."""
    columns = [
        torch.autograd.grad(
            x_hat[:, j].sum(), t, retain_graph=True,
            create_graph=create_graph,
        )[0]
        for j in range(x_hat.shape[1])
    ]
    return torch.cat(columns, dim=1)

