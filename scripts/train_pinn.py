#!/usr/bin/env python3
"""Retrain the reported 53-gene ODE-constrained PINN from the public matrix."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pinn_model import RegulatoryNet, TrajectoryNet, per_gene_derivatives


ROOT = Path(__file__).resolve().parents[1]
METADATA = {"cell_id", "t", "dataset", "sample_type"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    data = pd.read_csv(ROOT / "data" / "pinn_training_data.csv")
    genes = [column for column in data.columns if column not in METADATA]
    x = data[genes].to_numpy(dtype=np.float32)
    t = data["t"].to_numpy(dtype=np.float32)
    if len(genes) != 53 or x.shape[0] != 986:
        raise ValueError(f"Expected 986 cells and 53 genes; observed {x.shape}")

    t_tensor = torch.tensor(t, device=device).unsqueeze(1).requires_grad_(True)
    x_observed = torch.tensor(x, device=device)
    trajectory = TrajectoryNet(len(genes)).to(device)
    regulatory = RegulatoryNet(len(genes)).to(device)
    log_gamma = nn.Parameter(torch.full((len(genes),), np.log(0.3), device=device))

    def gamma():
        return torch.nn.functional.softplus(log_gamma) + 1e-3

    optimizer = torch.optim.Adam(
        list(trajectory.parameters()) + list(regulatory.parameters()) + [log_gamma],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        x_hat = trajectory(t_tensor)
        dx_dt = per_gene_derivatives(x_hat, t_tensor, create_graph=True)
        data_loss = torch.mean((x_hat - x_observed) ** 2)
        physics_loss = torch.mean((dx_dt - (regulatory(x_hat) - gamma() * x_hat)) ** 2)
        total_loss = data_loss + physics_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        history.append((epoch, data_loss.item(), physics_loss.item(), total_loss.item()))
        if epoch == 1 or epoch % 500 == 0:
            print(f"epoch={epoch} data={data_loss.item():.6f} physics={physics_loss.item():.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "trajectory_net": trajectory.state_dict(),
        "regulatory_net": regulatory.state_dict(),
        "log_gamma": log_gamma.detach().cpu(),
        "gamma": gamma().detach().cpu().numpy(),
        "genes": genes,
        "config": {
            "epochs": args.epochs, "lr": 1e-3, "lambda_physics": 1.0,
            "hidden": 128, "seed": 42, "n_cells": len(data),
            "n_genes": len(genes), "device": device,
        },
    }
    torch.save(checkpoint, args.output_dir / "pinn_sc_weights.pt")
    pd.DataFrame(
        history, columns=("epoch", "data_loss", "physics_loss", "total_loss")
    ).to_csv(args.output_dir / "pinn_loss_history.csv", index=False)


if __name__ == "__main__":
    main()

