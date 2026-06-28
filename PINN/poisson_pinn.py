#!/usr/bin/env python3
"""
Physics-informed neural network for the 2D Poisson equation.

This script follows the ODIL Poisson setup in ../Poisson/Poisson_odil_jax.py:

    u_xx + u_yy = f(x, y),      (x, y) in [0, 1]^2
    u = 0 on the boundary

with the manufactured reference solution

    u(x, y) = sin(pi * (k*x)^2) * sin(pi*y).

For the default integer k=4, this reference solution satisfies the zero
boundary condition on all four sides, matching the ODIL run in this folder.
python3 poisson_pinn.py --outdir out_poisson_pinn
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not layers or any(v <= 0 for v in layers):
        raise argparse.ArgumentTypeError("hidden layers must be positive integers")
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: tuple[int, ...]):
        super().__init__()
        widths = (in_dim, *hidden_layers, out_dim)
        layers: list[nn.Module] = []
        for left, right in zip(widths[:-2], widths[1:-1]):
            linear = nn.Linear(left, right)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
        final = nn.Linear(widths[-2], widths[-1])
        nn.init.xavier_normal_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ref_u_torch(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    return torch.sin(math.pi * (k * x) ** 2) * torch.sin(math.pi * y)


def rhs_exact_torch(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    pi = math.pi
    return (
        (
            (-4.0 * k**4 * pi**2 * x**2 - pi**2) * torch.sin(k**2 * pi * x**2)
            + 2.0 * k**2 * pi * torch.cos(k**2 * pi * x**2)
        )
        * torch.sin(pi * y)
    )


def ref_u_numpy(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    return np.sin(np.pi * (k * x) ** 2) * np.sin(np.pi * y)


def rhs_exact_numpy(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    pi = np.pi
    return (
        (
            (-4.0 * k**4 * pi**2 * x**2 - pi**2) * np.sin(k**2 * pi * x**2)
            + 2.0 * k**2 * pi * np.cos(k**2 * pi * x**2)
        )
        * np.sin(pi * y)
    )


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def sample_interior(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, 2, device=device, dtype=dtype)


def sample_boundary(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n_side = max(1, n // 4)
    s = torch.rand(n_side, 1, device=device, dtype=dtype)
    zero = torch.zeros_like(s)
    one = torch.ones_like(s)
    points = [
        torch.cat([zero, s], dim=1),
        torch.cat([one, s], dim=1),
        torch.cat([s, zero], dim=1),
        torch.cat([s, one], dim=1),
    ]
    return torch.cat(points, dim=0)


def poisson_losses(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    xy_f = sample_interior(args.n_f, device, dtype).requires_grad_(True)
    u = model(xy_f)
    du = grad(u, xy_f)
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]
    u_xx = grad(u_x, xy_f)[:, 0:1]
    u_yy = grad(u_y, xy_f)[:, 1:2]
    x = xy_f[:, 0:1]
    y = xy_f[:, 1:2]
    rhs = rhs_exact_torch(x, y, args.k)
    loss_pde = torch.mean((u_xx + u_yy - rhs) ** 2)

    xy_b = sample_boundary(args.n_b, device, dtype)
    if args.bc_mode == "zero":
        u_b = torch.zeros((xy_b.shape[0], 1), device=device, dtype=dtype)
    else:
        u_b = ref_u_torch(xy_b[:, 0:1], xy_b[:, 1:2], args.k)
    loss_bc = torch.mean((model(xy_b) - u_b) ** 2)

    total = args.w_pde * loss_pde + args.w_bc * loss_bc
    return {"total": total, "pde": loss_pde, "bc": loss_bc}


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, args.eval_n)
    y = np.linspace(0.0, 1.0, args.eval_n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    batches = []
    for start in range(0, len(points), args.eval_batch):
        batch = torch.as_tensor(points[start : start + args.eval_batch], device=device, dtype=dtype)
        batches.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(batches, axis=0).reshape(args.eval_n, args.eval_n)
    ref = ref_u_numpy(xx, yy, args.k)
    rhs = rhs_exact_numpy(xx, yy, args.k)
    return x, y, pred, ref, rhs


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "total", "pde", "bc", "rmse", "rel_rmse"]
        )
        writer.writeheader()
        writer.writerows(history)


def plot_outputs(
    outdir: Path,
    history: list[dict[str, float]],
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    rhs: np.ndarray,
) -> None:
    err = pred - ref
    umax = float(max(np.max(np.abs(ref)), np.max(np.abs(pred)), 1e-12))
    emax = float(max(np.max(np.abs(err)), 1e-12))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    panels = [
        (ref, "reference u", "RdBu_r", -umax, umax),
        (pred, "PINN u", "RdBu_r", -umax, umax),
        (err, "error", "RdBu_r", -emax, emax),
        (rhs, "rhs f", "viridis", None, None),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "poisson_field.png", dpi=200)
    plt.close(fig)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for key in ["total", "pde", "bc", "rmse"]:
        ax.semilogy(epochs, [max(row[key], 1e-30) for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / error")
    ax.legend()
    fig.savefig(outdir / "poisson_train.png", dpi=200)
    plt.close(fig)


def evaluate_rmse(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x, y, pred, ref, rhs = predict_grid(model, args, device, dtype)
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    rel_rmse = float(rmse / max(np.sqrt(np.mean(ref**2)), 1e-12))
    return rmse, rel_rmse, (x, y, pred, ref, rhs)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    dtype = torch.float64 if args.double else torch.float32
    model = MLP(2, 1, args.hidden_layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    grid_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = poisson_losses(model, args, device, dtype)
        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            rmse, rel_rmse, grid_cache = evaluate_rmse(model, args, device, dtype)
            row = {
                "epoch": epoch,
                "total": float(losses["total"].detach().cpu()),
                "pde": float(losses["pde"].detach().cpu()),
                "bc": float(losses["bc"].detach().cpu()),
                "rmse": rmse,
                "rel_rmse": rel_rmse,
            }
            history.append(row)
            print(
                "epoch={epoch:06d} total={total:.4e} pde={pde:.4e} "
                "bc={bc:.4e} rmse={rmse:.4e} rel={rel_rmse:.4e}".format(**row)
            )

    if args.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=args.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = poisson_losses(model, args, device, dtype)["total"]
            loss.backward()
            return loss

        lbfgs.step(closure)
        losses = poisson_losses(model, args, device, dtype)
        rmse, rel_rmse, grid_cache = evaluate_rmse(model, args, device, dtype)
        row = {
            "epoch": args.epochs + args.lbfgs_steps,
            "total": float(losses["total"].detach().cpu()),
            "pde": float(losses["pde"].detach().cpu()),
            "bc": float(losses["bc"].detach().cpu()),
            "rmse": rmse,
            "rel_rmse": rel_rmse,
        }
        history.append(row)
        print(
            "lbfgs total={total:.4e} pde={pde:.4e} bc={bc:.4e} "
            "rmse={rmse:.4e} rel={rel_rmse:.4e}".format(**row)
        )

    if grid_cache is None:
        _, _, grid_cache = evaluate_rmse(model, args, device, dtype)
    x, y, pred, ref, rhs = grid_cache
    write_history(outdir / "train.csv", history)
    np.savez(outdir / "poisson_solution.npz", x=x, y=y, u=pred, ref_u=ref, rhs=rhs, error=pred - ref)
    torch.save(model.state_dict(), outdir / "poisson_pinn.pt")
    plot_outputs(outdir, history, x, y, pred, ref, rhs)
    print(f"saved outputs to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs_steps", type=int, default=0)
    parser.add_argument("--n_f", type=int, default=1000, help="PDE collocation points")
    parser.add_argument("--n_b", type=int, default=400, help="total boundary points")
    parser.add_argument("--hidden_layers", type=parse_hidden_layers, default=(32, 32, 32))
    parser.add_argument("--double", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="out_poisson_pinn")

    parser.add_argument("--k", type=float, default=4.0)
    parser.add_argument("--bc_mode", choices=["zero", "exact"], default="zero")
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=100.0)

    parser.add_argument("--eval_n", type=int, default=161)
    parser.add_argument("--eval_batch", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
