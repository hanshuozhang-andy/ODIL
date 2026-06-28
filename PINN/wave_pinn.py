#!/usr/bin/env python3
"""
Physics-informed neural network for the 1D wave equation.

This script follows the ODIL wave example in ../Wave/wave.py:

    u_tt - u_xx = 0,      (t, x) in [0, 1] x [-1, 1]

python3 wave_pinn.py --epochs 5000 --outdir out_wave_pinn
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


def wave_exact_torch(
    t: torch.Tensor, x: torch.Tensor, variant: str
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.zeros_like(torch.broadcast_tensors(t, x)[0])
    ut = torch.zeros_like(u)
    second_shift = 0.5 if variant == "paper" else -0.5
    for i in range(1, 6):
        k = i * math.pi
        a = (x - t + 0.5) * k
        b = (x + t + second_shift) * k
        u = u + torch.cos(a) + torch.cos(b)
        ut = ut + k * torch.sin(a) - k * torch.sin(b)
    return u / 10.0, ut / 10.0


def wave_exact_numpy(
    t: np.ndarray, x: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    shape = np.broadcast_shapes(np.shape(t), np.shape(x))
    u = np.zeros(shape, dtype=np.float64)
    ut = np.zeros(shape, dtype=np.float64)
    second_shift = 0.5 if variant == "paper" else -0.5
    for i in range(1, 6):
        k = i * np.pi
        a = (x - t + 0.5) * k
        b = (x + t + second_shift) * k
        u += np.cos(a) + np.cos(b)
        ut += k * np.sin(a) - k * np.sin(b)
    return u / 10.0, ut / 10.0


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def sample_uniform(
    n: int,
    lower: tuple[float, float],
    upper: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    return lower_t + (upper_t - lower_t) * torch.rand(n, 2, device=device, dtype=dtype)


def wave_losses(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tx_f = sample_uniform(args.n_f, (0.0, -1.0), (1.0, 1.0), device, dtype)
    tx_f.requires_grad_(True)
    u = model(tx_f)
    du = grad(u, tx_f)
    u_t = du[:, 0:1]
    u_x = du[:, 1:2]
    u_tt = grad(u_t, tx_f)[:, 0:1]
    u_xx = grad(u_x, tx_f)[:, 1:2]
    loss_pde = torch.mean((u_tt - u_xx) ** 2)

    x0 = -1.0 + 2.0 * torch.rand(args.n_i, 1, device=device, dtype=dtype)
    t0 = torch.zeros_like(x0)
    tx_i = torch.cat([t0, x0], dim=1).requires_grad_(True)
    u_i = model(tx_i)
    u_i_exact, ut_i_exact = wave_exact_torch(t0, x0, args.exact_variant)
    u_t_i = grad(u_i, tx_i)[:, 0:1]
    loss_ic_u = torch.mean((u_i - u_i_exact) ** 2)
    loss_ic_ut = torch.mean((u_t_i - ut_i_exact) ** 2)

    tb = torch.rand(args.n_b, 1, device=device, dtype=dtype)
    x_left = -torch.ones_like(tb)
    tx_left = torch.cat([tb, x_left], dim=1)
    u_left_exact, _ = wave_exact_torch(tb, x_left, args.exact_variant)
    loss_bc = torch.mean((model(tx_left) - u_left_exact) ** 2)

    if args.right_bc:
        x_right = torch.ones_like(tb)
        tx_right = torch.cat([tb, x_right], dim=1)
        u_right_exact, _ = wave_exact_torch(tb, x_right, args.exact_variant)
        loss_bc = loss_bc + torch.mean((model(tx_right) - u_right_exact) ** 2)

    total = (
        args.w_pde * loss_pde
        + args.w_ic_u * loss_ic_u
        + args.w_ic_ut * loss_ic_ut
        + args.w_bc * loss_bc
    )
    return {
        "total": total,
        "pde": loss_pde,
        "ic_u": loss_ic_u,
        "ic_ut": loss_ic_ut,
        "bc": loss_bc,
    }


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, args.eval_nt)
    x = np.linspace(-1.0, 1.0, args.eval_nx)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    points = np.stack([tt.ravel(), xx.ravel()], axis=1)

    batches = []
    for start in range(0, len(points), args.eval_batch):
        batch = torch.as_tensor(points[start : start + args.eval_batch], device=device, dtype=dtype)
        batches.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(batches, axis=0).reshape(args.eval_nt, args.eval_nx)
    ref, _ = wave_exact_numpy(tt, xx, args.exact_variant)
    return t, x, pred, ref


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "total", "pde", "ic_u", "ic_ut", "bc", "rmse", "rel_rmse"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_wave_odil_style(
    outdir: Path,
    history: list[dict[str, float]],
    t: np.ndarray,
    x: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
) -> None:
    nslices = min(5, len(t))
    epoch = int(history[-1]["epoch"]) if history else 0
    vmax = float(max(np.max(np.abs(ref)), np.max(np.abs(pred)), 1e-12))
    ptp = 2.0 * vmax
    slice_ylim = (-vmax - 0.1 * ptp, vmax + 0.1 * ptp)
    slice_ids = np.linspace(0, len(t) - 1, nslices, dtype=int)

    fig = plt.figure(figsize=(6.2, 4.8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.suptitle(f"u epoch={epoch}", fontsize=14, y=0.97)

    spec = fig.add_gridspec(2 * nslices, 3, width_ratios=[1.0, 1.0, 0.9])
    heat_axes = []
    extent = [x[0], x[-1], t[0], t[-1]]
    for col, data in enumerate((pred, ref)):
        ax = fig.add_subplot(spec[1:-1, col])
        heat_axes.append(ax)
        ax.imshow(
            data,
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
            origin="lower",
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(t[0], t[-1])
        ax.spines[:].set_linewidth(0.4)

    line_ref = None
    line_pred = None
    for i, tidx in enumerate(slice_ids):
        row = nslices - 1 - i
        ax = fig.add_subplot(spec[2 * row : 2 * row + 2, 2])
        (line_ref,) = ax.plot(x, ref[tidx], c="C2", lw=1.3, label="reference")
        (line_pred,) = ax.plot(x, pred[tidx], c="C0", lw=1.0, label="inferred")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(slice_ylim)
        ax.spines[:].set_linewidth(0.4)
        ax.arrow(
            -0.02,
            0.5,
            -0.05,
            0.0,
            overhang=0.0,
            head_width=0.05,
            head_length=0.03,
            linewidth=0.5,
            transform=ax.transAxes,
            facecolor="black",
            edgecolor="black",
            clip_on=False,
        )

    if line_pred is not None and line_ref is not None:
        fig.legend(
            handles=[line_pred, line_ref],
            labels=["inferred", "reference"],
            loc="upper center",
            bbox_to_anchor=(0.44, 0.88),
            ncol=2,
            frameon=False,
            handlelength=2.5,
            columnspacing=2.5,
            fontsize=12,
        )

    fig.savefig(outdir / "wave_odil_style.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_outputs(outdir: Path, history: list[dict[str, float]], t, x, pred, ref) -> None:
    err = pred - ref
    vmax = float(max(np.max(np.abs(ref)), np.max(np.abs(pred)), 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    panels = [
        (ref, "reference u", "RdBu_r", -vmax, vmax),
        (pred, "PINN u", "RdBu_r", -vmax, vmax),
        (err, "error", "RdBu_r", -float(np.max(np.abs(err))), float(np.max(np.abs(err)))),
    ]
    for ax, (arr, title, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[x[0], x[-1], t[0], t[-1]],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_i,
            interpolation="bilinear",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "wave_field.png", dpi=200)
    plt.close(fig)

    plot_wave_odil_style(outdir, history, t, x, pred, ref)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for key in ["total", "pde", "ic_u", "ic_ut", "bc", "rmse"]:
        ax.semilogy(epochs, [max(row[key], 1e-30) for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / error")
    ax.legend()
    fig.savefig(outdir / "wave_train.png", dpi=200)
    plt.close(fig)


def evaluate_rmse(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    t, x, pred, ref = predict_grid(model, args, device, dtype)
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
    rel_rmse = float(rmse / max(np.sqrt(np.mean(ref**2)), 1e-12))
    return rmse, rel_rmse, (t, x, pred, ref)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    dtype = torch.float64 if args.double else torch.float32
    model = MLP(2, 1, args.hidden_layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    grid_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = wave_losses(model, args, device, dtype)
        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            rmse, rel_rmse, grid_cache = evaluate_rmse(model, args, device, dtype)
            row = {
                "epoch": epoch,
                "total": float(losses["total"].detach().cpu()),
                "pde": float(losses["pde"].detach().cpu()),
                "ic_u": float(losses["ic_u"].detach().cpu()),
                "ic_ut": float(losses["ic_ut"].detach().cpu()),
                "bc": float(losses["bc"].detach().cpu()),
                "rmse": rmse,
                "rel_rmse": rel_rmse,
            }
            history.append(row)
            print(
                "epoch={epoch:06d} total={total:.4e} pde={pde:.4e} "
                "ic_u={ic_u:.4e} ic_ut={ic_ut:.4e} bc={bc:.4e} "
                "rmse={rmse:.4e} rel={rel_rmse:.4e}".format(**row)
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
            loss = wave_losses(model, args, device, dtype)["total"]
            loss.backward()
            return loss

        lbfgs.step(closure)
        losses = wave_losses(model, args, device, dtype)
        rmse, rel_rmse, grid_cache = evaluate_rmse(model, args, device, dtype)
        row = {
            "epoch": args.epochs + args.lbfgs_steps,
            "total": float(losses["total"].detach().cpu()),
            "pde": float(losses["pde"].detach().cpu()),
            "ic_u": float(losses["ic_u"].detach().cpu()),
            "ic_ut": float(losses["ic_ut"].detach().cpu()),
            "bc": float(losses["bc"].detach().cpu()),
            "rmse": rmse,
            "rel_rmse": rel_rmse,
        }
        history.append(row)
        print(
            "lbfgs total={total:.4e} pde={pde:.4e} ic_u={ic_u:.4e} "
            "ic_ut={ic_ut:.4e} bc={bc:.4e} rmse={rmse:.4e} rel={rel_rmse:.4e}".format(**row)
        )

    if grid_cache is None:
        _, _, grid_cache = evaluate_rmse(model, args, device, dtype)
    t, x, pred, ref = grid_cache
    write_history(outdir / "train.csv", history)
    np.savez(outdir / "wave_solution.npz", t=t, x=x, u=pred, ref_u=ref, error=pred - ref)
    torch.save(model.state_dict(), outdir / "wave_pinn.pt")
    plot_outputs(outdir, history, t, x, pred, ref)
    print(f"saved outputs to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs_steps", type=int, default=0)
    parser.add_argument("--n_f", type=int, default=4096, help="PDE collocation points")
    parser.add_argument("--n_i", type=int, default=512, help="initial-condition points")
    parser.add_argument("--n_b", type=int, default=512, help="boundary points per side")
    parser.add_argument("--hidden_layers", type=parse_hidden_layers, default=(64, 64, 64, 64))
    parser.add_argument("--double", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="out_wave_pinn")

    parser.add_argument("--exact_variant", choices=["paper", "source"], default="paper")
    parser.add_argument("--right_bc", type=int, choices=[0, 1], default=1)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_ic_u", type=float, default=10.0)
    parser.add_argument("--w_ic_ut", type=float, default=10.0)
    parser.add_argument("--w_bc", type=float, default=10.0)

    parser.add_argument("--eval_nt", type=int, default=101)
    parser.add_argument("--eval_nx", type=int, default=201)
    parser.add_argument("--eval_batch", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
