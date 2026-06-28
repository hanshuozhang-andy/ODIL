#!/usr/bin/env python3
"""
Physics-informed neural network for velocity inference from tracer snapshots.

This script follows the ODIL setup in ../Velocity from tracer/velocity_tracer.py:

    c_t + vx c_x + vy c_y = 0,      (t, x, y) in [0, 1]^3

Only the initial and final tracer snapshots are imposed. The network jointly
infers c(t, x, y), vx(t, x, y), and vy(t, x, y). As in the ODIL code, velocity
regularization is included through spatial Laplacian and time-stationarity
terms.
python3 velocity_tracer_pinn.py --epochs 5000 --outdir out_velocity_tracer_pinn
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


def tracer_blob_torch(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    u0 = 0.2
    v0 = 0.2
    r0 = 0.2

    dx = x - u0 * t - 0.3
    dy = y - v0 * t - 0.3

    k = 1.0 + t
    dx = dx * k
    dy = dy / k

    inside = torch.clamp(1.0 - (dx**2 + dy**2) / r0**2, min=0.0)
    return inside**0.2


def tracer_blob_numpy(x: np.ndarray, y: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    u0 = 0.2
    v0 = 0.2
    r0 = 0.2

    dx = x - u0 * t - 0.3
    dy = y - v0 * t - 0.3

    k = 1.0 + t
    dx = dx * k
    dy = dy / k

    res = np.maximum(0.0, 1.0 - (dx**2 + dy**2) / r0**2)
    return res**0.2


def exact_velocity_numpy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """One smooth velocity field that transports the manufactured tracer blob."""
    a = 0.3 + 0.2 * t
    b = 0.3 + 0.2 * t
    k = 1.0 + t
    vx = 0.2 - (x - a) / k
    vy = 0.2 + (y - b) / k
    return vx, vy


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def sample_unit(n: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, dim, device=device, dtype=dtype)


def unpack_model(model: nn.Module, txy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(txy)
    c = out[:, 0:1]
    vx = out[:, 1:2]
    vy = out[:, 2:3]
    return c, vx, vy


def tracer_losses(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    txy_f = sample_unit(args.n_f, 3, device, dtype).requires_grad_(True)
    c, vx, vy = unpack_model(model, txy_f)
    dc = grad(c, txy_f)
    c_t = dc[:, 0:1]
    c_x = dc[:, 1:2]
    c_y = dc[:, 2:3]
    loss_pde = torch.mean((c_t + vx * c_x + vy * c_y) ** 2)

    xy_i = sample_unit(args.n_i, 2, device, dtype)
    t0 = torch.zeros(args.n_i, 1, device=device, dtype=dtype)
    t1 = torch.ones(args.n_i, 1, device=device, dtype=dtype)

    txy0 = torch.cat([t0, xy_i], dim=1)
    txy1 = torch.cat([t1, xy_i], dim=1)
    c0, _, _ = unpack_model(model, txy0)
    c1, _, _ = unpack_model(model, txy1)
    x = xy_i[:, 0:1]
    y = xy_i[:, 1:2]
    loss_init = torch.mean((c0 - tracer_blob_torch(x, y, t0)) ** 2)
    loss_final = torch.mean((c1 - tracer_blob_torch(x, y, t1)) ** 2)

    zero = torch.zeros((), device=device, dtype=dtype)
    loss_v_lap = zero
    loss_v_t = zero

    if args.w_v_lap:
        dvx = grad(vx, txy_f)
        dvy = grad(vy, txy_f)
        vx_x = dvx[:, 1:2]
        vx_y = dvx[:, 2:3]
        vy_x = dvy[:, 1:2]
        vy_y = dvy[:, 2:3]
        vx_xx = grad(vx_x, txy_f)[:, 1:2]
        vx_yy = grad(vx_y, txy_f)[:, 2:3]
        vy_xx = grad(vy_x, txy_f)[:, 1:2]
        vy_yy = grad(vy_y, txy_f)[:, 2:3]
        loss_v_lap = torch.mean((vx_xx + vx_yy) ** 2) + torch.mean((vy_xx + vy_yy) ** 2)

    if args.w_v_t:
        dvx = grad(vx, txy_f)
        dvy = grad(vy, txy_f)
        loss_v_t = torch.mean(dvx[:, 0:1] ** 2) + torch.mean(dvy[:, 0:1] ** 2)

    total = (
        args.w_pde * loss_pde
        + args.w_init * loss_init
        + args.w_final * loss_final
        + args.w_v_lap * loss_v_lap
        + args.w_v_t * loss_v_t
    )
    return {
        "total": total,
        "pde": loss_pde,
        "init": loss_init,
        "final": loss_final,
        "v_lap": loss_v_lap,
        "v_t": loss_v_t,
    }


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, args.eval_nt)
    x = np.linspace(0.0, 1.0, args.eval_nxy)
    y = np.linspace(0.0, 1.0, args.eval_nxy)
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    points = np.stack([tt.ravel(), xx.ravel(), yy.ravel()], axis=1)

    batches = []
    for start in range(0, len(points), args.eval_batch):
        batch = torch.as_tensor(points[start : start + args.eval_batch], device=device, dtype=dtype)
        batches.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(batches, axis=0).reshape(args.eval_nt, args.eval_nxy, args.eval_nxy, 3)
    c = pred[..., 0]
    vx = pred[..., 1]
    vy = pred[..., 2]
    c_ref = tracer_blob_numpy(xx, yy, tt)
    vx_ref, vy_ref = exact_velocity_numpy(xx, yy, tt)
    return t, x, y, c, vx, vy, np.stack([c_ref, vx_ref, vy_ref], axis=-1)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total",
                "pde",
                "init",
                "final",
                "v_lap",
                "v_t",
                "endpoint_rmse",
                "c_rmse",
                "vx_rmse",
                "vy_rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_outputs(
    outdir: Path,
    history: list[dict[str, float]],
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    ref: np.ndarray,
) -> None:
    slice_ids = np.linspace(0, len(t) - 1, min(5, len(t)), dtype=int)

    fig, axes = plt.subplots(3, len(slice_ids), figsize=(3.2 * len(slice_ids), 8.4), constrained_layout=True)
    if len(slice_ids) == 1:
        axes = axes.reshape(3, 1)

    for j, it in enumerate(slice_ids):
        panels = [
            (ref[it, ..., 0], "imposed/reference c", "YlOrBr", 0.0, 1.0),
            (c[it], "PINN c", "YlOrBr", 0.0, 1.0),
            (c[it] - ref[it, ..., 0], "c error", "RdBu_r", None, None),
        ]
        for i, (arr, ylabel, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            if vmin is None:
                scale = float(max(np.max(np.abs(arr)), 1e-12))
                vmin, vmax = -scale, scale
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
            ax.set_title(f"t={t[it]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "velocity_tracer_c.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(slice_ids), figsize=(3.2 * len(slice_ids), 5.6), constrained_layout=True)
    if len(slice_ids) == 1:
        axes = axes.reshape(2, 1)
    for j, it in enumerate(slice_ids):
        panels = [(vx[it], "vx"), (vy[it], "vy")]
        for i, (arr, ylabel) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(
                arr.T,
                origin="lower",
                extent=[x[0], x[-1], y[0], y[-1]],
                aspect="equal",
                cmap="PuOr_r",
                vmin=-0.5,
                vmax=0.5,
                interpolation="bilinear",
            )
            ax.set_title(f"t={t[it]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "velocity_tracer_v.png", dpi=200)
    plt.close(fig)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key in ["total", "pde", "init", "final", "v_lap", "v_t", "endpoint_rmse"]:
        ax.semilogy(epochs, [max(row[key], 1e-30) for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / error")
    ax.legend(ncol=2)
    fig.savefig(outdir / "velocity_tracer_train.png", dpi=200)
    plt.close(fig)


def evaluate_errors(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grid = predict_grid(model, args, device, dtype)
    _, _, _, c, vx, vy, ref = grid
    c_ref = ref[..., 0]
    vx_ref = ref[..., 1]
    vy_ref = ref[..., 2]
    endpoint_rmse = float(
        np.sqrt(0.5 * (np.mean((c[0] - c_ref[0]) ** 2) + np.mean((c[-1] - c_ref[-1]) ** 2)))
    )
    errors = {
        "endpoint_rmse": endpoint_rmse,
        "c_rmse": float(np.sqrt(np.mean((c - c_ref) ** 2))),
        "vx_rmse": float(np.sqrt(np.mean((vx - vx_ref) ** 2))),
        "vy_rmse": float(np.sqrt(np.mean((vy - vy_ref) ** 2))),
    }
    return errors, grid


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    dtype = torch.float64 if args.double else torch.float32
    model = MLP(3, 3, args.hidden_layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    grid_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = tracer_losses(model, args, device, dtype)
        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            errors, grid_cache = evaluate_errors(model, args, device, dtype)
            row = {
                "epoch": epoch,
                "total": float(losses["total"].detach().cpu()),
                "pde": float(losses["pde"].detach().cpu()),
                "init": float(losses["init"].detach().cpu()),
                "final": float(losses["final"].detach().cpu()),
                "v_lap": float(losses["v_lap"].detach().cpu()),
                "v_t": float(losses["v_t"].detach().cpu()),
                **errors,
            }
            history.append(row)
            print(
                "epoch={epoch:06d} total={total:.4e} pde={pde:.4e} "
                "init={init:.4e} final={final:.4e} v_lap={v_lap:.4e} v_t={v_t:.4e} "
                "endpoint={endpoint_rmse:.4e}".format(**row)
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
            loss = tracer_losses(model, args, device, dtype)["total"]
            loss.backward()
            return loss

        lbfgs.step(closure)
        losses = tracer_losses(model, args, device, dtype)
        errors, grid_cache = evaluate_errors(model, args, device, dtype)
        row = {
            "epoch": args.epochs + args.lbfgs_steps,
            "total": float(losses["total"].detach().cpu()),
            "pde": float(losses["pde"].detach().cpu()),
            "init": float(losses["init"].detach().cpu()),
            "final": float(losses["final"].detach().cpu()),
            "v_lap": float(losses["v_lap"].detach().cpu()),
            "v_t": float(losses["v_t"].detach().cpu()),
            **errors,
        }
        history.append(row)
        print(
            "lbfgs total={total:.4e} pde={pde:.4e} init={init:.4e} "
            "final={final:.4e} endpoint={endpoint_rmse:.4e}".format(**row)
        )

    if grid_cache is None:
        _, grid_cache = evaluate_errors(model, args, device, dtype)
    t, x, y, c, vx, vy, ref = grid_cache
    write_history(outdir / "train.csv", history)
    np.savez(
        outdir / "velocity_tracer_solution.npz",
        t=t,
        x=x,
        y=y,
        c=c,
        vx=vx,
        vy=vy,
        ref_c=ref[..., 0],
        ref_vx=ref[..., 1],
        ref_vy=ref[..., 2],
    )
    torch.save(model.state_dict(), outdir / "velocity_tracer_pinn.pt")
    plot_outputs(outdir, history, t, x, y, c, vx, vy, ref)
    print(f"saved outputs to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs_steps", type=int, default=0)
    parser.add_argument("--n_f", type=int, default=8192, help="advection collocation points")
    parser.add_argument("--n_i", type=int, default=2048, help="endpoint tracer points")
    parser.add_argument("--hidden_layers", type=parse_hidden_layers, default=(64, 64, 64, 64))
    parser.add_argument("--double", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="out_velocity_tracer_pinn")

    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_init", type=float, default=100.0)
    parser.add_argument("--w_final", type=float, default=100.0)
    parser.add_argument("--w_v_lap", type=float, default=0.01)
    parser.add_argument("--w_v_t", type=float, default=1.0)

    parser.add_argument("--eval_nt", type=int, default=5)
    parser.add_argument("--eval_nxy", type=int, default=64)
    parser.add_argument("--eval_batch", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
