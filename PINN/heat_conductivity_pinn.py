#!/usr/bin/env python3
"""
Physics-informed neural network for inferring conductivity from temperature.

This script follows the ODIL heat inverse problem in
../Inferring conductivity from temperature/heat_odil_library_like_source.py
and the paper settings for Fig. 4/5:

    u_t - (k(u) u_x)_x = 0,       (t, x) in [0, 1]^2
    u(t, 0) = u(t, 1) = 0
    u(0, x) = exp(-50 (x - 0.5)^2) - exp(-50)

The unknown conductivity is represented by a small neural network

    k(u) = kmax * sigmoid(k_net(u))

with the paper default architecture 1 x 5 x 5 x 1. The temperature field is
represented by u_net(t, x) with the paper default architecture
2 x 32 x 32 x 32 x 32 x 1.
python3 heat_conductivity_pinn.py --outdir out_heat_conductivity_pinn
python3 heat_conductivity_pinn.py --noise 0.1 --outdir out_heat_conductivity_pinn_noise
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
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


def default_ref_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Inferring conductivity from temperature"
        / "ref"
        / "ref.pickle"
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: list[int]):
        super().__init__()
        widths = [in_dim, *hidden_layers, out_dim]
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


def init_u_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-((x - 0.5) ** 2) * 50.0) - math.exp(-50.0)


def init_u_numpy(x: np.ndarray) -> np.ndarray:
    return np.exp(-((x - 0.5) ** 2) * 50.0) - np.exp(-50.0)


def ref_k_torch(u: torch.Tensor) -> torch.Tensor:
    return 0.02 * torch.exp(-((u - 0.5) ** 2) * 20.0)


def ref_k_numpy(u: np.ndarray) -> np.ndarray:
    return 0.02 * np.exp(-((u - 0.5) ** 2) * 20.0)


def transform_k(raw: torch.Tensor, kmax: float) -> torch.Tensor:
    return kmax * torch.sigmoid(raw)


def load_reference_field(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "fields" in data and "u" in data["fields"]:
        field = data["fields"]["u"]
        if isinstance(field, list):
            field = field[0]
        return np.asarray(field, dtype=np.float64)

    if isinstance(data, dict) and "ref_u" in data:
        return np.asarray(data["ref_u"], dtype=np.float64)

    if isinstance(data, dict) and "state_u" in data:
        return np.asarray(data["state_u"], dtype=np.float64)

    raise ValueError(f"Could not find reference temperature field in {path}")


def interp_cell_centered(field: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a [0, 1]^2 cell-centered grid."""
    nt, nx = field.shape
    tf = np.clip(t * nt - 0.5, 0.0, nt - 1.0)
    xf = np.clip(x * nx - 0.5, 0.0, nx - 1.0)

    t0 = np.floor(tf).astype(np.int64)
    x0 = np.floor(xf).astype(np.int64)
    t1 = np.clip(t0 + 1, 0, nt - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    wt = tf - t0
    wx = xf - x0

    v00 = field[t0, x0]
    v01 = field[t0, x1]
    v10 = field[t1, x0]
    v11 = field[t1, x1]
    return (
        (1.0 - wt) * (1.0 - wx) * v00
        + (1.0 - wt) * wx * v01
        + wt * (1.0 - wx) * v10
        + wt * wx * v11
    )


def make_imposed_data(
    ref_u: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = (np.arange(args.Nt) + 0.5) / args.Nt
    x = (np.arange(args.Nx) + 0.5) / args.Nx
    tt, xx = np.meshgrid(t, x, indexing="ij")
    flat = np.arange(args.Nt * args.Nx)

    if args.imposed == "random":
        candidates = flat
    elif args.imposed == "stripe":
        candidates = flat[np.abs(tt.ravel() - 0.5) < 1.0 / 6.0]
    elif args.imposed == "none":
        candidates = np.array([], dtype=np.int64)
    else:
        raise ValueError(f"Unknown imposed={args.imposed}")

    rng = np.random.default_rng(args.seed)
    if len(candidates):
        count = min(args.nimp, len(candidates))
        indices = np.unique(rng.permutation(candidates)[:count])
    else:
        indices = np.array([], dtype=np.int64)

    points = np.stack([tt.ravel()[indices], xx.ravel()[indices]], axis=1)
    values = interp_cell_centered(ref_u, points[:, 0], points[:, 1])[:, None]
    if args.noise:
        values = values + rng.normal(loc=0.0, scale=args.noise, size=values.shape)
    return points.astype(np.float64), values.astype(np.float64), indices


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


def conductivity(k_net: nn.Module, u: torch.Tensor, kmax: float) -> torch.Tensor:
    return transform_k(k_net(u), kmax)


def heat_losses(
    u_net: nn.Module,
    k_net: nn.Module,
    imposed_points: torch.Tensor,
    imposed_values: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tx_f = sample_unit(args.Nci, 2, device, dtype).requires_grad_(True)
    u = u_net(tx_f)
    du = grad(u, tx_f)
    u_t = du[:, 0:1]
    u_x = du[:, 1:2]
    q = conductivity(k_net, u, args.kmax) * u_x
    q_x = grad(q, tx_f)[:, 1:2]
    loss_pde = torch.mean((u_t - q_x) ** 2)

    tb = torch.rand(args.Ncb, 1, device=device, dtype=dtype)
    xb0 = torch.zeros_like(tb)
    xb1 = torch.ones_like(tb)
    u_b0 = u_net(torch.cat([tb, xb0], dim=1))
    u_b1 = u_net(torch.cat([tb, xb1], dim=1))
    loss_bound = torch.mean(u_b0**2) + torch.mean(u_b1**2)

    xi = torch.rand(args.Ncb, 1, device=device, dtype=dtype)
    ti = torch.zeros_like(xi)
    u_i = u_net(torch.cat([ti, xi], dim=1))
    loss_init = torch.mean((u_i - init_u_torch(xi)) ** 2)

    if imposed_points.numel():
        u_imp = u_net(imposed_points)
        loss_data = torch.mean((u_imp - imposed_values) ** 2)
    else:
        loss_data = torch.zeros((), device=device, dtype=dtype)

    total = (
        args.w_pde * loss_pde
        + args.w_bound * loss_bound
        + args.w_init * loss_init
        + args.kimp**2 * loss_data
    )
    return {
        "total": total,
        "pde": loss_pde,
        "bound": loss_bound,
        "init": loss_init,
        "data": loss_data,
    }


@torch.no_grad()
def predict_grid(
    u_net: nn.Module,
    k_net: nn.Module,
    args: argparse.Namespace,
    ref_u: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, args.eval_nt)
    x = np.linspace(0.0, 1.0, args.eval_nx)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    points = np.stack([tt.ravel(), xx.ravel()], axis=1)

    pred_batches = []
    for start in range(0, len(points), args.eval_batch):
        batch = torch.as_tensor(points[start : start + args.eval_batch], device=device, dtype=dtype)
        pred_batches.append(u_net(batch).detach().cpu().numpy())
    pred_u = np.concatenate(pred_batches, axis=0).reshape(args.eval_nt, args.eval_nx)
    ref_eval = interp_cell_centered(ref_u, tt, xx)

    uk = np.linspace(0.0, 1.0, args.eval_k)
    k_batches = []
    for start in range(0, len(uk), args.eval_batch):
        batch = torch.as_tensor(uk[start : start + args.eval_batch, None], device=device, dtype=dtype)
        k_batches.append(conductivity(k_net, batch, args.kmax).detach().cpu().numpy())
    pred_k = np.concatenate(k_batches, axis=0).reshape(-1)
    ref_k = ref_k_numpy(uk)
    return t, x, pred_u, ref_eval, uk, pred_k, ref_k


def evaluate_errors(
    u_net: nn.Module,
    k_net: nn.Module,
    args: argparse.Namespace,
    ref_u: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grid = predict_grid(u_net, k_net, args, ref_u, device, dtype)
    _, _, pred_u, ref_eval, _, pred_k, ref_k = grid
    u_rmse = float(np.sqrt(np.mean((pred_u - ref_eval) ** 2)))
    u_rel_max = float(u_rmse / max(np.max(np.abs(ref_eval)), 1e-12))
    k_rmse = float(np.sqrt(np.mean((pred_k - ref_k) ** 2)))
    k_rel_max = float(k_rmse / max(np.max(np.abs(ref_k)), 1e-12))
    return {
        "u_rmse": u_rmse,
        "u_rel_max": u_rel_max,
        "k_rmse": k_rmse,
        "k_rel_max": k_rel_max,
    }, grid


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total",
                "pde",
                "bound",
                "init",
                "data",
                "u_rmse",
                "u_rel_max",
                "k_rmse",
                "k_rel_max",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_temperature_odil_style(
    outdir: Path,
    history: list[dict[str, float]],
    t: np.ndarray,
    x: np.ndarray,
    pred_u: np.ndarray,
    ref_u: np.ndarray,
    imposed_points: np.ndarray,
) -> None:
    nslices = min(5, len(t))
    epoch = int(history[-1]["epoch"]) if history else 0
    slice_ids = np.linspace(0, len(t) - 1, nslices, dtype=int)
    extent = [x[0], x[-1], t[0], t[-1]]

    fig = plt.figure(figsize=(6.2, 4.8))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.suptitle(f"u epoch={epoch}", fontsize=14, y=0.97)
    spec = fig.add_gridspec(2 * nslices, 3, width_ratios=[1.0, 1.0, 0.9])

    for col, data in enumerate((pred_u, ref_u)):
        ax = fig.add_subplot(spec[1:-1, col])
        ax.imshow(
            data,
            interpolation="bilinear",
            cmap="YlOrBr",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin="lower",
            aspect="auto",
        )
        if col == 0 and len(imposed_points):
            ax.scatter(
                imposed_points[:, 1],
                imposed_points[:, 0],
                s=2.0,
                alpha=0.9,
                edgecolor="none",
                facecolor="k",
                zorder=100,
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
        (line_ref,) = ax.plot(x, ref_u[tidx], c="C2", lw=1.3, label="reference")
        (line_pred,) = ax.plot(x, pred_u[tidx], c="C0", lw=1.0, label="inferred")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-0.1, 1.1)
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
    fig.savefig(outdir / "heat_u_odil_style.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_outputs(
    outdir: Path,
    history: list[dict[str, float]],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    imposed_points: np.ndarray,
) -> None:
    t, x, pred_u, ref_u, uk, pred_k, ref_k = grid
    err = pred_u - ref_u

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    panels = [
        (ref_u, "reference u", "YlOrBr", 0.0, 1.0),
        (pred_u, "PINN u", "YlOrBr", 0.0, 1.0),
        (err, "u error", "RdBu_r", None, None),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        if vmin is None:
            scale = float(max(np.max(np.abs(arr)), 1e-12))
            vmin, vmax = -scale, scale
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[x[0], x[-1], t[0], t[-1]],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        if title == "PINN u" and len(imposed_points):
            ax.scatter(imposed_points[:, 1], imposed_points[:, 0], s=2.0, c="k", edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "heat_field.png", dpi=200)
    plt.close(fig)

    plot_temperature_odil_style(outdir, history, t, x, pred_u, ref_u, imposed_points)

    fig, ax = plt.subplots(figsize=(3.0, 2.4), constrained_layout=True)
    ax.plot(uk, pred_k, c="C0", lw=1.5, label="inferred")
    ax.plot(uk, ref_k, c="C2", lw=1.5, label="reference")
    ax.set_xlabel("u")
    ax.set_ylabel("k")
    ax.set_ylim(0.0, 0.03)
    ax.legend(frameon=False)
    fig.savefig(outdir / "heat_k.png", dpi=200)
    plt.close(fig)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key in ["total", "pde", "bound", "init", "data", "u_rel_max", "k_rel_max"]:
        ax.semilogy(epochs, [max(row[key], 1e-30) for row in history], label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / error")
    ax.legend(ncol=2)
    fig.savefig(outdir / "heat_train.png", dpi=200)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(args.ref_path) if args.ref_path else default_ref_path()
    ref_u = load_reference_field(ref_path)
    imposed_points_np, imposed_values_np, imposed_indices = make_imposed_data(ref_u, args)

    with (outdir / "imposed.csv").open("w") as f:
        f.write("t,x,u\n")
        for (t, x), (u,) in zip(imposed_points_np, imposed_values_np):
            f.write(f"{t},{x},{u}\n")

    device = get_device(args.device)
    dtype = torch.float64 if args.double else torch.float32
    u_net = MLP(2, 1, args.arch_u).to(device=device, dtype=dtype)
    k_net = MLP(1, 1, args.arch_k).to(device=device, dtype=dtype)
    params = list(u_net.parameters()) + list(k_net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    imposed_points = torch.as_tensor(imposed_points_np, device=device, dtype=dtype)
    imposed_values = torch.as_tensor(imposed_values_np, device=device, dtype=dtype)

    history: list[dict[str, float]] = []
    grid_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = heat_losses(u_net, k_net, imposed_points, imposed_values, args, device, dtype)
        losses["total"].backward()
        optimizer.step()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            errors, grid_cache = evaluate_errors(u_net, k_net, args, ref_u, device, dtype)
            row = {
                "epoch": epoch,
                "total": float(losses["total"].detach().cpu()),
                "pde": float(losses["pde"].detach().cpu()),
                "bound": float(losses["bound"].detach().cpu()),
                "init": float(losses["init"].detach().cpu()),
                "data": float(losses["data"].detach().cpu()),
                **errors,
            }
            history.append(row)
            print(
                "epoch={epoch:06d} total={total:.4e} pde={pde:.4e} "
                "bound={bound:.4e} init={init:.4e} data={data:.4e} "
                "u_rel={u_rel_max:.4e} k_rel={k_rel_max:.4e}".format(**row)
            )

    if args.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            params,
            max_iter=args.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = heat_losses(u_net, k_net, imposed_points, imposed_values, args, device, dtype)["total"]
            loss.backward()
            return loss

        lbfgs.step(closure)
        losses = heat_losses(u_net, k_net, imposed_points, imposed_values, args, device, dtype)
        errors, grid_cache = evaluate_errors(u_net, k_net, args, ref_u, device, dtype)
        row = {
            "epoch": args.epochs + args.lbfgs_steps,
            "total": float(losses["total"].detach().cpu()),
            "pde": float(losses["pde"].detach().cpu()),
            "bound": float(losses["bound"].detach().cpu()),
            "init": float(losses["init"].detach().cpu()),
            "data": float(losses["data"].detach().cpu()),
            **errors,
        }
        history.append(row)
        print(
            "lbfgs total={total:.4e} pde={pde:.4e} data={data:.4e} "
            "u_rel={u_rel_max:.4e} k_rel={k_rel_max:.4e}".format(**row)
        )

    if grid_cache is None:
        _, grid_cache = evaluate_errors(u_net, k_net, args, ref_u, device, dtype)

    t, x, pred_u, ref_eval, uk, pred_k, ref_k = grid_cache
    write_history(outdir / "train.csv", history)
    np.savez(
        outdir / "heat_conductivity_solution.npz",
        t=t,
        x=x,
        u=pred_u,
        ref_u=ref_eval,
        error_u=pred_u - ref_eval,
        uk=uk,
        k=pred_k,
        ref_k=ref_k,
        imposed_points=imposed_points_np,
        imposed_values=imposed_values_np,
        imposed_indices=imposed_indices,
    )
    torch.save({"u_net": u_net.state_dict(), "k_net": k_net.state_dict()}, outdir / "heat_conductivity_pinn.pt")
    plot_outputs(outdir, history, grid_cache, imposed_points_np)
    print(f"saved outputs to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=55000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs_steps", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--double", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--report_every", type=int, default=500)
    parser.add_argument("--outdir", type=str, default="out_heat_conductivity_pinn")

    parser.add_argument("--Nt", type=int, default=64, help="paper ODIL inverse grid in t")
    parser.add_argument("--Nx", type=int, default=64, help="paper ODIL inverse grid in x")
    parser.add_argument("--Nci", type=int, default=4096, help="PINN interior collocation points")
    parser.add_argument("--Ncb", type=int, default=128, help="PINN points on each boundary/initial set")
    parser.add_argument("--arch_u", type=int, nargs="*", default=[32, 32, 32, 32])
    parser.add_argument("--arch_k", type=int, nargs="*", default=[5, 5])
    parser.add_argument("--kmax", type=float, default=0.1)
    parser.add_argument("--kimp", type=float, default=2.0, help="paper data weight wdata")
    parser.add_argument("--imposed", choices=["random", "stripe", "none"], default="stripe")
    parser.add_argument("--nimp", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.0, help="use 0.1 for the noisy Fig. 5 setting")
    parser.add_argument("--ref_path", type=str, default=str(default_ref_path()))

    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bound", type=float, default=1.0)
    parser.add_argument("--w_init", type=float, default=1.0)

    parser.add_argument("--eval_nt", type=int, default=64)
    parser.add_argument("--eval_nx", type=int, default=64)
    parser.add_argument("--eval_k", type=int, default=200)
    parser.add_argument("--eval_batch", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
