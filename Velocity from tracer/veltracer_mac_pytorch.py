#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity from tracer: standalone PyTorch version of the ODIL example.

Problem:
    Find a velocity field v=(vx,vy) and tracer c(x,y,t) such that
        c_t + v · grad(c) = 0
    while matching known initial and final tracer snapshots.

This code is intentionally close to cselab/odil examples/velocity_from_tracer/veltracer.py,
but it does not require installing the ODIL package. It only needs numpy, matplotlib, torch.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def u_init_blob_np(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Same manufactured tracer snapshots as the ODIL source example."""
    u0 = 0.2
    v0 = 0.2
    r0 = 0.2
    dx = x - u0 * t - 0.3
    dy = y - v0 * t - 0.3
    k = 1.0 + t
    dx = dx * k
    dy = dy / k
    res = np.maximum(0.0, 1.0 - (dx**2 + dy**2) / r0**2)
    res = res**0.2
    return res.astype(np.float32)


def upwind_derivative_x(q: torch.Tensor, vx: torch.Tensor, dx: float) -> torch.Tensor:
    """First-order upwind derivative in x, periodic boundary by torch.roll."""
    qm = torch.roll(q, shifts=1, dims=1)
    qp = torch.roll(q, shifts=-1, dims=1)
    dq = torch.where(vx > 0, q - qm, torch.where(vx < 0, qp - q, 0.5 * (qp - qm)))
    return dq / dx


def upwind_derivative_y(q: torch.Tensor, vy: torch.Tensor, dy: float) -> torch.Tensor:
    """First-order upwind derivative in y, periodic boundary by torch.roll."""
    qm = torch.roll(q, shifts=1, dims=2)
    qp = torch.roll(q, shifts=-1, dims=2)
    dq = torch.where(vy > 0, q - qm, torch.where(vy < 0, qp - q, 0.5 * (qp - qm)))
    return dq / dy


def laplace(q: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """2D periodic Laplacian for each time slice."""
    qxm = torch.roll(q, shifts=1, dims=1)
    qxp = torch.roll(q, shifts=-1, dims=1)
    qym = torch.roll(q, shifts=1, dims=2)
    qyp = torch.roll(q, shifts=-1, dims=2)
    return (qxp - 2 * q + qxm) / dx**2 + (qyp - 2 * q + qym) / dy**2


def make_grid(nx: int, ny: int):
    x1 = (np.arange(nx, dtype=np.float32) + 0.5) / nx
    y1 = (np.arange(ny, dtype=np.float32) + 0.5) / ny
    x, y = np.meshgrid(x1, y1, indexing="ij")
    return x, y


def plot_solution(outdir: Path, epoch: int, x: np.ndarray, y: np.ndarray,
                  c: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                  c0: np.ndarray, c1: np.ndarray, history: list[dict]):
    nt = c.shape[0] - 1
    ids = np.linspace(0, nt, 5, dtype=int)
    times = ids / nt

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), constrained_layout=True)
    skip = max(1, c.shape[1] // 12)
    xs = x[::skip, ::skip]
    ys = y[::skip, ::skip]

    for j, it in enumerate(ids):
        ax = axes[0, j]
        im = ax.imshow(c[it].T, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="YlOrBr")
        ax.quiver(xs, ys, vx[it, ::skip, ::skip], vy[it, ::skip, ::skip], color="k", scale=5, width=0.004)
        ax.set_title(f"t={times[j]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="tracer c")

    for j, it in enumerate(ids):
        ax = axes[1, j]
        speed = np.sqrt(vx[it] ** 2 + vy[it] ** 2)
        im2 = ax.imshow(speed.T, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=max(0.4, float(speed.max())), cmap="viridis")
        ax.set_title(f"|v| at t={times[j]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im2, ax=axes[1, :], shrink=0.8, label="velocity magnitude")

    fig.suptitle(f"Velocity from tracer, epoch={epoch}")
    fig.savefig(outdir / "tracer_velocity.png", dpi=180)
    plt.close(fig)

    # Initial/final comparison figure.
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), constrained_layout=True)
    panels = [(c0, "given c0"), (c[0], "inferred c(t=0)"), (c1, "given c1"), (c[-1], "inferred c(t=1)")]
    for ax, (arr, title) in zip(axes, panels):
        ax.imshow(arr.T, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="YlOrBr")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(outdir / "initial_final_check.png", dpi=180)
    plt.close(fig)

    # Loss history.
    if history:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        epochs = [h["epoch"] for h in history]
        for key in ["loss", "pde", "init", "final", "xreg", "treg"]:
            vals = [h[key] for h in history]
            ax.semilogy(epochs, vals, label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.savefig(outdir / "loss_history.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--Nx", type=int, default=64, help="grid size in x")
    parser.add_argument("--Ny", type=int, default=None, help="grid size in y; default = Nx")
    parser.add_argument("--Nt", type=int, default=None, help="number of time intervals; default = Nx")
    parser.add_argument("--epochs", type=int, default=8000, help="Adam optimization epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument("--kimp", type=float, default=10.0, help="final tracer profile penalty weight")
    parser.add_argument("--kxreg", type=float, default=1e-3, help="Laplacian regularization weight for velocity")
    parser.add_argument("--ktreg", type=float, default=5.0, help="time regularization weight for velocity")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--outdir", type=str, default="out_veltracer_pytorch", help="output directory")
    parser.add_argument("--plot-every", type=int, default=500, help="plot interval; final plot is always saved")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nx = args.Nx
    ny = args.Ny or args.Nx
    nt = args.Nt or args.Nx
    dx = 1.0 / nx
    dy = 1.0 / ny
    dt = 1.0 / nt

    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}, grid: Nt={nt}, Nx={nx}, Ny={ny}")
    print(f"output directory: {outdir.resolve()}")

    x_np, y_np = make_grid(nx, ny)
    c0_np = u_init_blob_np(x_np, y_np, t=0.0)
    c1_np = u_init_blob_np(x_np, y_np, t=1.0)

    # Linear interpolation in time is a better initial guess for c than all zeros.
    ts = np.linspace(0.0, 1.0, nt + 1, dtype=np.float32)[:, None, None]
    c_init_np = (1.0 - ts) * c0_np[None, :, :] + ts * c1_np[None, :, :]

    c0 = torch.tensor(c0_np, device=device)
    c1 = torch.tensor(c1_np, device=device)
    c = torch.nn.Parameter(torch.tensor(c_init_np, device=device))

    # Source ODIL example initializes unknown fields essentially from zero.
    vx = torch.nn.Parameter(torch.zeros((nt + 1, nx, ny), device=device))
    vy = torch.nn.Parameter(torch.zeros((nt + 1, nx, ny), device=device))

    optimizer = torch.optim.Adam([c, vx, vy], lr=args.lr)
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        # Source-like time stepping: derivative of tracer at old time, velocity at new/current time.
        c_old = c[:-1]
        c_new = c[1:]
        vx_cur = vx[1:]
        vy_cur = vy[1:]

        cx = upwind_derivative_x(c_old, vx_cur.detach(), dx)
        cy = upwind_derivative_y(c_old, vy_cur.detach(), dy)
        residual = (c_new - c_old) / dt + vx_cur * cx + vy_cur * cy
        loss_pde = torch.mean(residual**2)

        # Known initial and final snapshots. Dividing by dx follows the scale used in ODIL source.
        loss_init = torch.mean(((c[0] - c0) / dx) ** 2)
        loss_final = torch.mean(((c[-1] - c1) / dx) ** 2) * (args.kimp**2)

        if args.kxreg != 0:
            loss_xreg = torch.mean((args.kxreg * laplace(vx, dx, dy)) ** 2) + torch.mean(
                (args.kxreg * laplace(vy, dx, dy)) ** 2
            )
        else:
            loss_xreg = torch.tensor(0.0, device=device)

        if args.ktreg != 0:
            loss_treg = torch.mean(((vx[1:] - vx[:-1]) * args.ktreg / dt) ** 2) + torch.mean(
                ((vy[1:] - vy[:-1]) * args.ktreg / dt) ** 2
            )
        else:
            loss_treg = torch.tensor(0.0, device=device)

        loss = loss_pde + loss_init + loss_final + loss_xreg + loss_treg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([c, vx, vy], max_norm=100.0)
        optimizer.step()

        # Tracer is a concentration-like variable; keeping it in [0, 1] helps visualization and stability.
        with torch.no_grad():
            c.clamp_(0.0, 1.0)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            rec = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "init": float(loss_init.detach().cpu()),
                "final": float(loss_final.detach().cpu()),
                "xreg": float(loss_xreg.detach().cpu()),
                "treg": float(loss_treg.detach().cpu()),
            }
            history.append(rec)
            print(
                f"epoch {epoch:6d} | loss={rec['loss']:.4e} pde={rec['pde']:.4e} "
                f"init={rec['init']:.4e} final={rec['final']:.4e} "
                f"xreg={rec['xreg']:.4e} treg={rec['treg']:.4e}"
            )

        if args.plot_every > 0 and epoch % args.plot_every == 0:
            plot_solution(
                outdir,
                epoch,
                x_np,
                y_np,
                c.detach().cpu().numpy(),
                vx.detach().cpu().numpy(),
                vy.detach().cpu().numpy(),
                c0_np,
                c1_np,
                history,
            )

    elapsed = time.time() - t0
    print(f"finished in {elapsed:.1f} seconds")

    c_np = c.detach().cpu().numpy()
    vx_np = vx.detach().cpu().numpy()
    vy_np = vy.detach().cpu().numpy()

    plot_solution(outdir, args.epochs, x_np, y_np, c_np, vx_np, vy_np, c0_np, c1_np, history)
    np.savez_compressed(outdir / "result.npz", c=c_np, vx=vx_np, vy=vy_np, c0=c0_np, c1=c1_np)
    print("saved:")
    print(f"  {outdir / 'tracer_velocity.png'}")
    print(f"  {outdir / 'initial_final_check.png'}")
    print(f"  {outdir / 'loss_history.png'}")
    print(f"  {outdir / 'result.npz'}")


if __name__ == "__main__":
    main()
