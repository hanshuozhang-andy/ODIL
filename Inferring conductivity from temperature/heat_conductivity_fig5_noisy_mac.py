#!/usr/bin/env python3
"""
Standalone PyTorch reproduction of Fig. 5 in pgae005.pdf:
"Inferring conductivity from noisy temperature measurements".

This script solves the inverse heat-conductivity ODIL problem with
Gaussian noise sigma=0.1 added to the 200 imposed temperature points.
It is a laptop-friendly standalone version and does not require the
official ODIL package.

Run on macOS:
    python3 -m pip install numpy scipy matplotlib torch
    python3 heat_conductivity_fig5_noisy_mac.py --epochs 5000

Fast test:
    python3 heat_conductivity_fig5_noisy_mac.py --epochs 1000 --Nx 48 --Nt 48 --ref_Nx 128 --ref_Nt 128

More official-like, but slower:
    python3 heat_conductivity_fig5_noisy_mac.py --zero_init --epochs 20000

Outputs:
    out_heat_fig5_noise/heat_results.png
    out_heat_fig5_noise/history.png
    out_heat_fig5_noise/model.pt
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline


def init_u_np(x: np.ndarray) -> np.ndarray:
    """U(x)=g(x)-g(0), g(x)=exp(-50(x-0.5)^2)."""
    g = np.exp(-50.0 * (x - 0.5) ** 2)
    g0 = np.exp(-50.0 * (0.0 - 0.5) ** 2)
    return g - g0


def k_ref_np(u: np.ndarray) -> np.ndarray:
    """Reference conductivity k(u)=0.02 exp(-20(u-0.5)^2)."""
    return 0.02 * np.exp(-20.0 * (u - 0.5) ** 2)


def k_ref_torch(u: torch.Tensor) -> torch.Tensor:
    return 0.02 * torch.exp(-20.0 * (u - 0.5) ** 2)


class KNet(nn.Module):
    """The same small architecture used in the paper/source: 1 x 5 x 5 x 1, tanh."""

    def __init__(self, kmax: float = 0.1):
        super().__init__()
        self.kmax = kmax
        self.net = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
        )
        # Close to common ML defaults; zero biases keep the initial k around kmax/2.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        q = self.net(u.reshape(-1, 1)).reshape_as(u)
        return self.kmax * torch.sigmoid(q)  # non-negative and <= kmax


def make_reference(args):
    """
    Generate a clean reference temperature field with a fast implicit finite-volume
    solver. The official source computes this reference by forward ODIL/Newton on
    a 256 x 256 grid; this standalone version uses backward-Euler Picard steps so
    it runs quickly on a laptop without the ODIL package.
    """
    from scipy.linalg import solve_banded

    nx = args.ref_Nx
    nt = args.ref_Nt
    x = (np.arange(nx) + 0.5) / nx
    dx = 1.0 / nx
    dt = 1.0 / nt
    u = init_u_np(x).astype(np.float64)
    out = np.empty((nt, nx), dtype=np.float32)

    print(f"Generating reference field with implicit finite volume: ref_Nt={nt}, ref_Nx={nx} ...")
    for n in range(nt):
        guess = u.copy()
        for _ in range(args.ref_picard):
            # Conductivity on faces. Boundary values are zero Dirichlet.
            k_face = np.empty(nx + 1, dtype=np.float64)
            k_face[0] = k_ref_np(0.5 * guess[0])
            k_face[-1] = k_ref_np(0.5 * guess[-1])
            k_face[1:-1] = k_ref_np(0.5 * (guess[:-1] + guess[1:]))

            lower_L = k_face[:-1] / dx**2
            upper_L = k_face[1:] / dx**2
            diag_L = -(lower_L + upper_L)
            # Boundary face is half a cell away: multiply boundary conductance by 2.
            diag_L[0] = -(2.0 * k_face[0] + k_face[1]) / dx**2
            diag_L[-1] = -(k_face[-2] + 2.0 * k_face[-1]) / dx**2

            # Matrix A = I - dt * L in scipy banded form.
            ab = np.zeros((3, nx), dtype=np.float64)
            ab[0, 1:] = -dt * upper_L[:-1]       # upper diagonal
            ab[1, :] = 1.0 - dt * diag_L         # main diagonal
            ab[2, :-1] = -dt * lower_L[1:]       # lower diagonal
            new_u = solve_banded((1, 1), ab, u)
            if np.linalg.norm(new_u - guess) / (np.linalg.norm(new_u) + 1e-12) < 1e-10:
                guess = new_u
                break
            guess = new_u
        u = guess
        out[n] = u.astype(np.float32)

    t = (np.arange(nt) + 0.5) / nt
    return t.astype(np.float32), x.astype(np.float32), out

def interpolate_reference_to_grid(t_ref, x_ref, ref_u, Nt, Nx):
    t = (np.arange(Nt) + 0.5) / Nt
    x = (np.arange(Nx) + 0.5) / Nx
    interp = RectBivariateSpline(t_ref, x_ref, ref_u, kx=3, ky=3)
    return t.astype(np.float32), x.astype(np.float32), interp(t, x).astype(np.float32)


def make_imposed_data(ref_u, t, x, args):
    """Official run script uses --imposed stripe and --nimp 200."""
    rng = np.random.default_rng(args.seed)
    Nt, Nx = ref_u.shape
    tt, xx = np.meshgrid(t, x, indexing="ij")

    if args.imposed == "stripe":
        candidate = np.flatnonzero(np.abs(tt.reshape(-1) - 0.5) < 1.0 / 6.0)
    elif args.imposed == "random":
        candidate = np.arange(Nt * Nx)
    else:
        raise ValueError("--imposed must be stripe or random")

    nimp = min(args.nimp, candidate.size)
    chosen = rng.choice(candidate, size=nimp, replace=False)
    mask = np.zeros(Nt * Nx, dtype=np.float32)
    mask[chosen] = 1.0
    mask = mask.reshape(Nt, Nx)

    imp_u = ref_u.copy()
    if args.noise > 0:
        imp_u = imp_u + rng.normal(0.0, args.noise, size=imp_u.shape).astype(np.float32)
    return mask.astype(np.float32), imp_u.astype(np.float32), chosen


@dataclass
class Grid:
    Nt: int
    Nx: int
    dt: float
    dx: float
    init_u: torch.Tensor


def pad_x_quad_zero(u: torch.Tensor) -> torch.Tensor:
    """Pad x-direction with the quadratic zero-Dirichlet halo used in source Eq. (30)."""
    left = (u[:, 1:2] - 6.0 * u[:, 0:1]) / 3.0
    right = (u[:, -2:-1] - 6.0 * u[:, -1:]) / 3.0
    return torch.cat([left, u, right], dim=1)


def previous_time_with_initial_halo(u: torch.Tensor, init_u: torch.Tensor) -> torch.Tensor:
    """
    Previous-time array. For the first time cell, use source's second-order
    initial-condition halo: u_ghost=(u_2 - 6u_1 + 8U)/3.
    """
    first_prev = (u[1:2, :] - 6.0 * u[0:1, :] + 8.0 * init_u[None, :]) / 3.0
    return torch.cat([first_prev, u[:-1, :]], dim=0)


def crank_nicolson_residual(u: torch.Tensor, grid: Grid, k_function) -> torch.Tensor:
    """
    Residual matching the ODIL source implementation:
        u_t - d_x( k(u^{n-1/2}_{face}) d_x u^{n-1/2} ) = 0
    using cell-centered Crank-Nicolson and zero-Dirichlet/initial halos.
    """
    up = pad_x_quad_zero(u)
    um = previous_time_with_initial_halo(u, grid.init_u)
    ump = pad_x_quad_zero(um)

    q_c = up[:, 1:-1]
    q_l = up[:, :-2]
    q_r = up[:, 2:]
    qm_c = ump[:, 1:-1]
    qm_l = ump[:, :-2]
    qm_r = ump[:, 2:]

    u_t = (q_c - qm_c) / grid.dt

    # spatial derivatives at left/right faces, at the time midpoint
    u_xm = ((q_c + qm_c) - (q_l + qm_l)) / (2.0 * grid.dx)
    u_xp = ((q_r + qm_r) - (q_c + qm_c)) / (2.0 * grid.dx)

    # conductivity is evaluated at face value and time midpoint
    u_face_m = ((q_c + qm_c) + (q_l + qm_l)) * 0.25
    u_face_p = ((q_r + qm_r) + (q_c + qm_c)) * 0.25
    km = k_function(u_face_m)
    kp = k_function(u_face_p)

    flux_div = (u_xp * kp - u_xm * km) / grid.dx
    return u_t - flux_div


def train_inverse(args):
    torch.set_num_threads(args.torch_threads)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print("Using device:", device)

    t_ref, x_ref, ref_u_fine = make_reference(args)
    t, x, ref_u = interpolate_reference_to_grid(t_ref, x_ref, ref_u_fine, args.Nt, args.Nx)
    mask_np, imp_u_np, imp_indices = make_imposed_data(ref_u, t, x, args)

    dtype = torch.float32
    init_u = torch.tensor(init_u_np(x), dtype=dtype, device=device)
    grid = Grid(args.Nt, args.Nx, 1.0 / args.Nt, 1.0 / args.Nx, init_u)

    ref_u_t = torch.tensor(ref_u, dtype=dtype, device=device)
    imp_u = torch.tensor(imp_u_np, dtype=dtype, device=device)
    mask = torch.tensor(mask_np, dtype=dtype, device=device)

    # The official source initializes the ODIL temperature grid with zeros.
    # For a standalone Mac script without the official multigrid machinery,
    # warm initialization is the practical default. Use --zero_init to mimic source.
    if args.warm_init:
        u_init_grid = np.repeat(init_u_np(x)[None, :], args.Nt, axis=0).astype(np.float32)
        u_param = nn.Parameter(torch.tensor(u_init_grid, dtype=dtype, device=device))
    else:
        u_param = nn.Parameter(torch.zeros((args.Nt, args.Nx), dtype=dtype, device=device))

    k_net = KNet(kmax=args.kmax).to(device)
    params = [u_param] + list(k_net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    ncell = args.Nt * args.Nx
    nimp = float(mask_np.sum())
    data_scale = args.kimp * np.sqrt(ncell / max(nimp, 1.0))

    us = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    us_t = torch.tensor(us, dtype=dtype, device=device)
    ref_k_curve = k_ref_np(us)

    hist_epoch, hist_loss, hist_pde, hist_data, hist_uerr, hist_kerr = [], [], [], [], [], []

    start = time.time()
    print(f"Training inverse ODIL: Nt={args.Nt}, Nx={args.Nx}, nimp={int(nimp)}, epochs={args.epochs}")
    for ep in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        res = crank_nicolson_residual(u_param, grid, k_net)
        pde_loss = torch.mean(res ** 2)
        imp_res = mask * (u_param - imp_u) * data_scale
        data_loss = torch.mean(imp_res ** 2)
        loss = pde_loss + data_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        if ep == 1 or ep % args.report_every == 0 or ep == args.epochs:
            with torch.no_grad():
                u_err = torch.sqrt(torch.mean((u_param - ref_u_t) ** 2)).item()
                k_pred = k_net(us_t).detach().cpu().numpy()
                k_err = np.sqrt(np.mean((k_pred - ref_k_curve) ** 2)) / ref_k_curve.max()
                hist_epoch.append(ep)
                hist_loss.append(loss.item())
                hist_pde.append(pde_loss.item())
                hist_data.append(data_loss.item())
                hist_uerr.append(u_err)
                hist_kerr.append(k_err)
            print(
                f"epoch {ep:6d} | loss={loss.item():.3e} "
                f"pde={pde_loss.item():.3e} data={data_loss.item():.3e} "
                f"u_err={u_err:.3e} k_rel_err={k_err:.3e}"
            )

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} s")

    with torch.no_grad():
        u_pred = u_param.detach().cpu().numpy()
        k_pred = k_net(us_t).detach().cpu().numpy()

    # Save model and arrays.
    torch.save({"u": u_pred, "knet": k_net.state_dict(), "args": vars(args)}, os.path.join(args.outdir, "model.pt"))
    np.savez(
        os.path.join(args.outdir, "data.npz"),
        t=t,
        x=x,
        ref_u=ref_u,
        u_pred=u_pred,
        imp_u=imp_u_np,
        mask=mask_np,
        us=us,
        k_pred=k_pred,
        k_ref=ref_k_curve,
        hist_epoch=np.array(hist_epoch),
        hist_loss=np.array(hist_loss),
        hist_pde=np.array(hist_pde),
        hist_data=np.array(hist_data),
        hist_uerr=np.array(hist_uerr),
        hist_kerr=np.array(hist_kerr),
    )

    plot_results(args.outdir, t, x, ref_u, u_pred, mask_np, us, ref_k_curve, k_pred)
    plot_history(args.outdir, hist_epoch, hist_loss, hist_pde, hist_data, hist_uerr, hist_kerr)


def plot_results(outdir, t, x, ref_u, u_pred, mask, us, k_ref, k_pred):
    extent = [x[0], x[-1], t[0], t[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)

    im0 = axes[0].imshow(ref_u, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="YlOrBr")
    axes[0].set_title("reference u")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = axes[1].imshow(u_pred, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="YlOrBr")
    axes[1].set_title("inferred u (ODIL)")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    im2 = axes[2].imshow(u_pred - ref_u, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    yy, xx = np.where(mask > 0)
    axes[2].scatter(x[xx], t[yy], s=3, c="k", alpha=0.8, linewidths=0)
    axes[2].set_title("error + noisy data points")
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    axes[3].plot(us, k_ref, label="reference")
    axes[3].plot(us, k_pred, label="inferred")
    axes[3].set_xlabel("u")
    axes[3].set_ylabel("k(u)")
    axes[3].set_ylim(0, 0.03)
    axes[3].set_title("conductivity")
    axes[3].legend(frameon=False)

    fig.savefig(os.path.join(outdir, "heat_results.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "fig5_noisy_temperature_conductivity.png"), dpi=200)
    plt.close(fig)


def plot_history(outdir, epoch, loss, pde, data, uerr, kerr):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].semilogy(epoch, loss, label="total")
    axes[0].semilogy(epoch, pde, label="PDE")
    axes[0].semilogy(epoch, data, label="data")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("loss")
    axes[0].legend(frameon=False)

    axes[1].semilogy(epoch, uerr)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("temperature RMSE")

    axes[2].semilogy(epoch, kerr)
    axes[2].set_xlabel("epoch")
    axes[2].set_title("conductivity relative RMSE")

    fig.savefig(os.path.join(outdir, "history.png"), dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--Nt", type=int, default=64, help="ODIL grid size in t")
    p.add_argument("--Nx", type=int, default=64, help="ODIL grid size in x")
    p.add_argument("--ref_Nt", type=int, default=256, help="reference grid size in t")
    p.add_argument("--ref_Nx", type=int, default=256, help="reference grid size in x")
    p.add_argument("--ref_picard", type=int, default=6, help="Picard iterations per implicit reference step")
    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kimp", type=float, default=2.0, help="weight of imposed temperature points")
    p.add_argument("--nimp", type=int, default=200, help="number of temperature measurements")
    p.add_argument("--imposed", choices=["stripe", "random"], default="stripe")
    p.add_argument("--noise", type=float, default=0.1, help="Gaussian noise added to data; Fig. 5 uses sigma=0.1")
    p.add_argument("--kmax", type=float, default=0.1, help="k = kmax * sigmoid(q)")
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--zero_init", dest="warm_init", action="store_false", help="mimic the official source zero initialization for u")
    p.set_defaults(warm_init=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--report_every", type=int, default=200)
    p.add_argument("--torch_threads", type=int, default=1, help="PyTorch CPU threads; 1 is often faster for this small problem")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--outdir", type=str, default="out_heat_fig5_noise")
    return p.parse_args()


if __name__ == "__main__":
    train_inverse(parse_args())
