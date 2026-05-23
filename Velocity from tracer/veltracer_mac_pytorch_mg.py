#!/usr/bin/env python3
"""
Standalone PyTorch version of ODIL's examples/velocity_from_tracer/veltracer.py.

This version adds a multigrid / multiresolution decomposition of the unknown
fields, following the idea used by ODIL/mODIL:

    q = q_1 + T_1 q_2 + T_1 T_2 q_3 + ...

where q is one of the grid fields: tracer u, velocity vx, velocity vy.
It does not require installing the ODIL package; it only needs PyTorch,
NumPy, and Matplotlib.

Run on Mac:
    python3 -m venv vel_env
    source vel_env/bin/activate
    pip install numpy matplotlib torch
    python veltracer_mac_pytorch_mg.py --Nx 64 --Nt 64 --epochs 5000

Faster test:
    python veltracer_mac_pytorch_mg.py --Nx 32 --Nt 32 --epochs 1500
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def u_init_blob_np(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Same synthetic tracer profile as ODIL's velocity_from_tracer example."""
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


def make_grid(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centered grid on [0, 1]^2."""
    x1 = (np.arange(nx, dtype=np.float32) + 0.5) / nx
    y1 = (np.arange(ny, dtype=np.float32) + 0.5) / ny
    x, y = np.meshgrid(x1, y1, indexing="ij")
    return x, y


def choose_mg_shapes(nt: int, nx: int, ny: int, nlvl: int | None) -> List[Tuple[int, int, int]]:
    """
    Return multigrid cshapes [(Nt, Nx, Ny), ...].
    Field shape is (Nt+1, Nx, Ny) because time is node-centered.
    """
    shapes: List[Tuple[int, int, int]] = []
    cur = (nt, nx, ny)
    if nlvl is None or nlvl <= 0:
        # Similar spirit to ODIL's automatic multigrid levels: keep halving
        # until the grid is too small. Limit to avoid too many tiny parameters.
        max_levels = 6
    else:
        max_levels = nlvl

    for _ in range(max_levels):
        shapes.append(cur)
        if nlvl is not None and nlvl > 0 and len(shapes) >= nlvl:
            break
        if min(cur) <= 4:
            break
        nxt = tuple(max(2, int(round(v / 2))) for v in cur)
        if nxt == cur:
            break
        cur = nxt
    return shapes


class MGField(nn.Module):
    """
    Multigrid decomposition for a 3D field q(t, x, y):

        q = q_level0 + interp(q_level1) + interp(q_level2) + ...

    q_level0 lives on the finest grid. Coarser parameters are interpolated
    directly to the fine field shape when the field is evaluated.
    """

    def __init__(
        self,
        name: str,
        cshapes: List[Tuple[int, int, int]],
        init_fine: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        use_multigrid: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.cshapes = cshapes if use_multigrid else [cshapes[0]]
        self.fine_shape = (cshapes[0][0] + 1, cshapes[0][1], cshapes[0][2])
        self.levels = nn.ParameterList()

        for level, (nt, nx, ny) in enumerate(self.cshapes):
            fshape = (nt + 1, nx, ny)
            if level == 0:
                value = init_fine.detach().to(device=device, dtype=dtype).clone()
                if tuple(value.shape) != fshape:
                    raise ValueError(f"init_fine shape {tuple(value.shape)} does not match {fshape}")
            else:
                value = torch.zeros(fshape, dtype=dtype, device=device)
            self.levels.append(nn.Parameter(value))

    def forward(self) -> torch.Tensor:
        out = self.levels[0]
        target = self.fine_shape
        for coarse in self.levels[1:]:
            # PyTorch interpolate expects N,C,D,H,W. Here D=t, H=x, W=y.
            z = coarse[None, None, ...]
            z = F.interpolate(z, size=target, mode="trilinear", align_corners=False)
            out = out + z[0, 0]
        return out


def roll_x(q: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(q, shifts=shift, dims=-2)


def roll_y(q: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(q, shifts=shift, dims=-1)


def upwind_diff_x(q: torch.Tensor, vx_for_sign: torch.Tensor) -> torch.Tensor:
    """First-order upwind difference in x, without division by dx."""
    qm = roll_x(q, 1)
    qp = roll_x(q, -1)
    return torch.where(vx_for_sign > 0, q - qm, torch.where(vx_for_sign < 0, qp - q, 0.5 * (qp - qm)))


def upwind_diff_y(q: torch.Tensor, vy_for_sign: torch.Tensor) -> torch.Tensor:
    """First-order upwind difference in y, without division by dy."""
    qm = roll_y(q, 1)
    qp = roll_y(q, -1)
    return torch.where(vy_for_sign > 0, q - qm, torch.where(vy_for_sign < 0, qp - q, 0.5 * (qp - qm)))


def laplace_xy(q: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    q_xx = (roll_x(q, -1) - 2.0 * q + roll_x(q, 1)) / (dx * dx)
    q_yy = (roll_y(q, -1) - 2.0 * q + roll_y(q, 1)) / (dy * dy)
    return q_xx + q_yy


@dataclass
class LossParts:
    total: torch.Tensor
    fu: torch.Tensor
    fimp: torch.Tensor
    xreg: torch.Tensor
    treg: torch.Tensor


def compute_loss(
    u: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    u_init: torch.Tensor,
    u_final: torch.Tensor,
    dt: float,
    dx: float,
    dy: float,
    kimp: float,
    kxreg: float,
    ktreg: float,
) -> LossParts:
    """
    Close to ODIL operator_advection:
      fu   = u_t + vx * D_upwind_x(u_old) + vy * D_upwind_y(u_old)
      fu[0] = (u[0] - u_init) / dx
      fimp[-1] = (u[-1] - u_final) / dx
      regularization: kxreg * Laplacian(v), ktreg * v_t
    """
    nt_plus_1 = u.shape[0]

    # Previous-time tracer used in the advection residual at t_i, i>=1.
    # For i=1 the official code replaces u^{0} by the imposed initial profile.
    prev = torch.empty_like(u[1:])
    prev[0] = u_init
    if nt_plus_1 > 2:
        prev[1:] = u[1:-1]

    # Velocity sign is detached in the upwind switch, similar to ODIL's frozen=True.
    ux_raw = upwind_diff_x(prev, vx[1:].detach())
    uy_raw = upwind_diff_y(prev, vy[1:].detach())

    fu = torch.empty_like(u)
    fu[0] = (u[0] - u_init) / dx
    fu[1:] = (u[1:] - prev) / dt + vx[1:] * ux_raw / dx + vy[1:] * uy_raw / dy
    loss_fu = torch.mean(fu**2)

    fimp = torch.zeros_like(u)
    fimp[-1] = (u[-1] - u_final) / dx
    loss_fimp = torch.mean((kimp * fimp) ** 2)

    if kxreg:
        loss_xreg = torch.mean((kxreg * laplace_xy(vx, dx, dy)) ** 2) + torch.mean(
            (kxreg * laplace_xy(vy, dx, dy)) ** 2
        )
    else:
        loss_xreg = torch.zeros((), dtype=u.dtype, device=u.device)

    if ktreg:
        zeros = torch.zeros_like(vx[:1])
        vx_t = torch.cat([zeros, (vx[1:] - vx[:-1]) * (ktreg / dt)], dim=0)
        vy_t = torch.cat([zeros, (vy[1:] - vy[:-1]) * (ktreg / dt)], dim=0)
        loss_treg = torch.mean(vx_t**2) + torch.mean(vy_t**2)
    else:
        loss_treg = torch.zeros((), dtype=u.dtype, device=u.device)

    total = loss_fu + loss_fimp + loss_xreg + loss_treg
    return LossParts(total=total, fu=loss_fu, fimp=loss_fimp, xreg=loss_xreg, treg=loss_treg)


def make_initial_u(args, c0_np: np.ndarray, c1_np: np.ndarray) -> np.ndarray:
    nt = args.Nt
    nx, ny = args.Nx, args.Ny
    if args.init == "zero":
        return np.zeros((nt + 1, nx, ny), dtype=np.float32)
    if args.init == "linear":
        out = np.zeros((nt + 1, nx, ny), dtype=np.float32)
        for n in range(nt + 1):
            a = n / nt
            out[n] = (1.0 - a) * c0_np + a * c1_np
        return out
    raise ValueError(f"unknown init mode {args.init!r}")


def plot_results(outdir: str, u: np.ndarray, vx: np.ndarray, vy: np.ndarray, c0: np.ndarray, c1: np.ndarray, hist: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    ntp1, nx, ny = u.shape
    times = np.linspace(0, ntp1 - 1, 5, dtype=int)

    x, y = make_grid(nx, ny)
    skip = max(1, nx // 16)
    offset = max(0, skip // 2 - 1)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), constrained_layout=True)
    for ax, it in zip(axes, times):
        ax.imshow(u[it].T, origin="lower", extent=[0, 1, 0, 1], cmap="YlOrBr", vmin=0, vmax=1)
        xs = x[offset::skip, offset::skip]
        ys = y[offset::skip, offset::skip]
        vxs = vx[it, offset::skip, offset::skip]
        vys = vy[it, offset::skip, offset::skip]
        ax.quiver(xs, ys, vxs, vys, color="k", scale=5)
        ax.set_title(f"t={it/(ntp1-1):.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Tracer and inferred velocity")
    fig.savefig(os.path.join(outdir, "tracer_velocity.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    titles = ["given c0", "inferred u(t=0)", "given c1", "inferred u(t=1)"]
    imgs = [c0, u[0], c1, u[-1]]
    for ax, img, title in zip(axes.ravel(), imgs, titles):
        ax.imshow(img.T, origin="lower", extent=[0, 1, 0, 1], cmap="YlOrBr", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(outdir, "initial_final_check.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), constrained_layout=True)
    for ax, it in zip(axes[0], times):
        ax.imshow(vx[it].T, origin="lower", extent=[0, 1, 0, 1], cmap="PuOr_r", vmin=-0.5, vmax=0.5)
        ax.set_title(f"vx, t={it/(ntp1-1):.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, it in zip(axes[1], times):
        ax.imshow(vy[it].T, origin="lower", extent=[0, 1, 0, 1], cmap="PuOr_r", vmin=-0.5, vmax=0.5)
        ax.set_title(f"vy, t={it/(ntp1-1):.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(outdir, "velocity_components.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key, vals in hist.items():
        if vals:
            ax.semilogy(vals, label=key)
    ax.set_xlabel("history step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(outdir, "loss_history.png"), dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--Nt", type=int, default=None, help="Grid size in t, number of time intervals")
    p.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    p.add_argument("--Ny", type=int, default=None, help="Grid size in y")
    p.add_argument("--kxreg", type=float, default=0.01, help="Laplacian regularization weight")
    p.add_argument("--ktreg", type=float, default=1.0, help="Time regularization weight")
    p.add_argument("--kimp", type=float, default=10.0, help="Imposed final tracer weight")
    p.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    p.add_argument("--epochs", type=int, default=3000, help="Number of Adam epochs")
    p.add_argument("--report_every", type=int, default=100, help="Print loss every N epochs")
    p.add_argument("--history_every", type=int, default=10, help="Save loss history every N epochs")
    p.add_argument("--outdir", type=str, default="out_veltracer_mg_pytorch", help="Output directory")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Torch device")
    p.add_argument("--double", action="store_true", help="Use float64 instead of float32")

    # ODIL-like arguments.
    p.add_argument("--multigrid", type=int, default=1, help="Use multigrid field decomposition, 1=yes, 0=no")
    p.add_argument("--nlvl", type=int, default=0, help="Number of multigrid levels; 0=auto")
    p.add_argument("--mg_interp", type=str, default="conv", help="Kept for ODIL naming compatibility; this script uses trilinear interpolation")
    p.add_argument("--init", type=str, default="zero", choices=["zero", "linear"], help="Initial guess for tracer field")
    p.add_argument("--clamp_u", action="store_true", help="Project tracer to [0,1] after every step; off by default to match ODIL more closely")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    args.Nt = args.Nt or args.Nx
    args.Ny = args.Ny or args.Nx

    device = pick_device(args.device)
    dtype = torch.float64 if args.double else torch.float32
    os.makedirs(args.outdir, exist_ok=True)

    nt, nx, ny = args.Nt, args.Nx, args.Ny
    dt, dx, dy = 1.0 / nt, 1.0 / nx, 1.0 / ny

    x_np, y_np = make_grid(nx, ny)
    c0_np = u_init_blob_np(x_np, y_np, 0.0)
    c1_np = u_init_blob_np(x_np, y_np, 1.0)
    u0_np = make_initial_u(args, c0_np, c1_np)

    c0 = torch.tensor(c0_np, dtype=dtype, device=device)
    c1 = torch.tensor(c1_np, dtype=dtype, device=device)
    u0 = torch.tensor(u0_np, dtype=dtype, device=device)
    v0 = torch.zeros((nt + 1, nx, ny), dtype=dtype, device=device)

    nlvl = None if args.nlvl <= 0 else args.nlvl
    cshapes = choose_mg_shapes(nt, nx, ny, nlvl)
    use_mg = bool(args.multigrid)

    print(f"device: {device}")
    print(f"fine cshape: Nt={nt}, Nx={nx}, Ny={ny}")
    print(f"multigrid: {int(use_mg)}, mg_interp={args.mg_interp}, levels: {cshapes if use_mg else [cshapes[0]]}")

    u_field = MGField("u", cshapes, u0, dtype, device, use_multigrid=use_mg).to(device)
    vx_field = MGField("vx", cshapes, v0, dtype, device, use_multigrid=use_mg).to(device)
    vy_field = MGField("vy", cshapes, v0, dtype, device, use_multigrid=use_mg).to(device)

    params = list(u_field.parameters()) + list(vx_field.parameters()) + list(vy_field.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    hist = {"total": [], "fu": [], "fimp": [], "xreg": [], "treg": []}

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        u = u_field()
        vx = vx_field()
        vy = vy_field()
        parts = compute_loss(u, vx, vy, c0, c1, dt, dx, dy, args.kimp, args.kxreg, args.ktreg)
        parts.total.backward()
        optimizer.step()

        if args.clamp_u:
            with torch.no_grad():
                # Clamp only the finest tracer level. This is optional and not source-like.
                u_field.levels[0].clamp_(0.0, 1.0)

        if epoch % args.history_every == 0:
            hist["total"].append(float(parts.total.detach().cpu()))
            hist["fu"].append(float(parts.fu.detach().cpu()))
            hist["fimp"].append(float(parts.fimp.detach().cpu()))
            hist["xreg"].append(float(parts.xreg.detach().cpu()))
            hist["treg"].append(float(parts.treg.detach().cpu()))

        if epoch % args.report_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:6d} "
                f"loss={float(parts.total.detach().cpu()):.6e} "
                f"fu={float(parts.fu.detach().cpu()):.3e} "
                f"fimp={float(parts.fimp.detach().cpu()):.3e} "
                f"xreg={float(parts.xreg.detach().cpu()):.3e} "
                f"treg={float(parts.treg.detach().cpu()):.3e}"
            )

    with torch.no_grad():
        u = u_field().detach().cpu().numpy()
        vx = vx_field().detach().cpu().numpy()
        vy = vy_field().detach().cpu().numpy()

    np.savez(os.path.join(args.outdir, "result.npz"), u=u, vx=vx, vy=vy, c0=c0_np, c1=c1_np, cshapes=np.array(cshapes, dtype=object))
    plot_results(args.outdir, u, vx, vy, c0_np, c1_np, hist)
    print(f"saved results to: {args.outdir}")


if __name__ == "__main__":
    main()
