#!/usr/bin/env python3
"""
Mac-runnable reproduction of the visual content of pgae005.pdf Fig. 10.

What it reproduces
------------------
Fig. 10 in Karnakov, Litvinov & Koumoutsakos (PNAS Nexus, 2024) shows
inference of an elliptical obstacle from velocity measurements. The paper's
full experiment solves the penalized steady Navier--Stokes equations using
ODIL on a 2N x N grid, with N=64, a level-set body fraction chi, 100 velocity
measurements, Adam lr=1e-3, and 20,000 inverse iterations.

This script is a compact, self-contained reproduction intended for a Mac laptop:
- grid-based fields on [0,2] x [0,1]
- level-set phi and body fraction chi
- 100 velocity measurements away from the body
- optimization of a discrete loss with PyTorch autograd
- output with the same panels as Fig. 10: vorticity/contours, velocity error,
  and body-fraction error.

Important difference
--------------------
The public cselab/odil repository currently lists pgae005 examples for Poisson,
tracer velocity, and heat conductivity, but not the 2-D body-shape case. To keep
this file short and robust on macOS, the forward flow here is a differentiable
ODIL-style surrogate streamfunction, not the full finite-volume Navier--Stokes
stencil with Rhie--Chow/deferred correction used in the paper. The structure of
unknowns, observations, level set, losses, and figure panels follows Fig. 10.

Run
---
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
python reproduce_pgae005_fig10.py

For a larger figure closer to the paper grid:
python reproduce_pgae005_fig10.py --nx 128 --ny 64 --epochs 3000  # slower
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class EllipseParams:
    cx: float = 0.68
    cy: float = 0.50
    a: float = 0.20
    b: float = 0.085


def make_grid(nx: int, ny: int, device: torch.device):
    """Cell-centered grid on [0,2] x [0,1]. Arrays have shape (ny, nx)."""
    x = (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * (2.0 / nx)
    y = (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * (1.0 / ny)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    dx = 2.0 / nx
    dy = 1.0 / ny
    return X, Y, dx, dy


def unpack_raw_params(raw: torch.Tensor):
    """Constrain ellipse parameters to physically reasonable ranges."""
    cx = 0.35 + 0.70 * torch.sigmoid(raw[0])  # [0.35, 1.05]
    cy = 0.30 + 0.40 * torch.sigmoid(raw[1])  # [0.30, 0.70]
    a = 0.08 + 0.22 * torch.sigmoid(raw[2])   # [0.08, 0.30]
    b = 0.035 + 0.13 * torch.sigmoid(raw[3])  # [0.035, 0.165]
    return cx, cy, a, b


def raw_from_params(p: EllipseParams) -> torch.Tensor:
    def logit(z: float) -> float:
        z = min(max(z, 1e-6), 1 - 1e-6)
        return math.log(z / (1 - z))

    return torch.tensor([
        logit((p.cx - 0.35) / 0.70),
        logit((p.cy - 0.30) / 0.40),
        logit((p.a - 0.08) / 0.22),
        logit((p.b - 0.035) / 0.13),
    ], dtype=torch.float32)


def ellipse_level_set(X, Y, cx, cy, a, b):
    """Approximate signed distance for an ellipse: positive inside, negative outside."""
    r = torch.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 + 1e-12)
    return (1.0 - r) * torch.minimum(a, b)


def body_fraction(phi: torch.Tensor, dx: float, smooth: bool = True):
    """
    Paper-style body fraction: chi = clip(0.5 + phi/(4 dx), 0, 1).
    The smooth sigmoid version keeps gradients nonzero during optimization.
    """
    if smooth:
        return torch.sigmoid(phi / (2.0 * dx))
    return torch.clamp(0.5 + phi / (4.0 * dx), 0.0, 1.0)


def ddx(q: torch.Tensor, dx: float):
    out = torch.zeros_like(q)
    out[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2.0 * dx)
    out[:, 0] = (q[:, 1] - q[:, 0]) / dx
    out[:, -1] = (q[:, -1] - q[:, -2]) / dx
    return out


def ddy(q: torch.Tensor, dy: float):
    out = torch.zeros_like(q)
    out[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2.0 * dy)
    out[0, :] = (q[1, :] - q[0, :]) / dy
    out[-1, :] = (q[-1, :] - q[-2, :]) / dy
    return out


def flow_from_params(X, Y, dx, dy, raw: torch.Tensor, smooth_chi: bool = True):
    """
    Differentiable surrogate of a steady flow around an ellipse.

    The streamfunction gives a divergence-free external field; multiplying by
    (1-chi) mimics the Brinkman/penalization no-slip body region and creates
    vorticity layers around the obstacle, which is the key visual content of Fig. 10.
    """
    cx, cy, a, b = unpack_raw_params(raw)
    phi = ellipse_level_set(X, Y, cx, cy, a, b)
    chi = body_fraction(phi, dx, smooth=smooth_chi)

    # Elliptic coordinates scaled by the current body.
    rx = (X - cx) / (a + 1e-12)
    ry = (Y - cy) / (b + 1e-12)
    core = torch.exp(-0.65 * (rx ** 2 + ry ** 2))

    # Smooth downstream wake beginning behind the body.
    wake_on = torch.sigmoid((X - (cx + 0.75 * a)) / 0.025)
    wake_decay = torch.exp(-torch.clamp(X - (cx + a), min=0.0) / 0.55)
    wake = wake_on * wake_decay * torch.exp(-0.65 * ((Y - cy) / (2.0 * b + 1e-12)) ** 2)

    # Streamfunction: uniform flow + obstacle deflection + weak wake asymmetry.
    # u = d psi / dy, v = - d psi / dx.
    psi = Y
    psi = psi + 0.33 * a * core * (Y - cy) / (b + 1e-12)
    psi = psi + 0.11 * wake * torch.sin(math.pi * (Y - cy) / 0.55)

    u = ddy(psi, dy)
    v = -ddx(psi, dx)

    # Penalized/no-slip body interior. This is not a projection; it is a compact
    # visual surrogate of the paper's lambda*chi*u penalization term.
    u = (1.0 - chi) * u
    v = (1.0 - chi) * v

    vort = ddx(v, dx) - ddy(u, dy)
    return u, v, vort, phi, chi


def choose_measurements(X, Y, phi_ref, n: int, seed: int):
    """Choose points outside but near the body, as in Fig. 10."""
    rng = np.random.default_rng(seed)
    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    phin = phi_ref.detach().cpu().numpy()

    # Ring around obstacle, plus a downstream wake region. These are all outside the body.
    near = (phin < -0.025) & (phin > -0.18)
    wake = (Xn > 0.75) & (Xn < 1.55) & (np.abs(Yn - 0.50) < 0.30) & (phin < -0.02)
    bounds = (Xn > 0.18) & (Xn < 1.70) & (Yn > 0.08) & (Yn < 0.92)
    mask = (near | wake) & bounds
    candidates = np.argwhere(mask)
    if len(candidates) < n:
        candidates = np.argwhere(bounds & (phin < -0.02))
    ids = rng.choice(len(candidates), size=n, replace=False)
    pts = candidates[ids]
    iy = torch.tensor(pts[:, 0], dtype=torch.long, device=X.device)
    ix = torch.tensor(pts[:, 1], dtype=torch.long, device=X.device)
    return iy, ix


def eikonal_loss(phi: torch.Tensor, dx: float, dy: float):
    return ((ddx(phi, dx) ** 2 + ddy(phi, dy) ** 2 - 1.0) ** 2).mean()


def relative_velocity_error(u, v, uref, vref):
    num = ((u - uref) ** 2 + (v - vref) ** 2).mean()
    den = (uref ** 2 + vref ** 2).mean() + 1e-14
    return torch.sqrt(num / den)


def relative_chi_error(chi, chi_ref):
    return torch.sqrt(((chi - chi_ref) ** 2).mean()) / (chi_ref.mean() + 1e-14)


def optimize_inverse(args):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    X, Y, dx, dy = make_grid(args.nx, args.ny, device)

    ref_raw = raw_from_params(EllipseParams()).to(device)
    with torch.no_grad():
        uref, vref, wref, phiref, chiref = flow_from_params(X, Y, dx, dy, ref_raw, smooth_chi=True)

    iy, ix = choose_measurements(X, Y, phiref, args.nobs, args.seed)

    # Initial guess: small nearly circular body, deliberately different from the reference ellipse.
    init = EllipseParams(cx=0.54, cy=0.50, a=0.085, b=0.070)
    raw = torch.nn.Parameter(raw_from_params(init).to(device))
    opt = torch.optim.Adam([raw], lr=args.lr)

    epochs = []
    vel_hist = []
    chi_hist = []
    loss_hist = []

    for ep in range(args.epochs + 1):
        u, v, w, phi, chi = flow_from_params(X, Y, dx, dy, raw, smooth_chi=True)
        data_loss = ((u[iy, ix] - uref[iy, ix]) ** 2 + (v[iy, ix] - vref[iy, ix]) ** 2).mean()
        # Weak discrete regularization keeps the level-set close to signed-distance behavior.
        loss = data_loss + args.eikonal_weight * eikonal_loss(phi, dx, dy)

        if ep % args.log_every == 0 or ep == args.epochs:
            with torch.no_grad():
                ve = relative_velocity_error(u, v, uref, vref)
                ce = relative_chi_error(chi, chiref)
                epochs.append(ep)
                vel_hist.append(float(ve))
                chi_hist.append(float(ce))
                loss_hist.append(float(torch.sqrt(loss)))
                cx, cy, a, b = unpack_raw_params(raw)
                print(
                    f"epoch {ep:6d} | sqrt(loss)={math.sqrt(float(loss)):.3e} "
                    f"| vel_err={float(ve):.3e} | chi_err={float(ce):.3e} "
                    f"| cx={float(cx):.4f}, cy={float(cy):.4f}, a={float(a):.4f}, b={float(b):.4f}"
                )

        if ep == args.epochs:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        u, v, w, phi, chi = flow_from_params(X, Y, dx, dy, raw, smooth_chi=True)

    return {
        "X": X.cpu(), "Y": Y.cpu(),
        "dx": dx, "dy": dy,
        "u": u.cpu(), "v": v.cpu(), "w": w.cpu(), "phi": phi.cpu(), "chi": chi.cpu(),
        "uref": uref.cpu(), "vref": vref.cpu(), "wref": wref.cpu(), "phiref": phiref.cpu(), "chiref": chiref.cpu(),
        "iy": iy.cpu(), "ix": ix.cpu(),
        "epochs": np.array(epochs), "vel_hist": np.array(vel_hist),
        "chi_hist": np.array(chi_hist), "loss_hist": np.array(loss_hist),
        "raw": raw.detach().cpu(),
    }


def plot_figure10(result, out: Path):
    X = result["X"].numpy()
    Y = result["Y"].numpy()
    ix = result["ix"].numpy()
    iy = result["iy"].numpy()

    w = result["w"].numpy()
    wref = result["wref"].numpy()
    chi = result["chi"].numpy()
    chiref = result["chiref"].numpy()

    fig = plt.figure(figsize=(10.8, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    extent = [0, 2, 0, 1]
    vmax = max(8.0, np.percentile(np.abs(np.r_[w.ravel(), wref.ravel()]), 98))
    im = ax0.imshow(w, origin="lower", extent=extent, cmap="magma", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    # Use two contours, matching paper: inferred colored, reference black.
    ax0.contour(X, Y, chi, levels=[0.5], colors=["tab:orange"], linewidths=2.0)
    ax0.contour(X, Y, chiref, levels=[0.5], colors=["black"], linewidths=1.6)
    ax0.scatter(X[iy, ix], Y[iy, ix], c="k", s=6, alpha=0.8, linewidths=0)
    ax0.set_title("A  vorticity $\\omega$ with $\\chi=0.5$ contours")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("vorticity $\\omega$")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.semilogy(result["epochs"], result["vel_hist"], lw=2)
    ax1.set_title("B")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("velocity error")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogy(result["epochs"], result["chi_hist"], lw=2)
    ax2.set_title("C")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("body fraction error")
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle("Reproduction of pgae005 Fig. 10: ellipse from velocity measurements", y=1.04)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved {out}")


def plot_supplement_like(result, out: Path):
    """Optional Fig. S14-like fields: u, v, chi, phi, inferred/reference side by side."""
    X = result["X"].numpy(); Y = result["Y"].numpy()
    fields = [
        (result["u"].numpy(), result["uref"].numpy(), "velocity $u$"),
        (result["v"].numpy(), result["vref"].numpy(), "velocity $v$"),
        (result["chi"].numpy(), result["chiref"].numpy(), "body fraction $\\chi$"),
        (result["phi"].numpy(), result["phiref"].numpy(), "level-set $\\phi$"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0), constrained_layout=True)
    for r, (a, b, title) in enumerate(fields):
        vals = np.r_[a.ravel(), b.ravel()]
        vmin, vmax = np.percentile(vals, [2, 98])
        for c, data in enumerate([a, b]):
            ax = axes[r, c]
            im = ax.imshow(data, origin="lower", extent=[0,2,0,1], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.contour(X, Y, result["chi" if c == 0 else "chiref"].numpy(), levels=[0.5], colors="w", linewidths=0.8)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(("inferred " if c == 0 else "reference ") + title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=128, help="number of grid cells in x; paper uses 2N=128; 64 is Mac-fast")
    p.add_argument("--ny", type=int, default=64, help="number of grid cells in y; paper uses N=64; 32 is Mac-fast")
    p.add_argument("--epochs", type=int, default=20000, help="optimization epochs; paper inverse uses 20000")
    p.add_argument("--lr", type=float, default=2e-2, help="Adam learning rate for compact surrogate")
    p.add_argument("--nobs", type=int, default=100, help="number of velocity measurement points")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eikonal-weight", type=float, default=0.0)
    p.add_argument("--out", type=str, default="fig10_reproduction.png")
    p.add_argument("--supp", action="store_true", help="also save Fig. S14-like field visualization")
    return p.parse_args()


def main():
    args = parse_args()
    result = optimize_inverse(args)
    out = Path(args.out)
    plot_figure10(result, out)
    if args.supp:
        plot_supplement_like(result, out.with_name(out.stem + "_S14_like.png"))


if __name__ == "__main__":
    main()
