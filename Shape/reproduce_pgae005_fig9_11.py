#!/usr/bin/env python3
"""
Mac-runnable reproduction of pgae005.pdf Fig. 9--11.

This script follows the two-dimensional "body shape from velocity" setup in
Karnakov, Litvinov, and Koumoutsakos, PNAS Nexus 2024:

  * domain [0, 2] x [0, 1]
  * paper grid 2N x N with N = 64
  * 100 velocity measurements
  * body fraction chi = clip(0.5 + phi / (4 dx), 0, 1)
  * level-set style body description
  * Adam optimizer and Fig. 9--11 panels A/B/C

The public cselab/odil repository currently ships pgae005 examples for Poisson,
tracer velocity, and heat conductivity, but not this body-shape case. To keep
the reproduction self-contained and reliable on a Mac laptop, the flow model is
a compact differentiable Brinkman/streamfunction surrogate instead of the full
finite-volume Navier--Stokes residual used in the paper. The plotted quantities,
measurement layout, level-set-to-chi map, case order, and error histories are
kept close to the paper.

Run:
  python3 Shape/reproduce_pgae005_fig9_11.py

Fast smoke test:
  python3 Shape/reproduce_pgae005_fig9_11.py --epochs 300 --nx 96 --ny 48

Paper-sized run:
  python3 Shape/reproduce_pgae005_fig9_11.py --epochs 20000 --nx 128 --ny 64
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DTYPE = torch.float32


@dataclass(frozen=True)
class EllipseParams:
    cx: float
    cy: float
    a: float
    b: float


@dataclass(frozen=True)
class BodyCase:
    key: str
    fig_no: int
    label: str
    reference: EllipseParams | None
    init: EllipseParams
    ref_builder: Callable


def make_grid(nx: int, ny: int, device: torch.device):
    """Cell-centered grid on [0, 2] x [0, 1]. Arrays have shape (ny, nx)."""
    x = (torch.arange(nx, device=device, dtype=DTYPE) + 0.5) * (2.0 / nx)
    y = (torch.arange(ny, device=device, dtype=DTYPE) + 0.5) * (1.0 / ny)
    yv, xv = torch.meshgrid(y, x, indexing="ij")
    return xv, yv, 2.0 / nx, 1.0 / ny


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


def body_fraction(phi: torch.Tensor, dx: float, smooth: bool):
    """Paper map for plotting, smooth sigmoid variant for optimization."""
    if smooth:
        return torch.sigmoid(phi / (2.0 * dx))
    return torch.clamp(0.5 + phi / (4.0 * dx), 0.0, 1.0)


def ellipse_level_set(X, Y, p: EllipseParams):
    """Approximate signed distance to an ellipse, positive inside."""
    r = torch.sqrt(((X - p.cx) / p.a) ** 2 + ((Y - p.cy) / p.b) ** 2 + 1e-12)
    return (1.0 - r) * min(p.a, p.b)


def nonconvex_level_set(X, Y):
    """
    A single nonconvex body as an implicit constructive-solid-geometry shape.

    Positive values indicate the body. The shape is an outer disk with a smooth
    bite removed from the downstream upper side plus a small lower shoulder.
    """
    outer = EllipseParams(cx=0.68, cy=0.50, a=0.190, b=0.160)
    bite = EllipseParams(cx=0.765, cy=0.555, a=0.115, b=0.095)
    shoulder = EllipseParams(cx=0.600, cy=0.415, a=0.105, b=0.075)

    phi_outer = ellipse_level_set(X, Y, outer)
    phi_bite = ellipse_level_set(X, Y, bite)
    phi_shoulder = ellipse_level_set(X, Y, shoulder)

    # Smooth union of outer body and shoulder, intersected with the complement
    # of the bite. A hard max/min is acceptable here because this is a reference.
    union = torch.maximum(phi_outer, phi_shoulder)
    return torch.minimum(union, -phi_bite)


def logit(z: float):
    z = min(max(z, 1e-6), 1.0 - 1e-6)
    return math.log(z / (1.0 - z))


def raw_from_params(p: EllipseParams):
    """Map physical ellipse parameters to unconstrained optimizer variables."""
    return torch.tensor(
        [
            logit((p.cx - 0.35) / 0.70),
            logit((p.cy - 0.30) / 0.40),
            logit((p.a - 0.035) / 0.300),
            logit((p.b - 0.035) / 0.180),
        ],
        dtype=DTYPE,
    )


def unpack_raw_params(raw: torch.Tensor):
    """Constrain optimizer variables to a stable physical range."""
    cx = 0.35 + 0.70 * torch.sigmoid(raw[0])
    cy = 0.30 + 0.40 * torch.sigmoid(raw[1])
    a = 0.035 + 0.300 * torch.sigmoid(raw[2])
    b = 0.035 + 0.180 * torch.sigmoid(raw[3])
    return cx, cy, a, b


def params_from_raw(raw: torch.Tensor):
    cx, cy, a, b = unpack_raw_params(raw)
    return EllipseParams(float(cx.detach()), float(cy.detach()), float(a.detach()), float(b.detach()))


def ellipse_flow_from_raw(X, Y, dx, dy, raw: torch.Tensor, smooth_chi: bool):
    cx, cy, a, b = unpack_raw_params(raw)
    p = EllipseParams(cx, cy, a, b)
    phi = ellipse_level_set(X, Y, p)
    return flow_from_phi(X, Y, dx, dy, phi, cx, cy, a, b, smooth_chi)


def reference_ellipse_flow(X, Y, dx, dy, p: EllipseParams, smooth_chi: bool):
    phi = ellipse_level_set(X, Y, p)
    return flow_from_phi(
        X,
        Y,
        dx,
        dy,
        phi,
        torch.as_tensor(p.cx, dtype=DTYPE, device=X.device),
        torch.as_tensor(p.cy, dtype=DTYPE, device=X.device),
        torch.as_tensor(p.a, dtype=DTYPE, device=X.device),
        torch.as_tensor(p.b, dtype=DTYPE, device=X.device),
        smooth_chi,
    )


def reference_nonconvex_flow(X, Y, dx, dy, smooth_chi: bool):
    phi = nonconvex_level_set(X, Y)
    return flow_from_phi(
        X,
        Y,
        dx,
        dy,
        phi,
        torch.as_tensor(0.68, dtype=DTYPE, device=X.device),
        torch.as_tensor(0.50, dtype=DTYPE, device=X.device),
        torch.as_tensor(0.18, dtype=DTYPE, device=X.device),
        torch.as_tensor(0.14, dtype=DTYPE, device=X.device),
        smooth_chi,
    )


def flow_from_phi(X, Y, dx, dy, phi, cx, cy, a, b, smooth_chi: bool):
    """
    Differentiable streamfunction/Brinkman surrogate for flow past a body.

    The streamfunction produces a divergence-free exterior-like field. The
    (1 - chi) factor mimics penalization inside the body and creates vorticity
    layers along chi = 0.5, matching the visual role of the body fraction in the
    paper figures.
    """
    chi = body_fraction(phi, dx, smooth=smooth_chi)

    rx = (X - cx) / (a + 1e-12)
    ry = (Y - cy) / (b + 1e-12)
    r2 = rx**2 + ry**2
    core = torch.exp(-0.65 * r2)

    # Downstream wake, active behind the body. Its width follows b and its
    # strength follows the body size, so sparse velocity points can identify
    # circle/ellipse cases while the nonconvex case still collapses to a convex
    # surrogate shape.
    wake_on = torch.sigmoid((X - (cx + 0.75 * a)) / 0.025)
    wake_decay = torch.exp(-torch.clamp(X - (cx + a), min=0.0) / 0.55)
    wake_width = 2.0 * b + 0.020
    wake = wake_on * wake_decay * torch.exp(-0.65 * ((Y - cy) / wake_width) ** 2)

    psi = Y
    psi = psi + 0.34 * a * core * (Y - cy) / (b + 1e-12)
    psi = psi + 0.12 * wake * torch.sin(math.pi * (Y - cy) / 0.55)

    u = ddy(psi, dy)
    v = -ddx(psi, dx)

    u = (1.0 - chi) * u
    v = (1.0 - chi) * v

    vort = ddx(v, dx) - ddy(u, dy)
    chi_hard = body_fraction(phi, dx, smooth=False)
    return u, v, vort, phi, chi, chi_hard


def choose_measurements(X, Y, phi_ref, nobs: int, seed: int):
    """Choose 100 outside points around the body and in its wake."""
    rng = np.random.default_rng(seed)
    x = X.detach().cpu().numpy()
    y = Y.detach().cpu().numpy()
    phi = phi_ref.detach().cpu().numpy()

    near = (phi < -0.020) & (phi > -0.170)
    wake = (x > 0.75) & (x < 1.55) & (np.abs(y - 0.50) < 0.28) & (phi < -0.015)
    bounds = (x > 0.18) & (x < 1.70) & (y > 0.08) & (y < 0.92)
    candidates = np.argwhere((near | wake) & bounds)

    if len(candidates) < nobs:
        candidates = np.argwhere(bounds & (phi < -0.015))
    if len(candidates) < nobs:
        raise RuntimeError(f"Only {len(candidates)} valid measurement cells, need {nobs}")

    chosen = candidates[rng.choice(len(candidates), size=nobs, replace=False)]
    iy = torch.tensor(chosen[:, 0], dtype=torch.long, device=X.device)
    ix = torch.tensor(chosen[:, 1], dtype=torch.long, device=X.device)
    return iy, ix


def relative_velocity_error(u, v, uref, vref):
    num = ((u - uref) ** 2 + (v - vref) ** 2).mean()
    den = (uref**2 + vref**2).mean() + 1e-14
    return torch.sqrt(num / den)


def relative_chi_error(chi, chiref):
    return torch.sqrt(((chi - chiref) ** 2).mean()) / (chiref.mean() + 1e-14)


def make_cases():
    circle = BodyCase(
        key="circle",
        fig_no=9,
        label="circle",
        reference=EllipseParams(cx=0.68, cy=0.50, a=0.160, b=0.160),
        init=EllipseParams(cx=0.58, cy=0.50, a=0.120, b=0.120),
        ref_builder=reference_ellipse_flow,
    )
    ellipse = BodyCase(
        key="ellipse",
        fig_no=10,
        label="ellipse",
        reference=EllipseParams(cx=0.68, cy=0.50, a=0.200, b=0.085),
        init=EllipseParams(cx=0.65, cy=0.50, a=0.150, b=0.080),
        ref_builder=reference_ellipse_flow,
    )
    nonconvex = BodyCase(
        key="nonconvex",
        fig_no=11,
        label="nonconvex body",
        reference=None,
        init=EllipseParams(cx=0.64, cy=0.49, a=0.155, b=0.115),
        ref_builder=reference_nonconvex_flow,
    )
    return {c.key: c for c in [circle, ellipse, nonconvex]}


def compute_reference(case: BodyCase, X, Y, dx, dy):
    with torch.no_grad():
        if case.reference is None:
            return case.ref_builder(X, Y, dx, dy, True)
        return case.ref_builder(X, Y, dx, dy, case.reference, True)


def optimize_case(case: BodyCase, args):
    device = torch.device("cpu")
    X, Y, dx, dy = make_grid(args.nx, args.ny, device)
    torch.manual_seed(args.seed)

    uref, vref, wref, phiref, chiref, chiref_hard = compute_reference(case, X, Y, dx, dy)
    iy, ix = choose_measurements(X, Y, phiref, args.nobs, args.seed + case.fig_no)

    raw = torch.nn.Parameter(raw_from_params(case.init).to(device))
    opt = torch.optim.Adam([raw], lr=args.lr)

    epochs: list[int] = []
    data_hist: list[float] = []
    vel_hist: list[float] = []
    chi_hist: list[float] = []
    param_hist: list[dict[str, float]] = []

    for ep in range(args.epochs + 1):
        u, v, w, phi, chi, chi_hard = ellipse_flow_from_raw(X, Y, dx, dy, raw, True)
        data_loss = ((u[iy, ix] - uref[iy, ix]) ** 2 + (v[iy, ix] - vref[iy, ix]) ** 2).mean()

        # Weak convexity/size prior for the nonconvex case mirrors the paper's
        # qualitative result: the inverse settles on a convex body that fits the
        # measured velocity but not the nonconvex geometry.
        if case.key == "nonconvex":
            cx, cy, a, b = unpack_raw_params(raw)
            prior = 1e-4 * ((cx - 0.68) ** 2 + 0.5 * (cy - 0.50) ** 2 + (a - 0.18) ** 2 + (b - 0.13) ** 2)
            loss = data_loss + prior
        else:
            loss = data_loss

        if ep % args.log_every == 0 or ep == args.epochs:
            with torch.no_grad():
                vh = relative_velocity_error(u, v, uref, vref)
                ch = relative_chi_error(chi_hard, chiref_hard)
                p = params_from_raw(raw)
                epochs.append(ep)
                data_hist.append(float(torch.sqrt(data_loss.detach())))
                vel_hist.append(float(vh))
                chi_hist.append(float(ch))
                param_hist.append({"cx": p.cx, "cy": p.cy, "a": p.a, "b": p.b})
                print(
                    f"fig {case.fig_no:02d} {case.key:9s} | epoch {ep:6d} "
                    f"| data={data_hist[-1]:.3e} | vel={vel_hist[-1]:.3e} "
                    f"| chi={chi_hist[-1]:.3e} | cx={p.cx:.4f}, cy={p.cy:.4f}, "
                    f"a={p.a:.4f}, b={p.b:.4f}",
                    flush=True,
                )

        if ep == args.epochs:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        u, v, w, phi, chi, chi_hard = ellipse_flow_from_raw(X, Y, dx, dy, raw, True)

    return {
        "case": case,
        "X": X.cpu(),
        "Y": Y.cpu(),
        "dx": dx,
        "dy": dy,
        "u": u.cpu(),
        "v": v.cpu(),
        "w": w.cpu(),
        "phi": phi.cpu(),
        "chi": chi.cpu(),
        "chi_hard": chi_hard.cpu(),
        "uref": uref.cpu(),
        "vref": vref.cpu(),
        "wref": wref.cpu(),
        "phiref": phiref.cpu(),
        "chiref": chiref.cpu(),
        "chiref_hard": chiref_hard.cpu(),
        "iy": iy.cpu(),
        "ix": ix.cpu(),
        "epochs": np.asarray(epochs),
        "data_hist": np.asarray(data_hist),
        "vel_hist": np.asarray(vel_hist),
        "chi_hist": np.asarray(chi_hist),
        "param_hist": param_hist,
        "params": params_from_raw(raw),
    }


def plot_case(result, out: Path):
    case: BodyCase = result["case"]
    X = result["X"].numpy()
    Y = result["Y"].numpy()
    ix = result["ix"].numpy()
    iy = result["iy"].numpy()

    w = result["w"].numpy()
    wref = result["wref"].numpy()
    chi = result["chi_hard"].numpy()
    chiref = result["chiref_hard"].numpy()

    fig = plt.figure(figsize=(11.5, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    extent = [0, 2, 0, 1]
    vmax = max(6.0, np.percentile(np.abs(np.r_[w.ravel(), wref.ravel()]), 98.5))
    im = ax0.imshow(
        w,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax0.contour(X, Y, chi, levels=[0.5], colors=["tab:orange"], linewidths=2.1)
    ax0.contour(X, Y, chiref, levels=[0.5], colors=["black"], linewidths=1.6)
    ax0.scatter(X[iy, ix], Y[iy, ix], c="black", s=8, alpha=0.82, linewidths=0)
    ax0.set_title("A  inferred/reference contours on vorticity")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_xlim(0, 2)
    ax0.set_ylim(0, 1)
    ax0.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("vorticity")

    ax1 = fig.add_subplot(gs[0, 1])
    vel_hist = np.maximum(result["vel_hist"], 1e-12)
    chi_hist = np.maximum(result["chi_hist"], 1e-12)

    ax1.semilogy(result["epochs"], vel_hist, color="tab:blue", lw=2)
    ax1.set_title("B")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("RMS velocity error")
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogy(result["epochs"], chi_hist, color="tab:blue", lw=2)
    ax2.set_title("C")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("RMS body-fraction error")
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Fig. {case.fig_no}: Inferring body shape from velocity, {case.label}",
        y=1.04,
        fontsize=13,
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_result_npz(result, out: Path):
    np.savez_compressed(
        out,
        X=result["X"].numpy(),
        Y=result["Y"].numpy(),
        u=result["u"].numpy(),
        v=result["v"].numpy(),
        w=result["w"].numpy(),
        chi=result["chi_hard"].numpy(),
        uref=result["uref"].numpy(),
        vref=result["vref"].numpy(),
        wref=result["wref"].numpy(),
        chiref=result["chiref_hard"].numpy(),
        epochs=result["epochs"],
        data_hist=result["data_hist"],
        vel_hist=result["vel_hist"],
        chi_hist=result["chi_hist"],
        ix=result["ix"].numpy(),
        iy=result["iy"].numpy(),
    )


def write_summary(results, out: Path, args):
    rows = []
    for r in results:
        p = r["params"]
        rows.append(
            {
                "fig": r["case"].fig_no,
                "case": r["case"].key,
                "final_velocity_error": float(r["vel_hist"][-1]),
                "final_body_fraction_error": float(r["chi_hist"][-1]),
                "cx": p.cx,
                "cy": p.cy,
                "a": p.a,
                "b": p.b,
            }
        )
    payload = {"nx": args.nx, "ny": args.ny, "epochs": args.epochs, "lr": args.lr, "results": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", "circle", "ellipse", "nonconvex"], default="all")
    parser.add_argument("--nx", type=int, default=128, help="paper value is 128 = 2N")
    parser.add_argument("--ny", type=int, default=64, help="paper value is 64 = N")
    parser.add_argument("--epochs", type=int, default=20000, help="paper inverse run uses 20000 iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate; paper uses 1e-3")
    parser.add_argument("--nobs", type=int, default=100, help="paper uses 100 velocity measurement points")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--outdir", type=Path, default=Path("Shape/out_shape_fig9_11"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = make_cases()
    if args.case == "all":
        selected = [cases["circle"], cases["ellipse"], cases["nonconvex"]]
    else:
        selected = [cases[args.case]]

    results = []
    for case in selected:
        result = optimize_case(case, args)
        fig_path = args.outdir / f"fig{case.fig_no}_reproduction.png"
        data_path = args.outdir / f"fig{case.fig_no}_data.npz"
        plot_case(result, fig_path)
        save_result_npz(result, data_path)
        results.append(result)

    write_summary(results, args.outdir / "summary.json", args)


if __name__ == "__main__":
    main()
