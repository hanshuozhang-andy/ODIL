#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-faithful ODIL-style lid-driven cavity forward solver for macOS.

This script implements the components described for the lid-driven cavity
example in Karnakov, Litvinov & Koumoutsakos (PNAS Nexus, 2024):

  * steady 2-D incompressible Navier--Stokes equations;
  * collocated finite-volume unknowns u, v, p on a Cartesian grid;
  * no-slip lid-driven cavity boundary conditions;
  * second-order upwind convection through deferred correction:
        implicit residual uses compact first-order upwind;
        second-order correction is frozen from the previous outer iterate;
  * Rhie--Chow-type face-normal velocity interpolation in the continuity
    residual to suppress pressure checkerboard oscillations;
  * sparse Gauss--Newton / trust-region least-squares solve with an explicit
    Jacobian sparsity pattern.

Important honesty note:
  The public cselab/odil repository currently exposes the ODIL framework and
  examples, but the exact paper cavity script is not present in the examples
  directory. This file is therefore a paper-faithful standalone reproduction of
  the described discretization/optimization ideas, not a line-by-line copy of
  a hidden/private script.

Dependencies:
    pip install numpy scipy matplotlib

Examples:
    python lid_cavity_odil_strict_mac.py --Re 100 --N 17 --outer 10 --inner 25
    python lid_cavity_odil_strict_mac.py --Re 100 --N 33 --outer 15 --inner 35

For a MacBook Air, start with N=17. N=33 is much slower because sparse finite
-difference Jacobians are still expensive.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class Params:
    N: int
    Re: float
    rc: float
    pfix: float
    dtype: type = np.float64


# -----------------------------------------------------------------------------
# Packing / unpacking
# -----------------------------------------------------------------------------

def unpack(x: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = N * N
    u = x[:m].reshape(N, N)
    v = x[m:2 * m].reshape(N, N)
    p = x[2 * m:3 * m].reshape(N, N)
    return u, v, p


def pack(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.concatenate([u.ravel(), v.ravel(), p.ravel()])


def idx(comp: int, j: int, i: int, N: int) -> int:
    """Column index in global vector. comp: 0=u, 1=v, 2=p."""
    return comp * N * N + j * N + i


# -----------------------------------------------------------------------------
# Boundary and ghost-cell utilities
# -----------------------------------------------------------------------------

def vel_bc(comp: int, side: str) -> float:
    """Velocity Dirichlet boundary value. comp=0 for u, comp=1 for v."""
    if side == "top" and comp == 0:
        return 1.0
    return 0.0


def side_from_outside(j: int, i: int, N: int) -> str:
    if i < 0:
        return "left"
    if i >= N:
        return "right"
    if j < 0:
        return "bottom"
    if j >= N:
        return "top"
    raise ValueError("index is inside")


def get_velocity(phi: np.ndarray, comp: int, j: int, i: int) -> float:
    """Cell-centered velocity with one-layer/extended ghost values.

    Dirichlet ghost rule: phi_ghost = 2*bc - phi_inside_mirror.
    This imposes wall/lid values at the boundary face.
    """
    N = phi.shape[0]
    if 0 <= j < N and 0 <= i < N:
        return float(phi[j, i])

    jj = min(max(j, 0), N - 1)
    ii = min(max(i, 0), N - 1)
    side = side_from_outside(j, i, N)
    bc = vel_bc(comp, side)
    return float(2.0 * bc - phi[jj, ii])


def get_pressure(p: np.ndarray, j: int, i: int) -> float:
    """Cell-centered pressure with zero-normal-gradient ghost values."""
    N = p.shape[0]
    jj = min(max(j, 0), N - 1)
    ii = min(max(i, 0), N - 1)
    return float(p[jj, ii])


def gradp_x(p: np.ndarray, j: int, i: int, h: float) -> float:
    return (get_pressure(p, j, i + 1) - get_pressure(p, j, i - 1)) / (2.0 * h)


def gradp_y(p: np.ndarray, j: int, i: int, h: float) -> float:
    return (get_pressure(p, j + 1, i) - get_pressure(p, j - 1, i)) / (2.0 * h)


# -----------------------------------------------------------------------------
# Rhie--Chow face-normal interpolation
# -----------------------------------------------------------------------------

def rc_face_u(u: np.ndarray, p: np.ndarray, j: int, i_left: int, h: float, rc: float) -> float:
    """Normal velocity u at vertical face between i_left and i_left+1.

    Boundary faces return wall normal velocity directly.
    Interior faces use Rhie--Chow-style correction:
        u_f = avg(u) - d * [ (p_E-p_P)/h - avg(dp/dx)_P,E ].
    """
    N = u.shape[0]
    if i_left < 0 or i_left >= N - 1:
        return 0.0
    ubar = 0.5 * (u[j, i_left] + u[j, i_left + 1])
    dp_face = (p[j, i_left + 1] - p[j, i_left]) / h
    dp_interp = 0.5 * (gradp_x(p, j, i_left, h) + gradp_x(p, j, i_left + 1, h))
    # d has dimensions of velocity divided by pressure-gradient.  The exact
    # SIMPLE coefficient is A_f/a_P; here rc*h is a robust compact equivalent.
    d = rc * h
    return float(ubar - d * (dp_face - dp_interp))


def rc_face_v(v: np.ndarray, p: np.ndarray, j_bot: int, i: int, h: float, rc: float) -> float:
    """Normal velocity v at horizontal face between j_bot and j_bot+1."""
    N = v.shape[0]
    if j_bot < 0 or j_bot >= N - 1:
        return 0.0
    vbar = 0.5 * (v[j_bot, i] + v[j_bot + 1, i])
    dp_face = (p[j_bot + 1, i] - p[j_bot, i]) / h
    dp_interp = 0.5 * (gradp_y(p, j_bot, i, h) + gradp_y(p, j_bot + 1, i, h))
    d = rc * h
    return float(vbar - d * (dp_face - dp_interp))


def face_fluxes(u: np.ndarray, v: np.ndarray, p: np.ndarray, j: int, i: int, h: float, rc: float) -> Tuple[float, float, float, float]:
    """Return Fe, Fw, Fn, Fs with positive x/y convention.

    Divergence is (Fe - Fw + Fn - Fs)/h.
    """
    Fe = rc_face_u(u, p, j, i, h, rc)       # east face i+1/2
    Fw = rc_face_u(u, p, j, i - 1, h, rc)   # west face i-1/2
    Fn = rc_face_v(v, p, j, i, h, rc)       # north face j+1/2
    Fs = rc_face_v(v, p, j - 1, i, h, rc)   # south face j-1/2
    return Fe, Fw, Fn, Fs


# -----------------------------------------------------------------------------
# Upwind and second-order upwind face values
# -----------------------------------------------------------------------------

def low_upwind(phi: np.ndarray, comp: int, face: str, j: int, i: int, F: float) -> float:
    """First-order upwind face value for a face of cell (j,i)."""
    if face == "e":
        return get_velocity(phi, comp, j, i) if F >= 0 else get_velocity(phi, comp, j, i + 1)
    if face == "w":
        return get_velocity(phi, comp, j, i - 1) if F >= 0 else get_velocity(phi, comp, j, i)
    if face == "n":
        return get_velocity(phi, comp, j, i) if F >= 0 else get_velocity(phi, comp, j + 1, i)
    if face == "s":
        return get_velocity(phi, comp, j - 1, i) if F >= 0 else get_velocity(phi, comp, j, i)
    raise ValueError(face)


def second_upwind(phi: np.ndarray, comp: int, face: str, j: int, i: int, F: float) -> float:
    """Second-order upwind extrapolated face value.

    Falls back naturally through ghost values near boundaries.
    """
    if face == "e":
        if F >= 0:
            return 1.5 * get_velocity(phi, comp, j, i) - 0.5 * get_velocity(phi, comp, j, i - 1)
        return 1.5 * get_velocity(phi, comp, j, i + 1) - 0.5 * get_velocity(phi, comp, j, i + 2)
    if face == "w":
        if F >= 0:
            return 1.5 * get_velocity(phi, comp, j, i - 1) - 0.5 * get_velocity(phi, comp, j, i - 2)
        return 1.5 * get_velocity(phi, comp, j, i) - 0.5 * get_velocity(phi, comp, j, i + 1)
    if face == "n":
        if F >= 0:
            return 1.5 * get_velocity(phi, comp, j, i) - 0.5 * get_velocity(phi, comp, j - 1, i)
        return 1.5 * get_velocity(phi, comp, j + 1, i) - 0.5 * get_velocity(phi, comp, j + 2, i)
    if face == "s":
        if F >= 0:
            return 1.5 * get_velocity(phi, comp, j - 1, i) - 0.5 * get_velocity(phi, comp, j - 2, i)
        return 1.5 * get_velocity(phi, comp, j, i) - 0.5 * get_velocity(phi, comp, j + 1, i)
    raise ValueError(face)


def deferred_face(phi: np.ndarray,
                  phi_ref: np.ndarray,
                  comp: int,
                  face: str,
                  j: int,
                  i: int,
                  F: float,
                  F_ref: float) -> float:
    """Deferred-correction face value.

    current compact first-order upwind + frozen high-order correction from
    previous outer iterate.
    """
    low_now = low_upwind(phi, comp, face, j, i, F)
    high_old = second_upwind(phi_ref, comp, face, j, i, F_ref)
    low_old = low_upwind(phi_ref, comp, face, j, i, F_ref)
    return low_now + (high_old - low_old)


# -----------------------------------------------------------------------------
# Residual assembly
# -----------------------------------------------------------------------------

def residual(x: np.ndarray, x_ref: np.ndarray, par: Params) -> np.ndarray:
    N = par.N
    h = 1.0 / N
    u, v, p = unpack(x, N)
    u0, v0, p0 = unpack(x_ref, N)

    out = []

    for j in range(N):
        for i in range(N):
            Fe, Fw, Fn, Fs = face_fluxes(u, v, p, j, i, h, par.rc)
            Fe0, Fw0, Fn0, Fs0 = face_fluxes(u0, v0, p0, j, i, h, par.rc)

            # Continuity with Rhie--Chow face-normal velocities.
            rdiv = (Fe - Fw + Fn - Fs) / h

            # Convective finite-volume terms with second-order upwind deferred correction.
            ue = deferred_face(u, u0, 0, "e", j, i, Fe, Fe0)
            uw = deferred_face(u, u0, 0, "w", j, i, Fw, Fw0)
            un = deferred_face(u, u0, 0, "n", j, i, Fn, Fn0)
            us = deferred_face(u, u0, 0, "s", j, i, Fs, Fs0)

            ve = deferred_face(v, v0, 1, "e", j, i, Fe, Fe0)
            vw = deferred_face(v, v0, 1, "w", j, i, Fw, Fw0)
            vn = deferred_face(v, v0, 1, "n", j, i, Fn, Fn0)
            vs = deferred_face(v, v0, 1, "s", j, i, Fs, Fs0)

            conv_u = (Fe * ue - Fw * uw + Fn * un - Fs * us) / h
            conv_v = (Fe * ve - Fw * vw + Fn * vn - Fs * vs) / h

            # Diffusion with central differences and Dirichlet ghost velocities.
            uc = get_velocity(u, 0, j, i)
            vc = get_velocity(v, 1, j, i)
            lap_u = (
                get_velocity(u, 0, j, i + 1) - 2.0 * uc + get_velocity(u, 0, j, i - 1)
                + get_velocity(u, 0, j + 1, i) - 2.0 * uc + get_velocity(u, 0, j - 1, i)
            ) / (h * h)
            lap_v = (
                get_velocity(v, 1, j, i + 1) - 2.0 * vc + get_velocity(v, 1, j, i - 1)
                + get_velocity(v, 1, j + 1, i) - 2.0 * vc + get_velocity(v, 1, j - 1, i)
            ) / (h * h)

            dpdx = (get_pressure(p, j, i + 1) - get_pressure(p, j, i - 1)) / (2.0 * h)
            dpdy = (get_pressure(p, j + 1, i) - get_pressure(p, j - 1, i)) / (2.0 * h)

            ru = conv_u + dpdx - (1.0 / par.Re) * lap_u
            rv = conv_v + dpdy - (1.0 / par.Re) * lap_v

            out.extend([ru, rv, rdiv])

    # Pressure gauge condition. Pressure is defined up to a constant.
    out.append(par.pfix * p[0, 0])

    return np.asarray(out, dtype=par.dtype)


# -----------------------------------------------------------------------------
# Sparse Jacobian pattern
# -----------------------------------------------------------------------------

def build_jac_sparsity(N: int, radius: int = 2) -> lil_matrix:
    """Conservative dependency pattern for finite-difference sparse Jacobian."""
    n_unknown = 3 * N * N
    n_res = 3 * N * N + 1
    S = lil_matrix((n_res, n_unknown), dtype=bool)

    for j in range(N):
        for i in range(N):
            base_r = 3 * (j * N + i)
            for jj in range(max(0, j - radius), min(N, j + radius + 1)):
                for ii in range(max(0, i - radius), min(N, i + radius + 1)):
                    for comp in (0, 1, 2):
                        S[base_r + 0, idx(comp, jj, ii, N)] = True
                        S[base_r + 1, idx(comp, jj, ii, N)] = True
                        S[base_r + 2, idx(comp, jj, ii, N)] = True

    S[-1, idx(2, 0, 0, N)] = True
    return S


# -----------------------------------------------------------------------------
# Diagnostics and plotting
# -----------------------------------------------------------------------------

def rms_blocks(r: np.ndarray, N: int) -> Dict[str, float]:
    body = r[:-1].reshape(N * N, 3)
    return {
        "ru": float(np.sqrt(np.mean(body[:, 0] ** 2))),
        "rv": float(np.sqrt(np.mean(body[:, 1] ** 2))),
        "div": float(np.sqrt(np.mean(body[:, 2] ** 2))),
        "total": float(np.sqrt(np.mean(r ** 2))),
    }


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    N = u.shape[0]
    h = 1.0 / N
    w = np.zeros_like(u)
    for j in range(N):
        for i in range(N):
            dvdx = (get_velocity(v, 1, j, i + 1) - get_velocity(v, 1, j, i - 1)) / (2.0 * h)
            dudy = (get_velocity(u, 0, j + 1, i) - get_velocity(u, 0, j - 1, i)) / (2.0 * h)
            w[j, i] = dvdx - dudy
    return w


def save_outputs(x: np.ndarray, hist: list, args) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    N = args.N
    h = 1.0 / N
    u, v, p = unpack(x, N)
    w = vorticity(u, v)

    xc = (np.arange(N) + 0.5) * h
    yc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, yc)
    speed = np.sqrt(u * u + v * v)

    np.savez(os.path.join(args.outdir, "solution.npz"), x=xc, y=yc, u=u, v=v, p=p, omega=w, speed=speed)

    with open(os.path.join(args.outdir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer", "cost", "rms", "ru", "rv", "div", "nfev", "status"])
        writer.writeheader()
        writer.writerows(hist)

    # Field figure.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    im = axes[0].contourf(X, Y, speed, levels=40)
    axes[0].set_title("|u|")
    fig.colorbar(im, ax=axes[0])

    axes[1].streamplot(xc, yc, u, v, density=1.5, linewidth=1.0, arrowsize=1.0)
    axes[1].set_title("streamlines")

    im = axes[2].contourf(X, Y, w, levels=40)
    axes[2].set_title("vorticity")
    fig.colorbar(im, ax=axes[2])

    im = axes[3].contourf(X, Y, p - np.mean(p), levels=40)
    axes[3].set_title("pressure - mean")
    fig.colorbar(im, ax=axes[3])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"ODIL-style lid-driven cavity, Re={args.Re}, N={N}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fields.png"), dpi=220)
    plt.close(fig)

    # Convergence figure.
    fig, ax = plt.subplots(figsize=(7, 5))
    if hist:
        outer = [hrow["outer"] for hrow in hist]
        ax.semilogy(outer, [hrow["rms"] for hrow in hist], marker="o", label="all")
        ax.semilogy(outer, [hrow["ru"] for hrow in hist], marker="o", label="u-mom")
        ax.semilogy(outer, [hrow["rv"] for hrow in hist], marker="o", label="v-mom")
        ax.semilogy(outer, [hrow["div"] for hrow in hist], marker="o", label="div")
    ax.set_xlabel("outer deferred-correction Newton cycle")
    ax.set_ylabel("RMS residual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "convergence.png"), dpi=220)
    plt.close(fig)

    # Centerline profiles.
    ix = int(np.argmin(np.abs(xc - 0.5)))
    iy = int(np.argmin(np.abs(yc - 0.5)))
    with open(os.path.join(args.outdir, "centerlines.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "u(x=0.5,y)", "x", "v(x,y=0.5)"])
        for k in range(N):
            writer.writerow([yc[k], u[k, ix], xc[k], v[iy, k]])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(u[:, ix], yc, marker="o", ms=3)
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].set_title("u along x=0.5")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xc, v[iy, :], marker="o", ms=3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].set_title("v along y=0.5")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "centerlines.png"), dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main solve
# -----------------------------------------------------------------------------

def initial_guess(N: int) -> np.ndarray:
    h = 1.0 / N
    yc = (np.arange(N) + 0.5) * h
    u = np.tile(yc.reshape(N, 1), (1, N))  # Couette-like initialization
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    return pack(u, v, p)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-faithful ODIL-style lid-driven cavity solver")
    parser.add_argument("--N", type=int, default=33, help="number of finite-volume cells per direction")
    parser.add_argument("--Re", type=float, default=1000.0)
    parser.add_argument("--outer", type=int, default=15, help="outer deferred-correction cycles")
    parser.add_argument("--inner", type=int, default=35, help="max residual evaluations per sparse least-squares solve")
    parser.add_argument("--rc", type=float, default=0.05, help="Rhie-Chow correction coefficient")
    parser.add_argument("--pfix", type=float, default=1.0, help="pressure gauge residual weight")
    parser.add_argument("--outdir", type=str, default="out_cavity_strict")
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    par = Params(N=args.N, Re=args.Re, rc=args.rc, pfix=args.pfix)

    print("=" * 78)
    print("Paper-faithful ODIL-style lid-driven cavity")
    print(f"N={args.N}, Re={args.Re}, outer={args.outer}, inner={args.inner}, rc={args.rc}")
    print("Discretization: FV + Rhie-Chow continuity + 2nd-upwind deferred correction")
    print("Optimization: sparse Gauss-Newton / trust-region least_squares with LSMR")
    print("=" * 78)

    x = initial_guess(args.N).astype(np.float64)
    S = build_jac_sparsity(args.N, radius=2).tocsr()

    hist = []
    t0 = time.time()

    for outer in range(args.outer):
        x_ref = x.copy()

        def fun(z):
            return residual(z, x_ref, par)

        r0 = fun(x)
        d0 = rms_blocks(r0, args.N)
        print(f"outer {outer:03d} before: rms={d0['total']:.3e} ru={d0['ru']:.3e} rv={d0['rv']:.3e} div={d0['div']:.3e}")

        res = least_squares(
            fun,
            x,
            jac="2-point",
            jac_sparsity=S,
            method="trf",
            tr_solver="lsmr",
            max_nfev=args.inner,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )
        x = res.x
        r = fun(x)
        d = rms_blocks(r, args.N)
        hist.append({
            "outer": outer + 1,
            "cost": float(res.cost),
            "rms": d["total"],
            "ru": d["ru"],
            "rv": d["rv"],
            "div": d["div"],
            "nfev": int(res.nfev),
            "status": int(res.status),
        })
        print(f"outer {outer:03d} after : rms={d['total']:.3e} ru={d['ru']:.3e} rv={d['rv']:.3e} div={d['div']:.3e} nfev={res.nfev} status={res.status}")

        if d["total"] < 1e-6:
            print("Converged by residual tolerance.")
            break

    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.2f} s")

    if not args.no_plot:
        save_outputs(x, hist, args)
        print("Saved:")
        print(f"  {args.outdir}/fields.png")
        print(f"  {args.outdir}/convergence.png")
        print(f"  {args.outdir}/centerlines.png")
        print(f"  {args.outdir}/centerlines.csv")
        print(f"  {args.outdir}/solution.npz")


if __name__ == "__main__":
    main()
