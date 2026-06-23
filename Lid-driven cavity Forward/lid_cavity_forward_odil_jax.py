#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lid-driven cavity: forward problem with ODIL + JAX.

This script follows the same ODIL/JAX structure as velocity_tracer.py:
  1) set ODIL_BACKEND before importing odil;
  2) define a residual operator on a grid;
  3) create odil.Domain / odil.State / odil.Problem;
  4) optimize a discrete PDE loss and save diagnostic figures.

Equations solved in [0,1]x[0,1], steady incompressible Navier--Stokes:
    u_x + v_y = 0
    u u_x + v u_y = -p_x + (1/Re)(u_xx + u_yy)
    u v_x + v v_y = -p_y + (1/Re)(v_xx + v_yy)

Boundary conditions:
    top wall:    u=1, v=0
    other walls: u=0, v=0

python lid_cavity_forward_odil_jax.py \
  --N 64 \
  --Re 100 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --lr 1e-3 \
  --epochs 8000 \
  --plot_every 1000 \
  --report_every 500 \
  --outdir out_cavity_re100

python lid_cavity_forward_odil_jax.py \
  --N 128 \
  --Re 100 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --lr 1e-3 \
  --epochs 20000 \
  --plot_every 2000 \
  --report_every 500 \
  --outdir out_cavity_re100_128

python lid_cavity_forward_odil_jax.py \
  --N 128 \
  --Re 1000 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --lr 5e-4 \
  --epochs 50000 \
  --plot_every 5000 \
  --report_every 1000 \
  --outdir out_cavity_re1000_128

cd "/Users/hanshuozhang/Desktop/ODIL_literature"
"""

import argparse
import os
import pickle
import sys


# ---------------------------------------------------------------------
# ODIL selects the backend when importing odil. Set JAX first.
# ---------------------------------------------------------------------
def _preparse_backend(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--backend", choices=["jax", "tf"],
                   default=os.environ.get("ODIL_BACKEND", "jax"))
    p.add_argument("--jit", type=int,
                   default=int(os.environ.get("ODIL_JIT", "0")))
    p.add_argument("--gpu", type=int, default=0,
                   help="Use GPU if available. On Apple Silicon, keep 0 unless configured.")
    a, _ = p.parse_known_args(argv)

    os.environ["ODIL_BACKEND"] = a.backend
    os.environ["ODIL_JIT"] = str(a.jit)
    if not a.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    return a


_PRE = _preparse_backend(sys.argv[1:])

import numpy as np
import matplotlib.pyplot as plt
import odil
from odil import printlog


# ---------------------------------------------------------------------
# ODIL residual operator.
# ---------------------------------------------------------------------
def operator_cavity(ctx):
    mod = ctx.mod
    extra = ctx.extra
    args = extra.args

    dx, dy = ctx.step()
    ix, iy = ctx.indices(loc="cc")
    nx, ny = ctx.size()

    zero = ctx.cast(0)
    one = ctx.cast(1)
    nu = ctx.cast(1.0 / args.Re)

    def stencil(key, frozen=False):
        """Cell-centered 5-point stencil: q, q_{i-1}, q_{i+1}, q_{j-1}, q_{j+1}."""
        return [
            ctx.field(key, 0, 0, frozen=frozen),
            ctx.field(key, -1, 0, frozen=frozen),
            ctx.field(key, 1, 0, frozen=frozen),
            ctx.field(key, 0, -1, frozen=frozen),
            ctx.field(key, 0, 1, frozen=frozen),
        ]

    def apply_dirichlet(st, left, right, bottom, top):
        """Linear ghost-cell Dirichlet values at the four walls."""
        q, qxm, qxp, qym, qyp = st
        left = ctx.cast(left)
        right = ctx.cast(right)
        bottom = ctx.cast(bottom)
        top = ctx.cast(top)

        qxm_bc = mod.where(ix == 0, 2.0 * left - q, qxm)
        qxp_bc = mod.where(ix == nx - 1, 2.0 * right - q, qxp)
        qym_bc = mod.where(iy == 0, 2.0 * bottom - q, qym)
        qyp_bc = mod.where(iy == ny - 1, 2.0 * top - q, qyp)
        return q, qxm_bc, qxp_bc, qym_bc, qyp_bc

    def apply_pressure_neumann(st):
        """Simple zero-normal-gradient pressure ghost cells."""
        q, qxm, qxp, qym, qyp = st
        qxm_bc = mod.where(ix == 0, q, qxm)
        qxp_bc = mod.where(ix == nx - 1, q, qxp)
        qym_bc = mod.where(iy == 0, q, qym)
        qyp_bc = mod.where(iy == ny - 1, q, qyp)
        return q, qxm_bc, qxp_bc, qym_bc, qyp_bc

    def ddx(st):
        q, qxm, qxp, qym, qyp = st
        return (qxp - qxm) / (2.0 * dx)

    def ddy(st):
        q, qxm, qxp, qym, qyp = st
        return (qyp - qym) / (2.0 * dy)

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        return (qxp - 2.0 * q + qxm) / dx**2 + (qyp - 2.0 * q + qym) / dy**2

    def ddx_upwind(st, vel):
        q, qxm, qxp, qym, qyp = st
        return mod.where(vel >= 0, (q - qxm) / dx, (qxp - q) / dx)

    def ddy_upwind(st, vel):
        q, qxm, qxp, qym, qyp = st
        return mod.where(vel >= 0, (q - qym) / dy, (qyp - q) / dy)

    # Unknown fields with wall boundary conditions.
    u_st = apply_dirichlet(stencil("u"), left=0.0, right=0.0, bottom=0.0, top=1.0)
    v_st = apply_dirichlet(stencil("v"), left=0.0, right=0.0, bottom=0.0, top=0.0)
    p_st = apply_pressure_neumann(stencil("p"))

    u = u_st[0]
    v = v_st[0]
    p = p_st[0]

    # Frozen velocities only choose the upwind branch, as in velocity_tracer.py.
    uf = ctx.field("u", 0, 0, frozen=True)
    vf = ctx.field("v", 0, 0, frozen=True)

    if args.scheme == "upwind":
        ux_adv = ddx_upwind(u_st, uf)
        uy_adv = ddy_upwind(u_st, vf)
        vx_adv = ddx_upwind(v_st, uf)
        vy_adv = ddy_upwind(v_st, vf)
    else:
        ux_adv = ddx(u_st)
        uy_adv = ddy(u_st)
        vx_adv = ddx(v_st)
        vy_adv = ddy(v_st)

    ux = ddx(u_st)
    vy = ddy(v_st)
    px = ddx(p_st)
    py = ddy(p_st)

    # Continuity and momentum residuals. The sign convention is
    # u*u_x + v*u_y + p_x - nu*lap(u) = 0, and similarly for v.
    f_cont = ux + vy
    f_mom_u = u * ux_adv + v * uy_adv + px - nu * laplace(u_st)
    f_mom_v = u * vx_adv + v * vy_adv + py - nu * laplace(v_st)

    res = [
        ("cont", args.kcont * f_cont),
        ("mom_u", args.kmom * f_mom_u),
        ("mom_v", args.kmom * f_mom_v),
    ]

    # Small pressure damping fixes the arbitrary pressure constant for Adam/L-BFGS.
    # It is intentionally tiny so that it does not drive the velocity solution.
    if args.kp:
        res.append(("p_gauge", args.kp * p))

    # Optional multigrid loss terms, useful with Adam on large grids.
    if args.mgloss:
        from functools import partial
        restrict = partial(odil.core.restrict_to_coarser, loc="cc", mod=mod)
        coarse_terms = [f_cont, f_mom_u, f_mom_v]
        for level in range(args.mgloss):
            coarse_terms = [restrict(f) for f in coarse_terms]
            for name, f in zip(["cont", "mom_u", "mom_v"], coarse_terms):
                res.append((f"{name}_mg{level + 1}", f))

    return res


# ---------------------------------------------------------------------
# Arguments.
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--backend", choices=["jax", "tf"], default=_PRE.backend)
    parser.add_argument("--jit", type=int, default=_PRE.jit)
    parser.add_argument("--gpu", type=int, default=_PRE.gpu)

    parser.add_argument("--N", type=int, default=64, help="Square grid size")
    parser.add_argument("--Nx", type=int, default=None, help="Grid size in x; defaults to N")
    parser.add_argument("--Ny", type=int, default=None, help="Grid size in y; defaults to N")
    parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--scheme", choices=["upwind", "central"], default="upwind",
                        help="Convective derivative discretization")
    parser.add_argument("--kcont", type=float, default=1.0, help="Continuity residual weight")
    parser.add_argument("--kmom", type=float, default=1.0, help="Momentum residual weight")
    parser.add_argument("--kp", type=float, default=1e-5, help="Small pressure damping / gauge weight")
    parser.add_argument("--mgloss", type=int, default=0, help="Add coarse-level residuals to the loss")

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_lid_cavity_forward_odil_jax")
    parser.set_defaults(frames=10)
    parser.set_defaults(plot_every=1000)
    parser.set_defaults(report_every=500)
    parser.set_defaults(history_every=50)
    parser.set_defaults(history_full=10)
    parser.set_defaults(plotext="png")
    parser.set_defaults(plot_title=1)
    parser.set_defaults(dump_data=1)

    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=1e-3)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(mg_interp="conv")

    parser.set_defaults(linsolver="multigrid")
    parser.set_defaults(linsolver_maxiter=20)

    return parser.parse_args()


# ---------------------------------------------------------------------
# NumPy diagnostics / plotting.
# ---------------------------------------------------------------------
def _grid(domain):
    nx, ny = domain.cshape
    x = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    y = np.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny)
    return x, y


def _vorticity(u, v, dx, dy):
    dv_dx = np.gradient(v, dx, axis=0, edge_order=1)
    du_dy = np.gradient(u, dy, axis=1, edge_order=1)
    return dv_dx - du_dy


def _divergence(u, v, dx, dy):
    du_dx = np.gradient(u, dx, axis=0, edge_order=1)
    dv_dy = np.gradient(v, dy, axis=1, edge_order=1)
    return du_dx + dv_dy


def _ghia_v_centerline(Re):
    """Ghia et al. v(x, y=0.5) reference points for Re=100 and Re=1000."""
    x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                  0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                  0.0703, 0.0625, 0.0000])
    if abs(Re - 100.0) < 1e-12:
        v = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
                      -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
                      0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                      0.09233, 0.00000])
    elif abs(Re - 1000.0) < 1e-12:
        v = np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188,
                      -0.51550, -0.42665, -0.31966, 0.02526, 0.32235,
                      0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                      0.27485, 0.00000])
    else:
        return None, None
    order = np.argsort(x)
    return x[order], v[order]


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    args = extra.args

    u = np.array(domain.field(state, "u"))
    v = np.array(domain.field(state, "v"))
    p = np.array(domain.field(state, "p"))

    nx, ny = domain.cshape
    dx, dy = domain.step()
    x, y = _grid(domain)
    X, Y = np.meshgrid(x, y, indexing="ij")
    omega = _vorticity(u, v, dx, dy)

    iy_mid = ny // 2
    ix_mid = nx // 2

    suff = "final" if frame is None else f"{frame:05d}"
    path = f"cavity_Re{args.Re:g}_{suff}.{args.plotext}"
    printlog(path)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.streamplot(X.T, Y.T, u.T, v.T, density=1.4, linewidth=0.8, arrowsize=0.7)
    ax.set_title("streamlines")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[0, 1]
    cf = ax.contourf(X.T, Y.T, omega.T, levels=40)
    ax.contour(X.T, Y.T, omega.T, levels=12, colors="k", linewidths=0.25, alpha=0.5)
    fig.colorbar(cf, ax=ax, shrink=0.9, label="vorticity")
    ax.set_title("vorticity")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axes[1, 0]
    ax.plot(x, v[:, iy_mid], label="ODIL")
    xg, vg = _ghia_v_centerline(args.Re)
    if xg is not None:
        ax.plot(xg, vg, "o", ms=3, label="Ghia reference")
    ax.axhline(0, lw=0.5, color="k")
    ax.set_title("v along y=0.5")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(y, u[ix_mid, :], label="u(x=0.5)")
    ax.plot(x, v[:, iy_mid], label="v(y=0.5)")
    ax.axhline(0, lw=0.5, color="k")
    div_rms = np.sqrt(np.mean(_divergence(u, v, dx, dy) ** 2))
    ax.set_title(f"centerline profiles, div_rms={div_rms:.2e}")
    ax.set_xlabel("coordinate")
    ax.legend()

    if args.plot_title:
        fig.suptitle(f"Lid-driven cavity, Re={args.Re:g}, epoch={epoch}")

    fig.savefig(path, dpi=180)
    plt.close(fig)

    if args.dump_data:
        with open(f"data_Re{args.Re:g}_{suff}.pickle", "wb") as f:
            pickle.dump(
                dict(
                    lower=domain.lower,
                    upper=domain.upper,
                    cshape=domain.cshape,
                    Re=args.Re,
                    x=x,
                    y=y,
                    u=u,
                    v=v,
                    p=p,
                    omega=omega,
                    div=_divergence(u, v, dx, dy),
                ),
                f,
            )


def _diagnostics(domain, state):
    u = np.array(domain.field(state, "u"))
    v = np.array(domain.field(state, "v"))
    dx, dy = domain.step()
    div = _divergence(u, v, dx, dy)
    speed = np.sqrt(u**2 + v**2)
    return dict(
        div_rms=float(np.sqrt(np.mean(div**2))),
        umax=float(np.max(u)),
        umin=float(np.min(u)),
        vmax=float(np.max(v)),
        vmin=float(np.min(v)),
        speed_max=float(np.max(speed)),
    )


def history_func(problem, state, epoch, history, cbinfo):
    for key, val in _diagnostics(problem.domain, state).items():
        history.append(key, val)


def report_func(problem, state, epoch, cbinfo):
    d = _diagnostics(problem.domain, state)
    printlog(
        "diagnostics: "
        + ", ".join(f"{k}:{v:.5g}" for k, v in d.items())
    )


# ---------------------------------------------------------------------
# Problem construction.
# ---------------------------------------------------------------------
def make_problem(args):
    args.Nx = args.Nx or args.N
    args.Ny = args.Ny or args.N

    dtype = np.float64 if args.double else np.float32

    domain = odil.Domain(
        cshape=(args.Nx, args.Ny),
        dimnames=("x", "y"),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        dtype=dtype,
        multigrid=args.multigrid,
        mg_interp=args.mg_interp,
        mg_nlvl=args.nlvl,
    )

    printlog("ODIL backend:", os.environ.get("ODIL_BACKEND", ""))
    printlog("ODIL JIT:", os.environ.get("ODIL_JIT", ""))
    printlog("grid cshape:", domain.cshape)
    printlog("Re:", args.Re)
    printlog("scheme:", args.scheme)
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    state = odil.State()
    state.fields["u"] = odil.Field(None, loc="cc")
    state.fields["v"] = odil.Field(None, loc="cc")
    state.fields["p"] = odil.Field(None, loc="cc")
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.args = args

    problem = odil.Problem(operator_cavity, domain, extra)
    return problem, state


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    odil.setup_outdir(args)

    problem, state = make_problem(args)
    callback = odil.make_callback(
        problem,
        args,
        plot_func=plot_func,
        history_func=history_func,
        report_func=report_func,
    )

    odil.util.optimize(args, args.optimizer, problem, state, callback)
    plot_func(problem, state, epoch=0, frame=None, cbinfo=None)

    with open("done", "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
