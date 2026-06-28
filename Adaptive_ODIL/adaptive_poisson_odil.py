#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D Poisson equation with ODIL + JAX, written in the same style as
velocity_tracer.py.

Problem:
    u_xx + u_yy = f(x, y),  (x,y) in [0,1]^2
    u = 0 on boundary

Reference solution:
    u(x,y) = sin(pi * (k*x)^2) * sin(pi*y)

Recommended macOS run:
    python Poisson_odil_jax.py \
      --N 64 \
      --k 4 \
      --rhs exact \
      --optimizer adam \
      --multigrid 1 \
      --mg_interp conv \
      --lr 0.005 \
      --epochs 1000 \
      --plot_every 100 \
      --report_every 100 \
      --outdir out_poisson_odil_jax

Closer-to-source run:
    python Poisson_odil_jax.py --N 64 --k 4 --rhs exact --double 1

Notes:
    - This version uses the official odil package instead of the standalone
      PyTorch optimizer.
    - The backend is selected before importing odil, just like velocity_tracer.py.
    - The unknown field u is stored as an ODIL cell-centered field with loc='cc'.
    - Multigrid decomposition is ODIL's built-in over-parameterization, enabled
      with --multigrid 1.
    - Adaptive-ODIL mode balances the fine-grid Poisson residual and optional
      multigrid residuals. Use --mgloss 1 or larger for a nontrivial adaptive
      Poisson comparison.

python3 adaptive_poisson_odil.py --adaptive 0 --mgloss 1 --outdir out_poisson_fixed
python3 adaptive_poisson_odil.py --adaptive 1 --mgloss 1 --outdir out_poisson_adaptive
"""

import argparse
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))


# ----------------------------------------------------------------------
# ODIL selects backend when importing odil, so set JAX before import odil.
# This follows the same pattern as velocity_tracer.py.
# ----------------------------------------------------------------------
def _preparse_backend(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--backend",
        choices=["jax", "tf"],
        default=os.environ.get("ODIL_BACKEND", "jax"),
    )
    p.add_argument(
        "--jit",
        type=int,
        default=int(os.environ.get("ODIL_JIT", "0")),
    )
    p.add_argument("--gpu", type=int, default=0)
    a, _ = p.parse_known_args(argv)

    os.environ["ODIL_BACKEND"] = a.backend
    os.environ["ODIL_JIT"] = str(a.jit)

    # On Mac, keep this disabled by default. JAX will use CPU.
    if not a.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    # Avoid excessive threading on macOS laptops.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    return a


_PRE = _preparse_backend(sys.argv[1:])

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import odil
from odil import printlog


ADAPT_GROUPS = {
    "pde": ("fu",),
    "mg": ("fu_mg*",),
}


def add_adaptive_arguments(parser):
    parser.add_argument("--adaptive", type=int, default=0, help="Enable adaptive residual weighting")
    parser.add_argument("--adapt_every", type=int, default=100, help="Epoch interval for weight updates")
    parser.add_argument("--adapt_alpha", type=float, default=0.5, help="Weight update exponent")
    parser.add_argument("--adapt_eps", type=float, default=1e-12, help="Small number for safe division")
    parser.add_argument("--adapt_min", type=float, default=1e-4, help="Minimum adaptive weight")
    parser.add_argument("--adapt_max", type=float, default=1e4, help="Maximum adaptive weight")
    parser.add_argument("--kmg", type=float, default=1.0, help="Initial weight for multigrid residuals")


def init_weight_tracers(args, weights):
    args.adapt_weight_names = tuple(weights.keys())
    for name, value in weights.items():
        setattr(args, "w_" + name, float(value))
    return {"w_" + name: float(value) for name, value in weights.items()}


def get_weight(ctx, name):
    return ctx.tracers["w_" + name]


def _name_matches(name, patterns):
    for pattern in patterns:
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True
        if name == pattern:
            return True
    return False


def _group_norm_from_pinfo(pinfo, patterns):
    if not pinfo or "names" not in pinfo or "norms" not in pinfo:
        return None
    vals = []
    for name, norm in zip(pinfo["names"], pinfo["norms"]):
        if _name_matches(name, patterns):
            vals.append(float(np.array(norm)))
    if not vals:
        return None
    return float(np.sqrt(np.mean(np.square(vals))))


def adaptive_epoch_func(problem, state, epoch, cbinfo):
    del state
    args = cbinfo.args
    if not args.adaptive or epoch == 0 or not args.adapt_every:
        return
    if epoch % args.adapt_every:
        return

    group_norms = {}
    for weight_name, patterns in ADAPT_GROUPS.items():
        value = _group_norm_from_pinfo(cbinfo.pinfo, patterns)
        if value is not None and value > args.adapt_eps:
            group_norms[weight_name] = value
    if len(group_norms) < 2:
        return

    target = float(np.exp(np.mean(np.log(list(group_norms.values())))))
    updates = []
    for name, norm in group_norms.items():
        old = float(getattr(args, "w_" + name))
        new = old * (target / norm) ** args.adapt_alpha
        new = float(np.clip(new, args.adapt_min, args.adapt_max))
        setattr(args, "w_" + name, new)
        problem.tracers["w_" + name] = new
        updates.append("{}:{:.4g}->{:.4g}".format(name, old, new))
    printlog("adaptive weights: " + ", ".join(updates))


# ----------------------------------------------------------------------
# Reference solution and right-hand side.
# ----------------------------------------------------------------------
def get_ref_u(x, y, k):
    """u(x,y) = sin(pi*(k*x)^2) * sin(pi*y)."""
    pi = np.pi
    return np.sin(pi * (k * x) ** 2) * np.sin(pi * y)


def get_ref_rhs_exact(x, y, k):
    """Continuous RHS f = u_xx + u_yy for the manufactured solution."""
    pi = np.pi
    return (
        (
            (-4.0 * k**4 * pi**2 * x**2 - pi**2) * np.sin(k**2 * pi * x**2)
            + 2.0 * k**2 * pi * np.cos(k**2 * pi * x**2)
        )
        * np.sin(pi * y)
    )


# ----------------------------------------------------------------------
# Finite-difference helpers. These mirror the official ODIL Poisson example.
# ----------------------------------------------------------------------
def split_wm_wp(st, dirs):
    """
    Split a compact stencil list into center, minus-neighbor and plus-neighbor.

    For 2D with dirs=(0,1), st contains:
        [q, qxm, qxp, qym, qyp]
    """
    q = st[0]
    qwm = [st[2 * i + 1] for i in dirs]
    qwp = [st[2 * i + 2] for i in dirs]
    return q, qwm, qwp


def apply_bc_zero_dirichlet(st, iw, nw, dirs, mod):
    """
    Apply zero Dirichlet boundary conditions through ODIL's quadratic
    half-cell ghost extrapolation.

    For boundary value b=0:
        ghost = (inner_neighbor - 6*current + 8*b) / 3
    """
    q, qwm, qwp = split_wm_wp(st, dirs)
    zero = mod.cast(0, q.dtype)

    for i in dirs:
        extrap = odil.core.extrap_quadh
        qm = mod.where(iw[i] == 0, extrap(qwp[i], q, zero), qwm[i])
        qp = mod.where(iw[i] == nw[i] - 1, extrap(qwm[i], q, zero), qwp[i])
        qwm[i], qwp[i] = qm, qp

    for i in dirs:
        st[2 * i + 1] = qwm[i]
        st[2 * i + 2] = qwp[i]


def get_discrete_rhs(ref_u, domain, mod):
    """
    Discrete RHS computed by applying the same ODIL finite-difference
    Laplacian to the reference solution. With --rhs discrete, the reference
    solution satisfies the discrete system up to optimizer tolerance.
    """
    ndim = domain.ndim
    dirs = range(ndim)
    dw = domain.step()
    iw = domain.indices()
    nw = domain.size()

    u_st = [None] * (2 * ndim + 1)
    u_st[0] = ref_u
    for i in dirs:
        u_st[2 * i + 1] = mod.roll(ref_u, 1, i)
        u_st[2 * i + 2] = mod.roll(ref_u, -1, i)

    apply_bc_zero_dirichlet(u_st, iw, nw, dirs, mod)
    u, uwm, uwp = split_wm_wp(u_st, dirs)
    u_ww = [(uwp[i] - 2.0 * u + uwm[i]) / dw[i] ** 2 for i in dirs]
    return sum(u_ww)


# ----------------------------------------------------------------------
# ODIL residual operator.
# ----------------------------------------------------------------------
def operator_poisson(ctx):
    domain = ctx.domain
    extra = ctx.extra
    args = extra.args
    mod = ctx.mod

    ndim = domain.ndim
    dirs = range(ndim)
    dw = ctx.step()
    iw = ctx.indices()
    nw = ctx.size()

    def stencil_var(key):
        st = [ctx.field(key)]
        for i in dirs:
            shift = [-1 if j == i else 0 for j in dirs]
            st.append(ctx.field(key, *shift))
            shift = [1 if j == i else 0 for j in dirs]
            st.append(ctx.field(key, *shift))
        return st

    u_st = stencil_var("u")
    apply_bc_zero_dirichlet(u_st, iw, nw, dirs, mod=mod)

    u, uwm, uwp = split_wm_wp(u_st, dirs)
    u_ww = [(uwp[i] - 2.0 * u + uwm[i]) / dw[i] ** 2 for i in dirs]
    fu = sum(u_ww) - extra.rhs

    # Same return style as velocity_tracer.py: named residuals.
    res = [("fu", fu * get_weight(ctx, "pde"))]

    # Optional multigrid loss terms, useful for forcing coarse-scale residuals.
    if args.mgloss:
        from functools import partial

        restrict = partial(
            odil.core.restrict_to_coarser,
            loc="c" * ndim,
            mod=mod,
        )
        fuc = fu
        for level in range(args.mgloss):
            fuc = restrict(fuc)
            res.append(("fu_mg{}".format(level + 1), fuc * get_weight(ctx, "mg")))

    return res


# ----------------------------------------------------------------------
# Arguments.
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--backend", choices=["jax", "tf"], default=_PRE.backend)
    parser.add_argument("--jit", type=int, default=_PRE.jit)
    parser.add_argument("--gpu", type=int, default=_PRE.gpu)

    parser.add_argument("--N", type=int, default=64, help="Grid size, N x N")
    parser.add_argument(
        "--k",
        "--osc_k",
        dest="k",
        type=float,
        default=4.0,
        help="Oscillation parameter in u=sin(pi*(k*x)^2)sin(pi*y)",
    )
    parser.add_argument(
        "--rhs",
        type=str,
        default="exact",
        choices=("exact", "discrete"),
        help="Use continuous manufactured RHS or discrete RHS from reference u",
    )
    parser.add_argument(
        "--mgloss",
        type=int,
        default=0,
        help="Add this many coarser residuals to the loss",
    )
    add_adaptive_arguments(parser)

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_poisson_odil_jax")
    parser.set_defaults(frames=5)
    parser.set_defaults(epochs=1000)
    parser.set_defaults(plot_every=100)
    parser.set_defaults(report_every=100)
    parser.set_defaults(history_every=10)
    parser.set_defaults(history_full=10)
    parser.set_defaults(plotext="png")
    parser.set_defaults(plot_title=1)
    parser.set_defaults(dump_data=1)

    # Source-like defaults for the Poisson example.
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.005)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(mg_interp="conv")
    parser.set_defaults(double=1)

    # Only relevant for --optimizer newton.
    parser.set_defaults(linsolver="multigrid")
    parser.set_defaults(linsolver_maxiter=10)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------
def _to_numpy(x):
    return np.array(x)


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    args = extra.args

    suff = "final" if frame is None else "{:05d}".format(frame)
    field_path = "poisson_field_{}.{}".format(suff, args.plotext)
    hist_path = "poisson_history_{}.{}".format(suff, args.plotext)
    data_path = "poisson_data_{}.pickle".format(suff)

    u = _to_numpy(domain.field(state, "u"))
    ref_u = _to_numpy(extra.ref_u)
    rhs = _to_numpy(extra.rhs)
    err = u - ref_u

    if args.dump_data:
        with open(data_path, "wb") as f:
            pickle.dump(
                dict(
                    lower=domain.lower,
                    upper=domain.upper,
                    cshape=domain.cshape,
                    u=u,
                    ref_u=ref_u,
                    rhs=rhs,
                    error=err,
                    k=args.k,
                    rhs_mode=args.rhs,
                ),
                f,
            )

    title_prefix = "epoch={:05d}, ".format(epoch) if args.plot_title else ""

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    items = [
        (ref_u, "reference u"),
        (u, "ODIL/JAX u"),
        (err, "u - reference"),
        (rhs, "rhs f"),
    ]

    for ax, (arr, title) in zip(axes, items):
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_title(title_prefix + title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(field_path, dpi=200)
    plt.close(fig)

    # If ODIL has already written a history file, leave plotting of history to ODIL.
    # The name is printed here to keep output format similar to velocity_tracer.py.
    printlog(field_path, data_path)


# ----------------------------------------------------------------------
# History and report.
# ----------------------------------------------------------------------
def get_errors(domain, extra, state):
    u = np.array(domain.field(state, "u"))
    ref_u = np.array(extra.ref_u)
    rmse = np.sqrt(np.mean((u - ref_u) ** 2))
    ref_rms = np.sqrt(np.mean(ref_u**2))
    rel_rmse = rmse / ref_rms

    # Residual RMSE is useful when --rhs exact is used: discretization error means
    # the exact continuous reference is not an exact zero-loss solution.
    try:
        mod = domain.mod
        res = np.array(get_discrete_rhs(u, domain, mod) - extra.rhs)
        res_rmse = np.sqrt(np.mean(res**2))
    except Exception:
        res_rmse = np.nan

    return rmse, rel_rmse, res_rmse


def history_func(problem, state, epoch, history, cbinfo):
    rmse, rel_rmse, res_rmse = get_errors(problem.domain, problem.extra, state)
    history.append("error_u", rmse)
    history.append("rel_error_u", rel_rmse)
    history.append("residual_u", res_rmse)
    args = problem.extra.args
    for name in args.adapt_weight_names:
        history.append("weight_" + name, getattr(args, "w_" + name))


def report_func(problem, state, epoch, cbinfo):
    rmse, rel_rmse, res_rmse = get_errors(problem.domain, problem.extra, state)
    printlog(
        "error: rmse={:.6e}, rel_rmse={:.6e}, residual_rmse={:.6e}".format(
            rmse, rel_rmse, res_rmse
        )
    )


# ----------------------------------------------------------------------
# Problem construction.
# ----------------------------------------------------------------------
def make_problem(args):
    dtype = np.float64 if args.double else np.float32

    domain = odil.Domain(
        cshape=(args.N, args.N),
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
    printlog("field shape cc:", domain.get_field_shape(loc="cc"))

    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    x, y = domain.points("x", "y", loc="cc")
    ref_u = get_ref_u(x, y, args.k).astype(dtype)

    if args.rhs == "discrete":
        rhs = get_discrete_rhs(ref_u, domain, domain.mod)
    else:
        rhs = get_ref_rhs_exact(x, y, args.k).astype(dtype)

    state = odil.State()
    state.fields["u"] = odil.Field(None, loc="cc")
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.args = args
    extra.ref_u = ref_u
    extra.rhs = rhs

    tracers = init_weight_tracers(args, {"pde": 1.0, "mg": args.kmg})
    problem = odil.Problem(operator_poisson, domain, extra, tracers=tracers)
    return problem, state


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    odil.setup_outdir(args)

    problem, state = make_problem(args)

    callback = odil.make_callback(
        problem,
        args,
        epoch_func=adaptive_epoch_func,
        plot_func=plot_func,
        history_func=history_func,
        report_func=report_func,
    )

    odil.util.optimize(args, args.optimizer, problem, state, callback)

    # Save one final figure after optimization finishes.
    plot_func(problem, state, 0, None, None)

    with open("done", "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
