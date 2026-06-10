#!/usr/bin/env python3
"""
This version:
  - uses ODIL library directly;
  - sets ODIL_BACKEND=jax before importing odil;
  - uses NumPy exact solution, so it does not call TensorFlow tf.Variable;
  - defaults to the paper-style wave setup:
      exact_variant = paper
      right_bc = 1
      multigrid = 0
      optimizer = newton

Newton:
    
ODIL_BACKEND=tf python wave.py \
  --optimizer newton \
  --multigrid 0 \
  --linsolver direct \
  --every_factor 0.01 \
  --Nt 25 \
  --Nx 25 \
  --outdir out_wave_newton_tf

L-BFGS-B:
    python wave.py \
        --optimizer lbfgsb \
        --multigrid 0 \
        --kimp 1 \
        --every_factor 10 \
        --Nt 25 \
        --Nx 25 \
        --outdir out_wave_lbfgsb_jax

Official-source style:
    python wave.py \
        --exact_variant source \
        --right_bc 0 \
        --optimizer newton \
        --multigrid 0 \
        --linsolver direct \
        --Nt 25 \
        --Nx 25 \
        --outdir out_wave_newton_source
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# IMPORTANT:
# Backend must be selected BEFORE importing odil.
# ----------------------------------------------------------------------
import os

os.environ.setdefault("ODIL_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
import pickle

import numpy as np

import odil
from odil import history, printlog


# ----------------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------------

def get_exact(args, t, x):
    """
    Exact/reference solution.

    paper:
        U(x,t) = 1/10 sum_{k=1}^5 [
            cos((x - t + 0.5) pi k)
          + cos((x + t + 0.5) pi k)
        ]

    source:
        follows the current official GitHub examples/wave/wave.py convention:
            cos((x - t + 0.5) pi k)
          + cos((x + t - 0.5) pi k)

    This function is deliberately NumPy-only.
    It avoids TensorFlow calls under JAX backend.
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    shape = np.broadcast_shapes(t.shape, x.shape)
    u = np.zeros(shape, dtype=float)
    ut = np.zeros(shape, dtype=float)

    for i in [1, 2, 3, 4, 5]:
        k = i * np.pi

        a = (x - t + 0.5) * k

        if args.exact_variant == "paper":
            b = (x + t + 0.5) * k
        else:
            b = (x + t - 0.5) * k

        u += np.cos(a) + np.cos(b)

        # d/dt cos((x - t + c)k) =  k sin((x - t + c)k)
        # d/dt cos((x + t + c)k) = -k sin((x + t + c)k)
        ut += k * np.sin(a) - k * np.sin(b)

    u /= 10.0
    ut /= 10.0

    return u, ut


# ----------------------------------------------------------------------
# ODIL residual operator
# ----------------------------------------------------------------------

def operator_wave(ctx):
    extra = ctx.extra
    args = extra.args
    mod = ctx.mod

    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()

    # Keeps coordinate construction consistent with the official wave example.
    ctx.points("x")

    def stencil_var(key):
        return [
            ctx.field(key),          # u_i^n
            ctx.field(key, -1, 0),   # u_i^{n-1}
            ctx.field(key, -2, 0),   # u_i^{n-2}
            ctx.field(key, -1, -1),  # u_{i-1}^{n-1}
            ctx.field(key, -1, 1),   # u_{i+1}^{n-1}
        ]

    left_utm = mod.roll(extra.left_u, 1, axis=0)
    right_utm = mod.roll(extra.right_u, 1, axis=0)

    def apply_bc_u(st):
        extrap = odil.core.extrap_quadh

        # Left boundary halo:
        # u_0^n = (u_2^n - 6u_1^n + 8U(-1,t^n)) / 3
        st[3] = mod.where(
            ix == 0,
            extrap(st[4], st[1], left_utm[:, None]),
            st[3],
        )

        # Right boundary halo:
        # u_{Nx+1}^n = (u_{Nx-1}^n - 6u_{Nx}^n + 8U(1,t^n)) / 3
        if args.right_bc:
            st[4] = mod.where(
                ix == nx - 1,
                extrap(st[3], st[1], right_utm[:, None]),
                st[4],
            )

        return st

    st = stencil_var("u")
    apply_bc_u(st)

    u, utm, utmm, uxm, uxp = st

    # Time derivatives.
    u_t_tm = (u - utm) / dt
    u_t_tmm = (utm - utmm) / dt

    # First physical time layer uses initial velocity U_t(x,0).
    u_t_tmm = mod.where(it == 1, extra.init_ut[None, :], u_t_tmm)

    u_tt = (u_t_tm - u_t_tmm) / dt

    # Space derivative at previous time layer.
    u_xx = (uxm - 2 * utm + uxp) / (dx**2)

    fu = u_tt - u_xx

    # At it == 0, impose first time layer from initial condition.
    # u(t=dt/2) ≈ U(x,0) + 0.5 dt U_t(x,0)
    u0 = extra.init_u + 0.5 * dt * extra.init_ut
    fu = mod.where(it == 0, (u - u0[None, :]) * args.kimp, fu)

    return [("fu", fu)]


# ----------------------------------------------------------------------
# Plotting and diagnostics
# ----------------------------------------------------------------------

def get_uut(domain, init_u, uu):
    from odil.core import extrap_quad, extrap_quadh

    dt = domain.step("t")

    u = uu
    utm = np.roll(u, 1, axis=0)
    utp = np.roll(u, -1, axis=0)

    utm[0, :] = extrap_quadh(utp[0, :], u[0, :], init_u)
    utp[-1, :] = extrap_quad(u[-3, :], u[-2, :], u[-1, :])

    return (utp - utm) / (2 * dt)


def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d

    del cbinfo

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title_u = "u epoch={:}".format(epoch) if args.plot_title else None
    title_ut = "ut epoch={:}".format(epoch) if args.plot_title else None

    path_u = "u_{:05d}.{}".format(frame, args.plotext)
    path_ut = "ut_{:05d}.{}".format(frame, args.plotext)
    printlog(path_u, path_ut)

    state_u = np.array(domain.field(state, "u"))
    state_ut = get_uut(domain, extra.init_u, state_u)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        data = {
            "state_u": state_u,
            "state_ut": state_ut,
            "ref_u": extra.ref_u,
            "ref_ut": extra.ref_ut,
            "lower": domain.lower,
            "upper": domain.upper,
            "cshape": domain.cshape,
        }
        data = odil.core.struct_to_numpy(mod, data)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    umax = max(abs(np.max(extra.ref_u)), abs(np.min(extra.ref_u)))

    plot_1d(
        domain,
        extra.ref_u,
        state_u,
        path=path_u,
        title=title_u,
        cmap="RdBu_r",
        nslices=5,
        transpose=True,
        umin=-umax,
        umax=umax,
    )

    umax = max(abs(np.max(extra.ref_ut)), abs(np.min(extra.ref_ut)))

    plot_1d(
        domain,
        extra.ref_ut,
        state_ut,
        path=path_ut,
        title=title_ut,
        cmap="RdBu_r",
        nslices=5,
        transpose=True,
        umin=-umax,
        umax=umax,
    )


def get_error(domain, extra, state, key):
    if key == "u":
        state_u = np.array(domain.field(state, "u"))
        return np.sqrt(np.mean((state_u - np.array(extra.ref_u)) ** 2))

    return None


def history_func(problem, state, epoch, hist, cbinfo):
    del epoch, cbinfo

    for key in ["u"]:
        error = get_error(problem.domain, problem.extra, state, key)
        if error is not None:
            hist.append("error_" + key, error)


def report_func(problem, state, epoch, cbinfo):
    del epoch, cbinfo

    items = []

    for key in ["u"]:
        error = get_error(problem.domain, problem.extra, state, key)
        if error is not None:
            items.append("{}:{:.5g}".format(key, error))

    if items:
        printlog("error: " + ", ".join(items))


# ----------------------------------------------------------------------
# Problem construction
# ----------------------------------------------------------------------

def make_problem(args):
    dtype = np.float64 if args.double else np.float32

    domain = odil.Domain(
        cshape=(args.Nt, args.Nx),
        dimnames=("t", "x"),
        lower=(0, -1),
        upper=(1, 1),
        multigrid=args.multigrid,
        dtype=dtype,
    )

    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    tt, xx = domain.points()
    t1, x1 = domain.points_1d()

    ref_u, ref_ut = get_exact(args, tt, xx)
    left_u, _ = get_exact(args, t1, t1 * 0 + domain.lower[1])
    right_u, _ = get_exact(args, t1, t1 * 0 + domain.upper[1])
    init_u, init_ut = get_exact(args, x1 * 0 + domain.lower[0], x1)

    extra = argparse.Namespace(
        args=args,
        ref_u=ref_u,
        ref_ut=ref_ut,
        left_u=left_u,
        right_u=right_u,
        init_u=init_u,
        init_ut=init_ut,
    )

    state = odil.State()
    state.fields["u"] = np.zeros(domain.cshape)
    state = domain.init_state(state)

    problem = odil.Problem(operator_wave, domain, extra)

    if args.checkpoint is not None:
        printlog("Loading checkpoint '{}'".format(args.checkpoint))
        odil.core.checkpoint_load(domain, state, args.checkpoint)

    if args.checkpoint_train is not None:
        printlog("Loading history from '{}'".format(args.checkpoint_train))
        history.load(args.checkpoint_train)
        args.epoch_start = history.get("epoch", [args.epoch_start])[-1]
        frame = history.get("frame", [args.frame_start])[-1]
        printlog("Starting from epoch={:} frame={:}".format(args.epoch_start, frame))

    return problem, state


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--Nt", type=int, default=64, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument(
        "--kimp",
        type=float,
        default=1.0,
        help="Factor to impose initial conditions",
    )

    parser.add_argument(
        "--right_bc",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 imposes both paper boundary conditions; 0 follows the current official source more closely.",
    )

    parser.add_argument(
        "--exact_variant",
        choices=["source", "paper"],
        default="paper",
        help="paper follows the paper formula; source follows current GitHub wave.py.",
    )

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    # Defaults chosen for the paper-style ODIL Newton run.
    parser.set_defaults(outdir="out_wave_jax")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="newton")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=0)

    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(
        plot_every=100,
        report_every=10,
        history_full=10,
        history_every=10,
        frames=2,
    )

    return parser.parse_args()


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

    with open("done", "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()