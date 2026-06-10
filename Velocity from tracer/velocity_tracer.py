#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Velocity from tracer with ODIL + JAX.
python velocity_tracer.py \
  --Nx 64 \
  --Nt 64 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --lr 0.01 \
  --kxreg 0.01 \
  --ktreg 1 \
  --kimp 10 \
  --frames 10 \
  --plot_every 100

Problem:
    Given c(x,y,0)=c0 and c(x,y,1)=c1, infer c(x,y,t) and velocity
    v=(vx,vy) from

        c_t + vx c_x + vy c_y = 0.

Recommended macOS run:
    python veltracer_odil_jax_wave_like.py --Nx 32 --Nt 32 --frames 3

Source-like run:
    python veltracer_odil_jax_wave_like.py --Nx 64 --Nt 64 --frames 10
"""

import argparse
import os
import sys


# ----------------------------------------------------------------------
# ODIL selects backend when importing odil, so set JAX before import odil.
# ----------------------------------------------------------------------
def _preparse_backend(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--backend", choices=["jax", "tf"],
                   default=os.environ.get("ODIL_BACKEND", "jax"))
    p.add_argument("--jit", type=int,
                   default=int(os.environ.get("ODIL_JIT", "0")))
    p.add_argument("--gpu", type=int, default=0)
    a, _ = p.parse_known_args(argv)

    os.environ["ODIL_BACKEND"] = a.backend
    os.environ["ODIL_JIT"] = str(a.jit)

    # On Mac, keep this disabled. JAX will use CPU.
    if not a.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    return a


_PRE = _preparse_backend(sys.argv[1:])

import pickle
import numpy as np
import matplotlib.pyplot as plt
import odil
from odil import printlog


# ----------------------------------------------------------------------
# Synthetic initial and final tracer snapshots.
# Same construction as ODIL source example.
# ----------------------------------------------------------------------
def tracer_blob(x, y, t):
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
    return res


# ----------------------------------------------------------------------
# ODIL residual operator.
# ----------------------------------------------------------------------
def operator_advection(ctx):
    mod = ctx.mod
    extra = ctx.extra
    args = extra.args

    dt, dx, dy = ctx.step()
    it, ix, iy = ctx.indices(loc="ncc")
    nt, nx, ny = ctx.size()

    def single_var(key, shift_t=0, shift_x=0, shift_y=0, frozen=False):
        return ctx.field(key, shift_t, shift_x, shift_y, frozen=frozen)

    def stencil_var(key, shift_t=0, frozen=False):
        return [
            ctx.field(key, shift_t, 0, 0, frozen=frozen),
            ctx.field(key, shift_t, -1, 0, frozen=frozen),
            ctx.field(key, shift_t, 1, 0, frozen=frozen),
            ctx.field(key, shift_t, 0, -1, frozen=frozen),
            ctx.field(key, shift_t, 0, 1, frozen=frozen),
        ]

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        return q_xx + q_yy

    def deriv_fou(qm, q, qp, v):
        """
        First-order upwind numerator.

        v > 0: backward difference q - qm
        v < 0: forward difference qp - q
        v = 0: centered difference 0.5 * (qp - qm)
        """
        return mod.where(
            v > 0,
            q - qm,
            mod.where(v < 0, qp - q, 0.5 * (qp - qm)),
        )

    # Velocity field.
    vx_st = stencil_var("vx")
    vy_st = stencil_var("vy")
    vx = vx_st[0]
    vy = vy_st[0]

    # Frozen velocity only chooses the upwind branch.
    vxf = stencil_var("vx", frozen=True)[0]
    vyf = stencil_var("vy", frozen=True)[0]

    # Tracer at previous time level.
    u_prev_st = stencil_var("u", shift_t=-1)

    # Spatial upwind derivatives of previous-time tracer.
    u_x = deriv_fou(u_prev_st[1], u_prev_st[0], u_prev_st[2], vxf) / dx
    u_y = deriv_fou(u_prev_st[3], u_prev_st[0], u_prev_st[4], vyf) / dy

    u = single_var("u")
    u_prev = u_prev_st[0]

    # At first physical step, use exact initial tracer instead of unknown u[-1].
    u_prev = mod.where(it == 1, extra.u_init[None, :, :], u_prev)

    # Advection residual.
    u_t = (u - u_prev) / dt
    fu = u_t + vx * u_x + vy * u_y

    zero = ctx.cast(0)

    # Initial tracer constraint.
    fu = mod.where(it == 0, (u - extra.u_init[None, :, :]) / dx, fu)

    # Final tracer constraint.
    fimp = mod.where(
        it == nt - 1,
        (u - extra.u_final[None, :, :]) / dx,
        zero,
    )

    res = [
        ("fu", fu),
        ("fimp", fimp * args.kimp),
    ]

    # Spatial smoothness: kxreg * Laplacian(v).
    if args.kxreg:
        res += [
            ("vx_lap", laplace(vx_st) * args.kxreg),
            ("vy_lap", laplace(vy_st) * args.kxreg),
        ]

    # Time stationarity: ktreg * v_t.
    if args.ktreg:
        k = args.ktreg / dt

        ftreg_vx = (single_var("vx") - single_var("vx", -1)) * k
        ftreg_vx = mod.where(it == 0, zero, ftreg_vx)

        ftreg_vy = (single_var("vy") - single_var("vy", -1)) * k
        ftreg_vy = mod.where(it == 0, zero, ftreg_vy)

        res += [
            ("vx_t", ftreg_vx),
            ("vy_t", ftreg_vy),
        ]

    return res


# ----------------------------------------------------------------------
# Arguments.
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--backend", choices=["jax", "tf"],
                        default=_PRE.backend)
    parser.add_argument("--jit", type=int, default=_PRE.jit)
    parser.add_argument("--gpu", type=int, default=_PRE.gpu)

    parser.add_argument("--Nt", type=int, default=None,
                        help="Grid size in t. Default: Nx")
    parser.add_argument("--Nx", type=int, default=64,
                        help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=None,
                        help="Grid size in y. Default: Nx")

    parser.add_argument("--kxreg", type=float, default=0.01,
                        help="Laplacian regularization weight")
    parser.add_argument("--ktreg", type=float, default=1.0,
                        help="Time regularization weight")
    parser.add_argument("--kimp", type=float, default=10.0,
                        help="Final tracer constraint weight")

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_veltracer_odil_jax")
    parser.set_defaults(frames=5)
    parser.set_defaults(plot_every=100)
    parser.set_defaults(report_every=100)
    parser.set_defaults(history_every=10)
    parser.set_defaults(history_full=5)
    parser.set_defaults(plotext="png")
    parser.set_defaults(plot_title=1)
    parser.set_defaults(dump_data=1)

    # Source-like defaults.
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.01)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(mg_interp="conv")

    # Only relevant for --optimizer newton.
    parser.set_defaults(linsolver="multigrid")
    parser.set_defaults(linsolver_maxiter=10)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------
def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_2d

    domain = problem.domain
    extra = problem.extra
    args = extra.args

    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "v_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1)

    slices_it = np.linspace(0, domain.cshape[0], 5, dtype=int)
    slices_t = domain.points_1d(0, loc="n")[slices_it]

    state_u = np.array(domain.field(state, "u"))
    state_vx = np.array(domain.field(state, "vx"))
    state_vy = np.array(domain.field(state, "vy"))

    if args.dump_data:
        with open("data_{:05d}.pickle".format(frame), "wb") as f:
            pickle.dump(
                dict(
                    lower=domain.lower,
                    upper=domain.upper,
                    cshape=domain.cshape,
                    slices_it=slices_it,
                    slices_t=np.array(slices_t),
                    u=state_u,
                    vx=state_vx,
                    vy=state_vy,
                    u_init=extra.u_init,
                    u_final=extra.u_final,
                    exact_uu=extra.exact_uu,
                ),
                f,
            )

    def callback_quiver(i, j, ax, fig):
        # Draw arrows only on inferred row.
        if i != 1:
            return

        xx, yy = domain.points("x", "y", loc=".cc")
        skip = max(1, domain.cshape[1] // 8)
        offset = max(0, skip // 2 - 1)

        x = np.array(xx[offset::skip, offset::skip]).flatten()
        y = np.array(yy[offset::skip, offset::skip]).flatten()

        vx = state_vx[slices_it[j], offset::skip, offset::skip].flatten()
        vy = state_vy[slices_it[j], offset::skip, offset::skip].flatten()

        ax.quiver(x, y, vx, vy, scale=5, color="k")

    title = "epoch={:05d}".format(epoch) if args.plot_title else None

    plot_2d(
        domain,
        extra.exact_uu,
        state_u,
        slices_it,
        slices_t,
        path0,
        cmap="YlOrBr",
        umin=0,
        umax=1,
        callback=callback_quiver,
        interpolation="bilinear",
        title=title,
        ylabel_exact="imposed",
        ylabel_pred="inferred",
    )

    plot_2d(
        domain,
        state_vx,
        state_vy,
        slices_it,
        slices_t,
        path1,
        umin=-0.5,
        umax=0.5,
        cmap="PuOr_r",
        interpolation="bilinear",
        title=title,
        ylabel_exact="vx",
        ylabel_pred="vy",
    )


# ----------------------------------------------------------------------
# History and report.
# ----------------------------------------------------------------------
def _end_errors(domain, extra, state):
    u = np.array(domain.field(state, "u"))
    e0 = np.sqrt(np.mean((u[0] - extra.u_init) ** 2))
    e1 = np.sqrt(np.mean((u[-1] - extra.u_final) ** 2))
    return e0, e1


def history_func(problem, state, epoch, history, cbinfo):
    e0, e1 = _end_errors(problem.domain, problem.extra, state)
    history.append("error_u_t0", e0)
    history.append("error_u_t1", e1)


def report_func(problem, state, epoch, cbinfo):
    e0, e1 = _end_errors(problem.domain, problem.extra, state)
    printlog("tracer endpoint error: t0:{:.5g}, t1:{:.5g}".format(e0, e1))


# ----------------------------------------------------------------------
# Problem construction.
# ----------------------------------------------------------------------
def make_problem(args):
    args.Nt = args.Nt or args.Nx
    args.Ny = args.Ny or args.Nx

    dtype = np.float64 if args.double else np.float32

    domain = odil.Domain(
        cshape=(args.Nt, args.Nx, args.Ny),
        dimnames=("t", "x", "y"),
        lower=(0.0, 0.0, 0.0),
        upper=(1.0, 1.0, 1.0),
        dtype=dtype,
        multigrid=args.multigrid,
        mg_interp=args.mg_interp,
        mg_nlvl=args.nlvl,
    )

    printlog("ODIL backend:", os.environ.get("ODIL_BACKEND", ""))
    printlog("ODIL JIT:", os.environ.get("ODIL_JIT", ""))
    printlog("grid cshape:", domain.cshape)
    printlog("field shape ncc:", domain.get_field_shape(loc="ncc"))

    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    x, y = domain.points("x", "y", loc=".cc")
    u_init = tracer_blob(x, y, 0.0).astype(dtype)
    u_final = tracer_blob(x, y, 1.0).astype(dtype)

    state = odil.State()

    # loc='ncc':
    # n in time, c in x, c in y.
    # So actual field has Nt+1 time nodes.
    state.fields["u"] = odil.Field(None, loc="ncc")
    state.fields["vx"] = odil.Field(None, loc="ncc")
    state.fields["vy"] = odil.Field(None, loc="ncc")

    state = domain.init_state(state)

    # Only first and final tracer snapshots are known.
    exact_uu = np.zeros(domain.get_field_shape(loc="ncc"), dtype=dtype)
    exact_uu[0] = u_init
    exact_uu[-1] = u_final

    extra = argparse.Namespace()
    extra.args = args
    extra.u_init = u_init
    extra.u_final = u_final
    extra.exact_uu = exact_uu

    problem = odil.Problem(operator_advection, domain, extra)
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
        plot_func=plot_func,
        history_func=history_func,
        report_func=report_func,
    )

    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open("done", "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()