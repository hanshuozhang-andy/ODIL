#!/usr/bin/env python3
"""
ODIL-library implementation of the heat-conductivity inverse problem.

This follows the official ODIL examples/heat/heat.py structure, but keeps only
ODIL grid solver, not the PINN branch.

Clean inverse Newton:
    python heat_odil_library_like_source.py \
        --outdir out_odiln --Nt 64 --Nx 64 \
        --ref_path ref/ref.pickle \
        --infer_k 1 --imposed stripe \
        --optimizer newton --multigrid 0 --kwreg 1 \
        --report_every 5 --history_every 1 --plot_every 10

Clean inverse Adam + multigrid:
    python heat_odil_library_like_source.py \
        --outdir out_odil --Nt 64 --Nx 64 \
        --ref_path ref/ref.pickle \
        --infer_k 1 --imposed stripe --every_factor 2

Noisy Fig. 5 style:
    add --noise 0.1
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import odil
from odil import history, printlog


def get_init_u(t, x, mod):
    """U(x)=g(x)-g(0), g(x)=exp(-50(x-0.5)^2)."""
    del t

    def g(z):
        return mod.exp(-((z - 0.5) ** 2) * 50)

    return g(x) - g(-mod.cast(0.5, x.dtype))


def get_ref_k(u, mod=np):
    """Reference conductivity k(u)=0.02 exp(-20(u-0.5)^2)."""
    return 0.02 * mod.exp(-((u - 0.5) ** 2) * 20)


def transform_k(knet, mod, kmax):
    """k = kmax * sigmoid(q), same positivity transform as source."""
    return mod.sigmoid(knet) * kmax


def get_anneal_factor(epoch, period):
    return 0.5 ** (epoch / period) if period else 1


# ----------------------------------------------------------------------
# ODIL residual operator
# ----------------------------------------------------------------------

def operator_odil(ctx):
    extra = ctx.extra
    args = extra.args
    mod = ctx.mod

    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()
    epoch = ctx.tracers["epoch"]

    def stencil_var(key, frozen=False):
        if not args.keep_frozen:
            frozen = False
        return [
            [
                ctx.field(key, 0, 0, frozen=frozen),
                ctx.field(key, 0, -1, frozen=frozen),
                ctx.field(key, 0, 1, frozen=frozen),
            ],
            [
                ctx.field(key, -1, 0, frozen=frozen),
                ctx.field(key, -1, -1, frozen=frozen),
                ctx.field(key, -1, 1, frozen=frozen),
            ],
        ]

    def apply_bc_u(st):
        # Initial condition through temporal halo.
        if args.keep_init:
            u0 = extra.init_u
            q0 = [u0, mod.roll(u0, 1, axis=0), mod.roll(u0, -1, axis=0)]
            extrap = odil.core.extrap_linear
            q, qm = st
            for i in range(3):
                qm[i] = mod.where(it == 0, extrap(q[i], q0[i][None, :]), qm[i])

        # Zero Dirichlet boundary through spatial halo.
        extrap = odil.core.extrap_quadh
        for q in st:
            q[1] = mod.where(ix == 0, extrap(q[2], q[0], 0), q[1])
            q[2] = mod.where(ix == nx - 1, extrap(q[1], q[0], 0), q[2])
        return st

    u_st = stencil_var("u")
    apply_bc_u(u_st)
    q, qm = u_st

    u_t = (q[0] - qm[0]) / dt

    # Crank-Nicolson midpoint gradients at left/right faces.
    u_xm = ((q[0] + qm[0]) - (q[1] + qm[1])) / (2 * dx)
    u_xp = ((q[2] + qm[2]) - (q[0] + qm[0])) / (2 * dx)

    # Frozen field for evaluating k(u), as in the official source.
    uf_st = stencil_var("u", frozen=True)
    apply_bc_u(uf_st)
    qf, qfm = uf_st

    ufxmh = ((qf[0] + qfm[0]) + (qf[1] + qfm[1])) * 0.25
    ufxph = ((qf[2] + qfm[2]) + (qf[0] + qfm[0])) * 0.25

    if args.infer_k:
        km = transform_k(ctx.neural_net("k_net")(ufxmh)[0], mod, args.kmax)
        kp = transform_k(ctx.neural_net("k_net")(ufxph)[0], mod, args.kmax)
    else:
        km = get_ref_k(ufxmh, mod=mod)
        kp = get_ref_k(ufxph, mod=mod)

    flux_m = u_xm * km
    flux_p = u_xp * kp
    q_x = (flux_p - flux_m) / dx

    fu = u_t - q_x
    if not args.keep_init:
        fu = mod.where(it == 0, ctx.cast(0), fu)

    res = [("fu", fu)]

    # Observation/data loss at imposed points.
    if extra.imp_size:
        scale = args.kimp * (np.prod(ctx.size()) / extra.imp_size) ** 0.5
        fuimp = extra.imp_mask * (u_st[0][0] - extra.imp_u) * scale
        res.append(("imp", fuimp))

    # Optional regularization.
    if args.kxreg:
        scale = args.kxreg * get_anneal_factor(epoch, args.kxregdecay)
        u_x = (u_st[0][0] - u_st[0][1]) / dx
        u_x = mod.where(ix == 0, ctx.cast(0), u_x)
        res.append(("xreg", u_x * scale))

    if args.ktreg:
        scale = args.ktreg * get_anneal_factor(epoch, args.ktregdecay)
        u_t_reg = (u_st[0][0] - u_st[1][0]) / dt
        u_t_reg = mod.where(it == 0, ctx.cast(0), u_t_reg)
        res.append(("treg", u_t_reg * scale))

    # Newton damping for k_net weights. This matches source --kwreg behavior.
    if args.kwreg and args.infer_k:
        domain = ctx.domain
        ww = domain.arrays_from_field(ctx.state.fields["k_net"])
        ww = mod.concatenate([mod.flatten(w) for w in ww], axis=0)
        scale = args.kwreg * get_anneal_factor(epoch, args.kwregdecay)
        res.append(("wreg", (mod.stop_gradient(ww) - ww) * scale))

    return res


# ----------------------------------------------------------------------
# Imposed points
# ----------------------------------------------------------------------

def get_imposed_indices(domain, args, iflat):
    iflat = np.array(iflat)
    rng = np.random.default_rng(args.seed)

    if args.imposed == "random":
        imp_i = iflat.flatten()
        nimp = min(args.nimp, imp_i.size)
        imp_i = rng.permutation(imp_i)[:nimp]

    elif args.imposed == "stripe":
        imp_i = iflat.flatten()
        t = np.array(domain.points("t")).flatten()
        imp_i = imp_i[np.abs(t[imp_i] - 0.5) < 1 / 6]
        nimp = min(args.nimp, imp_i.size)
        imp_i = rng.permutation(imp_i)[:nimp]

    elif args.imposed == "none":
        imp_i = []

    else:
        raise ValueError("Unknown imposed=" + args.imposed)

    return imp_i


def get_imposed_mask(args, domain):
    mod = domain.mod
    size = np.prod(domain.cshape)
    iflat = np.reshape(np.arange(size), domain.cshape)

    imp_i = np.unique(get_imposed_indices(domain, args, iflat))

    mask = np.zeros(size)
    if len(imp_i):
        mask[imp_i] = 1
        points = [mod.flatten(domain.points(i)) for i in range(domain.ndim)]
        points = np.array(points)[:, imp_i].T
    else:
        points = np.zeros((0, domain.ndim))

    return mask.reshape(domain.cshape), points, imp_i


# ----------------------------------------------------------------------
# Plotting and diagnostics
# ----------------------------------------------------------------------

def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d

    del cbinfo

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title_u = "u epoch={:}".format(epoch) if args.plot_title else None
    title_k = "k epoch={:}".format(epoch) if args.plot_title else None

    path_u = "u_{:05d}.{}".format(frame, args.plotext)
    path_k = "k_{:05d}.{}".format(frame, args.plotext)
    printlog(path_u, path_k)

    state_u = np.array(domain.field(state, "u"))

    def callback(i, fig, ax, data, extent):
        del fig, data, extent
        if i == 0 and len(extra.imp_points):
            imp_t, imp_x = extra.imp_points.T
            ax.scatter(
                imp_x,
                imp_t,
                s=0.5,
                alpha=1,
                edgecolor="none",
                facecolor="k",
                zorder=100,
            )

    plot_1d(
        domain,
        np.array(extra.imp_u),
        state_u,
        path=path_u,
        title=title_u,
        cmap="YlOrBr",
        nslices=5,
        interpolation="bilinear",
        callback=callback,
        transpose=True,
        umin=0,
        umax=1,
    )

    fig, ax = plt.subplots(figsize=(1.7, 1.5))
    ref_uk = extra.ref_uk
    ref_k = extra.ref_k

    k = None
    if args.infer_k:
        (k_raw,) = domain.neural_net(state, "k_net")(ref_uk)
        k = transform_k(k_raw, mod, args.kmax)
        ax.plot(ref_uk, k, zorder=10)

    ax.plot(ref_uk, ref_k, lw=1.5, zorder=1)
    ax.set_xlabel("u")
    ax.set_ylabel("k")
    ax.set_ylim(0, 0.03)
    ax.set_title(title_k)
    fig.savefig(path_k, bbox_inches="tight")
    plt.close(fig)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        data = {
            "state_u": state_u,
            "ref_u": extra.ref_u,
            "imp_u": extra.imp_u,
            "ref_uk": ref_uk,
            "k": k,
            "ref_k": ref_k,
            "imp_indices": extra.imp_indices,
            "imp_points": extra.imp_points,
        }
        data = odil.core.struct_to_numpy(mod, data)
        with open(path, "wb") as f:
            pickle.dump(data, f)


def get_error(domain, extra, state, key):
    args = extra.args
    mod = domain.mod

    if key == "u":
        state_u = domain.field(state, "u")
        return np.sqrt(np.mean((np.array(state_u) - np.array(extra.ref_u)) ** 2))

    if key == "k" and args.infer_k:
        (k_raw,) = domain.neural_net(state, "k_net")(extra.ref_uk)
        k = transform_k(k_raw, mod, args.kmax)
        return np.sqrt(np.mean((np.array(k) - extra.ref_k) ** 2)) / extra.ref_k.max()

    return None


def history_func(problem, state, epoch, hist, cbinfo):
    del epoch, cbinfo
    for key in ["u", "k"]:
        error = get_error(problem.domain, problem.extra, state, key)
        if error is not None:
            hist.append("error_" + key, error)


def report_func(problem, state, epoch, cbinfo):
    del epoch, cbinfo
    items = []
    for key in ["u", "k"]:
        error = get_error(problem.domain, problem.extra, state, key)
        if error is not None:
            items.append("{}:{:.5g}".format(key, error))
    if items:
        printlog("error: " + ", ".join(items))


# ----------------------------------------------------------------------
# Reference loading
# ----------------------------------------------------------------------

def load_fields_interp(path, keys, domain):
    """Load fields from checkpoint and interpolate them to current domain."""
    from scipy.interpolate import RectBivariateSpline

    src_state = odil.State(fields={key: odil.Field() for key in keys})
    state = odil.State(fields={key: odil.Field() for key in keys})

    odil.core.checkpoint_load(domain, src_state, path)
    t1, x1 = domain.points_1d()

    for key in keys:
        src_u = src_state.fields[key]
        src_domain = odil.Domain(
            cshape=src_u.array.shape,
            dimnames=("t", "x"),
            lower=domain.lower,
            upper=domain.upper,
            dtype=domain.dtype,
            mod=odil.backend.ModNumpy(),
        )
        src_u = src_domain.init_field(src_u)

        if src_domain.cshape != domain.cshape:
            src_t1, src_x1 = src_domain.points_1d()
            fu = RectBivariateSpline(src_t1, src_x1, src_u.array)
            state.fields[key].array = fu(t1, x1)
        else:
            state.fields[key] = src_u

    return state


# ----------------------------------------------------------------------
# Problem construction
# ----------------------------------------------------------------------

def make_problem(args):
    dtype = np.float64 if args.double else np.float32

    domain = odil.Domain(
        cshape=(args.Nt, args.Nx),
        dimnames=("t", "x"),
        multigrid=args.multigrid,
        dtype=dtype,
    )

    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    mod = domain.mod
    tt, xx = domain.points()
    _, x1 = domain.points_1d()

    init_u = get_init_u(x1 * 0, x1, mod)

    if args.ref_path is not None:
        printlog("Loading reference solution from '{}'".format(args.ref_path))
        ref_state = load_fields_interp(args.ref_path, ["u"], domain)
        ref_u = domain.cast(ref_state.fields["u"].array)
    else:
        ref_u = get_init_u(tt, xx, mod)

    imp_u_np = np.array(ref_u, copy=True)
    if args.noise:
        rng = np.random.default_rng(args.seed)
        imp_u_np = imp_u_np + rng.normal(loc=0, scale=args.noise, size=imp_u_np.shape)
    imp_u = domain.cast(imp_u_np)

    imp_mask, imp_points, imp_indices = get_imposed_mask(args, domain)
    imp_mask = domain.cast(imp_mask)

    with open("imposed.csv", "w") as f:
        f.write(",".join(domain.dimnames) + "\n")
        for p in imp_points:
            f.write("{:},{:}\n".format(*p))

    ref_uk = np.linspace(0, 1, 200).astype(domain.dtype)
    ref_k = get_ref_k(ref_uk)

    extra = argparse.Namespace(
        args=args,
        ref_u=ref_u,
        ref_uk=ref_uk,
        ref_k=ref_k,
        init_u=init_u,
        imp_mask=imp_mask,
        imp_size=len(imp_points),
        imp_u=imp_u,
        imp_indices=imp_indices,
        imp_points=imp_points,
    )

    state = odil.State()
    state.fields["u"] = np.zeros(domain.cshape)

    if args.infer_k:
        state.fields["k_net"] = domain.make_neural_net([1] + args.arch_k + [1])

    state = domain.init_state(state)
    problem = odil.Problem(operator_odil, domain, extra)

    if args.checkpoint is not None:
        printlog("Loading checkpoint '{}'".format(args.checkpoint))
        odil.core.checkpoint_load(domain, state, args.checkpoint)
        train_path = os.path.splitext(args.checkpoint)[0] + "_train.pickle"
        if args.checkpoint_train is None and os.path.isfile(train_path):
            args.checkpoint_train = train_path

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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--Nt", type=int, default=64, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--arch_k", type=int, nargs="*", default=[5, 5])
    parser.add_argument("--infer_k", type=int, default=0)
    parser.add_argument("--kmax", type=float, default=0.1)

    parser.add_argument("--kxreg", type=float, default=0)
    parser.add_argument("--kxregdecay", type=float, default=0)
    parser.add_argument("--ktreg", type=float, default=0)
    parser.add_argument("--ktregdecay", type=float, default=0)
    parser.add_argument("--kwreg", type=float, default=0)
    parser.add_argument("--kwregdecay", type=float, default=0)

    parser.add_argument("--kimp", type=float, default=2)
    parser.add_argument("--keep_frozen", type=int, default=1)
    parser.add_argument("--keep_init", type=int, default=1)

    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--imposed", choices=["random", "stripe", "none"], default="none")
    parser.add_argument("--nimp", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0)

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_heat")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=0)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(
        plot_every=2000,
        report_every=500,
        history_full=10,
        history_every=100,
        frames=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    odil.setup_outdir(args, relpath_args=["checkpoint", "checkpoint_train", "ref_path"])

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
