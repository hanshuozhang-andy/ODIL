#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Typical workflow:

# 1) Generate a reference solution with your forward code.
python lid_cavity_forward_odil_jax.py \
  --N 128 \
  --Re 3200 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --scheme upwind \
  --lr 5e-4 \
  --epochs 80000 \
  --plot_every 10000 \
  --report_every 1000 \
  --outdir out_cavity_re3200

# 2) Reconstruct from 100 sampled velocity points.
python lid_cavity_reconstruction_figure7_odil_jax_high_contrast.py \
  --N 128 \
  --Re 3200 \
  --ref_pickle "/Users/hanshuozhang/Desktop/ODIL_literature/Lid-driven cavity Forward/out_cavity_re3200/data_Re3200_final.pickle" \
  --ndata 100 \
  --seed 1 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --scheme upwind \
  --lr 1e-3 \
  --epochs 50000 \
  --plot_every 5000 \
  --report_every 1000 \
  --outdir out_cavity_reconstruction_fig7
  
For a faster debug run:
python lid_cavity_reconstruction_figure7_odil_jax.py \
  --N 64 --Re 3200 --ref_pickle out_cavity_re3200/data_Re3200_final.pickle \
  --ndata 100 --epochs 5000 --plot_every 1000 --outdir out_debug_fig7
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------
# ODIL selects the backend when importing odil. Set JAX first.
# ---------------------------------------------------------------------
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
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Use GPU if available. On Apple Silicon, keep 0 unless configured.",
    )
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
# Reference data and observation handling.
# ---------------------------------------------------------------------
def _grid_from_shape(nx, ny):
    x = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    y = np.linspace(0.5 / ny, 1.0 - 0.5 / ny, ny)
    return x, y


def _interp2_from_ref(x_ref, y_ref, q_ref, x_new, y_new):
    """Simple tensor-product linear interpolation for cell-centered arrays."""
    q_ref = np.asarray(q_ref)
    if q_ref.shape == (len(x_new), len(y_new)) and len(x_ref) == len(x_new) and len(y_ref) == len(y_new):
        if np.allclose(x_ref, x_new) and np.allclose(y_ref, y_new):
            return q_ref.copy()

    # Interpolate in y for every original x, then in x for every new y.
    tmp = np.empty((len(x_ref), len(y_new)), dtype=float)
    for i in range(len(x_ref)):
        tmp[i, :] = np.interp(y_new, y_ref, q_ref[i, :])

    out = np.empty((len(x_new), len(y_new)), dtype=float)
    for j in range(len(y_new)):
        out[:, j] = np.interp(x_new, x_ref, tmp[:, j])
    return out


def _vorticity_np(u, v, dx, dy):
    dv_dx = np.gradient(v, dx, axis=0, edge_order=1)
    du_dy = np.gradient(u, dy, axis=1, edge_order=1)
    return dv_dx - du_dy


def _divergence_np(u, v, dx, dy):
    du_dx = np.gradient(u, dx, axis=0, edge_order=1)
    dv_dy = np.gradient(v, dy, axis=1, edge_order=1)
    return du_dx + dv_dy


def _load_reference(path, nx, ny):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference pickle not found: {path}\n"
            "Run lid_cavity_forward_odil_jax.py first, then pass its data_Re*_final.pickle "
            "through --ref_pickle."
        )

    with open(path, "rb") as f:
        ref = pickle.load(f)

    if "u" not in ref or "v" not in ref:
        raise KeyError(f"{path} must contain arrays named 'u' and 'v'.")

    u0 = np.asarray(ref["u"], dtype=float)
    v0 = np.asarray(ref["v"], dtype=float)
    rx = np.asarray(ref.get("x", _grid_from_shape(*u0.shape)[0]), dtype=float)
    ry = np.asarray(ref.get("y", _grid_from_shape(*u0.shape)[1]), dtype=float)

    x, y = _grid_from_shape(nx, ny)
    u = _interp2_from_ref(rx, ry, u0, x, y)
    v = _interp2_from_ref(rx, ry, v0, x, y)

    dx = 1.0 / nx
    dy = 1.0 / ny
    omega = _vorticity_np(u, v, dx, dy)
    return dict(x=x, y=y, u=u, v=v, omega=omega, source=str(path), raw=ref)


def _choose_observation_cells(nx, ny, ndata, seed, margin):
    rng = np.random.default_rng(seed)
    margin = int(max(0, margin))
    if 2 * margin >= nx or 2 * margin >= ny:
        raise ValueError("--obs_margin is too large for this grid.")

    ii, jj = np.meshgrid(
        np.arange(margin, nx - margin),
        np.arange(margin, ny - margin),
        indexing="ij",
    )
    candidates = np.stack([ii.ravel(), jj.ravel()], axis=1)
    if ndata > len(candidates):
        raise ValueError(f"Requested {ndata} points, but only {len(candidates)} candidates are available.")
    pick = rng.choice(len(candidates), size=ndata, replace=False)
    return candidates[pick, 0], candidates[pick, 1]


def _smooth_from_sparse(mask, values, iters):
    """Cheap Laplace/Jacobi smoothing used only for the initial guess."""
    q = np.zeros_like(values, dtype=float)
    q[mask] = values[mask]
    for _ in range(int(max(0, iters))):
        qn = 0.25 * (
            np.roll(q, 1, axis=0)
            + np.roll(q, -1, axis=0)
            + np.roll(q, 1, axis=1)
            + np.roll(q, -1, axis=1)
        )
        # Linear-extrapolate-like boundary update to avoid periodic initial artifacts.
        qn[0, :] = 2.0 * qn[1, :] - qn[2, :]
        qn[-1, :] = 2.0 * qn[-2, :] - qn[-3, :]
        qn[:, 0] = 2.0 * qn[:, 1] - qn[:, 2]
        qn[:, -1] = 2.0 * qn[:, -2] - qn[:, -3]
        q = np.where(mask, values, qn)
    return q


def _shift_array_for_stencil(a, ox, oy):
    """
    Return b[i,j] = a[i+ox,j+oy] for interior cells.

    ctx.field(key, -1, 0) means q[i-1,j], so ox=-1 corresponds to np.roll(...,+1).
    Boundary entries are overwritten by linear extrapolation in the operator.
    """
    return np.roll(a, shift=(-ox, -oy), axis=(0, 1))


def _build_observation_package(args):
    nx, ny = args.Nx, args.Ny
    ref = _load_reference(args.ref_pickle, nx, ny)

    mask = np.zeros((nx, ny), dtype=bool)
    ii, jj = _choose_observation_cells(nx, ny, args.ndata, args.seed, args.obs_margin)
    mask[ii, jj] = True

    u_obs = np.zeros((nx, ny), dtype=float)
    v_obs = np.zeros((nx, ny), dtype=float)
    u_obs[mask] = ref["u"][mask]
    v_obs[mask] = ref["v"][mask]

    # Precompute shifted observation arrays for reparameterized stencils.
    shifts = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)]
    mask_shift = {}
    uobs_shift = {}
    vobs_shift = {}
    for ox, oy in shifts:
        mask_shift[(ox, oy)] = _shift_array_for_stencil(mask.astype(float), ox, oy)
        uobs_shift[(ox, oy)] = _shift_array_for_stencil(u_obs, ox, oy)
        vobs_shift[(ox, oy)] = _shift_array_for_stencil(v_obs, ox, oy)

    if args.init_from_ref:
        u_init = ref["u"].copy()
        v_init = ref["v"].copy()
    elif args.init_mode == "obs_smooth":
        u_init = _smooth_from_sparse(mask, u_obs, args.init_smooth_iters)
        v_init = _smooth_from_sparse(mask, v_obs, args.init_smooth_iters)
    else:
        u_init = np.zeros((nx, ny), dtype=float)
        v_init = np.zeros((nx, ny), dtype=float)
        u_init[mask] = u_obs[mask]
        v_init[mask] = v_obs[mask]

    return dict(
        ref=ref,
        obs_mask=mask.astype(float),
        u_obs=u_obs,
        v_obs=v_obs,
        obs_i=ii,
        obs_j=jj,
        mask_shift=mask_shift,
        uobs_shift=uobs_shift,
        vobs_shift=vobs_shift,
        u_init=u_init,
        v_init=v_init,
    )


# ---------------------------------------------------------------------
# ODIL residual operator for reconstruction.
# ---------------------------------------------------------------------
def operator_cavity_reconstruction(ctx):
    mod = ctx.mod
    extra = ctx.extra
    args = extra.args

    dx, dy = ctx.step()
    ix, iy = ctx.indices(loc="cc")
    nx, ny = ctx.size()

    nu = ctx.cast(1.0 / args.Re)

    def carray(a):
        return ctx.cast(a)

    def reparam_value(key, raw, ox, oy):
        """Replace raw velocity value by observed value where the shifted cell is measured."""
        if args.impose != "reparam":
            return raw
        if key == "u":
            mask = carray(extra.mask_shift[(ox, oy)])
            data = carray(extra.uobs_shift[(ox, oy)])
        elif key == "v":
            mask = carray(extra.mask_shift[(ox, oy)])
            data = carray(extra.vobs_shift[(ox, oy)])
        else:
            return raw
        return mod.where(mask > 0.5, data, raw)

    def stencil(key, frozen=False):
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        out = []
        for ox, oy in offsets:
            raw = ctx.field(key, ox, oy, frozen=frozen)
            out.append(reparam_value(key, raw, ox, oy))
        return out

    def apply_linear_extrapolate(st):
        """
        Linear extrapolation at the square walls, used instead of no-slip velocity BCs.
        For example, left ghost value is q[-1,j] = 2 q[0,j] - q[1,j].
        """
        q, qxm, qxp, qym, qyp = st
        qxm_bc = mod.where(ix == 0, q + (q - qxp), qxm)
        qxp_bc = mod.where(ix == nx - 1, q + (q - qxm), qxp)
        qym_bc = mod.where(iy == 0, q + (q - qyp), qym)
        qyp_bc = mod.where(iy == ny - 1, q + (q - qym), qyp)
        return q, qxm_bc, qxp_bc, qym_bc, qyp_bc

    def apply_pressure_neumann(st):
        """Pressure gauge handled separately; use zero-normal-gradient ghost cells."""
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

    # Unknown velocities use linear extrapolation at the boundary.
    u_st = apply_linear_extrapolate(stencil("u"))
    v_st = apply_linear_extrapolate(stencil("v"))
    p_st = apply_pressure_neumann(stencil("p"))

    u = u_st[0]
    v = v_st[0]
    p = p_st[0]

    # Frozen, reparameterized velocities only choose the upwind branch.
    uf = apply_linear_extrapolate(stencil("u", frozen=True))[0]
    vf = apply_linear_extrapolate(stencil("v", frozen=True))[0]

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

    f_cont = ux + vy
    f_mom_u = u * ux_adv + v * uy_adv + px - nu * laplace(u_st)
    f_mom_v = u * vx_adv + v * vy_adv + py - nu * laplace(v_st)

    res = [
        ("cont", args.kcont * f_cont),
        ("mom_u", args.kmom * f_mom_u),
        ("mom_v", args.kmom * f_mom_v),
    ]

    # Penalty mode is useful for debugging.  The paper-style default is reparam.
    if args.impose == "penalty":
        mask = carray(extra.obs_mask)
        u_obs = carray(extra.u_obs)
        v_obs = carray(extra.v_obs)
        w = ctx.cast(np.sqrt(args.kdata))
        res.append(("data_u", w * mask * (u - u_obs)))
        res.append(("data_v", w * mask * (v - v_obs)))

    # Pressure gauge: fixes arbitrary additive constant.
    if args.kp:
        res.append(("p_gauge", args.kp * p))

    # Optional smoothness regularization.  The flow-complexity example uses a
    # second-derivative regularizer; it can also stabilize sparse reconstruction.
    if args.kreg:
        res.extend([
            ("reg_uxx", args.kreg * (u_st[2] - 2.0 * u_st[0] + u_st[1]) / dx**2),
            ("reg_uyy", args.kreg * (u_st[4] - 2.0 * u_st[0] + u_st[3]) / dy**2),
            ("reg_vxx", args.kreg * (v_st[2] - 2.0 * v_st[0] + v_st[1]) / dx**2),
            ("reg_vyy", args.kreg * (v_st[4] - 2.0 * v_st[0] + v_st[3]) / dy**2),
        ])

    # Optional coarse residuals, helpful with Adam and multigrid decomposition.
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

    parser.add_argument("--N", type=int, default=128, help="Square grid size")
    parser.add_argument("--Nx", type=int, default=None, help="Grid size in x; defaults to N")
    parser.add_argument("--Ny", type=int, default=None, help="Grid size in y; defaults to N")
    parser.add_argument("--Re", type=float, default=3200.0, help="Reynolds number")
    parser.add_argument("--ref_pickle", type=str, required=True, help="Forward reference pickle, e.g. data_Re3200_final.pickle")

    parser.add_argument("--ndata", type=int, default=100, help="Number of velocity measurement points")
    parser.add_argument("--obs_margin", type=int, default=3, help="Exclude this many cells near walls when sampling observations")
    parser.add_argument("--impose", choices=["reparam", "penalty"], default="reparam", help="How to impose velocity measurements")
    parser.add_argument("--kdata", type=float, default=1e4, help="Penalty weight used only when --impose penalty")

    parser.add_argument("--init_mode", choices=["zero", "obs_smooth"], default="obs_smooth")
    parser.add_argument("--init_smooth_iters", type=int, default=1500, help="Jacobi smoothing iterations for initial guess")
    parser.add_argument("--init_from_ref", type=int, default=0, help="Debug only: initialize from reference field")

    parser.add_argument("--scheme", choices=["upwind", "central"], default="upwind", help="Convective derivative discretization")
    parser.add_argument("--kcont", type=float, default=1.0, help="Continuity residual weight")
    parser.add_argument("--kmom", type=float, default=1.0, help="Momentum residual weight")
    parser.add_argument("--kp", type=float, default=1e-5, help="Small pressure damping / gauge weight")
    parser.add_argument("--kreg", type=float, default=0.0, help="Second-derivative velocity regularization weight")
    parser.add_argument("--mgloss", type=int, default=0, help="Add coarse-level residuals to the loss")

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_lid_cavity_reconstruction_fig7_odil_jax")
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
# Plotting and diagnostics.
# ---------------------------------------------------------------------
def _state_arrays(domain, state, extra):
    u_raw = np.array(domain.field(state, "u"))
    v_raw = np.array(domain.field(state, "v"))
    p = np.array(domain.field(state, "p"))

    if extra.args.impose == "reparam":
        mask = extra.obs_mask > 0.5
        u = u_raw.copy()
        v = v_raw.copy()
        u[mask] = extra.u_obs[mask]
        v[mask] = extra.v_obs[mask]
    else:
        u = u_raw
        v = v_raw

    return u, v, p


def _error_metrics(u, v, ref):
    ur = ref["u"]
    vr = ref["v"]
    rms = np.sqrt(np.mean((u - ur) ** 2 + (v - vr) ** 2))
    denom = np.sqrt(np.mean(ur**2 + vr**2)) + 1e-30
    rel_rms = rms / denom
    mean_abs = np.mean(np.abs(u - ur) + np.abs(v - vr))
    return float(rms), float(rel_rms), float(mean_abs)


def _plot_field_pair(
    ax_inf,
    ax_ref,
    x,
    y,
    q,
    qref,
    obs_i,
    obs_j,
    title,
    cmap="PuOr_r",
    vmin=None,
    vmax=None,
):
    """Plot inferred/reference fields with stronger, zero-centered contrast."""
    if vmin is None or vmax is None:
        vmax_auto = max(float(np.max(np.abs(q))), float(np.max(np.abs(qref))), 1e-12)
        vmin, vmax = -vmax_auto, vmax_auto

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    q_plot = np.clip(q, vmin, vmax)
    qref_plot = np.clip(qref, vmin, vmax)

    im0 = ax_inf.imshow(
        q_plot.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax_ref.imshow(
        qref_plot.T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax_inf.plot(x[obs_i], y[obs_j], "k.", ms=1.6, alpha=0.95)
    ax_ref.plot(x[obs_i], y[obs_j], "k.", ms=1.6, alpha=0.75)
    ax_inf.set_title(f"inferred {title}")
    ax_ref.set_title(f"reference {title}")
    for ax in (ax_inf, ax_ref):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    return im0


def _plot_profiles(ax, x, y, q, qref, label):
    # Figure 7 has multiple side profiles; here they are stacked with vertical offsets.
    levels = [0.15, 0.35, 0.55, 0.75]
    offsets = np.arange(len(levels))[::-1] * 1.4
    scale = max(np.max(np.abs(q)), np.max(np.abs(qref)), 1e-12)
    for yy, off in zip(levels, offsets):
        j = int(np.argmin(np.abs(y - yy)))
        ax.plot(x, q[:, j] / scale + off, color="tab:red", lw=1.1)
        ax.plot(x, qref[:, j] / scale + off, color="k", lw=0.9)
        ax.text(1.02, off, f"y={y[j]:.2f}", va="center", fontsize=7)
    ax.set_title(f"{label} profiles")
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-1.0, offsets[0] + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    args = extra.args

    u, v, p = _state_arrays(domain, state, extra)
    nx, ny = domain.cshape
    dx, dy = domain.step()
    x, y = _grid_from_shape(nx, ny)
    omega = _vorticity_np(u, v, dx, dy)
    ref = extra.ref

    suff = "final" if frame is None else f"{frame:05d}"
    path = f"figure7_reconstruction_Re{args.Re:g}_{suff}.{args.plotext}"
    printlog(path)

    fig = plt.figure(figsize=(11.8, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.0, 1.0, 0.12, 0.85),
        height_ratios=(1, 1, 1),
    )

    # Fixed color limits make the visual contrast consistent with the paper-style
    # Figure 7 panels. Values outside these ranges are clipped to the end colors.
    rows = [
        (u, ref["u"], "u", "velocity u", -0.8, 0.8, [-0.8, -0.4, 0.0, 0.4, 0.8]),
        (v, ref["v"], "v", "velocity v", -0.5, 0.5, [-0.5, -0.25, 0.0, 0.25, 0.5]),
        (omega, ref["omega"], r"$\omega$", "vorticity", -10.0, 10.0, [-10, -5, 0, 5, 10]),
    ]

    for r, (q, qref, short, title, vmin, vmax, ticks) in enumerate(rows):
        ax0 = fig.add_subplot(gs[r, 0])
        ax1 = fig.add_subplot(gs[r, 1])
        cax = fig.add_subplot(gs[r, 2])
        axp = fig.add_subplot(gs[r, 3])
        im = _plot_field_pair(
            ax0,
            ax1,
            x,
            y,
            q,
            qref,
            extra.obs_i,
            extra.obs_j,
            title,
            cmap="PuOr_r",
            vmin=vmin,
            vmax=vmax,
        )
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=8)
        _plot_profiles(axp, x, y, q, qref, short)

    rms, rel_rms, mean_abs = _error_metrics(u, v, ref)
    if args.plot_title:
        fig.suptitle(
            f"Lid-driven cavity reconstruction, Re={args.Re:g}, "
            f"Ndata={args.ndata}, epoch={epoch}, rel_rms={rel_rms:.3e}, mean_abs={mean_abs:.3e}"
        )

    fig.savefig(path, dpi=180)
    plt.close(fig)

    if args.dump_data:
        with open(f"data_reconstruction_Re{args.Re:g}_{suff}.pickle", "wb") as f:
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
                    div=_divergence_np(u, v, dx, dy),
                    ref_u=ref["u"],
                    ref_v=ref["v"],
                    ref_omega=ref["omega"],
                    obs_i=extra.obs_i,
                    obs_j=extra.obs_j,
                    obs_x=x[extra.obs_i],
                    obs_y=y[extra.obs_j],
                    obs_u=extra.u_obs[extra.obs_mask > 0.5],
                    obs_v=extra.v_obs[extra.obs_mask > 0.5],
                    rel_rms=rel_rms,
                    mean_abs=mean_abs,
                ),
                f,
            )


def _diagnostics(domain, state, extra):
    u, v, p = _state_arrays(domain, state, extra)
    dx, dy = domain.step()
    div = _divergence_np(u, v, dx, dy)
    speed = np.sqrt(u**2 + v**2)
    rms, rel_rms, mean_abs = _error_metrics(u, v, extra.ref)
    return dict(
        div_rms=float(np.sqrt(np.mean(div**2))),
        rel_rms=rel_rms,
        mean_abs=mean_abs,
        umax=float(np.max(u)),
        umin=float(np.min(u)),
        vmax=float(np.max(v)),
        vmin=float(np.min(v)),
        speed_max=float(np.max(speed)),
    )


def history_func(problem, state, epoch, history, cbinfo):
    for key, val in _diagnostics(problem.domain, state, problem.extra).items():
        history.append(key, val)


def report_func(problem, state, epoch, cbinfo):
    d = _diagnostics(problem.domain, state, problem.extra)
    printlog("diagnostics: " + ", ".join(f"{k}:{v:.5g}" for k, v in d.items()))


# ---------------------------------------------------------------------
# Problem construction.
# ---------------------------------------------------------------------
def make_problem(args):
    args.Nx = args.Nx or args.N
    args.Ny = args.Ny or args.N

    obs = _build_observation_package(args)

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
    printlog("reference:", obs["ref"]["source"])
    printlog("ndata:", args.ndata, "seed:", args.seed, "impose:", args.impose)
    printlog("obs_margin:", args.obs_margin)
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    state = odil.State()
    state.fields["u"] = odil.Field(obs["u_init"].astype(dtype), loc="cc")
    state.fields["v"] = odil.Field(obs["v_init"].astype(dtype), loc="cc")
    state.fields["p"] = odil.Field(np.zeros((args.Nx, args.Ny), dtype=dtype), loc="cc")
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.args = args
    extra.ref = obs["ref"]
    extra.obs_mask = obs["obs_mask"].astype(dtype)
    extra.u_obs = obs["u_obs"].astype(dtype)
    extra.v_obs = obs["v_obs"].astype(dtype)
    extra.obs_i = obs["obs_i"]
    extra.obs_j = obs["obs_j"]
    extra.mask_shift = {k: v.astype(dtype) for k, v in obs["mask_shift"].items()}
    extra.uobs_shift = {k: v.astype(dtype) for k, v in obs["uobs_shift"].items()}
    extra.vobs_shift = {k: v.astype(dtype) for k, v in obs["vobs_shift"].items()}

    problem = odil.Problem(operator_cavity_reconstruction, domain, extra)
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
