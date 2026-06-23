#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 8 style reproduction: measure of flow complexity with ODIL + JAX.

This script follows the Figure 7 reconstruction idea, but repeats the
reconstruction while changing the number of velocity measurement points K.
For each K, several random sets of K points are sampled. The minimum error
among those random trials estimates E_min(K). The flow complexity is then
K_min(eps): the smallest K for which E_min(K) < eps.

Typical workflow:

# 1) You already need a cavity reference produced by lid_cavity_forward_odil_jax.py.
#    For example:
python lid_cavity_forward_odil_jax.py \
  --N 64 \
  --Re 3200 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --scheme upwind \
  --lr 5e-4 \
  --epochs 80000 \
  --plot_every 10000 \
  --report_every 1000 \
  --dump_data 1 \
  --outdir out_cavity_re3200_64

# 2) Run a quick debug version first.
python lid_cavity_flow_complexity_figure8_odil_jax.py \
  --N 64 \
  --cavity_ref_pickle "/Users/hanshuozhang/Desktop/ODIL_literature/Lid-driven cavity Forward/out_cavity_re3200/data_Re3200_final.pickle" \
  --flows uniform,couette,poiseuille,cavity \
  --K_list 1,2,4,8,16,29,40,50 \
  --nsamples 2 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --scheme upwind \
  --lr 1e-3 \
  --epochs 3000 \
  --kreg 1e-3 \
  --eps 0.05 \
  --outdir out_figure8_debug

# 3) Paper-like, but much slower. Increase nsamples/epochs gradually.
python lid_cavity_flow_complexity_figure8_odil_jax.py \
  --N 64 \
  --cavity_ref_pickle "/Users/hanshuozhang/Desktop/ODIL_literature/Lid-driven cavity Forward/out_cavity_re3200/data_Re3200_final.pickle" \
  --flows uniform,couette,poiseuille,cavity \
  --K_list 1,2,4,6,8,10,12,16,20,24,29,35,40,45,50 \
  --nsamples 20 \
  --optimizer adam \
  --multigrid 1 \
  --mg_interp conv \
  --scheme upwind \
  --lr 1e-3 \
  --epochs 10000 \
  --kreg 1e-3 \
  --eps 0.05 \
  --outdir out_figure8_complexity

Notes:
- The original paper used a 64x64 grid and Newton's method with a second-derivative
  regularizer kreg=1e-3 for this complexity experiment. This script keeps the same
  inverse-problem structure but defaults are chosen to be practical on a laptop.
- The cavity case needs a reference pickle. Uniform/Couette/Poiseuille are generated
  analytically and normalized to maximum speed 1.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

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
# Basic array utilities.
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


def _shift_array_for_stencil(a, ox, oy):
    """
    Return b[i,j] = a[i+ox,j+oy] for interior cells.
    ctx.field(key, -1, 0) means q[i-1,j], so ox=-1 corresponds to np.roll(...,+1).
    Boundary entries are later overwritten by linear extrapolation in the operator.
    """
    return np.roll(a, shift=(-ox, -oy), axis=(0, 1))


def _smooth_from_sparse(mask, values, iters):
    """Cheap Jacobi smoothing used only for the initial guess."""
    q = np.zeros_like(values, dtype=float)
    q[mask] = values[mask]
    for _ in range(int(max(0, iters))):
        qn = 0.25 * (
            np.roll(q, 1, axis=0)
            + np.roll(q, -1, axis=0)
            + np.roll(q, 1, axis=1)
            + np.roll(q, -1, axis=1)
        )
        qn[0, :] = 2.0 * qn[1, :] - qn[2, :]
        qn[-1, :] = 2.0 * qn[-2, :] - qn[-3, :]
        qn[:, 0] = 2.0 * qn[:, 1] - qn[:, 2]
        qn[:, -1] = 2.0 * qn[:, -2] - qn[:, -3]
        q = np.where(mask, values, qn)
    return q


# ---------------------------------------------------------------------
# Reference flows.
# ---------------------------------------------------------------------
def _normalize_speed(u, v):
    smax = float(np.max(np.sqrt(u**2 + v**2)))
    if smax > 0:
        u = u / smax
        v = v / smax
    return u, v


def _load_cavity_reference(path, nx, ny):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cavity reference pickle not found: {path}\n"
            "Run lid_cavity_forward_odil_jax.py first and pass its data_Re*_final.pickle "
            "through --cavity_ref_pickle."
        )
    with open(path, "rb") as f:
        ref0 = pickle.load(f)
    if "u" not in ref0 or "v" not in ref0:
        raise KeyError(f"{path} must contain arrays named 'u' and 'v'.")

    u0 = np.asarray(ref0["u"], dtype=float)
    v0 = np.asarray(ref0["v"], dtype=float)
    rx = np.asarray(ref0.get("x", _grid_from_shape(*u0.shape)[0]), dtype=float)
    ry = np.asarray(ref0.get("y", _grid_from_shape(*u0.shape)[1]), dtype=float)
    x, y = _grid_from_shape(nx, ny)
    u = _interp2_from_ref(rx, ry, u0, x, y)
    v = _interp2_from_ref(rx, ry, v0, x, y)
    # Do not normalize the cavity field; keep the reference from the forward solve.
    dx, dy = 1.0 / nx, 1.0 / ny
    return dict(flow="cavity", x=x, y=y, u=u, v=v, omega=_vorticity_np(u, v, dx, dy), source=str(path))


def _make_reference(flow, nx, ny, cavity_ref_pickle=None):
    x, y = _grid_from_shape(nx, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if flow == "uniform":
        u = np.ones((nx, ny), dtype=float)
        v = np.zeros((nx, ny), dtype=float)
    elif flow == "couette":
        # Linear shear flow; maximum speed is normalized to 1.
        u = Y.copy()
        v = np.zeros((nx, ny), dtype=float)
    elif flow == "poiseuille":
        # Parabolic channel profile; maximum speed is 1.
        u = 4.0 * Y * (1.0 - Y)
        v = np.zeros((nx, ny), dtype=float)
    elif flow == "cavity":
        return _load_cavity_reference(cavity_ref_pickle, nx, ny)
    else:
        raise ValueError(f"Unknown flow: {flow}")

    u, v = _normalize_speed(u, v)
    dx, dy = 1.0 / nx, 1.0 / ny
    return dict(flow=flow, x=x, y=y, u=u, v=v, omega=_vorticity_np(u, v, dx, dy), source="analytic")


# ---------------------------------------------------------------------
# Observation package.
# ---------------------------------------------------------------------
def _choose_observation_cells(nx, ny, k, rng, margin):
    margin = int(max(0, margin))
    if 2 * margin >= nx or 2 * margin >= ny:
        raise ValueError("--obs_margin is too large for this grid.")

    ii, jj = np.meshgrid(
        np.arange(margin, nx - margin),
        np.arange(margin, ny - margin),
        indexing="ij",
    )
    candidates = np.stack([ii.ravel(), jj.ravel()], axis=1)
    if k > len(candidates):
        raise ValueError(f"Requested K={k}, but only {len(candidates)} candidates are available.")
    pick = rng.choice(len(candidates), size=k, replace=False)
    return candidates[pick, 0], candidates[pick, 1]


def _build_observation_package(args, ref, k, sample_seed):
    nx, ny = args.Nx, args.Ny
    rng = np.random.default_rng(sample_seed)
    ii, jj = _choose_observation_cells(nx, ny, k, rng, args.obs_margin)

    mask = np.zeros((nx, ny), dtype=bool)
    mask[ii, jj] = True

    u_obs = np.zeros((nx, ny), dtype=float)
    v_obs = np.zeros((nx, ny), dtype=float)
    u_obs[mask] = ref["u"][mask]
    v_obs[mask] = ref["v"][mask]

    shifts = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)]
    mask_shift = {}
    uobs_shift = {}
    vobs_shift = {}
    for ox, oy in shifts:
        mask_shift[(ox, oy)] = _shift_array_for_stencil(mask.astype(float), ox, oy)
        uobs_shift[(ox, oy)] = _shift_array_for_stencil(u_obs, ox, oy)
        vobs_shift[(ox, oy)] = _shift_array_for_stencil(v_obs, ox, oy)

    if args.init_mode == "obs_smooth":
        u_init = _smooth_from_sparse(mask, u_obs, args.init_smooth_iters)
        v_init = _smooth_from_sparse(mask, v_obs, args.init_smooth_iters)
    elif args.init_mode == "zero":
        u_init = np.zeros((nx, ny), dtype=float)
        v_init = np.zeros((nx, ny), dtype=float)
        u_init[mask] = u_obs[mask]
        v_init[mask] = v_obs[mask]
    else:
        raise ValueError(args.init_mode)

    return dict(
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
# ODIL residual operator.
# ---------------------------------------------------------------------
def operator_flow_reconstruction(ctx):
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
        q, qxm, qxp, qym, qyp = st
        qxm_bc = mod.where(ix == 0, q + (q - qxp), qxm)
        qxp_bc = mod.where(ix == nx - 1, q + (q - qxm), qxp)
        qym_bc = mod.where(iy == 0, q + (q - qyp), qym)
        qyp_bc = mod.where(iy == ny - 1, q + (q - qym), qyp)
        return q, qxm_bc, qxp_bc, qym_bc, qyp_bc

    def apply_pressure_neumann(st):
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

    u_st = apply_linear_extrapolate(stencil("u"))
    v_st = apply_linear_extrapolate(stencil("v"))
    p_st = apply_pressure_neumann(stencil("p"))

    u = u_st[0]
    v = v_st[0]
    p = p_st[0]

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

    if args.kp:
        res.append(("p_gauge", args.kp * p))

    # Same kind of second-derivative regularizer as the paper's flow-complexity example.
    if args.kreg:
        res.extend([
            ("reg_uxx", args.kreg * (u_st[2] - 2.0 * u_st[0] + u_st[1]) / dx**2),
            ("reg_uyy", args.kreg * (u_st[4] - 2.0 * u_st[0] + u_st[3]) / dy**2),
            ("reg_vxx", args.kreg * (v_st[2] - 2.0 * v_st[0] + v_st[1]) / dx**2),
            ("reg_vyy", args.kreg * (v_st[4] - 2.0 * v_st[0] + v_st[3]) / dy**2),
        ])

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
# Problem construction and solve.
# ---------------------------------------------------------------------
def _state_arrays(domain, state, extra):
    u_raw = np.array(domain.field(state, "u"))
    v_raw = np.array(domain.field(state, "v"))
    p = np.array(domain.field(state, "p"))

    mask = extra.obs_mask > 0.5
    u = u_raw.copy()
    v = v_raw.copy()
    u[mask] = extra.u_obs[mask]
    v[mask] = extra.v_obs[mask]
    return u, v, p


def _reconstruction_error(u, v, ref):
    # Paper-style error: mean absolute difference summed over both velocity components.
    return float(np.mean(np.abs(u - ref["u"]) + np.abs(v - ref["v"])))


def _make_problem(args, ref, obs):
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

    state = odil.State()
    state.fields["u"] = odil.Field(obs["u_init"].astype(dtype), loc="cc")
    state.fields["v"] = odil.Field(obs["v_init"].astype(dtype), loc="cc")
    state.fields["p"] = odil.Field(np.zeros((args.Nx, args.Ny), dtype=dtype), loc="cc")
    state = domain.init_state(state)

    extra = SimpleNamespace()
    extra.args = args
    extra.ref = ref
    extra.obs_mask = obs["obs_mask"].astype(dtype)
    extra.u_obs = obs["u_obs"].astype(dtype)
    extra.v_obs = obs["v_obs"].astype(dtype)
    extra.obs_i = obs["obs_i"]
    extra.obs_j = obs["obs_j"]
    extra.mask_shift = {k: v.astype(dtype) for k, v in obs["mask_shift"].items()}
    extra.uobs_shift = {k: v.astype(dtype) for k, v in obs["uobs_shift"].items()}
    extra.vobs_shift = {k: v.astype(dtype) for k, v in obs["vobs_shift"].items()}

    problem = odil.Problem(operator_flow_reconstruction, domain, extra)
    return problem, state


def _run_one_reconstruction(args, ref, k, sample_index):
    sample_seed = int(args.seed + 100000 * args.flow_index + 1000 * sample_index + k)
    obs = _build_observation_package(args, ref, k, sample_seed)
    problem, state = _make_problem(args, ref, obs)

    # Keep the repeated sampling run quiet. odil.util.optimize accepts a callback;
    # the varargs noop is robust to callback signature differences.
    def noop_callback(*_a, **_kw):
        return None

    odil.util.optimize(args, args.optimizer, problem, state, noop_callback)
    u, v, p = _state_arrays(problem.domain, state, problem.extra)
    err = _reconstruction_error(u, v, ref)
    return err


# ---------------------------------------------------------------------
# Results I/O and plotting.
# ---------------------------------------------------------------------
def _parse_int_list(s):
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_flows(s):
    flows = []
    for part in str(s).split(','):
        part = part.strip().lower()
        if part:
            flows.append(part)
    valid = {"uniform", "couette", "poiseuille", "cavity"}
    bad = [f for f in flows if f not in valid]
    if bad:
        raise ValueError(f"Unknown flow(s): {bad}. Valid: {sorted(valid)}")
    return flows


def _read_existing(csv_path):
    rows = []
    if not Path(csv_path).exists():
        return rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(
                flow=row["flow"],
                K=int(row["K"]),
                sample=int(row["sample"]),
                error=float(row["error"]),
            ))
    return rows


def _append_row(csv_path, row):
    exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["flow", "K", "sample", "error"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _plot_figure8(rows, flows, k_list, eps, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
    axes = axes.ravel()

    pretty = {
        "uniform": "uniform",
        "couette": "couette",
        "poiseuille": "poiseuille",
        "cavity": "cavity",
    }

    for ax, flow in zip(axes, flows):
        sub = [r for r in rows if r["flow"] == flow]
        if not sub:
            ax.set_title(f"{pretty[flow]}: no data")
            continue

        # Scatter all random trials in light gray.
        for r in sub:
            ax.plot(r["K"], max(r["error"], 1e-16), '.', color='0.75', ms=3.0, alpha=0.55)

        min_err = []
        used_k = []
        for k in k_list:
            vals = [r["error"] for r in sub if r["K"] == k]
            if vals:
                used_k.append(k)
                min_err.append(min(vals))
        used_k = np.asarray(used_k, dtype=int)
        min_err = np.asarray(min_err, dtype=float)

        ax.plot(used_k, np.maximum(min_err, 1e-16), '-', color='tab:red', lw=1.6, label=r"estimated $E_{min}(K)$")
        ax.axhline(eps, color='k', ls='--', lw=0.8, alpha=0.7)

        hit = used_k[min_err < eps] if len(used_k) else []
        if len(hit):
            kmin = int(hit[0])
            ax.axvline(kmin, color='k', ls='--', lw=0.8, alpha=0.7)
            title = rf"{pretty[flow]}, $K_{{min}}({eps:g})={kmin}$"
        else:
            title = rf"{pretty[flow]}, $K_{{min}}({eps:g})$ not reached"

        ax.set_title(title)
        ax.set_xlabel("points K")
        ax.set_ylabel("error")
        ax.set_yscale("log")
        ax.set_xlim(min(k_list) - 1, max(k_list) + 1)
        ax.grid(True, which="both", lw=0.3, alpha=0.35)

    # Hide unused panels if fewer than four flows are selected.
    for ax in axes[len(flows):]:
        ax.axis("off")

    fig.suptitle("Measure of flow complexity: reconstruction error vs number of measurements")
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Arguments and main.
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--backend", choices=["jax", "tf"], default=_PRE.backend)
    parser.add_argument("--jit", type=int, default=_PRE.jit)
    parser.add_argument("--gpu", type=int, default=_PRE.gpu)

    parser.add_argument("--N", type=int, default=64, help="Square grid size")
    parser.add_argument("--Nx", type=int, default=None, help="Grid size in x; defaults to N")
    parser.add_argument("--Ny", type=int, default=None, help="Grid size in y; defaults to N")
    parser.add_argument("--Re", type=float, default=3200.0, help="Reynolds number used in the reconstruction PDE")
    parser.add_argument("--cavity_ref_pickle", type=str, default="", help="Forward reference pickle for the cavity case")
    parser.add_argument("--flows", type=str, default="uniform,couette,poiseuille,cavity", help="Comma-separated flow list")
    parser.add_argument("--K_list", type=str, default="1,2,4,6,8,10,12,16,20,24,29,35,40,45,50", help="Comma-separated K values")
    parser.add_argument("--nsamples", type=int, default=8, help="Random point sets per K")
    parser.add_argument("--eps", type=float, default=0.05, help="Accuracy threshold for K_min")
    parser.add_argument("--obs_margin", type=int, default=3, help="Exclude this many cells near walls when sampling observations")
    parser.add_argument("--resume", type=int, default=1, help="Resume from figure8_results.csv if present")

    parser.add_argument("--init_mode", choices=["zero", "obs_smooth"], default="obs_smooth")
    parser.add_argument("--init_smooth_iters", type=int, default=800, help="Jacobi smoothing iterations for initial guess")

    parser.add_argument("--scheme", choices=["upwind", "central"], default="upwind", help="Convective derivative discretization")
    parser.add_argument("--kcont", type=float, default=1.0, help="Continuity residual weight")
    parser.add_argument("--kmom", type=float, default=1.0, help="Momentum residual weight")
    parser.add_argument("--kp", type=float, default=1e-5, help="Small pressure damping / gauge weight")
    parser.add_argument("--kreg", type=float, default=1e-3, help="Second-derivative velocity regularization weight")
    parser.add_argument("--mgloss", type=int, default=0, help="Add coarse-level residuals to the loss")

    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir="out_figure8_flow_complexity_odil_jax")
    parser.set_defaults(frames=0)
    parser.set_defaults(plot_every=0)
    parser.set_defaults(report_every=0)
    parser.set_defaults(history_every=0)
    parser.set_defaults(history_full=0)
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


def main():
    args = parse_args()
    args.Nx = args.Nx or args.N
    args.Ny = args.Ny or args.N

    # Convert path before odil.setup_outdir changes the current working directory.
    if args.cavity_ref_pickle:
        args.cavity_ref_pickle = os.path.abspath(os.path.expanduser(args.cavity_ref_pickle))

    flows = _parse_flows(args.flows)
    k_list = _parse_int_list(args.K_list)
    if not k_list:
        raise ValueError("--K_list is empty")
    if "cavity" in flows and not args.cavity_ref_pickle:
        raise ValueError("The cavity case requires --cavity_ref_pickle")

    odil.setup_outdir(args)

    printlog("ODIL backend:", os.environ.get("ODIL_BACKEND", ""))
    printlog("ODIL JIT:", os.environ.get("ODIL_JIT", ""))
    printlog("grid:", (args.Nx, args.Ny))
    printlog("flows:", flows)
    printlog("K_list:", k_list)
    printlog("nsamples:", args.nsamples)
    printlog("eps:", args.eps)
    printlog("optimizer:", args.optimizer, "epochs:", args.epochs, "lr:", args.lr)
    printlog("kreg:", args.kreg)
    if args.cavity_ref_pickle:
        printlog("cavity reference:", args.cavity_ref_pickle)

    csv_path = "figure8_results.csv"
    rows = _read_existing(csv_path) if args.resume else []
    done = {(r["flow"], r["K"], r["sample"]) for r in rows}

    refs = {flow: _make_reference(flow, args.Nx, args.Ny, args.cavity_ref_pickle) for flow in flows}

    for flow_index, flow in enumerate(flows):
        args.flow_index = flow_index
        ref = refs[flow]
        printlog("\n=== flow:", flow, "source:", ref.get("source", ""), "===")
        for k in k_list:
            for sample in range(args.nsamples):
                key = (flow, k, sample)
                if key in done:
                    continue
                printlog(f"flow={flow:10s} K={k:3d} sample={sample+1:3d}/{args.nsamples}")
                err = _run_one_reconstruction(args, ref, k, sample)
                row = {"flow": flow, "K": int(k), "sample": int(sample), "error": float(err)}
                rows.append(row)
                done.add(key)
                _append_row(csv_path, row)
                printlog(f"  error={err:.6e}")

                # Update the figure after each sample, so a partial run is still useful.
                _plot_figure8(rows, flows, k_list, args.eps, f"figure8_flow_complexity.{args.plotext}")

    _plot_figure8(rows, flows, k_list, args.eps, f"figure8_flow_complexity_final.{args.plotext}")
    with open("figure8_refs.pickle", "wb") as f:
        pickle.dump(refs, f)
    with open("done", "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
