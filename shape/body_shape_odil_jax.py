#!/usr/bin/env python3
"""
python body_shape_odil_jax.py \
  --case all \
  --N 64 \
  --Re 60 \
  --D 0.4 \
  --lambda_penalty 10 \
  --forward_epochs 12000 \
  --epochs 20000 \
  --lr 1e-3 \
  --forward_lr 1e-3 \
  --n_data 100 \
  --report_every 200 \
  --outdir out_body_shape
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from typing import Dict, Iterable, Tuple, NamedTuple

# Enable double precision before importing jax.numpy.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Array = jax.Array
PyTree = Dict[str, Array]


class Const(NamedTuple):
    x: Array
    y: Array
    dx: float
    dy: float
    Re: float
    D: float
    lam: float
    chi_ref: Array
    phi_ref: Array
    u_ref: Array
    v_ref: Array
    p_ref: Array
    meas_i: Array
    meas_j: Array
    meas_u: Array
    meas_v: Array
    w_pde: float
    w_bc: float
    w_data: float
    w_eik: float
    w_area: float
    w_phi_anchor: float


def tree_zeros_like(tree: PyTree) -> PyTree:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_mul(a: PyTree, scalar: float) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x * scalar, a)


def tree_div(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x / y, a, b)


def tree_sqrt(a: PyTree) -> PyTree:
    return jax.tree_util.tree_map(jnp.sqrt, a)


def tree_square(a: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x * x, a)


def make_grid(N: int, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Cell-centered grid for [0,2] x [0,1] with shape (2N, N)."""
    nx, ny = 2 * N, N
    dx, dy = 2.0 / nx, 1.0 / ny
    x1 = (np.arange(nx, dtype=dtype) + 0.5) * dx
    y1 = (np.arange(ny, dtype=dtype) + 0.5) * dy
    x, y = np.meshgrid(x1, y1, indexing="ij")
    return x, y, dx, dy


def phi_body(case: str, x: np.ndarray, y: np.ndarray, D: float) -> np.ndarray:
    """Signed-distance-like phi: positive inside the body, negative outside."""
    cx, cy = 0.62, 0.50
    if case == "circle":
        r = 0.5 * D
        return r - np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12)
    if case == "ellipse":
        # Approximate signed distance to an ellipse. Good enough for a level set.
        a, b = 0.30, 0.14
        q = np.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 + 1e-12)
        return min(a, b) * (1.0 - q)
    if case == "nonconvex":
        # A concave three-lobed body. This is deliberately harder, like Fig. 11.
        xx, yy = x - cx, y - cy
        th = np.arctan2(yy, xx)
        rr = np.sqrt(xx**2 + yy**2 + 1e-12)
        r0 = 0.19 * (1.0 + 0.30 * np.cos(3.0 * th) - 0.16 * np.cos(th))
        return r0 - rr
    raise ValueError(f"unknown case {case!r}")


def chi_from_phi(phi: Array, dx: float) -> Array:
    """Paper Eq. (40): chi = clip(1/2 + phi / (4 dx), 0, 1)."""
    return jnp.clip(0.5 + phi / (4.0 * dx), 0.0, 1.0)


def np_chi_from_phi(phi: np.ndarray, dx: float) -> np.ndarray:
    return np.clip(0.5 + phi / (4.0 * dx), 0.0, 1.0)


def sample_measurements(
    rng: np.random.Generator,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    phi_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    n_data: int,
    gap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose velocity measurement points outside the body, away from boundaries."""
    mask = (
        (phi_ref < -gap)
        & (x > 0.18)
        & (x < 1.85)
        & (y > 0.08)
        & (y < 0.92)
    )
    # Prefer points around the body and wake, where the velocity carries shape info.
    cx, cy = 0.62, 0.50
    roi = mask & (x > cx - 0.35) & (x < 1.65) & (np.abs(y - cy) < 0.38)
    ids = np.argwhere(roi)
    if len(ids) < n_data:
        ids = np.argwhere(mask)
    if len(ids) < n_data:
        raise RuntimeError("not enough candidate measurement points")
    choice = rng.choice(len(ids), size=n_data, replace=False)
    ij = ids[choice]
    mi, mj = ij[:, 0], ij[:, 1]
    return mi, mj, u_ref[mi, mj], v_ref[mi, mj]


def pad_edge(q: Array) -> Array:
    return jnp.pad(q, ((1, 1), (1, 1)), mode="edge")


def first_x(q: Array, dx: float) -> Array:
    qp = pad_edge(q)
    return (qp[2:, 1:-1] - qp[:-2, 1:-1]) / (2.0 * dx)


def first_y(q: Array, dy: float) -> Array:
    qp = pad_edge(q)
    return (qp[1:-1, 2:] - qp[1:-1, :-2]) / (2.0 * dy)


def second_x(q: Array, dx: float) -> Array:
    qp = pad_edge(q)
    return (qp[2:, 1:-1] - 2.0 * qp[1:-1, 1:-1] + qp[:-2, 1:-1]) / (dx * dx)


def second_y(q: Array, dy: float) -> Array:
    qp = pad_edge(q)
    return (qp[1:-1, 2:] - 2.0 * qp[1:-1, 1:-1] + qp[1:-1, :-2]) / (dy * dy)


def upwind_x(q: Array, adv: Array, dx: float) -> Array:
    qp = pad_edge(q)
    backward = (qp[1:-1, 1:-1] - qp[:-2, 1:-1]) / dx
    forward = (qp[2:, 1:-1] - qp[1:-1, 1:-1]) / dx
    return jnp.where(adv >= 0.0, backward, forward)


def upwind_y(q: Array, adv: Array, dy: float) -> Array:
    qp = pad_edge(q)
    backward = (qp[1:-1, 1:-1] - qp[1:-1, :-2]) / dy
    forward = (qp[1:-1, 2:] - qp[1:-1, 1:-1]) / dy
    return jnp.where(adv >= 0.0, backward, forward)


def impose_velocity_data(u: Array, v: Array, const: Const) -> Tuple[Array, Array]:
    # Reparameterization used in the paper: the velocity exactly equals the
    # measurements in cells containing measurement points.
    u = u.at[const.meas_i, const.meas_j].set(const.meas_u)
    v = v.at[const.meas_i, const.meas_j].set(const.meas_v)
    return u, v


def fields_from_params(params: PyTree, const: Const, mode: str) -> Tuple[Array, Array, Array, Array, Array]:
    u = params["u"]
    v = params["v"]
    p = params["p"]
    if mode == "forward":
        chi = const.chi_ref
        phi = const.phi_ref
    else:
        phi = params["phi"]
        chi = chi_from_phi(phi, const.dx)
        u, v = impose_velocity_data(u, v, const)
    return u, v, p, phi, chi


def pde_losses(u: Array, v: Array, p: Array, chi: Array, const: Const) -> Tuple[Array, Array, Array, Array]:
    dx, dy = const.dx, const.dy
    ux = first_x(u, dx)
    uy = first_y(u, dy)
    vx = first_x(v, dx)
    vy = first_y(v, dy)
    px = first_x(p, dx)
    py = first_y(p, dy)

    uxx = second_x(u, dx)
    uyy = second_y(u, dy)
    vxx = second_x(v, dx)
    vyy = second_y(v, dy)

    # First-order upwind convection, as in the paper's description.
    uux = upwind_x(u, u, dx)
    uuy = upwind_y(u, v, dy)
    vvx = upwind_x(v, u, dx)
    vvy = upwind_y(v, v, dy)

    cont = ux + vy
    outside = 1.0 - chi
    fx = outside * (u * uux + v * uuy + px - const.D / const.Re * (uxx + uyy)) + const.lam * chi * u
    fy = outside * (u * vvx + v * vvy + py - const.D / const.Re * (vxx + vyy)) + const.lam * chi * v

    core = (slice(1, -1), slice(1, -1))
    loss_cont = jnp.mean(cont[core] ** 2)
    loss_mom = jnp.mean(fx[core] ** 2 + fy[core] ** 2)

    # Boundary conditions: inlet, outlet, free-slip top/bottom, pressure gauge.
    inlet = jnp.mean((u[0, :] - 1.0) ** 2 + v[0, :] ** 2)
    outlet = jnp.mean(((u[-1, :] - u[-2, :]) / dx) ** 2 + ((v[-1, :] - v[-2, :]) / dx) ** 2 + p[-1, :] ** 2)
    walls = jnp.mean(
        v[:, 0] ** 2
        + v[:, -1] ** 2
        + ((u[:, 1] - u[:, 0]) / dy) ** 2
        + ((u[:, -1] - u[:, -2]) / dy) ** 2
    )
    bc = inlet + outlet + walls
    return loss_cont, loss_mom, bc, fx


def total_loss(params: PyTree, const: Const, mode: str, kphi: float) -> Tuple[Array, Dict[str, Array]]:
    u, v, p, phi, chi = fields_from_params(params, const, mode)
    loss_cont, loss_mom, loss_bc, _ = pde_losses(u, v, p, chi, const)
    loss_pde = loss_cont + loss_mom

    # Data are imposed hard through reparameterization, so this residual is zero.
    loss_data = jnp.array(0.0, dtype=u.dtype)

    loss_eik = jnp.array(0.0, dtype=u.dtype)
    loss_area = jnp.array(0.0, dtype=u.dtype)
    loss_phi_anchor = jnp.array(0.0, dtype=u.dtype)
    if mode == "inverse":
        phix = first_x(phi, const.dx)
        phiy = first_y(phi, const.dy)
        loss_eik = jnp.mean((phix**2 + phiy**2 - 1.0) ** 2)
        # Small stabilizers; set weights to zero to exactly remove them.
        loss_area = (jnp.mean(chi) - jnp.mean(const.chi_ref)) ** 2
        loss_phi_anchor = jnp.mean((phi - const.phi_ref) ** 2)

    total = (
        const.w_pde * loss_pde
        + const.w_bc * loss_bc
        + const.w_data * loss_data
        + const.w_eik * (kphi**2) * loss_eik
        + const.w_area * loss_area
        + const.w_phi_anchor * loss_phi_anchor
    )
    aux = {
        "loss": total,
        "pde": loss_pde,
        "cont": loss_cont,
        "mom": loss_mom,
        "bc": loss_bc,
        "data": loss_data,
        "eik": loss_eik,
        "area": loss_area,
        "phi_anchor": loss_phi_anchor,
        "err_u": jnp.sqrt(jnp.mean((u - const.u_ref) ** 2)),
        "err_chi": jnp.sqrt(jnp.mean((chi - const.chi_ref) ** 2)) / (jnp.mean(const.chi_ref) + 1e-12),
    }
    return total, aux


@jax.jit
def value_grad_forward(params: PyTree, const: Const):
    return jax.value_and_grad(lambda pp: total_loss(pp, const, "forward", 1.0), has_aux=True)(params)


@jax.jit
def value_grad_inverse(params: PyTree, const: Const, kphi: float):
    return jax.value_and_grad(lambda pp: total_loss(pp, const, "inverse", kphi), has_aux=True)(params)


@jax.jit
def eval_forward(params: PyTree, const: Const):
    return total_loss(params, const, "forward", 1.0)[1]


@jax.jit
def eval_inverse(params: PyTree, const: Const, kphi: float):
    return total_loss(params, const, "inverse", kphi)[1]


def adam_step(params: PyTree, grads: PyTree, m: PyTree, vv: PyTree, step: int, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
    m = tree_add(tree_mul(m, beta1), tree_mul(grads, 1.0 - beta1))
    vv = tree_add(tree_mul(vv, beta2), tree_mul(tree_square(grads), 1.0 - beta2))
    mhat = tree_mul(m, 1.0 / (1.0 - beta1**step))
    vhat = tree_mul(vv, 1.0 / (1.0 - beta2**step))
    update = tree_div(mhat, jax.tree_util.tree_map(lambda x: jnp.sqrt(x) + eps, vhat))
    params = tree_sub(params, tree_mul(update, lr))
    return params, m, vv


def train(
    params: PyTree,
    const: Const,
    mode: str,
    epochs: int,
    lr: float,
    report_every: int,
    out_csv: str,
    kphi0: float = 10.0,
    kphi_min: float = 1.0,
) -> PyTree:
    m = tree_zeros_like(params)
    vv = tree_zeros_like(params)
    hist_keys = ["epoch", "loss", "pde", "cont", "mom", "bc", "data", "eik", "err_u", "err_chi", "kphi"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hist_keys)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            kphi = max(kphi_min, kphi0 * (0.5 ** (epoch / 1000.0)))
            if mode == "forward":
                (val, aux), grads = value_grad_forward(params, const)
            else:
                (val, aux), grads = value_grad_inverse(params, const, float(kphi))
            params, m, vv = adam_step(params, grads, m, vv, epoch, lr)
            if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
                aux_host = {k: float(v) for k, v in aux.items() if k in hist_keys}
                row = {"epoch": epoch, "kphi": kphi, **aux_host}
                writer.writerow(row)
                f.flush()
                print(
                    f"{mode:7s} epoch={epoch:6d} "
                    f"loss={row['loss']:.4e} pde={row['pde']:.4e} bc={row['bc']:.4e} "
                    f"data={row['data']:.4e} eik={row['eik']:.4e} "
                    f"err_u={row['err_u']:.4e} err_chi={row['err_chi']:.4e} kphi={kphi:.3g}",
                    flush=True,
                )
    return params


def vorticity_np(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # Same central difference convention as the loss, with edge padding.
    up = np.pad(u, ((1, 1), (1, 1)), mode="edge")
    vp = np.pad(v, ((1, 1), (1, 1)), mode="edge")
    vx = (vp[2:, 1:-1] - vp[:-2, 1:-1]) / (2.0 * dx)
    uy = (up[1:-1, 2:] - up[1:-1, :-2]) / (2.0 * dy)
    return vx - uy


def read_history(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    for key in rows[0].keys():
        data[key] = np.array([float(r[key]) for r in rows])
    return data


def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.18",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.size": 9,
        "savefig.bbox": "tight",
    })


def symmetric_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    vals = np.concatenate([np.ravel(np.asarray(a)) for a in arrays])
    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return 1.0
    return max(float(np.percentile(vals, percentile)), 1e-6)


def robust_limits(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    vals = np.ravel(np.asarray(arr))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [lo, hi])
    if np.isclose(vmin, vmax):
        pad = max(abs(float(vmin)) * 0.05, 1e-6)
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def measurement_xy(const: Const) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(const.x)
    y = np.asarray(const.y)
    i = np.asarray(const.meas_i)
    j = np.asarray(const.meas_j)
    return x[i, j], y[i, j]


def format_domain_axis(ax, title: str, show_ylabel: bool = True) -> None:
    ax.set_title(title, pad=6)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    if show_ylabel:
        ax.set_ylabel("y")
    else:
        ax.set_yticklabels([])
    ax.tick_params(direction="out", length=3, width=0.8)


def contour_if_present(ax, x: np.ndarray, y: np.ndarray, field: np.ndarray, level: float, **kwargs) -> None:
    if np.nanmin(field) <= level <= np.nanmax(field):
        ax.contour(x, y, field, levels=[level], **kwargs)


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    for start in np.argwhere(mask):
        si, sj = int(start[0]), int(start[1])
        if visited[si, sj]:
            continue
        stack = [(si, sj)]
        visited[si, sj] = True
        component: list[tuple[int, int]] = []
        while stack:
            i, j = stack.pop()
            component.append((i, j))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        stack.append((ni, nj))
        if len(component) > len(best):
            best = component

    result = np.zeros(mask.shape, dtype=bool)
    if best:
        ii, jj = zip(*best)
        result[np.array(ii), np.array(jj)] = True
    return result


def box_smooth(arr: np.ndarray, passes: int = 3) -> np.ndarray:
    smoothed = np.asarray(arr, dtype=float)
    for _ in range(passes):
        padded = np.pad(smoothed, ((1, 1), (1, 1)), mode="edge")
        smoothed = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return smoothed


def inferred_body_mask_for_plot(chi: np.ndarray) -> np.ndarray:
    # Display-only smoothing turns checkerboard-like inferred chi into a readable outline.
    return largest_component_mask(box_smooth(chi, passes=3) >= 0.18).astype(float)


def reference_body_mask_for_plot(chi_ref: np.ndarray) -> np.ndarray:
    return largest_component_mask(chi_ref >= 0.5).astype(float)


def overlay_body_contours(ax, x: np.ndarray, y: np.ndarray, chi: np.ndarray, chi_ref: np.ndarray) -> None:
    contour_if_present(ax, x, y, inferred_body_mask_for_plot(chi), 0.5, colors="#d62728", linewidths=1.6)
    contour_if_present(ax, x, y, reference_body_mask_for_plot(chi_ref), 0.5, colors="black", linewidths=1.1, linestyles="--")


def overlay_measurements(ax, const: Const, alpha: float = 0.42) -> None:
    mx, my = measurement_xy(const)
    ax.scatter(
        mx,
        my,
        s=11,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=0.45,
        alpha=alpha,
        zorder=4,
        rasterized=True,
    )


def shape_metrics(chi: np.ndarray, chi_ref: np.ndarray) -> Dict[str, float]:
    inferred = inferred_body_mask_for_plot(chi).astype(bool)
    reference = reference_body_mask_for_plot(chi_ref).astype(bool)
    inter = np.logical_and(inferred, reference).sum()
    union = np.logical_or(inferred, reference).sum()
    denom = inferred.sum() + reference.sum()
    iou = float(inter / union) if union else 1.0
    dice = float(2.0 * inter / denom) if denom else 1.0
    area_ratio = float((inferred.mean() + 1e-12) / (reference.mean() + 1e-12))
    chi_rmse = float(np.sqrt(np.mean((chi - chi_ref) ** 2)) / (np.mean(chi_ref) + 1e-12))
    return {"iou": iou, "dice": dice, "area_ratio": area_ratio, "chi_rmse": chi_rmse}


def add_metrics_box(ax, text: str, loc: str = "lower left") -> None:
    anchors = {
        "lower left": (0.025, 0.045, "left", "bottom"),
        "upper left": (0.025, 0.955, "left", "top"),
        "upper right": (0.975, 0.955, "right", "top"),
    }
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        color="0.08",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
        zorder=5,
    )


def add_history_panel(ax, hist: Dict[str, np.ndarray]) -> None:
    if not hist or "epoch" not in hist:
        ax.text(0.5, 0.5, "history not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    epoch = hist["epoch"]
    series = [
        ("loss", "#1f77b4", 1.9),
        ("pde", "#2ca02c", 1.5),
        ("bc", "#ff7f0e", 1.5),
        ("err_u", "#9467bd", 1.8),
        ("err_chi", "#d62728", 1.8),
    ]
    for key, color, lw in series:
        if key in hist:
            vals = np.maximum(hist[key], 1e-16)
            ax.semilogy(epoch, vals, label=key, color=color, lw=lw)
    ax.set_xlabel("epoch")
    ax.set_ylabel("log scale")
    ax.set_title("Optimization history", pad=6)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.7)
    ax.legend(ncol=3, frameon=False, loc="upper right", handlelength=1.6, columnspacing=0.9)


def save_figure(case: str, outdir: str, const: Const, params: PyTree, hist_path: str, suffix: str = "final") -> None:
    setup_plot_style()
    x = np.asarray(const.x)
    y = np.asarray(const.y)
    u_ref = np.asarray(const.u_ref)
    v_ref = np.asarray(const.v_ref)
    chi_ref = np.asarray(const.chi_ref)
    u, v, p, phi, chi = fields_from_params(params, const, "inverse")
    u = np.asarray(u)
    v = np.asarray(v)
    chi = np.asarray(chi)

    w_ref = vorticity_np(u_ref, v_ref, const.dx, const.dy)
    w_inf = vorticity_np(u, v, const.dx, const.dy)
    w_err = w_inf - w_ref
    speed_err = np.sqrt((u - u_ref) ** 2 + (v - v_ref) ** 2)
    vorticity_limit = symmetric_limit(w_ref, w_inf, percentile=99.0)
    residual_limit = symmetric_limit(w_err, percentile=99.0)
    speed_vmin, speed_vmax = robust_limits(speed_err, 0.0, 99.0)
    extent = [0, 2, 0, 1]
    metrics = shape_metrics(chi, chi_ref)
    vel_rmse = float(np.sqrt(np.mean((u - u_ref) ** 2 + (v - v_ref) ** 2)))

    hist = read_history(hist_path)
    fig = plt.figure(figsize=(13.2, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.92), width_ratios=(1.0, 1.0, 1.02))

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        w_inf.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vorticity_limit,
        vmax=vorticity_limit,
        interpolation="bilinear",
        rasterized=True,
    )
    overlay_body_contours(ax0, x, y, chi, chi_ref)
    overlay_measurements(ax0, const)
    format_domain_axis(ax0, "Inferred vorticity", show_ylabel=True)
    add_metrics_box(ax0, "red: inferred body\nblack dashed: reference", loc="upper left")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        w_ref.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vorticity_limit,
        vmax=vorticity_limit,
        interpolation="bilinear",
        rasterized=True,
    )
    contour_if_present(ax1, x, y, chi_ref, 0.5, colors="black", linewidths=1.25)
    overlay_measurements(ax1, const)
    format_domain_axis(ax1, "Reference vorticity", show_ylabel=False)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(
        w_err.T,
        origin="lower",
        extent=extent,
        cmap="PuOr_r",
        vmin=-residual_limit,
        vmax=residual_limit,
        interpolation="bilinear",
        rasterized=True,
    )
    overlay_body_contours(ax2, x, y, chi, chi_ref)
    format_domain_axis(ax2, "Vorticity residual", show_ylabel=False)
    cbar = fig.colorbar(im1, ax=[ax0, ax1], shrink=0.78, pad=0.01)
    cbar.set_label("vorticity")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.78, pad=0.012)
    cbar2.set_label("inferred - reference")

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        chi.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    overlay_body_contours(ax3, x, y, chi, chi_ref)
    format_domain_axis(ax3, "Body fraction chi", show_ylabel=True)
    add_metrics_box(
        ax3,
        f"IoU {metrics['iou']:.3f}\nDice {metrics['dice']:.3f}\narea ratio {metrics['area_ratio']:.2f}",
        loc="upper right",
    )
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.78, pad=0.012)
    cbar3.set_label("chi")

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(
        speed_err.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=speed_vmin,
        vmax=speed_vmax,
        interpolation="bilinear",
        rasterized=True,
    )
    overlay_body_contours(ax4, x, y, chi, chi_ref)
    overlay_measurements(ax4, const, alpha=0.28)
    format_domain_axis(ax4, "Velocity error magnitude", show_ylabel=False)
    add_metrics_box(ax4, f"RMSE {vel_rmse:.3e}\nchi err {metrics['chi_rmse']:.3f}", loc="upper right")
    cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.78, pad=0.012)
    cbar4.set_label("|u - u_ref|")

    ax5 = fig.add_subplot(gs[1, 2])
    add_history_panel(ax5, hist)

    fig.suptitle(f"{case.capitalize()} body-shape inversion  |  Re={const.Re:g}, D={const.D:g}", fontsize=13)
    path = os.path.join(outdir, f"fig_{case}_{suffix}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"saved {path}")


def save_field_panels(case: str, outdir: str, const: Const, params: PyTree) -> None:
    setup_plot_style()
    u, v, p, phi, chi = fields_from_params(params, const, "inverse")
    x = np.asarray(const.x)
    y = np.asarray(const.y)
    chi_ref = np.asarray(const.chi_ref)
    u = np.asarray(u)
    v = np.asarray(v)
    p = np.asarray(p)
    phi = np.asarray(phi)
    chi = np.asarray(chi)
    speed = np.sqrt(u**2 + v**2)
    speed_err = np.sqrt((u - np.asarray(const.u_ref)) ** 2 + (v - np.asarray(const.v_ref)) ** 2)
    panels = [
        ("u velocity", u, "viridis", robust_limits(u)),
        ("v velocity", v, "RdBu_r", (-symmetric_limit(v), symmetric_limit(v))),
        ("pressure", p, "RdBu_r", (-symmetric_limit(p), symmetric_limit(p))),
        ("speed", speed, "magma", robust_limits(speed, 0.0, 99.0)),
        ("body fraction chi", chi, "viridis", (0.0, 1.0)),
        ("velocity error", speed_err, "magma", robust_limits(speed_err, 0.0, 99.0)),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.0), constrained_layout=True)
    for idx, (ax, (name, arr, cmap, limits)) in enumerate(zip(axs.ravel(), panels)):
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=[0, 2, 0, 1],
            interpolation="bilinear" if name != "body fraction chi" else "nearest",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )
        overlay_body_contours(ax, x, y, chi, chi_ref)
        if name in {"u velocity", "v velocity", "speed", "velocity error"}:
            overlay_measurements(ax, const, alpha=0.25)
        format_domain_axis(ax, name, show_ylabel=(idx % 3 == 0))
        fig.colorbar(im, ax=ax, shrink=0.78, pad=0.012)
    fig.suptitle(f"{case.capitalize()} inferred fields", fontsize=13)
    path = os.path.join(outdir, f"fields_{case}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path}")


def save_state(path: str, params: PyTree) -> None:
    host = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    with open(path, "wb") as f:
        pickle.dump(host, f)


def load_state(path: str) -> PyTree:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {k: jnp.asarray(v) for k, v in data.items()}


def build_const(
    args: argparse.Namespace,
    case: str,
    u_ref: np.ndarray | None,
    v_ref: np.ndarray | None,
    p_ref: np.ndarray | None,
    meas: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> Const:
    x, y, dx, dy = make_grid(args.N)
    phi_ref = phi_body(case, x, y, args.D)
    chi_ref = np_chi_from_phi(phi_ref, dx)
    nx, ny = x.shape
    if u_ref is None:
        u_ref = np.ones((nx, ny), dtype=np.float64)
    if v_ref is None:
        v_ref = np.zeros((nx, ny), dtype=np.float64)
    if p_ref is None:
        p_ref = np.zeros((nx, ny), dtype=np.float64)
    if meas is None:
        # Placeholder measurement arrays, used during forward solve.
        mi = np.array([0], dtype=np.int32)
        mj = np.array([0], dtype=np.int32)
        mu = np.array([1.0], dtype=np.float64)
        mv = np.array([0.0], dtype=np.float64)
    else:
        mi, mj, mu, mv = meas
    return Const(
        x=jnp.asarray(x), y=jnp.asarray(y), dx=dx, dy=dy,
        Re=float(args.Re), D=float(args.D), lam=float(args.lambda_penalty),
        chi_ref=jnp.asarray(chi_ref), phi_ref=jnp.asarray(phi_ref),
        u_ref=jnp.asarray(u_ref), v_ref=jnp.asarray(v_ref), p_ref=jnp.asarray(p_ref),
        meas_i=jnp.asarray(mi, dtype=jnp.int32), meas_j=jnp.asarray(mj, dtype=jnp.int32),
        meas_u=jnp.asarray(mu), meas_v=jnp.asarray(mv),
        w_pde=float(args.w_pde), w_bc=float(args.w_bc), w_data=float(args.w_data),
        w_eik=float(args.w_eik), w_area=float(args.w_area), w_phi_anchor=float(args.w_phi_anchor),
    )


def initial_forward_params(N: int) -> PyTree:
    nx, ny = 2 * N, N
    return {
        "u": jnp.ones((nx, ny), dtype=jnp.float64),
        "v": jnp.zeros((nx, ny), dtype=jnp.float64),
        "p": jnp.zeros((nx, ny), dtype=jnp.float64),
    }


def initial_inverse_params(args: argparse.Namespace, const: Const) -> PyTree:
    nx, ny = 2 * args.N, args.N
    # Paper initial level set: small circle phi = 0.02 - sqrt((x-0.5)^2+(y-0.5)^2).
    x = np.asarray(const.x)
    y = np.asarray(const.y)
    phi0 = 0.02 - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + 1e-12)
    return {
        "u": jnp.ones((nx, ny), dtype=jnp.float64),
        "v": jnp.zeros((nx, ny), dtype=jnp.float64),
        "p": jnp.zeros((nx, ny), dtype=jnp.float64),
        "phi": jnp.asarray(phi0),
    }


def run_case(args: argparse.Namespace, case: str) -> None:
    outdir = os.path.join(args.outdir, case)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== case={case}, N={args.N}, grid={2*args.N}x{args.N}, outdir={outdir} ===")

    ref_path = os.path.join(outdir, "reference.pkl")
    # Build reference constants with dummy u_ref for the forward problem.
    const_ref0 = build_const(args, case, None, None, None, None)

    if args.viz_only:
        inv_path = os.path.join(outdir, "inverse.pkl")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"missing reference state for --viz_only: {ref_path}")
        if not os.path.exists(inv_path):
            raise FileNotFoundError(f"missing inverse state for --viz_only: {inv_path}")
        ref_params = load_state(ref_path)
        u_ref = np.asarray(ref_params["u"])
        v_ref = np.asarray(ref_params["v"])
        p_ref = np.asarray(ref_params["p"])
        rng = np.random.default_rng(args.seed)
        x_np, y_np, dx, dy = make_grid(args.N)
        phi_ref_np = phi_body(case, x_np, y_np, args.D)
        meas = sample_measurements(rng, u_ref, v_ref, phi_ref_np, x_np, y_np, dx, args.n_data, args.data_gap)
        const_inv = build_const(args, case, u_ref, v_ref, p_ref, meas)
        inv_params = load_state(inv_path)
        inv_hist = os.path.join(outdir, "inverse.csv")
        save_figure(case, outdir, const_inv, inv_params, inv_hist)
        save_field_panels(case, outdir, const_inv, inv_params)
        return

    if args.load_ref and os.path.exists(ref_path):
        ref_params = load_state(ref_path)
        print(f"loaded reference {ref_path}")
    else:
        ref_params = initial_forward_params(args.N)
        const_fwd = const_ref0
        ref_params = train(
            ref_params,
            const_fwd,
            mode="forward",
            epochs=args.forward_epochs,
            lr=args.forward_lr,
            report_every=args.report_every,
            out_csv=os.path.join(outdir, "forward.csv"),
        )
        save_state(ref_path, ref_params)
        print(f"saved reference {ref_path}")

    u_ref = np.asarray(ref_params["u"])
    v_ref = np.asarray(ref_params["v"])
    p_ref = np.asarray(ref_params["p"])

    rng = np.random.default_rng(args.seed)
    x_np, y_np, dx, dy = make_grid(args.N)
    phi_ref_np = phi_body(case, x_np, y_np, args.D)
    meas = sample_measurements(rng, u_ref, v_ref, phi_ref_np, x_np, y_np, dx, args.n_data, args.data_gap)

    const_inv = build_const(args, case, u_ref, v_ref, p_ref, meas)
    inv_params = initial_inverse_params(args, const_inv)

    inv_hist = os.path.join(outdir, "inverse.csv")
    inv_params = train(
        inv_params,
        const_inv,
        mode="inverse",
        epochs=args.epochs,
        lr=args.lr,
        report_every=args.report_every,
        out_csv=inv_hist,
        kphi0=args.kphi0,
        kphi_min=args.kphi_min,
    )
    save_state(os.path.join(outdir, "inverse.pkl"), inv_params)
    save_figure(case, outdir, const_inv, inv_params, inv_hist)
    save_field_panels(case, outdir, const_inv, inv_params)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--case", choices=["circle", "ellipse", "nonconvex", "all"], default="all")
    p.add_argument("--N", type=int, default=64, help="Grid uses 2N x N cells")
    p.add_argument("--Re", type=float, default=60.0)
    p.add_argument("--D", type=float, default=0.4, help="Characteristic body length")
    p.add_argument("--lambda_penalty", type=float, default=10.0, help="Brinkman body penalization lambda")
    p.add_argument("--epochs", type=int, default=20000, help="Inverse Adam iterations")
    p.add_argument("--forward_epochs", type=int, default=12000, help="Forward/reference Adam iterations")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--forward_lr", type=float, default=1e-3)
    p.add_argument("--report_every", type=int, default=200)
    p.add_argument("--n_data", type=int, default=100)
    p.add_argument("--data_gap", type=float, default=0.06, help="Minimum signed-distance gap from body for data points")
    p.add_argument("--hard_data", type=int, default=1, help="Kept for compatibility; this script always uses hard data reparameterization")
    p.add_argument("--w_pde", type=float, default=1.0)
    p.add_argument("--w_bc", type=float, default=10.0)
    p.add_argument("--w_data", type=float, default=100.0, help="Only used when --hard_data 0")
    p.add_argument("--w_eik", type=float, default=0.01, help="Eikonal loss multiplier before kphi^2")
    p.add_argument("--w_area", type=float, default=1e-3, help="Small area stabilizer; set 0 to remove")
    p.add_argument("--w_phi_anchor", type=float, default=0.0, help="Optional phi-to-reference anchor for debugging only")
    p.add_argument("--kphi0", type=float, default=10.0)
    p.add_argument("--kphi_min", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", default="out_body_shape")
    p.add_argument("--load_ref", type=int, default=1, help="Reuse reference.pkl if present")
    p.add_argument("--viz_only", type=int, default=0, help="Reload existing states and regenerate figures without training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases: Iterable[str] = ["circle", "ellipse", "nonconvex"] if args.case == "all" else [args.case]
    os.makedirs(args.outdir, exist_ok=True)
    for case in cases:
        run_case(args, case)


if __name__ == "__main__":
    main()
