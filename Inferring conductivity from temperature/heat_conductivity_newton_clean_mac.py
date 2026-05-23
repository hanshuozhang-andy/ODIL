#!/usr/bin/env python3
"""
Clean-data ODIL-Newton reproduction for pgae005.pdf, Fig. 4 / heat example:
"Inferring conductivity from temperature".

This is a standalone macOS-friendly script. It does not require the official
ODIL package. It implements a Gauss-Newton / Levenberg-damped Newton solve for

    u_t - d_x( k(u) u_x ) = 0,  (x,t) in [0,1]^2

with zero Dirichlet boundary conditions and the same second-order halo formulas
used in the paper. The unknown conductivity k(u) is represented by the same tiny
network used in the paper/source:

    1 x 5 x 5 x 1, tanh activations, k(u) = 0.1 * sigmoid(q(u)).

The inverse loss is

    mean(PDE residual^2) + w_data^2 / N_data * sum_observed (u - u_obs)^2
    + optional Newton damping on NN weights.

Run on macOS:
    python3 -m pip install numpy scipy matplotlib
    python3 heat_conductivity_newton_clean_mac.py

Fast test:
    python3 heat_conductivity_newton_clean_mac.py --Nx 32 --Nt 32 --ref_Nx 128 --ref_Nt 128 --newton_steps 15

Closer to the paper grid, but slower:
    python3 heat_conductivity_newton_clean_mac.py --Nx 64 --Nt 64 --ref_Nx 256 --ref_Nt 256 --newton_steps 30

Outputs:
    out_heat_newton_clean/heat_results.png
    out_heat_newton_clean/history.png
    out_heat_newton_clean/data.npz
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


# -------------------------- reference problem --------------------------

def init_u_np(x: np.ndarray) -> np.ndarray:
    """U(x)=g(x)-g(0), g(x)=exp(-50(x-0.5)^2)."""
    g = np.exp(-50.0 * (x - 0.5) ** 2)
    g0 = np.exp(-50.0 * (0.0 - 0.5) ** 2)
    return g - g0


def k_ref_np(u: np.ndarray) -> np.ndarray:
    """Reference conductivity k(u)=0.02 exp(-20(u-0.5)^2)."""
    return 0.02 * np.exp(-20.0 * (u - 0.5) ** 2)


def make_reference(args):
    """
    Generate a clean reference temperature field with an implicit finite-volume
    solver. The paper/source uses ODIL/Newton on a finer grid; this standalone
    version uses a robust implicit Picard method so it runs without ODIL.
    """
    from scipy.linalg import solve_banded

    nx = args.ref_Nx
    nt = args.ref_Nt
    x = (np.arange(nx) + 0.5) / nx
    dx = 1.0 / nx
    dt = 1.0 / nt
    u = init_u_np(x).astype(np.float64)
    out = np.empty((nt, nx), dtype=np.float64)

    print(f"Generating clean reference: ref_Nt={nt}, ref_Nx={nx} ...")
    for n in range(nt):
        guess = u.copy()
        for _ in range(args.ref_picard):
            k_face = np.empty(nx + 1, dtype=np.float64)
            k_face[0] = k_ref_np(0.5 * guess[0])
            k_face[-1] = k_ref_np(0.5 * guess[-1])
            k_face[1:-1] = k_ref_np(0.5 * (guess[:-1] + guess[1:]))

            lower_L = k_face[:-1] / dx**2
            upper_L = k_face[1:] / dx**2
            diag_L = -(lower_L + upper_L)
            # zero Dirichlet boundary at a half-cell distance
            diag_L[0] = -(2.0 * k_face[0] + k_face[1]) / dx**2
            diag_L[-1] = -(k_face[-2] + 2.0 * k_face[-1]) / dx**2

            ab = np.zeros((3, nx), dtype=np.float64)
            ab[0, 1:] = -dt * upper_L[:-1]
            ab[1, :] = 1.0 - dt * diag_L
            ab[2, :-1] = -dt * lower_L[1:]
            new_u = solve_banded((1, 1), ab, u)
            if np.linalg.norm(new_u - guess) / (np.linalg.norm(new_u) + 1e-14) < 1e-11:
                guess = new_u
                break
            guess = new_u
        u = guess
        out[n] = u

    t = (np.arange(nt) + 0.5) / nt
    return t, x, out


def interpolate_reference_to_grid(t_ref, x_ref, ref_u, Nt, Nx):
    t = (np.arange(Nt) + 0.5) / Nt
    x = (np.arange(Nx) + 0.5) / Nx
    interp = RectBivariateSpline(t_ref, x_ref, ref_u, kx=3, ky=3)
    return t, x, interp(t, x)


def make_imposed_data(ref_u, t, x, args):
    """Pick N_data clean observation points; default mimics official --imposed stripe."""
    rng = np.random.default_rng(args.seed)
    Nt, Nx = ref_u.shape
    tt, xx = np.meshgrid(t, x, indexing="ij")

    if args.imposed == "stripe":
        candidate = np.flatnonzero(np.abs(tt.reshape(-1) - 0.5) < 1.0 / 6.0)
    elif args.imposed == "random":
        candidate = np.arange(Nt * Nx)
    else:
        raise ValueError("--imposed must be stripe or random")

    nimp = min(args.nimp, candidate.size)
    chosen = rng.choice(candidate, size=nimp, replace=False)
    mask = np.zeros(Nt * Nx, dtype=bool)
    mask[chosen] = True
    return mask.reshape(Nt, Nx), ref_u.copy(), chosen


# -------------------------- tiny k(u) network --------------------------

@dataclass
class ThetaSpec:
    ntheta: int = 46


def init_theta(seed: int) -> np.ndarray:
    """Xavier-like random initialization for 1x5x5x1 tanh network."""
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-np.sqrt(6 / 6), np.sqrt(6 / 6), size=(5, 1))
    b1 = np.zeros(5)
    W2 = rng.uniform(-np.sqrt(6 / 10), np.sqrt(6 / 10), size=(5, 5))
    b2 = np.zeros(5)
    W3 = rng.uniform(-np.sqrt(6 / 6), np.sqrt(6 / 6), size=(1, 5))
    b3 = np.zeros(1)
    return pack_theta(W1, b1, W2, b2, W3, b3)


def pack_theta(W1, b1, W2, b2, W3, b3) -> np.ndarray:
    return np.concatenate([
        W1.reshape(-1), b1.reshape(-1), W2.reshape(-1), b2.reshape(-1), W3.reshape(-1), b3.reshape(-1)
    ]).astype(np.float64)


def unpack_theta(theta: np.ndarray):
    p = 0
    W1 = theta[p:p+5].reshape(5, 1); p += 5
    b1 = theta[p:p+5]; p += 5
    W2 = theta[p:p+25].reshape(5, 5); p += 25
    b2 = theta[p:p+5]; p += 5
    W3 = theta[p:p+5].reshape(1, 5); p += 5
    b3 = theta[p:p+1]; p += 1
    return W1, b1, W2, b2, W3, b3


def knet_value_and_derivatives(u: np.ndarray, theta: np.ndarray, kmax: float = 0.1):
    """
    Return k(u), dk/du, and dk/dtheta for all u values.
    u may have arbitrary shape. dkdtheta has shape (u.size, 46).
    """
    shape = u.shape
    uf = u.reshape(-1).astype(np.float64)
    m = uf.size
    W1, b1, W2, b2, W3, b3 = unpack_theta(theta)

    z1 = uf[:, None] * W1[:, 0][None, :] + b1[None, :]       # (m, 5)
    a1 = np.tanh(z1)
    z2 = a1 @ W2.T + b2[None, :]                             # (m, 5)
    a2 = np.tanh(z2)
    q = (a2 @ W3.T).reshape(-1) + b3[0]

    # stable sigmoid
    s = 1.0 / (1.0 + np.exp(-np.clip(q, -60.0, 60.0)))
    k = kmax * s
    dk_dq = kmax * s * (1.0 - s)

    dq_dtheta = np.empty((m, 46), dtype=np.float64)

    # Backprop sensitivities for q.
    dq_da2 = W3.reshape(5)[None, :]                          # (1,5)
    dq_dz2 = dq_da2 * (1.0 - a2**2)                           # (m,5)
    dq_da1 = dq_dz2 @ W2                                      # (m,5)
    dq_dz1 = dq_da1 * (1.0 - a1**2)                           # (m,5)

    p = 0
    dq_dtheta[:, p:p+5] = dq_dz1 * uf[:, None]                # W1
    p += 5
    dq_dtheta[:, p:p+5] = dq_dz1                              # b1
    p += 5
    # W2 row-major: W2[h,j]
    block = np.empty((m, 25), dtype=np.float64)
    c = 0
    for h in range(5):
        for j in range(5):
            block[:, c] = dq_dz2[:, h] * a1[:, j]
            c += 1
    dq_dtheta[:, p:p+25] = block
    p += 25
    dq_dtheta[:, p:p+5] = dq_dz2                              # b2
    p += 5
    dq_dtheta[:, p:p+5] = a2                                  # W3
    p += 5
    dq_dtheta[:, p:p+1] = 1.0                                 # b3

    dk_dtheta = dk_dq[:, None] * dq_dtheta
    dq_du = dq_dz1 @ W1[:, 0]
    dk_du = dk_dq * dq_du

    return k.reshape(shape), dk_du.reshape(shape), dk_dtheta


# -------------------------- residual and Jacobian --------------------------

@dataclass
class Grid:
    Nt: int
    Nx: int
    dt: float
    dx: float
    init_u: np.ndarray


def pad_x_quad_zero(u: np.ndarray) -> np.ndarray:
    left = (u[:, 1:2] - 6.0 * u[:, 0:1]) / 3.0
    right = (u[:, -2:-1] - 6.0 * u[:, -1:]) / 3.0
    return np.concatenate([left, u, right], axis=1)


def previous_time_with_initial_halo(u: np.ndarray, init_u: np.ndarray) -> np.ndarray:
    first_prev = (u[1:2, :] - 6.0 * u[0:1, :] + 8.0 * init_u[None, :]) / 3.0
    return np.concatenate([first_prev, u[:-1, :]], axis=0)


def pde_quantities(u: np.ndarray, theta: np.ndarray, grid: Grid, kmax: float):
    """Compute residual and local quantities needed for the analytic Jacobian."""
    up = pad_x_quad_zero(u)
    um = previous_time_with_initial_halo(u, grid.init_u)
    ump = pad_x_quad_zero(um)

    qc = up[:, 1:-1]
    ql = up[:, :-2]
    qr = up[:, 2:]
    qmc = ump[:, 1:-1]
    qml = ump[:, :-2]
    qmr = ump[:, 2:]

    u_t = (qc - qmc) / grid.dt
    xm = ((qc + qmc) - (ql + qml)) / (2.0 * grid.dx)
    xp = ((qr + qmr) - (qc + qmc)) / (2.0 * grid.dx)
    face_m = ((qc + qmc) + (ql + qml)) * 0.25
    face_p = ((qr + qmr) + (qc + qmc)) * 0.25

    km, dkm_du, dkm_dtheta = knet_value_and_derivatives(face_m, theta, kmax)
    kp, dkp_du, dkp_dtheta = knet_value_and_derivatives(face_p, theta, kmax)

    flux_div = (xp * kp - xm * km) / grid.dx
    F = u_t - flux_div
    return F, xm, xp, face_m, face_p, km, kp, dkm_du, dkp_du, dkm_dtheta, dkp_dtheta


def build_residual_and_jacobian(u, theta, grid, mask, imp_u, args):
    """Build residual vector and sparse Jacobian for one Gauss-Newton step."""
    Nt, Nx = grid.Nt, grid.Nx
    Ncell = Nt * Nx
    P = theta.size
    n_u = Ncell
    n_param = n_u + P

    F, xm, xp, face_m, face_p, km, kp, dkm_du, dkp_du, dkm_dtheta, dkp_dtheta = pde_quantities(
        u, theta, grid, args.kmax
    )

    obs_flat = np.flatnonzero(mask.reshape(-1))
    Ndata = max(obs_flat.size, 1)

    scale_pde = 1.0 / np.sqrt(Ncell)
    scale_data = args.wdata / np.sqrt(Ndata)
    scale_theta = args.wtheta / np.sqrt(P)

    r_pde = (F.reshape(-1) * scale_pde).astype(np.float64)
    r_data = ((u.reshape(-1)[obs_flat] - imp_u.reshape(-1)[obs_flat]) * scale_data).astype(np.float64)
    r_theta = np.zeros(P, dtype=np.float64)  # current theta minus frozen theta = 0
    r = np.concatenate([r_pde, r_data, r_theta])

    rows, cols, vals = [], [], []

    def u_index(n, i):
        return n * Nx + i

    def add(row, col, val):
        if val != 0.0 and np.isfinite(val):
            rows.append(row); cols.append(col); vals.append(val)

    def add_v_initial(row, j, coeff):
        """v_j=(u[1,j]-6u[0,j]+8U_j)/3 for the initial-time halo."""
        add(row, u_index(1, j), coeff / 3.0)
        add(row, u_index(0, j), -2.0 * coeff)

    def add_prev_n0_x(row, i, role, coeff):
        # role in {'l','c','r'} for qml, qmc, qmr at n=0 after x-padding v.
        if role == 'c':
            add_v_initial(row, i, coeff)
        elif role == 'l':
            if i == 0:
                add_v_initial(row, 1, coeff / 3.0)
                add_v_initial(row, 0, -2.0 * coeff)
            else:
                add_v_initial(row, i - 1, coeff)
        elif role == 'r':
            if i == Nx - 1:
                add_v_initial(row, Nx - 2, coeff / 3.0)
                add_v_initial(row, Nx - 1, -2.0 * coeff)
            else:
                add_v_initial(row, i + 1, coeff)

    def add_time_x(row, n, i, role, coeff):
        # role in {'l','c','r'} for current/normal previous time after x-padding.
        if role == 'c':
            add(row, u_index(n, i), coeff)
        elif role == 'l':
            if i == 0:
                add(row, u_index(n, 1), coeff / 3.0)
                add(row, u_index(n, 0), -2.0 * coeff)
            else:
                add(row, u_index(n, i - 1), coeff)
        elif role == 'r':
            if i == Nx - 1:
                add(row, u_index(n, Nx - 2), coeff / 3.0)
                add(row, u_index(n, Nx - 1), -2.0 * coeff)
            else:
                add(row, u_index(n, i + 1), coeff)

    invdt = 1.0 / grid.dt
    invdx = 1.0 / grid.dx
    inv2dx = 1.0 / (2.0 * grid.dx)

    # PDE block
    for n in range(Nt):
        for i in range(Nx):
            row = n * Nx + i
            kml = km[n, i]
            kpr = kp[n, i]
            dkm = dkm_du[n, i]
            dkp = dkp_du[n, i]
            xmv = xm[n, i]
            xpv = xp[n, i]

            # local derivatives wrt q_l, q_c, q_r and previous qml,qmc,qmr
            coeff_qc = invdt - invdx * ((-inv2dx) * kpr + xpv * dkp * 0.25 - (inv2dx * kml + xmv * dkm * 0.25))
            coeff_qr = -invdx * (inv2dx * kpr + xpv * dkp * 0.25)
            coeff_ql = invdx * ((-inv2dx) * kml + xmv * dkm * 0.25)

            coeff_qmc = -invdt - invdx * ((-inv2dx) * kpr + xpv * dkp * 0.25 - (inv2dx * kml + xmv * dkm * 0.25))
            coeff_qmr = coeff_qr
            coeff_qml = coeff_ql

            coeffs = [
                ('l', coeff_ql), ('c', coeff_qc), ('r', coeff_qr)
            ]
            for role, coeff in coeffs:
                add_time_x(row, n, i, role, coeff * scale_pde)

            if n > 0:
                for role, coeff in [('l', coeff_qml), ('c', coeff_qmc), ('r', coeff_qmr)]:
                    add_time_x(row, n - 1, i, role, coeff * scale_pde)
            else:
                for role, coeff in [('l', coeff_qml), ('c', coeff_qmc), ('r', coeff_qmr)]:
                    add_prev_n0_x(row, i, role, coeff * scale_pde)

            # derivative wrt NN weights: -1/dx * (xp * dkp/dtheta - xm * dkm/dtheta)
            flat = row
            theta_coeff = -invdx * (xpv * dkp_dtheta[flat, :] - xmv * dkm_dtheta[flat, :]) * scale_pde
            for j in range(P):
                add(row, n_u + j, theta_coeff[j])

    # Data rows
    base = Ncell
    for rloc, flat in enumerate(obs_flat):
        add(base + rloc, flat, scale_data)

    # Newton damping rows on theta weights, matching the spirit of Eq. (32)
    base2 = Ncell + obs_flat.size
    for j in range(P):
        add(base2 + j, n_u + j, scale_theta)

    J = coo_matrix((vals, (rows, cols)), shape=(r.size, n_param)).tocsc()
    return r, J


def compute_loss_parts(u, theta, grid, mask, imp_u, args):
    F = pde_quantities(u, theta, grid, args.kmax)[0]
    obs = mask
    pde_loss = float(np.mean(F**2))
    data_loss = float(args.wdata**2 / max(obs.sum(), 1) * np.sum((u[obs] - imp_u[obs])**2))
    return pde_loss, data_loss, pde_loss + data_loss


# -------------------------- Newton solve --------------------------

def solve_inverse_newton(args):
    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    t_ref, x_ref, ref_u_fine = make_reference(args)
    t, x, ref_u = interpolate_reference_to_grid(t_ref, x_ref, ref_u_fine, args.Nt, args.Nx)
    mask, imp_u, imp_indices = make_imposed_data(ref_u, t, x, args)

    grid = Grid(args.Nt, args.Nx, 1.0 / args.Nt, 1.0 / args.Nx, init_u_np(x))

    if args.warm_init:
        u = np.repeat(init_u_np(x)[None, :], args.Nt, axis=0).astype(np.float64)
    else:
        u = np.zeros((args.Nt, args.Nx), dtype=np.float64)

    theta = init_theta(args.seed + 100)
    P = theta.size
    Ncell = args.Nt * args.Nx

    us = np.linspace(0.0, 1.0, 300)
    ref_k = k_ref_np(us)

    hist = {"step": [], "loss": [], "pde": [], "data": [], "uerr": [], "kerr": [], "alpha": [], "stepnorm": []}

    pde0, data0, loss0 = compute_loss_parts(u, theta, grid, mask, imp_u, args)
    print(f"Initial loss={loss0:.6e} pde={pde0:.6e} data={data0:.6e}")
    print(f"Newton solve: Nt={args.Nt}, Nx={args.Nx}, unknowns={Ncell + P}, observations={mask.sum()}")

    start = time.time()
    for step in range(1, args.newton_steps + 1):
        t0 = time.time()
        r, J = build_residual_and_jacobian(u, theta, grid, mask, imp_u, args)
        A = (J.T @ J).tocsc()
        if args.lm > 0:
            A = A + args.lm * eye(A.shape[0], format="csc")
        b = -(J.T @ r)

        try:
            delta = spsolve(A, b)
        except Exception as e:
            print("Linear solve failed:", e)
            break

        # Limit very large steps for robustness on laptops.
        max_abs = np.max(np.abs(delta))
        if max_abs > args.max_step_abs:
            delta *= args.max_step_abs / max_abs

        u_flat = u.reshape(-1)
        old_u = u.copy()
        old_theta = theta.copy()
        old_loss = compute_loss_parts(u, theta, grid, mask, imp_u, args)[2]

        accepted = False
        alpha = 1.0
        for _ in range(args.line_search):
            cand_u = (u_flat + alpha * delta[:Ncell]).reshape(args.Nt, args.Nx)
            cand_theta = theta + alpha * delta[Ncell:]
            pde_l, data_l, cand_loss = compute_loss_parts(cand_u, cand_theta, grid, mask, imp_u, args)
            if np.isfinite(cand_loss) and cand_loss <= old_loss * (1.0 - args.armijo * alpha) + 1e-16:
                u = cand_u
                theta = cand_theta
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # still take a tiny step if it improves, otherwise stop
            cand_u = (u_flat + alpha * delta[:Ncell]).reshape(args.Nt, args.Nx)
            cand_theta = theta + alpha * delta[Ncell:]
            pde_l, data_l, cand_loss = compute_loss_parts(cand_u, cand_theta, grid, mask, imp_u, args)
            if np.isfinite(cand_loss) and cand_loss < old_loss:
                u = cand_u
                theta = cand_theta
                accepted = True
            else:
                print(f"step {step}: line search failed; stopping.")
                break

        pde_l, data_l, loss_l = compute_loss_parts(u, theta, grid, mask, imp_u, args)
        k_pred = knet_value_and_derivatives(us, theta, args.kmax)[0]
        uerr = np.sqrt(np.mean((u - ref_u) ** 2)) / (np.max(np.abs(ref_u)) + 1e-14)
        kerr = np.sqrt(np.mean((k_pred - ref_k) ** 2)) / (np.max(ref_k) + 1e-14)
        stepnorm = np.linalg.norm(alpha * delta) / (np.linalg.norm(np.concatenate([old_u.reshape(-1), old_theta])) + 1e-14)

        hist["step"].append(step)
        hist["loss"].append(loss_l)
        hist["pde"].append(pde_l)
        hist["data"].append(data_l)
        hist["uerr"].append(uerr)
        hist["kerr"].append(kerr)
        hist["alpha"].append(alpha)
        hist["stepnorm"].append(stepnorm)

        print(
            f"step {step:03d} | loss={loss_l:.3e} pde={pde_l:.3e} data={data_l:.3e} "
            f"u_err={uerr:.3e} k_rel_err={kerr:.3e} alpha={alpha:.2g} "
            f"time={time.time()-t0:.2f}s"
        )
        if stepnorm < args.tol:
            print("Converged by small relative step.")
            break

    print(f"Done in {time.time()-start:.1f} s")

    k_pred = knet_value_and_derivatives(us, theta, args.kmax)[0]
    save_outputs(args, t, x, ref_u, u, mask, imp_u, us, ref_k, k_pred, theta, hist)


def save_outputs(args, t, x, ref_u, u_pred, mask, imp_u, us, k_ref, k_pred, theta, hist):
    np.savez(
        os.path.join(args.outdir, "data.npz"),
        t=t, x=x, ref_u=ref_u, u_pred=u_pred, mask=mask, imp_u=imp_u,
        us=us, k_ref=k_ref, k_pred=k_pred, theta=theta,
        **{k: np.array(v) for k, v in hist.items()}
    )
    plot_results(args.outdir, t, x, ref_u, u_pred, mask, us, k_ref, k_pred)
    plot_history(args.outdir, hist)


def plot_results(outdir, t, x, ref_u, u_pred, mask, us, k_ref, k_pred):
    extent = [x[0], x[-1], t[0], t[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)

    im0 = axes[0].imshow(ref_u, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="YlOrBr")
    axes[0].set_title("reference u")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = axes[1].imshow(u_pred, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1, cmap="YlOrBr")
    axes[1].set_title("inferred u (Newton)")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    im2 = axes[2].imshow(u_pred - ref_u, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    yy, xx = np.where(mask)
    axes[2].scatter(x[xx], t[yy], s=3, c="k", alpha=0.8, linewidths=0)
    axes[2].set_title("error + clean data points")
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    axes[3].plot(us, k_ref, label="reference")
    axes[3].plot(us, k_pred, label="inferred")
    axes[3].set_xlabel("u")
    axes[3].set_ylabel("k(u)")
    axes[3].set_ylim(0, 0.03)
    axes[3].set_title("conductivity")
    axes[3].legend(frameon=False)

    fig.savefig(os.path.join(outdir, "heat_results.png"), dpi=200)
    plt.close(fig)


def plot_history(outdir, hist):
    step = np.array(hist["step"])
    if step.size == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].semilogy(step, hist["loss"], label="total")
    axes[0].semilogy(step, hist["pde"], label="PDE")
    axes[0].semilogy(step, hist["data"], label="data")
    axes[0].set_xlabel("Newton step")
    axes[0].set_title("loss")
    axes[0].legend(frameon=False)

    axes[1].semilogy(step, hist["uerr"])
    axes[1].set_xlabel("Newton step")
    axes[1].set_title("temperature rel. RMSE")

    axes[2].semilogy(step, hist["kerr"])
    axes[2].set_xlabel("Newton step")
    axes[2].set_title("conductivity rel. RMSE")

    fig.savefig(os.path.join(outdir, "history.png"), dpi=200)
    plt.close(fig)


# -------------------------- CLI --------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--Nt", type=int, default=64, help="inverse ODIL grid size in t; use 64 for paper-like grid")
    p.add_argument("--Nx", type=int, default=64, help="inverse ODIL grid size in x; use 64 for paper-like grid")
    p.add_argument("--ref_Nt", type=int, default=256, help="reference grid size in t; paper uses 256")
    p.add_argument("--ref_Nx", type=int, default=256, help="reference grid size in x; paper uses 256")
    p.add_argument("--ref_picard", type=int, default=6, help="Picard iterations per implicit reference step")
    p.add_argument("--newton_steps", type=int, default=1000, help="Gauss-Newton iterations")
    p.add_argument("--wdata", type=float, default=2.0, help="observation/data weight; paper uses w_data=2")
    p.add_argument("--wtheta", type=float, default=1.0, help="Newton damping weight on NN parameters; paper uses w_theta=1")
    p.add_argument("--lm", type=float, default=1e-8, help="small Levenberg diagonal damping for numerical stability")
    p.add_argument("--max_step_abs", type=float, default=2.0, help="clip absolute Newton update components")
    p.add_argument("--line_search", type=int, default=12, help="maximum backtracking steps")
    p.add_argument("--armijo", type=float, default=1e-6, help="backtracking sufficient decrease parameter")
    p.add_argument("--tol", type=float, default=1e-8, help="relative step tolerance")
    p.add_argument("--nimp", type=int, default=200, help="number of temperature observations")
    p.add_argument("--imposed", choices=["stripe", "random"], default="stripe")
    p.add_argument("--kmax", type=float, default=0.1, help="k(u) = kmax * sigmoid(q(u))")
    p.add_argument("--warm_init", action="store_true", help="initialize u with repeated initial condition instead of zeros")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", type=str, default="out_heat_newton_clean")
    return p.parse_args()


if __name__ == "__main__":
    solve_inverse_newton(parse_args())
