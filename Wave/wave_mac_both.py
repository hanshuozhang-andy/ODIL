#!/usr/bin/env python3
"""
Mac-friendly standalone reproduction of the ODIL wave-equation example.

This version can run BOTH:
  1. ODIL + Newton
  2. ODIL + L-BFGS-B

in one command, and generate both result figures automatically.

Required packages:
  pip install numpy scipy matplotlib

Recommended command:
  python3 wave_mac_both.py --Nt 25 --Nx 25 --outdir out_wave_both

Outputs:
  out_wave_both/newton/u_00000.png
  out_wave_both/lbfgsb/u_00000.png
  out_wave_both/comparison_newton_lbfgsb.png
  out_wave_both/summary.txt
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Exact solution
#    Kept close to official examples/wave/wave.py:
#    cos((x - t + 0.5)k) + cos((x + t - 0.5)k), divided by 10.
# ============================================================

def get_exact(t, x):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    u = np.zeros_like(x, dtype=float)
    ut = np.zeros_like(x, dtype=float)

    for i in [1, 2, 3, 4, 5]:
        k = i * np.pi
        a = (x - t + 0.5) * k
        b = (x + t - 0.5) * k

        u += np.cos(a) + np.cos(b)

        # derivative with respect to t
        ut += k * np.sin(a) - k * np.sin(b)

    u /= 10.0
    ut /= 10.0

    return u, ut


# ============================================================
# 2. Grid and sparse ODIL wave residual
# ============================================================

def make_grid(Nt, Nx):
    lower_t, upper_t = 0.0, 1.0
    lower_x, upper_x = -1.0, 1.0

    dt = (upper_t - lower_t) / Nt
    dx = (upper_x - lower_x) / Nx

    t = lower_t + (np.arange(Nt) + 0.5) * dt
    x = lower_x + (np.arange(Nx) + 0.5) * dx

    tt, xx = np.meshgrid(t, x, indexing="ij")

    ref_u, ref_ut = get_exact(tt, xx)
    left_u, _ = get_exact(t, np.full_like(t, lower_x))
    right_u, _ = get_exact(t, np.full_like(t, upper_x))
    init_u, init_ut = get_exact(np.zeros_like(x), x)

    return {
        "Nt": Nt,
        "Nx": Nx,
        "dt": dt,
        "dx": dx,
        "t": t,
        "x": x,
        "tt": tt,
        "xx": xx,
        "ref_u": ref_u,
        "ref_ut": ref_ut,
        "left_u": left_u,
        "right_u": right_u,
        "init_u": init_u,
        "init_ut": init_ut,
    }


def build_wave_system(grid, kimp=1.0):
    Nt = grid["Nt"]
    Nx = grid["Nx"]
    dt = grid["dt"]
    dx = grid["dx"]

    left_u = grid["left_u"]
    right_u = grid["right_u"]
    init_u = grid["init_u"]
    init_ut = grid["init_ut"]

    rows, cols, data = [], [], []
    rhs = []
    row = 0

    def index(n, i):
        return n * Nx + i

    def add_coeff(n, i, value):
        nonlocal row
        rows.append(row)
        cols.append(index(n, i))
        data.append(float(value))

    def add_u_with_halo(n, i, value):
        """
        Add value * u[n,i] to the current residual row.

        Boundary conditions are included through quadratic ghost cells:
          u_left_ghost  = (u_2 - 6u_1 + 8U(-1,t)) / 3
          u_right_ghost = (u_{Nx-1} - 6u_{Nx} + 8U(1,t)) / 3
        """
        const = 0.0

        if 0 <= i < Nx:
            add_coeff(n, i, value)

        elif i == -1:
            add_coeff(n, 1, value / 3.0)
            add_coeff(n, 0, -6.0 * value / 3.0)
            const += value * (8.0 * left_u[n] / 3.0)

        elif i == Nx:
            add_coeff(n, Nx - 2, value / 3.0)
            add_coeff(n, Nx - 1, -6.0 * value / 3.0)
            const += value * (8.0 * right_u[n] / 3.0)

        else:
            raise ValueError("Halo index too far away.")

        return const

    def add_minus_laplace_x(n, i):
        """
        Add -u_xx at time layer n:
          - (u[i-1] - 2u[i] + u[i+1]) / dx^2
        """
        const = 0.0
        const += add_u_with_halo(n, i - 1, -1.0 / dx**2)
        const += add_u_with_halo(n, i,      2.0 / dx**2)
        const += add_u_with_halo(n, i + 1, -1.0 / dx**2)
        return const

    for n in range(Nt):
        for i in range(Nx):
            const = 0.0

            if n == 0:
                # First time layer from initial condition:
                # u(t=dt/2) ≈ U(x,0) + 0.5*dt*U_t(x,0)
                u0 = init_u[i] + 0.5 * dt * init_ut[i]
                add_coeff(n, i, kimp)
                const += -kimp * u0

            elif n == 1:
                # ((u^1-u^0)/dt - U_t(x,0))/dt - u_xx(u^0) = 0
                add_coeff(n, i, 1.0 / dt**2)
                add_coeff(n - 1, i, -1.0 / dt**2)
                const += -init_ut[i] / dt
                const += add_minus_laplace_x(n - 1, i)

            else:
                # (u^n - 2u^{n-1} + u^{n-2})/dt^2 - u_xx(u^{n-1}) = 0
                add_coeff(n, i, 1.0 / dt**2)
                add_coeff(n - 1, i, -2.0 / dt**2)
                add_coeff(n - 2, i, 1.0 / dt**2)
                const += add_minus_laplace_x(n - 1, i)

            rhs.append(-const)
            row += 1

    A = sp.csr_matrix((data, (rows, cols)), shape=(Nt * Nx, Nt * Nx))
    b = np.array(rhs, dtype=float)

    return A, b


# ============================================================
# 3. Metrics and solvers
# ============================================================

def rmse(u, ref_u):
    return float(np.sqrt(np.mean((u - ref_u) ** 2)))


def relative_rmse(u, ref_u):
    denom = np.sqrt(np.mean(ref_u ** 2))
    return float(np.sqrt(np.mean((u - ref_u) ** 2)) / denom)


def residual_loss(A, b, z):
    r = A @ z - b
    return float(np.mean(r ** 2))


def solve_newton(A, b):
    t0 = time.perf_counter()

    try:
        z = spla.spsolve(A, b)
        if not np.all(np.isfinite(z)):
            raise RuntimeError("spsolve returned non-finite values")
        solver = "spsolve"
    except Exception:
        result = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000)
        z = result[0]
        solver = "lsqr"

    elapsed = time.perf_counter() - t0
    return z, elapsed, solver


def solve_lbfgsb(A, b, maxiter=5000, ftol=1e-14, gtol=1e-10):
    N = A.shape[1]
    history = []
    t0 = time.perf_counter()

    def fun_and_jac(z):
        r = A @ z - b
        f = np.mean(r ** 2)
        g = (2.0 / r.size) * (A.T @ r)
        return f, np.asarray(g)

    def callback(z):
        ep = len(history) + 1
        f = residual_loss(A, b, z)
        history.append((ep, f, time.perf_counter() - t0))

    result = opt.minimize(
        fun=lambda z: fun_and_jac(z)[0],
        x0=np.zeros(N, dtype=float),
        jac=lambda z: fun_and_jac(z)[1],
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "maxls": 50,
        },
    )

    elapsed = time.perf_counter() - t0
    return result.x, elapsed, result, history


# ============================================================
# 4. Plotting
# ============================================================

def plot_solution(grid, u, outpath, title):
    ref_u = grid["ref_u"]
    err = u - ref_u
    x = grid["x"]
    t = grid["t"]

    vmax = max(abs(ref_u.max()), abs(ref_u.min()), abs(u.max()), abs(u.min()), 1e-12)
    evmax = max(abs(err.max()), abs(err.min()), 1e-12)

    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(
        ref_u, origin="lower", aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax1.set_title("Reference")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(
        u, origin="lower", aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax2.set_title(title)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(
        err, origin="lower", aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r", vmin=-evmax, vmax=evmax
    )
    ax3.set_title("Error")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(2, 2, 4)
    slice_ids = np.linspace(0, grid["Nt"] - 1, min(5, grid["Nt"]), dtype=int)
    for n in slice_ids:
        ax4.plot(x, ref_u[n], linestyle="--", linewidth=1.3, label=f"ref t={t[n]:.2f}")
        ax4.plot(x, u[n], linewidth=1.0, label=f"pred t={t[n]:.2f}")
    ax4.set_title("Time slices")
    ax4.set_xlabel("x")
    ax4.set_ylabel("u")
    ax4.legend(fontsize=7, ncol=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_comparison(grid, u_newton, u_lbfgsb, outpath):
    ref_u = grid["ref_u"]
    x = grid["x"]
    t = grid["t"]

    vmax = max(
        abs(ref_u.max()), abs(ref_u.min()),
        abs(u_newton.max()), abs(u_newton.min()),
        abs(u_lbfgsb.max()), abs(u_lbfgsb.min()),
        1e-12
    )

    diff = u_lbfgsb - u_newton
    dvmax = max(abs(diff.max()), abs(diff.min()), 1e-12)

    fig = plt.figure(figsize=(14, 8))

    items = [
        ("Reference", ref_u, -vmax, vmax),
        ("ODIL + Newton", u_newton, -vmax, vmax),
        ("ODIL + L-BFGS-B", u_lbfgsb, -vmax, vmax),
        ("L-BFGS-B - Newton", diff, -dvmax, dvmax),
    ]

    for idx, (title, arr, vmin, vmax_) in enumerate(items, start=1):
        ax = fig.add_subplot(2, 2, idx)
        im = ax.imshow(
            arr, origin="lower", aspect="auto",
            extent=[x[0], x[-1], t[0], t[-1]],
            cmap="RdBu_r", vmin=vmin, vmax=vmax_
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Wave equation: ODIL Newton vs ODIL L-BFGS-B")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch,loss,elapsed\n")
        for ep, loss, elapsed in rows:
            f.write(f"{ep},{loss:.16e},{elapsed:.8f}\n")


def save_summary(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


# ============================================================
# 5. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run ODIL Newton and ODIL L-BFGS-B wave equation in one command.",
    )
    parser.add_argument("--Nt", type=int, default=25)
    parser.add_argument("--Nx", type=int, default=25)
    parser.add_argument("--kimp", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="out_wave_both")
    parser.add_argument("--maxiter", type=int, default=5000, help="L-BFGS-B max iterations")
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--gtol", type=float, default=1e-10)
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    newton_dir = outdir / "newton"
    lbfgs_dir = outdir / "lbfgsb"
    newton_dir.mkdir(parents=True, exist_ok=True)
    lbfgs_dir.mkdir(parents=True, exist_ok=True)

    print("Building grid and ODIL residual system...")
    grid = make_grid(args.Nt, args.Nx)
    A, b = build_wave_system(grid, kimp=args.kimp)

    print(f"Grid: Nt={args.Nt}, Nx={args.Nx}, parameters={args.Nt * args.Nx}")
    print("Running ODIL + Newton...")
    z_newton, time_newton, solver_newton = solve_newton(A, b)
    u_newton = z_newton.reshape(args.Nt, args.Nx)

    loss_newton = residual_loss(A, b, z_newton)
    err_newton = rmse(u_newton, grid["ref_u"])
    rel_newton = relative_rmse(u_newton, grid["ref_u"])

    plot_solution(
        grid,
        u_newton,
        newton_dir / "u_00000.png",
        f"ODIL + Newton ({solver_newton})"
    )

    print("Running ODIL + L-BFGS-B...")
    z_lbfgsb, time_lbfgsb, result_lbfgsb, history_lbfgsb = solve_lbfgsb(
        A,
        b,
        maxiter=args.maxiter,
        ftol=args.ftol,
        gtol=args.gtol,
    )
    u_lbfgsb = z_lbfgsb.reshape(args.Nt, args.Nx)

    loss_lbfgsb = residual_loss(A, b, z_lbfgsb)
    err_lbfgsb = rmse(u_lbfgsb, grid["ref_u"])
    rel_lbfgsb = relative_rmse(u_lbfgsb, grid["ref_u"])

    plot_solution(
        grid,
        u_lbfgsb,
        lbfgs_dir / "u_00000.png",
        "ODIL + L-BFGS-B"
    )
    save_log(lbfgs_dir / "train.log", history_lbfgsb)

    plot_comparison(
        grid,
        u_newton,
        u_lbfgsb,
        outdir / "comparison_newton_lbfgsb.png"
    )

    summary = {
        "Nt": args.Nt,
        "Nx": args.Nx,
        "parameters": args.Nt * args.Nx,

        "newton_solver": solver_newton,
        "newton_loss": f"{loss_newton:.6e}",
        "newton_rmse": f"{err_newton:.6e}",
        "newton_relative_rmse": f"{rel_newton:.6e}",
        "newton_time_seconds": f"{time_newton:.6f}",

        "lbfgsb_success": str(result_lbfgsb.success),
        "lbfgsb_message": str(result_lbfgsb.message),
        "lbfgsb_iterations": str(result_lbfgsb.nit),
        "lbfgsb_loss": f"{loss_lbfgsb:.6e}",
        "lbfgsb_rmse": f"{err_lbfgsb:.6e}",
        "lbfgsb_relative_rmse": f"{rel_lbfgsb:.6e}",
        "lbfgsb_time_seconds": f"{time_lbfgsb:.6f}",

        "difference_rmse_lbfgsb_minus_newton": f"{rmse(u_lbfgsb, u_newton):.6e}",
    }

    save_summary(outdir / "summary.txt", summary)

    print("\nDone.")
    print(f"Newton figure:       {newton_dir / 'u_00000.png'}")
    print(f"L-BFGS-B figure:     {lbfgs_dir / 'u_00000.png'}")
    print(f"Comparison figure:   {outdir / 'comparison_newton_lbfgsb.png'}")
    print(f"Summary:             {outdir / 'summary.txt'}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
