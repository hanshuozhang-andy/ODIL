#!/usr/bin/env python3
"""
Mac-friendly standalone reproduction of cselab/odil examples/wave/wave.py.

Goal:
  Reproduce the ODIL wave-equation example without importing odil, TensorFlow, or JAX.
  This avoids common macOS backend crashes while keeping the discretization and
  command-line style close to the official example.

Required packages:
  pip install numpy scipy matplotlib

Examples:
  python wave_mac_standalone.py --optimizer newton --Nt 25 --Nx 25 --outdir out_wave_mac
  python wave_mac_standalone.py --optimizer lbfgsb --Nt 25 --Nx 25 --outdir out_wave_mac_lbfgs

Notes:conda --version
  - This script implements the non-multigrid wave case.
  - Newton is implemented as a sparse linear solve because the wave residual is linear.
  - L-BFGS-B minimizes mean squared discrete residuals from zero initial guess.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Exact solution, consistent with the official GitHub wave.py source.
#
# Official source uses:
#   cos((x - t + 0.5) * k) + cos((x + t - 0.5) * k)
# and divides by 2*len([1,2,3,4,5]) = 10.
# ---------------------------------------------------------------------

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

        # d/dt cos((x-t+0.5)k) =  k sin((x-t+0.5)k)
        # d/dt cos((x+t-0.5)k) = -k sin((x+t-0.5)k)
        ut += k * np.sin(a) - k * np.sin(b)

    u /= 10.0
    ut /= 10.0

    return u, ut


# ---------------------------------------------------------------------
# 2. Grid and sparse residual construction.
#
# We build residual r(z) = A z - b.
# z contains all unknown grid values u[n, i], flattened in row-major order.
#
# This follows the official operator_wave structure:
#   u_t_tm  = (u - u_tm) / dt
#   u_t_tmm = (u_tm - u_tmm) / dt, replaced by init_ut at it == 1
#   u_tt    = (u_t_tm - u_t_tmm) / dt
#   u_xx    = (u_xm - 2*u_tm + u_xp) / dx^2
#   fu      = u_tt - u_xx
#   at it==0, fu = (u - (init_u + 0.5*dt*init_ut)) * kimp
# ---------------------------------------------------------------------

def extrap_quadh(u2, u1, ub):
    """
    Quadratic ghost-cell extrapolation used in the paper/source logic:
        ghost = (u2 - 6*u1 + 8*ub) / 3
    where ub is the boundary value.
    """
    return (u2 - 6.0 * u1 + 8.0 * ub) / 3.0


def make_grid(Nt, Nx):
    lower_t, upper_t = 0.0, 1.0
    lower_x, upper_x = -1.0, 1.0

    dt = (upper_t - lower_t) / Nt
    dx = (upper_x - lower_x) / Nx

    # Cell centers, consistent with ODIL Domain points.
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


def build_wave_system(grid, kimp=1.0, apply_right_bc=True):
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
        Add value * u[n, i] to the current residual row.

        If i is outside the spatial domain, replace by quadratic ghost-cell
        extrapolation using boundary data. This matches the halo-cell idea in
        the official wave.py and the paper.
        """
        const = 0.0

        if 0 <= i < Nx:
            add_coeff(n, i, value)

        elif i == -1:
            # Left ghost cell:
            # u_ghost = (u[n,1] - 6*u[n,0] + 8*U_left[n]) / 3
            add_coeff(n, 1, value / 3.0)
            add_coeff(n, 0, -6.0 * value / 3.0)
            const += value * (8.0 * left_u[n] / 3.0)

        elif i == Nx:
            if not apply_right_bc:
                # Fallback: zero-gradient-like copy. The official raw source
                # currently comments out the right boundary line, but the paper
                # includes it. Default is apply_right_bc=True.
                add_coeff(n, Nx - 1, value)
            else:
                # Right ghost cell:
                # u_ghost = (u[n,Nx-2] - 6*u[n,Nx-1] + 8*U_right[n]) / 3
                add_coeff(n, Nx - 2, value / 3.0)
                add_coeff(n, Nx - 1, -6.0 * value / 3.0)
                const += value * (8.0 * right_u[n] / 3.0)

        else:
            raise ValueError("Requested halo index is too far from the domain.")

        return const

    def add_minus_laplace_x(n, i):
        """
        Add -u_xx at time layer n:
            -(u[i-1] - 2*u[i] + u[i+1]) / dx^2
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
                # Official source:
                #   u0 = init_u + 0.5*dt*init_ut
                #   fu = (u - u0) * kimp
                u0 = init_u[i] + 0.5 * dt * init_ut[i]
                add_coeff(n, i, kimp)
                const += -kimp * u0

            elif n == 1:
                # u_tt = ((u^1-u^0)/dt - init_ut) / dt
                # fu = u_tt - u_xx(u^0)
                add_coeff(n, i, 1.0 / dt**2)
                add_coeff(n - 1, i, -1.0 / dt**2)
                const += -init_ut[i] / dt
                const += add_minus_laplace_x(n - 1, i)

            else:
                # u_tt = (u^n - 2u^{n-1} + u^{n-2}) / dt^2
                # fu = u_tt - u_xx(u^{n-1})
                add_coeff(n, i, 1.0 / dt**2)
                add_coeff(n - 1, i, -2.0 / dt**2)
                add_coeff(n - 2, i, 1.0 / dt**2)
                const += add_minus_laplace_x(n - 1, i)

            # Residual row is A*z + const, i.e. A*z - b with b=-const.
            rhs.append(-const)
            row += 1

    A = sp.csr_matrix((data, (rows, cols)), shape=(Nt * Nx, Nt * Nx))
    b = np.array(rhs, dtype=float)

    return A, b


# ---------------------------------------------------------------------
# 3. Optimizers.
# ---------------------------------------------------------------------

def rmse(u, ref_u):
    return float(np.sqrt(np.mean((u - ref_u) ** 2)))


def relative_rmse(u, ref_u):
    denom = np.sqrt(np.mean(ref_u ** 2))
    if denom == 0:
        return np.nan
    return float(np.sqrt(np.mean((u - ref_u) ** 2)) / denom)


def residual_loss(A, b, z):
    r = A @ z - b
    return float(np.mean(r ** 2))


def solve_newton(A, b):
    """
    For this linear wave-equation residual, ODIL+Newton reduces to a sparse
    linear solve. We first try spsolve; if the matrix is singular/ill-conditioned,
    we fall back to lsqr.
    """
    t0 = time.perf_counter()

    try:
        z = spla.spsolve(A, b)
        if not np.all(np.isfinite(z)):
            raise RuntimeError("spsolve returned non-finite values")
        solver = "spsolve"
    except Exception:
        out = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000)
        z = out[0]
        solver = "lsqr"

    elapsed = time.perf_counter() - t0
    return z, elapsed, solver


def solve_lbfgsb(A, b, maxiter=5000, gtol=1e-10, ftol=1e-14, report_every=10):
    N = A.shape[1]
    history = []
    start = time.perf_counter()

    def fun_and_jac(z):
        r = A @ z - b
        f = np.mean(r ** 2)
        g = (2.0 / r.size) * (A.T @ r)
        return f, np.asarray(g)

    def callback(z):
        it = len(history)
        if it % report_every == 0:
            f = residual_loss(A, b, z)
            history.append((it, f, time.perf_counter() - start))
        else:
            # Keep count without storing too much.
            history.append((it, np.nan, time.perf_counter() - start))

    result = opt.minimize(
        fun=lambda z: fun_and_jac(z)[0],
        x0=np.zeros(N, dtype=float),
        jac=lambda z: fun_and_jac(z)[1],
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": maxiter,
            "gtol": gtol,
            "ftol": ftol,
            "maxls": 50,
        },
    )

    elapsed = time.perf_counter() - start
    return result.x, elapsed, result, history


# ---------------------------------------------------------------------
# 4. Plotting and output.
# ---------------------------------------------------------------------

def write_log_header(path, args, grid):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# wave_mac_standalone.py\n")
        f.write(f"# optimizer={args.optimizer}\n")
        f.write(f"# Nt={args.Nt}, Nx={args.Nx}, kimp={args.kimp}\n")
        f.write(f"# dt={grid['dt']}, dx={grid['dx']}\n")
        f.write("# epoch, loss, error_u, rel_error_u, elapsed\n")


def append_log(path, epoch, loss, error_u, rel_error_u, elapsed):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{loss:.16e},{error_u:.16e},{rel_error_u:.16e},{elapsed:.8f}\n")


def save_data_csv(path, grid, u):
    Nt, Nx = grid["Nt"], grid["Nx"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "i", "t", "x", "u_pred", "u_ref", "error"])
        for n in range(Nt):
            for i in range(Nx):
                writer.writerow([
                    n,
                    i,
                    grid["t"][n],
                    grid["x"][i],
                    u[n, i],
                    grid["ref_u"][n, i],
                    u[n, i] - grid["ref_u"][n, i],
                ])


def plot_solution(grid, u, outpath, title=None):
    ref_u = grid["ref_u"]
    err = u - ref_u

    t = grid["t"]
    x = grid["x"]

    vmax = max(abs(np.max(ref_u)), abs(np.min(ref_u)), 1e-12)
    evmax = max(abs(np.max(err)), abs(np.min(err)), 1e-12)

    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(
        ref_u,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax1.set_title("Reference $U(x,t)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(
        u,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax2.set_title("ODIL solution $u$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(
        err,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="RdBu_r",
        vmin=-evmax,
        vmax=evmax,
    )
    ax3.set_title("Error $u-U$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(2, 2, 4)
    nslices = min(5, grid["Nt"])
    slice_ids = np.linspace(0, grid["Nt"] - 1, nslices, dtype=int)
    for n in slice_ids:
        ax4.plot(x, ref_u[n, :], linestyle="--", linewidth=1.4, label=f"ref t={t[n]:.2f}")
        ax4.plot(x, u[n, :], linewidth=1.0, label=f"pred t={t[n]:.2f}")
    ax4.set_title("Time slices")
    ax4.set_xlabel("x")
    ax4.set_ylabel("u")
    ax4.legend(fontsize=7, ncol=2)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_history(log_path, outpath):
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            rows.append([float(p) for p in parts])

    if not rows:
        return

    arr = np.array(rows)
    epoch = arr[:, 0]
    loss = arr[:, 1]
    err = arr[:, 2]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(epoch, loss, marker="o", label="loss")
    ax.semilogy(epoch, err, marker="s", label="RMSE")
    ax.set_xlabel("epoch / iteration")
    ax.set_ylabel("value")
    ax.set_title("Training history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. CLI.
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Mac-friendly standalone ODIL wave-equation reproduction.",
    )

    # Same core names as official examples/wave/wave.py.
    parser.add_argument("--Nt", type=int, default=64, help="Grid size in t")
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--kimp", type=float, default=1.0, help="Factor to impose initial conditions")
    parser.add_argument("--optimizer", choices=["newton", "lbfgsb"], default="newton")
    parser.add_argument("--multigrid", type=int, default=0, help="Accepted for compatibility; this standalone script supports only 0")
    parser.add_argument("--outdir", type=str, default="out_wave_mac")
    parser.add_argument("--plotext", type=str, default="png")
    parser.add_argument("--plot_title", type=int, default=1)
    parser.add_argument("--right_bc", type=int, default=1, help="Apply right boundary halo condition")

    # L-BFGS options.
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--gtol", type=float, default=1e-10)

    # Output options.
    parser.add_argument("--dump_data", action="store_true", help="Save solution values to CSV")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.multigrid != 0:
        print("Warning: this standalone Mac version implements the non-multigrid wave example only.")
        print("         Continuing with --multigrid 0 behavior.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(args.Nt, args.Nx)
    A, b = build_wave_system(grid, kimp=args.kimp, apply_right_bc=bool(args.right_bc))

    log_path = outdir / "train.log"
    write_log_header(log_path, args, grid)

    print(f"Grid: Nt={args.Nt}, Nx={args.Nx}, parameters={args.Nt * args.Nx}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Output directory: {outdir}")

    if args.optimizer == "newton":
        z, elapsed, solver = solve_newton(A, b)
        u = z.reshape(args.Nt, args.Nx)

        loss = residual_loss(A, b, z)
        err = rmse(u, grid["ref_u"])
        relerr = relative_rmse(u, grid["ref_u"])

        append_log(log_path, 1, loss, err, relerr, elapsed)

        print(f"Linear solver: {solver}")
        print(f"loss: {loss:.6e}")
        print(f"error_u RMSE: {err:.6e}")
        print(f"relative_error_u: {relerr:.6e}")
        print(f"time: {elapsed:.6f} s")

    else:
        z, elapsed, result, history = solve_lbfgsb(
            A,
            b,
            maxiter=args.maxiter,
            gtol=args.gtol,
            ftol=args.ftol,
            report_every=args.report_every,
        )
        u = z.reshape(args.Nt, args.Nx)

        # Write compact history. Some callback rows contain nan; recompute for stored epochs is expensive,
        # so for final use exact values below.
        for ep, f, tm in history:
            if np.isfinite(f):
                uu = z.reshape(args.Nt, args.Nx) if ep == history[-1][0] else u
                append_log(log_path, int(ep), f, rmse(uu, grid["ref_u"]), relative_rmse(uu, grid["ref_u"]), tm)

        loss = residual_loss(A, b, z)
        err = rmse(u, grid["ref_u"])
        relerr = relative_rmse(u, grid["ref_u"])
        append_log(log_path, int(result.nit), loss, err, relerr, elapsed)

        print(f"success: {result.success}")
        print(f"message: {result.message}")
        print(f"iterations: {result.nit}")
        print(f"loss: {loss:.6e}")
        print(f"error_u RMSE: {err:.6e}")
        print(f"relative_error_u: {relerr:.6e}")
        print(f"time: {elapsed:.6f} s")

    if args.dump_data:
        save_data_csv(outdir / "data_00000.csv", grid, u)

    if not args.no_plot:
        title = f"u optimizer={args.optimizer}, Nt={args.Nt}, Nx={args.Nx}" if args.plot_title else None
        plot_solution(grid, u, outdir / f"u_00000.{args.plotext}", title=title)
        plot_history(log_path, outdir / f"history.{args.plotext}")
        print(f"Saved plot: {outdir / f'u_00000.{args.plotext}'}")
        print(f"Saved history: {outdir / f'history.{args.plotext}'}")

    with open(outdir / "done", "w", encoding="utf-8") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
