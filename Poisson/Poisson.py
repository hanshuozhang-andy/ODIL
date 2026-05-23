#!/usr/bin/env python3
"""
Standalone ODIL-style solver for the 2D Poisson example in pgae005.pdf.

Problem:
    u_xx + u_yy = f(x, y),  (x,y) in [0,1]^2
    u = 0 on boundary

Reference solution:
    u(x,y) = sin(pi * (k*x)^2) * sin(pi*y)

This code follows the official ODIL Poisson example as closely as possible,
but removes the dependency on the official odil package / TensorFlow / JAX.

Author note:
    - Grid: cell-centered N x N grid
    - Boundary condition: zero Dirichlet using quadratic half-cell ghost extrapolation
    - Loss: mean squared finite-difference residual
    - Optimizer: Adam
    - Multigrid decomposition: u = u_fine + interpolate(u_coarse) + ...
"""

import os

# Similar to official run script: avoid excessive threading on Mac.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--N", type=int, default=64, help="Grid size, N x N")
    parser.add_argument("--k", type=float, default=4.0, help="Oscillation parameter k")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of Adam epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument(
        "--levels",
        type=int,
        default=6,
        help="Number of multigrid levels. Use 1 to disable multigrid.",
    )
    parser.add_argument(
        "--rhs",
        type=str,
        default="exact",
        choices=["exact", "discrete"],
        help="Use exact continuous RHS or discrete RHS from reference solution",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32. Default is float64, closer to official ODIL.",
    )
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="out_poisson_mac")

    return parser.parse_args()


def make_cell_center_points(N, dtype):
    """
    Official ODIL uses cell-centered points:
        x_i = (i + 0.5) / N
    """
    x = (torch.arange(N, dtype=dtype) + 0.5) / N
    y = (torch.arange(N, dtype=dtype) + 0.5) / N
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X, Y


def get_ref_u(X, Y, k):
    """
    Same as official get_ref_u(..., ref='osc'):

        u = sin(pi * (k*x)^2) * sin(pi*y)
    """
    pi = torch.pi
    return torch.sin(pi * (k * X) ** 2) * torch.sin(pi * Y)


def get_ref_rhs_exact(X, Y, k):
    """
    Exact continuous RHS:
        f = u_xx + u_yy

    For u = sin(pi*(k*x)^2) sin(pi*y),

        u_xx = [2*pi*k^2*cos(pi*k^2*x^2)
                - 4*k^4*pi^2*x^2*sin(pi*k^2*x^2)] sin(pi*y)

        u_yy = -pi^2 sin(pi*k^2*x^2) sin(pi*y)
    """
    pi = torch.pi
    sin = torch.sin
    cos = torch.cos

    return (
        (
            (-4 * k**4 * pi**2 * X**2 - pi**2) * sin(k**2 * pi * X**2)
            + 2 * k**2 * pi * cos(k**2 * pi * X**2)
        )
        * sin(pi * Y)
    )


def apply_zero_dirichlet_bc(u):
    """
    Apply zero Dirichlet boundary condition through ghost cells.

    The official ODIL code uses extrap_quadh:

        ghost = (inner_neighbor - 6 * current + 8 * boundary_value) / 3

    Here boundary_value = 0.

    u is cell-centered:
        boundary is at x=0, first cell center at x=dx/2,
        ghost cell center is at x=-dx/2.
    """
    nx, ny = u.shape
    assert nx == ny, "This script assumes an N x N grid."

    ix = torch.arange(nx, device=u.device).reshape(nx, 1)
    iy = torch.arange(ny, device=u.device).reshape(1, ny)

    uxm0 = torch.roll(u, shifts=1, dims=0)
    uxp0 = torch.roll(u, shifts=-1, dims=0)
    uym0 = torch.roll(u, shifts=1, dims=1)
    uyp0 = torch.roll(u, shifts=-1, dims=1)

    # x-direction ghost cells.
    uxm = torch.where(ix == 0, (uxp0 - 6.0 * u) / 3.0, uxm0)
    uxp = torch.where(ix == nx - 1, (uxm0 - 6.0 * u) / 3.0, uxp0)

    # y-direction ghost cells.
    uym = torch.where(iy == 0, (uyp0 - 6.0 * u) / 3.0, uym0)
    uyp = torch.where(iy == ny - 1, (uym0 - 6.0 * u) / 3.0, uyp0)

    return uxm, uxp, uym, uyp


def discrete_laplacian(u):
    """
    Five-point finite-difference Laplacian with zero Dirichlet ghost cells.

        Lap(u)_ij =
            (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / dx^2
          + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / dy^2
    """
    nx, ny = u.shape
    dx = 1.0 / nx
    dy = 1.0 / ny

    uxm, uxp, uym, uyp = apply_zero_dirichlet_bc(u)

    u_xx = (uxp - 2.0 * u + uxm) / dx**2
    u_yy = (uyp - 2.0 * u + uym) / dy**2

    return u_xx + u_yy


class MultigridField(torch.nn.Module):
    """
    ODIL-style multigrid decomposition:

        u = u_1 + T_1 u_2 + T_1 T_2 u_3 + ...

    where u_1 is fine-grid variable, u_2, u_3, ... are coarser variables,
    and T means interpolation to the finer grid.

    This is not a full multigrid solver. It is an over-parameterization
    used to make Adam converge faster, similar to mODIL.
    """

    def __init__(self, N, levels, dtype):
        super().__init__()

        shapes = []
        n = N
        for _ in range(max(levels, 1)):
            shapes.append(n)
            if n <= 2:
                break
            n = n // 2

        self.shapes = shapes

        self.terms = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros((s, s), dtype=dtype))
                for s in self.shapes
            ]
        )

    def forward(self):
        # Start from coarsest field and interpolate upward.
        u = self.terms[-1]

        for term in reversed(self.terms[:-1]):
            u = F.interpolate(
                u[None, None, :, :],
                size=term.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            u = u + term

        return u


def compute_errors(u, ref_u):
    rmse = torch.sqrt(torch.mean((u - ref_u) ** 2))
    ref_rms = torch.sqrt(torch.mean(ref_u**2))
    rel_rmse = rmse / ref_rms
    return rmse.item(), rel_rmse.item()


def save_plots(outdir, args, u, ref_u, rhs, history):
    os.makedirs(outdir, exist_ok=True)

    u_np = u.detach().cpu().numpy()
    ref_np = ref_u.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    err_np = u_np - ref_np

    # Field figure.
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)

    items = [
        (ref_np, "reference u"),
        (u_np, "ODIL-style u"),
        (err_np, "u - reference"),
        (rhs_np, "rhs f"),
    ]

    for ax, (data, title) in zip(axes, items):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    field_path = os.path.join(outdir, f"poisson_k{args.k:g}_field.png")
    fig.savefig(field_path, dpi=200)
    plt.close(fig)

    # Training history.
    hist = np.array(history, dtype=float)
    epochs = hist[:, 0]
    loss = hist[:, 1]
    rmse = hist[:, 2]
    rel_rmse = hist[:, 3]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.loglog(epochs + 1, rmse, marker="o", label="RMSE")
    ax.loglog(epochs + 1, rel_rmse, marker="s", label="relative RMSE")
    ax.set_xlabel("epoch + 1")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    train_path = os.path.join(outdir, f"poisson_k{args.k:g}_train.png")
    fig.savefig(train_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved figures:")
    print(f"  {field_path}")
    print(f"  {train_path}")


def main():
    args = parse_args()

    torch.set_num_threads(1)

    dtype = torch.float32 if args.float32 else torch.float64

    N = args.N
    k = args.k

    X, Y = make_cell_center_points(N, dtype=dtype)
    ref_u = get_ref_u(X, Y, k)

    if args.rhs == "exact":
        rhs = get_ref_rhs_exact(X, Y, k)
    else:
        # Discrete RHS makes the reference solution exactly satisfy
        # the chosen discrete Laplacian.
        with torch.no_grad():
            rhs = discrete_laplacian(ref_u).detach()

    model = MultigridField(N=N, levels=args.levels, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=" * 70)
    print("ODIL-style 2D Poisson solver")
    print(f"N          = {N} x {N}")
    print(f"k          = {k}")
    print(f"rhs        = {args.rhs}")
    print(f"epochs     = {args.epochs}")
    print(f"lr         = {args.lr}")
    print(f"dtype      = {dtype}")
    print(f"MG shapes  = {model.shapes}")
    print("=" * 70)

    history = []
    t0 = time.time()

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        u = model()
        residual = discrete_laplacian(u) - rhs
        loss = torch.mean(residual**2)

        if epoch % args.report_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                rmse, rel_rmse = compute_errors(u, ref_u)
                history.append((epoch, loss.item(), rmse, rel_rmse))
                print(
                    f"epoch={epoch:6d}  "
                    f"loss={loss.item():.6e}  "
                    f"rmse={rmse:.6e}  "
                    f"rel_rmse={rel_rmse:.6e}"
                )

        if epoch == args.epochs:
            break

        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    print(f"\nDone. Elapsed time: {elapsed:.2f} seconds")

    with torch.no_grad():
        final_u = model().detach()

    save_plots(args.outdir, args, final_u, ref_u, rhs, history)


if __name__ == "__main__":
    main()