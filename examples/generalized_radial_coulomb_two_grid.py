import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.radial_Coulomb import (
    radial_Coulomb_femdvr_two_grid,
)


def build_femdvr(r_max, n_elem, points_per_elem):
    nodes = np.linspace(0.0, r_max, n_elem + 1)
    n_points = np.ones((n_elem,), dtype=int) * points_per_elem
    return FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)


def analytic_weighted_kernel(r1_grid, r2_grid, L, gamma):
    r1 = r1_grid.r[1:-1]
    r2 = r2_grid.r[1:-1]
    w2 = r2_grid.weights[1:-1]
    gamma_abs = abs(gamma)
    parity = -1.0 if gamma < 0.0 and L % 2 else 1.0

    source_radius = gamma_abs * r2
    r_less = np.minimum(r1[:, np.newaxis], source_radius[np.newaxis, :])
    r_greater = np.maximum(r1[:, np.newaxis], source_radius[np.newaxis, :])
    return (
        parity
        * (4 * np.pi / (2 * L + 1))
        * r_less**L
        / r_greater ** (L + 1)
        * w2[np.newaxis, :]
    )


r_grid = build_femdvr(r_max=20.0, n_elem=4, points_per_elem=11)
R_grid = build_femdvr(r_max=6.0, n_elem=3, points_per_elem=11)

L_max = 3
gamma = 0.5
terms = [(1.0, 1.0, gamma)]

G_solved = radial_Coulomb_femdvr_two_grid(
    r_grid, R_grid, n_L=L_max + 1, charge_gamma_terms=terms
)

G_exact = np.zeros_like(G_solved)
for L in range(L_max + 1):
    G_exact[L] = analytic_weighted_kernel(r_grid, R_grid, L, gamma)

error = G_solved - G_exact

fig, axes = plt.subplots(
    L_max + 1, 3, figsize=(12, 10), constrained_layout=True
)
extent = [R_grid.r[1], R_grid.r[-2], r_grid.r[1], r_grid.r[-2]]

for L in range(L_max + 1):
    vmax = max(np.max(np.abs(G_solved[L])), np.max(np.abs(G_exact[L])))
    err_vmax = np.max(np.abs(error[L]))

    panels = [
        (G_solved[L], "Poisson", None, vmax),
        (G_exact[L], "analytic", None, vmax),
        (
            error[L],
            "difference",
            SymLogNorm(
                linthresh=max(err_vmax * 1e-8, np.finfo(float).tiny),
                vmin=-err_vmax,
                vmax=err_vmax,
                base=10,
            )
            if err_vmax > 0
            else None,
            err_vmax,
        ),
    ]

    for col, (values, title, norm, scale) in enumerate(panels):
        kwargs = {"norm": norm}
        if norm is None and scale > 0:
            kwargs.update({"vmin": -scale, "vmax": scale})

        image = axes[L, col].imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            **kwargs,
        )
        axes[L, col].set_title(f"L={L}: {title}")
        axes[L, col].set_xlabel("R")
        axes[L, col].set_ylabel("r")
        fig.colorbar(image, ax=axes[L, col])

fig.suptitle(rf"Two-grid FEMDVR radial Coulomb, $\gamma={gamma}$")
plt.show()
