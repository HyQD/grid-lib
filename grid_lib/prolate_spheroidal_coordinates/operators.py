from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .grid import ProlateSpheroidalGrid


@dataclass(frozen=True)
class SeparableKineticMatrices:
    """One-dimensional pieces of the weak-form prolate kinetic energy."""

    xi_stiffness: sparse.csr_matrix
    eta_stiffness: sparse.csr_matrix
    xi_weight: sparse.csr_matrix
    eta_weight: sparse.csr_matrix


def validate_axis_free_grid(grid: ProlateSpheroidalGrid) -> None:
    """Require a prolate grid without molecular-axis endpoint nodes."""

    if np.any(np.isclose(grid.xi, 1.0, rtol=0.0, atol=1e-13)) or np.any(
        np.isclose(np.abs(grid.eta), 1.0, rtol=0.0, atol=1e-13)
    ):
        raise ValueError(
            "Prolate operators require an axis-free Radau-type grid without "
            "xi = 1 or eta = +/-1 nodes. Use "
            "setup_prolate_spheroidal_grid(...)."
        )


def weighted_stiffness_matrix(
    points: np.ndarray,
    weights: np.ndarray,
    derivative_matrix: np.ndarray,
    coefficient: np.ndarray,
) -> sparse.csr_matrix:
    """Return D.T @ diag(weights * coefficient) @ D."""

    points = np.asarray(points)
    weights = np.asarray(weights)
    coefficient = np.asarray(coefficient)
    derivative_matrix = np.asarray(derivative_matrix)

    if points.ndim != 1:
        raise ValueError("points must be one-dimensional.")
    if weights.shape != points.shape:
        raise ValueError("weights must have the same shape as points.")
    if coefficient.shape != points.shape:
        raise ValueError("coefficient must have the same shape as points.")
    if derivative_matrix.shape != (points.size, points.size):
        raise ValueError("derivative_matrix has an incompatible shape.")

    derivative = sparse.csr_matrix(derivative_matrix)
    weighted_coefficient = sparse.diags(weights * coefficient, format="csr")
    return (derivative.T @ weighted_coefficient @ derivative).tocsr()


def xi_derivative_matrix(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
) -> np.ndarray:
    """Return the xi derivative matrix for the even/odd-|m| basis.

    Even |m| uses the ordinary cardinal basis. Odd |m| uses
    sqrt(xi**2 - 1) l_i(xi) / sqrt(xi_i**2 - 1), so the nodal values remain
    cardinal while the derivative includes the square-root axis factor.
    """

    validate_axis_free_grid(grid)
    if abs(int(m)) % 2 == 0:
        return np.asarray(grid.xi_D1)

    factor = np.sqrt(grid.xi**2 - 1.0)
    if np.any(factor <= 0.0):
        raise ValueError(
            "Odd-|m| sectors require xi nodes away from xi = 1. Use "
            "setup_prolate_spheroidal_grid(...)."
        )

    factor_derivative = grid.xi / factor
    derivative = (factor[:, np.newaxis] * grid.xi_D1) / factor[np.newaxis, :]
    diagonal = np.diag_indices_from(derivative)
    derivative[diagonal] += factor_derivative / factor
    return derivative


def eta_derivative_matrix(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
) -> np.ndarray:
    """Return the eta derivative matrix for the even/odd-|m| basis."""

    validate_axis_free_grid(grid)
    if abs(int(m)) % 2 == 0:
        return np.asarray(grid.eta_D1)

    factor = np.sqrt(1.0 - grid.eta**2)
    if np.any(factor <= 0.0):
        raise ValueError(
            "Odd-|m| sectors require eta nodes away from eta = +/-1. Use "
            "setup_prolate_spheroidal_grid(...)."
        )

    factor_derivative = -grid.eta / factor
    derivative = (factor[:, np.newaxis] * grid.eta_D1) / factor[np.newaxis, :]
    diagonal = np.diag_indices_from(derivative)
    derivative[diagonal] += factor_derivative / factor
    return derivative


def separable_kinetic_matrices(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
) -> SeparableKineticMatrices:
    """Build one-dimensional weak-form matrices for one m sector."""

    validate_axis_free_grid(grid)
    xi_stiffness = weighted_stiffness_matrix(
        grid.xi,
        grid.xi_weights,
        xi_derivative_matrix(grid, m),
        grid.xi**2 - 1.0,
    )
    eta_stiffness = weighted_stiffness_matrix(
        grid.eta,
        grid.eta_weights,
        eta_derivative_matrix(grid, m),
        1.0 - grid.eta**2,
    )
    return SeparableKineticMatrices(
        xi_stiffness=xi_stiffness,
        eta_stiffness=eta_stiffness,
        xi_weight=sparse.diags(grid.xi_weights, format="csr"),
        eta_weight=sparse.diags(grid.eta_weights, format="csr"),
    )


def derivative_kinetic_energy_matrix(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
    matrices: SeparableKineticMatrices | None = None,
) -> sparse.csr_matrix:
    """Return the separable xi/eta derivative part of the kinetic energy."""

    validate_axis_free_grid(grid)
    if matrices is None:
        matrices = separable_kinetic_matrices(grid, m)

    kinetic = sparse.kron(
        matrices.xi_stiffness,
        matrices.eta_weight,
        format="csr",
    )
    kinetic += sparse.kron(
        matrices.xi_weight,
        matrices.eta_stiffness,
        format="csr",
    )
    return (0.5 * grid.a * kinetic).tocsr()


def overlap_diagonal(
    grid: ProlateSpheroidalGrid,
) -> np.ndarray:
    """Return the diagonal weak-form overlap weights, flattened in C order."""

    validate_axis_free_grid(grid)
    return (
        grid.a**3
        * grid.quadrature_weights_2d()
        * grid.metric_factor()
    ).ravel()


def overlap_matrix(
    grid: ProlateSpheroidalGrid,
) -> sparse.csr_matrix:
    return sparse.diags(overlap_diagonal(grid), format="csr")


def centrifugal_diagonal(
    grid: ProlateSpheroidalGrid,
    m: int,
    active_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weak-form m-dependent centrifugal diagonal."""

    validate_axis_free_grid(grid)
    abs_m = abs(int(m))
    if abs_m == 0:
        size = grid.size if active_indices is None else np.asarray(active_indices).size
        return np.zeros(size)

    xi, eta = grid.mesh()
    denominator = (xi**2 - 1.0) * (1.0 - eta**2)
    diagonal = (
        0.5
        * grid.a
        * abs_m**2
        * grid.quadrature_weights_2d()
        * grid.metric_factor()
        / denominator
    ).ravel()
    if active_indices is not None:
        diagonal = diagonal[np.asarray(active_indices)]

    if not np.all(np.isfinite(diagonal)):
        raise ValueError(
            "The centrifugal diagonal is singular on this grid. Use an "
            "axis-free Radau grid before forming nonzero-m sectors."
        )
    return diagonal


def centrifugal_matrix(
    grid: ProlateSpheroidalGrid,
    m: int,
    active_indices: np.ndarray | None = None,
) -> sparse.csr_matrix:
    return sparse.diags(
        centrifugal_diagonal(grid, m, active_indices=active_indices),
        format="csr",
    )


def electron_nuclear_coulomb_diagonal(
    grid: ProlateSpheroidalGrid,
    Z1: float = 1.0,
    Z2: float = 1.0,
) -> np.ndarray:
    """Return the weak-form two-center Coulomb diagonal.

    The physical potential is

        -((Z1 + Z2) xi + (Z2 - Z1) eta) / (a (xi**2 - eta**2)).

    Multiplication by the prolate volume element leaves the finite weak-form
    diagonal used here. Setting Z1 = Z2 = 0 returns zeros.
    """

    validate_axis_free_grid(grid)
    xi, eta = grid.mesh()
    diagonal = (
        -grid.a**2
        * grid.quadrature_weights_2d()
        * ((Z1 + Z2) * xi + (Z2 - Z1) * eta)
    )
    return diagonal.ravel()


def electron_nuclear_coulomb_matrix(
    grid: ProlateSpheroidalGrid,
    Z1: float = 1.0,
    Z2: float = 1.0,
) -> sparse.csr_matrix:
    return sparse.diags(
        electron_nuclear_coulomb_diagonal(grid, Z1=Z1, Z2=Z2),
        format="csr",
    )


def kinetic_energy_matrix(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
    matrices: SeparableKineticMatrices | None = None,
    active_indices: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Return the weak-form kinetic-energy matrix for one m sector."""

    kinetic = derivative_kinetic_energy_matrix(grid, m=m, matrices=matrices)
    if active_indices is not None:
        kinetic = restrict_matrix(kinetic, active_indices)
    if m != 0:
        kinetic += centrifugal_matrix(grid, m, active_indices=active_indices)
    return kinetic.tocsr()


def active_mask(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
    xi_outer_dirichlet: bool = True,
    drop_zero_overlap: bool = True,
) -> np.ndarray:
    """Return a boolean active-node mask for one m sector.

    The axis behavior is built into the basis for odd |m|. The Radau grid has
    no xi = 1 or eta = +/-1 endpoint nodes, so the default active-space
    truncation removes only the finite xi_max boundary and any zero-overlap
    nodes.
    """

    validate_axis_free_grid(grid)
    mask = np.ones(grid.shape, dtype=bool)

    if xi_outer_dirichlet:
        mask[-1, :] = False

    if drop_zero_overlap:
        mask &= overlap_diagonal(grid).reshape(grid.shape) > 0.0

    return mask


def active_indices(
    grid: ProlateSpheroidalGrid,
    m: int = 0,
    xi_outer_dirichlet: bool = True,
    drop_zero_overlap: bool = True,
) -> np.ndarray:
    return np.flatnonzero(
        active_mask(
            grid,
            m=m,
            xi_outer_dirichlet=xi_outer_dirichlet,
            drop_zero_overlap=drop_zero_overlap,
        ).ravel()
    )


def restrict_matrix(
    matrix: sparse.spmatrix,
    active_indices: np.ndarray,
) -> sparse.csr_matrix:
    active_indices = np.asarray(active_indices)
    return matrix.tocsr()[active_indices, :][:, active_indices].tocsr()


def restrict_diagonal(
    diagonal: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(diagonal)[np.asarray(active_indices)]
