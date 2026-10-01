from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.special import gammaln, lpmv, lqmn

from .grid import ProlateSpheroidalGrid
from .operators import (
    overlap_diagonal,
    validate_axis_free_grid,
    weighted_stiffness_matrix,
    xi_derivative_matrix,
)


@dataclass(frozen=True)
class NeumannCoulombTerm:
    coefficient: float
    eta_factor: np.ndarray
    xi_kernel: np.ndarray


@dataclass(frozen=True)
class NeumannCoulombSolver:
    """Cached Neumann-expansion Coulomb solver for one angular transfer."""

    grid: ProlateSpheroidalGrid
    l_max: int
    angular_m: int
    terms: tuple[NeumannCoulombTerm, ...]

    @classmethod
    def from_grid(
        cls,
        grid: ProlateSpheroidalGrid,
        l_max: int,
        angular_m: int = 0,
    ) -> "NeumannCoulombSolver":
        validate_axis_free_grid(grid)
        order = abs(int(angular_m))
        if l_max < order:
            raise ValueError("l_max must be at least abs(angular_m).")

        xi, eta = grid.mesh()
        xi_flat = xi.ravel()
        eta_flat = eta.ravel()
        xi_min = np.minimum.outer(xi_flat, xi_flat)
        xi_max = np.maximum.outer(xi_flat, xi_flat)

        terms = []
        for degree in range(order, l_max + 1):
            p_xi_min = associated_legendre_p(degree, order, xi_min)
            q_xi_max = associated_legendre_q(degree, order, xi_max)
            terms.append(
                NeumannCoulombTerm(
                    coefficient=neumann_coefficient(degree, order) / grid.a,
                    eta_factor=lpmv(order, degree, eta_flat),
                    xi_kernel=p_xi_min * q_xi_max,
                )
            )
        return cls(
            grid=grid,
            l_max=l_max,
            angular_m=angular_m,
            terms=tuple(terms),
        )

    def potential(self, density: np.ndarray) -> np.ndarray:
        density_2d = np.asarray(density, dtype=float).reshape(self.grid.shape)
        source_weights = density_2d.ravel() * overlap_diagonal(self.grid)
        potential = np.zeros(self.grid.size)

        for term in self.terms:
            source = source_weights * term.eta_factor
            potential += (
                term.coefficient
                * term.eta_factor
                * (term.xi_kernel @ source)
            )

        return potential.reshape(self.grid.shape)


@dataclass(frozen=True)
class HaxtonPoissonCoulombSolver:
    """Appendix-B Poisson/Green's-function Coulomb solver.

    This implements the finite-box radial Green's-function construction used
    by Haxton, Lawler, and McCurdy.  The raw Neumann solver evaluates
    P_l^m(xi_<) Q_l^m(xi_>) directly; this solver represents that radial
    Green's function through the inverse of the finite-DVR T_lm matrix plus
    the boundary correction at xi_max.
    """

    grid: ProlateSpheroidalGrid
    l_max: int
    angular_m: int
    radial_kernels: tuple[np.ndarray, ...]
    eta_factors: tuple[np.ndarray, ...]
    active_xi: np.ndarray

    @classmethod
    def from_grid(
        cls,
        grid: ProlateSpheroidalGrid,
        l_max: int,
        angular_m: int = 0,
    ) -> "HaxtonPoissonCoulombSolver":
        validate_axis_free_grid(grid)
        order = abs(int(angular_m))
        if l_max < order:
            raise ValueError("l_max must be at least abs(angular_m).")

        active_xi = np.arange(grid.xi.size - 1)
        xi_active = grid.xi[active_xi]
        xi_boundary = grid.xi[-1]
        weights_active = grid.xi_weights[active_xi]

        stiffness = weighted_stiffness_matrix(
            grid.xi,
            grid.xi_weights,
            xi_derivative_matrix(grid, order),
            grid.xi**2 - 1.0,
        ).toarray()
        inv_sqrt_weights = 1.0 / np.sqrt(grid.xi_weights)
        normalized_stiffness = (
            inv_sqrt_weights[:, np.newaxis]
            * stiffness
            * inv_sqrt_weights[np.newaxis, :]
        )
        normalized_stiffness = normalized_stiffness[np.ix_(active_xi, active_xi)]

        radial_kernels = []
        eta_factors = []
        for degree in range(order, l_max + 1):
            p_xi = normalized_associated_legendre_p(degree, order, xi_active)
            p_boundary = float(
                normalized_associated_legendre_p(
                    degree,
                    order,
                    np.array([xi_boundary]),
                )[0]
            )
            q_boundary = float(
                normalized_associated_legendre_q(
                    degree,
                    order,
                    np.array([xi_boundary]),
                )[0]
            )
            diagonal = (
                degree * (degree + 1.0)
                + order**2 / (xi_active**2 - 1.0)
            )
            t_matrix = -normalized_stiffness - np.diag(diagonal)
            t_inverse = np.linalg.inv(t_matrix)
            boundary = (
                2.0
                / (2 * degree + 1.0)
                * np.outer(p_xi, p_xi)
                * q_boundary
                / p_boundary
            )
            poisson = haxton_t_inverse_coefficient(degree, order) * t_inverse
            poisson /= np.sqrt(
                weights_active[:, np.newaxis] * weights_active[np.newaxis, :]
            )
            radial_kernels.append(boundary + poisson)
            eta_factors.append(
                normalized_associated_legendre_p(degree, order, grid.eta)
            )

        return cls(
            grid=grid,
            l_max=l_max,
            angular_m=order,
            radial_kernels=tuple(radial_kernels),
            eta_factors=tuple(eta_factors),
            active_xi=active_xi,
        )

    def potential(self, density: np.ndarray) -> np.ndarray:
        density_2d = np.asarray(density, dtype=float).reshape(self.grid.shape)
        source_weights = (
            density_2d * overlap_diagonal(self.grid).reshape(self.grid.shape)
        )
        potential = np.zeros(self.grid.shape)
        prefactor = (
            (-1.0) ** self.angular_m
            * 4.0
            / self.grid.internuclear_distance
        )
        eta_indices = np.arange(self.grid.eta.size)
        source = source_weights[np.ix_(self.active_xi, eta_indices)]

        for radial_kernel, eta_factor in zip(self.radial_kernels, self.eta_factors):
            contracted_eta = (source * eta_factor[np.newaxis, :]).sum(axis=1)
            potential[self.active_xi, :] += (
                prefactor
                * (radial_kernel @ contracted_eta)[:, np.newaxis]
                * eta_factor[np.newaxis, :]
            )

        return potential


def associated_legendre_p(
    degree: int,
    order: int,
    x: np.ndarray,
) -> np.ndarray:
    """Return real associated Legendre P for eta in [-1, 1] or xi > 1."""

    if order < 0:
        raise ValueError("order must be non-negative.")
    if degree < order:
        raise ValueError("degree must be at least order.")

    points = np.asarray(x)
    if np.all(points <= 1.0):
        return np.asarray(lpmv(order, degree, points))
    if not np.all(points > 1.0):
        raise ValueError("associated_legendre_p expects either x <= 1 or x > 1.")

    if order == 0:
        return np.asarray(lpmv(0, degree, points))

    p_mm = odd_double_factorial(2 * order - 1) * (
        points**2 - 1.0
    ) ** (0.5 * order)
    if degree == order:
        return p_mm

    p_prev = p_mm
    p_curr = (2 * order + 1.0) * points * p_prev
    if degree == order + 1:
        return p_curr

    for current_degree in range(order + 2, degree + 1):
        p_next = (
            (2 * current_degree - 1.0) * points * p_curr
            - (current_degree + order - 1.0) * p_prev
        ) / (current_degree - order)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def associated_legendre_q(
    degree: int,
    order: int,
    x: np.ndarray,
) -> np.ndarray:
    """Return associated Legendre Q_degree^order(x) for x > 1."""

    if order < 0:
        raise ValueError("order must be non-negative.")
    if degree < order:
        raise ValueError("degree must be at least order.")
    values, _ = lqmn(order, degree, np.asarray(x))
    return np.asarray(values[order, degree])


def normalized_associated_legendre_p(
    degree: int,
    order: int,
    x: np.ndarray,
) -> np.ndarray:
    """Return Haxton's normalized associated Legendre P."""

    log_norm = 0.5 * (
        np.log(2 * degree + 1.0)
        + gammaln(degree - order + 1)
        - np.log(2.0)
        - gammaln(degree + order + 1)
    )
    return math.exp(log_norm) * associated_legendre_p(degree, order, x)


def normalized_associated_legendre_q(
    degree: int,
    order: int,
    x: np.ndarray,
) -> np.ndarray:
    """Return Haxton's normalized associated Legendre Q."""

    log_norm = 0.5 * (
        np.log(2 * degree + 1.0)
        + gammaln(degree - order + 1)
        - np.log(2.0)
        - gammaln(degree + order + 1)
    )
    return math.exp(log_norm) * associated_legendre_q(degree, order, x)


def neumann_coefficient(degree: int, order: int) -> float:
    """Return the coefficient in the prolate Neumann expansion of 1/r12."""

    if order < 0:
        raise ValueError("order must be non-negative.")
    if degree < order:
        raise ValueError("degree must be at least order.")

    log_ratio = gammaln(degree - order + 1) - gammaln(degree + order + 1)
    return (
        (-1.0) ** order
        * (2 * degree + 1)
        * math.exp(2.0 * log_ratio)
    )


def haxton_t_inverse_coefficient(degree: int, order: int) -> float:
    """Return the coefficient multiplying T_lm^-1.

    With the normalized P_l^m and SciPy Q_l^m conventions used here, the
    Wronskian gives a coefficient (-1)^(m + 1).  This is equivalent to
    Haxton Eq. (B5) after accounting for the associated-Q normalization.
    """

    if order < 0:
        raise ValueError("order must be non-negative.")
    if degree < order:
        raise ValueError("degree must be at least order.")

    return (-1.0) ** (order + 1)


def odd_double_factorial(n: int) -> float:
    if n < 1:
        return 1.0

    value = 1.0
    for factor in range(n, 0, -2):
        value *= factor
    return value


def electron_electron_potential(
    grid: ProlateSpheroidalGrid,
    density: np.ndarray,
    l_max: int,
    angular_m: int = 0,
) -> np.ndarray:
    """Return the Coulomb potential generated by a nodal density.

    The implementation follows the Neumann expansion used in Zhang et al.,
    Sec. 2.5,

        1/r12 = 1/a sum_lm C_lm P_l^m(xi_<) Q_l^m(xi_>)
                P_l^m(eta_1) P_l^m(eta_2) exp(i m(phi_1 - phi_2)).

    ``density`` is interpreted as a real-space density value on the full
    tensor-product grid, flattened in C order or shaped as ``grid.shape``.  For
    a normalized orbital, ``sum(density * overlap_diagonal(grid))`` should be
    one.  The returned potential has the same shape as ``density``.
    """

    validate_axis_free_grid(grid)
    order = abs(int(angular_m))
    if l_max < order:
        raise ValueError("l_max must be at least abs(angular_m).")

    solver = NeumannCoulombSolver.from_grid(
        grid,
        l_max=l_max,
        angular_m=order,
    )
    return solver.potential(density)


def electron_electron_potential_diagonal(
    grid: ProlateSpheroidalGrid,
    density: np.ndarray,
    l_max: int,
    angular_m: int = 0,
) -> np.ndarray:
    """Return the weak-form diagonal for multiplication by the Hartree potential."""

    potential = electron_electron_potential(
        grid,
        density,
        l_max=l_max,
        angular_m=angular_m,
    )
    return overlap_diagonal(grid) * potential.ravel()


def electron_electron_potential_matrix(
    grid: ProlateSpheroidalGrid,
    density: np.ndarray,
    l_max: int,
    angular_m: int = 0,
) -> sparse.csr_matrix:
    """Return the weak-form diagonal matrix for the Hartree potential."""

    return sparse.diags(
        electron_electron_potential_diagonal(
            grid,
            density,
            l_max=l_max,
            angular_m=angular_m,
        ),
        format="csr",
    )


def coulomb_self_energy(
    grid: ProlateSpheroidalGrid,
    density: np.ndarray,
    potential: np.ndarray | None = None,
    l_max: int | None = None,
) -> float:
    """Return int density(r) potential(r) dV for a Coulomb potential.

    This is the Coulomb integral J for a one-electron orbital density when
    ``potential`` is generated by that same density.  For a classical total
    charge density self-energy, multiply the returned value by one half.
    """

    density_flat = np.asarray(density, dtype=float).reshape(grid.shape).ravel()
    if potential is None:
        if l_max is None:
            raise ValueError("l_max is required when potential is not supplied.")
        potential = electron_electron_potential(grid, density, l_max=l_max)
    potential_flat = np.asarray(potential, dtype=float).reshape(grid.shape).ravel()
    return float(np.dot(density_flat * overlap_diagonal(grid), potential_flat))
