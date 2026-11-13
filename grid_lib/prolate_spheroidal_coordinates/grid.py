from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.special import roots_jacobi

from grid_lib.pseudospectral_grids.gauss_legendre import GaussLegendre
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


@dataclass(frozen=True)
class ProlateSpheroidalGrid:
    """Tensor-product prolate spheroidal grid.

    The nuclei are placed at z = +/- a with internuclear distance R = 2a.
    The coordinates are xi in [1, infinity), eta in [-1, 1].
    """

    internuclear_distance: float
    xi: np.ndarray
    eta: np.ndarray
    xi_weights: np.ndarray
    eta_weights: np.ndarray
    xi_D1: np.ndarray
    eta_D1: np.ndarray
    xi_grid: object
    eta_grid: object

    @property
    def a(self) -> float:
        return 0.5 * self.internuclear_distance

    @property
    def shape(self) -> tuple[int, int]:
        return self.xi.size, self.eta.size

    @property
    def size(self) -> int:
        return self.xi.size * self.eta.size

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.xi, self.eta, indexing="ij")

    def metric_factor(self) -> np.ndarray:
        xi, eta = self.mesh()
        return xi**2 - eta**2

    def quadrature_weights_2d(self) -> np.ndarray:
        return self.xi_weights[:, np.newaxis] * self.eta_weights[np.newaxis, :]

    def axial_coordinate(self) -> np.ndarray:
        """Return the Cartesian ``z = a * xi * eta`` coordinate."""

        xi, eta = self.mesh()
        return self.a * xi * eta

    def cylindrical_radius(self) -> np.ndarray:
        """Return the cylindrical radius from the molecular axis."""

        xi, eta = self.mesh()
        return self.a * np.sqrt((xi**2 - 1.0) * (1.0 - eta**2))

    def cartesian_radius(self) -> np.ndarray:
        """Return the distance from the bond midpoint."""

        xi, eta = self.mesh()
        return self.a * np.sqrt(xi**2 + eta**2 - 1.0)

    def focus_distances(self) -> tuple[np.ndarray, np.ndarray]:
        """Return distances to the foci at ``z = -a`` and ``z = +a``."""

        xi, eta = self.mesh()
        return self.a * (xi + eta), self.a * (xi - eta)

    def cartesian_coordinates(
        self,
        phi: Real = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Cartesian coordinates on one fixed azimuthal half-plane."""

        phi = _finite_real(phi, "phi")
        rho = self.cylindrical_radius()
        return rho * np.cos(phi), rho * np.sin(phi), self.axial_coordinate()


@dataclass(frozen=True)
class NodalDVRGrid:
    """Minimal one-dimensional nodal DVR container."""

    r: np.ndarray
    weights: np.ndarray
    D1: np.ndarray
    nodes: np.ndarray | None = None
    edge_indices: list[int] | None = None


def setup_prolate_spheroidal_grid(
    internuclear_distance: float,
    xi_max: float,
    xi_elements: int = 8,
    eta_elements: int = 1,
    points_per_element: int = 8,
    xi_points_per_element: int | None = None,
    eta_points_per_element: int | None = None,
    eta_degree: int | None = None,
) -> ProlateSpheroidalGrid:
    """Create a tensor-product grid in prolate spheroidal coordinates.

    The xi grid uses a right-Radau first element and Lobatto nodes on later
    elements.  The eta grid uses a single Gauss-Legendre quadrature over
    [-1, 1].  This axis-free grid is the one expected by the prolate
    one-electron and Coulomb operators.
    """

    internuclear_distance = _positive_real(
        internuclear_distance,
        "internuclear_distance",
    )
    if xi_max <= 1.0:
        raise ValueError("xi_max must be greater than 1.")
    xi_elements = _positive_integer(xi_elements, "xi_elements")
    xi_points = (
        points_per_element
        if xi_points_per_element is None
        else xi_points_per_element
    )
    eta_points = (
        eta_elements
        * (
            points_per_element
            if eta_points_per_element is None
            else eta_points_per_element
        )
        if eta_degree is None
        else eta_degree
    )
    return setup_prolate_spheroidal_grid_from_boundaries(
        internuclear_distance=internuclear_distance,
        xi_boundaries=np.linspace(1.0, xi_max, xi_elements + 1),
        xi_points_per_element=xi_points,
        eta_degree=eta_points,
    )


def setup_prolate_spheroidal_grid_from_boundaries(
    internuclear_distance: float,
    xi_boundaries: np.ndarray,
    xi_points_per_element: int | np.ndarray = 8,
    eta_degree: int = 8,
) -> ProlateSpheroidalGrid:
    """Create an axis-free prolate grid from explicit ``xi`` boundaries.

    ``xi_points_per_element`` may be one integer shared by every element or
    one integer per interval in ``xi_boundaries``. The first boundary must be
    exactly one, as required by the prolate coordinate domain.
    """

    internuclear_distance = _positive_real(
        internuclear_distance,
        "internuclear_distance",
    )
    boundaries = _element_boundaries(xi_boundaries)
    if boundaries[0] != 1.0:
        raise ValueError("xi_boundaries must start at 1")
    eta_degree = _positive_integer(eta_degree, "eta_degree")
    xi_grid = setup_radau_lobatto_femdvr_interval_from_boundaries(
        boundaries,
        points_per_element=xi_points_per_element,
    )
    eta_grid = setup_gauss_legendre_interval(
        -1.0,
        1.0,
        n_points=eta_degree,
    )
    return ProlateSpheroidalGrid(
        internuclear_distance=internuclear_distance,
        xi=np.asarray(xi_grid.r),
        eta=np.asarray(eta_grid.r),
        xi_weights=np.asarray(xi_grid.weights),
        eta_weights=np.asarray(eta_grid.weights),
        xi_D1=np.asarray(xi_grid.D1),
        eta_D1=np.asarray(eta_grid.D1),
        xi_grid=xi_grid,
        eta_grid=eta_grid,
    )


def setup_gauss_legendre_interval(
    x_min: float,
    x_max: float,
    n_points: int,
) -> NodalDVRGrid:
    """Create a Gauss-Legendre grid, retaining the nodal DVR interface."""

    grid = GaussLegendre(n_points, Linear_map(x_min, x_max))
    return NodalDVRGrid(
        r=grid.r,
        weights=grid.weights,
        D1=grid.D1,
        nodes=np.array([x_min, x_max]),
        edge_indices=[],
    )


def setup_radau_lobatto_femdvr_interval(
    x_min: float,
    x_max: float,
    n_elements: int,
    points_per_element: int,
) -> NodalDVRGrid:
    """Use right-Radau on the first element and GLL on later elements."""

    if n_elements < 1:
        raise ValueError("n_elements must be at least 1.")
    nodes = np.linspace(x_min, x_max, n_elements + 1)
    return setup_radau_lobatto_femdvr_interval_from_boundaries(
        nodes,
        points_per_element,
    )


def setup_radau_lobatto_femdvr_interval_from_boundaries(
    element_boundaries: np.ndarray,
    points_per_element: int | np.ndarray,
) -> NodalDVRGrid:
    """Use right-Radau on the first element and GLL on later elements."""

    element_boundaries = _element_boundaries(element_boundaries)
    point_counts = _element_point_counts(
        points_per_element,
        n_elements=element_boundaries.size - 1,
    )

    local_points = []
    local_weights = []
    local_derivatives = []
    local_to_global = []
    global_points: list[float] = []

    for element, n_points in enumerate(point_counts):
        if element == 0:
            standard_points, standard_weights = right_radau_nodes_weights(
                n_points
            )
        else:
            gll = GaussLegendreLobatto(
                n_points - 1,
                Linear_map(-1.0, 1.0),
                symmetrize=False,
            )
            standard_points = gll.r
            standard_weights = gll.weights

        points, weights = map_standard_interval(
            standard_points,
            standard_weights,
            element_boundaries[element],
            element_boundaries[element + 1],
        )
        local_points.append(points)
        local_weights.append(weights)
        local_derivatives.append(lagrange_derivative_matrix(points))

        element_indices = []
        for point in points:
            match = next(
                (
                    index
                    for index, existing in enumerate(global_points)
                    if np.isclose(point, existing, rtol=0.0, atol=1e-13)
                ),
                None,
            )
            if match is None:
                match = len(global_points)
                global_points.append(float(point))
            element_indices.append(match)
        local_to_global.append(element_indices)

    dim0 = sum(points.size for points in local_points)
    n_global = len(global_points)
    local_derivative = np.zeros((dim0, dim0))
    local_weight = np.zeros(dim0)
    restriction = np.zeros((dim0, n_global))

    offset = 0
    for points, weights, derivative, indices in zip(
        local_points,
        local_weights,
        local_derivatives,
        local_to_global,
    ):
        n_local = points.size
        local_slice = slice(offset, offset + n_local)
        local_derivative[local_slice, local_slice] = derivative
        local_weight[local_slice] = weights
        for row, global_index in enumerate(indices, start=offset):
            restriction[row, global_index] = 1.0
        offset += n_local

    derivative_intermediate = (
        restriction.T @ np.diag(local_weight) @ local_derivative @ restriction
    )
    mass = restriction.T @ np.diag(local_weight) @ restriction
    weights = local_weight @ restriction
    derivative = np.linalg.solve(mass, derivative_intermediate)

    edge_indices = [
        int(np.argmin(np.abs(np.asarray(global_points) - node)))
        for node in element_boundaries
    ]
    return NodalDVRGrid(
        r=np.asarray(global_points),
        weights=weights,
        D1=derivative,
        nodes=element_boundaries,
        edge_indices=edge_indices,
    )


def right_radau_nodes_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Radau nodes and weights on [-1, 1] including x=1."""

    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    interior, _ = roots_jacobi(n_points - 1, 1.0, 0.0)
    nodes = np.concatenate((interior, np.array([1.0])))
    nodes.sort()
    weights = quadrature_weights_from_moments(nodes)
    return nodes, weights


def quadrature_weights_from_moments(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes)
    powers = np.arange(nodes.size)
    vandermonde = nodes[np.newaxis, :] ** powers[:, np.newaxis]
    moments = np.array(
        [2.0 / (power + 1) if power % 2 == 0 else 0.0 for power in powers]
    )
    return np.linalg.solve(vandermonde, moments)


def map_standard_interval(
    nodes: np.ndarray,
    weights: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 0.5 * (x_max - x_min)
    shift = 0.5 * (x_max + x_min)
    return scale * np.asarray(nodes) + shift, scale * np.asarray(weights)


def lagrange_derivative_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    n_points = points.size
    barycentric_weights = np.ones(n_points)
    for j in range(n_points):
        barycentric_weights[j] = 1.0 / np.prod(
            points[j] - np.delete(points, j)
        )

    derivative = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                derivative[i, j] = (
                    barycentric_weights[j]
                    / barycentric_weights[i]
                    / (points[i] - points[j])
                )
        derivative[i, i] = -np.sum(derivative[i])
    return derivative


def _element_boundaries(values: np.ndarray) -> np.ndarray:
    boundaries = np.asarray(values, dtype=float)
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("element boundaries must be a one-dimensional array")
    if not np.all(np.isfinite(boundaries)):
        raise ValueError("element boundaries must be finite")
    if np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("element boundaries must be strictly increasing")
    return boundaries


def _element_point_counts(values, *, n_elements: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 0:
        if isinstance(values, (bool, np.bool_)) or not isinstance(
            values,
            Integral,
        ):
            raise TypeError("points_per_element must contain integers")
        counts = np.full(n_elements, int(values), dtype=int)
    else:
        if raw.shape != (n_elements,):
            raise ValueError("points_per_element must contain one value per element")
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("points_per_element must contain integers")
        counts = raw.astype(int, copy=False)
    if np.any(counts < 2):
        raise ValueError("points_per_element values must be at least 2")
    return counts


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_real(value: Real, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result
