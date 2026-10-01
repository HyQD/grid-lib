from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import legendre
from scipy.special import roots_jacobi

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

GridKind = Literal["femdvr", "gll", "radau"]


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
    grid_kind: GridKind = "radau",
    xi_degree: int | None = None,
    eta_degree: int | None = None,
) -> ProlateSpheroidalGrid:
    """Create a tensor-product grid in prolate spheroidal coordinates.

    The default ``grid_kind="radau"`` uses a right-Radau first xi element,
    Lobatto later xi elements, and Gauss-Legendre eta nodes.  This is the grid
    expected by the prolate one-electron operators.  The ``"femdvr"`` and
    ``"gll"`` grid kinds remain available as lower-level grid construction
    utilities.
    """

    if internuclear_distance <= 0.0:
        raise ValueError("internuclear_distance must be positive.")
    if xi_max <= 1.0:
        raise ValueError("xi_max must be greater than 1.")

    if grid_kind == "femdvr":
        xi_grid = setup_femdvr_interval(
            1.0,
            xi_max,
            n_elements=xi_elements,
            points_per_element=xi_points_per_element or points_per_element,
        )
        eta_grid = setup_femdvr_interval(
            -1.0,
            1.0,
            n_elements=eta_elements,
            points_per_element=eta_points_per_element or points_per_element,
        )
    elif grid_kind == "gll":
        xi_grid = GaussLegendreLobatto(
            xi_degree if xi_degree is not None else points_per_element - 1,
            Linear_map(1.0, xi_max),
            symmetrize=False,
        )
        eta_grid = GaussLegendreLobatto(
            eta_degree if eta_degree is not None else points_per_element - 1,
            Linear_map(-1.0, 1.0),
            symmetrize=False,
        )
    elif grid_kind == "radau":
        xi_grid = setup_radau_lobatto_femdvr_interval(
            1.0,
            xi_max,
            n_elements=xi_elements,
            points_per_element=xi_points_per_element or points_per_element,
        )
        eta_grid = setup_gauss_legendre_interval(
            -1.0,
            1.0,
            n_points=eta_degree
            if eta_degree is not None
            else eta_elements * (eta_points_per_element or points_per_element),
        )
    else:
        raise ValueError("grid_kind must be 'femdvr', 'gll', or 'radau'.")

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


def setup_femdvr_interval(
    x_min: float,
    x_max: float,
    n_elements: int,
    points_per_element: int,
) -> FEMDVR:
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1.")
    if points_per_element < 2:
        raise ValueError("points_per_element must be at least 2.")

    nodes = np.linspace(x_min, x_max, n_elements + 1)
    n_points = np.full(n_elements, points_per_element, dtype=int)
    return FEMDVR(
        nodes,
        n_points,
        Linear_map,
        GaussLegendreLobatto,
        symmetrize=False,
    )


def setup_gauss_legendre_interval(
    x_min: float,
    x_max: float,
    n_points: int,
) -> NodalDVRGrid:
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")

    nodes, weights = legendre.leggauss(n_points)
    points, physical_weights = map_standard_interval(
        nodes,
        weights,
        x_min,
        x_max,
    )
    return NodalDVRGrid(
        r=points,
        weights=physical_weights,
        D1=lagrange_derivative_matrix(points),
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
    points_per_element: int,
) -> NodalDVRGrid:
    """Use right-Radau on the first element and GLL on later elements."""

    element_boundaries = np.asarray(element_boundaries, dtype=float)
    if element_boundaries.ndim != 1 or element_boundaries.size < 2:
        raise ValueError("element_boundaries must be a one-dimensional array.")
    if np.any(np.diff(element_boundaries) <= 0.0):
        raise ValueError("element_boundaries must be strictly increasing.")
    if points_per_element < 2:
        raise ValueError("points_per_element must be at least 2.")

    local_points = []
    local_weights = []
    local_derivatives = []
    local_to_global = []
    global_points: list[float] = []

    for element in range(element_boundaries.size - 1):
        if element == 0:
            standard_points, standard_weights = right_radau_nodes_weights(
                points_per_element
            )
        else:
            gll = GaussLegendreLobatto(
                points_per_element - 1,
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
