import numpy as np
import pytest

from grid_lib.prolate_spheroidal_coordinates import (
    active_indices,
    from_euclidean_coefficients,
    local_potential_diagonal,
    local_potential_matrix,
    overlap_diagonal,
    setup_prolate_spheroidal_grid,
    setup_prolate_spheroidal_grid_from_boundaries,
    to_euclidean_coefficients,
)


@pytest.fixture
def haxton_comparable_grid():
    return setup_prolate_spheroidal_grid_from_boundaries(
        internuclear_distance=2.0,
        xi_boundaries=[1.0, 4.0, 14.0, 24.0],
        xi_points_per_element=14,
        eta_degree=9,
    )


def test_explicit_boundaries_reproduce_haxton_comparable_shape(
    haxton_comparable_grid,
):
    grid = haxton_comparable_grid

    assert grid.shape == (40, 9)
    assert grid.size == 360
    assert active_indices(grid).size == 351
    np.testing.assert_array_equal(grid.xi_grid.nodes, [1.0, 4.0, 14.0, 24.0])
    assert grid.xi[0] > 1.0
    assert grid.xi[-1] == 24.0
    assert np.all(np.abs(grid.eta) < 1.0)


def test_explicit_boundaries_accept_per_element_point_counts():
    grid = setup_prolate_spheroidal_grid_from_boundaries(
        internuclear_distance=2.0,
        xi_boundaries=[1.0, 2.0, 5.0],
        xi_points_per_element=[3, 5],
        eta_degree=4,
    )

    assert grid.shape == (7, 4)
    np.testing.assert_array_equal(grid.xi_grid.nodes, [1.0, 2.0, 5.0])


def test_uniform_constructor_matches_explicit_boundaries():
    uniform = setup_prolate_spheroidal_grid(
        internuclear_distance=2.0,
        xi_max=7.0,
        xi_elements=3,
        xi_points_per_element=5,
        eta_degree=6,
    )
    explicit = setup_prolate_spheroidal_grid_from_boundaries(
        internuclear_distance=2.0,
        xi_boundaries=np.linspace(1.0, 7.0, 4),
        xi_points_per_element=5,
        eta_degree=6,
    )

    for attribute in (
        "xi",
        "eta",
        "xi_weights",
        "eta_weights",
        "xi_D1",
        "eta_D1",
    ):
        np.testing.assert_allclose(
            getattr(uniform, attribute),
            getattr(explicit, attribute),
        )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"internuclear_distance": 0.0}, ValueError, "must be positive"),
        ({"xi_boundaries": [1.1, 2.0]}, ValueError, "must start at 1"),
        ({"xi_boundaries": [1.0, np.inf]}, ValueError, "must be finite"),
        (
            {"xi_points_per_element": [3, 4, 5]},
            ValueError,
            "one value per element",
        ),
        ({"xi_points_per_element": 2.5}, TypeError, "contain integers"),
        ({"eta_degree": True}, TypeError, "must be an integer"),
    ],
)
def test_explicit_boundary_grid_validates_inputs(arguments, error, message):
    options = {
        "internuclear_distance": 2.0,
        "xi_boundaries": [1.0, 2.0, 5.0],
        "xi_points_per_element": 4,
        "eta_degree": 5,
    }
    options.update(arguments)

    with pytest.raises(error, match=message):
        setup_prolate_spheroidal_grid_from_boundaries(**options)


def test_coordinate_helpers_satisfy_prolate_geometry(haxton_comparable_grid):
    grid = haxton_comparable_grid
    xi, eta = grid.mesh()
    x, y, z = grid.cartesian_coordinates(phi=0.37)
    rho = grid.cylindrical_radius()
    radius = grid.cartesian_radius()
    distance1, distance2 = grid.focus_distances()

    np.testing.assert_allclose(z, grid.a * xi * eta)
    np.testing.assert_allclose(x**2 + y**2, rho**2)
    np.testing.assert_allclose(x**2 + y**2 + z**2, radius**2)
    np.testing.assert_allclose(distance1, grid.a * (xi + eta))
    np.testing.assert_allclose(distance2, grid.a * (xi - eta))
    np.testing.assert_allclose(distance1 + distance2, 2.0 * grid.a * xi)
    np.testing.assert_allclose(distance1 - distance2, 2.0 * grid.a * eta)


def test_cartesian_coordinates_reject_nonfinite_phi(haxton_comparable_grid):
    with pytest.raises(ValueError, match="phi must be a finite real scalar"):
        haxton_comparable_grid.cartesian_coordinates(np.inf)


def test_local_potential_weak_diagonal_and_matrix(haxton_comparable_grid):
    grid = haxton_comparable_grid
    potential = 0.25 * grid.axial_coordinate() - 0.1j * grid.cartesian_radius()
    expected = overlap_diagonal(grid) * potential.ravel()

    diagonal = local_potential_diagonal(grid, potential)
    matrix = local_potential_matrix(grid, potential)

    np.testing.assert_allclose(diagonal, expected)
    np.testing.assert_allclose(matrix.diagonal(), expected)
    assert matrix.format == "csr"

    active = active_indices(grid)
    np.testing.assert_allclose(
        local_potential_diagonal(grid, potential.ravel(), active),
        expected[active],
    )


@pytest.mark.parametrize(
    ("potential", "error", "message"),
    [
        (np.ones((2, 3)), ValueError, "potential must have shape"),
        (np.array(["invalid"]), TypeError, "potential must be numeric"),
        (None, TypeError, "potential must be numeric"),
    ],
)
def test_local_potential_validates_values(
    haxton_comparable_grid,
    potential,
    error,
    message,
):
    with pytest.raises(error, match=message):
        local_potential_diagonal(haxton_comparable_grid, potential)


def test_metric_transforms_round_trip_full_and_active_coefficients(
    haxton_comparable_grid,
):
    grid = haxton_comparable_grid
    rng = np.random.default_rng(4921)
    full = rng.normal(size=(2, *grid.shape)) + 1j * rng.normal(
        size=(2, *grid.shape)
    )

    euclidean = to_euclidean_coefficients(grid, full)
    restored = from_euclidean_coefficients(grid, euclidean)

    np.testing.assert_allclose(restored, full)
    expected_norms = np.sum(
        np.abs(full) ** 2 * overlap_diagonal(grid).reshape(grid.shape),
        axis=(-2, -1),
    )
    np.testing.assert_allclose(
        np.sum(np.abs(euclidean) ** 2, axis=(-2, -1)),
        expected_norms,
    )

    active = active_indices(grid)
    active_values = full.reshape(2, -1)[:, active]
    active_euclidean = to_euclidean_coefficients(
        grid,
        active_values,
        active_indices=active,
    )
    np.testing.assert_allclose(
        from_euclidean_coefficients(
            grid,
            active_euclidean,
            active_indices=active,
        ),
        active_values,
    )


def test_metric_transforms_validate_coefficient_shape(haxton_comparable_grid):
    with pytest.raises(ValueError, match="coefficients must end"):
        to_euclidean_coefficients(
            haxton_comparable_grid,
            np.ones((2, 3)),
        )
