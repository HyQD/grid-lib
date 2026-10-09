import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from grid_lib.prolate_spheroidal_coordinates import (
    NodalDVRGrid,
    lagrange_derivative_matrix,
    setup_gauss_legendre_interval,
    setup_prolate_spheroidal_grid,
    validate_axis_free_grid,
)
from grid_lib.pseudospectral_grids import (
    GaussLegendre,
    Linear_map,
    PseudospectralGrid,
    Rational_map,
)


def test_default_reference_grid():
    grid = GaussLegendre(3)

    assert isinstance(grid, PseudospectralGrid)
    assert grid.n_points == 3
    np.testing.assert_allclose(grid.x, [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    np.testing.assert_allclose(grid.r, grid.x)
    np.testing.assert_allclose(grid.r_dot, 1.0)
    np.testing.assert_allclose(grid.weights, [5 / 9, 8 / 9, 5 / 9])
    assert grid.D1.shape == (3, 3)
    assert grid.D2.shape == (3, 3)


@pytest.mark.parametrize("n_points", [1, 2, 6, 12])
@pytest.mark.parametrize("interval", [(-1.0, 1.0), (0.0, 2.0), (-3.0, -0.5)])
def test_quadrature_polynomial_exactness(n_points, interval):
    left, right = interval
    grid = GaussLegendre(n_points, Linear_map(left, right))

    assert np.all(np.diff(grid.r) > 0.0)
    assert np.all((left < grid.r) & (grid.r < right))
    assert np.all(grid.weights > 0.0)
    for degree in range(2 * n_points):
        polynomial = Legendre.basis(degree, domain=interval)
        integral = polynomial.integ()
        np.testing.assert_allclose(
            grid.weights @ polynomial(grid.r),
            integral(right) - integral(left),
            rtol=0.0,
            atol=2e-14,
        )


@pytest.mark.parametrize("n_points", [1, 2, 8, 12])
@pytest.mark.parametrize("interval", [(-1.0, 1.0), (0.0, 2.0), (2.0, 5.0)])
def test_derivative_polynomial_exactness(n_points, interval):
    grid = GaussLegendre(n_points, Linear_map(*interval))

    for degree in range(n_points):
        polynomial = Legendre.basis(degree, domain=interval)
        values = polynomial(grid.r)
        np.testing.assert_allclose(
            grid.D1 @ values,
            polynomial.deriv(1)(grid.r),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            grid.D2 @ values,
            polynomial.deriv(2)(grid.r),
            rtol=2e-12,
            atol=2e-10,
        )


def test_one_point_mapped_grid():
    grid = GaussLegendre(np.int64(1), Linear_map(2.0, 6.0))

    np.testing.assert_array_equal(grid.x, [0.0])
    np.testing.assert_array_equal(grid.r, [4.0])
    np.testing.assert_array_equal(grid.weights, [4.0])
    np.testing.assert_array_equal(grid.D1, [[0.0]])
    np.testing.assert_array_equal(grid.D2, [[0.0]])


@pytest.mark.parametrize("n_points", [0, -1])
def test_reject_nonpositive_point_count(n_points):
    with pytest.raises(ValueError, match="n_points must be at least 1"):
        GaussLegendre(n_points)


@pytest.mark.parametrize("n_points", [1.5, 3.0, True, np.bool_(False)])
def test_reject_noninteger_point_count(n_points):
    with pytest.raises(TypeError, match="n_points must be an integer"):
        GaussLegendre(n_points)


@pytest.mark.parametrize("interval", [(1.0, 1.0), (2.0, -1.0)])
def test_reject_nonincreasing_interval(interval):
    with pytest.raises(ValueError, match="strictly increasing interval"):
        GaussLegendre(4, Linear_map(*interval))


@pytest.mark.parametrize("interval", [(0.0, np.inf), (-np.inf, 0.0), (np.nan, 1.0)])
def test_reject_nonfinite_interval(interval):
    with pytest.raises(ValueError, match="endpoints must be finite"):
        GaussLegendre(4, Linear_map(*interval))


def test_reject_nonlinear_mapping():
    with pytest.raises(TypeError, match="Mapping must be a Linear_map"):
        GaussLegendre(4, Rational_map())


@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("interval", [(-1.0, 1.0), (2.0, 5.0)])
def test_prolate_interval_helper_preserves_interface_and_values(n_points, interval):
    left, right = interval
    grid = setup_gauss_legendre_interval(left, right, n_points)
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    scale = 0.5 * (right - left)
    points = scale * nodes + 0.5 * (right + left)

    assert isinstance(grid, NodalDVRGrid)
    np.testing.assert_allclose(grid.r, points, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(grid.weights, scale * weights)
    np.testing.assert_allclose(
        grid.D1, lagrange_derivative_matrix(points), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_array_equal(grid.nodes, interval)
    assert grid.edge_indices == []


def test_prolate_tensor_grid_uses_gauss_legendre_eta_grid():
    grid = setup_prolate_spheroidal_grid(
        internuclear_distance=2.0,
        xi_max=5.0,
        xi_elements=2,
        points_per_element=4,
        eta_degree=6,
    )

    validate_axis_free_grid(grid)
    assert grid.shape == (7, 6)
    np.testing.assert_allclose(grid.eta_weights.sum(), 2.0)
    np.testing.assert_allclose(grid.eta_D1 @ grid.eta**3, 3 * grid.eta**2, atol=2e-14)
