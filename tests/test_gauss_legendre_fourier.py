import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from grid_lib.pseudospectral_grids import GaussLegendre
from grid_lib.spherical_coordinates import GaussLegendreFourierGrid
from grid_lib.spherical_coordinates.utils import Ylm


@pytest.mark.parametrize("n_theta,n_phi", [(1, 1), (3, 8), (8, 15)])
def test_nodes_weights_and_mesh(n_theta, n_phi):
    grid = GaussLegendreFourierGrid(n_theta, n_phi)
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)

    assert isinstance(grid.polar_grid, GaussLegendre)
    assert grid.shape == (n_theta, n_phi)
    assert grid.size == n_theta * n_phi
    np.testing.assert_allclose(grid.cos_theta, nodes)
    np.testing.assert_allclose(np.cos(grid.theta), nodes, atol=1e-15)
    assert np.all((0 < grid.theta) & (grid.theta < np.pi))
    assert np.all(np.diff(grid.theta) < 0)
    np.testing.assert_allclose(grid.phi, 2 * np.pi * np.arange(n_phi) / n_phi)
    assert np.all(grid.phi < 2 * np.pi)
    np.testing.assert_allclose(grid.theta_weights, weights)
    np.testing.assert_allclose(grid.theta_weights.sum(), 2)
    np.testing.assert_allclose(grid.phi_weights, 2 * np.pi / n_phi)
    np.testing.assert_allclose(grid.phi_weights.sum(), 2 * np.pi)
    np.testing.assert_allclose(
        grid.weights, grid.theta_weights[:, None] * grid.phi_weights[None, :]
    )
    np.testing.assert_allclose(grid.weights.sum(), 4 * np.pi)
    np.testing.assert_allclose(grid.integrate(np.ones(grid.shape)), 4 * np.pi)
    assert np.all(grid.weights > 0)

    theta, phi = grid.mesh()
    assert theta.shape == phi.shape == grid.shape
    np.testing.assert_array_equal(
        theta, np.broadcast_to(grid.theta[:, None], grid.shape)
    )
    np.testing.assert_array_equal(phi, np.broadcast_to(grid.phi[None, :], grid.shape))
    np.testing.assert_allclose(
        grid.weights.ravel() @ np.cos(theta.ravel()) ** 2,
        grid.integrate(np.cos(theta) ** 2),
    )
    assert repr(grid) == f"GaussLegendreFourierGrid(n_theta={n_theta}, n_phi={n_phi})"


@pytest.mark.parametrize("l_max", [0, 1, np.int64(4)])
def test_bandlimit_constructor(l_max):
    grid = GaussLegendreFourierGrid.from_bandlimit(l_max)
    assert grid.shape == (l_max + 1, 2 * l_max + 1)


def test_numpy_integer_counts():
    grid = GaussLegendreFourierGrid(np.int64(2), np.int32(3))
    assert grid.shape == (2, 3)


@pytest.mark.parametrize("name", ["n_theta", "n_phi", "l_max"])
@pytest.mark.parametrize("value", [1.5, 3.0, True, np.bool_(False), "4", None])
def test_reject_noninteger_counts(name, value):
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        if name == "l_max":
            GaussLegendreFourierGrid.from_bandlimit(value)
        else:
            parameters = {"n_theta": 3, "n_phi": 5, name: value}
            GaussLegendreFourierGrid(**parameters)


@pytest.mark.parametrize("name", ["n_theta", "n_phi"])
@pytest.mark.parametrize("value", [0, -1])
def test_reject_nonpositive_counts(name, value):
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        GaussLegendreFourierGrid(**{"n_theta": 3, "n_phi": 5, name: value})


def test_reject_negative_bandlimit():
    with pytest.raises(ValueError, match="l_max must be at least 0"):
        GaussLegendreFourierGrid.from_bandlimit(-1)


def test_integrate_scalars_and_broadcast_batches():
    grid = GaussLegendreFourierGrid(4, 9)
    theta, phi = grid.mesh()
    np.testing.assert_allclose(grid.integrate(1), 4 * np.pi)
    np.testing.assert_allclose(grid.integrate(2 + 3j), (2 + 3j) * 4 * np.pi)
    np.testing.assert_allclose(
        grid.integrate(grid.cos_theta[:, None] ** 2), 4 * np.pi / 3
    )
    np.testing.assert_allclose(grid.integrate(np.exp(1j * phi[0])), 0, atol=2e-14)

    factors = np.arange(6).reshape(2, 3) + 1j
    values = factors[..., None, None] * np.cos(theta) ** 2
    np.testing.assert_allclose(grid.integrate(values), factors * 4 * np.pi / 3)
    np.testing.assert_allclose(
        grid.integrate(factors[..., None, None]), factors * 4 * np.pi
    )


@pytest.mark.parametrize("shape", [(2, 3), (4, 8), (2, 4, 8), (36,)])
def test_reject_incompatible_integration_shape(shape):
    grid = GaussLegendreFourierGrid(4, 9)
    with pytest.raises(ValueError, match="angular shape"):
        grid.integrate(np.ones(shape))


@pytest.mark.parametrize(
    "grid_shape,values_shape", [((1, 1), (2, 3)), ((1, 3), (2, 3)), ((3, 1), (3, 2))]
)
def test_integration_cannot_expand_singleton_grid_axes(grid_shape, values_shape):
    grid = GaussLegendreFourierGrid(*grid_shape)
    with pytest.raises(ValueError, match="angular shape"):
        grid.integrate(np.ones(values_shape))


def test_polynomial_fourier_quadrature_exactness():
    grid = GaussLegendreFourierGrid(5, 12)
    modes = np.arange(-grid.n_phi + 1, grid.n_phi)
    waves = np.exp(1j * modes[:, None, None] * grid.phi)
    for degree in range(2 * grid.n_theta):
        values = Legendre.basis(degree)(grid.cos_theta)[:, None] * waves
        expected = 4 * np.pi * (modes == 0) * (degree == 0)
        np.testing.assert_allclose(grid.integrate(values), expected, atol=5e-14, rtol=0)


def test_spherical_harmonic_orthonormality():
    l_max = 5
    grid = GaussLegendreFourierGrid.from_bandlimit(l_max)
    theta, phi = grid.mesh()
    harmonics = np.array(
        [Ylm(l, m, theta, phi) for l in range(l_max + 1) for m in range(-l, l + 1)]
    )
    overlap = grid.integrate(harmonics[:, None].conj() * harmonics[None, :])
    np.testing.assert_allclose(overlap, np.eye(len(harmonics)), atol=5e-14, rtol=0)
