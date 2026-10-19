from types import SimpleNamespace

import numpy as np
import pytest

from grid_lib.pseudospectral_grids import (
    GaussLegendre,
    Linear_map,
    setup_femdvr_uniform,
)
from grid_lib.spherical_coordinates import (
    GaussLegendreFourierGrid,
    SphericalNodalDifferentialOperator,
    SphericalVectorField,
)


@pytest.fixture
def differential_operator():
    radial_grid = GaussLegendre(8, Linear_map(0.4, 2.1))
    angular_grid = GaussLegendreFourierGrid(7, 15)
    return SphericalNodalDifferentialOperator(radial_grid, angular_grid)


def coordinates(operator):
    r = operator.r[:, None, None]
    x = operator.cos_theta[None, :, None]
    phi = operator.angular_grid.phi[None, None, :]
    return r, x, phi


def test_coordinate_derivatives_support_complex_batches(differential_operator):
    operator = differential_operator
    r, x, phi = coordinates(operator)
    modes = np.array([[-3], [2]])[:, None, None, :]
    radial_factors = np.array([1.0, 2.0j])[:, None, None, None]
    values = radial_factors * r**3 * (1 + x + x**2) * np.exp(1j * modes * phi)

    np.testing.assert_allclose(
        operator.radial_derivative(values),
        radial_factors * 3 * r**2 * (1 + x + x**2) * np.exp(1j * modes * phi),
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        operator.polar_derivative(values),
        -radial_factors
        * r**3
        * np.sqrt(1 - x**2)
        * (1 + 2 * x)
        * np.exp(1j * modes * phi),
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        operator.azimuthal_derivative(values),
        1j * modes * values,
        atol=2e-12,
        rtol=2e-12,
    )


def test_gradient_of_cartesian_z(differential_operator):
    operator = differential_operator
    r, x, _ = coordinates(operator)
    values = np.broadcast_to(r * x, operator.shape)
    gradient = operator.gradient(values)

    np.testing.assert_allclose(
        gradient.radial, np.broadcast_to(x, operator.shape), atol=2e-13
    )
    np.testing.assert_allclose(
        gradient.polar,
        np.broadcast_to(-np.sqrt(1 - x**2), operator.shape),
        atol=2e-13,
    )
    np.testing.assert_allclose(gradient.azimuthal, 0.0, atol=2e-13)
    np.testing.assert_allclose(operator.gradient_squared(values), 1.0, atol=5e-13)
    np.testing.assert_allclose(
        operator.gradient_norm_squared(values), 1.0, atol=5e-13
    )


def test_divergence_and_laplacian_polynomials(differential_operator):
    operator = differential_operator
    r, _, _ = coordinates(operator)
    zeros = np.zeros(operator.shape)
    vector = SphericalVectorField(
        radial=np.broadcast_to(r, operator.shape),
        polar=zeros,
        azimuthal=zeros,
    )
    np.testing.assert_allclose(operator.divergence(vector), 3.0, atol=2e-11)

    r_squared = np.broadcast_to(r**2, operator.shape)
    np.testing.assert_allclose(operator.laplacian(r_squared), 6.0, atol=2e-10)


def test_divergence_of_angular_components(differential_operator):
    operator = differential_operator
    r, x, phi = coordinates(operator)
    sin_theta = np.sqrt(1 - x**2)
    zeros = np.zeros(operator.shape)
    vector = SphericalVectorField(
        radial=zeros,
        polar=np.broadcast_to(sin_theta, operator.shape),
        azimuthal=np.broadcast_to(sin_theta * np.cos(phi), operator.shape),
    )
    expected = 2 * x / r - np.sin(phi) / r
    np.testing.assert_allclose(
        operator.divergence(vector),
        np.broadcast_to(expected, operator.shape),
        atol=2e-12,
        rtol=2e-12,
    )


def test_weighted_adjoint_identity(differential_operator):
    operator = differential_operator
    rng = np.random.default_rng(9274)
    scalar = rng.normal(size=operator.shape) + 1j * rng.normal(size=operator.shape)
    vector = SphericalVectorField(
        radial=rng.normal(size=operator.shape) + 1j * rng.normal(size=operator.shape),
        polar=rng.normal(size=operator.shape) + 1j * rng.normal(size=operator.shape),
        azimuthal=rng.normal(size=operator.shape)
        + 1j * rng.normal(size=operator.shape),
    )

    gradient = operator.gradient(scalar)
    left = operator.integrate(
        gradient.radial.conj() * vector.radial
        + gradient.polar.conj() * vector.polar
        + gradient.azimuthal.conj() * vector.azimuthal
    )
    right = operator.inner_product(scalar, operator.gradient_adjoint(vector))
    np.testing.assert_allclose(left, right, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        operator.adjoint_divergence(vector),
        -operator.gradient_adjoint(vector),
        atol=0,
        rtol=0,
    )


def test_even_azimuthal_grid_sets_nyquist_derivative_to_zero():
    radial_grid = GaussLegendre(3, Linear_map(0.5, 1.0))
    angular_grid = GaussLegendreFourierGrid(3, 8)
    operator = SphericalNodalDifferentialOperator(radial_grid, angular_grid)
    nyquist = (-1.0) ** np.arange(angular_grid.n_phi)
    values = np.broadcast_to(nyquist, operator.shape)
    np.testing.assert_allclose(
        operator.azimuthal_derivative(values), 0.0, atol=1e-14
    )


def test_radial_subset_and_validation():
    radial_grid = SimpleNamespace(
        r=np.array([0.0, 0.5, 1.0]),
        weights=np.ones(3),
        D1=np.eye(3),
    )
    angular_grid = GaussLegendreFourierGrid(2, 3)

    with pytest.raises(ValueError, match="strictly positive"):
        SphericalNodalDifferentialOperator(radial_grid, angular_grid)
    with pytest.raises(ValueError, match="at least one"):
        SphericalNodalDifferentialOperator(
            radial_grid, angular_grid, radial_indices=slice(0, 0)
        )

    operator = SphericalNodalDifferentialOperator(
        radial_grid, angular_grid, radial_indices=slice(1, None)
    )
    assert operator.shape == (2, 2, 3)
    np.testing.assert_array_equal(operator.radial_indices, [1, 2])
    assert "n_r=2" in repr(operator)

    with pytest.raises(ValueError, match="spherical grid shape"):
        operator.gradient(np.zeros((4, 2, 3)))
    with pytest.raises(TypeError, match="SphericalVectorField"):
        operator.divergence(np.zeros(operator.shape))


def test_full_femdvr_values_supply_nonzero_boundaries():
    radial_grid = setup_femdvr_uniform(0.0, 2.0, 2, 7)
    angular_grid = GaussLegendreFourierGrid(4, 9)
    operator = SphericalNodalDifferentialOperator(
        radial_grid, angular_grid, radial_indices=slice(1, -1)
    )
    full_r = radial_grid.r[:, None, None]
    full_values = np.broadcast_to(1 + full_r**2, operator.full_shape)

    np.testing.assert_allclose(
        operator.radial_derivative(full_values),
        np.broadcast_to(2 * operator.r[:, None, None], operator.shape),
        atol=2e-13,
    )
    np.testing.assert_allclose(operator.laplacian(full_values), 6.0, atol=2e-11)

    zeros = np.zeros(operator.full_shape)
    full_vector = SphericalVectorField(
        radial=np.broadcast_to(full_r, operator.full_shape),
        polar=zeros,
        azimuthal=zeros,
    )
    np.testing.assert_allclose(operator.divergence(full_vector), 3.0, atol=2e-12)


def test_integrate_and_inner_product(differential_operator):
    operator = differential_operator
    expected_volume = 4 * np.pi * (2.1**3 - 0.4**3) / 3
    np.testing.assert_allclose(
        operator.integrate(np.ones(operator.shape)), expected_volume
    )
    values = np.full(operator.shape, 1 + 2j)
    np.testing.assert_allclose(
        operator.inner_product(values, values), 5 * expected_volume
    )
