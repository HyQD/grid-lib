"""Tests for Gauss-Legendre-Lobatto cardinal polynomials."""

import numpy as np
import pytest
from numpy.polynomial import legendre

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


@pytest.fixture(scope="module")
def gll_80():
    return GaussLegendreLobatto(80, Linear_map(-1, 1))


@pytest.mark.parametrize("N", [5, 80])
def test_cardinal_functions_are_kronecker_deltas_at_nodes(N):
    gll = GaussLegendreLobatto(N, Linear_map(-1, 1))
    values = np.array(
        [gll.cardinal_function(gll.x, j) for j in range(N + 1)]
    )

    np.testing.assert_array_equal(values, np.eye(N + 1))
    assert gll.cardinal_function(gll.x[2], 2) == 1.0
    assert gll.cardinal_function(gll.x[2], 3) == 0.0


@pytest.mark.parametrize("offset", [0.0, 1e-16, 1e-13, 1e-10, 1e-7, 1e-3])
def test_polynomial_interpolation_is_accurate_next_to_nodes(gll_80, offset):
    coefficients = np.random.default_rng(1).standard_normal(81) / np.arange(1, 82)
    points = np.clip(gll_80.x + offset, -1, 1)
    node_values = legendre.legval(gll_80.x, coefficients)
    interpolant = sum(
        node_values[j] * gll_80.cardinal_function(points, j)
        for j in range(81)
    )
    exact = legendre.legval(points, coefficients)

    np.testing.assert_allclose(
        interpolant,
        exact,
        rtol=0,
        atol=1e-12 * np.abs(exact).max(),
    )


def test_scalar_and_array_evaluation_agree():
    gll = GaussLegendreLobatto(12, Linear_map(-1, 1))
    points = np.array([-0.7, 0.05, 0.9])

    for j in (0, 5, 12):
        array_values = gll.cardinal_function(points, j)
        scalar_values = [gll.cardinal_function(float(y), j) for y in points]
        assert scalar_values == pytest.approx(array_values, abs=1e-15)

    assert gll.cardinal_function(points.reshape(3, 1), 5).shape == (3, 1)
