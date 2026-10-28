import numpy as np
import pytest

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    angular_matrix_element,
    number_of_lm_states,
)
from grid_lib.spherical_coordinates.multipole_potential import (
    SphericalMultipolePotential,
)
from grid_lib.spherical_coordinates.potentials import (
    clamped_molecular_potential_Poisson,
    clamped_molecular_potential_quadrature,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb
from grid_lib.spherical_coordinates.utils import Ylm, cartesian_to_spherical


def test_centered_charge_acts_as_minus_z_over_r_on_every_channel():
    r = np.array([0.25, 0.8, 2.0])
    potential = SphericalMultipolePotential.from_point_charges(
        r,
        [[0.0, 0.0, 0.0]],
        [2.0],
        l_max=3,
        m_max=2,
    )
    rng = np.random.default_rng(4)
    orbitals = rng.normal(size=(2, *potential.orbital_shape)) + 1j * rng.normal(
        size=(2, *potential.orbital_shape)
    )

    result = potential(orbitals)

    np.testing.assert_allclose(result, (-2.0 / r)[None, :, None] * orbitals)
    assert result.shape == orbitals.shape
    assert potential.L_max == 6
    assert potential.M_max == 4


def test_constructor_copies_inputs_and_validates_trailing_orbital_shape():
    r = np.array([0.5, 1.0])
    coefficients = np.zeros((1, 2))
    potential = SphericalMultipolePotential(
        r, coefficients, l_max=1, m_max=0, L_max=0, M_max=0
    )
    r[:] = -1.0
    coefficients[:] = 3.0

    np.testing.assert_array_equal(potential.r, [0.5, 1.0])
    np.testing.assert_array_equal(potential.coefficients, 0.0)
    with pytest.raises(ValueError, match="trailing shape"):
        potential.apply(np.zeros((2, 3)))


def test_axial_diatomic_preserves_m_and_exposes_matching_fixed_m_blocks():
    r = np.array([0.4, 1.0, 2.5])
    a = 1.0
    potential = SphericalMultipolePotential.from_point_charges(
        r,
        [[0.0, 0.0, -a], [0.0, 0.0, a]],
        [1.0, 2.0],
        l_max=3,
        m_max=2,
        L_max=6,
        M_max=4,
    )
    full = potential.angular_matrix()

    assert potential.is_axisymmetric
    for m in range(-2, 3):
        indices = [
            LM_to_I(l, m, 3, 2) for l in range(abs(m), 4)
        ]
        np.testing.assert_allclose(
            potential.fixed_m_matrix(m), full[:, indices][:, :, indices]
        )

    for i, (_, m1) in enumerate(potential.orbital_lm):
        for j, (_, m2) in enumerate(potential.orbital_lm):
            if m1 != m2:
                np.testing.assert_allclose(full[:, i, j], 0.0, atol=1e-14)


def test_midpoint_diatomic_even_and_odd_multipoles():
    r = np.array([0.5, 1.0, 3.0])
    a = 1.0
    L_max = 5
    homonuclear = clamped_molecular_potential_quadrature(
        r,
        [[0.0, 0.0, -a], [0.0, 0.0, a]],
        [1.0, 1.0],
        L_max,
        0,
    )
    heteronuclear = clamped_molecular_potential_quadrature(
        r,
        [[0.0, 0.0, -a], [0.0, 0.0, a]],
        [1.0, 2.0],
        L_max,
        0,
    )
    radial = np.minimum(r, a) ** np.arange(L_max + 1)[:, None]
    radial /= np.maximum(r, a) ** (np.arange(L_max + 1)[:, None] + 1)

    for L in range(L_max + 1):
        index = LM_to_I(L, 0, L_max, 0)
        if L % 2:
            np.testing.assert_allclose(homonuclear[index], 0.0, atol=1e-14)
        expected = (
            -np.sqrt(4 * np.pi / (2 * L + 1))
            * ((-1) ** L + 2.0)
            * radial[L]
        )
        np.testing.assert_allclose(heteronuclear[index], expected)


def test_off_axis_point_charge_coefficients_use_conjugated_harmonics():
    r = np.array([0.6, 1.8])
    position = np.array([0.3, -0.4, 1.2])
    radius, theta, phi = cartesian_to_spherical(position)
    coefficients = clamped_molecular_potential_quadrature(
        r, [position], [1.3], L_max=3, M_max=3
    )
    for M in range(-3, 4):
        for L in range(abs(M), 4):
            index = LM_to_I(L, M, 3, 3)
            expected = (
                -1.3
                * 4.0
                * np.pi
                / (2 * L + 1)
                * np.minimum(r, radius) ** L
                / np.maximum(r, radius) ** (L + 1)
                * np.conj(Ylm(L, M, theta, phi))
            )
            np.testing.assert_allclose(coefficients[index], expected)


def test_real_off_axis_point_charge_gives_hermitian_projected_operator():
    potential = SphericalMultipolePotential.from_point_charges(
        [0.6, 1.8],
        [[0.3, -0.4, 1.2]],
        [1.3],
        l_max=2,
        m_max=2,
        L_max=4,
        M_max=4,
    )
    matrix = potential.angular_matrix()
    rng = np.random.default_rng(8)
    orbitals = rng.normal(size=(3, *potential.orbital_shape)) + 1j * rng.normal(
        size=(3, *potential.orbital_shape)
    )

    np.testing.assert_allclose(
        matrix, matrix.conj().swapaxes(-1, -2), atol=2e-14, rtol=2e-14
    )
    np.testing.assert_allclose(
        potential.apply(orbitals),
        np.einsum("rij,...rj->...ri", matrix, orbitals),
    )
    assert not potential.is_axisymmetric
    with pytest.raises(ValueError, match="axisymmetric"):
        potential.fixed_m_matrix(0)


def test_projected_action_matches_direct_angular_quadrature():
    r = np.array([0.7, 2.1])
    position = np.array([0.3, -0.4, 1.2])
    charge = 1.3
    potential = SphericalMultipolePotential.from_point_charges(
        r,
        [position],
        [charge],
        l_max=2,
        m_max=2,
        L_max=4,
        M_max=4,
    )
    matrix = potential.angular_matrix()
    pairs = ((0, 0), (2, 2), (4, 1), (5, 7))

    for radial_index, radius in enumerate(r):
        def point_charge(theta, phi):
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            distance = np.sqrt(
                (x - position[0]) ** 2
                + (y - position[1]) ** 2
                + (z - position[2]) ** 2
            )
            return -charge / distance

        for row, column in pairs:
            l1, m1 = potential.orbital_lm[row]
            l2, m2 = potential.orbital_lm[column]
            expected = angular_matrix_element(
                l1, m1, l2, m2, point_charge, order=59
            )
            np.testing.assert_allclose(
                matrix[radial_index, row, column],
                expected,
                atol=2e-12,
                rtol=2e-12,
            )


def test_poisson_factory_reuses_poisson_coefficients():
    nodes = np.arange(0.0, 4.0 + 1.0, 1.0)
    femdvr = FEMDVR(
        nodes,
        np.full(len(nodes) - 1, 6),
        Linear_map,
        GaussLegendreLobatto,
    )
    r = femdvr.r[1:-1]
    weights = femdvr.weights[1:-1]
    W = radial_Coulomb(femdvr, 5)
    radius = r[len(r) // 3]
    positions = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    charges = [1.0, 2.0]
    expected = clamped_molecular_potential_Poisson(
        r, W, weights, positions, charges, L_max=4, M_max=0
    )

    potential = SphericalMultipolePotential.from_point_charges_poisson(
        r,
        W,
        weights,
        positions,
        charges,
        l_max=2,
        m_max=0,
        L_max=4,
        M_max=0,
    )

    np.testing.assert_allclose(potential.coefficients, expected)


def test_multipoles_above_orbital_triangle_limit_have_zero_action():
    r = np.array([0.5, 1.0])
    coefficients = np.zeros((number_of_lm_states(3, 0), len(r)))
    coefficients[LM_to_I(3, 0, 3, 0)] = [2.0, 3.0]
    potential = SphericalMultipolePotential(
        r, coefficients, l_max=1, m_max=0, L_max=3, M_max=0
    )

    np.testing.assert_array_equal(
        potential.apply(np.ones(potential.orbital_shape)), 0.0
    )
