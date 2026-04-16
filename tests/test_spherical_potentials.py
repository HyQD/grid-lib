import numpy as np
import pytest

from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    clamped_molecular_potential_Poisson,
    clamped_molecular_potential_quadrature,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb
from grid_lib.spherical_coordinates.utils import Ylm, cartesian_to_spherical
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


def test_clamped_molecular_potential_quadrature_for_centered_nucleus():
    r = np.array([0.5, 1.0, 2.0])
    V_LM = clamped_molecular_potential_quadrature(
        r,
        positions=[[0.0, 0.0, 0.0]],
        charges=[2.0],
        L_max=3,
        M_max=2,
    )

    I_00 = LM_to_I(0, 0, 3, 2)
    np.testing.assert_allclose(V_LM[I_00], -2.0 * np.sqrt(4 * np.pi) / r)

    mask = np.ones(V_LM.shape[0], dtype=bool)
    mask[I_00] = False
    np.testing.assert_allclose(V_LM[mask], 0.0, atol=1e-14, rtol=0.0)


def test_clamped_molecular_potential_quadrature_for_displaced_nucleus_on_z_axis():
    r = np.array([0.5, 1.0, 2.0, 4.0])
    a = 1.5
    L_max = 4
    M_max = 3

    V_LM = clamped_molecular_potential_quadrature(
        r,
        positions=[[0.0, 0.0, a]],
        charges=[1.0],
        L_max=L_max,
        M_max=M_max,
    )

    r_min = np.minimum(r, a)
    r_max = np.maximum(r, a)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), L_max + 1):
            I_LM = LM_to_I(L, M, L_max, M_max)

            if M == 0:
                expected = -np.sqrt(4 * np.pi / (2 * L + 1)) * (
                    r_min**L / r_max ** (L + 1)
                )
                np.testing.assert_allclose(V_LM[I_LM], expected)
            else:
                np.testing.assert_allclose(
                    V_LM[I_LM], 0.0, atol=1e-14, rtol=0.0
                )


def test_clamped_molecular_potential_poisson_assembles_components_from_W():
    nodes = np.arange(0.0, 6.0 + 1.0, 1.0)
    n_points = np.ones((len(nodes) - 1,), dtype=int) * 6
    femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

    r = femdvr.r[1:-1]
    weights = femdvr.weights[1:-1]
    L_max = 4
    M_max = 3
    W = radial_Coulomb(femdvr, L_max + 1)

    r_idx = len(r) // 3
    a = r[r_idx]
    position = np.array([a / np.sqrt(3), a / np.sqrt(3), a / np.sqrt(3)])
    V_poisson = clamped_molecular_potential_Poisson(
        r,
        W,
        weights,
        positions=[position],
        charges=[1.0],
        L_max=L_max,
        M_max=M_max,
    )

    _, theta, phi = cartesian_to_spherical(position)
    V_expected = np.zeros_like(V_poisson, dtype=complex)

    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), L_max + 1):
            I_LM = LM_to_I(L, M, L_max, M_max)
            V_expected[I_LM] = -Ylm(L, M, theta, phi) * W[L, :, r_idx] / weights[r_idx]

    np.testing.assert_allclose(V_poisson, V_expected, atol=1e-14, rtol=1e-14)


def test_clamped_molecular_potential_poisson_warns_and_uses_nearest_grid_point():
    nodes = np.arange(0.0, 6.0 + 1.0, 1.0)
    n_points = np.ones((len(nodes) - 1,), dtype=int) * 6
    femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

    r = femdvr.r[1:-1]
    weights = femdvr.weights[1:-1]
    L_max = 4
    M_max = 3
    W = radial_Coulomb(femdvr, L_max + 1)

    a = 1.23
    position = np.array([0.0, 0.0, a])
    nearest_idx = np.argmin(np.abs(r - a))
    snapped_position = np.array([0.0, 0.0, r[nearest_idx]])

    with pytest.warns(
        UserWarning, match="not a grid point; using nearest grid point"
    ):
        V_poisson = clamped_molecular_potential_Poisson(
            r,
            W,
            weights,
            positions=[position],
            charges=[1.0],
            L_max=L_max,
            M_max=M_max,
        )

    V_expected = clamped_molecular_potential_Poisson(
        r,
        W,
        weights,
        positions=[snapped_position],
        charges=[1.0],
        L_max=L_max,
        M_max=M_max,
    )

    np.testing.assert_allclose(V_poisson, V_expected, atol=1e-14, rtol=1e-14)
