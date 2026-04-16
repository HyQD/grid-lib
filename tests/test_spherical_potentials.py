import numpy as np

from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    clamped_molecular_potential_quadrature,
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
