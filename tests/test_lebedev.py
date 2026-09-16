import numpy as np
import pytest
from grid_lib.spherical_coordinates.angular_momentum import (
    angular_matrix_element,
)

LMAX = 4
LEBEDEV_ORDER = 2 * LMAX + 1  # = 9, an available Lebedev order


def fx(theta, phi):
    return np.sin(theta) * np.cos(phi)


def fy(theta, phi):
    return np.sin(theta) * np.sin(phi)


def fz(theta, phi):
    return np.cos(theta)


def analytic_direction_cosine_matrix_element(component, l1, m1, l2, m2):
    """
    Return the analytical matrix element

        <l1,m1 | f_component | l2,m2>

    for the Cartesian direction cosines

        f_x = sin(theta) cos(phi),
        f_y = sin(theta) sin(phi),
        f_z = cos(theta).

    The spherical harmonics are assumed to use the Condon-Shortley phase,
    as in scipy.special.sph_harm_y.
    """
    l = l2
    m = m2

    # Expansion coefficients for
    #
    #   sin(theta) exp(+i phi) Y_lm
    #
    plus = {
        (l + 1, m + 1): -np.sqrt(
            (l + m + 1) * (l + m + 2) / ((2 * l + 1) * (2 * l + 3))
        )
    }

    # Expansion coefficients for
    #
    #   sin(theta) exp(-i phi) Y_lm
    #
    minus = {
        (l + 1, m - 1): np.sqrt(
            (l - m + 1) * (l - m + 2) / ((2 * l + 1) * (2 * l + 3))
        )
    }

    if l > 0:
        plus[(l - 1, m + 1)] = np.sqrt(
            (l - m) * (l - m - 1) / ((2 * l - 1) * (2 * l + 1))
        )

        minus[(l - 1, m - 1)] = -np.sqrt(
            (l + m) * (l + m - 1) / ((2 * l - 1) * (2 * l + 1))
        )

    state = (l1, m1)

    if component == "x":
        # sin(theta) cos(phi)
        # = 1/2 [sin(theta)e^(i phi) + sin(theta)e^(-i phi)]
        return 0.5 * (plus.get(state, 0.0) + minus.get(state, 0.0))

    if component == "y":
        # sin(theta) sin(phi)
        # = 1/(2i) [sin(theta)e^(i phi) - sin(theta)e^(-i phi)]
        return (plus.get(state, 0.0) - minus.get(state, 0.0)) / (2j)

    if component == "z":
        # cos(theta) Y_lm
        # = A_{l+1,m} Y_{l+1,m} + A_{l,m} Y_{l-1,m}
        if state == (l + 1, m):
            return np.sqrt(((l + 1) ** 2 - m**2) / ((2 * l + 1) * (2 * l + 3)))

        if l > 0 and state == (l - 1, m):
            return np.sqrt((l**2 - m**2) / ((2 * l - 1) * (2 * l + 1)))

        return 0.0

    raise ValueError(f"Unknown component {component!r}")


def angular_momentum_states(lmax):
    """Return all (l, m) states with 0 <= l <= lmax."""
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


@pytest.mark.parametrize(
    "component, f",
    [
        ("x", fx),
        ("y", fy),
        ("z", fz),
    ],
)
def test_direction_cosine_matrix_elements(component, f):
    states = angular_momentum_states(LMAX)

    for l1, m1 in states:
        for l2, m2 in states:
            numerical = angular_matrix_element(
                l1,
                m1,
                l2,
                m2,
                f,
                order=LEBEDEV_ORDER,
            )

            analytical = analytic_direction_cosine_matrix_element(
                component,
                l1,
                m1,
                l2,
                m2,
            )

            np.testing.assert_allclose(
                numerical,
                analytical,
                rtol=0.0,
                atol=1e-13,
                err_msg=(f"{component}: " f"<{l1},{m1}|f|{l2},{m2}>"),
            )
