import numpy as np

from grid_lib.spherical_coordinates.utils import cartesian_to_spherical


def test_cartesian_to_spherical_on_axes():
    test_cases = [
        ((1, 0, 0), (1.0, np.pi / 2, 0.0)),
        ((-1, 0, 0), (1.0, np.pi / 2, np.pi)),
        ((0, 1, 0), (1.0, np.pi / 2, np.pi / 2)),
        ((0, -1, 0), (1.0, np.pi / 2, 3 * np.pi / 2)),
        ((0, 0, 1), (1.0, 0.0, 0.0)),
        ((0, 0, -1), (1.0, np.pi, 0.0)),
    ]

    for point, expected in test_cases:
        computed = cartesian_to_spherical(point)
        np.testing.assert_allclose(computed, expected, rtol=0.0, atol=1e-15)


def test_cartesian_to_spherical_off_axis():
    point = (0, 1, 1)
    expected = (np.sqrt(2), np.pi / 4, np.pi / 2)

    computed = cartesian_to_spherical(point)

    np.testing.assert_allclose(computed, expected, rtol=0.0, atol=1e-15)
