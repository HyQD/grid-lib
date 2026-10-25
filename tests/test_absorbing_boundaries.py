import numpy as np
import pytest

from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask, mask_function
from grid_lib.spherical_coordinates.utils import mask_function as legacy_mask_function


def test_cosine_mask_values_and_legacy_import():
    radial_grid = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array(
        [
            1.0,
            1.0,
            1.0,
            np.cos(np.pi / 4.0) ** 0.25,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        mask_function(radial_grid, 5.0, 3.0, 4.0),
        expected,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        RadialCosineMask(r0=3.0)(radial_grid, 5.0),
        expected,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert legacy_mask_function is mask_function


def test_cubic_cap_has_requested_onset_and_strength():
    radial_grid = np.array([0.0, 3.0, 4.0, 5.0])

    potential = CubicCAP(alpha=0.6, eta=0.25)(radial_grid, r_max=5.0)

    np.testing.assert_array_equal(
        potential,
        -0.25j * np.array([0.0, 0.0, 1.0, 8.0]),
    )
    assert CubicCAP(alpha=0.6).eta == 1.0


@pytest.mark.parametrize("alpha", [-0.1, 1.0, np.inf])
def test_cubic_cap_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CubicCAP(alpha=alpha)


@pytest.mark.parametrize("eta", [0.0, -1.0, np.inf])
def test_cubic_cap_rejects_invalid_eta(eta):
    with pytest.raises(ValueError, match="eta"):
        CubicCAP(alpha=0.8, eta=eta)


@pytest.mark.parametrize(
    "profile",
    [RadialCosineMask(r0=3.0), CubicCAP(alpha=0.6)],
)
def test_absorber_rejects_nodes_outside_radial_domain(profile):
    with pytest.raises(ValueError, match="interval"):
        profile(np.array([0.0, 5.1]), r_max=5.0)
