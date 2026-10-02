import numpy as np
import pytest

from grid_lib.pseudospectral_grids import RadialSincDVR, setup_grid


def _expected_l_adapted_matrices(r_max, n_r, l_max):
    dr = r_max / n_r
    j = np.arange(1, n_r + 1)
    I, J = np.meshgrid(j, j, indexing="ij")
    mask = I != J

    D1_direct = np.zeros((n_r, n_r))
    D1_direct[mask] = (-1.0) ** (I[mask] - J[mask]) / (
        dr * (I[mask] - J[mask])
    )
    D1_image = (-1.0) ** (I + J) / (dr * (I + J))

    D2_direct = np.zeros((n_r, n_r))
    D2_direct[mask] = (
        -2.0
        * (-1.0) ** (I[mask] - J[mask])
        / (dr**2 * (I[mask] - J[mask]) ** 2)
    )
    np.fill_diagonal(D2_direct, -np.pi**2 / (3.0 * dr**2))
    D2_image = -2.0 * (-1.0) ** (I + J) / (dr**2 * (I + J) ** 2)

    D1_l = np.empty((l_max + 1, n_r, n_r))
    D2_l = np.empty((l_max + 1, n_r, n_r))
    for l in range(l_max + 1):
        s_l = (-1.0) ** (l + 1)
        D1_l[l] = D1_direct + s_l * D1_image
        D2_l[l] = D2_direct + s_l * D2_image

    return D1_l, D2_l


def test_radial_sinc_dvr_builds_l_adapted_matrices():
    dvr = RadialSincDVR(r_max=8.0, n_r=4, l_max=3)

    assert dvr.dr == 2.0
    np.testing.assert_allclose(dvr.r, [2.0, 4.0, 6.0, 8.0])
    assert dvr.D1_l.shape == (4, 4, 4)
    assert dvr.D2_l.shape == (4, 4, 4)

    D1_expected, D2_expected = _expected_l_adapted_matrices(8.0, 4, 3)
    np.testing.assert_allclose(dvr.D1_l, D1_expected)
    np.testing.assert_allclose(dvr.D2_l, D2_expected)


def test_radial_sinc_dvr_l_zero_matches_backward_compatible_matrices():
    dvr = RadialSincDVR(6.0, 3, l_max=2)

    np.testing.assert_allclose(dvr.D1, dvr.D1_l[0])
    np.testing.assert_allclose(dvr.D2, dvr.D2_l[0])

    r = dvr.r
    np.testing.assert_allclose(np.diag(dvr.D1_l[0]), -0.5 / r)
    np.testing.assert_allclose(np.diag(dvr.D1_l[1]), 0.5 / r)
    np.testing.assert_allclose(
        np.diag(dvr.D2_l[0]), -np.pi**2 / (3.0 * dvr.dr**2) + 0.5 / r**2
    )
    np.testing.assert_allclose(
        np.diag(dvr.D2_l[1]), -np.pi**2 / (3.0 * dvr.dr**2) - 0.5 / r**2
    )


def test_radial_sinc_setup_accepts_n_r_and_l_max():
    dvr = setup_grid("radial-sinc", {"r_max": 10.0, "n_r": 5, "l_max": 2})

    assert isinstance(dvr, RadialSincDVR)
    assert dvr.n_r == 5
    assert dvr.N == 5
    assert dvr.l_max == 2
    assert dvr.D1_l.shape == (3, 5, 5)


def test_radial_sinc_dvr_validates_inputs():
    with pytest.raises(ValueError, match="r_max must be positive"):
        RadialSincDVR(0.0, 4)
    with pytest.raises(ValueError, match="n_r must be >= 1"):
        RadialSincDVR(1.0, 0)
    with pytest.raises(ValueError, match="l_max must be >= 0"):
        RadialSincDVR(1.0, 4, -1)
    with pytest.raises(ValueError, match="N and n_r must agree"):
        RadialSincDVR(1.0, n_r=4, N=5)
