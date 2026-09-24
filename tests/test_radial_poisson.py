import numpy as np
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.radial_poisson import (
    solve_radial_Poisson_dvr,
    solve_radial_Poisson_femdvr_two_grid,
)
from grid_lib.spherical_coordinates.radial_Coulomb import (
    radial_Coulomb,
    radial_Coulomb_femdvr_two_grid,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from matplotlib import pyplot as plt
from scipy.special import erf


def test_radial_poisson():
    """
    The test solves the radial Poisson equation
    .. math::
        \frac{\partial^2}{\partial r^2} \tilde{V}_0(r) = -1/r * u(r)^2,
    for the two test cases:

    1. u(r) = N*r*exp(-r^2), where N is the normalization constant such that
       .. math::
           \int_0^{r_{max}} dr r^2 * u(r)^2 = 1
        The exact solution is given by
        .. math::
            \tilde{V}_0(r) = \text{erf}(\sqrt{2} * r)
    2. u(r) = N*r*exp(-r), where N is the normalization.
        The exact solution is given by
        .. math::
            \tilde{V}_0(r) = 1 - (r + 1) * exp(-2*r)

    The numerical solution (at the inner grid points) is obtained from the solution of the radial Poisson equation in the DVR basis
    .. math::
        \tilde{v}_L(r_\alpha) = \sum_{\beta} u(r_\beta)  \tilde{v}^{DVR}_{L}(r_\alpha; \chi_\beta) u(r_\beta)

    """

    N_r = 128
    r_min = 0
    r_max = 40
    GLL = GaussLegendreLobatto(N_r, Linear_map(r_min, r_max), symmetrize=False)

    tilde_V0_dvr = solve_radial_Poisson_dvr(GLL, n_L=1)

    r = GLL.r[1:-1]
    w_r = GLL.weights[1:-1]
    r_dot = GLL.r_dot[1:-1]

    u_1s_gaussian = r * np.exp(-(r**2))
    norm_psi = np.dot(w_r, u_1s_gaussian**2)
    u_1s_gaussian /= np.sqrt(norm_psi)

    tilde_V0 = np.einsum(
        "b, ab, b->a",
        u_1s_gaussian,
        tilde_V0_dvr[0],
        u_1s_gaussian,
        optimize=True,
    )
    tilde_V0_exact_1s_gaussian = erf(
        np.sqrt(2) * r
    )  # The exact solution of the radial Poisson equation to a 1s Gaussian

    assert np.linalg.norm(tilde_V0 - tilde_V0_exact_1s_gaussian) < 1e-10
    assert np.max(np.abs(tilde_V0 - tilde_V0_exact_1s_gaussian)) < 1e-10

    u_1s = r * np.exp(-r)
    norm_1s = np.dot(w_r, u_1s**2)
    u_1s /= np.sqrt(norm_1s)
    tilde_V0_1s = np.einsum(
        "b, ab, b->a", u_1s, tilde_V0_dvr[0], u_1s, optimize=True
    )

    tilde_V0_exact_1s = 1 - (r + 1) * np.exp(
        -2 * r
    )  # The exact solution of the radial Poisson equation to a 1s Gaussian

    assert np.linalg.norm(tilde_V0_1s - tilde_V0_exact_1s) < 1e-10
    assert np.max(np.abs(tilde_V0_1s - tilde_V0_exact_1s)) < 1e-10


def test_radial_poisson_fem():
    """
    The test solves the radial Poisson equation using the FEM-DVR method.
    The test is similar to the one in `test_radial_poisson`, but uses the
    `FEMDVR` class to solve the radial Poisson equation.

    Test for a for uniform distribution of nodes and a non-uniform distribution of nodes.
    """

    a = 0
    b = 40
    n_elem = 4
    points_per_elem = 51

    nodes_list = [np.linspace(a, b, n_elem + 1), np.array([0, 4, 16, 28, 40])]
    n_points_list = [
        np.ones((n_elem,), dtype=int) * points_per_elem,
        np.array([31, 21, 11, 11]),
    ]

    for nodes, n_points in zip(nodes_list, n_points_list):
        femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

        tilde_V0_dvr = solve_radial_Poisson_dvr(femdvr, n_L=1)

        r = femdvr.r[1:-1]
        w_r = femdvr.weights[1:-1]
        r_dot = femdvr.r_dot[1:-1]

        u_1s_gaussian = r * np.exp(-(r**2))
        norm_psi = np.dot(w_r, u_1s_gaussian**2)
        u_1s_gaussian /= np.sqrt(norm_psi)

        tilde_V0 = np.einsum(
            "b, ab, b->a",
            u_1s_gaussian,
            tilde_V0_dvr[0],
            u_1s_gaussian,
            optimize=True,
        )
        tilde_V0_exact_1s_gaussian = erf(
            np.sqrt(2) * r
        )  # The exact solution of the radial Poisson equation to a 1s Gaussian

        assert np.linalg.norm(tilde_V0 - tilde_V0_exact_1s_gaussian) < 1e-10
        assert np.max(np.abs(tilde_V0 - tilde_V0_exact_1s_gaussian)) < 1e-10

        u_1s = r * np.exp(-r)
        norm_1s = np.dot(w_r, u_1s**2)
        u_1s /= np.sqrt(norm_1s)

        tilde_V0_1s = np.einsum(
            "b, ab, b->a", u_1s, tilde_V0_dvr[0], u_1s, optimize=True
        )

        tilde_V0_exact_1s = 1 - (r + 1) * np.exp(
            -2 * r
        )  # The exact solution of the radial Poisson equation to a 1s Gaussian

        assert np.linalg.norm(tilde_V0_1s - tilde_V0_exact_1s) < 1e-10
        assert np.max(np.abs(tilde_V0_1s - tilde_V0_exact_1s)) < 1e-10


def _femdvr(r_max, n_elem, points_per_elem):
    nodes = np.linspace(0.0, r_max, n_elem + 1)
    n_points = np.ones((n_elem,), dtype=int) * points_per_elem
    return FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)


def test_femdvr_interpolation_matrix_is_identity_on_grid_nodes():
    femdvr = _femdvr(r_max=8.0, n_elem=4, points_per_elem=11)

    interpolation = femdvr.interpolation_matrix(femdvr.r)

    np.testing.assert_allclose(
        interpolation, np.eye(len(femdvr.r)), atol=1e-14, rtol=0.0
    )


def test_two_grid_gamma_one_matches_existing_radial_coulomb():
    femdvr = _femdvr(r_max=20.0, n_elem=4, points_per_elem=11)

    W_old = radial_Coulomb(femdvr, n_L=4)
    W_new = radial_Coulomb_femdvr_two_grid(
        femdvr, femdvr, n_L=4, charge_gamma_terms=[(1.0, 1.0, 1.0)]
    )

    np.testing.assert_allclose(W_new, W_old, atol=1e-12, rtol=1e-12)


def test_two_grid_negative_gamma_has_multipole_parity():
    r1_grid = _femdvr(r_max=12.0, n_elem=3, points_per_elem=11)
    r2_grid = _femdvr(r_max=6.0, n_elem=3, points_per_elem=11)

    W_positive = radial_Coulomb_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, charge_gamma_terms=[(1.0, 1.0, 2.0)]
    )
    W_negative = radial_Coulomb_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, charge_gamma_terms=[(1.0, 1.0, -2.0)]
    )

    for L in range(4):
        np.testing.assert_allclose(
            W_negative[L],
            (-1) ** L * W_positive[L],
            atol=1e-12,
            rtol=1e-12,
        )


def test_two_grid_scaled_nodes_match_discrete_same_grid_reference():
    gamma = 2.0
    r1_grid = _femdvr(r_max=12.0, n_elem=3, points_per_elem=11)
    r2_grid = _femdvr(r_max=6.0, n_elem=3, points_per_elem=11)

    W = radial_Coulomb_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, charge_gamma_terms=[(1.0, 1.0, gamma)]
    )
    W_reference_grid = radial_Coulomb(r1_grid, n_L=4)

    r1 = r1_grid.r[1:-1]
    r2 = r2_grid.r[1:-1]
    w1 = r1_grid.weights[1:-1]
    w2 = r2_grid.weights[1:-1]
    source_radius = gamma * r2

    source_indices = np.array(
        [np.argmin(np.abs(r1 - source)) for source in source_radius]
    )
    np.testing.assert_allclose(r1[source_indices], source_radius)

    weight_scale = w2 / w1[source_indices]
    W_discrete_reference = (
        W_reference_grid[:, :, source_indices]
        * weight_scale[np.newaxis, np.newaxis, :]
    )
    np.testing.assert_allclose(W, W_discrete_reference, atol=1e-12, rtol=1e-12)


def test_two_grid_scaled_nodes_are_close_to_analytic_kernel():
    gamma = 2.0
    r1_grid = _femdvr(r_max=12.0, n_elem=3, points_per_elem=11)
    r2_grid = _femdvr(r_max=6.0, n_elem=3, points_per_elem=11)

    W = radial_Coulomb_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, charge_gamma_terms=[(1.0, 1.0, gamma)]
    )

    r1 = r1_grid.r[1:-1]
    r2 = r2_grid.r[1:-1]
    w2 = r2_grid.weights[1:-1]
    source_radius = gamma * r2

    for L in range(4):
        r_less = np.minimum(r1[:, np.newaxis], source_radius[np.newaxis, :])
        r_greater = np.maximum(
            r1[:, np.newaxis], source_radius[np.newaxis, :]
        )
        W_exact = (
            (4 * np.pi / (2 * L + 1))
            * r_less**L
            / r_greater ** (L + 1)
            * w2[np.newaxis, :]
        )
        relative_error = np.max(np.abs(W[L] - W_exact)) / np.max(
            np.abs(W_exact)
        )
        assert relative_error < 0.70


def test_two_grid_gamma_zero_is_monopole_only():
    r1_grid = _femdvr(r_max=10.0, n_elem=2, points_per_elem=11)
    r2_grid = _femdvr(r_max=4.0, n_elem=2, points_per_elem=11)

    u_L = solve_radial_Poisson_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, gamma=0.0
    )
    W = radial_Coulomb_femdvr_two_grid(
        r1_grid, r2_grid, n_L=4, charge_gamma_terms=[(2.0, -3.0, 0.0)]
    )

    r1 = r1_grid.r[1:-1]
    w2 = r2_grid.weights[1:-1]

    np.testing.assert_allclose(
        u_L[0], np.repeat(w2[np.newaxis, :], len(r1), axis=0)
    )
    np.testing.assert_allclose(u_L[1:], 0.0)

    W0_exact = -6.0 * 4 * np.pi * w2[np.newaxis, :] / r1[:, np.newaxis]
    np.testing.assert_allclose(W[0], W0_exact)
    np.testing.assert_allclose(W[1:], 0.0)
