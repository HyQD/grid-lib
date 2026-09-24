import numpy as np
from grid_lib.spherical_coordinates.radial_poisson import (
    solve_radial_Poisson_dvr,
    solve_radial_Poisson_femdvr_two_grid,
)


def radial_Coulomb(GLL, n_L):

    tilde_V = solve_radial_Poisson_dvr(GLL, n_L)
    r = GLL.r[1:-1]
    n_r = len(r)

    W = np.zeros((n_L, n_r, n_r))
    for L in range(0, n_L):
        W[L] = (4 * np.pi / (2 * L + 1)) * tilde_V[L] / r[:, np.newaxis]

    return W


def _as_charge_gamma_terms(charge_gamma_terms):
    terms = list(charge_gamma_terms)
    for term in terms:
        if len(term) != 3:
            raise ValueError(
                "Each Coulomb term must be a tuple/list "
                "(q1, q2, gamma)."
            )
    return terms


def radial_Coulomb_femdvr_two_grid(r1_grid, r2_grid, n_L, charge_gamma_terms):
    r"""
    Build generalized two-grid FEMDVR radial Coulomb matrices.

    The returned matrix represents the radial multipoles of

    .. math::
        \sum_k \frac{q_{1,k} q_{2,k}}
        {|\mathbf r_1 - \gamma_k \mathbf r_2|}.

    Source quadrature weights are included in the columns, matching the
    convention of :func:`radial_Coulomb`.

    Parameters
    ----------
    r1_grid
        Target FEMDVR grid.
    r2_grid
        Source FEMDVR grid.
    n_L : int
        Number of angular momenta to compute.
    charge_gamma_terms
        Iterable of ``(q1, q2, gamma)`` tuples.

    Returns
    -------
    ndarray
        Array with shape ``(n_L, n_r1_inner, n_r2_inner)``.
    """
    r1 = r1_grid.r[1:-1]
    r2 = r2_grid.r[1:-1]
    n_r1 = len(r1)
    n_r2 = len(r2)
    W = np.zeros((n_L, n_r1, n_r2))

    for q1, q2, gamma in _as_charge_gamma_terms(charge_gamma_terms):
        u_L = solve_radial_Poisson_femdvr_two_grid(
            r1_grid, r2_grid, n_L, gamma
        )
        prefactor = q1 * q2
        for L in range(n_L):
            W[L] += (
                prefactor
                * (4 * np.pi / (2 * L + 1))
                * u_L[L]
                / r1[:, np.newaxis]
            )

    return W
