import numpy as np
from scipy.linalg import solve


def solve_radial_Poisson_dvr(GLL, n_L):
    r"""
    The integral 
    .. math::
        \tilde{v}_{L}(r; \chi_\alpha, \chi_\beta) \equiv r \int_{0}^{r_{\text{max}}} \chi_{\alpha}(r_2) \frac{r_<^L}{r_>^{L+1}} \chi_\beta(r_2)  dr_2
    
    where 
    .. math::
        r_< = \min(r, r_2), \quad r_> = \max(r, r_2),
    
    and \chi_\alpha(r) are DVR basis functions
        
    can be computed/evaluated by solving the corresponding radial Poisson equation
    .. math::
        \left( \frac{\partial^2}{\partial r^2} - \frac{L(L+1)}{r^2} \right) \tilde{v}_{L}(r; \chi_\alpha, \chi_\beta) = -\frac{(2L+1)}{r} \chi_\alpha(r) \chi_\beta(r)
    
    subject to the boundary conditions 
    .. math::
        \tilde{v}_L(0; \chi_\alpha, \chi_\beta) &= 0, \\
        \tilde{v}_L(r_{\text{max}}; \chi_\alpha, \chi_\beta) &= \left( \frac{r_\alpha}{r_\text{max}} \right)^L \dot{r}_\alpha w_\alpha \delta_{\alpha, \beta}.
    
    At a/the grid point/points r=r_\gamma, 
    .. math::
        \tilde{v}_{L}(r_\gamma; \chi_\alpha, \chi_\beta) \neq 0 \iff \chi_\alpha = \chi_\beta,
    which we can formulate as 
    .. math::
        \tilde{v}_{L}(r_\gamma; \chi_\alpha, \chi_\beta) = \tilde{v}_{L}(r_\gamma; \chi_\beta) \delta_{\alpha, \beta},
    and we represent the solution/function v_L(r_\gamma; \chi_beta) as a/the matrix (for each L) as 
    .. math::
        \tilde{V}_{L,\gamma,\beta} \equiv \tilde{v}_{L}(r_\gamma; \chi_\beta).

    The solution, for each L, at the (inner) grid points is given by 
    .. math::
        \tilde{V}_L = (D^{(2)}_L)^{-1} B_L,
    
    where (with the boundary conditions incorporated)
    .. math::
        (B_L)_{\delta,\beta} = \left(\frac{(2L+1)}{r_\beta} \delta_{\delta, \beta} + D^{(2)}_{L,\delta, N} \tilde{V}_{L,N,\beta} \right),
    and D^{(2)}_L is the matrix representation of the operator
    .. math::
        \nabla_L^2 \equiv \frac{\partial^2}{\partial r^2} - \frac{L(L+1)}{r^2}
    in the DVR (Legendre-Lobatto) basis.

    Parameters
    ----------
    GLL : GaussLegendreLobatto
        An instance of the Gauss-Legendre-Lobatto grid object/class.
    n_L : int
        The number of angular momenta L to compute the radial Poisson equation for.
    """
    r_max = GLL.r[-1]

    # We solve the radial Poisson equation on the inner grid points
    r = GLL.r[1:-1]
    w_r = GLL.weights[1:-1]
    # r_dot = GLL.r_dot[1:-1]
    D2 = GLL.D2[1:-1, 1:-1]

    n_r = len(r)
    tilde_vL = np.zeros((n_L, n_r, n_r))

    for L in range(0, n_L):

        D2_L = D2 - np.diag(L * (L + 1) / r**2)

        D2_L_inv = np.linalg.inv(D2_L)
        B_L = np.diag(-(2 * L + 1) / r)
        tilde_vL_inhom = np.dot(D2_L_inv, B_L)

        tilde_vL_hom = np.zeros((n_r, n_r))
        for a in range(n_r):
            tilde_vL_hom[:, a] = (
                r[a] ** L * w_r[a] / r_max ** (2 * L + 1)
            ) * r ** (L + 1)

        tilde_vL[L] = tilde_vL_inhom + tilde_vL_hom

    return tilde_vL


def _signed_gamma_parity(L, gamma):
    if gamma < 0.0 and L % 2:
        return -1.0
    return 1.0


def _outer_boundary_u(r_boundary, source_radius, source_weights, L, parity):
    r_less = np.minimum(r_boundary, source_radius)
    r_greater = np.maximum(r_boundary, source_radius)
    return parity * r_boundary * r_less**L / r_greater ** (L + 1) * source_weights


def solve_radial_Poisson_femdvr_two_grid(r1_grid, r2_grid, n_L, gamma):
    r"""
    Solve the two-grid FEMDVR radial Poisson problem.

    This generalizes :func:`solve_radial_Poisson_dvr` to independent target
    and source FEMDVR grids for the kernel

    .. math::
        \frac{1}{|\mathbf r_1 - \gamma \mathbf r_2|}.

    The returned array contains the ``u_L(r_1, r_2)`` matrices, where
    ``u_L = r_1 V_L`` and source quadrature weights are included in the
    columns, matching the existing same-grid convention.

    Parameters
    ----------
    r1_grid
        Target FEMDVR grid.
    r2_grid
        Source FEMDVR grid.
    n_L : int
        Number of angular momenta to compute.
    gamma : float
        Scale factor multiplying the source coordinate.  Negative values
        include the spherical-harmonic parity factor ``(-1)**L``.

    Returns
    -------
    ndarray
        Array with shape ``(n_L, n_r1_inner, n_r2_inner)``.
    """
    r1 = r1_grid.r[1:-1]
    r2 = r2_grid.r[1:-1]
    w2 = r2_grid.weights[1:-1]

    u_L = np.zeros((n_L, len(r1), len(r2)))

    if gamma == 0.0:
        if n_L > 0:
            u_L[0] = w2[np.newaxis, :]
        return u_L

    gamma_abs = abs(gamma)
    source_radius = gamma_abs * r2
    r1_boundary = r1_grid.r[-1]
    D2 = r1_grid.D2[1:-1, 1:-1]
    delta = r1_grid.delta_matrix(source_radius, include_boundaries=False)

    for L in range(n_L):
        parity = _signed_gamma_parity(L, gamma)
        D2_L = D2 - np.diag(L * (L + 1) / r1**2)

        source_strength = parity * w2 / source_radius
        B_L = -(2 * L + 1) * delta * source_strength[np.newaxis, :]
        u_inhomogeneous = solve(D2_L, B_L, assume_a="gen")

        boundary = _outer_boundary_u(
            r1_boundary, source_radius, w2, L, parity
        )
        u_homogeneous = (
            (r1 / r1_boundary)[:, np.newaxis] ** (L + 1)
        ) * boundary[np.newaxis, :]

        u_L[L] = u_inhomogeneous + u_homogeneous

    return u_L
