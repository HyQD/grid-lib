import numpy as np


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