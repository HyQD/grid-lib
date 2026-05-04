import numpy as np
from matplotlib import pyplot as plt
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)


def test_integrate_polynomials():

    """
    Gauss-Lobatto quadrature is accurate for polynomials up to order 2n-1,
    where n is the number of grid/integration points. We have chosen the convention that
    x0 = -1, xN = 1, and x_i are the (N-1) roots of P_N'(x) = 0 where P_N(x) is the N-th order
    Legendre polynomial. Thus, we have N+1 grid/integration points, and the quadrature is accurate
    for polynomials up to order 2N-1.
    """

    N = 6
    x_min = -1.0
    x_max = 1.0
    L = x_max - x_min  # Length of the simulation box
    GLL = GaussLegendreLobatto(N, Linear_map(x_min, x_max), symmetrize=False)
    w = GLL.weights

    x = GLL.r

    # Test the integral of even polynomials up to order 2N-1 on the interval [-1, 1]
    for n in range(2, 2 * N - 1, 2):
        integral_xn = np.dot(w, x**n)
        np.testing.assert_allclose(
            integral_xn, 2 / (n + 1), rtol=0.0, atol=1e-15
        )


def test_integrate_polynomials_mapped_interval():

    N = 6

    def test_integrate_polynomials_mapped_interval(a, b):

        GLL = GaussLegendreLobatto(N, Linear_map(a, b), symmetrize=False)
        w = GLL.weights
        r_dot = GLL.r_dot
        a_tol = 1e-12
        for n in range(1, 2 * N):
            integral_xn = np.dot(w, GLL.r**n)
            np.testing.assert_allclose(
                np.array([integral_xn]),
                np.array([(b ** (n + 1) - a ** (n + 1)) / (n + 1)]),
                rtol=0.0,
                atol=a_tol,
            )

    test_integrate_polynomials_mapped_interval(0, 1)
    test_integrate_polynomials_mapped_interval(0, 2)
    test_integrate_polynomials_mapped_interval(1, 2)
    test_integrate_polynomials_mapped_interval(-0.5, 1)
    test_integrate_polynomials_mapped_interval(-2, 1)


def test_integrate_xk_exp_minus_a_x2():
    """
    int_0^\infty x^k exp(-a*x^2) dx = 0.5*a^(-(k+1)/2)*Gamma((k+1)/2), where Gamma is the gamma function.
    """
    from scipy.special import gamma

    def test_integrate_xk_exp_minus_a_x2(a):

        assert a > 0

        N = 60

        x_min = 0
        x_max = np.sqrt(37 / a) + 10

        GLL = GaussLegendreLobatto(
            N, Linear_map(x_min, x_max), symmetrize=False
        )

        x = GLL.r
        x_dot = GLL.r_dot
        w = GLL.weights

        for k in range(6):
            f = x**k * np.exp(-a * x**2)
            integral_quadrature = np.dot(w, f)
            integral_exact = 0.5 * a ** (-(k + 1) / 2) * gamma((k + 1) / 2)
            np.testing.assert_allclose(
                integral_quadrature, integral_exact, rtol=0.0, atol=1e-10
            )

    test_integrate_xk_exp_minus_a_x2(1.0)
    test_integrate_xk_exp_minus_a_x2(0.5)
    test_integrate_xk_exp_minus_a_x2(0.1)
    test_integrate_xk_exp_minus_a_x2(1.5)
    test_integrate_xk_exp_minus_a_x2(2.0)


# if __name__ == "__main__":
#     #test_integrate_polynomials_mapped_interval()
#     test_integrate_xk_exp_minus_a_x2()