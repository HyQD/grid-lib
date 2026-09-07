import numpy as np
from matplotlib import pyplot as plt
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR

def test_particle_in_box():
    print()
    print(f"** Test 1D particle-in-box **")

    def test_particle_in_box(L):
        print(f"* Box length L = {L}")
        N = 40
        x_min = 0
        x_max = L

        GLL = GaussLegendreLobatto(
            N, Linear_map(x_min, x_max), symmetrize=False
        )
        x = GLL.r[1:-1]
        w = GLL.weights[1:-1]
        x_dot = GLL.r_dot[1:-1]

        D1 = GLL.D1
        D2 = np.dot(D1, D1)[
            1:-1, 1:-1
        ]  # Second derivative matrix with Dirichlet boundary conditions u(x_min) = u(x_max) = 0
        T = -0.5 * D2

        eps, U = np.linalg.eig(T)
        idx = np.argsort(eps)
        eps = eps[idx]
        U = U[:, idx]

        N_max = 10
        for k in range(1, N_max):
            eps_k_approx = eps[k - 1]
            psi_k_approx = U[:, k - 1]
            norm_psi_k_approx = np.dot(w, psi_k_approx**2)
            psi_k_approx /= np.sqrt(norm_psi_k_approx)

            eps_k_exact = k**2 * np.pi**2 / (2 * L**2)
            psi_k_exact = np.sqrt(2 / L) * np.sin(k * np.pi * x / L)
            norm_psi_k_exact = np.dot(w, psi_k_exact**2)
            np.testing.assert_allclose(
                eps_k_approx, eps_k_exact, rtol=1e-14, atol=1e-12
            )
            np.testing.assert_allclose(
                np.abs(psi_k_approx) ** 2,
                np.abs(psi_k_exact) ** 2,
                rtol=0.0,
                atol=1e-12,
            )

    test_particle_in_box(1)
    test_particle_in_box(1.5)
    test_particle_in_box(2)
    test_particle_in_box(4)


def test_harmonic_oscillator():
    from scipy.special import eval_hermite as Hermite, factorial as fac

    print()
    print(f"** Test 1D harmonic oscillator **")

    def test_harmonic_oscillator(omega):
        print(f"* Oscillator frequency omega = {omega}")
        N = 80
        a = 0.5 * omega

        """
        np.exp(-a*x**2) < 10**(-16) for |x| > sqrt(-ln(10^(-16))/a), ln(10^-16) \approx -36.84
        """
        x_min = -np.sqrt(37 / a)
        x_max = np.sqrt(37 / a)

        GLL = GaussLegendreLobatto(
            N, Linear_map(x_min, x_max), symmetrize=False
        )
        x = GLL.r[1:-1]
        w = GLL.weights[1:-1]
        x_dot = GLL.r_dot[1:-1]
        D1 = GLL.D1
        D2 = np.dot(D1, D1)[1:-1, 1:-1]
        T = -0.5 * D2
        V = np.diag(0.5 * omega**2 * x**2)
        H = T + V
        eps, U = np.linalg.eig(H)
        idx = np.argsort(eps)
        eps = eps[idx]
        U = U[:, idx]

        for k in range(0, 5):

            eps_k_approx = eps[k]
            eps_k_exact = omega * (k + 0.5)

            psi_k_approx = U[:, k]
            norm_psi_k_approx = np.dot(w, psi_k_approx**2)
            psi_k_approx /= np.sqrt(norm_psi_k_approx)

            psi_k_exact = (
                1
                / np.sqrt(2**k * fac(k))
                * (omega / np.pi) ** (0.25)
                * Hermite(k, np.sqrt(omega) * x)
                * np.exp(-0.5 * omega * x**2)
            )
            np.testing.assert_allclose(
                eps_k_approx, eps_k_exact, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.abs(psi_k_approx) ** 2,
                np.abs(psi_k_exact) ** 2,
                rtol=0.0,
                atol=1e-12,
            )

    test_harmonic_oscillator(1)
    test_harmonic_oscillator(2)
    test_harmonic_oscillator(4)
    test_harmonic_oscillator(0.5)
    test_harmonic_oscillator(0.75)


def test_hydrogenic_spherical_coordinates():

    from scipy.special import eval_genlaguerre as Laguerre, factorial as fac

    def u_nl_exact(r, n, l, Z):
        R_nl = (
            (2 * Z / n) ** (3 / 2)
            * np.sqrt(fac(n - l - 1) / (2 * n * fac(n + l)))
            * (2 * Z * r / n) ** l
            * Laguerre(n - l - 1, 2 * l + 1, 2 * Z * r / n)
            * np.exp(-Z * r / n)
        )
        u_nl = r * R_nl
        return u_nl

    print()
    print(f"** Test hydrogenic systems in spherical coordinates **")

    def test_hydrogenic_sphc(Z):
        print(f"* Nuclear charge Z = {Z}")
        N = 200
        r_min = 0
        r_max = 140
        GLL = GaussLegendreLobatto(
            N, Linear_map(r_min, r_max), symmetrize=False
        )
        r = GLL.r[1:-1]
        w = GLL.weights[1:-1]
        r_dot = GLL.r_dot[1:-1]
        D1 = GLL.D1
        D2 = np.dot(D1, D1)[1:-1, 1:-1]
        for l in range(0, 3):

            eps_exact = -(Z**2) / (2 * np.arange(l + 1, l + 4) ** 2)

            Tl = -0.5 * D2 + np.diag(l * (l + 1) / (2 * r**2))
            V = np.diag(-Z / r)
            Hl = Tl + V
            eps, U = np.linalg.eig(Hl)
            idx = np.argsort(eps)
            eps = eps[idx]
            U = U[:, idx]

            n1 = l + 1
            n2 = l + 2
            n3 = l + 3
            u_n1_l = u_nl_exact(r, n1, l, Z)
            u_n2_l = u_nl_exact(r, n2, l, Z)
            u_n3_l = u_nl_exact(r, n3, l, Z)

            u_n1_l_approx = U[:, 0]
            norm_u_n1_l_approx = np.dot(w, u_n1_l_approx**2)
            u_n1_l_approx /= np.sqrt(norm_u_n1_l_approx)

            u_n2_l_approx = U[:, 1]
            norm_u_n2_l_approx = np.dot(w, u_n2_l_approx**2)
            u_n2_l_approx /= np.sqrt(norm_u_n2_l_approx)

            u_n3_l_approx = U[:, 2]
            norm_u_n3_l_approx = np.dot(w, u_n3_l_approx**2)
            u_n3_l_approx /= np.sqrt(norm_u_n3_l_approx)

            np.testing.assert_allclose(
                eps[0:3], eps_exact, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.abs(u_n1_l) ** 2,
                np.abs(u_n1_l_approx) ** 2,
                rtol=0.0,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                np.abs(u_n2_l) ** 2,
                np.abs(u_n2_l_approx) ** 2,
                rtol=0.0,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                np.abs(u_n3_l) ** 2,
                np.abs(u_n3_l_approx) ** 2,
                rtol=0.0,
                atol=1e-10,
            )

    test_hydrogenic_sphc(1)
    test_hydrogenic_sphc(2)
    test_hydrogenic_sphc(4)
    test_hydrogenic_sphc(10)

def test_ho_fem():

    a = -10
    b = 10
    n_elem = 3
    points_per_elem = 31

    nodes_list = [np.linspace(a, b, n_elem + 1), np.array([-10, -2, 2, 10])]
    n_points_list = [
        np.ones((n_elem,), dtype=int) * points_per_elem,
        np.array([21, 21, 21]),
    ]

    for nodes, n_points in zip(nodes_list, n_points_list):

        femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

        # Get nodes and weights and differentiation matrix.
        r = femdvr.r
        r_dot = femdvr.r_dot
        w = femdvr.weights
        D = femdvr.D1
        D2 = femdvr.D2
        ei = femdvr.edge_indices

        H_full = -0.5 * D2 + np.diag(0.5 * r**2)
        H = H_full[1:-1, 1:-1]

        E, U = np.linalg.eig(H)
        i = np.argsort(E)
        E = E[i]
        U = U[:, i]

        assert np.allclose(E[:5], [0.5, 1.5, 2.5, 3.5, 4.5], rtol=1e-12)


# if __name__ == "__main__":
#     # test_harmonic_oscillator()
#     test_hydrogenic_spherical_coordinates()
