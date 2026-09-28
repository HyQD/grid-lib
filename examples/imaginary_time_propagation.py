import numpy as np
import time
from matplotlib import pyplot as plt

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map,
)

from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
)
from grid_lib.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)
from grid_lib.spherical_coordinates.utils import (
    Counter,
    quadrature,
)
from grid_lib.spherical_coordinates.rhs import H0_B_Psi, HtPsi
from grid_lib.spherical_coordinates.preconditioners import M2Psi
import tqdm
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, cg, gmres, bicgstab

N = 100
r_max = 20
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
nr = len(r)
D1 = radial_matrix_elements.D1
T_D2 = -(1 / 2) * radial_matrix_elements.D2

Z = 1
B = 1
l_max = 20

m_list = [0, -1, -2]

for m in m_list:
    angular_matrix_elements = AngularMatrixElements_l(
        arr_to_calc=["H_Bz_Omega"], l_max=l_max, m=m
    )
    n_lm = angular_matrix_elements.n_lm

    tfinal = 50.0
    dt = 0.1
    num_steps = int(tfinal / dt) + 1
    time_points = np.zeros(num_steps)
    expec_H = np.zeros(num_steps, dtype=np.complex128)

    # Use the lowest eigenstate of the B=0 Hamiltonian as initial guess for imaginary time propagation
    l = abs(m)
    H0 = T_D2 + np.diag(potential) + np.diag(l * (l + 1) / (2 * r**2))
    eps, psi = np.linalg.eigh(H0)

    H0_psi = H0_B_Psi(angular_matrix_elements, radial_matrix_elements, potential, B0=B)
    rhs = HtPsi(angular_matrix_elements, radial_matrix_elements, H0_psi, [])

    psi_t = np.zeros((l_max + 1, nr), dtype=np.complex128)
    psi_t[l] = np.complex128(psi[:, 0])
    psi_t[l] /= np.sqrt(quadrature(weights, np.abs(psi_t[l]) ** 2))

    H_psi_t = rhs(psi_t, 0)
    H_psi_t = H_psi_t.reshape((l_max + 1, nr))

    for l in range(l_max + 1):
        expec_H[0] += quadrature(weights, psi_t[l] * H_psi_t[l])

    preconditioner = M2Psi(angular_matrix_elements, radial_matrix_elements, -1j * dt)
    M_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=preconditioner)

    for i in tqdm.tqdm(range(num_steps - 1)):

        time_points[i + 1] = (i + 1) * dt
        ti = time_points[i] + dt / 2

        Ap_lambda = lambda psi, ti=ti: psi.ravel() + dt / 2 * rhs(psi, ti)
        Ap_linear = LinearOperator(
            (nr * (l_max + 1), nr * (l_max + 1)), matvec=Ap_lambda
        )

        z = psi_t.ravel() - dt / 2 * rhs(psi_t.ravel(), ti)

        local_counter = Counter()
        psi_t, info = bicgstab(
            Ap_linear,
            z,
            M=M_linear,
            x0=psi_t.ravel(),
            tol=1e-12,
            callback=local_counter,
        )

        psi_t = psi_t.reshape((l_max + 1, nr))

        norm = 0
        for I in range(l_max + 1):
            norm += quadrature(weights, np.abs(psi_t[I]) ** 2)

        psi_t /= np.sqrt(norm)

        H_psi_t = rhs(psi_t, 0)
        H_psi_t = H_psi_t.reshape((l_max + 1, nr))

        for l in range(l_max + 1):
            expec_H[i + 1] += quadrature(weights, psi_t[l] * H_psi_t[l])

    print(f"<Psi|H|Psi>   : {expec_H[-1].real}")
    print(f"Binding energy: {0.5 * B * (abs(m) + m + 1) - expec_H[-1].real}")

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(time_points, expec_H.real)
    # plt.subplot(212)
    # plt.semilogy(np.abs(expec_H[1:].real - expec_H[0:-1].real))
    # plt.show()
