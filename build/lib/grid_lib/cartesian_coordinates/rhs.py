import numpy as np
from grid_lib.cartesian_coordinates.sinc_dvr import compute_mean_field


class FockOperator:
    def __init__(self, H0, w12, x, e_field):
        self.H0 = H0
        self.w12 = w12
        self.x = x
        self.e_field = e_field
        self.w12 = w12

    def __call__(self, psi, t):

        H0_phi = np.dot(self.H0, psi)
        et_phi = self.e_field(t) * np.einsum("i,ij->ij", self.x, psi)

        Vdir_phi = Vdirect_phi(psi, psi, self.w12)
        Vex_phi = Vexchange_phi(psi, psi, self.w12)

        F_phi = H0_phi - et_phi + 2 * Vdir_phi - Vex_phi
        return -1j * F_phi


def Vdirect_phi(psi, phi, w12):
    """
    Apply the direct potential to the orbitals phi.
        \hat{V}_dir(psi) * phi_i = (sum_j (\int |\psi_j(r')|^2 w12(r,r') dr')) * phi_i
                                = direct * phi_i
        direct = sum_j (\int |\psi_j(r')|^2 w12(r,r') dr')

    Args:
        psi (np.ndarray): The orbitals that \hat{V}_dir(psi) depend upon.
        phi (np.ndarray): The orbitals to act with \hat{V}_dir(psi) upon.
    Returns:
        np.ndarray: The orbitals after the direct potential has been applied.
    """
    direct = np.zeros(phi.shape[0], dtype=phi.dtype)
    for i in range(phi.shape[1]):
        direct += compute_mean_field(w12, psi[:, i], psi[:, i])

    Vd_phi = np.zeros(phi.shape, dtype=phi.dtype)

    for i in range(phi.shape[1]):
        Vd_phi[:, i] = direct * phi[:, i]

    return Vd_phi


def Vexchange_phi(psi, phi, w12):
    """
    Apply the exchange potential to the orbitals phi.

    \hat{V}_ex(psi)*phi_i = \sum_j (\int psi_j^*(r') w12(r,r') phi_i(r') dr') * psi_j(r)

    Args:
        psi (np.ndarray): The orbitals that \hat{V}_ex(psi) depend upon.
        phi (np.ndarray): The orbitals to act with \hat{V}_ex(psi) upon.
    Returns:
        np.ndarray: The orbitals after the exchange potential has been applied.
    """
    Vex_phi = np.zeros(phi.shape, dtype=phi.dtype)

    for i in range(psi.shape[1]):
        for j in range(phi.shape[1]):
            Vex_phi[:, i] += (
                compute_mean_field(w12, psi[:, j], phi[:, i]) * psi[:, j]
            )

    return Vex_phi


# def rhs_Wc(psi, t, direct):
#     """
#     Evaluate the right-hand side
#         (H0+V(t)+W(psi))*psi = (H0+V(t)+int w(x,x')|psi(x')|^2 dx')*psi
#     with a constant mean-field.
#     A constant mean-field means that the direct potential is held fixed over some time interval.
#     """
#     rhs = np.dot(H, psi)
#     rhs += e_field(t) * x * psi
#     rhs += direct * psi
#     return -1j * rhs


# def rhs_VWc(psi, t, direct):
#     """
#     Evaluate V(t)*psi+W_{direct}(psi)*psi where the direct potential is fixed/provided.
#     """
#     rhs = e_field(t) * x * psi
#     rhs += direct * psi
#     return -1j * rhs
