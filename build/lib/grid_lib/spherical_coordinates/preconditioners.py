import numpy as np


class Preconditioner:
    def __init__(self, angular_matrix_elements, radial_matrix_elements):
        pass

    def __call__(psi):
        pass


class MPsi:
    def __init__(self, angular_matrix_elements, radial_matrix_elements, dt):
        nr = radial_matrix_elements.nr
        T_D2 = -(1 / 2) * radial_matrix_elements.D2

        Identity = np.complex128(np.eye(nr))
        self.M = np.linalg.inv(Identity + 1j * dt / 2 * T_D2)

        self.n_lm = angular_matrix_elements.n_lm
        self.nr = radial_matrix_elements.nr

    def __call__(self, psi):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        for I in range(self.n_lm):
            psi_new[I] = np.dot(self.M, psi[I])

        return psi_new.ravel()


class M2Psi:
    def __init__(self, angular_matrix_elements, radial_matrix_elements, dt):
        T_D2 = -(1 / 2) * radial_matrix_elements.D2
        r = radial_matrix_elements.r

        self.n_lm = angular_matrix_elements.n_lm
        self.nr = radial_matrix_elements.nr

        Identity = np.complex128(np.eye(self.nr))
        self.M_l = np.zeros((self.n_lm, self.nr, self.nr), dtype=np.complex128)

        for I in range(self.n_lm):
            l = angular_matrix_elements.lm_I[I][0]
            T_l = T_D2 + np.diag(l * (l + 1) / (2 * r**2))
            self.M_l[I] = np.linalg.inv(Identity + 1j * dt / 2 * T_l)

    def __call__(self, psi):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        for I in range(self.n_lm):
            psi_new[I] = np.dot(self.M_l[I], psi[I])

        return psi_new.ravel()
