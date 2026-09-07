import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
)
from opt_einsum import contract


def a_lm(l, m):
    return np.sqrt((l + m + 1) * (l - m + 1) / ((2 * l + 1) * (2 * l + 3)))


def b_lm(l, m):
    return np.sqrt((l + m + 1) * (l + m + 2) / ((2 * l + 1) * (2 * l + 3)))


class MomentumOperators:
    def __init__(self, l_max, m_max, r, weights, D1):

        self.l_max = l_max
        self.m_max = m_max
        self.n_lm = number_of_lm_states(l_max, m_max)
        self.r = r
        self.weights = weights
        self.D1 = D1
        self.D1_tilde = D1 - np.diag(1 / r)  # <alpha|d/dr - 1/r|beta>
        self.r_inv = 1 / r  # <alpha|1/r|beta>

        self.px_lm_1 = np.zeros((self.n_lm, self.n_lm))
        self.px_lm_2 = np.zeros((self.n_lm, self.n_lm))
        self.py_lm_1 = np.zeros((self.n_lm, self.n_lm))
        self.py_lm_2 = np.zeros((self.n_lm, self.n_lm))
        self.pz_lm_1 = np.zeros((self.n_lm, self.n_lm))
        self.pz_lm_2 = np.zeros((self.n_lm, self.n_lm))

        for m2 in range(-m_max, m_max + 1):
            for l2 in range(abs(m2), l_max + 1):
                I2 = LM_to_I(l2, m2, l_max, m_max)
                if l2 != l_max:
                    # Populate +:
                    I1 = I2 + 1
                    self.pz_lm_1[I1, I2] = a_lm(l2, m2)
                    self.pz_lm_2[I1, I2] = l2 * a_lm(l2, m2)
                    if m2 != m_max:
                        # Populate ++:
                        I1 = LM_to_I(l2 + 1, m2 + 1, l_max, m_max)
                        self.px_lm_1[I1, I2] = b_lm(l2, m2)
                        self.px_lm_2[I1, I2] = l2 * b_lm(l2, m2)
                        self.py_lm_1[I1, I2] = b_lm(l2, m2)
                        self.py_lm_2[I1, I2] = l2 * b_lm(l2, m2)
                    if m2 != -m_max:
                        # Populate +-:
                        I1 = LM_to_I(l2 + 1, m2 - 1, l_max, m_max)
                        self.px_lm_1[I1, I2] = -b_lm(l2, -m2)
                        self.px_lm_2[I1, I2] = -l2 * b_lm(l2, -m2)
                        self.py_lm_1[I1, I2] = b_lm(l2, -m2)
                        self.py_lm_2[I1, I2] = l2 * b_lm(l2, -m2)
                if l2 != abs(m2):
                    # Populate -:
                    I1 = I2 - 1
                    self.pz_lm_1[I1, I2] = a_lm(l2 - 1, m2)
                    self.pz_lm_2[I1, I2] = -(l2 + 1) * a_lm(l2 - 1, m2)
                if (l2 != m2) and (m2 != m_max):
                    # Populate -+:
                    I1 = LM_to_I(l2 - 1, m2 + 1, l_max, m_max)
                    self.px_lm_1[I1, I2] = -b_lm(l2 - 1, -m2 - 1)
                    self.px_lm_2[I1, I2] = (l2 + 1) * b_lm(l2 - 1, -m2 - 1)
                    self.py_lm_1[I1, I2] = -b_lm(l2 - 1, -m2 - 1)
                    self.py_lm_2[I1, I2] = (l2 + 1) * b_lm(l2 - 1, -m2 - 1)
                if (l2 != -m2) and (m2 != -m_max):
                    # Populate --:
                    I1 = LM_to_I(l2 - 1, m2 - 1, l_max, m_max)
                    self.px_lm_1[I1, I2] = b_lm(l2 - 1, m2 - 1)
                    self.px_lm_2[I1, I2] = -(l2 + 1) * b_lm(l2 - 1, m2 - 1)
                    self.py_lm_1[I1, I2] = -b_lm(l2 - 1, m2 - 1)
                    self.py_lm_2[I1, I2] = (l2 + 1) * b_lm(l2 - 1, m2 - 1)

    def dx(self, A):

        dA_dr = contract("ab, pbI-> paI", self.D1_tilde, A)
        dx_A = contract("IJ, paJ -> paI", self.px_lm_1, dA_dr)

        A_r = contract("a, paI->paI", self.r_inv, A)
        dx_A -= contract("IJ, paJ -> paI", self.px_lm_2, A_r)

        return -dx_A / 2

    def px(self, A):
        return -1j * self.dx(A)

    def p_dx_q(self, A):
        dx_A = self.dx(A)
        p_dx_q = contract("a, paI, qaI -> pq", self.weights, A.conj(), dx_A)
        return p_dx_q

    def p_px_q(self, A):
        p_dx_q = self.p_dx_q(A)
        return -1j * p_dx_q

    def dy(self, A):

        dA_dr = contract("ab, pbI-> paI", self.D1_tilde, A)
        dy_A = contract("IJ, paJ -> paI", self.py_lm_1, dA_dr)

        A_r = contract("a, paI->paI", self.r_inv, A)
        dy_A -= contract("IJ, paJ -> paI", self.py_lm_2, A_r)

        return 1j * dy_A / 2

    def py(self, A):
        return -1j * self.dy(A)

    def p_dy_q(self, A):
        dy_A = self.dy(A)
        p_dy_q = contract("a, paI, qaI -> pq", self.weights, A.conj(), dy_A)
        return p_dy_q

    def p_py_q(self, A):
        p_dy_q = self.p_dy_q(A)
        return -1j * p_dy_q

    def dz(self, A):
        """

        Compute the action of the z-component of the momentum operator on an orbital array A, i.e.,

            <alpha,I|p_z|psi_p> = sum_J <alpha,I|dz|beta,J> A_{p, beta, J}, |alpha,I> = r^{-1} chi_alpha(r) Y_I(theta, phi)

        """

        dA_dr = contract("ab, pbI-> paI", self.D1_tilde, A)
        dz_A = contract("IJ, paJ -> paI", self.pz_lm_1, dA_dr)

        A_r = contract("a, paI->paI", self.r_inv, A)
        dz_A -= contract("IJ, paJ -> paI", self.pz_lm_2, A_r)

        return dz_A

    def pz(self, A):
        return -1j * self.dz(A)

    def p_dz_q(self, A):
        dz_A = self.dz(A)
        p_dz_q = contract("a, paI, qaI -> pq", self.weights, A.conj(), dz_A)
        return p_dz_q

    def p_pz_q(self, A):
        p_dz_q = self.p_dz_q(A)
        return -1j * p_dz_q

    def get_momentum_vector(self, A):
        return np.array([self.px(A), self.py(A), self.pz(A)])

    def get_gradient_vector(self, A):
        return np.array([self.dx(A), self.dy(A), self.dz(A)])
