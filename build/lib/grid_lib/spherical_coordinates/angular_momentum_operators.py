import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
)
from opt_einsum import contract


def Aplus(l, m):
    return np.sqrt(l * (l + 1) - m * (m + 1))


def Aminus(l, m):
    return np.sqrt(l * (l + 1) - m * (m - 1))


class AngularMomentumOperators:
    def __init__(self, l_max, m_max):

        self.l_max = l_max
        self.m_max = m_max
        self.n_lm = number_of_lm_states(l_max, m_max)

        self.lp = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| l+ |l2,m2>
        self.lm = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| l- |l2,m2>
        self.lz = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| lz |l2,m2>
        self.lz_sq = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| lz^2 |l2,m2>
        self.l_sq = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| l^2 |l2,m2>
        self.lm_lp = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| l- l+ |l2,m2>
        self.lp_lm = np.zeros((self.n_lm, self.n_lm))  # <l1,m1| l+ l- |l2,m2>

        for m1 in range(-self.m_max, self.m_max + 1):
            for l1 in range(abs(m1), l_max + 1):
                i1 = LM_to_I(l1, m1, l_max, m_max)
                for m2 in range(-self.m_max, self.m_max + 1):
                    for l2 in range(abs(m2), l_max + 1):
                        i2 = LM_to_I(l2, m2, l_max, m_max)
                        if l1 == l2 and m1 == m2 + 1:
                            self.lp[i1, i2] = Aplus(l2, m2)
                        if l1 == l2 and m1 == m2 - 1:
                            self.lm[i1, i2] = Aminus(l2, m2)
                        if l1 == l2 and m1 == m2:
                            self.lz[i1, i2] = m1
                            self.lz_sq[i1, i2] = m1**2
                            self.l_sq[i1, i2] = l1 * (l1 + 1)
                            self.lm_lp[i1, i2] = Aminus(l2, m2 + 1) * Aplus(
                                l2, m2
                            )
                            self.lp_lm[i1, i2] = Aplus(l2, m2 - 1) * Aminus(
                                l2, m2
                            )

        self.lx = (1.0 / 2.0) * (self.lp + self.lm)
        self.ly = (1.0 / 2.0j) * (self.lp - self.lm)


class OrbitalAngularMomentum:
    def __init__(self, l_max, m_max, weights):

        self.weights = weights
        self.ame = AngularMomentumOperators(l_max, m_max)

    def lx(self, A):
        return contract("IJ, paJ -> paI", self.ame.lx, A)
    
    def ly(self, A):
        return contract("IJ, paJ -> paI", self.ame.ly, A)
    
    def lz(self, A):
        return contract("IJ, paJ -> paI", self.ame.lz, A)

    def p_lsq_q(self, A):
        l_sq = contract(
            "k,pkI, qkJ, IJ->pq", self.weights, A.conj(), A, self.ame.l_sq
        )
        return l_sq

    def p_lx_q(self, A):
        l_x = contract(
            "k,pkI, qkJ, IJ->pq", self.weights, A.conj(), A, self.ame.lx
        )
        return l_x

    def p_ly_q(self, A):
        l_y = contract(
            "k,pkI, qkJ, IJ->pq", self.weights, A.conj(), A, self.ame.ly
        )
        return l_y

    def p_lz_q(self, A):
        l_z = contract(
            "k,pkI, qkJ, IJ->pq", self.weights, A.conj(), A, self.ame.lz
        )
        return l_z

    def compute_l_vector(self, A):
        lx_e = self.p_lx_q(A)
        ly_e = self.p_ly_q(A)
        lz_e = self.p_lz_q(A)
        return np.array([lx_e, ly_e, lz_e])



