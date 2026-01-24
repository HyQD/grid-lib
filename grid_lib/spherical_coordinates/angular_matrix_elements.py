import numpy as np
import time
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre, sph_harm_y, spherical_jn
from pathlib import Path


class AngularMatrixElements:
    def __init__(self, l_max=5, N=101):
        current_file_location = Path(__file__).parent
        coord = np.loadtxt(
            current_file_location / ("Lebedev/lebedev_%03d.txt" % N)
        )

        self.N = N

        self.theta = coord[:, 1] * np.pi / 180
        self.phi = coord[:, 0] * np.pi / 180 + np.pi
        self.weights = coord[:, 2]

        self.sin_th = np.sin(self.theta)
        self.cos_th = np.cos(self.theta)
        self.sin_ph = np.sin(self.phi)
        self.cos_ph = np.cos(self.phi)
        self.exp_m1j_p = np.exp(-1j * self.phi)
        self.exp_p1j_p = np.exp(1j * self.phi)

        self.arr = {}

        self.n_l = l_max + 1

    def kron_delta(self, i, j):
        return int(i == j)

    def a_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(
                ((l + 1) ** 2 - m**2) / ((2 * l + 1) * (2 * l + 3))
            )
        else:
            return 0

    def b_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(
                ((l + m + 1) * (l + m + 2)) / ((2 * l + 1) * (2 * l + 3))
            )
        else:
            return 0

    def c_lm(self, l, m):
        if l >= 0 and abs(m) <= l:
            return np.sqrt(l * (l + 1) - m * (m + 1))
        else:
            return 0

    def l1m1_costh_l2m2(self, l1, m1, l2, m2):
        return (
            self.a_lm(l2, m2) * self.kron_delta(l1, l2 + 1)
            + self.a_lm(l2 - 1, m2) * self.kron_delta(l1, l2 - 1)
        ) * self.kron_delta(m1, m2)

    def l1m1_sinth_ddtheta_l2m2(self, l1, m1, l2, m2):
        return (
            l2 * self.a_lm(l2, m2) * self.kron_delta(l1, l2 + 1)
            - (l2 + 1) * self.a_lm(l2 - 1, m2) * self.kron_delta(l1, l2 - 1)
        ) * self.kron_delta(m1, m2)

    def l1m1_sinth_cosph_l2m2(self, l1, m1, l2, m2):
        integral = (
            self.b_lm(l2 - 1, -m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2, m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2 - 1, m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 - 1)
        )
        integral += (
            self.b_lm(l2, -m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 - 1)
        )
        return 0.5 * integral

    def l1m1_sinth_sinph_l2m2(self, l1, m1, l2, m2):
        integral = (
            self.b_lm(l2 - 1, -m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral -= (
            self.b_lm(l2, m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 + 1)
        )
        integral += (
            self.b_lm(l2 - 1, m2 - 1)
            * self.kron_delta(l1, l2 - 1)
            * self.kron_delta(m1, m2 - 1)
        )
        integral -= (
            self.b_lm(l2, -m2)
            * self.kron_delta(l1, l2 + 1)
            * self.kron_delta(m1, m2 - 1)
        )
        return -1j * integral / 2

    def l1m1_costh_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.cos_th * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_costh_sq_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        """
        z_z_Omega
        """
        integrand = Yl1m1_cc * self.cos_th**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_costh_sinth_sinph_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        """
        z_y_Omega
        """
        integrand = Yl1m1_cc * self.cos_th * self.sin_th * self.sin_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_costh_sinth_cosph_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        """
        z_x_Omega
        """
        integrand = Yl1m1_cc * self.cos_th * self.sin_th * self.cos_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_cosph_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th * self.cos_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_cosph_sq_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        integrand = Yl1m1_cc * self.sin_th**2 * self.cos_ph**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinph_cosph_sinthsq_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = (
            Yl1m1_cc * self.sin_ph * self.cos_ph * self.sin_th**2 * Yl2m2
        )
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sinph_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th * self.sin_ph * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_sinph_sq_l2m2_Lebedev(
        self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2
    ):
        integrand = Yl1m1_cc * self.sin_th**2 * self.sin_ph**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_ddtheta_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        sinth_ddtheta_l2m2 = (
            m2 * self.cos_th * sph_harm_y(m2, l2, self.phi, self.theta)
        )
        if np.abs(m2 + 1) <= l2:
            sinth_ddtheta_l2m2 += (
                np.sqrt((l2 - m2) * (l2 + m2 + 1))
                * self.sin_th
                * self.exp_m1j_p
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        integrand = Yl1m1_cc * sinth_ddtheta_l2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_sinth_sq_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integrand = Yl1m1_cc * self.sin_th**2 * Yl2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_px_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integral = self.c_lm(l2, m2) * self.l1m1_costh_l2m2(
            l1, m1, l2, m2 + 1
        ) - self.c_lm(l2, m2 - 1) * self.l1m1_costh_l2m2(l1, m1, l2, m2 - 1)

        integrand = (
            Yl1m1_cc * self.sin_th * (self.exp_m1j_p - self.exp_p1j_p) * Yl2m2
        )
        integral += m2 * np.sum(4 * np.pi * self.weights * integrand)
        return 0.5 * integral

    def l1m1_y_px_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################
        """
        \left(\cos \phi \cos \theta \frac{\partial}{\partial \theta} - \frac{\sin \phi}{\sin \theta} \frac{\partial}{\partial \phi}\right) Y_{l,m}(\theta, \phi)
        = \frac{1}{2} \left[\cos\theta \left( c_{l,m}Y_{l,m+1} - c_{l,m-1}Y_{l,m-1} \right) + m \sin \theta (e^{-i \phi}-e^{i\phi}) Y_{l,m} \right]
        """
        integrand = (
            0.5 * m2 * self.sin_th * (self.exp_m1j_p - self.exp_p1j_p) * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand += (
                0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm_y(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.sin_ph

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_y_pz_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        sinth_ddtheta_l2m2 = (
            m2 * self.cos_th * sph_harm_y(m2, l2, self.phi, self.theta)
        )
        if np.abs(m2 + 1) <= l2:
            sinth_ddtheta_l2m2 += (
                np.sqrt((l2 - m2) * (l2 + m2 + 1))
                * self.sin_th
                * self.exp_m1j_p
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        integrand = -sinth_ddtheta_l2m2
        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.sin_ph
        integral = np.sum(4 * np.pi * self.weights * integrand)
        return integral

    def l1m1_z_px_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################
        """ """
        integrand = (
            0.5 * m2 * self.sin_th * (self.exp_m1j_p - self.exp_p1j_p) * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand += (
                0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm_y(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.cos_th

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_py_l2m2_Lebedev(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        integral = -self.c_lm(l2, m2) * self.l1m1_costh_l2m2(
            l1, m1, l2, m2 + 1
        ) - self.c_lm(l2, m2 - 1) * self.l1m1_costh_l2m2(l1, m1, l2, m2 - 1)

        integrand = (
            Yl1m1_cc * self.sin_th * (self.exp_m1j_p + self.exp_p1j_p) * Yl2m2
        )
        integral += m2 * np.sum(4 * np.pi * self.weights * integrand)
        return 1j * 0.5 * integral

    def l1m1_x_py_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################

        integrand = (
            1j
            * 0.5
            * m2
            * self.sin_th
            * (self.exp_m1j_p + self.exp_p1j_p)
            * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm_y(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.cos_ph

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_x_pz_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        sinth_ddtheta_l2m2 = (
            m2 * self.cos_th * sph_harm_y(m2, l2, self.phi, self.theta)
        )
        if np.abs(m2 + 1) <= l2:
            sinth_ddtheta_l2m2 += (
                np.sqrt((l2 - m2) * (l2 + m2 + 1))
                * self.sin_th
                * self.exp_m1j_p
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        integrand = -sinth_ddtheta_l2m2
        ###############################################################################
        integrand *= Yl1m1_cc * self.sin_th * self.cos_ph
        integral = np.sum(4 * np.pi * self.weights * integrand)
        return integral

    def l1m1_z_py_l2m2(self, Yl1m1_cc, Yl2m2, l1, m1, l2, m2):
        ###############################################################################

        integrand = (
            1j
            * 0.5
            * m2
            * self.sin_th
            * (self.exp_m1j_p + self.exp_p1j_p)
            * Yl2m2
        )

        if abs(m2 + 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2)
                * self.cos_th
                * sph_harm_y(m2 + 1, l2, self.phi, self.theta)
            )

        if abs(m2 - 1) <= l2:
            integrand -= (
                1j
                * 0.5
                * self.c_lm(l2, m2 - 1)
                * self.cos_th
                * sph_harm_y(m2 - 1, l2, self.phi, self.theta)
            )

        ###############################################################################
        integrand *= Yl1m1_cc * self.cos_th

        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_Y_star_cos_theta_l2m2_Lebedev(self, l1, m1, l2, m2, L, M):
        I = self.I_lm[f"{l1}{m1}"]
        J = self.I_lm[f"{l2}{m2}"]
        K = self.I_lm[f"{L}{M}"]

        Y_l1m1 = self.sph_harms[I, :]
        Y_l2m2 = self.sph_harms[J, :]
        Y_LM = self.sph_harms[K, :]

        integrand = Y_l1m1.conj() * Y_LM.conj() * self.cos_th * Y_l2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_Y_star_sin_theta_ddtheta_l2m2_Lebedev(self, l1, m1, l2, m2, L, M):
        I = self.I_lm[f"{l1}{m1}"]
        J = self.I_lm[f"{l2}{m2}"]
        K = self.I_lm[f"{L}{M}"]

        Y_l1m1 = self.sph_harms[I, :]
        Y_l2m2 = self.sph_harms[J, :]
        Y_LM = self.sph_harms[K, :]

        sin_theta_ddtheta_l2m2 = m2 * self.cos_th * Y_l2m2
        if np.abs(m2 + 1) <= l2:
            J_ = self.I_lm[f"{l2}{m2+1}"]
            Y_l21m2 = self.sph_harms[J_, :]
            sin_theta_ddtheta_l2m2 += (
                np.sqrt((l2 - m2) * (l2 + m2 + 1))
                * self.sin_th
                * self.exp_m1j_p
                * Y_l21m2
            )

        integrand = Y_l1m1.conj() * Y_LM.conj() * sin_theta_ddtheta_l2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_Y_star_l2m2_Lebedev(self, l1, m1, l2, m2, L, M):
        I = self.I_lm[f"{l1}{m1}"]
        J = self.I_lm[f"{l2}{m2}"]
        K = self.I_lm[f"{L}{M}"]

        Y_l1m1 = self.sph_harms[I, :]
        Y_l2m2 = self.sph_harms[J, :]
        Y_LM = self.sph_harms[K, :]

        integrand = Y_l1m1.conj() * Y_LM.conj() * Y_l2m2
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(self, l1, m1, l2, m2, L, M):
        I = self.I_lm[f"{l1}{m1}"]
        J = self.I_lm[f"{l2}{m2}"]
        K = self.I_lm[f"{L}{M}"]

        Y_l1m1 = self.sph_harms[I, :]
        Y_l2m2 = self.sph_harms[J, :]
        Y_LM = self.sph_harms[K, :]

        integrand = (
            Y_l1m1.conj() * Y_LM.conj() * self.cos_ph * self.sin_th * Y_l2m2
        )
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(self, l1, m1, l2, m2, L, M):
        I = self.I_lm[f"{l1}{m1}"]
        J = self.I_lm[f"{l2}{m2}"]
        K = self.I_lm[f"{L}{M}"]

        Y_l1m1 = self.sph_harms[I, :]
        Y_l2m2 = self.sph_harms[J, :]
        Y_LM = self.sph_harms[K, :]

        integrand = (
            Y_l1m1.conj() * Y_LM.conj() * self.sin_ph * self.sin_th * Y_l2m2
        )
        integral = np.sum(4 * np.pi * self.weights * integrand)

        return integral

    def __call__(self, name):
        return self.arr[name]


class AngularMatrixElements_l(AngularMatrixElements):
    def __init__(self, arr_to_calc=[], l_max=5, m=0, N=101):
        super().__init__(l_max, N)

        self.n_lm = l_max + 1
        self.lm_I, self.I_lm = setup_lm_index_mapping_l(l_max, m)
        self.m = m

        for el in arr_to_calc:
            self.arr[el] = np.zeros((self.n_lm, self.n_lm), dtype=np.complex128)

        arr_to_calc_dict = setup_l_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_l_matrix_elements(arr_to_calc_dict, m)

    def setup_l_matrix_elements(self, arr_to_calc_dict, m=0):
        n_l = self.n_l

        for l1 in range(n_l):
            Yl1m_cc = sph_harm_y(m, l1, self.phi, self.theta).conj()
            for l2 in range(n_l):
                Yl2m = sph_harm_y(m, l2, self.phi, self.theta)
                if arr_to_calc_dict["z_Omega"]:
                    self.arr["z_Omega"][l1, l2] = self.l1m1_costh_l2m2(
                        l1, m, l2, m
                    )
                if arr_to_calc_dict["H_z_beta"] and arr_to_calc_dict["z_Omega"]:
                    if abs(m) <= l1 and abs(m) <= l2:
                        self.arr["H_z_beta"][l1, l2] = (
                            -self.l1m1_sinth_ddtheta_l2m2(l1, m, l2, m)
                            - self.arr["z_Omega"][l1, l2]
                        )
                elif (
                    arr_to_calc_dict["H_z_beta"]
                    and arr_to_calc_dict["z_Omega"] == False
                ):
                    if abs(m) <= l1 and abs(m) <= l2:
                        self.arr["H_z_beta"][
                            l1, l2
                        ] = -self.l1m1_sinth_ddtheta_l2m2(
                            l1, m, l2, m
                        ) - self.l1m1_costh_l2m2(
                            l1, m, l2, m
                        )
                if arr_to_calc_dict["H_Bz_Omega"]:
                    if abs(m) <= l1 and abs(m) <= l2:
                        self.arr["H_Bz_Omega"][
                            l1, l2
                        ] = self.l1m1_sinth_sq_l2m2_Lebedev(
                            Yl1m_cc, Yl2m, l1, m, l2, m
                        )


class AngularMatrixElements_lm(AngularMatrixElements):
    def __init__(
        self,
        arr_to_calc=[],
        lmr_arr_to_calc=[],
        l_max=5,
        m_max=None,
        N=101,
        nr=None,
    ):
        super().__init__(l_max, N)

        if (m_max is None) or (m_max > l_max):
            m_max = l_max

        self.n_lm = (m_max + 1) ** 2 + (l_max - m_max) * (2 * m_max + 1)
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm_mrestricted(
            l_max=l_max, m_max=m_max
        )
        self.m_max = m_max

        for el in arr_to_calc:
            self.arr[el] = np.zeros((self.n_lm, self.n_lm), dtype=np.complex128)

        arr_to_calc_dict = setup_lm_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_matrix_elements(arr_to_calc_dict)

    def setup_matrix_elements(self, arr_to_calc_dict):
        tic = time.time()
        n_lm = self.n_lm

        l_max = self.n_l - 1

        for l1 in range(l_max + 1):
            temp_m1_max = min(l1, self.m_max)
            for m1 in range(-temp_m1_max, temp_m1_max + 1):
                I = self.I_lm[f"{l1}{m1}"]
                Yl1m1_cc = sph_harm_y(m1, l1, self.phi, self.theta).conj()
                for l2 in range(l1 - 2, l1 + 3):
                    if l2 >= 0 and l2 <= l_max:
                        temp_m2_max = min(l2, self.m_max)
                        for m2 in range(-temp_m2_max, temp_m2_max + 1):
                            J = self.I_lm[f"{l2}{m2}"]
                            Yl2m2 = sph_harm_y(m2, l2, self.phi, self.theta)

                            if arr_to_calc_dict["x_Omega"]:
                                self.arr["x_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_cosph_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["x_x_Omega"]:
                                self.arr["x_x_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_cosph_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["y_Omega"]:
                                self.arr["y_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sinph_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["y_y_Omega"]:
                                self.arr["y_y_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_sinph_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["z_Omega"]:
                                self.arr["z_Omega"][
                                    I, J
                                ] = self.l1m1_costh_l2m2(l1, m1, l2, m2)

                            if arr_to_calc_dict["z_z_Omega"]:
                                self.arr["z_z_Omega"][
                                    I, J
                                ] = self.l1m1_costh_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["z_x_Omega"]:
                                self.arr["z_x_Omega"][
                                    I, J
                                ] = self.l1m1_costh_sinth_cosph_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["z_y_Omega"]:
                                self.arr["z_y_Omega"][
                                    I, J
                                ] = self.l1m1_costh_sinth_sinph_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["y_x_Omega"]:
                                self.arr["y_x_Omega"][
                                    I, J
                                ] = self.l1m1_sinph_cosph_sinthsq_l2m2(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )

                            if arr_to_calc_dict["y_px_beta"]:
                                self.arr["y_px_beta"][I, J] = (
                                    self.l1m1_y_px_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["y_pz_beta"]:
                                self.arr["y_pz_beta"][I, J] = (
                                    self.l1m1_y_pz_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["z_y_Omega"][I, J]
                                )

                            if arr_to_calc_dict["x_py_beta"]:
                                self.arr["x_py_beta"][I, J] = (
                                    self.l1m1_x_py_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["x_pz_beta"]:
                                self.arr["x_pz_beta"][I, J] = (
                                    self.l1m1_x_pz_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["z_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["z_py_beta"]:
                                self.arr["z_py_beta"][I, J] = (
                                    self.l1m1_z_py_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["z_y_Omega"][I, J]
                                )

                            if arr_to_calc_dict["z_px_beta"]:
                                self.arr["z_px_beta"][I, J] = (
                                    self.l1m1_z_px_l2m2(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["z_x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_x_beta"]:
                                self.arr["H_x_beta"][I, J] = (
                                    self.l1m1_px_l2m2_Lebedev(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["x_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_y_beta"]:
                                self.arr["H_y_beta"][I, J] = (
                                    self.l1m1_py_l2m2_Lebedev(
                                        Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                    )
                                    - self.arr["y_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_z_beta"]:
                                self.arr["H_z_beta"][I, J] = (
                                    -self.l1m1_sinth_ddtheta_l2m2(
                                        l1, m1, l2, m2
                                    )
                                    - self.arr["z_Omega"][I, J]
                                )

                            if arr_to_calc_dict["H_Bz_Omega"]:
                                self.arr["H_Bz_Omega"][
                                    I, J
                                ] = self.l1m1_sinth_sq_l2m2_Lebedev(
                                    Yl1m1_cc, Yl2m2, l1, m1, l2, m2
                                )
        toc = time.time()
        print(f"Time setup angular matrix elements: {toc-tic}")


class AngularMatrixElements_lmr(AngularMatrixElements):
    def __init__(
        self, arr_to_calc, nr, r, k, phi_k, theta_k, l_max=5, m_max=None, N=101
    ):
        super().__init__(l_max, N)

        if (m_max is None) or (m_max > l_max):
            m_max = l_max

        self.n_l = l_max + 1
        self.n_lm = (m_max + 1) ** 2 + (l_max - m_max) * (2 * m_max + 1)
        self.nr = nr
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm(l_max=2 * l_max + 2)
        self.lm_I_, self.I_lm_ = setup_lm_index_mapping_lm_mrestricted(
            l_max=2 * l_max + 2, m_max=m_max
        )
        self.m_max = m_max

        self.r = r
        self.k = k
        self.phi_k = phi_k
        self.theta_k = theta_k

        for el in arr_to_calc:
            self.arr[el] = np.zeros(
                (self.n_lm, self.n_lm, self.nr), dtype=np.complex128
            )

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        if True in arr_to_calc_dict.values():
            self.setup_matrix_elements(arr_to_calc_dict)

    def setup_matrix_elements(self, arr_to_calc_dict):
        r = self.r
        nl = self.n_l
        theta = self.theta
        phi = self.phi
        k = self.k
        theta_k = self.theta_k
        phi_k = self.phi_k

        self.sph_harms = np.zeros(
            (4 * self.n_l**2, len(self.weights)), dtype=np.complex128
        )

        for L in range(2 * nl):
            for M in range(-L, L + 1):
                I = self.I_lm[f"{L}{M}"]
                self.sph_harms[I, :] = sph_harm_y(M, L, phi, theta)

        sph_jn = np.zeros((2 * self.n_l, self.nr))
        sph_jn2 = np.zeros((2 * self.n_l, self.nr))
        for L in range(2 * nl):
            sph_jn[L, :] = spherical_jn(L, k * r)
            sph_jn2[L, :] = spherical_jn(L, 2 * k * r)

        self.sph_jn = sph_jn
        self.sph_jn2 = sph_jn2

        for l1 in range(nl):
            temp_m1_max = min(l1, self.m_max)
            for m1 in range(-temp_m1_max, temp_m1_max + 1):
                I = self.I_lm_[f"{l1}{m1}"]
                for l2 in range(nl):
                    temp_m2_max = min(l2, self.m_max)
                    for m2 in range(-temp_m2_max, temp_m2_max + 1):
                        J = self.I_lm_[f"{l2}{m2}"]
                        for L in range(2 * nl):
                            temp_M_max = min(L, self.m_max)
                            for M in range(-temp_M_max, temp_M_max + 1):
                                if arr_to_calc_dict["expkr_costh"]:
                                    cond1 = -m1 - M + m2 == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            sign=1,
                                        )
                                        F_W = self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        self.arr["expkr_costh"][I, J, :] += (
                                            F_W * F_r
                                        )

                                if arr_to_calc_dict["expkr_sinth_ddtheta"]:
                                    cond1 = -m1 - M + m2 == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            sign=1,
                                        )
                                        F_W = self.l1m1_Y_star_sin_theta_ddtheta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        self.arr["expkr_sinth_ddtheta"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr2"]:
                                    cond1 = -m1 - M + m2 == 0
                                    cond2 = (np.abs(l1 - L) <= l2) and (
                                        l2 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2) % 2) == 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            sph_jn2[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            sign=1,
                                        )
                                        F_W = self.l1m1_Y_star_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        self.arr["expkr2"][I, J, :] += F_W * F_r

                                if arr_to_calc_dict["expkr_cosph_sinth"]:
                                    cond1 = np.abs(-m1 - M + m2) == 1
                                    cond2 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        self.arr["expkr_cosph_sinth"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr_sinph_sinth"]:
                                    cond1 = np.abs(-m1 - M + m2) == 1
                                    cond2 = ((l1 + L + l2 + 1) % 2) == 0

                                    if cond1 and cond2:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )

                                        self.arr["expkr_sinph_sinth"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr_m2_sinph_sinth"]:
                                    cond1 = np.abs(-m1 - M + m2) == 1
                                    cond2 = ((l1 + L + l2 + 1) % 2) == 0
                                    cond3 = m2 != 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = (
                                            m2
                                            * self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2, L, M
                                            )
                                        )

                                        self.arr["expkr_m2_sinph_sinth"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr_m2_cosph_sinth"]:
                                    cond1 = np.abs(-m1 - M + m2) == 1
                                    cond2 = ((l1 + L + l2 + 1) % 2) == 0
                                    cond3 = m2 != 0

                                    if cond1 and cond2 and cond3:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = (
                                            m2
                                            * self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2, L, M
                                            )
                                        )

                                        self.arr["expkr_m2_cosph_sinth"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr_c_costh_(m+1)"]:
                                    cond1 = -m1 - M + (m2 + 1) == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0
                                    cond4 = abs(m2 + 1) <= l2

                                    if cond1 and cond2 and cond3 and cond4:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = coeff_c(
                                            l2, m2
                                        ) * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2 + 1, L, M
                                        )

                                        self.arr["expkr_c_costh_(m+1)"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["expkr_c_costh_(m-1)"]:
                                    cond1 = -m1 - M + (m2 - 1) == 0
                                    cond2 = (np.abs(l1 - L) <= l2 + 1) and (
                                        l2 - 1 <= l1 + L
                                    )
                                    cond3 = ((l1 + L + l2 + 1) % 2) == 0
                                    cond4 = abs(m2 - 1) <= l2

                                    if cond1 and cond2 and cond3 and cond4:
                                        F_r = f_r(
                                            sph_jn[L, :],
                                            L,
                                            M,
                                            theta_k,
                                            phi_k,
                                            1,
                                        )
                                        F_W = coeff_c(
                                            l2, m2 - 1
                                        ) * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2 - 1, L, M
                                        )

                                        self.arr["expkr_c_costh_(m-1)"][
                                            I, J, :
                                        ] += (F_W * F_r)

                                if arr_to_calc_dict["M_tilde_x"]:
                                    F_r = f_r(
                                        sph_jn[L, :], L, M, theta_k, phi_k, 1
                                    )
                                    F_W1 = self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                                        l1, m1, l2, m2, L, M
                                    )
                                    F_W2 = (
                                        1j
                                        * m2
                                        * self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )
                                    )
                                    F_W3 = 0
                                    F_W4 = 0
                                    if abs(m2 + 1) <= l2:
                                        F_W3 = (
                                            -(1 / 2)
                                            * coeff_c(l2, m2)
                                            * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2 + 1, L, M
                                            )
                                        )
                                    if abs(m2 - 1) <= l2:
                                        F_W4 = (
                                            (1 / 2)
                                            * coeff_c(l2, m2 - 1)
                                            * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2 - 1, L, M
                                            )
                                        )

                                    self.arr["M_tilde_x"][I, J, :] += (
                                        F_W1 + F_W2 + F_W3 + F_W4
                                    ) * F_r

                                if arr_to_calc_dict["M_tilde_y"]:
                                    F_r = f_r(
                                        sph_jn[L, :], L, M, theta_k, phi_k, 1
                                    )
                                    F_W1 = self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                                        l1, m1, l2, m2, L, M
                                    )
                                    F_W2 = (
                                        -1j
                                        * m2
                                        * self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                                            l1, m1, l2, m2, L, M
                                        )
                                    )
                                    F_W3 = 0
                                    F_W4 = 0
                                    if abs(m2 + 1) <= l2:
                                        F_W3 = (
                                            (1j / 2)
                                            * coeff_c(l2, m2)
                                            * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2 + 1, L, M
                                            )
                                        )
                                    if abs(m2 - 1) <= l2:
                                        F_W4 = (
                                            (1j / 2)
                                            * coeff_c(l2, m2 - 1)
                                            * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                                                l1, m1, l2, m2 - 1, L, M
                                            )
                                        )

                                    self.arr["M_tilde_y"][I, J, :] += (
                                        F_W1 + F_W2 + F_W3 + F_W4
                                    ) * F_r


class AngularMatrixElements_orders(AngularMatrixElements):
    def __init__(self, l_max, N):
        super().__init__(l_max, N)

    def compute_matrix_elements(
        self, l1, m1, l2, m2, I, J, L, M, arr_to_calc_dict
    ):
        theta_k = self.theta_k
        phi_k = self.phi_k

        Y_k = sph_harm_y(M, L, phi_k, theta_k) * (1j) ** L
        if arr_to_calc_dict["expkr_costh"]:
            cond1 = -m1 - M + m2 == 0
            cond2 = (np.abs(l1 - L) <= l2 + 1) and (l2 - 1 <= l1 + L)
            cond3 = ((l1 + L + l2 + 1) % 2) == 0

            if cond1 and cond2 and cond3:
                F_W = self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_costh"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_sinth_ddtheta"]:
            cond1 = -m1 - M + m2 == 0
            cond2 = (np.abs(l1 - L) <= l2 + 1) and (l2 - 1 <= l1 + L)
            cond3 = ((l1 + L + l2 + 1) % 2) == 0

            if cond1 and cond2 and cond3:
                F_W = self.l1m1_Y_star_sin_theta_ddtheta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_sinth_ddtheta"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr2"]:
            cond1 = -m1 - M + m2 == 0
            cond2 = (np.abs(l1 - L) <= l2) and (l2 <= l1 + L)
            cond3 = ((l1 + L + l2) % 2) == 0

            if cond1 and cond2 and cond3:
                F_W = self.l1m1_Y_star_l2m2_Lebedev(l1, m1, l2, m2, L, M)

                self.arr["expkr2"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_cosph_sinth"]:
            cond1 = np.abs(-m1 - M + m2) == 1
            cond2 = ((l1 + L + l2 + 1) % 2) == 0

            if cond1 and cond2:
                F_W = self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_cosph_sinth"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_sinph_sinth"]:
            cond1 = np.abs(-m1 - M + m2) == 1
            cond2 = ((l1 + L + l2 + 1) % 2) == 0

            if cond1 and cond2:
                F_W = self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_sinph_sinth"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_m2_sinph_sinth"]:
            cond1 = np.abs(-m1 - M + m2) == 1
            cond2 = ((l1 + L + l2 + 1) % 2) == 0
            cond3 = m2 != 0

            if cond1 and cond2 and cond3:
                F_W = m2 * self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_m2_sinph_sinth"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_m2_cosph_sinth"]:
            cond1 = np.abs(-m1 - M + m2) == 1
            cond2 = ((l1 + L + l2 + 1) % 2) == 0
            cond3 = m2 != 0

            if cond1 and cond2 and cond3:
                F_W = m2 * self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )

                self.arr["expkr_m2_cosph_sinth"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_c_costh_(m+1)"]:
            cond1 = -m1 - M + (m2 + 1) == 0
            cond2 = (np.abs(l1 - L) <= l2 + 1) and (l2 - 1 <= l1 + L)
            cond3 = ((l1 + L + l2 + 1) % 2) == 0
            cond4 = abs(m2 + 1) <= l2

            if cond1 and cond2 and cond3 and cond4:
                F_W = coeff_c(l2, m2) * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                    l1, m1, l2, m2 + 1, L, M
                )

                self.arr["expkr_c_costh_(m+1)"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["expkr_c_costh_(m-1)"]:
            cond1 = -m1 - M + (m2 - 1) == 0
            cond2 = (np.abs(l1 - L) <= l2 + 1) and (l2 - 1 <= l1 + L)
            cond3 = ((l1 + L + l2 + 1) % 2) == 0
            cond4 = abs(m2 - 1) <= l2

            if cond1 and cond2 and cond3 and cond4:
                F_W = coeff_c(
                    l2, m2 - 1
                ) * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                    l1, m1, l2, m2 - 1, L, M
                )

                self.arr["expkr_c_costh_(m-1)"][L, I, J] += F_W * Y_k

        if arr_to_calc_dict["M_tilde_x"]:
            F_W1 = self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                l1, m1, l2, m2, L, M
            )
            F_W2 = (
                1j
                * m2
                * self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )
            )
            F_W3 = 0
            F_W4 = 0
            if abs(m2 + 1) <= l2:
                F_W3 = (
                    -(1 / 2)
                    * coeff_c(l2, m2)
                    * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                        l1, m1, l2, m2 + 1, L, M
                    )
                )
            if abs(m2 - 1) <= l2:
                F_W4 = (
                    (1 / 2)
                    * coeff_c(l2, m2 - 1)
                    * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                        l1, m1, l2, m2 - 1, L, M
                    )
                )

            self.arr["M_tilde_x"][L, I, J] += (F_W1 + F_W2 + F_W3 + F_W4) * Y_k

        if arr_to_calc_dict["M_tilde_y"]:
            F_W1 = self.l1m1_Y_star_sin_phi_sin_theta_l2m2_Lebedev(
                l1, m1, l2, m2, L, M
            )
            F_W2 = (
                -1j
                * m2
                * self.l1m1_Y_star_cos_phi_sin_theta_l2m2_Lebedev(
                    l1, m1, l2, m2, L, M
                )
            )
            F_W3 = 0
            F_W4 = 0
            if abs(m2 + 1) <= l2:
                F_W3 = (
                    (1j / 2)
                    * coeff_c(l2, m2)
                    * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                        l1, m1, l2, m2 + 1, L, M
                    )
                )
            if abs(m2 - 1) <= l2:
                F_W4 = (
                    (1j / 2)
                    * coeff_c(l2, m2 - 1)
                    * self.l1m1_Y_star_cos_theta_l2m2_Lebedev(
                        l1, m1, l2, m2 - 1, L, M
                    )
                )

            self.arr["M_tilde_y"][L, I, J] += (F_W1 + F_W2 + F_W3 + F_W4) * Y_k


class AngularMatrixElements_lmr_orders(AngularMatrixElements_orders):
    def __init__(
        self,
        arr_to_calc,
        nr,
        r,
        k,
        phi_k,
        theta_k,
        l_max=5,
        m_max=None,
        N=101,
        NL=1,
    ):
        super().__init__(l_max, N)

        if (m_max is None) or (m_max > l_max):
            m_max = l_max

        self.n_l = l_max + 1
        self.n_lm = (m_max + 1) ** 2 + (l_max - m_max) * (2 * m_max + 1)
        self.nr = nr
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm(l_max=2 * l_max + 2)
        self.lm_I_, self.I_lm_ = setup_lm_index_mapping_lm_mrestricted(
            l_max=2 * l_max + 2, m_max=m_max
        )

        self.m_max = m_max

        self.NL = NL

        self.r = r
        self.k = k
        self.phi_k = phi_k
        self.theta_k = theta_k

        for el in arr_to_calc:
            self.arr[el] = np.zeros(
                (NL, self.n_lm, self.n_lm), dtype=np.complex128
            )

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        self.sph_harms = np.zeros(
            ((2 * self.n_l) ** 2, len(self.weights)), dtype=np.complex128
        )

        for L_ in range(2 * self.n_l):
            for M in range(-L_, L_ + 1):
                I = self.I_lm[f"{L_}{M}"]
                self.sph_harms[I, :] = sph_harm_y(M, L_, self.phi, self.theta)

        self.sph_jn = np.zeros((NL, self.nr))
        self.sph_jn2 = np.zeros((NL, self.nr))

        if True in arr_to_calc_dict.values():
            for L in range(NL):
                self.setup_matrix_elements(arr_to_calc_dict, L)

    def setup_matrix_elements(self, arr_to_calc_dict, L):
        r = self.r
        nl = self.n_l
        k = self.k

        self.sph_jn[L, :] = 4 * np.pi * spherical_jn(L, k * r)
        self.sph_jn2[L, :] = 4 * np.pi * spherical_jn(L, 2 * k * r)

        for l1 in range(nl):
            temp_m1_max = min(l1, self.m_max)
            for m1 in range(-temp_m1_max, temp_m1_max + 1):
                I = self.I_lm_[f"{l1}{m1}"]
                for l2 in range(nl):
                    temp_m2_max = min(l2, self.m_max)
                    for m2 in range(-temp_m2_max, temp_m2_max + 1):
                        J = self.I_lm_[f"{l2}{m2}"]
                        for M in range(-L, L + 1):
                            self.compute_matrix_elements(
                                l1, m1, l2, m2, I, J, L, M, arr_to_calc_dict
                            )


class AngularMatrixElements_lr_orders(AngularMatrixElements_orders):
    def __init__(self, arr_to_calc, nr, r, k, phi_k, theta_k, l_max=5, N=101):
        super().__init__(l_max, N)

        self.n_l = l_max + 1
        self.n_lm = self.n_l
        self.nr = nr
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm(l_max=2 * l_max + 2)

        self.NL = 1

        self.r = r
        self.k = k
        self.phi_k = phi_k
        self.theta_k = theta_k

        for el in arr_to_calc:
            self.arr[el] = np.zeros(
                (self.NL, self.n_lm, self.n_lm), dtype=np.complex128
            )

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        self.sph_harms = np.zeros(
            ((2 * self.n_l) ** 2, len(self.weights)), dtype=np.complex128
        )

        for L_ in range(2 * self.n_l):
            for M in range(-L_, L_ + 1):
                I = self.I_lm[f"{L_}{M}"]
                self.sph_harms[I, :] = sph_harm_y(M, L_, self.phi, self.theta)

        self.sph_jn = np.zeros((self.NL, self.nr))
        self.sph_jn2 = np.zeros((self.NL, self.nr))

        if True in arr_to_calc_dict.values():
            self.setup_matrix_elements(arr_to_calc_dict, 0)

    def setup_matrix_elements(self, arr_to_calc_dict, L):
        r = self.r
        nl = self.n_l
        k = self.k

        self.sph_jn[L, :] = 4 * np.pi * spherical_jn(L, k * r)
        self.sph_jn2[L, :] = 4 * np.pi * spherical_jn(L, 2 * k * r)

        for l1 in range(nl):
            I = self.I_lm[f"{l1}{0}"]
            for l2 in range(nl):
                J = self.I_lm[f"{l2}{0}"]
                self.compute_matrix_elements(
                    l1, 0, l2, 0, l1, l2, L, 0, arr_to_calc_dict
                )


class AngularMatrixElements_lr_Coulomb(AngularMatrixElements):
    def __init__(self, arr_to_calc, nr, r, l_max=5, m=0, L_max=5, a=2.0, N=101):
        super().__init__(l_max, N)

        n_sph_harms = max(l_max + 1, L_max + 1)

        self.n_lm = l_max + 1
        self.L_max = L_max
        self.lm_I, self.I_lm = setup_lm_index_mapping_l(n_sph_harms, m)
        self.m = m
        self.nr = nr
        self.r = r
        self.a = a

        self.arr["1/(r-a)"] = np.zeros(
            (self.n_lm, self.n_lm, self.nr), dtype=np.complex128
        )

        self.r_inv_l = np.zeros((L_max + 1, nr))
        for L in range(L_max + 1):
            self.r_inv_l[L, :] = compute_r_inv_l(r, a, L)

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        self.sph_harms = np.zeros(
            (n_sph_harms, len(self.weights)), dtype=np.complex128
        )

        for L_ in range(n_sph_harms):
            self.sph_harms[L_, :] = sph_harm_y(0, L_, self.phi, self.theta)

        if True in arr_to_calc_dict.values():
            self.setup_l_matrix_elements(arr_to_calc_dict, m)

    def setup_l_matrix_elements(self, arr_to_calc_dict, m=0, M=0):
        n_l = self.n_l
        L_max = self.L_max
        nr = self.nr

        for l1 in range(n_l):
            for l2 in range(n_l):
                for L in range(L_max + 1):
                    if arr_to_calc_dict["1/(r-a)"]:
                        self.arr["1/(r-a)"][
                            l1, l2, :
                        ] += self.l1m1_Y_star_l2m2_Lebedev(
                            l1, m, l2, m, L, M
                        ) * compute_r_inv_l(
                            self.r, self.a, L
                        )


class AngularMatrixElements_lmr_Coulomb(AngularMatrixElements):
    def __init__(
        self, arr_to_calc, nr, r, l_max=5, m_max=None, L_max=5, a=2.0, N=101
    ):
        super().__init__(l_max, N)

        if (m_max is None) or (m_max > l_max):
            m_max = l_max

        n_sph_harms = max(l_max + 1, L_max + 1)

        self.n_lm = (m_max + 1) ** 2 + (l_max - m_max) * (2 * m_max + 1)
        self.L_max = L_max
        self.lm_I, self.I_lm = setup_lm_index_mapping_lm_mrestricted(
            n_sph_harms, m_max
        )
        self.nr = nr
        self.r = r
        self.a = a

        self.m_max = m_max

        self.mask_m0 = np.zeros(self.n_lm, dtype=bool)
        for l in range(self.n_l):
            I = self.I_lm[f"{l}{0}"]
            self.mask_m0[I] = True

        self.arr["1/(r-a)"] = np.zeros(
            (self.n_lm, self.n_lm, self.nr), dtype=np.complex128
        )

        self.r_inv_l = np.zeros((L_max + 1, nr))
        for L in range(L_max + 1):
            self.r_inv_l[L, :] = compute_r_inv_l(r, a, L)

        arr_to_calc_dict = setup_lmr_arr_to_calc(arr_to_calc)

        self.sph_harms = np.zeros(
            (self.n_lm, len(self.weights)), dtype=np.complex128
        )

        for L_ in range(n_sph_harms):
            temp_M_max = min(L_, m_max)
            for M_ in range(-temp_M_max, temp_M_max + 1):
                I = self.I_lm[f"{L_}{M_}"]
                self.sph_harms[I, :] = sph_harm_y(M_, L_, self.phi, self.theta)

        if True in arr_to_calc_dict.values():
            self.setup_lm_matrix_elements(arr_to_calc_dict)

    def setup_lm_matrix_elements(self, arr_to_calc_dict, M=0):
        n_l = self.n_l
        L_max = self.L_max
        nr = self.nr

        for l1 in range(n_l):
            for l2 in range(n_l):
                temp_m_max = min(l1, l2, self.m_max)
                for m in range(-temp_m_max, temp_m_max + 1):
                    I = self.I_lm[f"{l1}{m}"]
                    J = self.I_lm[f"{l2}{m}"]
                    for L in range(L_max + 1):
                        if arr_to_calc_dict["1/(r-a)"]:
                            self.arr["1/(r-a)"][
                                I, J, :
                            ] += self.l1m1_Y_star_l2m2_Lebedev(
                                l1, m, l2, m, L, M
                            ) * compute_r_inv_l(
                                self.r, self.a, L
                            )


def compute_r_inv_l(r, a, l):
    r_min = np.minimum(r, a)
    r_max = np.maximum(r, a)
    return np.sqrt(4 * np.pi / (2 * l + 1)) * r_min**l / r_max ** (l + 1)


def setup_lm_index_mapping_l(l_max, m=0):
    lm_I = []
    I_lm = dict()
    I = 0
    for l in range(l_max + 1):
        I_lm[f"{l}{m}"] = I
        lm_I.append((l, m))
        I += 1

    return lm_I, I_lm


def setup_lm_index_mapping_lm(l_max):
    lm_I = []
    I_lm = dict()
    I = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            I_lm[f"{l}{m}"] = I
            lm_I.append((l, m))
            I += 1

    return lm_I, I_lm


def setup_lm_index_mapping_lm_mrestricted(l_max, m_max):
    lm_I = []
    I_lm = dict()
    I = 0
    for l in range(l_max + 1):
        temp_m_max = min(l, m_max)
        for m in range(-temp_m_max, temp_m_max + 1):
            I_lm[f"{l}{m}"] = I
            lm_I.append((l, m))
            I += 1

    return lm_I, I_lm


def setup_l_arr_to_calc(arr_to_calc_list=[]):
    arr_to_calc_dict = {
        "z_Omega": False,
        "H_z_beta": False,
        "H_Bz_Omega": False,
    }

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def setup_lm_arr_to_calc(arr_to_calc_list=[]):
    arr_to_calc_dict = {
        "H_x_beta": False,
        "H_y_beta": False,
        "H_z_beta": False,
        "x_Omega": False,
        "y_Omega": False,
        "z_Omega": False,
        "x_x_Omega": False,
        "y_y_Omega": False,
        "z_z_Omega": False,
        "y_x_Omega": False,
        "z_x_Omega": False,
        "z_y_Omega": False,
        "x_py_beta": False,
        "x_pz_beta": False,
        "y_px_beta": False,
        "y_pz_beta": False,
        "z_px_beta": False,
        "z_py_beta": False,
        "H_Bz_Omega": False,
    }

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def setup_lmr_arr_to_calc(arr_to_calc_list):
    arr_to_calc_dict = {
        "expkr_costh": False,
        "expkr_sinth_ddtheta": False,
        "expkr2": False,
        "expkr_cosph_sinth": False,
        "expkr_sinph_sinth": False,
        "M_tilde_x": False,
        "M_tilde_y": False,
        "expkr_m2_sinph_sinth": False,
        "expkr_m2_cosph_sinth": False,
        "expkr_c_costh_(m+1)": False,
        "expkr_c_costh_(m-1)": False,
        "1/(r-a)": False,
    }

    return set_boolean_values_to_dict(arr_to_calc_dict, arr_to_calc_list)


def set_boolean_values_to_dict(_dict, _list):
    for el in _list:
        if el not in _dict.keys():
            raise ValueError(f"Calculation of {el} array is not implemented")
        else:
            _dict[el] = True

    return _dict


from sympy import *
from sympy.physics.quantum.cg import CG


def Clebsch_Gordan(j1, m1, j2, m2, j3, m3):
    return CG(j1, m1, j2, m2, j3, m3).doit().evalf()


def A(l1, l2, l3):
    return np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (4 * np.pi * (2 * l3 + 1)))


def f_r(sph_jn, L, M, theta_k, phi_k, sign):
    C = 4 * np.pi * (sign * 1j) ** L
    Y = sph_harm_y(M, L, phi_k, theta_k)
    return C * Y * sph_jn


def f_W(l1, m1, l2, m2, l3, m3):
    return A(l1, l2, l3) * Clebsch_Gordan(l1, m1, l2, m2, l3, m3)


def coeff_c(l, m):
    return np.sqrt(l * (l + 1) - m * (m + 1))
