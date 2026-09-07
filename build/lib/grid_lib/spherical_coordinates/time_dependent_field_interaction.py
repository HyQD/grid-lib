import numpy as np
from opt_einsum import contract


from grid_lib.spherical_coordinates.rhs import VPsi
from grid_lib.spherical_coordinates.Hpsi_components import (
    x_psi,
    y_psi,
    z_psi,
    x_x_psi,
    y_y_psi,
    px_psi,
    py_psi,
    pz_psi,
    y_px_psi,
)

import time


class V_psi_length_z(VPsi):
    def __init__(
        self, angular_matrix_elements, radial_matrix_elements, e_field_z
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            e_field_z=e_field_z,
        )

        self.r = radial_matrix_elements.r
        self.nr = radial_matrix_elements.nr
        self.nl = angular_matrix_elements.n_lm
        self.z_Omega = angular_matrix_elements("z_Omega")

    def __call__(self, psi, t, ravel=True):
        psi_new = self.e_field_z(t) * z_psi(psi, self.z_Omega, self.r)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_length(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        e_field_x=None,
        e_field_y=None,
        e_field_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            e_field_x=e_field_x,
            e_field_y=e_field_y,
            e_field_z=e_field_z,
        )

        self.x_active = False if e_field_x is None else True
        self.y_active = False if e_field_y is None else True
        self.z_active = False if e_field_z is None else True

        self.r = self.radial_matrix_elements.r

        if self.x_active:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
        if self.y_active:
            self.y_Omega = self.angular_matrix_elements("y_Omega")
        if self.z_active:
            self.z_Omega = self.angular_matrix_elements("z_Omega")

        self.active = True

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        if self.x_active:
            psi_new += self.e_field_x(t) * x_psi(psi, self.x_Omega, self.r)

        if self.y_active:
            psi_new += self.e_field_y(t) * y_psi(psi, self.y_Omega, self.r)

        if self.z_active:
            psi_new += self.e_field_z(t) * z_psi(psi, self.z_Omega, self.r)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity_z(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_z,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_z=a_field_z,
        )

        self.r = radial_matrix_elements.r
        self.nr = radial_matrix_elements.nr
        self.nl = angular_matrix_elements.n_lm

        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.z_Omega = self.angular_matrix_elements("z_Omega")
        self.H_z_beta = self.angular_matrix_elements("H_z_beta")

    def __call__(self, psi, t, ravel=True):
        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        psi_new = self.a_field_z(t) * pz_psi(
            psi, dpsi_dr, self.z_Omega, self.H_z_beta, self.r_inv
        )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_x=None,
        a_field_y=None,
        a_field_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_x=a_field_x,
            a_field_y=a_field_y,
            a_field_z=a_field_z,
        )

        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.x_active = False if a_field_x is None else True
        self.y_active = False if a_field_y is None else True
        self.z_active = False if a_field_z is None else True

        if self.x_active:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
            self.H_x_beta = self.angular_matrix_elements("H_x_beta")
        if self.y_active:
            self.y_Omega = self.angular_matrix_elements("y_Omega")
            self.H_y_beta = self.angular_matrix_elements("H_y_beta")
        if self.z_active:
            self.z_Omega = self.angular_matrix_elements("z_Omega")
            self.H_z_beta = self.angular_matrix_elements("H_z_beta")

        self.active = True

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        if self.x_active:
            psi_new += self.a_field_x(t) * px_psi(
                psi, dpsi_dr, self.x_Omega, self.H_x_beta, self.r_inv
            )

        if self.y_active:
            psi_new += self.a_field_y(t) * py_psi(
                psi, dpsi_dr, self.y_Omega, self.H_y_beta, self.r_inv
            )

        if self.z_active:
            psi_new += self.a_field_z(t) * pz_psi(
                psi, dpsi_dr, self.z_Omega, self.H_z_beta, self.r_inv
            )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_velocity_first(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        a_field_x=None,
        a_field_y=None,
        a_field_z=None,
        k_x=None,
        k_y=None,
        k_z=None,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            a_field_x=a_field_x,
            a_field_y=a_field_y,
            a_field_z=a_field_z,
        )

        self.k_x = k_x
        self.k_y = k_y
        self.k_z = k_z

        self.r = self.radial_matrix_elements.r
        self.r_inv = self.radial_matrix_elements.r_inv
        self.D1 = self.radial_matrix_elements.D1

        self.x_polarized = False if a_field_x is None else True
        self.y_polarized = False if a_field_y is None else True
        self.z_polarized = False if a_field_z is None else True

        self.x_propagation = False if k_x is None else True
        self.y_propagation = False if k_y is None else True
        self.z_propagation = False if k_z is None else True

        if self.x_polarized:
            self.x_Omega = self.angular_matrix_elements("x_Omega")
            self.H_x_beta = self.angular_matrix_elements("H_x_beta")

            if self.y_propagation:
                self.y_Omega = self.angular_matrix_elements("y_Omega")
                self.y_y_Omega = self.angular_matrix_elements("y_y_Omega")
                self.y_x_Omega = self.angular_matrix_elements("y_x_Omega")
                self.y_px_beta = self.angular_matrix_elements("y_px_beta")

            if self.z_propagation:
                self.z_Omega = self.angular_matrix_elements("z_Omega")
                self.z_z_Omega = self.angular_matrix_elements("z_z_Omega")
                self.z_x_Omega = self.angular_matrix_elements("z_x_Omega")
                self.z_px_beta = self.angular_matrix_elements("z_px_beta")

        if self.y_polarized:
            self.y_Omega = self.angular_matrix_elements("y_Omega")
            self.H_y_beta = self.angular_matrix_elements("H_y_beta")

            if self.x_propagation:
                self.x_Omega = self.angular_matrix_elements("x_Omega")
                self.x_y_Omega = self.angular_matrix_elements("y_x_Omega")
                self.x_py_beta = self.angular_matrix_elements("x_py_beta")
                self.x_x_Omega = self.angular_matrix_elements("x_x_Omega")

            if self.z_propagation:
                self.z_Omega = self.angular_matrix_elements("z_Omega")
                self.z_z_Omega = self.angular_matrix_elements("z_z_Omega")
                self.z_y_Omega = self.angular_matrix_elements("z_y_Omega")
                self.z_py_beta = self.angular_matrix_elements("z_py_beta")

        if self.z_polarized:
            self.z_Omega = self.angular_matrix_elements("z_Omega")
            self.H_z_beta = self.angular_matrix_elements("H_z_beta")

            if self.x_propagation:
                self.x_Omega = self.angular_matrix_elements("x_Omega")
                self.x_z_Omega = self.angular_matrix_elements("z_x_Omega")
                self.x_pz_beta = self.angular_matrix_elements("x_pz_beta")
                self.x_x_Omega = self.angular_matrix_elements("x_x_Omega")

            if self.y_propagation:
                self.y_Omega = self.angular_matrix_elements("y_Omega")
                self.y_y_Omega = self.angular_matrix_elements("y_y_Omega")
                self.y_z_Omega = self.angular_matrix_elements("z_y_Omega")
                self.y_pz_beta = self.angular_matrix_elements("y_pz_beta")

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        if self.x_polarized:
            A1_x_t, A2_x_t = self.a_field_x(t)
            psi_new += A1_x_t * px_psi(
                psi, dpsi_dr, self.x_Omega, self.H_x_beta, self.r_inv
            )
            psi_new += 0.5 * A1_x_t**2 * psi

            if self.y_propagation:
                psi_new += (
                    A2_x_t
                    * self.k_y
                    * y_px_psi(
                        psi, dpsi_dr, self.y_x_Omega, self.y_px_beta, self.r
                    )
                )

                psi_new += (
                    A1_x_t
                    * A2_x_t
                    * self.k_y
                    * y_psi(psi, self.y_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_x_t**2
                    * self.k_y**2
                    * y_y_psi(psi, self.y_y_Omega, self.r)
                )
            if self.z_propagation:
                psi_new += (
                    A2_x_t
                    * self.k_z
                    * y_px_psi(
                        psi, dpsi_dr, self.z_x_Omega, self.z_px_beta, self.r
                    )
                )

                psi_new += (
                    A1_x_t
                    * A2_x_t
                    * self.k_z
                    * y_psi(psi, self.z_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_x_t**2
                    * self.k_z**2
                    * y_y_psi(psi, self.z_z_Omega, self.r)
                )

        if self.y_polarized:
            A1_y_t, A2_y_t = self.a_field_y(t)

            psi_new += A1_y_t * px_psi(
                psi, dpsi_dr, self.y_Omega, self.H_y_beta, self.r_inv
            )

            psi_new += 0.5 * A1_y_t**2 * psi

            if self.x_propagation:
                psi_new += (
                    A2_y_t
                    * self.k_x
                    * y_px_psi(
                        psi, dpsi_dr, self.x_y_Omega, self.x_py_beta, self.r
                    )
                )

                psi_new += (
                    A1_y_t
                    * A2_y_t
                    * self.k_x
                    * y_psi(psi, self.x_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_y_t**2
                    * self.k_x**2
                    * y_y_psi(psi, self.x_x_Omega, self.r)
                )

            if self.z_propagation:
                psi_new += (
                    A2_y_t
                    * self.k_z
                    * y_px_psi(
                        psi, dpsi_dr, self.z_y_Omega, self.z_py_beta, self.r
                    )
                )

                psi_new += (
                    A1_y_t
                    * A2_y_t
                    * self.k_z
                    * y_psi(psi, self.z_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_y_t**2
                    * self.k_z**2
                    * y_y_psi(psi, self.z_z_Omega, self.r)
                )

        if self.z_polarized:
            A1_z_t, A2_z_t = self.a_field_z(t)

            psi_new += A1_z_t * px_psi(
                psi, dpsi_dr, self.z_Omega, self.H_z_beta, self.r_inv
            )

            psi_new += 0.5 * A1_z_t**2 * psi

            if self.x_propagation:
                psi_new += (
                    A2_z_t
                    * self.k_x
                    * y_px_psi(
                        psi, dpsi_dr, self.x_z_Omega, self.x_pz_beta, self.r
                    )
                )

                psi_new += (
                    A1_z_t
                    * A2_z_t
                    * self.k_x
                    * y_psi(psi, self.x_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_z_t**2
                    * self.k_x**2
                    * y_y_psi(psi, self.x_x_Omega, self.r)
                )

            if self.y_propagation:
                psi_new += (
                    A2_z_t
                    * self.k_y
                    * y_px_psi(
                        psi, dpsi_dr, self.y_z_Omega, self.y_pz_beta, self.r
                    )
                )

                psi_new += (
                    A1_z_t
                    * A2_z_t
                    * self.k_y
                    * y_psi(psi, self.y_Omega, self.r)
                )
                psi_new += (
                    0.5
                    * A2_z_t**2
                    * self.k_y**2
                    * y_y_psi(psi, self.y_y_Omega, self.r)
                )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_PlaneWaveExpansion(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
        a_field_z_p,
        a_field_z_m,
        a_field2_z_p,
        a_field2_z_m,
        add_contr_funcs=[],
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            arr_contr_with_ddr_p=arr_contr_with_ddr_p,
            arr_contr_with_ddr_m=arr_contr_with_ddr_m,
            arr_contr_with_r_p=arr_contr_with_r_p,
            arr_contr_with_r_m=arr_contr_with_r_m,
            a_field_z_p=a_field_z_p,
            a_field_z_m=a_field_z_m,
            a_field2_z_p=a_field2_z_p,
            a_field2_z_m=a_field2_z_m,
        )

        self.D1 = radial_matrix_elements.D1
        self.r = radial_matrix_elements.r
        self.nr = radial_matrix_elements.nr
        self.n_lm = angular_matrix_elements.n_lm

        self.sph_jn = self.angular_matrix_elements.sph_jn
        self.sph_jn2 = self.angular_matrix_elements.sph_jn2

        self.expkr2_p = self.angular_matrix_elements("expkr2")
        self.expkr2_m = self.expkr2_p.conj()

        self.add_contr_funcs = add_contr_funcs
        if len(add_contr_funcs) > 0:
            self.add_contr = True
        else:
            self.add_contr = False


class V_psi_full_orders(V_psi_PlaneWaveExpansion):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
        a_field_z_p,
        a_field_z_m,
        a_field2_z_p,
        a_field2_z_m,
        NL,
        add_contr_funcs=[],
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
            a_field_z_p,
            a_field_z_m,
            a_field2_z_p,
            a_field2_z_m,
            add_contr_funcs=add_contr_funcs,
        )

        self.NL = NL

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        for L in range(self.NL):
            sph_jn = self.sph_jn[L, :]
            sph_jn2 = self.sph_jn2[L, :]

            dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)
            dpsi_dr_sph_jn = contract("k, Ik->Ik", sph_jn, dpsi_dr)

            arr_contr_with_ddr = (1j / 2) * (
                self.a_field_z_m(t) * self.arr_contr_with_ddr_p[L]
                + self.a_field_z_p(t) * self.arr_contr_with_ddr_m[L]
            )

            psi_new -= contract(
                "IJ, Jk->Ik", arr_contr_with_ddr, dpsi_dr_sph_jn
            )

            psi_r = contract("k, Ik->Ik", 1 / self.r, psi)
            psi_r_sph_jn = contract("k, Ik->Ik", sph_jn, psi_r)

            arr_contr_with_r = (1j / 2) * (
                self.a_field_z_m(t) * self.arr_contr_with_r_p[L]
                + self.a_field_z_p(t) * self.arr_contr_with_r_m[L]
            )

            psi_new += contract("IJ, Jk->Ik", arr_contr_with_r, psi_r_sph_jn)

            psi_sph_jn2 = contract("k, Ik->Ik", sph_jn2, psi)

            arr_contr = (1 / 8) * (
                self.a_field2_z_m(t) * self.expkr2_p[L]
                + self.a_field2_z_p(t) * self.expkr2_m[L]
            )

            psi_new += contract("IJ, Jk->Ik", arr_contr, psi_sph_jn2)

            if self.add_contr:
                for el in self.add_contr_functions:
                    psi_new += el(
                        psi,
                        dpsi_dr,
                        psi_r,
                        psi_r_sph_jn,
                        psi_sph_jn2,
                        self.angular_matrix_elements,
                        self.radial_matrix_elements,
                    )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_psi_full(V_psi_PlaneWaveExpansion):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
        a_field_z_p,
        a_field_z_m,
        a_field2_z_p,
        a_field2_z_m,
        add_contr_funcs=[],
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
            a_field_z_p,
            a_field_z_m,
            a_field2_z_p,
            a_field2_z_m,
            add_contr_funcs=add_contr_funcs,
        )

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        dpsi_dr = contract("ij, Ij->Ii", self.D1, psi)

        psi_new -= (
            (1j / 2)
            * self.a_field_z_m(t)
            * contract("IJk, Jk->Ik", self.arr_contr_with_ddr_p, dpsi_dr)
        )
        psi_new -= (
            (1j / 2)
            * self.a_field_z_p(t)
            * contract("IJk, Jk->Ik", self.arr_contr_with_ddr_m, dpsi_dr)
        )

        psi_r = contract("k, Ik->Ik", 1 / self.r, psi)

        psi_new += (
            (1j / 2)
            * self.a_field_z_m(t)
            * contract("IJk, Jk->Ik", self.arr_contr_with_r_p, psi_r)
        )
        psi_new += (
            (1j / 2)
            * self.a_field_z_p(t)
            * contract("IJk, Jk->Ik", self.arr_contr_with_r_m, psi_r)
        )

        psi_new += (
            (1 / 8)
            * self.a_field2_z_m(t)
            * contract("IJk, Jk->Ik", self.expkr2_p, psi)
        )
        psi_new += (
            (1 / 8)
            * self.a_field2_z_p(t)
            * contract("IJk, Jk->Ik", self.expkr2_m, psi)
        )

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


class V_Coulomb(VPsi):
    def __init__(
        self,
        angular_matrix_elements,
        radial_matrix_elements,
    ):
        super().__init__(
            angular_matrix_elements,
            radial_matrix_elements,
        )

        self.V = angular_matrix_elements("1/(r-a)")

    def __call__(self, psi, t, ravel=True):
        psi = psi.reshape((self.n_lm, self.nr))
        psi_new = np.zeros((self.n_lm, self.nr), dtype=np.complex128)

        psi_new -= contract("IJk, Jk->Ik", self.V, psi)

        if ravel:
            return psi_new.ravel()
        else:
            return psi_new


def setup_V_psi_PlaneWaveExpansion(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_z_p,
    a_field_z_m,
    a_field2_z_p,
    a_field2_z_m,
    polarization,
    NL=2,
    orders=True,
):
    if polarization == "x":
        (
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
        ) = get_contr_arr_x(angular_matrix_elements)
    elif polarization == "y":
        if orders:
            swapaxes = (1, 2)
        else:
            swapaxes = (0, 1)
        (
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
        ) = get_contr_arr_y(angular_matrix_elements, swapaxes=swapaxes)
    elif polarization == "z":
        (
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
        ) = get_contr_arr_z(angular_matrix_elements)
    else:
        raise NotImplementedError("")

    if orders:
        return V_psi_full_orders(
            angular_matrix_elements,
            radial_matrix_elements,
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
            a_field_z_p,
            a_field_z_m,
            a_field2_z_p,
            a_field2_z_m,
            NL,
        )
    else:
        return V_psi_full(
            angular_matrix_elements,
            radial_matrix_elements,
            arr_contr_with_ddr_p,
            arr_contr_with_ddr_m,
            arr_contr_with_r_p,
            arr_contr_with_r_m,
            a_field_z_p,
            a_field_z_m,
            a_field2_z_p,
            a_field2_z_m,
        )


def get_contr_arr_x(angular_matrix_elements):
    arr_contr_with_ddr_p = angular_matrix_elements("expkr_cosph_sinth")
    arr_contr_with_ddr_m = arr_contr_with_ddr_p.conj()
    arr_contr_with_r_p = (
        angular_matrix_elements("expkr_cosph_sinth")
        + 1j * angular_matrix_elements("expkr_m2_sinph_sinth")
        - (1 / 2) * angular_matrix_elements("expkr_c_costh_(m+1)")
        + (1 / 2) * angular_matrix_elements("expkr_c_costh_(m-1)")
    )
    arr_contr_with_r_m = arr_contr_with_r_p.conj()

    return (
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
    )


def get_contr_arr_y(angular_matrix_elements, swapaxes=(1, 2)):
    arr_contr_with_ddr_p = angular_matrix_elements("expkr_sinph_sinth")
    arr_contr_with_ddr_m = arr_contr_with_ddr_p.conj().swapaxes(
        swapaxes[0], swapaxes[1]
    )
    arr_contr_with_r_p = (
        angular_matrix_elements("expkr_sinph_sinth")
        - 1j * angular_matrix_elements("expkr_m2_cosph_sinth")
        + (1j / 2) * angular_matrix_elements("expkr_c_costh_(m+1)")
        + (1j / 2) * angular_matrix_elements("expkr_c_costh_(m-1)")
    )
    arr_contr_with_r_m = -arr_contr_with_r_p.conj()

    return (
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
    )


def get_contr_arr_z(angular_matrix_elements):
    arr_contr_with_ddr_p = angular_matrix_elements("expkr_costh")
    arr_contr_with_ddr_m = arr_contr_with_ddr_p.conj()
    arr_contr_with_r_p = (
        angular_matrix_elements("expkr_sinth_ddtheta") + arr_contr_with_ddr_p
    )
    arr_contr_with_r_m = arr_contr_with_r_p.conj()

    return (
        arr_contr_with_ddr_p,
        arr_contr_with_ddr_m,
        arr_contr_with_r_p,
        arr_contr_with_r_m,
    )


def contr_expkr_Ax(
    psi,
    dpsi_dr,
    psi_r,
    psi_r_sph_jn,
    psi_sph_jn2,
    angular_matrix_elements,
    radial_matrix_elements,
):
    arr = angular_matrix_elements("expkr_cosph_sinth")
    return contract("IJ, Jk->Ik", arr, psi_r_sph_jn)


def contr_expkr_Ay(
    psi,
    dpsi_dr,
    psi_r,
    psi_r_sph_jn,
    psi_sph_jn2,
    angular_matrix_elements,
    radial_matrix_elements,
):
    arr = angular_matrix_elements("expkr_sinph_sinth")
    return -contract("IJ, Jk->Ik", arr, psi_r_sph_jn)
