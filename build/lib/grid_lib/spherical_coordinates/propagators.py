import numpy as np
from grid_lib.spherical_coordinates.preconditioners import M2Psi
from grid_lib.spherical_coordinates.properties import expec_x_i, expec_p_i
from scipy.sparse.linalg import LinearOperator, bicgstab
import tqdm
from grid_lib.spherical_coordinates.utils import (
    Counter,
)
from opt_einsum import contract


class Propagator:
    def __init__(self, radial_matrix_elements, angular_matrix_elements):

        self.radial_matrix_elements = radial_matrix_elements
        self.angular_matrix_elements = angular_matrix_elements
        self.nr = len(self.radial_matrix_elements.r)
        self.n_lm = self.angular_matrix_elements.n_lm

    def run(self, rhs, psi0, mask_r, tfinal, dt):
        pass


class BiCGstab(Propagator):
    def __init__(self, radial_matrix_elements, angular_matrix_elements):
        super().__init__(radial_matrix_elements, angular_matrix_elements)

    def run(self, rhs, psi0, mask_r, tfinal, dt, conv_tol=1e-8):
        psi_t = psi0.copy()
        nr, n_lm = self.nr, self.n_lm
        D1 = self.radial_matrix_elements.D1
        weights = self.radial_matrix_elements.weights
        r = self.radial_matrix_elements.r

        # sampling arrays
        num_steps = int(tfinal / dt) + 1

        time_points = np.linspace(0, tfinal, num_steps)
        expec_x = np.zeros(num_steps, dtype=np.complex128)
        expec_y = np.zeros(num_steps, dtype=np.complex128)
        expec_z = np.zeros(num_steps, dtype=np.complex128)
        expec_px = np.zeros(num_steps, dtype=np.complex128)
        expec_py = np.zeros(num_steps, dtype=np.complex128)
        expec_pz = np.zeros(num_steps, dtype=np.complex128)
        nr_its_conv = np.zeros(num_steps - 1)

        # arrays needed for sampling
        x_Omega = self.angular_matrix_elements("x_Omega")
        y_Omega = self.angular_matrix_elements("y_Omega")
        z_Omega = self.angular_matrix_elements("z_Omega")
        H_x_beta = self.angular_matrix_elements("H_x_beta")
        H_y_beta = self.angular_matrix_elements("H_y_beta")
        H_z_beta = self.angular_matrix_elements("H_z_beta")

        # preconditioner
        preconditioner = M2Psi(
            self.angular_matrix_elements, self.radial_matrix_elements, dt
        )
        M_linear = LinearOperator(
            (nr * (n_lm), nr * (n_lm)), matvec=preconditioner
        )

        ### RUN ##########################

        for i in tqdm.tqdm(range(num_steps - 1)):
            time_points[i + 1] = (i + 1) * dt
            ti = time_points[i] + dt / 2

            Ap_lambda = lambda psi, ti=ti: psi.ravel() + 1j * dt / 2 * rhs(
                psi, ti
            )
            Ap_linear = LinearOperator(
                (nr * (n_lm), nr * (n_lm)), matvec=Ap_lambda
            )
            z = psi_t.ravel() - 1j * dt / 2 * rhs(psi_t, ti)

            local_counter = Counter()
            psi_t, info = bicgstab(
                Ap_linear,
                z,
                M=M_linear,
                x0=psi_t.ravel(),
                tol=conv_tol,
                callback=local_counter,
            )
            nr_its_conv[i] = local_counter.counter
            psi_t = psi_t.reshape((n_lm, nr))

            psi_t = contract("Ik, k->Ik", psi_t, mask_r)
            dpsi_t_dr = contract("ij, Ij->Ii", D1, psi_t)

            expec_x[i + 1] = expec_x_i(psi_t, weights, r, x_Omega)
            expec_px[i + 1] = expec_p_i(
                psi_t, dpsi_t_dr, weights, r, x_Omega, H_x_beta
            )

            expec_y[i + 1] = expec_x_i(psi_t, weights, r, y_Omega)
            expec_py[i + 1] = expec_p_i(
                psi_t, dpsi_t_dr, weights, r, y_Omega, H_y_beta
            )

            expec_z[i + 1] = expec_x_i(psi_t, weights, r, z_Omega)
            expec_pz[i + 1] = expec_p_i(
                psi_t, dpsi_t_dr, weights, r, z_Omega, H_z_beta
            )

        samples = {
            "time_points": time_points,
            "expec_x": expec_x,
            "expec_px": expec_px,
            "expec_y": expec_y,
            "expec_py": expec_py,
            "expec_z": expec_z,
            "expec_pz": expec_pz,
        }

        return samples
