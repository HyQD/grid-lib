import argparse
import time
import warnings

import numpy as np
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, bicgstab

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_lm,
)
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    number_of_lm_states,
    setup_y_and_ybar_sympy,
)
from grid_lib.spherical_coordinates.angular_momentum_operators import (
    OrbitalAngularMomentum,
)
from grid_lib.spherical_coordinates.potentials import (
    clamped_molecular_potential_quadrature,
    clamped_molecular_potential_Poisson,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb
from grid_lib.spherical_coordinates.utils import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Imaginary-time propagation for the H2+ ground state in a "
            "Cartesian-axis magnetic field."
        )
    )
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=0)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--element-size", type=float, default=4.0)
    parser.add_argument("--n-points-per-element", type=int, default=11)
    parser.add_argument("--B0", type=float, default=1.0)
    parser.add_argument(
        "--B-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Magnetic-field axis. Only Cartesian axes are supported.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=2.0,
        help="Internuclear distance. Nuclei are placed at z = +/- R/2.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--conv-tol-E", type=float, default=1e-12)
    parser.add_argument("--conv-tol-psi", type=float, default=1e-10)
    return parser.parse_args()


def compute_h_ap(A, H, l_max, m_max):
    h_ap = np.zeros(A.shape, dtype=A.dtype)

    for m in range(-m_max, m_max + 1):
        for l in range(abs(m), l_max + 1):
            I_lm = LM_to_I(l, m, l_max, m_max)
            h_ap[:, :, I_lm] += contract("ab, pb->pa", H[l], A[:, :, I_lm])

    return h_ap


def get_axis_diamagnetic_matrix(axis, matrices):
    if axis == "x":
        return matrices["y_y"] + matrices["z_z"]
    if axis == "y":
        return matrices["x_x"] + matrices["z_z"]
    return matrices["x_x"] + matrices["y_y"]


class Hamiltonian:
    def __init__(
        self,
        Te,
        y_bar,
        V_molecular,
        l_max,
        m_max,
        weights,
        r,
        diamagnetic_lm,
        B0=1.0,
        B_axis="z",
    ):
        self.Te = Te
        self.y_bar = y_bar
        self.V_molecular = V_molecular
        self.l_max = l_max
        self.m_max = m_max
        self.orb_amo = OrbitalAngularMomentum(l_max, m_max, weights)
        self.r = r
        self.diamagnetic_lm = diamagnetic_lm
        self.B0 = B0
        self.B_axis = B_axis

    def __call__(self, psi, t):
        Te_psi = compute_h_ap(psi, self.Te, self.l_max, self.m_max)
        V_molecular_psi = contract(
            "JIK, Ja, paK->paI", self.y_bar, self.V_molecular, psi
        )
        H_psi = Te_psi + V_molecular_psi

        if self.B_axis == "x":
            H_psi += 0.5 * self.B0 * self.orb_amo.lx(psi)
        elif self.B_axis == "y":
            H_psi += 0.5 * self.B0 * self.orb_amo.ly(psi)
        else:
            H_psi += 0.5 * self.B0 * self.orb_amo.lz(psi)

        H_psi += (
            self.B0**2
            / 8
            * contract("IJ, paJ, a->paI", self.diamagnetic_lm, psi, self.r**2)
        )

        return H_psi


class PreCond:
    def __init__(
        self,
        Hl,
        dt,
        n_active,
        l_max,
        m_max,
        imag_time=False,
        dtype=np.complex128,
    ):
        self.nr = Hl.shape[1]
        self.Ml_inv = np.zeros(Hl.shape, dtype=dtype)
        self.Identity = np.eye(self.nr, dtype=dtype)
        self.dt = dt

        for l in range(Hl.shape[0]):
            if imag_time:
                self.Ml_inv[l] = np.linalg.inv(
                    self.Identity + 0.5 * self.dt * Hl[l]
                )
            else:
                self.Ml_inv[l] = np.linalg.inv(
                    self.Identity + 0.5j * self.dt * Hl[l]
                )

        self.l_max = l_max
        self.m_max = m_max
        self.n_lm = number_of_lm_states(l_max, m_max)
        self.n_active = n_active

    def __call__(self, A):
        A = A.reshape((self.n_active, self.nr, self.n_lm))
        M_ap = np.zeros(A.shape, dtype=A.dtype)

        for m in range(-self.m_max, self.m_max + 1):
            for l in range(abs(m), self.l_max + 1):
                I_lm = LM_to_I(l, m, self.l_max, self.m_max)
                M_ap[:, :, I_lm] += contract(
                    "ab, pb->pa", self.Ml_inv[l], A[:, :, I_lm]
                )

        return M_ap.ravel()


def main():
    args = parse_args()

    l_max = args.l_max
    m_max = args.m_max
    n_l = l_max + 1
    n_lm = number_of_lm_states(l_max, m_max)

    if args.B_axis != "z" and m_max == 0:
        print(
            "Warning: m_max=0 is only sensible for a magnetic field along z. "
            "Proceeding anyway."
        )

    L_max = 2 * l_max
    M_max = 2 * m_max

    positions = np.array(
        [
            [0.0, 0.0, -args.R / 2],
            [0.0, 0.0, args.R / 2],
        ]
    )
    charges = np.array([1.0, 1.0])

    print(f"R: {args.R:.6f}")
    print(f"B0: {args.B0:.6f}")
    print(f"B-axis: {args.B_axis}")
    print(f"l_max: {l_max}, m_max: {m_max}")

    r_start = 0.0
    nodes = np.arange(r_start, args.r_max + args.element_size, args.element_size)
    n_points = np.ones((len(nodes) - 1,), dtype=int) * args.n_points_per_element

    GLL_r = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

    D2 = GLL_r.D2[1:-1, 1:-1]
    r = GLL_r.r[1:-1]
    weights = GLL_r.weights[1:-1]
    n_r = len(r)

    nabla_sq = np.zeros((n_l, n_r, n_r))
    Te = np.zeros((n_l, n_r, n_r))

    for l in range(l_max + 1):
        nabla_sq[l] = D2 - np.diag(l * (l + 1) / r**2)
        Te[l] = -0.5 * nabla_sq[l]

    _, y_bar = setup_y_and_ybar_sympy(l_max, m_max, L_max, M_max)

    W = radial_Coulomb(GLL_r, L_max + 1)
    V_molecular = clamped_molecular_potential_Poisson(
        r,
        W,
        weights,
        positions=positions,
        charges=charges,
        L_max=L_max,
        M_max=M_max,
        force=False,
    )
    # stop
    ame = AngularMatrixElements_lm(
        arr_to_calc=["x_x_Omega", "y_y_Omega", "z_z_Omega"],
        l_max=l_max,
        m_max=m_max,
    )

    matrices = {}
    for arr_lm_key, short_name in (
        ("x_x_Omega", "x_x"),
        ("y_y_Omega", "y_y"),
        ("z_z_Omega", "z_z"),
    ):
        arr = np.zeros((n_lm, n_lm), dtype=np.complex128)
        lm_to_index = ame.I_lm
        for m1 in range(-m_max, m_max + 1):
            for l1 in range(abs(m1), l_max + 1):
                i = LM_to_I(l1, m1, l_max, m_max)
                i_ame = lm_to_index[f"{l1}{m1}"]
                for m2 in range(-m_max, m_max + 1):
                    for l2 in range(abs(m2), l_max + 1):
                        j = LM_to_I(l2, m2, l_max, m_max)
                        j_ame = lm_to_index[f"{l2}{m2}"]
                        arr[i, j] = ame.arr[arr_lm_key][i_ame, j_ame]
        matrices[short_name] = arr

    diamagnetic_lm = get_axis_diamagnetic_matrix(args.B_axis, matrices)

    H0 = Hamiltonian(
        Te=Te,
        y_bar=y_bar,
        V_molecular=V_molecular,
        l_max=l_max,
        m_max=m_max,
        weights=weights,
        r=r,
        diamagnetic_lm=diamagnetic_lm,
        B0=args.B0,
        B_axis=args.B_axis,
    )

    psi = np.zeros((1, n_r, n_lm), dtype=np.complex128)
    psi += (np.random.random((1, n_r, n_lm)) - 0.5) * 1e-3
    psi[0, :, LM_to_I(0, 0, l_max, m_max)] += np.exp(-r)

    norm_psi0 = contract("paI, paI, a->", psi.conj(), psi, weights)
    psi /= np.sqrt(norm_psi0)

    Hpsi = H0(psi, t=0.0)
    energy_old = contract("paI, paI, a->", psi.conj(), Hpsi, weights)
    total_energy_old = energy_old + 1.0 / args.R

    print(f"Initial electronic energy: {energy_old.real:.12f}")
    print(f"Initial total energy:      {total_energy_old.real:.12f}")

    precond = PreCond(
        Hl=Te,
        dt=args.dt,
        n_active=1,
        l_max=l_max,
        m_max=m_max,
        imag_time=True,
        dtype=psi.dtype,
    )
    M_linear = LinearOperator(
        (psi.size, psi.size), matvec=precond, dtype=psi.dtype
    )

    psi_old = psi.copy()
    converged = False

    for n in range(args.num_steps):
        z = psi - 0.5 * args.dt * H0(psi, t=0.0)

        A_linear = LinearOperator(
            (psi.size, psi.size),
            matvec=lambda psi_in: (
                psi_in
                + 0.5 * args.dt * H0(psi_in.reshape(psi.shape), t=0.0).ravel()
            ),
            dtype=psi.dtype,
        )

        local_counter = Counter()
        tic = time.time()
        psi_new, info = bicgstab(
            A_linear,
            z.ravel(),
            x0=psi.ravel(),
            atol=1e-12,
            callback=local_counter,
            M=M_linear,
            rtol=0.0,
        )
        toc = time.time()

        if info != 0:
            print(
                f"Warning: bicgstab did not converge at step {n + 1}, info: {info}"
            )

        psi = psi_new.reshape(psi.shape)
        norm_psi = contract("paI, paI, a->", psi.conj(), psi, weights)
        psi /= np.sqrt(norm_psi)

        Hpsi = H0(psi, t=0.0)
        energy = contract("paI, paI, a->", psi.conj(), Hpsi, weights)
        total_energy = energy + 1.0 / args.R

        T_psi = compute_h_ap(psi, Te, l_max, m_max)
        energy_T = contract("paI, paI, a->", psi.conj(), T_psi, weights)

        dE = energy - energy_old
        dpsi = np.linalg.norm(psi.ravel() - psi_old.ravel(), ord=np.inf)

        energy_old = energy
        psi_old = psi.copy()

        print(
            f"Step {n + 1:5d}, "
            f"E_elec: {energy.real:.12f}, "
            f"E_tot: {total_energy.real:.12f}, "
            f"E_tot(Zeeman): {(total_energy - 0.5 * args.B0).real:.12f},"
            f"E_kinetic: {energy_T.real:.12f}, "
            f"dE: {dE.real:.3e}, "
            f"dpsi: {dpsi:.3e}, "
            f"bicgstab_time: {toc - tic:.3f}s"
        )

        if np.abs(dE) < args.conv_tol_E and dpsi < args.conv_tol_psi:
            print(f"Converged at step {n + 1}")
            converged = True
            break

    if not converged:
        print(
            "Warning: imaginary-time propagation did not satisfy convergence tolerances."
        )

    final_Hpsi = H0(psi, t=0.0)
    final_energy = contract("paI, paI, a->", psi.conj(), final_Hpsi, weights)
    final_total_energy = final_energy + 1.0 / args.R

    print()
    print(f"Final electronic energy: {final_energy.real:.12f}")
    print(f"Final total energy:      {final_total_energy.real:.12f}")

    dat = {
        "r": r,
        "psi": psi,
        "energy": energy,
        "total_energy": total_energy,
    }
    np.savez(f"h2p_magnetic_imag_time_B0={args.B0}_Bdir={args.B_axis}_lmax={l_max}_mmax={m_max}.npz", **dat)


if __name__ == "__main__":
    main()
