import numpy as np
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import AngularMatrixElements_lm
from grid_lib.spherical_coordinates.angular_momentum_operators import OrbitalAngularMomentum
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from opt_einsum import contract
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb
from grid_lib.spherical_coordinates.utils import Counter
import time
import grid_lib as gb
print(f"grid_lib version: {gb.__version__}")


from scipy.sparse.linalg import bicgstab, LinearOperator

from scipy.special import sph_harm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib import pyplot as plt

l_max = 2
m_max = 0
n_l = l_max + 1
n_lm = number_of_lm_states(l_max, m_max)

L_max = 2 * l_max
M_max = 0

r_start = 0.0
r_max = 24.0
element_size = 4.0

nodes = np.arange(r_start, r_max + element_size, element_size)
print(f"Nodes: {nodes}")

n_points_pr_element = 11
n_points = (
    np.ones((len(nodes) - 1,), dtype=int) * n_points_pr_element
)  # Example number of points per element

GLL_r = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

weights = GLL_r.weights[1:-1]  # Exclude boundary points
D1 = GLL_r.D1[1:-1, 1:-1]  # Exclude boundary points
D2 = GLL_r.D2[1:-1, 1:-1]  # Exclude boundary points

r = GLL_r.r[1:-1]
n_r = len(r)

nabla_sq = np.zeros((n_l, n_r, n_r))
He = np.zeros((n_l, n_r, n_r))

for l in range(l_max + 1):
    nabla_sq[l] = D2 - np.diag(l * (l + 1) / r**2)
    He[l] = -0.5 * nabla_sq[l] - np.diag(1 / r)


eps0, psi0 = np.linalg.eig(He[0])
idx = np.argsort(eps0)
eps0 = eps0[idx]
psi0 = psi0[:, idx]
print(f"eps0: {eps0[0]:.6f}")

orb_amo = OrbitalAngularMomentum(l_max, m_max, weights)

ame = AngularMatrixElements_lm(arr_to_calc=["x_x_Omega", "y_y_Omega", "z_z_Omega", "H_Bz_Omega"], l_max=l_max, m_max=m_max)

lm_I_ame = ame.lm_I
I_lm_ame = ame.I_lm

x2_lm = ame.arr["x_x_Omega"]
y2_lm = ame.arr["y_y_Omega"]
z2_lm = ame.arr["z_z_Omega"]
H_Bz_lm = ame.arr["H_Bz_Omega"]


x2_lm_v2 = np.zeros((n_lm, n_lm), dtype=np.complex128)
y2_lm_v2 = np.zeros((n_lm, n_lm), dtype=np.complex128)
z2_lm_v2 = np.zeros((n_lm, n_lm), dtype=np.complex128)
for m1 in range(-m_max, m_max + 1):
    for l1 in range(abs(m1), l_max + 1):
        I = LM_to_I(l1, m1, l_max, m_max)
        I_ame = I_lm_ame[f"{l1}{m1}"]
        for m2 in range(-m_max, m_max + 1):
            for l2 in range(abs(m2), l_max + 1):
                J = LM_to_I(l2, m2, l_max, m_max)
                J_ame = I_lm_ame[f"{l2}{m2}"]
                x2_lm_v2[I, J] = x2_lm[I_ame, J_ame]
                y2_lm_v2[I, J] = y2_lm[I_ame, J_ame]
                z2_lm_v2[I, J] = z2_lm[I_ame, J_ame]



x2_plus_y2_lm = x2_lm_v2 + y2_lm_v2
y2_plus_z2_lm = y2_lm_v2 + z2_lm_v2
x2_plus_z2_lm = x2_lm_v2 + z2_lm_v2

def compute_h_ap(A, H, l_max, m_max):

    h_ap = np.zeros(A.shape, dtype=A.dtype)

    for m in range(-m_max, m_max + 1):
        for l in range(abs(m), l_max + 1):
            I_lm = LM_to_I(l, m, l_max, m_max)
            h_ap[:, :, I_lm] += contract("ab, pb->pa", H[l], A[:, :, I_lm])

    return h_ap



class Hamiltonian:
    
    def __init__(self, He, l_max, m_max, weights, B0=1.0, B_dir=[0,0,1]):
        self.He = He
        self.orb_amo = OrbitalAngularMomentum(l_max, m_max, weights)
        self.B0 = B0
        self.B_dir = B_dir
    
    def __call__(self, psi, t):

        He_psi = compute_h_ap(psi, self.He, l_max, m_max)
        
        if self.B_dir[0] != 0:
            He_psi += self.orb_amo.lx(psi)* self.B0 / 2
            He_psi += self.B0**2 / 8 * contract("IJ, paJ, a->paI", y2_plus_z2_lm, psi, r**2)
        
        elif self.B_dir[1] != 0:
            He_psi += self.orb_amo.ly(psi) * self.B0 / 2
            He_psi += self.B0**2 / 8 * contract("IJ, paJ, a->paI", x2_plus_z2_lm, psi, r**2)

        elif self.B_dir[2] != 0:
            He_psi += self.orb_amo.lz(psi) * self.B0 / 2
            He_psi += self.B0**2 / 8 * contract("IJ, paJ, a->paI", x2_plus_y2_lm, psi, r**2)
        
        return He_psi
    

B_dir = [0, 1, 1]
norm_B = B_dir[0]**2 + B_dir[1]**2 + B_dir[2]**2
B_dir = [B_dir[i] / np.sqrt(norm_B) for i in range(3)]

B0 = 0.0

# Do imaginary time evolution to find the ground state
H0 = Hamiltonian(He=He, l_max=l_max, m_max=m_max, weights=weights, B0=B0, B_dir=B_dir)
psi = np.zeros((1, n_r, n_lm), dtype=np.complex128)  # Start with a random guess
psi[0, :, LM_to_I(0,0, l_max, m_max)] = np.complex128(psi0[:, 0].copy())  # Start with the lowest energy state of the radial Hamiltonian for l=0

norm_psi0 = contract("paI, paI, a->", psi.conj(), psi, weights)
print(f"Norm of psi0: {norm_psi0:.6f}")
psi /= np.sqrt(norm_psi0)



dt = 0.01
num_steps = int(2e4)

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
        tic = time.time()
        for m in range(-self.m_max, self.m_max + 1):
            for l in range(abs(m), self.l_max + 1):
                I_lm = LM_to_I(l, m, self.l_max, self.m_max)
                M_ap[:, :, I_lm] += contract(
                    "ab, pb->pa", self.Ml_inv[l], A[:, :, I_lm]
                )
        toc = time.time()
        # print(f"Time apply PreCond: {toc - tic:.6f} seconds")
        return M_ap.ravel()

precond = PreCond(Hl=He, dt=dt, n_active=1, l_max=l_max, m_max=m_max, imag_time=True)
M_linear = LinearOperator(
    (psi.size, psi.size), matvec=precond, dtype=psi.dtype
)

psi_old = psi.copy()
Hpsi = H0(psi, t=0)
energy_old = contract("paI, paI, a->", psi.conj(), Hpsi, weights)
print(f"Initial energy: {energy_old:.6f}")

conv_tol_E = 1e-12
conv_tol_psi = 1e-10

converged = False
for n in range(num_steps - 1):

    z = psi - 0.5 * dt * H0(psi, t=0)

    H_linear = LinearOperator(
        (psi.size, psi.size),
        matvec=lambda x: H0(x.reshape(psi.shape), t=0).flatten(),
        dtype=psi.dtype,
    )
    A_lambda = (
        lambda psi_in: psi_in
        + 0.5 * dt * H0(psi_in.reshape(psi.shape), t=0).flatten()
    )
    A_linear = LinearOperator(
        (psi.size, psi.size), matvec=A_lambda, dtype=psi.dtype
    )

    local_counter = Counter()
    tic = time.time()
    psi_new, info = bicgstab(
        A_linear,
        z.flatten(),
        x0=psi.flatten(),
        atol=1e-12,
        callback=local_counter,
        M=M_linear,
        rtol=0.0,
    )
    toc = time.time()
    time_bicgstab = toc - tic
    if info != 0:
        print(f"Warning: CG did not converge at step {n+1}, info: {info}")

    psi = psi_new.reshape(psi.shape)
    norm_psi = contract("paI, paI, a->", psi.conj(), psi, weights)
    psi /= np.sqrt(norm_psi)
    
    Hpsi = H0(psi, t=0)
    energy = contract("paI, paI, a->", psi.conj(), Hpsi, weights)
    dE = energy - energy_old
    energy_old = energy

    dpsi = np.linalg.norm(psi.ravel() - psi_old.ravel(), ord=np.inf)
    psi_old = psi.copy()
    e_total = energy 

    print(
        f"Step {n+1}, E(electronic): {energy.real:.6f}, dE: {dE.real:.2e}, dpsi: {dpsi:.2e}"
    )
    
    if np.abs(dE) < conv_tol_E and dpsi < conv_tol_psi:
        print(
            f"Converged at step {n+1} with energy {energy:.3f} to conv_tol_E: {conv_tol_E:.2g} and conv_tol_psi: {conv_tol_psi:.2g}"
        )
        converged = True
        break

dat = dict()
dat["r"] = r
dat["psi"] = psi
dat["energy"] = energy

np.savez(f"gs_H_B0={B0}_Bdir={B_dir}_rmax={r_max}_lmax={l_max}_mmax={m_max}_npe={n_points_pr_element}.npz", **dat)