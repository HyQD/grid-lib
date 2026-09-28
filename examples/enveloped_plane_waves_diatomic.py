import numpy as np
from matplotlib import pyplot as plt
import tqdm
from opt_einsum import contract
from scipy.sparse.linalg import LinearOperator, bicgstab

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
    Linear_map,
)

from grid_lib.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_lm,
    AngularMatrixElements_lmr,
    AngularMatrixElements_lmr_Coulomb,
)
from grid_lib.spherical_coordinates.lasers import (
    square_velocity_exp_p,
    square_velocity_exp_m,
    square_velocity_exp2_p,
    square_velocity_exp2_m,
)


from grid_lib.spherical_coordinates.utils import mask_function
from grid_lib.spherical_coordinates.preconditioners import M2Psi

from grid_lib.spherical_coordinates.rhs import (
    H0Psi,
    HtPsi,
)

from grid_lib.spherical_coordinates.time_dependent_field_interaction import (
    setup_V_psi_PlaneWaveExpansion,
    V_Coulomb,
)


from grid_lib.spherical_coordinates.utils import (
    Counter,
    quadrature,
)

from grid_lib.spherical_coordinates.properties import expec_x_i, expec_p_i

from grid_lib.spherical_coordinates.ground_state import (
    compute_ground_state_diatomic,
)

### INPUTS #######################

# pulse inputs
E0 = 0.03
omega = 0.057
ncycles = 1
dt = 0.25
speed_of_light = 137
k = omega / speed_of_light

# grid inputs
N = 200
nr = N - 1
r_max = 100
l_max = 6
m_max = 1
L_max = l_max  # max l number in partial wave expansion of the Coulomb potential
a = 2.0  # H2+ bond length

### SETUP ########################

# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
D1 = radial_matrix_elements.D1
T_D2 = -(1 / 2) * radial_matrix_elements.D2

# setup angular matrix elements
angular_matrix_elements = AngularMatrixElements_lm(
    arr_to_calc=[
        "x_Omega",
        "z_Omega",
        "H_x_beta",
        "H_z_beta",
    ],
    l_max=l_max,
    m_max=m_max,
)
angular_matrix_elements_lmr = AngularMatrixElements_lmr(
    arr_to_calc=[
        "expkr_costh",
        "expkr_sinth_ddtheta",
        "expkr2",
    ],
    nr=nr,
    r=r,
    k=k,
    phi_k=0,
    theta_k=np.pi / 2,
    l_max=l_max,
    m_max=m_max,
)
angular_matrix_elements_Coulomb = AngularMatrixElements_lmr_Coulomb(
    arr_to_calc=["1/(r-a)"],
    nr=nr,
    r=r,
    l_max=l_max,
    m_max=m_max,
    a=a,
    L_max=L_max,
)
n_lm = angular_matrix_elements.n_lm

# setup mask function
mask_r = mask_function(r, r[-1], r[-1] - 30)

# Compute ground/intial state
eps, phi_n = compute_ground_state_diatomic(
    angular_matrix_elements_Coulomb, radial_matrix_elements, potential, l_max
)
# setup initial state
psi_t_temp = phi_n[:, 0] / np.sqrt(quadrature(weights, np.abs(phi_n[:, 0]) ** 2))
psi_t = np.zeros((n_lm, nr), dtype=np.complex128)
for l in range(l_max + 1):
    I = angular_matrix_elements_lmr.I_lm_[f"{l}{0}"]
    psi_t[I, :] = psi_t_temp[l * nr : (l + 1) * nr]

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

a_field_z_p = square_velocity_exp_p(field_strength=E0, omega=omega, ncycles=ncycles)
a_field_z_m = square_velocity_exp_m(field_strength=E0, omega=omega, ncycles=ncycles)

a_field2_z_p = square_velocity_exp2_p(field_strength=E0, omega=omega, ncycles=ncycles)
a_field2_z_m = square_velocity_exp2_m(field_strength=E0, omega=omega, ncycles=ncycles)


# # sampling arrays
num_steps = int(tfinal / dt) + 1

time_points = np.linspace(0, tfinal, num_steps)
expec_x = np.zeros(num_steps, dtype=np.complex128)
expec_z = np.zeros(num_steps, dtype=np.complex128)
expec_px = np.zeros(num_steps, dtype=np.complex128)
expec_pz = np.zeros(num_steps, dtype=np.complex128)
nr_its_conv = np.zeros(num_steps - 1)

# A1_x, A2_x = a_field_x(time_points)


# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi = setup_V_psi_PlaneWaveExpansion(
    angular_matrix_elements_lmr,
    radial_matrix_elements,
    a_field_z_p=a_field_z_p,
    a_field_z_m=a_field_z_m,
    a_field2_z_p=a_field2_z_p,
    a_field2_z_m=a_field2_z_m,
    polarization="z",
    orders=False,
)

V_Coulomb = V_Coulomb(angular_matrix_elements_Coulomb, radial_matrix_elements)

rhs = HtPsi(
    angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi, V_Coulomb]
)


# preconditioner
preconditioner = M2Psi(angular_matrix_elements, radial_matrix_elements, dt)
M_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=preconditioner)


# arrays needed for sampling
x_Omega = angular_matrix_elements("x_Omega")
z_Omega = angular_matrix_elements("z_Omega")
H_x_beta = angular_matrix_elements("H_x_beta")
H_z_beta = angular_matrix_elements("H_z_beta")


dpsi_t_dr = contract("ij, Ij->Ii", D1, psi_t)

expec_x[0] = expec_x_i(psi_t, weights, r, x_Omega)
expec_px[0] = expec_p_i(psi_t, dpsi_t_dr, weights, r, x_Omega, H_x_beta)

expec_z[0] = expec_x_i(psi_t, weights, r, z_Omega)
expec_pz[0] = expec_p_i(psi_t, dpsi_t_dr, weights, r, z_Omega, H_z_beta)

### RUN ##########################

for i in tqdm.tqdm(range(num_steps - 1)):
    ti = time_points[i] + dt / 2

    Ap_lambda = lambda psi, ti=ti: psi.ravel() + 1j * dt / 2 * rhs(psi, ti)
    Ap_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=Ap_lambda)
    z = psi_t.ravel() - 1j * dt / 2 * rhs(psi_t, ti)

    local_counter = Counter()
    psi_t, info = bicgstab(
        Ap_linear,
        z,
        M=M_linear,
        x0=psi_t.ravel(),
        tol=1e-8,
        callback=local_counter,
    )
    nr_its_conv[i] = local_counter.counter
    psi_t = psi_t.reshape((n_lm, nr))

    psi_t = contract("Ik, k->Ik", psi_t, mask_r)
    dpsi_t_dr = contract("ij, Ij->Ii", D1, psi_t)

    expec_x[i + 1] = expec_x_i(psi_t, weights, r, x_Omega)
    expec_px[i + 1] = expec_p_i(psi_t, dpsi_t_dr, weights, r, x_Omega, H_x_beta)

    expec_z[i + 1] = expec_x_i(psi_t, weights, r, z_Omega)
    expec_pz[i + 1] = expec_p_i(psi_t, dpsi_t_dr, weights, r, z_Omega, H_z_beta)


samples = {
    "time_points": time_points,
    "expec_x": expec_x,
    "expec_px": expec_px,
    "expec_z": expec_z,
    "expec_pz": expec_pz,
}

np.savez(f"output_enveloped_plane_waves_diatomic", **samples)
