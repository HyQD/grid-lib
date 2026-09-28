import numpy as np
import time
from matplotlib import pyplot as plt
import tqdm
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, cg, gmres, bicgstab
from opt_einsum import contract

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
)

from grid_lib.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
    AngularMatrixElements_lm,
)

from grid_lib.spherical_coordinates.lasers import (
    square_length_dipole,
    square_velocity_dipole,
)
from grid_lib.spherical_coordinates.utils import mask_function
from grid_lib.spherical_coordinates.preconditioners import M2Psi
from grid_lib.spherical_coordinates.rhs import (
    H0Psi,
    HtPsi,
)
from grid_lib.spherical_coordinates.time_dependent_field_interaction import (
    V_psi_length_z,
    V_psi_length,
    V_psi_velocity,
)
from grid_lib.spherical_coordinates.utils import (
    Counter,
    quadrature,
)
from grid_lib.spherical_coordinates.properties import (
    expec_x_i,
)
from grid_lib.spherical_coordinates.ground_state import compute_ground_state

### INPUTS #######################

# pulse inputs
E0 = 0.03
omega = 0.057
ncycles = 3
dt = 0.25

# grid inputs
N = 200
nr = N - 1
r_max = 100
l_max = 3
alpha = 0.4

### SETUP ########################
# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Rational_map(r_max=r_max, alpha=alpha))
weights = gll.weights

# setup radial matrix elements
radial_matrix_elements = RadialMatrixElements(gll)
potential = -radial_matrix_elements.r_inv
r = radial_matrix_elements.r
D1 = radial_matrix_elements.D1
T_D2 = -(1 / 2) * radial_matrix_elements.D2

# setup angular matrix elements
angular_matrix_elements = AngularMatrixElements_l(arr_to_calc=["z_Omega"], l_max=l_max)
# angular_matrix_elements = AngularMatrixElements_lm(arr_to_calc=["z_Omega"], l_max=l_max)
n_lm = angular_matrix_elements.n_lm

# setup mask function
mask_r = mask_function(r, r[-1], r[-1] - 30)

# Compute ground/intial state
tic = time.time()
eps, phi_n = compute_ground_state(
    angular_matrix_elements, radial_matrix_elements, potential
)

# setup initial state
psi_t = np.zeros((n_lm, nr), dtype=np.complex128)
psi_t[0] = np.complex128(phi_n[:, 0])
psi_t[0] /= np.sqrt(quadrature(weights, np.abs(psi_t[0]) ** 2))
psi0 = psi_t[0].copy()
toc = time.time()
print(f"Time computing initial state: {toc-tic}")

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

e_field_z = square_length_dipole(
    field_strength=E0, omega=omega, ncycles=ncycles, phase=-np.pi / 2
)


# sampling arrays
num_steps = int(tfinal / dt) + 1

time_points = np.linspace(0, tfinal, num_steps)
expec_z = np.zeros(num_steps, dtype=np.complex128)
expec_pz = np.zeros(num_steps, dtype=np.complex128)

nr_its_conv = np.zeros(num_steps - 1)


# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi = V_psi_length_z(angular_matrix_elements, radial_matrix_elements, e_field_z)

rhs = HtPsi(angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi])

# preconditioner
preconditioner = M2Psi(angular_matrix_elements, radial_matrix_elements, dt)
M_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=preconditioner)


# arrays needed for sampling
z_Omega = angular_matrix_elements("z_Omega")


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

    expec_z[i + 1] = expec_x_i(psi_t, weights, r, z_Omega)


samples = {
    "time_points": time_points,
    "expec_z": expec_z,
}

np.savez("output", **samples)
