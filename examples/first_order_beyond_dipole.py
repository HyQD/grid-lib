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
)
from grid_lib.spherical_coordinates.lasers import (
    square_velocity_dipole,
    square_velocity_first,
)


from grid_lib.spherical_coordinates.utils import mask_function
from grid_lib.spherical_coordinates.preconditioners import M2Psi

from grid_lib.spherical_coordinates.rhs import (
    H0Psi,
    HtPsi,
)

from grid_lib.spherical_coordinates.time_dependent_field_interaction import (
    V_psi_velocity_z,
    V_psi_velocity_first,
)


from grid_lib.spherical_coordinates.utils import (
    Counter,
    quadrature,
)

from grid_lib.spherical_coordinates.properties import expec_x_i, expec_p_i

from grid_lib.spherical_coordinates.ground_state import compute_ground_state

### INPUTS #######################

# pulse inputs
E0 = 0.03
omega = 0.057
ncycles = 1
dt = 0.25
speed_of_light = 137
k_y = omega / speed_of_light

# grid inputs
N = 200
nr = N - 1
r_max = 100
l_max = 3
alpha = 0.4

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
        "y_Omega",
        "x_x_Omega",
        "y_x_Omega",
        "y_y_Omega",
        "H_x_beta",
        "H_y_beta",
        "y_px_beta",
        "x_py_beta",
    ],
    l_max=l_max,
)
n_lm = angular_matrix_elements.n_lm

# setup mask function
mask_r = mask_function(r, r[-1], r[-1] - 30)

# Compute ground/intial state
eps, phi_n = compute_ground_state(
    angular_matrix_elements, radial_matrix_elements, potential
)

# setup initial state
psi_t = np.zeros((n_lm, nr), dtype=np.complex128)
psi_t[0] = np.complex128(phi_n[:, 0])
psi_t[0] /= np.sqrt(quadrature(weights, np.abs(psi_t[0]) ** 2))
psi0 = psi_t[0].copy()

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

a_field_x = square_velocity_first(
    field_strength=E0, omega=omega, ncycles=ncycles, phase=-np.pi / 2
)


# sampling arrays
num_steps = int(tfinal / dt) + 1

time_points = np.linspace(0, tfinal, num_steps)
expec_x = np.zeros(num_steps, dtype=np.complex128)
expec_y = np.zeros(num_steps, dtype=np.complex128)
expec_px = np.zeros(num_steps, dtype=np.complex128)
expec_py = np.zeros(num_steps, dtype=np.complex128)
nr_its_conv = np.zeros(num_steps - 1)

A1_x, A2_x = a_field_x(time_points)


# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_x=a_field_x,
    k_y=k_y,
)

rhs = HtPsi(angular_matrix_elements, radial_matrix_elements, H0_psi, [Vt_psi])


# preconditioner
preconditioner = M2Psi(angular_matrix_elements, radial_matrix_elements, dt)
M_linear = LinearOperator((nr * (n_lm), nr * (n_lm)), matvec=preconditioner)


# arrays needed for sampling
x_Omega = angular_matrix_elements("x_Omega")
y_Omega = angular_matrix_elements("y_Omega")
H_x_beta = angular_matrix_elements("H_x_beta")
H_y_beta = angular_matrix_elements("H_y_beta")


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

    expec_y[i + 1] = expec_x_i(psi_t, weights, r, y_Omega)
    expec_py[i + 1] = expec_p_i(psi_t, dpsi_t_dr, weights, r, y_Omega, H_y_beta)


from matplotlib import pyplot as plt

plt.figure()
plt.plot(time_points, expec_x.real)

plt.figure()
plt.plot(time_points, expec_px.real)

plt.show()
# samples = {
#    "time_points": time_points,
#    "expec_x": expec_x,
#    "expec_px": expec_px,
#    "expec_y": expec_y,
#    "expec_py": expec_py,
# }

# np.savez("output_velocity_first_order", **samples)
