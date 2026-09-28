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
from grid_lib.spherical_coordinates.propagators import BiCGstab

### INPUTS #######################

# pulse inputs
E0_z = 0.01
E0_x = 0.002
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


### SETUP ########################

# setup Legendre-Lobatto grid
gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
weights = gll.weights
print(weights)

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
        "z_Omega",
        "x_x_Omega",
        "y_x_Omega",
        "z_x_Omega",
        "z_y_Omega",
        "y_y_Omega",
        "z_z_Omega",
        "H_x_beta",
        "H_y_beta",
        "H_z_beta",
        "x_py_beta",
        "x_pz_beta",
        "y_px_beta",
        "y_pz_beta",
        "z_px_beta",
        "z_py_beta",
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
psi_t0 = np.zeros((n_lm, nr), dtype=np.complex128)
psi_t0[0] = np.complex128(phi_n[:, 0])
psi_t0[0] /= np.sqrt(quadrature(weights, np.abs(psi_t0[0]) ** 2))

# setup pulses
t_cycle = 2 * np.pi / omega
tfinal = ncycles * t_cycle

a_field_z = square_velocity_first(
    field_strength=E0_z, omega=omega, ncycles=ncycles, phase=-np.pi / 2
)

a_field_x = square_velocity_first(
    field_strength=E0_x, omega=2 * omega, ncycles=2 * ncycles, phase=-np.pi / 2
)

# right-hand side
H0_psi = H0Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    potential,
)

Vt_psi_Ez_ky = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_z=a_field_z,
    k_y=k_y,
)

Vt_psi_Ex_ky = V_psi_velocity_first(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_x=a_field_x,
    k_y=k_y,
)

rhs = HtPsi(
    angular_matrix_elements,
    radial_matrix_elements,
    H0_psi,
    [Vt_psi_Ez_ky, Vt_psi_Ex_ky],
)
propagator = BiCGstab(radial_matrix_elements, angular_matrix_elements)
conv_tol = 1e-8

dat = propagator.run(rhs, psi_t0, mask_r, tfinal, dt, conv_tol=conv_tol)

time_points = dat["time_points"]
expec_x = dat["expec_x"]
expec_y = dat["expec_y"]
expec_z = dat["expec_z"]

from matplotlib import pyplot as plt

plt.figure()
plt.subplot(311)
plt.plot(time_points, expec_x.real)
plt.subplot(312)
plt.plot(time_points, expec_y.real)
plt.subplot(313)
plt.plot(time_points, expec_z.real)
plt.show()
