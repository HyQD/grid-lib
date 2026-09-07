import numpy as np
import math
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from utils import TDMAsolver, Tridiag, round_down

from opt_einsum import contract

from scipy.integrate import simps, trapz

import time
import tqdm

from scipy.sparse.linalg import LinearOperator, bicgstab

from grid_lib.spherical_coordinates.utils import Counter
from grid_lib.spherical_coordinates.lasers import (
    square_length_dipole,
    square_velocity_dipole,
)

### inputs grid/stationary states ##
dr = 0.5
dz = 0.5

Lr = 30.0
Lz = 30.0

n_eig = 10

gauge = "velocity"

zgrid_parity = "even"


### defining grid ##
assert (zgrid_parity == "odd") or (
    zgrid_parity == "even"
), "zgrid_parity must be odd or even"

r = dr * (np.arange(round_down(Lr / dr + 1)) + 1 / 2)
if zgrid_parity == "odd":
    z = dz * np.arange(-math.ceil(Lz / dz), math.ceil(Lz / dz) + 1)
elif zgrid_parity == "even":
    z = dz * (np.arange(-round_down(Lz / dz + 1), round_down(Lz / dz + 1)) + 1 / 2)

Nr = len(r)
Nz = len(z)

print("Nr:", Nr)
print("Nz:", Nz)

delta_r = r[1] - r[0]
delta_z = z[1] - z[0]


### setup grid arrays (n_r x n_z) ##
V_grid = np.zeros((Nr, Nz))

for i in range(0, Nr):
    for j in range(0, Nz):
        V_grid[i, j] = -1 / (np.sqrt(r[i] ** 2 + z[j] ** 2))

### setup full hamiltonian ( n_r*n_z x n_r*n_z ) ##
n = Nr * Nz

print("mat dim:", n)

a = 0.5
h_diag_rz = a * 2 * np.ones(n) * (1 / delta_r**2 + 1 / delta_z**2) + V_grid.flatten(
    "F"
)
h_off_rz = -a * (np.ones(n - 1) / (delta_r**2))
h_off_off_rz = -a * np.ones(n - Nr) / (delta_z**2)


ii = 1
for i in range(1, n - 1):
    if i % Nr == 0:
        h_off_rz[i - 1] = 0
        ii = 1
    else:
        h_off_rz[i - 1] *= (r[ii - 1] + delta_r / 2) / np.sqrt(r[ii - 1] * r[ii])
        ii += 1

h_mat_rz_sparse = scipy.sparse.diags(
    [h_diag_rz, h_off_rz, h_off_rz, h_off_off_rz, h_off_off_rz],
    offsets=[0, -1, 1, -Nr, Nr],
)


t0 = time.time()
epsilon, phi = scipy.sparse.linalg.eigsh(h_mat_rz_sparse, k=n_eig, which="SA")
t1 = time.time()
print("Energies:")
print((np.sort(epsilon).real)[:5])


#### TIME PROPAGATION ###

# Time and pulse inputs
E0 = 0.03
dt = 0.25
omega = 0.057
ncycles = 1
time_after_pulse = 0

td = ncycles * 2 * np.pi / omega
tot_time = td + time_after_pulse
nt = int(tot_time / dt)

num_steps = int(tot_time / dt) + 1


print("nt:", nt)

# matrices for the time propagation
h_diag_zr = (
    a * 2 * np.ones(n) * (1 / delta_r**2 + 1 / delta_z**2)
)  # + V_grid.flatten("C")
h_off_zr = -a * (np.ones(n - 1) / (delta_z**2))

for i in range(1, n - 1):
    if i % Nz == 0:
        h_off_zr[i - 1] = 0

psi_t = (phi[:, 0]).astype(np.complex128).reshape(Nz, Nr).T
psi_t = psi_t * 1 / np.sqrt(trapz(trapz(psi_t.conj() * psi_t, dx=delta_z), dx=delta_r))


ones_k0_z = np.ones(Nz, dtype=np.complex128)
ones_k1_z = np.ones(Nz - 1, dtype=np.complex128)

d_dz = (np.diag(-ones_k1_z, k=-1) + np.diag(ones_k1_z, k=1)) / (2 * delta_z)
d2_dz2 = (
    np.diag(ones_k1_z, k=-1) - np.diag(2 * ones_k0_z) + np.diag(ones_k1_z, k=1)
) / (delta_z**2)
pz = -1j * d_dz

ones_k0_r = np.ones(Nr, dtype=np.complex128)
ones_k1_r = np.ones(Nr - 1, dtype=np.complex128)

d_dr = (np.diag(-ones_k1_r, k=-1) + np.diag(ones_k1_r, k=1)) / (2 * delta_r)

d2_dr2_diag = -2 * ones_k0_r / (delta_r**2)
d2_dr2_off = ones_k1_r / (delta_r**2)

for ii in range(1, Nr - 1):
    d2_dr2_off[ii - 1] *= (r[ii - 1] + delta_r / 2) / np.sqrt(r[ii - 1] * r[ii])

d2_dr2 = np.diag(d2_dr2_off, k=-1) + np.diag(d2_dr2_diag) + np.diag(d2_dr2_off, k=1)

# sample arrays
time_points = np.linspace(0, tot_time, nt)
norm = np.zeros(nt, dtype=np.complex128)
expec_z = np.zeros(nt, dtype=np.complex128)
nr_its_conv = np.zeros(num_steps - 1)


def preconditioner(psi):
    return psi


# input ()
if gauge == "length":
    pulse_z = square_length_dipole(
        field_strength=E0, omega=omega, ncycles=ncycles, phase=-np.pi / 2
    )
elif gauge == "velocity":
    pulse_z = square_velocity_dipole(
        field_strength=E0, omega=omega, ncycles=ncycles, phase=-np.pi / 2
    )


M_linear = LinearOperator((Nz * Nr, Nz * Nr), matvec=preconditioner)


def rhs(psi, t, ravel=True):
    psi = psi.reshape((Nr, Nz))
    psi_new = np.zeros((Nr, Nz), dtype=np.complex128)

    psi_new += -(1 / 2) * contract("Ai,iB->AB", d2_dr2, psi)
    psi_new += -(1 / 2) * contract("Ai,Bi->BA", d2_dz2, psi)
    psi_new += contract("AB,AB->AB", V_grid, psi)

    if gauge == "length":
        psi_new += pulse_z(t) * contract("AB,B->AB", psi, z)
    elif gauge == "velocity":
        psi_new -= 1j * pulse_z(t) * contract("AB,CB->AC", psi, d_dz)

    if ravel:
        return psi_new.ravel()
    else:
        return psi_new


for i in tqdm.tqdm(range(num_steps - 1)):
    ti = time_points[i] + dt / 2

    Ap_lambda = lambda psi__, ti=ti: psi__.ravel() + 1j * dt / 2 * rhs(psi__, ti)
    Ap_linear = LinearOperator((Nz * Nr, Nz * Nr), matvec=Ap_lambda)
    temp = psi_t.ravel() - 1j * dt / 2 * rhs(psi_t, ti)

    local_counter = Counter()
    psi_t, info = bicgstab(
        Ap_linear,
        temp,
        M=M_linear,
        x0=psi_t.ravel(),
        tol=1e-8,
        callback=local_counter,
    )
    nr_its_conv[i] = local_counter.counter
    psi_t = psi_t.reshape((Nr, Nz))

    norm[i] = trapz(trapz(psi_t.conj() * psi_t, dx=delta_z), dx=delta_r)
    expec_z[i] = trapz(
        trapz(psi_t.conj() * contract("B,AB->AB", z, psi_t), dx=delta_z), dx=delta_r
    )


samples = {
    "time_points": time_points,
    "expec_z": expec_z,
    "norm": norm,
}

np.savez(f"output_dipole_approx_{gauge}", **samples)
