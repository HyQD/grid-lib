import numpy as np
import math
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time
from utils import TDMAsolver, Tridiag, round_down

from scipy.integrate import simps, trapz

from lasers_AE import sine_square_A_velocity, sine_square_A_length

import time


### inputs grid/stationary states ##
dr = 0.1
dz = 0.1

Lr = 10.0
Lz = 10.0

n_eig = 1

zgrid_parity = "even"

w = 1  # harmonic oscillator frequency


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
r_grid = np.zeros((Nr, Nz))
z_grid = np.zeros((Nr, Nz))


for i in range(0, Nr):
    for j in range(0, Nz):
        V_grid[i, j] = -1 / (np.sqrt(r[i] ** 2 + z[j] ** 2))
        # V_grid[i,j] = -1/(np.sqrt(r[i]**2 + (z[j]-1)**2))-1/(np.sqrt(r[i]**2 + (z[j]+1)**2))
        r_grid[i, j] = r[i]
        z_grid[i, j] = z[j]
        # V_grid[i,j] = (1/2)*w*(r[i]**2 + z[j]**2)


### setup full hamiltonian ( n_r*n_z x n_r*n_z ) ##
n = Nr * Nz

print("mat dim:", n)

a = 0.5
h_diag_rz = a * 2 * np.ones(n) * (1 / delta_r ** 2 + 1 / delta_z ** 2) + V_grid.flatten(
    "F"
)
h_off_rz = -a * (np.ones(n - 1) / (delta_r ** 2))
h_off_off_rz = -a * np.ones(n - Nr) / (delta_z ** 2)


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
E0 = 0.001
dt = 0.005
omega = 0.1
ncycles = 1
gauge = "velocity"
time_after_pulse = 20

td = ncycles * 2 * np.pi / omega
tot_time = td + time_after_pulse
nt = int(tot_time / dt)


print("nt:", nt)

# matrices for the time propagation
h_diag_zr = (
    a * 2 * np.ones(n) * (1 / delta_r ** 2 + 1 / delta_z ** 2)
)  # + V_grid.flatten("C")
h_off_zr = -a * (np.ones(n - 1) / (delta_z ** 2))

for i in range(1, n - 1):
    if i % Nz == 0:
        h_off_zr[i - 1] = 0


h_mat_r_rz = Tridiag(diag=h_diag_rz, below=h_off_rz, above=h_off_rz)
h_mat_z_zr = Tridiag(diag=h_diag_zr, below=h_off_zr, above=h_off_zr)

state = (phi[:, 0]).astype(np.complex128).reshape(Nz, Nr).T
state = state * 1 / np.sqrt(trapz(trapz(state.conj() * state, dx=delta_z), dx=delta_r))


if gauge == "length":
    pulse = sine_square_A_length(E0, omega, td)
    z_mat_zr = Tridiag(diag=z_grid.flatten("C"))

elif gauge == "velocity":
    pulse = sine_square_A_velocity(E0, omega, td)
    ddz_off_zr = np.ones(n - 1) / (2 * delta_z)

    for i in range(1, n - 1):
        if i % Nz == 0:
            ddz_off_zr[i - 1] = 0

    p_mat_zr = Tridiag(below=1j * ddz_off_zr, above=-1j * ddz_off_zr)

dipmom = np.zeros(nt, dtype=np.complex128)
norm = np.zeros(nt, dtype=np.complex128)
time_points = np.zeros(nt)


I = Tridiag(diag=np.ones(n))

t = 0
for i in np.arange(nt):
    if gauge == "length":
        z_E_mat_zr = pulse(t) * z_mat_zr
        h_mat_z_zr_temp = h_mat_z_zr + z_E_mat_zr
    elif gauge == "velocity":
        A_p_mat_zr = pulse(t) * p_mat_zr
        h_mat_z_zr_temp = h_mat_z_zr + A_p_mat_zr

    mat0 = I - 1j * (dt / 2) * h_mat_r_rz
    state = mat0.dot(state.flatten("F"))

    state = state.reshape(Nz, Nr).T

    mat0 = I - 1j * (dt / 2) * h_mat_z_zr_temp
    state = mat0.dot(state.flatten("C"))

    mat0 = I + 1j * (dt / 2) * h_mat_z_zr_temp
    state = mat0.dot_inverse(state)

    state = state.reshape(Nr, Nz)

    mat0 = I + 1j * (dt / 2) * h_mat_r_rz
    state = mat0.dot_inverse(state.flatten("F"))

    state = state.reshape(Nz, Nr).T

    mat_temp = state.conj() * z_grid * state
    dipmom[i] = -trapz(trapz(mat_temp, dx=delta_z), dx=delta_r)
    norm[i] = trapz(trapz(state.conj() * state, dx=delta_z), dx=delta_r)
    time_points[i] = t

    if not i % 1000:
        print(i)

    t += dt

samples = {}
samples["time_points"] = time_points
samples["dipole_moment"] = dipmom
samples["norm"] = norm
samples["laser_pulse"] = pulse(time_points)
np.savez(
    f"dt{dt}_dr{dr}_dz{dz}_Lr{Lr}_Lz{Lz}_par{zgrid_parity}_{gauge}.npz", **samples
)
