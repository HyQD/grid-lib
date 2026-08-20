import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import scipy.linalg as sp

# Parameters
r_max = 10.0
dr = 0.001

n_r = int(r_max / dr)

r = np.linspace(dr, r_max, n_r)
print(r[1]-r[0])


nvals = 10
# Matrices
D2 = diags([1, -2, 1], [-1, 0, 1], shape=(n_r, n_r)) / dr ** 2
D1 = diags([-0.5, 0, 0.5], [-1, 0, 1], shape=(n_r, n_r)) / dr


r_new = r - dr / 2 #For some reasone it is clever to take the midpoint here...

R_inv = diags(1 / r_new, 0)
R_inv_D1 = R_inv.dot(D1)

# D2_dense = (
#     np.diag(-2 / dr ** 2 * np.ones(n_r))
#     + np.diag(1 / dr ** 2 * np.ones(n_r - 1), k=-1)
#     + +np.diag(1 / dr ** 2 * np.ones(n_r - 1), k=1)
# )

# D1_dense = np.diag(-0.5 / dr * np.ones(n_r - 1), k=-1) + np.diag(
#     0.5 / dr * np.ones(n_r - 1), k=1
# )
# R_inv_dense = np.diag(1/r_new)
# R_inv_D1_dense = np.dot(R_inv_dense, D1_dense)




T = -0.5*(D2 + R_inv_D1)
V = diags(0.5*r_new**2, 0)
print(T.shape, V.shape)
H_rho = T+V
# Finite differences
vals, vecs = eigs(H_rho, k=nvals, which="SM")
eps_rho = np.sort(vals)
print(eps_rho[0:4])

dz = 0.1
z_max = 10
n_z = int(2 * z_max / dz) + 1
z = np.linspace(-z_max, z_max, n_z)

H_diag = 1.0 / (dz ** 2) * np.ones(n_z) + 0.5 * z ** 2
H_off_diag_upper = -1.0 / (2 * dz ** 2) * np.ones(n_z - 1)
H_off_diag_lower = -1.0 / (2 * dz ** 2) * np.ones(n_z - 1)

h_z = (
    np.diag(H_diag)
    + np.diag(H_off_diag_lower, k=-1)
    + np.diag(H_off_diag_upper, k=1)
)

eps_z, C_z = np.linalg.eig(h_z)
print(eps_z[0:4])

print(eps_rho[0].real+eps_z[0].real)

