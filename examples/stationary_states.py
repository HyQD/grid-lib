import numpy as np

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Rational_map,
)
from grid_lib.spherical_coordinates.potentials import (
    Coulomb,
    SAE,
    Gaussian_charge_distribution,
)

# Setup Legendre-Lobatto grid
N = 200
nr = N - 1
r_max = 40
alpha = 0.4

gll = GaussLegendreLobatto(N, Rational_map(r_max=r_max, alpha=alpha))
PN_x = gll.PN_x
weights = gll.weights

r = gll.r[1:-1]
D2 = gll.D2[1:-1, 1:-1]
ddr = -0.5 * D2

Hydrogen_potential = Coulomb(Z=1)
SAE_He_potential = SAE(Z=2, A=0, B=2.083)
Gaussian_charge_distribution_hydrogen = Gaussian_charge_distribution(mu=100)

V_Hydrogen = np.diag(Hydrogen_potential(r))
V_SAE_He = np.diag(SAE_He_potential(r))
V_Gaussian_charge_distribution = np.diag(Gaussian_charge_distribution_hydrogen(r))

print()
for l in range(0, 3):

    T = ddr + np.diag(l * (l + 1) / (2 * r**2))
    H_l_Hydrogen = T + V_Hydrogen
    H_l_SAE_He = T + V_SAE_He
    H_l_Gaussian_charge_distribution = T + V_Gaussian_charge_distribution

    eps_l_hydrogen, phi_nl_hydrogen = np.linalg.eigh(H_l_Hydrogen)
    eps_l_SAE_He, phi_nl_SAE_He = np.linalg.eigh(H_l_SAE_He)
    (
        eps_l_Gaussian_charge_distribution,
        phi_nl_Gaussian_charge_distribution,
    ) = np.linalg.eigh(H_l_Gaussian_charge_distribution)

    print(f"** The 3 lowest lying eigenvalues for l={l} **")
    print(eps_l_hydrogen[0:3])
    print(eps_l_Gaussian_charge_distribution[0:3])
    print(eps_l_SAE_He[0:3])
    print()
