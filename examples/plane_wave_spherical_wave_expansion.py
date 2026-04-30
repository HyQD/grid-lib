import numpy as np
from matplotlib import pyplot as plt

from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_momentum import LM_to_I
from grid_lib.spherical_coordinates.potentials import (
    plane_wave_spherical_wave_expansion,
)

# Plane wave parameters
k = np.array([0.0, 0.0, 2.0])  # wave vector along z-axis

L_max = 3
M_max = 0  # k along z: only M = 0 terms contribute

# Radial grid
N = 150
r_max = 10.0
gll = GaussLegendreLobatto(N, Linear_map(r_min=0.0, r_max=r_max))
r = gll.r[1:-1]

# Compute radial components for e^{+ik.r} and e^{-ik.r}
f_LM_plus = plane_wave_spherical_wave_expansion(r, k, L_max, M_max, sign=+1)
f_LM_minus = plane_wave_spherical_wave_expansion(r, k, L_max, M_max, sign=-1)

K = np.linalg.norm(k)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

# Left panel: e^{+ik.r}
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    data = f_LM_plus[I]
    if np.isrealobj(data):
        if np.allclose(data, 0.0):
            continue
        ax1.plot(r, data, label=rf"$f_{{L={L},M=0}}(r)$")
    else:
        label = rf"$f_{{L={L},M=0}}(r)$"
        if not np.allclose(data.real, 0.0):
            ax1.plot(r, data.real, label=label + r" $\mathrm{Re}$")
        if not np.allclose(data.imag, 0.0):
            ax1.plot(r, data.imag, "--", label=label + r" $\mathrm{Im}$")

ax1.set_xlabel(r"$r$ (a.u.)")
ax1.set_ylabel(r"$f_{LM}(r)$")
ax1.set_title(
    rf"$e^{{i\mathbf{{k}}\cdot\mathbf{{r}}}}$, "
    rf"$\mathbf{{k}} = (0,0,{K:.1f})$"
)
ax1.legend()
ax1.set_xlim(0, r_max)

# Right panel: e^{-ik.r}
for L in range(L_max + 1):
    I = LM_to_I(L, 0, L_max, M_max)
    data = f_LM_minus[I]
    if np.isrealobj(data):
        if np.allclose(data, 0.0):
            continue
        ax2.plot(r, data, label=rf"$f_{{L={L},M=0}}(r)$")
    else:
        label = rf"$f_{{L={L},M=0}}(r)$"
        if not np.allclose(data.real, 0.0):
            ax2.plot(r, data.real, label=label + r" $\mathrm{Re}$")
        if not np.allclose(data.imag, 0.0):
            ax2.plot(r, data.imag, "--", label=label + r" $\mathrm{Im}$")

ax2.set_xlabel(r"$r$ (a.u.)")
ax2.set_ylabel(r"$f_{LM}(r)$")
ax2.set_title(
    rf"$e^{{-i\mathbf{{k}}\cdot\mathbf{{r}}}}$, "
    rf"$\mathbf{{k}} = (0,0,{K:.1f})$"
)
ax2.legend()
ax2.set_xlim(0, r_max)

plt.tight_layout()
plt.show()
