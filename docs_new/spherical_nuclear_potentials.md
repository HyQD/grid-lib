# Nuclear potentials in spherical coordinates

## Scope

This document describes how `grid-lib` represents the potential energy of an
electron in the field of fixed point nuclei. The implementation supports
arbitrary nuclear positions, although the first molecular application is a
linear diatomic molecule with its nuclei on the laboratory `z` axis.

The relevant public interfaces are:

- `clamped_molecular_potential_quadrature`, which evaluates the analytic
  free-space multipole expansion;
- `clamped_molecular_potential_Poisson`, which obtains the same coefficients
  from discrete radial Coulomb kernels; and
- `SphericalMultipolePotential`, which applies supplied radial multipole
  coefficients to spherical-harmonic orbital expansions.

The name `clamped_molecular_potential_quadrature` is historical. The current
implementation uses the analytic Laplace expansion rather than performing an
angular quadrature.

## Spherical-harmonic convention

For nuclei with charges `Z_A > 0` at fixed Cartesian positions
`R_A`, the electronic nuclear-attraction potential is

```math
V_{\mathrm{nuc}}(\mathbf r)
=-\sum_A\frac{Z_A}{|\mathbf r-\mathbf R_A|}.
```

It is expanded in complex spherical harmonics as

```math
V_{\mathrm{nuc}}(\mathbf r)
=\sum_{L=0}^{L_{\max}}\sum_{M=-M_{\max}}^{M_{\max}}
V_{LM}(r)Y_{LM}(\Omega),
```

where terms with `|M| > L` are absent. `grid-lib` uses the same spherical
harmonic convention as `scipy.special.sph_harm_y`, including the
Condon--Shortley phase.

The Laplace expansion is

```math
\frac{1}{|\mathbf r-\mathbf R_A|}
=4\pi\sum_{L=0}^{\infty}\sum_{M=-L}^{L}
\frac{1}{2L+1}
\frac{r_<^L}{r_>^{L+1}}
Y_{LM}(\Omega)Y_{LM}^{*}(\Omega_A),
```

with

```math
r_<=\min(r,R_A),\qquad r_>=\max(r,R_A).
```

Consequently, the stored radial coefficients are

```math
V_{LM}(r)
=-\sum_A Z_A\frac{4\pi}{2L+1}
\frac{r_<^L}{r_>^{L+1}}Y_{LM}^{*}(\Omega_A).
```

The complex conjugation of `Y_LM(Omega_A)` is essential. It has no visible
effect in tests restricted to `M = 0`, because `Y_L0` is real. For an
off-axis nucleus, however, omitting it gives the wrong azimuthal phase and the
projection of a real nuclear potential need not be Hermitian.

The coefficient array has shape

```text
(number_of_lm_states(L_max, M_max), n_r)
```

and uses `LM_to_I(L, M, L_max, M_max)` ordering. Magnetic quantum numbers are
the outer ordering: `M = -M_max, ..., M_max`, with
`L = |M|, ..., L_max` inside each magnetic sector.

## Important limits and symmetries

### Nucleus at the expansion origin

For a nucleus at the origin, only the monopole coefficient is nonzero:

```math
V_{00}(r)=-Z\frac{\sqrt{4\pi}}{r}.
```

Since `Y_00 = 1/sqrt(4*pi)`, its projected action is exactly `-Z/r` in every
orbital channel.

### Midpoint-centered axial diatomic molecule

For nuclei at

```math
\mathbf R_A=(0,0,-a),\qquad
\mathbf R_B=(0,0,a),
```

all coefficients with `M != 0` vanish. The remaining coefficients are

```math
V_{L0}(r)=-\sqrt{\frac{4\pi}{2L+1}}
\left[Z_A(-1)^L+Z_B\right]
\frac{r_<^L(r,a)}{r_>^{L+1}(r,a)}.
```

For a homonuclear molecule, `Z_A = Z_B`, every odd-`L` coefficient cancels.
For a heteronuclear molecule, odd multipoles generally remain nonzero even
though the expansion origin is the bond midpoint.

The finite multipole coefficients are continuous at `r = a`, but their first
radial derivatives have finite jump discontinuities there. During the initial
diatomic development, `a` should therefore be an internal FEM element
boundary. This prevents a single high-order polynomial element from spanning
the derivative discontinuity and also places the Poisson delta source on a
global FEM-DVR node.

## Analytic construction

`clamped_molecular_potential_quadrature` evaluates the analytic coefficient
formula directly on a supplied radial grid:

```python
import numpy as np

from grid_lib.spherical_coordinates import (
    clamped_molecular_potential_quadrature,
)

a = 1.0
r = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
positions = np.array(
    [
        [0.0, 0.0, -a],
        [0.0, 0.0, a],
    ]
)
charges = np.array([1.0, 1.0])

V_LM = clamped_molecular_potential_quadrature(
    r,
    positions,
    charges,
    L_max=8,
    M_max=0,
)
```

This construction is the preferred representation of fixed point nuclei. It
does not move a nucleus to the nearest grid point and it preserves the exact
free-space radial kernel at every requested radius.

At `r = R_A`, each finite multipole coefficient remains well-defined. The
point singularity at `Omega = Omega_A` is recovered only by the infinite
angular expansion, so convergence with `L_max` becomes slower near a displaced
nucleus.

## Poisson construction

The same potential is the free-space solution of

```math
\nabla^2V_{\mathrm{nuc}}(\mathbf r)
=4\pi\rho_{\mathrm{nuc}}(\mathbf r),
```

where

```math
\rho_{\mathrm{nuc}}(\mathbf r)
=\sum_A Z_A\delta^{(3)}(\mathbf r-\mathbf R_A).
```

After expansion in spherical harmonics, the radial source is

```math
\rho_{LM}(r)
=\sum_A Z_A\frac{\delta(r-R_A)}{r^2}
Y_{LM}^{*}(\Omega_A).
```

`clamped_molecular_potential_Poisson` assumes that `W[L]` is the discrete
radial Coulomb kernel for rank `L`, including the factor
`4*pi/(2*L+1)`. If nucleus `A` is at radial node `r_j`, the constructed
coefficient is

```math
V_{LM}(r_i)
=-\sum_A Z_A Y_{LM}^{*}(\Omega_A)
\frac{W^{(L)}_{ij}}{w_j},
```

where `w_j` is the radial quadrature weight. Division by `w_j` is the nodal
representation of the radial delta function.

By default, the routine raises an error unless every noncentral nuclear radius
coincides with a radial grid point. Passing `force=True` snaps the source to
the nearest point and emits a warning. This option is useful for diagnostics,
but it changes the represented geometry and is not recommended for production
clamped-nuclei calculations.

The Poisson route is primarily an independent discrete check of the analytic
construction. It may also become useful for finite nuclear charge
distributions.

## Action on an orbital expansion

Reduced radial orbitals are represented as

```math
\phi_p(\mathbf r)
=\frac{1}{r}\sum_{lm}u^p_{lm}(r)Y_{lm}(\Omega).
```

The nuclear-potential action is

```math
(V_{\mathrm{nuc}}u^p)_{lm}(r)
=\sum_{LM}\sum_{l'm'}V_{LM}(r)
\langle lm|Y_{LM}|l'm'\rangle u^p_{l'm'}(r).
```

`SphericalMultipolePotential` performs this contraction using sparse Gaunt
coupling matrices. Its orbital arrays use radial-first storage:

```text
(..., n_r, n_lm)
```

where the leading dimensions may contain orbitals or other batch indices. No
radial quadrature weights are absorbed into either `V_LM(r)` or the operator
action. This is the nodal-value convention used by the spherical code.

A two-centre operator can be constructed and applied as follows:

```python
from grid_lib.spherical_coordinates import SphericalMultipolePotential

potential = SphericalMultipolePotential.from_point_charges(
    r,
    positions,
    charges,
    l_max=6,
    m_max=2,
    L_max=12,
    M_max=0,
)

# orbitals.shape == (n_orbitals, len(r), potential.orbital_shape[1])
potential_times_orbitals = potential.apply(orbitals)
```

For general, nonaxial nuclear arrangements, `M_max` must be large enough to
represent the required magnetic couplings. The factory defaults are

```text
L_max = 2*l_max
M_max = min(L_max, 2*m_max)
```

unless the cutoffs are specified explicitly.

The method `angular_matrix()` returns the pointwise channel matrices with
shape `(n_r, n_lm, n_lm)`. It is useful for testing and small explicit
Hamiltonians, but the sparse `apply()` path avoids constructing this dense
three-dimensional array.

## Axial fixed-`m` blocks

When only `M = 0` coefficients are present, the Gaunt selection rule gives
`m' = m`. The nuclear potential therefore conserves signed `m` while coupling
different values of `l`.

`SphericalMultipolePotential.is_axisymmetric` detects this case, and
`fixed_m_matrix(m)` returns the pointwise block

```math
V^{(m)}_{ll'}(r)
=\sum_L V_{L0}(r)\langle lm|Y_{L0}|l'm\rangle.
```

The returned array has shape

```text
(n_r, l_max - abs(m) + 1, l_max - abs(m) + 1).
```

Calling `fixed_m_matrix` for a nonaxisymmetric potential raises an error.

## Choice of angular cutoff

For orbital channels truncated at `l_max`, the Gaunt triangle rule implies

```math
|l-l'|\le L\le l+l'.
```

Hence multipoles with `L > 2*l_max` have identically zero matrix elements in
the retained orbital space. Choosing

```text
nuclear_L_max = 2*l_max
```

therefore gives the complete Galerkin projection of the point-charge
potential for that orbital space. A smaller nuclear cutoff is an additional
approximation. Increasing `nuclear_L_max` beyond `2*l_max` cannot change the
projected operator, although increasing the orbital `l_max` can still be
necessary to resolve displaced Coulomb cusps.

For an axial molecule, `nuclear_M_max = 0` is sufficient. This statement
applies only to the nuclear potential. Electron-electron pair densities can
have nonzero magnetic components, so their Coulomb cutoffs must be selected
independently.

## Validation currently covered

The operator tests check:

1. the centered-nucleus `-Z/r` limit;
2. the analytic axial coefficients for nuclei at `+a` and `-a`;
3. cancellation of odd multipoles in a homonuclear molecule;
4. survival of odd multipoles in a heteronuclear molecule;
5. conservation of `m` and agreement of fixed-`m` blocks with the full
   channel matrix;
6. Hermiticity for real point-charge potentials;
7. agreement with direct angular quadrature for an off-axis nucleus;
8. agreement between the Poisson factory and the underlying Poisson
   coefficient builder; and
9. vanishing projected action for multipoles above `2*l_max`.
