# Multicentre Coulomb potentials

`BeckeMulticenterPoisson` constructs a free-space Coulomb potential from a
scalar density sampled on atom-centred radial--Lebedev grids. It combines a
smooth Becke partition with B-spline Galerkin solutions of the radial Poisson
equations.

For each centre (A), the partitioned density is

\[
\rho_A(\mathbf r)=w_A(\mathbf r)\rho(\mathbf r),
\qquad \sum_A w_A(\mathbf r)=1,
\]

and is analyzed in local spherical harmonics. With
(u_{LM}(r)=rV_{LM}(r)), each multipole satisfies

\[
\left[-\frac{d^2}{dr^2}+\frac{L(L+1)}{r^2}\right]u_{LM}(r)
=4\pi r\rho_{LM}(r).
\]

The potential basis vanishes at the origin but retains its outer endpoint
function. The free-space Robin condition

\[
u'_{LM}(R)+\frac{L}{R}u_{LM}(R)=0
\]

then supplies the represented source's analytic exterior multipole tail.

## Example

```python
import numpy as np

from grid_lib import BSplineRadialBasis
from grid_lib.multicenter import BeckeMulticenterPoisson

centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
basis = BSplineRadialBasis(
    [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    degree=8,
    continuity=4,
    vanish_at_boundaries=(True, False),
)
builder = BeckeMulticenterPoisson(
    centers,
    basis,
    L_max=8,
    radial_order=12,
    lebedev_order=29,
)

alpha = 1.0

def density(points):
    return (alpha / np.pi) ** 1.5 * np.exp(
        -alpha * np.sum(points**2, axis=-1)
    )

potential = builder.solve(density)
values = potential(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
print(potential.total_charge, values)
```

The flattened molecular quadrature is available as `builder.points` and
`builder.integration_weights`. Consumers that already evaluate densities,
such as Hartree--Fock orbital code, can call
`builder.solve_values(density_values)` without an extra callback.

## Independent convergence controls

Four numerical limits must be checked separately:

1. radial B-spline knots, degree, and continuity;
2. Gauss--Legendre `radial_order` within every knot interval;
3. atom-centred `lebedev_order`;
4. Coulomb multipole cutoff `L_max`.

`angular_rotation` rotates every Lebedev grid and provides a direct
orientation-sensitivity diagnostic. A simultaneous rotation of centres,
density, target points, and angular grid is covariant when all magnetic
components through `L_max` are retained.

## Pilot scope

- The current Becke partition is geometry-only; atomic-size corrections are
  not yet included.
- Each call solves one real or complex scalar density. Batched pair-density
  solves are deferred until a consuming Hartree--Fock interface establishes
  the required memory layout.
- Every atomic source is truncated at its radial basis extent. The returned
  `total_charge` exposes the corresponding finite-grid charge diagnostic.
- Hartree and exchange contractions belong to `tdmcscf`, not this spatial
  operator.

::: grid_lib.multicenter.BeckePartition

::: grid_lib.multicenter.BeckeMulticenterPoisson

::: grid_lib.multicenter.MulticenterPoissonPotential
