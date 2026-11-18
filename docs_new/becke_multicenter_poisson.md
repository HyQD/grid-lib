# Atom-centred B-splines and the Becke multicentre Poisson solver

## Scope

This document describes the reusable atom-centred B-spline basis and
multicentre free-space Poisson solver in `grid-lib`. The relevant public
interfaces are:

- `BSplineRadialBasis`, an open, clamped B-spline basis on a finite radial
  interval;
- `BeckePartition`, a smooth geometry-only partition of unity around distinct
  Cartesian centres;
- `BeckeMulticenterPoisson`, which constructs and solves the atom-centred
  multipole problems; and
- `MulticenterPoissonPotential`, the callable potential returned by a solve.

The implementation computes the Coulomb potential of a supplied scalar
density,

```math
V(\mathbf r)=\int\frac{\rho(\mathbf r')}{|\mathbf r-\mathbf r'|}\,
d\mathbf r',
```

or, equivalently,

```math
\nabla^2 V(\mathbf r)=-4\pi\rho(\mathbf r).
```

It supplies a numerical primitive for Coulomb operators. It does not by
itself define orbitals, construct a Hartree-Fock Fock matrix, calculate
exchange, or run an SCF iteration.

## Relation to Becke's multicentre construction

Becke's multicentre integration scheme introduces smooth atomic weights
`w_A` satisfying

```math
\sum_A w_A(\mathbf r)=1.
```

Any molecular integrand, and in particular an electron density, can then be
decomposed as

```math
\rho(\mathbf r)=\sum_A \rho_A(\mathbf r),\qquad
\rho_A(\mathbf r)=w_A(\mathbf r)\rho(\mathbf r).
```

The molecular integral becomes a sum of atom-centred integrals. Becke also
observed that linearity of Poisson's equation permits the same decomposition
for the Coulomb potential: solve independently for every `rho_A` in spherical
coordinates about centre `A`, then add the atomic potentials [1]. Becke and
Dickson subsequently developed this multicentre Poisson strategy explicitly
[2].

`grid-lib` follows that decomposition, but its radial discretization is a
separate implementation choice. Each atomic multipole is solved by a
B-spline Galerkin method on a finite sphere and is continued analytically
outside the sphere. Thus, the implementation is similar in spirit to the
Becke strategy without being a reproduction of the radial solver used in the
original papers.

## Radial B-spline basis

### Knot and continuity convention

`BSplineRadialBasis` is constructed from strictly increasing, nonnegative
breakpoints `nodes`. For polynomial degree `p` and requested interior
continuity `c`, every interior breakpoint has knot multiplicity

```math
m=p-c.
```

The end knots are repeated `p + 1` times, producing an open, clamped basis.
Every basis function has compact radial support. The basis exposes both a
global evaluation and an interval-local evaluation so that later operators
can exploit this compact support.

```python
import numpy as np

from grid_lib import BSplineRadialBasis

nodes = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
basis = BSplineRadialBasis(
    nodes,
    degree=8,
    continuity=4,
    vanish_at_boundaries=(True, False),
)

values, derivatives = basis.evaluate([0.1, 0.7, 3.0])
local_values, local_derivatives, indices = basis.evaluate_interval(
    2, [0.3, 0.4]
)
```

`evaluate` returns arrays whose final dimension has length `basis.size`.
`evaluate_interval` returns only the functions active in one knot interval;
`indices` maps the local columns to the retained global basis ordering.
`locate` maps arbitrary radii to interval indices and returns `-1` outside
the finite domain.

### Boundary flags

`vanish_at_boundaries=(inner, outer)` removes the clamped basis function that
is nonzero at the corresponding boundary.

- `(True, True)` imposes homogeneous values at both ends. This is the default
  and is useful for reduced radial orbital functions in a finite box.
- `(True, False)` imposes the regular inner value while retaining freedom at
  the outer boundary. This is required by `BeckeMulticenterPoisson` so that
  the free-space boundary condition can be imposed weakly.

The boundary flags only impose function values. In particular, the basis
does not by itself impose the angular-momentum-dependent origin behaviour
`u_l(r) = O(r^(l+1))` needed by some orbital representations.

## Becke partition and molecular quadrature

For centres `A` and `B`, define

```math
\mu_{AB}(\mathbf r)=\frac{r_A-r_B}{R_{AB}},
```

where `r_A=|r-R_A|` and `R_AB=|R_A-R_B|`. The polynomial

```math
p(\mu)=\frac{3\mu-\mu^3}{2}
```

is applied repeatedly to sharpen the smooth switch. If `f_k` denotes `k`
applications of `p`, the switch favouring centre `A` is

```math
s_{AB}=\frac{1-f_k(\mu_{AB})}{2}.
```

The unnormalised cell functions and final weights are

```math
P_A=\prod_{B\ne A}s_{AB},\qquad
w_A=\frac{P_A}{\sum_B P_B}.
```

The standard value is `k = 3`, selected with `switch_iterations=3`. This is
the smooth fuzzy-Voronoi construction introduced by Becke [1].

For each centre, `BeckeMulticenterPoisson` combines:

- Gauss-Legendre quadrature in every radial knot interval;
- a Lebedev angular rule on every radial shell; and
- the Becke weight belonging to that centre.

The concatenated Cartesian points are available as `builder.points`, and
their complete volume and partition weights as `builder.integration_weights`.
`builder.center_slices` identifies the points belonging to each centre.
Consequently,

```python
integral = builder.integrate(values)
```

integrates one scalar value per point in the builder ordering.

The present `BeckePartition` is geometry-only. It implements the unshifted
cell boundaries and does not yet include Becke's heteronuclear atomic-size
correction based on adjusted pair coordinates [1, Appendix]. This omission
does not break the partition of unity, but it can make the atomic quadrature
less efficient for centres of very different sizes.

## Multipole Poisson formulation

### Atomic density expansion

For every centre, the partitioned density is expanded in complex spherical
harmonics,

```math
\rho_A(r_A,\Omega_A)=
\sum_{L=0}^{L_{\max}}\sum_{M=-L}^{L}
\rho^A_{LM}(r_A)Y_{LM}(\Omega_A),
```

with

```math
\rho^A_{LM}(r_A)=
\int Y_{LM}^{*}(\Omega_A)\rho_A(r_A,\Omega_A)\,d\Omega_A.
```

All `M` values are retained for each `L`; the implementation is not limited
to axial systems. It uses the convention of `scipy.special.sph_harm_y`,
including the Condon-Shortley phase.

### Reduced radial equation

Writing the atomic potential coefficient as

```math
V^A_{LM}(r)=\frac{u^A_{LM}(r)}{r}
```

gives the one-dimensional problem

```math
\left[-\frac{d^2}{dr^2}+\frac{L(L+1)}{r^2}\right]u^A_{LM}(r)
=4\pi r\rho^A_{LM}(r).
```

The regular inner condition is `u_LM(0) = 0`. The corresponding Poisson
basis must therefore start at zero and use
`vanish_at_boundaries=(True, False)`.

### Galerkin matrix and free-space boundary

For a B-spline expansion `u_LM = sum_j c_j B_j`, integration by parts gives
the matrix

```math
K^{(L)}_{ij}=
\int_0^R B_i'(r)B_j'(r)\,dr
+L(L+1)\int_0^R\frac{B_i(r)B_j(r)}{r^2}\,dr
+\frac{L}{R}B_i(R)B_j(R),
```

and right-hand side

```math
b^{LM}_i=4\pi\int_0^R B_i(r)r\rho_{LM}(r)\,dr.
```

The surface term imposes the free-space Robin condition

```math
u_{LM}'(R)+\frac{L}{R}u_{LM}(R)=0.
```

It matches the source-free exterior multipole exactly. At evaluation points
outside the atomic sphere, the code uses

```math
u_{LM}(r)=u_{LM}(R)\left(\frac{R}{r}\right)^L,
\qquad
V_{LM}(r)\propto r^{-L-1}.
```

The matrix depends on the centre's radial basis and `L`, but not on `M` or on
the density. Its Cholesky factorization is therefore cached once per centre
and `L` and reused by subsequent solves.

At the centre itself, the finite monopole value is obtained from the origin
derivative of `u_00`. Higher multipoles vanish there.

## Basic use

The following example builds a two-centre solver and evaluates the potential
of a normalized Gaussian density:

```python
import numpy as np

from grid_lib import BSplineRadialBasis
from grid_lib.multicenter import BeckeMulticenterPoisson

centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
nodes = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
radial_basis = BSplineRadialBasis(
    nodes,
    degree=8,
    continuity=4,
    vanish_at_boundaries=(True, False),
)

builder = BeckeMulticenterPoisson(
    centers,
    radial_basis,
    L_max=10,
    radial_order=12,
    lebedev_order=29,
)

alpha = 1.3
source = np.array([0.2, 0.3, -0.1])

def density(points):
    displacement = points - source
    return (alpha / np.pi) ** 1.5 * np.exp(
        -alpha * np.sum(displacement**2, axis=-1)
    )

potential = builder.solve(density)
targets = np.array([[0.0, 0.0, 0.0], [3.0, 0.5, -0.2]])
values = potential(targets)
charge = potential.total_charge
```

Passing a single basis reuses it at every centre. A sequence of bases can be
used when different atomic species require different nodes, degrees, or
outer radii.

If the density is already available on `builder.points`, use
`builder.solve_values(density_values)`. Both real and complex scalar
densities are accepted. `MulticenterPoissonPotential` is callable on arrays
with final dimension three and evaluates large arrays in chunks. For a real
density, tiny imaginary components can remain from complex spherical-harmonic
roundoff.

For a density sampled in the same quadrature, its Coulomb self-energy can be
formed as

```python
density_values = density(builder.points)
potential = builder.solve_values(density_values)
hartree_energy = 0.5 * builder.integrate(
    density_values * potential(builder.points)
).real
```

## Independent convergence controls

Several errors are present and should be converged separately:

1. `r_max` and the radial nodes control source truncation and radial
   resolution. Each atomic component is sampled only inside its own sphere.
   Density outside that sphere is omitted from that component.
2. The B-spline degree and continuity control the radial approximation space.
3. `radial_order` controls Gauss-Legendre integration within every knot
   interval. It must be at least the largest B-spline degree.
4. `lebedev_order` controls angular projection and Becke integration. A fixed
   finite Lebedev grid is not continuously rotation invariant; its
   orientation dependence should decrease with increasing order.
5. `L_max` controls the atomic density and potential multipole expansion.

Changing several controls simultaneously can hide cancellation between
radial, angular, source-truncation, and multipole errors. Production studies
should vary them one at a time and monitor at least the integrated charge,
potential values, Coulomb energies, and Coulomb reciprocity.

## Validation currently covered

The regression tests cover the following properties:

- B-spline values and first derivatives agree with SciPy reference splines,
  including knot and finite-boundary values.
- Interval-local evaluation reproduces global evaluation and identifies only
  compactly supported active functions.
- A normalized single-centre Gaussian reproduces its analytic charge,
  potential, `1/r` exterior tail, and Coulomb self-energy. The current test
  tolerances are `5e-7` for sampled potential values and `5e-11` for the
  self-energy.
- A displaced Gaussian converges as `L_max` is increased.
- Radial charge integration converges as `radial_order` is increased.
- Simultaneously rotating the centres, density, targets, and Lebedev grid
  preserves the potential to a current tolerance of `2e-14`.
- The orientation envelope decreases strongly with increasing Lebedev order.
- A three-centre, water-like geometry supports distinct radial bases; its
  current regression tolerances are `3e-11` for total charge and `7e-5` for
  selected potential values.
- Coulomb self-interactions are positive, and the discrepancy between
  `integral rho_a V_b` and `integral rho_b V_a` decreases with increasing
  `L_max`.

These tolerances validate the implementation on controlled analytic cases;
they are not universal error estimates for an arbitrary molecular density.

## Current limitations

- The Becke weights do not yet include heteronuclear atomic-size corrections.
- Every atomic density component is truncated at its radial basis extent.
  The analytic exterior continuation is exact for the represented interior
  source, but it cannot recover omitted source density.
- One scalar density is solved at a time. Batched pair densities and direct
  orbital-product interfaces are not yet implemented.
- The dense Cholesky factorizations do not exploit the full banded structure
  of the B-spline matrices.
- Nuclear derivatives, Becke-weight derivatives, and forces are not exposed.
- Exchange is not a local Poisson solve. A Hartree-Fock implementation must
  construct the required orbital-pair densities and contractions at a higher
  level.
- The solver provides a numerical Coulomb primitive, not an SCF convergence
  strategy or a time-propagation scheme.

## References

1. A. D. Becke, "A multicenter numerical integration scheme for polyatomic
   molecules," *The Journal of Chemical Physics* **88**, 2547--2553 (1988).
   [doi:10.1063/1.454033](https://doi.org/10.1063/1.454033)
2. A. D. Becke and R. M. Dickson, "Numerical solution of Poisson's equation
   in polyatomic molecules," *The Journal of Chemical Physics* **89**,
   2993--2997 (1988).
   [doi:10.1063/1.455005](https://doi.org/10.1063/1.455005)
