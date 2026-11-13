# Prolate spheroidal coordinates

## Scope

`grid_lib.prolate_spheroidal_coordinates` provides numerical primitives for
fixed-axis, two-centre calculations in prolate spheroidal coordinates. The
package supplies grids, coordinate arrays, weak-form one-electron operators,
overlap transformations, and two alternative electron--electron Coulomb
constructions. It does not prescribe a physical system, laser pulse,
absorber, eigensolver, or time propagator.

The implementation uses the finite-element discrete-variable representation
(FEM-DVR) framework [5] in the prolate formulation developed for diatomic
targets by Tao, McCurdy, and Rescigno [1]. Related time-dependent H2+
calculations are described in Ref. [2]. The finite-box Poisson Coulomb
construction follows Haxton, Lawler, and McCurdy [3], while the direct
Neumann expansion and fixed-axis strong-field context are also discussed by
Zhang, Yuan, and Zhao [4].

## Coordinate convention

The nuclei are the foci at `z = -a` and `z = +a`, with internuclear distance
`R = 2a`. The coordinates are

```math
x=a\sqrt{(\xi^2-1)(1-\eta^2)}\cos\phi,
\qquad
y=a\sqrt{(\xi^2-1)(1-\eta^2)}\sin\phi,
\qquad
z=a\xi\eta,
```

where `1 <= xi < infinity`, `-1 <= eta <= 1`, and `0 <= phi < 2*pi`. The
distances to the foci are

```math
r_1=a(\xi+\eta),
\qquad
r_2=a(\xi-\eta),
```

and the volume element is

```math
d^3r=a^3(\xi^2-\eta^2)\,d\xi\,d\eta\,d\phi.
```

`ProlateSpheroidalGrid` exposes the coordinate arrays through:

- `mesh()` for the tensor-product `xi` and `eta` arrays;
- `axial_coordinate()` for `z`;
- `cylindrical_radius()` for `sqrt(x**2 + y**2)`;
- `cartesian_radius()` for the distance from the bond midpoint;
- `focus_distances()` for `(r1, r2)`; and
- `cartesian_coordinates(phi)` for `(x, y, z)` on a selected azimuthal
  half-plane.

The focus labels are tied to the equations above. Code that associates
different nuclear labels with the two foci must swap the corresponding
charges consistently.

## Grid construction

The public constructors are:

- `setup_prolate_spheroidal_grid(...)`, for uniformly spaced `xi` element
  boundaries; and
- `setup_prolate_spheroidal_grid_from_boundaries(...)`, for explicit,
  nonuniform boundaries and either one common point count or one count per
  element.

The first `xi` element uses a right Gauss--Radau rule. It omits the singular
focus endpoint `xi = 1` and includes its right interface. Subsequent elements
use Gauss--Lobatto rules and share interface nodes. The `eta` coordinate uses
a Gauss--Legendre rule and therefore omits both molecular-axis endpoints
`eta = +/-1`. This axis-free layout is required by the current prolate
operators.

For example, the following is dimensionally comparable to the H2 grid used
in the Haxton-development calculation, but uses `R = 2` bohr for H2+:

```python
from grid_lib.prolate_spheroidal_coordinates import (
    active_indices,
    setup_prolate_spheroidal_grid_from_boundaries,
)

grid = setup_prolate_spheroidal_grid_from_boundaries(
    internuclear_distance=2.0,
    xi_boundaries=[1.0, 4.0, 14.0, 24.0],
    xi_points_per_element=14,
    eta_degree=9,
)

print(grid.shape)                 # (40, 9)
print(grid.size)                  # 360
print(active_indices(grid).size)  # 351 with the outer xi row removed
```

Per-element orders are also accepted:

```python
grid = setup_prolate_spheroidal_grid_from_boundaries(
    internuclear_distance=2.0,
    xi_boundaries=[1.0, 2.0, 5.0],
    xi_points_per_element=[8, 12],
    eta_degree=9,
)
```

Element point counts include both endpoints for a Lobatto element. Shared
interfaces occur only once in the assembled global grid.

## Fixed magnetic sectors and axis regularity

For an azimuthal factor

```math
\frac{e^{im\phi}}{\sqrt{2\pi}},
```

the remaining numerical state is a function of `(xi, eta)`. The kinetic
builders operate on one fixed integer `m` sector. Even `abs(m)` uses the
ordinary cardinal basis. Odd `abs(m)` uses cardinal functions multiplied by
the regularity factor

```math
\sqrt{(\xi^2-1)(1-\eta^2)}.
```

`xi_derivative_matrix` and `eta_derivative_matrix` include the derivatives of
these factors. This is why manually constructed grids containing `xi = 1` or
`eta = +/-1` are rejected by `validate_axis_free_grid`.

The main one-electron operator functions are:

- `separable_kinetic_matrices`, which caches the one-dimensional weighted
  stiffness and weight matrices;
- `kinetic_energy_matrix`, which returns the weak kinetic matrix including
  the nonzero-`m` centrifugal contribution;
- `electron_nuclear_coulomb_matrix`, which returns the weak two-centre
  attraction matrix; and
- `active_indices`, `restrict_matrix`, and `restrict_diagonal`, which define
  and apply an active-node map.

By default, `active_indices` removes the outermost `xi` row to impose a
homogeneous Dirichlet boundary. It does not remove a focus or molecular-axis
row because those endpoints are absent from the quadrature.

## Overlap and coefficient convention

Nodal coefficients use the diagonal physical overlap

```math
S_{ij}
=a^3 w_i^{(\xi)}w_j^{(\eta)}(\xi_i^2-\eta_j^2).
```

`overlap_diagonal(grid)` returns this array flattened in C order, with `eta`
as the fastest index. A nodal coefficient vector `c` therefore has norm

```math
\lVert c\rVert_S^2=c^\dagger S c.
```

The Euclidean coefficients are

```math
y=S^{1/2}c.
```

Use `to_euclidean_coefficients` and `from_euclidean_coefficients` for this
transformation. They accept a full tensor-product state ending in
`grid.shape`, a flattened state ending in `grid.size`, or an active-space
state when the same `active_indices` array is supplied:

```python
from grid_lib.prolate_spheroidal_coordinates import (
    from_euclidean_coefficients,
    to_euclidean_coefficients,
)

y = to_euclidean_coefficients(grid, nodal_coefficients)
restored = from_euclidean_coefficients(grid, y)
```

For a weak Hamiltonian matrix `A`, the generalized eigenproblem is

```math
A c=E S c,
```

and its Euclidean symmetric form is

```math
S^{-1/2} A S^{-1/2}y=Ey.
```

The transforms do not restrict the state automatically. Full-grid and
active-space layouts must not be mixed.

## Local potentials

A pointwise local potential `V(xi, eta)` has weak diagonal

```math
A^{(V)}_{ij}=S_{ij}V(\xi_i,\eta_j).
```

`local_potential_diagonal` and `local_potential_matrix` perform this
construction. They accept real or complex finite values, so the primitive is
also suitable for a representation-level complex absorbing potential. CAP
placement and strength remain application policy and do not belong in
`grid-lib`.

```python
import numpy as np

from grid_lib.prolate_spheroidal_coordinates import local_potential_matrix

z = grid.axial_coordinate()
xi, _ = grid.mesh()
q = grid.a * xi
q0 = 0.8 * grid.a * grid.xi[-1]
potential = 0.02 * z - 1j * 1.0e-4 * np.maximum(q - q0, 0.0) ** 3
weak_potential = local_potential_matrix(grid, potential)
```

This function returns a weak matrix. Applying it as though it were the strong
nodal action introduces an extra factor of `S`. The corresponding strong
local action is simply pointwise multiplication by `V`.

For nuclei with charges `Z1` and `Z2`, the physical two-centre attraction is

```math
V_{\mathrm{en}}(\xi,\eta)
=-\frac{Z_1}{a(\xi+\eta)}-\frac{Z_2}{a(\xi-\eta)}.
```

`electron_nuclear_coulomb_diagonal` forms its weak diagonal analytically. The
factor `(xi**2 - eta**2)` from the volume element cancels the apparent focal
denominators, giving a finite weak expression on the axis-free grid.

## Electron--electron Coulomb functionality

The package contains two density-to-potential solvers:

- `NeumannCoulombSolver`, which evaluates a truncated prolate Neumann
  expansion using associated Legendre functions; and
- `PoissonCoulombSolver`, which uses a finite-box radial Green's function
  obtained from the inverse DVR operator plus the outer-boundary correction
  described by Haxton, Lawler, and McCurdy [3].

Both solvers act on real-space densities stored on the full tensor-product
grid. Their angular-transfer parameter is `angular_m`; it is not an orbital
magnetic quantum number unless the calling contraction makes that
identification. The Neumann truncation `l_max`, box size, and grid resolution
must be converged for the target calculation.

Convenience functions provide the potential, its weak diagonal, and its
sparse weak matrix. `coulomb_self_energy` evaluates the associated Hartree
self-energy with the prolate overlap metric.

## Numerical and modelling limitations

- The coordinate scale and foci are fixed by one internuclear distance.
  Moving-nuclei dynamics requires additional coordinate and kinetic
  couplings and is not implemented by these primitives.
- A single grid has one fixed molecular axis. Orientation averaging or
  rotational dynamics belongs at a higher level.
- The current operators assume an axis-free Radau/Gauss grid.
- The outer boundary is a constant-`xi` ellipsoid, not a sphere.
- Grid construction and weak operators do not by themselves define a
  normalized wavefunction, Hamiltonian action, boundary-condition study, or
  converged physical calculation.
- Exterior complex scaling, used in the original scattering and H2+
  time-dependent FEM-DVR work [1,2], is not currently exposed by this module.

## References

1. L. Tao, C. W. McCurdy, and T. N. Rescigno, “Grid-based methods for
   diatomic quantum scattering problems: A finite-element discrete-variable
   representation in prolate spheroidal coordinates,” *Physical Review A*
   **79**, 012719 (2009).
   [doi:10.1103/PhysRevA.79.012719](https://doi.org/10.1103/PhysRevA.79.012719)
2. L. Tao, C. W. McCurdy, and T. N. Rescigno, “Grid-based methods for
   diatomic quantum scattering problems. II. Time-dependent treatment of
   single- and two-photon ionization of H2+,” *Physical Review A* **80**,
   013402 (2009).
   [doi:10.1103/PhysRevA.80.013402](https://doi.org/10.1103/PhysRevA.80.013402)
3. D. J. Haxton, K. V. Lawler, and C. W. McCurdy, “Multiconfiguration
   time-dependent Hartree--Fock treatment of electronic and nuclear dynamics
   in diatomic molecules,” *Physical Review A* **83**, 063416 (2011).
   [doi:10.1103/PhysRevA.83.063416](https://doi.org/10.1103/PhysRevA.83.063416)
4. B. Zhang, J. Yuan, and Z. Zhao, “DMTDHF: A full dimensional
   time-dependent Hartree--Fock program for diatomic molecules in strong
   laser fields,” *Computer Physics Communications* **194**, 84--96 (2015).
   [doi:10.1016/j.cpc.2015.04.008](https://doi.org/10.1016/j.cpc.2015.04.008)
5. T. N. Rescigno and C. W. McCurdy, “Numerical grid methods for
   quantum-mechanical scattering problems,” *Physical Review A* **62**,
   032706 (2000).
   [doi:10.1103/PhysRevA.62.032706](https://doi.org/10.1103/PhysRevA.62.032706)
