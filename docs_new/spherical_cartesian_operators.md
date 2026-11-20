# Cartesian operators and rotations in spherical expansions

## Scope

`SphericalReducedRadialOperators` applies Cartesian position and momentum
operators to reduced-radial complex spherical-harmonic expansions. The public
`SphericalHarmonicRotation` class applies active spatial rotations to the same
angular channel convention.

These operations support single-centre representations of arbitrarily
oriented multicentre systems. They do not choose a molecular origin or move
nuclei; those remain application-level decisions.

## State convention

The wavefunction is

```math
\Psi(\mathbf r)
=
\frac{1}{r}
\sum_{l=0}^{l_{\max}}
\sum_{m=-m_{\max}}^{m_{\max}}
u_{lm}(r)Y_{lm}(\hat{\mathbf r}),
```

with terms satisfying `|m| <= l`. Arrays have radial-first trailing shape

```text
(..., n_r, n_lm).
```

Channels use the standard grid-lib ordering: increasing signed `m` first,
then increasing `l` within each `m` sector.

## Cartesian coordinate operators

The coordinate operators are

```math
x=r\sin\theta\cos\phi,
\qquad
y=r\sin\theta\sin\phi,
\qquad
z=r\cos\theta.
```

Their angular matrices are constructed from the normalized rank-one
spherical tensors

```math
C_{1q}(\hat{\mathbf r})
=
\sqrt{\frac{4\pi}{3}}Y_{1q}(\hat{\mathbf r}).
```

Using the Condon--Shortley convention,

```math
\frac{x}{r}=\frac{C_{1,-1}-C_{1,1}}{\sqrt{2}},
```

```math
\frac{y}{r}=\frac{i(C_{1,-1}+C_{1,1})}{\sqrt{2}},
\qquad
\frac{z}{r}=C_{1,0}.
```

The public actions are `apply_x`, `apply_y`, and `apply_z`. Transverse
operators change `m` by one and therefore require full-angular states. The
axial `apply_z_fixed_m` action remains available for an isolated signed-`m`
block.

## Cartesian momentum operators

For reduced radial functions, write

```math
\nabla
=
\hat{\mathbf r}\,\partial_r
-\frac{i}{r}\hat{\mathbf r}\times\mathbf L.
```

The reduced-radial action of `p_i=-i partial_i` is then

```math
p_i
=
-iN_i\left(\partial_r-\frac{1}{r}\right)
-\frac{(\hat{\mathbf r}\times\mathbf L)_i}{r},
```

where `N_i=r_i/r`. `SphericalReducedRadialOperators` combines these sparse
angular matrices with the supplied strong first-derivative matrix.

The public actions are `apply_px`, `apply_py`, and `apply_pz`. The direct
axial formula is retained for `apply_pz` and `apply_pz_fixed_m`, so truncated
or isolated magnetic spaces do not acquire an artificial dependence on
intermediate transverse channels.

Coordinate and momentum actions preserve arbitrary leading batch axes and
accept real or complex input. Hermiticity is with respect to the radial
quadrature metric associated with the derivative matrix supplied by the
caller.

## Active spherical-harmonic rotations

`SphericalHarmonicRotation` represents a proper Cartesian rotation
`R` through

```math
(U(R)\Psi)(\mathbf r)=\Psi(R^{-1}\mathbf r).
```

For each angular momentum `l`, its channel block is

```math
D^l(R)
=
\exp[-i\boldsymbol\omega\cdot\mathbf L^{(l)}],
```

where `omega` is the Cartesian rotation vector. The implementation assembles
these blocks into grid-lib's signed-`m`-then-`l` ordering.

A general rotation requires the complete space

```text
m = -l, ..., l
```

for every retained `l`. The resulting channel count is `(l_max + 1)**2`.
The class therefore always represents a complete magnetic space and returns
a unitary channel matrix up to floating-point roundoff.

For Cartesian vector operators, the convention implies

```math
U(R)^{\dagger}O_iU(R)
=
\sum_j R_{ij}O_j.
```

This identity is tested for the coordinate direction matrices and both
angular factors of the momentum operator.

## Example

```python
import numpy as np

from grid_lib.spherical_coordinates import (
    SphericalHarmonicRotation,
    SphericalReducedRadialOperators,
)

operators = SphericalReducedRadialOperators(
    radial_points,
    first_derivative,
    l_max=8,
    m_max=8,
)

x_psi = operators.apply_x(psi)
py_psi = operators.apply_py(psi)

rotation = SphericalHarmonicRotation.from_axis_angle(
    axis=[0.0, 1.0, 0.0],
    angle=np.pi / 4.0,
    l_max=8,
)
rotated_psi = rotation(psi)
```

The rotation acts on the final angular-channel axis, so the same object also
accepts arrays with radial, state, or other leading dimensions.

## Validation

The regression suite checks:

- analytic `s`-orbital actions for `x`, `y`, `p_x`, and `p_y`;
- the identity `x**2 + y**2 + z**2 = r**2` away from the angular cutoff;
- FEM-DVR metric Hermiticity of all six Cartesian actions;
- agreement of the unchanged axial action with the legacy implementation;
- exact axial rotation phases `exp(-i*m*angle)`;
- unitarity, inversion, and composition of general rotations;
- Cartesian-vector covariance under a general three-dimensional rotation;
- validation of improper, nonorthogonal, and malformed rotations.
