# Spherical Coordinates

The spherical coordinate system is defined by

$$
\begin{aligned}
x &= r\sin\theta\cos\phi \\
y &= r\sin\theta\sin\phi \\
z &= r\cos\theta
\end{aligned}
$$

with domains $r \in [0,\infty)$, $\theta \in [0,\pi]$, and $\phi \in [0,2\pi)$.

The volume element is

$$
dV = r^2\sin\theta\,dr\,d\theta\,d\phi.
$$

The Laplacian is

$$
\nabla^2 = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial}{\partial r}\right)
+ \frac{1}{r^2}\left[\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right)
+ \frac{1}{\sin^2\theta}\frac{\partial^2}{\partial\phi^2}\right].
$$

## Wavefunction parametrization

A common expansion is

$$
\Psi(\mathbf{r}) = \sum_{l=0}^{l_{\max}}\sum_{m=-l}^{l}\frac{u_{l,m}(r)}{r}Y_{l,m}(\theta,\phi).
$$

This decomposition is the basis for several operators implemented in the spherical-coordinate modules.

## Gauss-Legendre-Fourier angular grid

`GaussLegendreFourierGrid` combines Gauss-Legendre nodes in
`cos(theta)` with equally spaced periodic nodes in `phi`. It is independent
of the radial grid, which may use Gauss-Legendre-Lobatto finite elements.

```python
import numpy as np
from grid_lib.spherical_coordinates import GaussLegendreFourierGrid

grid = GaussLegendreFourierGrid(n_theta=12, n_phi=24)
theta, phi = grid.mesh()
integral = grid.integrate(np.cos(theta)**2)  # 4*pi/3
```

The mesh and combined weights have shape `(n_theta, n_phi)`. The cosine
nodes are ascending, so the corresponding `theta` nodes are descending.
The poles and the duplicate azimuthal endpoint at `2*pi` are excluded.
`theta_weights` are the ordinary Gauss-Legendre weights on `[-1, 1]`;
the combined `weights` sum to `4*pi` and already represent the measure
`sin(theta) dtheta dphi`. Do not apply another `sin(theta)` or `4*pi` factor.

`integrate` sums over the last two axes, preserving leading batch axes and
complex values. Scalars and singleton angular axes are broadcast; for a
polar-only function use shape `(n_theta, 1)`. Nodal values are unweighted.
For consumers expecting one-dimensional arrays, flatten both mesh arrays
and `grid.weights` in the same order.

`GaussLegendreFourierGrid.from_bandlimit(l_max)` chooses `l_max + 1` polar
nodes and `2*l_max + 1` azimuthal nodes, sufficient to analyze a function
whose spherical-harmonic expansion ends at `l_max`. Products of two such
expansions can extend to `2*l_max`, so use `from_bandlimit(2*l_max)` when
that product is the function being integrated. Arbitrary non-bandlimited
functions require quadrature convergence checks.
The grid does not implement spherical-harmonic transforms or change the
existing Lebedev-based matrix-element functions.

## Nodal gradient and divergence

`SphericalNodalDifferentialOperator` acts on ordinary scalar nodal values
with shape `(..., n_r, n_theta, n_phi)`. It combines the radial grid's first
derivative matrix, the Gauss-Legendre derivative in $x=\cos\theta$, and a
periodic Fourier derivative in $\phi$. The physical gradient components are

$$
\nabla f = \boldsymbol e_r\frac{\partial f}{\partial r}
+ \boldsymbol e_\theta\frac{1}{r}\frac{\partial f}{\partial\theta}
+ \boldsymbol e_\phi\frac{1}{r\sin\theta}\frac{\partial f}{\partial\phi}.
$$

For orthonormal spherical components
$\boldsymbol A=(A_r,A_\theta,A_\phi)$, the strong divergence is

$$
\nabla\mathbin{\cdot}\boldsymbol A =
\frac{1}{r^2}\frac{\partial}{\partial r}(r^2 A_r)
+ \frac{1}{r\sin\theta}\frac{\partial}{\partial\theta}(\sin\theta A_\theta)
+ \frac{1}{r\sin\theta}\frac{\partial A_\phi}{\partial\phi}.
$$

The class also provides the quadrature-weighted adjoint of the gradient.
This gives the discretely variational counterpart of minus the divergence,
which is useful for functionals depending on density gradients.

All active radial nodes must be strictly positive. For an FEM-DVR grid with
an endpoint at the origin, select an active radial subset explicitly.
Differential methods may still receive values on the complete radial grid;
the endpoint values then contribute to derivatives evaluated at active nodes.
Passing only active values instead imposes zero at omitted nodes.
