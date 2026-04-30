# Time-Independent Schrodinger Equation (TISE)

For a spherically symmetric potential,

$$
\left(-\frac{1}{2}\nabla^2 + V(\mathbf{r})\right)\Psi_k(\mathbf{r}) = E_k\Psi_k(\mathbf{r}).
$$

Using radial functions $u_{n,l}(r)$, the radial equation is

$$
-\frac{1}{2}\frac{d^2u_{n,l}}{dr^2} + \frac{l(l+1)}{2r^2}u_{n,l} + V(r)u_{n,l} = \epsilon_{n,l}u_{n,l}.
$$

The implementation maps Gauss-Legendre-Lobatto points from $x \in [-1,1]$ to radial coordinates $r(x)$, then discretizes the transformed equation using cardinal functions and differentiation matrices.

This yields the matrix eigenvalue problems used in stationary-state solvers in the library.
