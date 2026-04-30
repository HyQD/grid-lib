# Time-Dependent Schrodinger Equation (TDSE)

In atomic units and Coulomb gauge,

$$
i\dot{\Psi}(\mathbf{r},t) = \left(\frac{1}{2}(\hat{p}+A(\mathbf{r},t))^2 + V(\mathbf{r})\right)\Psi(\mathbf{r},t).
$$

Using spherical harmonics,

$$
\Psi(\mathbf{r},t)=\sum_{l,m}\frac{u_{l,m}(r,t)}{r}Y_l^m(\theta,\phi),
$$

which produces coupled radial equations with interaction terms for different gauges (length and velocity) and polarization directions.

The pseudospectral discretization of these equations forms the basis for propagators and right-hand-side builders in the spherical-coordinate TDSE modules.
