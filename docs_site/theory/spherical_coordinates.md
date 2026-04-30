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
