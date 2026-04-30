# Spherical Harmonics

The library uses the convention

$$
Y_l^m(\theta,\phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi}.
$$

Key identities:

$$
\int (Y_l^m)^*(\Omega)Y_{l'}^{m'}(\Omega)\,d\Omega = \delta_{ll'}\delta_{mm'},
$$

$$
(Y_l^m)^*(\Omega) = (-1)^mY_l^{-m}(\Omega).
$$

The product of two spherical harmonics can be expressed through Wigner 3j symbols, which underpins the angular coupling formulas used in dipole and interaction operators.

For derivations and extended identities, see the historical source page in `docs/doc_pages/sph.rst`.
