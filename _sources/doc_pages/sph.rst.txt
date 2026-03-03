.. _sph-label:

Spherical harmonics
###################

We use the convention where the spherical harmonics :math:`Y_l^m(\theta, \phi)` are defined as

.. math::

    Y_l^m(\theta, \phi) = (-1)^m\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}P_l^m(\cos\theta)e^{im\phi},

where :math:`P_l^m(x)` are the associated Legendre functions

.. math::

    P_l^m(x) = (1-x^2)^{|m|/2} \left(\frac{d}{dx}\right)^{|m|} P_l(x)

and :math:`P_l(x)` are the Legendre polynomials

.. math::

    P_l(x) = \frac{1}{2^l l!} \left( \frac{d}{dx} \right)^l (x^2-1)^l

In the following we use the notation 

.. math::

    Y_l^m(\theta, \phi) = Y_l^m(\Omega)

where :math:`\Omega = (\theta, \phi)`.


Properties
==========

.. math::

    \int (Y_l^m)^*(\Omega)Y_{l^\prime}^{m^\prime}(\Omega) d\Omega = \delta_{l l^\prime}\delta_{m m^\prime}

.. math::

    (Y_l^m)^*(\Omega) = (-1)^{m}Y_l^{-m}(\Omega)

.. math::

    \sum_{m=-l}^l |Y_l^m(\Omega)|^2 = \frac{2l+1}{4 \pi}

The product of two spherical harmonics can be expressed as 

.. math::

    Y_{l^\prime,m^\prime}(\Omega)Y_{l,m}(\Omega) = \sqrt{\frac{(2l^\prime+1)(2l+1)}{4\pi}}\sum_{L=0}^{\infty}\sum_{M=-L}^{L}(-1)^M\sqrt{2L+1}
    \begin{pmatrix}
        l^\prime & l & L \\
        m^\prime & m & -M 
    \end{pmatrix}
    \begin{pmatrix}
        l^\prime & l & L \\
        0 & 0 & 0 
    \end{pmatrix}
    Y_{L,M}(\Omega),

where :math:`\begin{pmatrix} l^\prime & l & L \\ m^\prime & m & -M \end{pmatrix}` are the Wigner 3j symbols https://mathworld.wolfram.com/Wigner3j-Symbol.html.

The effect of operators on spherical harmonics
----------------------------------------------

.. math::

    \frac{\partial}{\partial \phi}Y_{l,m}(\Omega) &= imY_{l,m}(\Omega), \\
    \frac{\partial}{\partial \theta}Y_{l,m}(\Omega) &= m \frac{\cos{\theta}}{\sin{\theta}}Y_{l,m}(\Omega) + \sqrt{(l-m)(l+m+1)}e^{-i\phi}Y_{l,m+1}(\Omega)

Trigonometric functions can be written in terms of spherical harmonics, and their actions on spherical harmonics can then be evaluated using 
the product rule for two spherical harmonics. Some important relations are

.. math::

    \cos{\theta} &= 2\sqrt{\frac{\pi}{3}}Y_{1,0} \\
    \sin{\theta}e^{-i\phi} &= 2\sqrt{\frac{2\pi}{3}}Y_{1,-1} \\
    \sin{\theta}\cos{\phi} &= \sqrt{\frac{2\pi}{3}}\Bigl[ Y_{1,-1} - Y_{1,1} \Bigr] \\
    \sin{\theta}\sin{\phi} &= \sqrt{\frac{2\pi}{3}}i\Bigl[ Y_{1,-1} + Y_{1,1} \Bigr] 

The action of the Cartesian position and derivative operators on the spherical harmonics are given by

.. math::

    xY_{l,m} &= \frac{1}{2}r\Bigl[\Bigr.b_{l-1,-m-1}Y_{l-1,m+1} - b_{l,m}Y_{l+1,m+1} - b_{l-1,m-1}Y_{l-1,m-1} + b_{l,-m}Y_{l+1,m-1}\Bigl. \Bigr], \\
    yY_{l,m} &= \frac{i}{2}r\Bigl[\Bigr.b_{l-1,-m-1}Y_{l-1,m+1} - b_{l,m}Y_{l+1,m+1} + b_{l-1,m-1}Y_{l-1,m-1} - b_{l,-m}Y_{l+1,m-1}\Bigl. \Bigr], \\
    zY_{l,m} &= r\Bigl[a_{l,m}Y_{l+1,m} + a_{l-1,m}Y_{l-1,m}\Bigr],

and

.. math::
    
    \frac{\partial}{\partial x} Y_{l,m} &= \cos{\phi}\sin{\theta}Y_{l,m}\frac{\partial}{\partial r} 
    + \frac{1}{2} \left[\cos\theta \left( c_{l,m}Y_{l,m+1} - c_{l,m-1}Y_{l,m-1} \right) + m \sin \theta (e^{-i \phi}-e^{i\phi}) Y_{l,m} \right] \frac{1}{r}, \\
    \frac{\partial}{\partial y} Y_{l,m} &= \sin{\phi}\sin{\theta}Y_{l,m}\frac{\partial}{\partial r} 
    + \frac{i}{2} \left[-\cos\theta \left( c_{l,m}Y_{l,m+1} + c_{l,m-1}Y_{l,m-1} \right) + m \sin \theta (e^{-i \phi}+e^{i\phi}) Y_{l,m} \right] \frac{1}{r}, \\
    \frac{\partial}{\partial z} Y_{l,m} &= \bigl(a_{l,m}Y_{l+1,m} + a_{l-1,m}Y_{l-1,m}\bigr)\frac{\partial}{\partial r} 
    + \Bigl[ -la_{l,m}Y_{l+1,m}  + (l+1)a_{l-1,m}Y_{l-1,m}\Bigr] \frac{1}{r},
  
where we have defined 

.. math::

    a_{l,m} &= \sqrt{\frac{(l+1)^2-m^2}{(2l+1)(2l+3)}}, \\
    b_{l,m} &= \sqrt{\frac{(l+m+1)(l+m+2)}{(2l+1)(2l+3)}}, \\
    c_{l,m} &= \sqrt{l(l+1) - m(m+1)}.


Special cases of Wigner coefficients
====================================

Trigonometric operators can typically be expressed in terms of spherical harmonics with small quantum numbers. 
In these cases the Wigner coefficients can be evaluated explicitly. We give closed form expressions to some simplified Wigner coefficients that can be useful. 
First, we repeat equation (13) in https://mathworld.wolfram.com/Wigner3j-Symbol.html in relevant notation:

.. math::
    
   \begin{pmatrix}
        l^\prime & l & l^\prime+l \\
        m^\prime & m & -M 
    \end{pmatrix} 
    = (-1)^{l'-l+M}\Biggl[ \frac{(2l')!(2l)!}{(2l'+2l+1)!}\frac{(l'+l+M)!(l'+l-M)!}{(l'+m')!(l'-m')!(l+m)!(l-m)!} \Biggr]^{1/2}

Integrals over spherical harmonics 
==================================

Using the product formula for the spherical harmonics it follows that 

.. math::

    \int Y_{l',m'}(\Omega) Y_{l,m}(\Omega) Y_{L,M}^*(\Omega) d\Omega = (-1)^M \sqrt{\frac{(2l'+1)(2l+1)(2L+1)}{4\pi}}
    \begin{pmatrix}
        l^\prime & l & L \\
        m^\prime & m & -M 
    \end{pmatrix}
    \begin{pmatrix}
        l^\prime & l & L \\
        0 & 0 & 0 
    \end{pmatrix}


.. math:: 

    \int Y_{l^\prime, m^\prime}^*(\Omega) x Y_{l,m}(\Omega) d\Omega &= \frac{1}{2}r\Bigl[\Bigr.b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1} 
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1} 
    - b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1}
    + b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1}\Bigl. \Bigr], \\
    \int Y_{l^\prime, m^\prime}^*(\Omega) y Y_{l,m}(\Omega) d\Omega &= \frac{i}{2}r\Bigl[\Bigr.b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1}
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1}
    + b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1} 
    - b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1} \Bigl. \Bigr], \\
    \int Y_{l^\prime, m^\prime}^*(\Omega) z Y_{l,m}(\Omega) d\Omega &= r\Bigl[a_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m} 
    + a_{l-1,m}\delta_{l^\prime, l-1}\delta_{m^\prime, m}\Bigr],
