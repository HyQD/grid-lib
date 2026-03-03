.. _tdse-label:

The time-dependent Schrödinger equation
#######################################

The time-dependent Schrödinger equation
=======================================

The time-dependent Schrödinger equation for a single particle moving in a potential :math:`V(\mathbf{r})` interacting with a (classical) electromagnetic field given by the vector potential 
:math:`A(\mathbf{r},t)` is, in the Coulomb gauge (:math:`\nabla \cdot A(\mathbf{r},t)=0`), given by 

.. math::

    i \dot{\Psi}(\mathbf{r}, t) &= \left( \frac{1}{2} \left( \hat{p} + A(\mathbf{r},t) \right)^2 + V(\mathbf{r}) \right) \Psi(\mathbf{r}, t) \\
    &= \left(-\frac{1}{2} \nabla^2 + A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2 + V(\mathbf{r}) \right) \Psi(\mathbf{r}, t)

In spherical coordinates, we parametrize time-dependent wavefunction as 

.. math::
    
    \Psi(\mathbf{r},t) = \sum_{l=0}^L \sum_{m=-l}^l \frac{u_{l,m}(r,t)}{r} Y_l^m(\theta, \phi).


Inserting this ansatz into to TDSE yields 

.. math::

    i \sum_{l,m} \frac{1}{r} \dot{u}_{l,m}(r,t) Y_l^m(\Omega) = \sum_{l,m} \left( \frac{1}{r}\left( -\frac{1}{2}\frac{d^2u_{l,m}(r,t)}{dr^2} + V_l(r)u_{l,m}(r,t) \right)  Y_l^m(\Omega) \right) 
    + V_I(\mathbf{r}, t) \Psi(\mathbf{r}, t), 

where we have defined the time-dependent interaction potential as 

.. math::

    V_I(\mathbf{r}, t) = A(\mathbf{r},t) \cdot \hat{p} + \frac{1}{2}A(\mathbf{r},t)^2.

Multiplying through with :math:`r` and :math:`Y_{l^\prime, m^\prime}^*(\Omega)` and integrating over :math:`\Omega` yields equations of motion for :math:`u_{l,m}(r,t)`,

.. math::
    
    i \dot{u}_{l^\prime,m^\prime}(r,t) = \left( -\frac{1}{2}  \frac{d^2 u_{l^\prime,m^\prime}(r,t)}{dr^2} + V_{l^\prime}(r)u_{l^\prime,m^\prime}(r,t) \right)   
    + r \int Y_{l^\prime, m^\prime}^*(\Omega) V_I(\mathbf{r}, t) \Psi(\mathbf{r}, t) d\Omega 

Classical electromagnetic fields
================================



The dipole approximation
------------------------

.. math::

    A(\mathbf{r},t) &\approx A(t) \vec{u}, \\
    \mathcal{E}(\mathbf{r}, t) &\approx -\frac{\partial A(t)}{\partial t} \vec{u}.

Length gauge 
------------

.. math::

    V_I(\mathbf{r}, t) = \mathcal{E}(t) \vec{u} \cdot \mathbf{r}

Using the integrals over position operators given in (see :ref:`sph-label`) we have that 

.. math::

    i \dot{u}_{l^\prime,m^\prime}(r,t) = \left( -\frac{1}{2}  \frac{d^2 u_{l^\prime,m^\prime}(r,t)}{dr^2} + V_{l^\prime}(r)u_{l^\prime,m^\prime}(r,t) \right) 
    + r \mathcal{E}(t)
    \begin{cases}
     \frac{1}{2}\sum_{l, m} (b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1} 
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1} 
    - b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1}
    + b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1}) u_{l,m}(r,t), \text{ if } \vec{u} = \vec{e}_x \\
    \frac{i}{2}\sum_{l,m} (b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1}
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1}
    + b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1} 
    - b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1})  u_{l,m}(r,t), \text{ if } \vec{u} = \vec{e}_y \\
    \sum_{l,m} (a_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m} 
    + a_{l-1,m}\delta_{l^\prime, l-1}\delta_{m^\prime, m}) u_{l,m}(r,t), \text{ if } \vec{u} = \vec{e}_z
    \end{cases}

Similar to the approach we used for the TISE (see :ref:`tise-label`), we introduce the 
new wavefunction :math:`\phi_{l,m}(x) = \dot{r}(x)^{1/2}u_{l,m}(r(x))` and discretrize the EOMs with the pseudospectral method, which yields 

.. math::

    i \dot{\phi}_{l^\prime, m^\prime}(x_i) = 
    \sum_{j=1}^{N-1} \left(-\frac{1}{2}  \frac{\tilde{g}^{\prime \prime}_j(x_i)}{\dot{r}(x_i) \dot{r}(x_j)} \tilde{\phi}_{l^\prime,m^\prime}(x_j) \right) 
    +   V_{l^\prime}(r(x_i))\tilde{\phi}_{l^\prime,m^\prime}(x_i) 
    + r(x_i) \mathcal{E}(t)
    \begin{cases}
     \frac{1}{2}\sum_{l, m} (b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1} 
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1} 
    - b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1}
    + b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1}) \tilde{\phi}_{l,m}(x_i), \text{ if } \vec{u} = \vec{e}_x \\
    \frac{i}{2}\sum_{l,m} (b_{l-1,-m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m+1}
    - b_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m+1}
    + b_{l-1,m-1}\delta_{l^\prime, l-1}\delta_{m^\prime, m-1} 
    - b_{l,-m}\delta_{l^\prime, l+1}\delta_{m^\prime, m-1})  \tilde{\phi}_{l,m}(x_i), \text{ if } \vec{u} = \vec{e}_y \\
    \sum_{l,m} (a_{l,m}\delta_{l^\prime, l+1}\delta_{m^\prime, m} 
    + a_{l-1,m}\delta_{l^\prime, l-1}\delta_{m^\prime, m}) \tilde{\phi}_{l,m}(x_i), \text{ if } \vec{u} = \vec{e}_z
    \end{cases}

Velocity gauge
--------------

.. math::

    V_I(\mathbf{r}, t) = A(t) \cdot \hat{p} + \frac{1}{2}A(t)^2

Beyond dipole approximation
===========================