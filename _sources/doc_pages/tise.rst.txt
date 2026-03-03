.. _tise-label:

The time-independent Schrödinger equation
#########################################

The time-independent Schrödinger equation

.. math::
    \left(-\frac{1}{2}\nabla^2 + V(\mathbf{r}) \right) \Psi_k(\mathbf{r}) = E_k \Psi_k(\mathbf{r}), \ \ k=1,2,3,\cdots

For a spherical symmetric potential :math:`V(r)`, the eigenfunctions can be taken as 

.. math::
    \Psi_{n,l,m}(\mathbf{r}) = r^{-1} u_{n,l}(r) Y_{l,m}(\theta, \phi),

and the (radial) TISE becomes 

.. math::

    -\frac{1}{2}\frac{d^2 u_{n,l}(r)}{d r^2}+\frac{l(l+1)}{2 r^2} u_{n,l}(r) + V(r)u_{n,l} = \epsilon_{n,l} u_{n,l}(r).

Gauss-Legendre-Lobatto quadrature is defined on :math:`x \in [-1,1]`. 
We need to map the grid/Lobatto points :math:`x_i \in [-1,1]` into radial points :math:`r(x): [-1,1] \rightarrow [0, r_{\text{max}}]`. 
In that case, :math:`u_{n,l}(r) = u_{n,l}(r(x))`.

The (rational) mapping 

.. math::
    
    r(x) = L \frac{1+x}{1-x+\alpha}, \ \ \alpha = \frac{2L}{r_{\text{max}}} \label{x_to_r},

has often been used in earlier work. 
The parameters :math:`L` and :math:`\alpha` control the length of the grid and the density of points near :math:`r=0`. 

Another option is to use the linear mapping given by 

.. math::

    r(x) = \frac{r_{\text{max}}}{2}(x+1).

In order to use the Gauss-Legendre-Lobatto pseudospectral method, we must first formulate 
the radial TISE with respecto to :math:`x`.

By the chain rule we find that for an arbitrary function :math:`\psi(r(x))`

.. math::
    
    \frac{d \psi}{dr} &= \frac{1}{\dot{r}(x)} \frac{d \psi}{dx}, \\
    \frac{d^2 \psi}{dr^2} &= \frac{1}{\dot{r}(x)^2} \frac{d^2 \psi}{dx^2} - \frac{\ddot{r}(x)}{\dot{r}(x)^3} \frac{d \psi}{dx},

and the radial TISE becomes

.. math::
    :name: eq:unsymmetric_radial_TISE
    
    -\frac{1}{2} \left( \frac{1}{\dot{r}(x)^2} \frac{d^2 u_{n,l}}{dx^2} - \frac{\ddot{r}(x)}{\dot{r}(x)^3} \frac{d u_{n,l}}{dx} \right) + V_l(r(x)) u_{n,l}(r(x)) = \epsilon_{n,l} u_{n,l}(r(x)),
    
where we have defined :math:`V_l(r(x)) \equiv V(r(x)) + \frac{l(l+1)}{2 r(x)^2}`. Note that this is, in general, an unsymmetric eigenvalue problem 
due to the presence of the term :math:`\frac{\ddot{r}(x)}{\dot{r}(x)^3}`.

In order to reformulate the above as a symmetric eigenvalue problem, we define the new wavefunction :math:`\phi_{n,l}(x) = \dot{r}(x)^{1/2} u_{n,l}(r(x))`.

Insertion of this into Eq. :ref:`Link title <eq:unsymmetric_radial_TISE>` yields a symmetric eigenvalue problem

.. math::
    :label: symmetric_radial_TISE

    \left(-\frac{1}{2} \frac{1}{\dot{r}(x)} \frac{d^2}{dx^2} \frac{1}{\dot{r}(x)} + V_l(r(x))+\tilde{V}(r(x)) \right) \phi_{n,l}(x) = \epsilon_{n,l} \phi_{n,l}(x),
    

where we have defined 

.. math:: 
    
    \tilde{V}(x) \equiv \frac{2\dddot{r}(x)\dot{r}(x)-3\ddot{r}(x)^2}{4\dot{r}(x)^4}.
     
We discretize this equation with the pseudospectral method by expanding :math:`\phi_{n,l}(x)` and :math:`\phi_{n,l}(x)/\dot{r}(x)` as 

.. math::

    \phi_{n,l}(x) &= \sum_{j=0}^N \phi_{n,l}(x_j) g_j(x), \\
    \frac{\phi_{n,l}(x)}{\dot{r}(x)} &= \sum_{j=0}^N \frac{\phi_{n,l}(x_j)}{\dot{r}(x_j)} g_j(x).

Inserting these expansions into Eq. :math:ref:`symmetric_radial_TISE` we have that 

.. math::

     \sum_{j=0}^N \left(-\frac{1}{2} \frac{1}{\dot{r}(x)} \frac{\phi_{n,l}(x_j)}{\dot{r}(x_j)} g^{\prime \prime}_j(x) + V(r(x)) \phi_{n,l}(x_j) g_j(x) \right) = \epsilon_{n,l} \sum_{j=0}^N \phi_{n,l}(x_j) g_j(x)

Next, we multiply through with :math:`g_i(x)` and integrate over :math:`x`, 

.. math::

     \sum_{j=0}^N \left(-\frac{1}{2}  \frac{\phi_{n,l}(x_j)}{\dot{r}(x_j)} \int \frac{g_i(x)}{\dot{r}(x)} g^{\prime \prime}_j(x) dx +  \phi_{n,l}(x_j) \int g_i(x) V(r(x)) g_j(x) dx \right) = \epsilon_{n,l} \sum_{j=0}^N \phi_{n,l}(x_j) \int g_i(x) g_j(x) dx

The integrals are evaluated with by quadrature and using the property :math:`g_j(x_i) = \delta_{i,j}` we have that 

.. math::
    
    \int g_i(x) g_j(x) dx &= \sum_{m=0}^N g_i(x_m) g_j(x_m) w_m = \sum_{m=0} w_m \delta_{i, m} \delta_{j,m} = w_i \delta_{i,j}, \\
    \int \frac{g_i(x)}{\dot{r}(x)} g^{\prime \prime}_j(x) dx &= \sum_{m=0} g_i(x_m) V(r(x_m)) g_j(x_m) w_m = w_i V(r(x_i)) \delta_{i,j}, \\
    \int \frac{g_i(x)}{\dot{r}(x)} g^{\prime \prime}_j(x) dx & \underbrace{=}_{???} \sum_{m=1}^{N-1} \frac{g_i(x_m)}{\dot{r}(x_m)} g^{\prime \prime}_j(x_m) = w_i \frac{g^{\prime \prime}_j(x_i)}{\dot{r}(x_i)},  \ \ i=1,\cdots,N-1.


Thus, for the interior grid points,

.. math::
     
     \sum_{j=1}^{N-1} \left(-\frac{1}{2}  \frac{\phi_{n,l}(x_j)}{\dot{r}(x_j)} w_i \frac{g^{\prime \prime}_j(x_i)}{\dot{r}(x_i)} +  \phi_{n,l}(x_j) w_i V(r(x_i)) \delta_{i,j} \right) = \epsilon_{n,l} \sum_{j=1}^{N-1} \phi_{n,l}(x_j) w_i \delta_{i,j}. 

Using the expressions for the :math:`g_j^{\prime \prime}(x_i)` we can write this as (notice that the weights :math:`w_i` cancels)

.. math::
     
     \sum_{j=1}^{N-1} \left(-\frac{1}{2}  \frac{\tilde{g}^{\prime \prime}_j(x_i) P_N(x_i)}{\dot{r}(x_i) \dot{r}(x_j)} \frac{\phi_{n,l}(x_j)} {P_N(x_j)} \right) +  \phi_{n,l}(x_i) V(r(x_i))  = \epsilon_{n,l}  \phi_{n,l}(x_i). 

Furthermore, dividing through with :math:`P_N(x_i)`, we have that 

.. math::

     \sum_{j=1}^{N-1} \left(-\frac{1}{2}  \frac{\tilde{g}^{\prime \prime}_j(x_i)}{\dot{r}(x_i) \dot{r}(x_j)} \tilde{\phi}_{n,l}(x_j) \right) +   V(r(x_i))\tilde{\phi}_{n,l}(x_i)  = \epsilon_{n,l}  \tilde{\phi}_{n,l}(x_i),

where we have defined 

.. math::
    
    \tilde{\phi}_{n,l}(x_i) \equiv \frac{\phi_{n,l}(x_i)}{P_N(x_i)}.