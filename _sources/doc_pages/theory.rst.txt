Theory
######

The pseudospectral method
=========================
In the pseudospectral method a function :math:`f(x)` is approximated as a linear combination

.. math::
    
    f(x) \simeq f_N(x) = \sum_{j=0}^N f(x_j) g_j(x)


of cardinal functions :math:`g_j(x)` where the expansion coefficients are values of 
:math:`f(x)` at the grid points :math:`x_j`. For Lobatto-type grids, the interior grid points 
:math:`\{ x_j \}_{j=1}^{N-1}` are defined as the zeros of :math:`P_N^\prime(x)`, where :math:`P_N(x)` is a 
:math:`N`-th order polynomial [#f1]_,

.. math::
    
    P_N^\prime(x_j) = 0, \ \ j=1,\cdots,N-1,


while the end (boundary) points :math:`x_0` and :math:`x_N` are given.

The cardinal functions :math:`g_j(x)` have the property that

.. math::
    
    g_j(x_i) = \delta_{ij}, \label{delta_property_g}


and can be written as 

.. math::
    
    g_j(x) = -\frac{1}{N(N+1)P_N(x_j)}\frac{(1-x^2)P_N^\prime(x)}{x-x_j}.


The integral of a function is approximated by quadrature

.. math::
    
    \int_{x_0}^{x_N} f(x) dx \simeq \sum_{j=0}^N w_j f(x_j), \label{quadrature_rule}


where :math:`w_j` are the quadrature weights.

.. rubric:: Footnotes

.. [#f1] Which type of polynomial to choose depends on the problem. Examples include Chebyshev, Legendre, Laguerre or Hermite polynomials.


Gauss-Legendre-Lobatto
=======================

In the Gauss-Legendre-Lobatto pseudospectral method, :math:`P_N(x)` is taken as the :math:`N`-th 
order Legendre polynomial. The end points are :math:`x_0 = -1` and :math:`x_N=1`, while the interior 
grid points must be found as the roots 

.. math::
    
    P_N^\prime(x_i) = 0,

and the quadrature weights are given by

.. math::
    
    w_i = \frac{2}{N(N+1)P_N(x_i)^2}.

The grid points and weights can be determined using numpy functions (REF).
Furthermore, 

.. math::
    
    
    \frac{d g_j}{dx} \Bigr|_{x=x_i} &= g_j^\prime(x_i) = \tilde{g}_j^\prime(x_i) \frac{P_N(x_i)}{P_N(x_j)}, \\
    \tilde{g}_j(x_i) &= 
    \begin{cases}
    \frac{1}{4}N(N+1), \ \ i=j=0, \\
    -\frac{1}{4}N(N+1), \ \ i=j=N, \\
    0, \ \ i=j \text{ and } 1 \leq j \leq N-1, \\
    \frac{1}{x_i-x_j}, \ \ i \neq j,
    \end{cases}
    

and for :math:`i,j = 1,\cdots,N-1`

.. math::
    
    \frac{d^2 g_j}{dx^2} \Bigr|_{x=x_i} &= g_j^{\prime \prime}(x_i) = \tilde{g}_j^{\prime \prime}(x_i) \frac{P_N(x_i)}{P_N(x_j)}, \\
    \tilde{g}_j^{\prime \prime}(x_i) &= 
    \begin{cases}
    -\frac{1}{3} \frac{N(N+1)}{(1-x_i^2)}, & i = j,\\
    -\frac{2}{(x_i-x_j)^2}, & i \neq j.
    \end{cases}

