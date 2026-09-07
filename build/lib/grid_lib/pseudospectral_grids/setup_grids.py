"""
Setup functions for pseudospectral grids.

Provides convenient interfaces for creating different types of pseudospectral grids.
"""

import numpy as np
from typing import Optional, Union, Dict, Any

from .sinc_dvr import SincDVR
from .gauss_legendre_lobatto import GaussLegendreLobatto, Linear_map, Rational_map
from .femdvr import FEMDVR


def setup_grid(
    grid_type: str,
    grid_params: Optional[Dict[str, Any]] = None,
) -> Union[SincDVR, GaussLegendreLobatto, FEMDVR]:
    """Set up a pseudospectral grid of the specified type.

    Parameters
    ----------
    grid_type : str
        Type of grid to create. Options: 'sinc', 'gll', 'femdvr'
    grid_params : dict, optional
        Grid-specific parameters. See notes for details on each grid type.

    Returns
    -------
    grid : SincDVR, GaussLegendreLobatto, or FEMDVR
        The created grid object.

    Notes
    -----
    For 'sinc' grids, grid_params should contain:
        - 'x0' (float): Left endpoint. Default: -10
        - 'xN' (float): Right endpoint. Default: 10
        - 'N' (int): Number of grid points. Default: 40

    For 'gll' grids, grid_params should contain:
        - 'N' (int): Number of grid points minus 1 (polynomial degree)
        - 'x0' (float): Left endpoint for default linear mapping. Default: 0
        - 'xN' (float): Right endpoint for default linear mapping. Default: 30
        - 'Mapping' (optional): Mapping object. Default: Linear_map()
        - 'symmetrize' (bool): Whether to symmetrize weights. Default: False

    For 'femdvr' grids, grid_params should contain:
        - 'nodes' (ndarray): Element boundary nodes
        - 'n_points' (ndarray): Number of points per element
        - 'Mapping' (optional): Mapping class. Default: Linear_map
        - 'element_class' (optional): Element class. Default: GaussLegendreLobatto
        - 'symmetrize' (bool): Whether to symmetrize. Default: False

    Examples
    --------
    >>> # Create a sinc grid with default parameters
    >>> grid = setup_grid('sinc')

    >>> # Create a sinc grid with custom parameters
    >>> grid = setup_grid('sinc', {'x0': -5, 'xN': 5, 'N': 50})

    >>> # Create a GLL grid with 50 points
    >>> grid = setup_grid('gll', {'N': 49})

    >>> # Create a FEM-DVR grid
    >>> nodes = np.array([-10, -5, 0, 5, 10])
    >>> n_points = np.array([11, 11, 11, 11])
    >>> grid = setup_grid('femdvr', {'nodes': nodes, 'n_points': n_points})
    """
    if grid_params is None:
        grid_params = {}

    grid_type = grid_type.lower()

    if grid_type == 'sinc':
        return _setup_sinc(grid_params)
    elif grid_type == 'gll':
        return _setup_gll(grid_params)
    elif grid_type == 'femdvr':
        return _setup_femdvr(grid_params)
    else:
        raise ValueError(
            f"Unknown grid_type: {grid_type}. "
            "Supported types are: 'sinc', 'gll', 'femdvr'"
        )


def _setup_sinc(params: Dict[str, Any]) -> SincDVR:
    """Create a sinc DVR grid."""
    x0 = params.get('x0', -10)
    xN = params.get('xN', 10)
    N = params.get('N', 40)

    return SincDVR(x0, xN, N)


def _setup_gll(params: Dict[str, Any]) -> GaussLegendreLobatto:
    """Create a Gauss-Legendre-Lobatto grid."""
    if 'N' not in params:
        raise ValueError("'N' (polynomial degree) is required for 'gll' grid")

    N = params['N']
    x0 = params.get('x0', params.get('r_min', 0))
    xN = params.get('xN', params.get('r_max', 30))
    Mapping = params.get('Mapping', Linear_map(x0, xN))
    symmetrize = params.get('symmetrize', False)

    return GaussLegendreLobatto(N, Mapping, symmetrize=symmetrize)


def _setup_femdvr(params: Dict[str, Any]) -> FEMDVR:
    """Create a FEM-DVR grid."""
    if 'nodes' not in params or 'n_points' not in params:
        raise ValueError(
            "'nodes' and 'n_points' are required for 'femdvr' grid"
        )

    nodes = params['nodes']
    n_points = params['n_points']
    Mapping = params.get('Mapping', Linear_map)
    element_class = params.get('element_class', GaussLegendreLobatto)
    symmetrize = params.get('symmetrize', False)

    return FEMDVR(nodes, n_points, Mapping, element_class, symmetrize=symmetrize)


def setup_femdvr_uniform(
    x_min: float,
    x_max: float,
    n_elements: int,
    points_per_element: int,
    Mapping: Optional[type] = None,
    symmetrize: bool = False,
) -> FEMDVR:
    """Set up a FEM-DVR grid with uniform element spacing and point distribution.

    This is a convenience function for creating FEM-DVR grids where all elements
    have the same number of grid points.

    Parameters
    ----------
    x_min : float
        Left endpoint of the domain
    x_max : float
        Right endpoint of the domain
    n_elements : int
        Number of elements
    points_per_element : int
        Number of grid points in each element (must be >= 2)
    Mapping : type, optional
        Mapping class to use for coordinate transformation. Default: Linear_map
    symmetrize : bool, optional
        Whether to symmetrize the weights. Default: False

    Returns
    -------
    grid : FEMDVR
        The created FEM-DVR grid object.

    Raises
    ------
    ValueError
        If points_per_element < 2

    Examples
    --------
    >>> # Create a FEM-DVR grid with 3 elements and 11 points per element
    >>> grid = setup_femdvr_uniform(-10, 10, 3, 11)

    >>> # Create with rational mapping
    >>> from grid_lib.pseudospectral_grids import Rational_map
    >>> grid = setup_femdvr_uniform(-10, 100, 5, 15, Mapping=Rational_map)
    """
    if points_per_element < 2:
        raise ValueError("points_per_element must be >= 2")

    if Mapping is None:
        Mapping = Linear_map

    # Create uniform element boundaries
    nodes = np.linspace(x_min, x_max, n_elements + 1)

    # Create uniform point distribution across all elements
    n_points = np.full(n_elements, points_per_element, dtype=int)

    return setup_grid(
        'femdvr',
        {
            'nodes': nodes,
            'n_points': n_points,
            'Mapping': Mapping,
            'element_class': GaussLegendreLobatto,
            'symmetrize': symmetrize,
        },
    )
