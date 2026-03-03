import numpy as np
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from typing import Type


class FEMDVR:
    """A FEMDVR class.

    The interface is very similar to the GaussLobattto classes.

    """

    def __init__(
        self,
        nodes,
        n_points,
        Mapping,
        element_class: Type[GaussLegendreLobatto],
        symmetrize=False,
    ):
        """Constructor.

        Args:
            nodes (ndarray): Vector of nodes of boundaries of elements. Number of elements is this len(nodes) - 1. nodes[i] and nodes[i+1] defines the boundary of element i.
            n_points (ndarray): Number of *total* grid points per element. The polynomial degree over  element i is n_points[i] - 1.
            element_class: Class for the elements. A subclass of GaussLobatto (which is an abstract base class).

        The constructor creates the following attributes:
            self.D (ndarray): Differentiation matrix
            self.D2 (ndarray): Second-order differentiation matrix
            self.R (ndarray): Matrix that identifies nodes at element interfaces.
            self.x (ndarray): Nodes
            self.w (ndarray): Weights
            self.dvr (list): List of dvr object instances for each element
            self.edge_indices (ndarray): Indices of elements of x corresponding to element boundaries

        """

        self.n_points = n_points
        self.nodes = nodes
        self.degrees = self.n_points - 1
        self.n_intervals = len(nodes) - 1
        n_intervals = self.n_intervals

        n_constraints = n_intervals - 1
        dim0 = np.sum(
            n_points
        )  # dimension of function space without boundary matching
        D = np.zeros((dim0, dim0))
        R = np.zeros((dim0, dim0 - n_constraints))
        w = np.zeros((dim0,))
        self.r = np.zeros((dim0 - n_constraints,))
        self.r_dot = np.zeros((dim0 - n_constraints,))

        # Extract intervals and generate individual DVRs.
        a = []
        b = []
        dim = []
        dvr = []
        block_start = [0]  # indices for start of original basis
        block_start2 = [0]  # indices for start of interface matched basis

        # Create DVRs over each element
        for i in range(n_intervals):
            a.append(nodes[i])
            b.append(nodes[i + 1])
            dvr.append(
                element_class(
                    n_points[i] - 1, Mapping(a[i], b[i]), symmetrize=symmetrize
                )
            )

        for i in range(n_intervals):
            # update dimension
            dim.append(n_points[i])
            # update block start indices
            # block_start[i] is the start of the original basis, block_start2[i]
            # is the start of the interface matched basis.
            block_start.append(block_start[i] + dim[i])
            block_start2.append(block_start2[i] + dim[i] - 1)

            # if this is the last interval, we need to add one more point to the interface matched basis
            # to account for the last point of the last element.
            if i == n_intervals - 1:
                block_start2[-1] += 1

            # Fill the differentiation matrix D and the weights w
            # D is the differentiation matrix in the original basis, w are the weights.
            D[
                block_start[i] : block_start[i + 1],
                block_start[i] : block_start[i + 1],
            ] += dvr[i].D1
            w[block_start[i] : block_start[i + 1]] = dvr[i].weights

            # Fill the interface matching matrix R
            # R is the matrix that identifies nodes at element interfaces.
            R[
                block_start[i] : block_start[i + 1],
                block_start2[i] : block_start2[i + 1] + 1,
            ] = np.eye(dim[i])

            # Update nodes x
            # x are the nodes of the interface matched basis.
            self.r[block_start2[i] : block_start2[i + 1]] = dvr[i].r[
                : block_start2[i + 1] - block_start2[i]
            ]
            self.r_dot[block_start2[i] : block_start2[i + 1]] = dvr[i].r_dot[
                : block_start2[i + 1] - block_start2[i]
            ]

        # Compute second derivative matrix.
        temp = D @ R
        D2_intermediate = (
            -temp.T @ np.diag(w) @ temp
        )  # D2 is the second derivative matrix
        D1_intermediate = R.T @ np.diag(w) @ D @ R
        self.S = R.T @ np.diag(w) @ R  # S is the mass matrix
        self.weights = w @ R

        # Compute differentiation matrices in the original basis.
        self.D1 = np.linalg.solve(self.S, D1_intermediate)
        self.D2 = np.linalg.solve(self.S, D2_intermediate)

        # S is diagonal. Compute square root of inverse of S directly.
        S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.S)))

        # Store symmetric versions of the differentiation matrices.
        self.D2_symmetric = S_inv_sqrt @ D2_intermediate @ S_inv_sqrt
        self.D1_symmetric = S_inv_sqrt @ D1_intermediate @ S_inv_sqrt

        self.R = R
        self.dvr = dvr
        self.edge_indices = block_start2.copy()
        self.edge_indices[-1] -= 1


if __name__ == "__main__":

    a = -10
    b = 10
    n_elem = 3
    points_per_elem = 21

    nodes = np.linspace(a, b, n_elem + 1)
    # nodes = np.array([-10, -2, 2, 10])
    n_points = np.ones((n_elem,), dtype=int) * points_per_elem

    femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)

    # Get nodes and weights and differentiation matrix.
    r = femdvr.r
    r_dot = femdvr.r_dot
    w = femdvr.weights
    D = femdvr.D1
    D2 = femdvr.D2
    ei = femdvr.edge_indices

    H_full = -0.5 * D2 + np.diag(0.5 * r**2)
    H = H_full[1:-1, 1:-1]

    E, U = np.linalg.eig(H)
    i = np.argsort(E)
    E = E[i]
    U = U[:, i]

    print(E[:5])

    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(3):
        y = np.zeros((len(r),))
        y[1:-1] = U[:, i]
        plt.plot(r, y, color=f"C{i}", marker=".", linestyle="-")
        plt.plot(r[ei], y[ei], color=f"C{i}", marker="o", linestyle="none")
    plt.show()
