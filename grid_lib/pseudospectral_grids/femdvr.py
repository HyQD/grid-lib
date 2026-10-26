import numpy as np
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from .pseudospectral_grid import PseudospectralGrid
from scipy.sparse import coo_matrix, diags
import matplotlib.pyplot as plt
from typing import Type


class FEMDVR(PseudospectralGrid):
    """A FEMDVR class.

    The interface is very similar to the GaussLobattto classes.

    """

    def __repr__(self):
        return f"FEMDVR"

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
            self.element_dof_maps (tuple): Local-to-global degree-of-freedom maps
            self.bridge_indices (ndarray): Shared interior element-boundary indices
            self.dof_owner (ndarray): Owning element of every global degree of freedom
            self.element_owned_dof_masks (tuple): Locally owned entries of each map
            self.local_mass_matrices (tuple): Element mass matrices
            self.local_first_derivative_matrices (tuple): Element weak first derivatives
            self.local_stiffness_matrices (tuple): Element positive stiffness matrices

        """

        self.n_points = n_points
        self.nodes = nodes
        self.degrees = self.n_points - 1
        self.n_intervals = len(nodes) - 1
        n_intervals = self.n_intervals

        n_constraints = n_intervals - 1
        dim0 = np.sum(n_points)  # dimension of function space without boundary matching
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
        self.element_class = element_class

        # Retain the element topology and weak-form operators used above.  The
        # shared endpoint between adjacent elements has one global index and
        # therefore occurs in both neighboring element maps.
        self.element_dof_maps = tuple(
            np.arange(block_start2[i], block_start2[i] + dim[i], dtype=int)
            for i in range(n_intervals)
        )
        self.bridge_indices = np.asarray(self.edge_indices[1:-1], dtype=int)

        # Assign shared bridge degrees of freedom to the element on their left.
        # Ownership is only topology metadata; both elements still contribute
        # when a local operator is assembled or applied.
        self.dof_owner = np.full(self.r.size, -1, dtype=int)
        element_owned_dof_masks = []
        for element, dof_map in enumerate(self.element_dof_maps):
            owned = self.dof_owner[dof_map] < 0
            self.dof_owner[dof_map[owned]] = element
            element_owned_dof_masks.append(owned)
        self.element_owned_dof_masks = tuple(element_owned_dof_masks)
        self.element_owned_dof_maps = tuple(
            dof_map[owned]
            for dof_map, owned in zip(
                self.element_dof_maps, self.element_owned_dof_masks
            )
        )

        self.local_mass_matrices = tuple(
            np.diag(element.weights) for element in self.dvr
        )
        self.local_first_derivative_matrices = tuple(
            element.weights[:, np.newaxis] * element.D1 for element in self.dvr
        )
        self.local_stiffness_matrices = tuple(
            element.D1.T
            @ (element.weights[:, np.newaxis] * element.D1)
            for element in self.dvr
        )

    @staticmethod
    def _normalized_axis(axis, ndim):
        axis = int(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise np.exceptions.AxisError(axis, ndim=ndim)
        return axis

    def element_values(self, values, axis=-1):
        """Gather global nodal values into one array per finite element.

        Parameters
        ----------
        values : array_like
            Values whose ``axis`` dimension is the global FEM-DVR dimension.
        axis : int, optional
            Axis containing the FEM-DVR degrees of freedom.

        Returns
        -------
        tuple of ndarray
            Element arrays in element order.  A bridge value occurs in both
            arrays adjacent to that bridge.
        """
        values = np.asarray(values)
        axis = self._normalized_axis(axis, values.ndim)
        if values.shape[axis] != self.r.size:
            raise ValueError(
                f"values axis has length {values.shape[axis]}; expected "
                f"{self.r.size}."
            )
        return tuple(
            np.take(values, dof_map, axis=axis) for dof_map in self.element_dof_maps
        )

    def assemble_element_vectors(self, element_vectors, axis=-1):
        """Scatter-add element vectors into the global FEM-DVR ordering.

        Contributions from both elements adjacent to a bridge are added.  This
        is the vector analogue of finite-element matrix assembly.
        """
        element_vectors = tuple(np.asarray(vector) for vector in element_vectors)
        if len(element_vectors) != self.n_intervals:
            raise ValueError(
                f"Received {len(element_vectors)} element vectors; expected "
                f"{self.n_intervals}."
            )
        if not element_vectors:
            raise ValueError("At least one element vector is required.")

        normalized_axis = self._normalized_axis(axis, element_vectors[0].ndim)
        moved_vectors = []
        leading_shape = None
        for element, (vector, dof_map) in enumerate(
            zip(element_vectors, self.element_dof_maps)
        ):
            current_axis = self._normalized_axis(axis, vector.ndim)
            if current_axis != normalized_axis:
                raise ValueError("All element vectors must have the same rank.")
            moved = np.moveaxis(vector, current_axis, -1)
            if moved.shape[-1] != dof_map.size:
                raise ValueError(
                    f"Element {element} vector has length {moved.shape[-1]}; "
                    f"expected {dof_map.size}."
                )
            if leading_shape is None:
                leading_shape = moved.shape[:-1]
            elif moved.shape[:-1] != leading_shape:
                raise ValueError(
                    "All element vectors must have matching non-FEM-DVR dimensions."
                )
            moved_vectors.append(moved)

        dtype = np.result_type(*(vector.dtype for vector in moved_vectors))
        assembled = np.zeros((*leading_shape, self.r.size), dtype=dtype)
        for vector, dof_map in zip(moved_vectors, self.element_dof_maps):
            assembled[..., dof_map] += vector

        return np.moveaxis(assembled, -1, normalized_axis)

    def apply_element_matrices(self, values, element_matrices, axis=-1):
        """Apply element matrices and scatter-add their local results.

        This method performs an assembly-free action of a weak-form operator.
        It does not construct the corresponding global matrix.
        """
        element_matrices = tuple(np.asarray(matrix) for matrix in element_matrices)
        if len(element_matrices) != self.n_intervals:
            raise ValueError(
                f"Received {len(element_matrices)} element matrices; expected "
                f"{self.n_intervals}."
            )

        local_values = self.element_values(values, axis=axis)
        local_results = []
        for element, (matrix, local_value, dof_map) in enumerate(
            zip(element_matrices, local_values, self.element_dof_maps)
        ):
            expected_shape = (dof_map.size, dof_map.size)
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"Element {element} matrix has shape {matrix.shape}; expected "
                    f"{expected_shape}."
                )
            moved = np.moveaxis(local_value, axis, -1)
            result = np.einsum("ij,...j->...i", matrix, moved, optimize=True)
            local_results.append(np.moveaxis(result, -1, axis))

        return self.assemble_element_vectors(local_results, axis=axis)

    @staticmethod
    def _scale_along_axis(values, scale, axis):
        shape = [1] * values.ndim
        shape[axis] = scale.size
        return values * scale.reshape(shape)

    def _apply_weak_operator(
        self, values, element_matrices, representation, axis=-1
    ):
        values = np.asarray(values)
        axis = self._normalized_axis(axis, values.ndim)
        if representation not in {"weak", "strong", "symmetric"}:
            raise ValueError(
                "representation must be 'weak', 'strong', or 'symmetric'."
            )

        operator_input = values
        if representation == "symmetric":
            operator_input = self._scale_along_axis(
                values, 1.0 / np.sqrt(self.weights), axis
            )

        result = self.apply_element_matrices(
            operator_input, element_matrices, axis=axis
        )
        if representation == "strong":
            return self._scale_along_axis(result, 1.0 / self.weights, axis)
        if representation == "symmetric":
            return self._scale_along_axis(
                result, 1.0 / np.sqrt(self.weights), axis
            )
        return result

    def apply_local_mass(self, values, axis=-1):
        """Apply the assembled mass matrix without constructing it."""
        return self.apply_element_matrices(
            values, self.local_mass_matrices, axis=axis
        )

    def apply_local_first_derivative(
        self, values, representation="strong", axis=-1
    ):
        """Apply the element-local first derivative.

        ``representation='weak'`` returns the assembled weak action,
        ``'strong'`` applies the inverse mass matrix afterward, and
        ``'symmetric'`` applies the square-root mass transformation used by
        :attr:`D1_symmetric`.
        """
        return self._apply_weak_operator(
            values,
            self.local_first_derivative_matrices,
            representation,
            axis=axis,
        )

    def apply_local_stiffness(self, values, representation="weak", axis=-1):
        """Apply the positive FEM-DVR stiffness operator."""
        return self._apply_weak_operator(
            values, self.local_stiffness_matrices, representation, axis=axis
        )

    def apply_local_second_derivative(
        self, values, representation="strong", axis=-1
    ):
        """Apply the second derivative using element-local stiffness actions."""
        return -self.apply_local_stiffness(
            values, representation=representation, axis=axis
        )

    def assemble_local_matrix(self, element_matrices, format="csr"):
        """Assemble element matrices into an optional sparse global matrix."""
        element_matrices = tuple(np.asarray(matrix) for matrix in element_matrices)
        if len(element_matrices) != self.n_intervals:
            raise ValueError(
                f"Received {len(element_matrices)} element matrices; expected "
                f"{self.n_intervals}."
            )

        rows = []
        columns = []
        data = []
        for element, (matrix, dof_map) in enumerate(
            zip(element_matrices, self.element_dof_maps)
        ):
            expected_shape = (dof_map.size, dof_map.size)
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"Element {element} matrix has shape {matrix.shape}; expected "
                    f"{expected_shape}."
                )
            local_rows, local_columns = np.meshgrid(
                dof_map, dof_map, indexing="ij"
            )
            rows.append(local_rows.ravel())
            columns.append(local_columns.ravel())
            data.append(matrix.ravel())

        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.r.size, self.r.size),
        )
        return matrix.asformat(format)

    def _sparse_weak_representation(self, matrix, representation, format):
        if representation not in {"weak", "strong", "symmetric"}:
            raise ValueError(
                "representation must be 'weak', 'strong', or 'symmetric'."
            )
        if representation == "strong":
            matrix = diags(1.0 / self.weights) @ matrix
        elif representation == "symmetric":
            inverse_sqrt_mass = diags(1.0 / np.sqrt(self.weights))
            matrix = inverse_sqrt_mass @ matrix @ inverse_sqrt_mass
        return matrix.asformat(format)

    def sparse_mass_matrix(self, format="csr"):
        """Return the assembled mass matrix in a SciPy sparse format."""
        return self.assemble_local_matrix(self.local_mass_matrices, format=format)

    def sparse_first_derivative_matrix(self, representation="strong", format="csr"):
        """Return the assembled sparse first-derivative matrix."""
        weak = self.assemble_local_matrix(
            self.local_first_derivative_matrices, format="csr"
        )
        return self._sparse_weak_representation(weak, representation, format)

    def sparse_stiffness_matrix(self, representation="weak", format="csr"):
        """Return the assembled sparse positive stiffness matrix."""
        weak = self.assemble_local_matrix(self.local_stiffness_matrices, format="csr")
        return self._sparse_weak_representation(weak, representation, format)

    def sparse_second_derivative_matrix(
        self, representation="strong", format="csr"
    ):
        """Return the assembled sparse second-derivative matrix."""
        return -self.sparse_stiffness_matrix(
            representation=representation, format=format
        )

    @staticmethod
    def cardinal_values(nodes, x):
        """Evaluate local Lagrange cardinal functions.

        This helper is used by FEMDVR interpolation routines.  It assumes the
        local basis is represented by Lagrange cardinal functions on the
        supplied element nodes, as in finite-element Gauss-Legendre-Lobatto DVR.
        """
        nodes = np.asarray(nodes)
        close = np.isclose(x, nodes, rtol=0.0, atol=100.0 * np.finfo(float).eps)
        if np.any(close):
            values = np.zeros_like(nodes, dtype=float)
            values[np.argmax(close)] = 1.0
            return values

        barycentric_weights = np.ones_like(nodes, dtype=float)
        for j in range(nodes.size):
            barycentric_weights[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))

        scaled = barycentric_weights / (x - nodes)
        return scaled / np.sum(scaled)

    def interpolation_matrix(self, points):
        """Evaluate global FEMDVR cardinal functions at physical points.

        Rows correspond to ``points`` and columns correspond to the global
        interface-matched FEMDVR nodes.  Points outside the grid interval are
        represented by zero rows.
        """
        points = np.asarray(points)
        values = np.zeros((points.size, self.r.size))
        element_row_start = np.concatenate(([0], np.cumsum(self.n_points[:-1])))

        for point_index, point in enumerate(points):
            if point < self.nodes[0] or point > self.nodes[-1]:
                continue

            if np.isclose(point, self.nodes[-1]):
                element = self.n_intervals - 1
            else:
                element = np.searchsorted(self.nodes, point, side="right") - 1
                element = int(np.clip(element, 0, self.n_intervals - 1))

            a = self.nodes[element]
            b = self.nodes[element + 1]
            x = 2.0 * (point - a) / (b - a) - 1.0

            local = self.cardinal_values(self.dvr[element].x, x)
            start = element_row_start[element]
            stop = start + self.n_points[element]
            values[point_index] = local @ self.R[start:stop, :]

        return values

    def delta_matrix(self, points, include_boundaries=False):
        """Represent off-grid delta functions in the FEMDVR basis.

        The returned columns approximate ``delta(r - points[j])`` using
        cardinal values divided by quadrature weights.  By default only the
        inner, non-boundary rows are returned, matching the radial Poisson
        solvers.
        """
        interpolation = self.interpolation_matrix(points)

        if include_boundaries:
            return interpolation.T / self.weights[:, np.newaxis]

        return interpolation[:, 1:-1].T / self.weights[1:-1, np.newaxis]


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
