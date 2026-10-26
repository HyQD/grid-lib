# Pseudospectral Grids API

## Element-local FEM-DVR operations

`FEMDVR` retains the existing dense global matrices and additionally exposes
the finite-element topology and the local weak-form mass, first-derivative,
and stiffness matrices. The local-to-global maps include the shared bridge
degree of freedom in both adjacent elements, while `dof_owner` and
`element_owned_dof_maps` assign each global degree of freedom to exactly one
element using left ownership at bridges.

The `apply_local_*` methods gather global values, apply the element matrices,
and scatter-add the result without assembling a global operator. They accept
an `axis` argument, so the FEM-DVR coordinate can be any axis of a
multidimensional state. Weak, strong, and square-root-mass symmetric forms are
available where applicable. Sparse global matrices can be assembled
explicitly for validation and interoperability.

::: grid_lib.pseudospectral_grids
