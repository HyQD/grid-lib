# Radial bases

`BSplineRadialBasis` provides a finite, locally supported B-spline basis whose
evaluation is independent of the integration grid. Interior continuity and
homogeneous finite-boundary conditions are explicit choices.

```python
from grid_lib.basis import BSplineRadialBasis

basis = BSplineRadialBasis(
    [0.0, 1.0, 2.0, 4.0, 8.0],
    degree=8,
    continuity=4,
)
interval = basis.locate([0.25, 0.75])
values, derivatives, indices = basis.evaluate_interval(0, [0.25, 0.75])
```

The default boundary constraints impose only zero values on the reduced
radial functions. Angular-momentum-dependent origin regularity must be
constructed explicitly by the consuming spherical representation.

::: grid_lib.basis.BSplineRadialBasis
