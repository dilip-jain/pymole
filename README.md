# PyMOLE

Python implementation of MOLE (Mimetic Operators Library Enhanced)

## Description

PyMOLE is a Python implementation of the SDSU's MOLE library, providing mimetic operators for numerical calculations. It offers both pure Python and C++ implementations, allowing users to choose between ease of installation and maximum performance.

## Installation

Basic installation (Python-only):
```bash
pip install pymole
```

Full installation (with C++ backend):
```bash
pip install pymole[cpp]
```

## Usage

```python
import pymole
import numpy as np

# Create a gradient operator (uses Python backend by default)
m, dx = 100, 0.01
grad = pymole.create_gradient(m, dx)

# Switch to C++ backend for performance
pymole.use_backend('cpp')
grad = pymole.create_gradient(m, dx)

# Non-periodic MOLE grids use staggered vectors: G is (m+1) x (m+2).
x = np.linspace(0.0, 1.0, m + 2)
result = grad @ x
```

The MOLE C++ constructor forms are also exposed at package level and use the
selected backend. For example, `pymole.Gradient(k, m, dx)` and
`pymole.InterpolCtoF(k, m, dc, nc)` accept the same arguments whether the
Python or C++ backend is selected. `pymole.cpp.Gradient` and
`pymole.pure.Gradient` expose the corresponding backend-specific constructors.

For non-periodic 1D operators, gradient maps `m + 2` nodal values to `m + 1` values, divergence maps `m + 1` values to `m + 2`, and the Laplacian is `(m + 2) x (m + 2)` and is assembled as divergence times gradient. The supplied spacing is used directly.

Interpolate center values to face values with MOLE's `InterpolCtoF` operator:

```python
ctof = pymole.create_interpol_c_to_f(9, k=2)
face_values = ctof @ np.ones(11)
```

`create_interpol_c_to_f` supports 1D, 2D, and 3D grids, orders `k=2, 4, 6, 8`,
and periodic or non-periodic boundaries per axis. The grid sizes must satisfy
MOLE's `m > 2*k` constraint. Unlike the other grid factories, it does not take
grid spacing because this interpolation matrix is independent of spacing.
Use `create_interpol_c_to_n` for the corresponding center-to-node operator;
use `create_interpol_f_to_c` to interpolate face values to centers. CtoN uses
tensor-product assembly in multiple dimensions, while FtoC assembles the
directional blocks with MOLE's staggered-grid shapes. Use
`create_interpol_n_to_c` to interpolate nodal values to centers.

Apply scalar boundary conditions to a sparse operator and right-hand side with `add_scalar_bc`:

```python
lap = pymole.create_laplacian(m, dx)
A, b = pymole.add_scalar_bc(
	lap.matrix,
	np.zeros(m + 2),
	m,
	dx,
	pymole.ScalarBoundaryConditions(
		dc=(1.0, 1.0),
		nc=(0.0, 0.0),
		values=(0.0, 0.0),
	),
)
```

The function returns updated matrix and RHS copies. In 2D and 3D, `dc`, `nc`, and `v` follow `left/right`, `bottom/top`, and `front/back` order; `v` contains one vector per side.

The lower-level MOLE operations are also available: `create_scalar_bc_lhs` returns the boundary matrix for each side, and `apply_scalar_bc_rhs` applies the supplied side-row indices to the RHS.

For MOLE's in-place API and overload names, use `AddScalarBC` with its matching dimension-specific data type:

```python
bc = pymole.AddScalarBC.BC1D()
bc.dc[:] = (1.0, 1.0)
bc.v[:] = (0.0, 0.0)
A = pymole.create_laplacian(m, dx).matrix.copy()
b = np.zeros(m + 2)
pymole.AddScalarBC.addScalarBC(A, b, 2, m, dx, bc)
```

`AddScalarBC.addScalarBClhs` accepts the output matrices as arguments, and `AddScalarBC.addScalarBCrhs` updates the provided RHS vector, matching MOLE's reference-output and in-place conventions.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

Based on the original MOLE library:
- Repository: https://github.com/csrc-sdsu/mole
- License: GPL-3.0
