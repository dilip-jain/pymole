"""
PyMOLE - Python Mimetic Operators Library Enhanced
Copyright (C) 2025 Dilip Jain

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This is a Python reimplementation of MOLE:
Original MOLE Copyright (C) CSRC-SDSU
Repository: https://github.com/csrc-sdsu/mole
"""

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .backend import use_backend, get_backend
from . import pure
from .pure.boundary import BC1D, BC2D, BC3D, ScalarBoundaryConditions
from .pure.operators import _normalize_grid

__version__ = "0.1.0a1"

BoundaryMode = Literal['periodic', 'nonperiodic']
Boundary = Union[BoundaryMode, Tuple[BoundaryMode, ...]]
AxisBoundaries = Union[BoundaryMode, Tuple[BoundaryMode, ...]]
GridSize = Union[int, Tuple[int, ...]]
GridSpacing = Union[float, Tuple[float, ...]]

__all__ = [
    'use_backend',
    'get_backend',
    'create_gradient',
    'create_divergence',
    'create_laplacian',
    'create_interpol',
    'create_interpol_c_to_f',
    'create_interpol_c_to_n',
    'create_robin_bc',
    'create_mixed_bc',
    'add_scalar_bc',
    'create_scalar_bc_lhs',
    'apply_scalar_bc_rhs',
    'ScalarBoundaryConditions',
    'BC1D',
    'BC2D',
    'BC3D',
    'AddScalarBC',
]


def create_gradient(
    n: GridSize,
    h: GridSpacing,
    boundary: Boundary = 'nonperiodic',
    k: int = 2,
    dc=None,
    nc=None,
):
    """Factory function that creates gradient operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticGradient(n, h, boundary=boundary, k=k, dc=dc, nc=nc)
    return pure.MimeticGradient(n, h, boundary=boundary, k=k, dc=dc, nc=nc)

def create_divergence(
    n: GridSize,
    h: GridSpacing,
    boundary: Boundary = 'nonperiodic',
    k: int = 2,
    dc=None,
    nc=None,
):
    """Factory function that creates divergence operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticDivergence(n, h, boundary=boundary, k=k, dc=dc, nc=nc)
    return pure.MimeticDivergence(n, h, boundary=boundary, k=k, dc=dc, nc=nc)


def create_laplacian(n: GridSize, h: GridSpacing, k: int = 2):
    """Factory function that creates Laplacian operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticLaplacian(n, h, k=k)
    return pure.MimeticLaplacian(n, h, k=k)


def create_interpol(
    n: GridSize,
    h: GridSpacing,
    c: Union[float, Tuple[float, ...]] = 0.5,
):
    """Factory function that creates interpolation operator using selected backend"""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticInterpol(n, h, c=c)
    return pure.MimeticInterpol(n, h, c=c)


def create_interpol_c_to_f(
    n: GridSize,
    boundary: AxisBoundaries = 'nonperiodic',
    k: int = 2,
    dc: Optional[Sequence[int]] = None,
    nc: Optional[Sequence[int]] = None,
):
    """Create MOLE's center-to-face interpolation operator."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticInterpolCtoF(n, boundary=boundary, k=k, dc=dc, nc=nc)
    return pure.MimeticInterpolCtoF(n, boundary=boundary, k=k, dc=dc, nc=nc)


def create_interpol_c_to_n(
    n: GridSize,
    boundary: Boundary = 'nonperiodic',
    k: int = 2,
    dc: Optional[Sequence[int]] = None,
    nc: Optional[Sequence[int]] = None,
):
    """Create MOLE's center-to-node interpolation operator."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticInterpolCtoN(n, boundary=boundary, k=k, dc=dc, nc=nc)
    return pure.MimeticInterpolCtoN(n, boundary=boundary, k=k, dc=dc, nc=nc)


def create_robin_bc(
    n: GridSize,
    h: GridSpacing,
    k: int = 2,
    a: float = 1.0,
    b: float = 0.0,
):
    """Create a Robin boundary-condition operator using the selected backend."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticRobinBC(n, h, k=k, a=a, b=b)
    return pure.MimeticRobinBC(n, h, k=k, a=a, b=b)


def create_mixed_bc(n: GridSize, h: GridSpacing, k: int = 2, **bc_dict):
    """Create a mixed boundary-condition operator using the selected backend."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.MimeticMixedBC(n, h, k=k, **bc_dict)
    return pure.MimeticMixedBC(n, h, k=k, **bc_dict)


def add_scalar_bc(matrix, rhs, n: GridSize, h: GridSpacing,
                  conditions: ScalarBoundaryConditions):
    """Apply scalar boundary conditions and return updated matrix and RHS."""
    if get_backend() == 'cpp':
        # Boundary assembly operates on the SciPy sparse matrix shared by both backends.
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.add_scalar_bc(matrix, rhs, n, h, conditions)
    return pure.add_scalar_bc(matrix, rhs, n, h, conditions)


def create_scalar_bc_lhs(
    n: GridSize,
    h: GridSpacing,
    conditions: ScalarBoundaryConditions,
):
    """Create side-specific scalar boundary matrices with the selected backend."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.create_scalar_bc_lhs(n, h, conditions)
    return pure.create_scalar_bc_lhs(n, h, conditions)


def apply_scalar_bc_rhs(rhs, conditions: ScalarBoundaryConditions, side_rows):
    """Apply boundary values to selected RHS rows with the chosen backend."""
    if get_backend() == 'cpp':
        # pylint: disable=import-outside-toplevel
        from . import cpp
        return cpp.apply_scalar_bc_rhs(rhs, conditions, side_rows)
    return pure.apply_scalar_bc_rhs(rhs, conditions, side_rows)


def _overwrite_sparse_matrix(matrix, replacement):
    if not sparse.issparse(matrix):
        raise TypeError("A must be a SciPy sparse matrix")
    updated = replacement.asformat(matrix.format)
    attributes = {
        'csc': ('_shape', 'data', 'indices', 'indptr'),
        'csr': ('_shape', 'data', 'indices', 'indptr'),
        'bsr': ('_shape', 'data', 'indices', 'indptr'),
        'coo': ('_shape', 'row', 'col', 'data'),
        'dia': ('_shape', 'data', 'offsets'),
        'dok': ('_shape', '_dict'),
        'lil': ('_shape', 'rows', 'data'),
    }.get(matrix.format)
    if attributes is None:
        raise TypeError(f"In-place AddScalarBC does not support {matrix.format!r} matrices")
    for attribute in attributes:
        value = getattr(updated, attribute)
        setattr(matrix, attribute, value.copy() if hasattr(value, 'copy') else value)
    matrix.__dict__.pop('_has_sorted_indices', None)
    matrix.__dict__.pop('_has_canonical_format', None)


def _overwrite_vector(vector, replacement):
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise TypeError("b must be a one-dimensional NumPy array for in-place updates")
    values = np.asarray(replacement, dtype=vector.dtype)
    if values.shape != vector.shape:
        raise ValueError("Replacement vector must have the same shape as b")
    vector[...] = values


# pylint: disable=invalid-name,too-many-locals
def addScalarBClhs(*args):
    """MOLE-compatible overloads that fill side matrices in place."""
    if len(args) == 7:
        k, m, dx, dc, nc, left, right = args
        n, h = m, dx
        outputs = (left, right)
    elif len(args) == 11:
        k, m, dx, n_axis, dy, dc, nc, left, right, bottom, top = args
        n, h = (m, n_axis), (dx, dy)
        outputs = (left, right, bottom, top)
    elif len(args) == 15:
        (k, m, dx, n_axis, dy, o, dz, dc, nc,
         left, right, bottom, top, front, back) = args
        n, h = (m, n_axis, o), (dx, dy, dz)
        outputs = (left, right, bottom, top, front, back)
    else:
        raise TypeError("addScalarBClhs expects the MOLE 1D, 2D, or 3D argument list")
    conditions = ScalarBoundaryConditions(dc=dc, nc=nc, values=(), k=k)
    matrices = create_scalar_bc_lhs(n, h, conditions)
    for output, matrix in zip(outputs, matrices):
        _overwrite_sparse_matrix(output, matrix)
# pylint: disable=invalid-name
def addScalarBCrhs(*args):
    """MOLE-compatible RHS overloads that update the supplied vector in place."""
    if len(args) == 3:
        rhs, values, indices = args
        if get_backend() == 'cpp':
            # pylint: disable=import-outside-toplevel
            from . import cpp
            updated = cpp.apply_scalar_bc_rhs_1d(rhs, values, indices)
        else:
            updated = np.asarray(rhs, dtype=float).copy()
            boundary_values = np.asarray(values, dtype=float).reshape(-1)
            for index, row in enumerate(indices):
                updated[int(row)] = boundary_values[index]
        _overwrite_vector(rhs, updated)
        return None

    if len(args) not in (8, 10):
        raise TypeError("addScalarBCrhs expects the MOLE 1D, 2D, or 3D argument list")
    rhs, dc, nc, values, *side_rows = args
    conditions = ScalarBoundaryConditions(dc=dc, nc=nc, values=values)
    updated = apply_scalar_bc_rhs(rhs, conditions, side_rows)
    _overwrite_vector(rhs, updated)
    return None


# pylint: disable=invalid-name
def addScalarBC(*args):
    """MOLE-compatible overloads that modify the supplied matrix and RHS in place."""
    if len(args) == 6:
        matrix, rhs, k, m, dx, conditions = args
        n, h, expected_type = m, dx, BC1D
    elif len(args) == 8:
        matrix, rhs, k, m, dx, n_axis, dy, conditions = args
        n, h, expected_type = (m, n_axis), (dx, dy), BC2D
    elif len(args) == 10:
        matrix, rhs, k, m, dx, n_axis, dy, o, dz, conditions = args
        n, h, expected_type = (m, n_axis, o), (dx, dy, dz), BC3D
    else:
        raise TypeError("addScalarBC expects the MOLE 1D, 2D, or 3D argument list")
    if not isinstance(conditions, expected_type):
        raise TypeError(f"Expected {expected_type.__name__} boundary data")

    dimensions, spacings = _normalize_grid(n, h)
    if not np.all(np.isfinite(spacings)):
        raise ValueError("Grid spacings must be finite and positive")
    coefficient_count = 2 * len(dimensions)
    dc = np.asarray(conditions.dc, dtype=float)
    nc = np.asarray(conditions.nc, dtype=float)
    if dc.shape != (coefficient_count,) or nc.shape != (coefficient_count,):
        raise ValueError(f"dc and nc must each contain {coefficient_count} coefficients")
    if len(dimensions) > 1 and len(conditions.v) != coefficient_count:
        raise ValueError(f"v must contain {coefficient_count} side vectors")
    if matrix.shape[0] != matrix.shape[1] or rhs.shape != (matrix.shape[1],):
        raise ValueError("A must be square and b must match A's columns")
    if not np.any(dc) and not np.any(nc):
        return None

    scalar_conditions = ScalarBoundaryConditions(
        dc=dc, nc=nc, values=conditions.v, k=k
    )
    grid_n = dimensions[0] if len(dimensions) == 1 else dimensions
    grid_h = spacings[0] if len(spacings) == 1 else spacings
    updated_matrix, updated_rhs = add_scalar_bc(
        matrix, rhs, grid_n, grid_h, scalar_conditions
    )
    _overwrite_sparse_matrix(matrix, updated_matrix)
    _overwrite_vector(rhs, updated_rhs)
    return None


# pylint: disable=too-few-public-methods
class AddScalarBC:
    """Namespace matching MOLE's scalar-boundary API."""

    BC1D = BC1D
    BC2D = BC2D
    BC3D = BC3D
    addScalarBClhs = staticmethod(addScalarBClhs)
    addScalarBCrhs = staticmethod(addScalarBCrhs)
    addScalarBC = staticmethod(addScalarBC)
