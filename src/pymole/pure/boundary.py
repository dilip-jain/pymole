"""Scalar boundary-condition application for sparse PDE systems."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix

from .operators import MimeticGradient, _normalize_grid


BoundaryValues = Union[
    Sequence[Union[float, Sequence[float], np.ndarray]],
    np.ndarray,
]
GridDimensions = Tuple[int, ...]


@dataclass(frozen=True)
class ScalarBoundaryConditions:
    """Scalar boundary coefficients and values in MOLE face order."""

    dc: Union[Sequence[float], np.ndarray]
    nc: Union[Sequence[float], np.ndarray]
    values: BoundaryValues
    k: int = 2


@dataclass
class BC1D:
    """MOLE-compatible 1D scalar boundary data."""

    dc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    nc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class BC2D:
    """MOLE-compatible 2D scalar boundary data."""

    dc: np.ndarray = field(default_factory=lambda: np.zeros(4))
    nc: np.ndarray = field(default_factory=lambda: np.zeros(4))
    v: List[np.ndarray] = field(default_factory=lambda: [np.array([]) for _ in range(4)])


@dataclass
class BC3D:
    """MOLE-compatible 3D scalar boundary data."""

    dc: np.ndarray = field(default_factory=lambda: np.zeros(6))
    nc: np.ndarray = field(default_factory=lambda: np.zeros(6))
    v: List[np.ndarray] = field(default_factory=lambda: [np.array([]) for _ in range(6)])


@dataclass(frozen=True)
class _AxisBoundary:
    left_matrix: csc_matrix
    right_matrix: csc_matrix
    axis: int
    system_size: int
    axis_count: int


@dataclass(frozen=True)
class _BoundaryAssembly:
    dc: np.ndarray
    nc: np.ndarray
    dimensions: GridDimensions
    spacings: Tuple[float, ...]
    active_axes: Tuple[bool, ...]
    system_size: int


@dataclass(frozen=True)
class _BoundaryGrid:
    dc: np.ndarray
    nc: np.ndarray
    dimensions: GridDimensions
    spacings: Tuple[float, ...]
    active_axes: Tuple[bool, ...]
    system_shape: Tuple[int, ...]


def _normalize_boundary_values(values: BoundaryValues, axis_count: int) -> Tuple[np.ndarray, ...]:
    if axis_count == 1:
        array = np.asarray(values, dtype=float)
        if array.shape != (2,):
            raise ValueError("1D boundary values must contain left and right scalars")
        if not np.all(np.isfinite(array)):
            raise ValueError("Boundary values must be finite")
        return tuple(array[index:index + 1] for index in range(2))

    try:
        side_count = len(values)
    except TypeError as exc:
        raise ValueError(f"Boundary values must contain {2 * axis_count} side vectors") from exc
    if side_count != 2 * axis_count:
        raise ValueError(f"Boundary values must contain {2 * axis_count} side vectors")
    side_values = tuple(np.asarray(side, dtype=float).reshape(-1) for side in values)
    if any(not np.all(np.isfinite(side)) for side in side_values):
        raise ValueError("Boundary values must be finite")
    return side_values


def _boundary_coefficients(conditions: ScalarBoundaryConditions, axis_count: int):
    expected_size = 2 * axis_count
    dc = np.asarray(conditions.dc, dtype=float)
    nc = np.asarray(conditions.nc, dtype=float)
    if dc.shape != (expected_size,) or nc.shape != (expected_size,):
        raise ValueError(f"dc and nc must each contain {expected_size} coefficients")
    if not np.all(np.isfinite(dc)) or not np.all(np.isfinite(nc)):
        raise ValueError("Boundary coefficients must be finite")
    return dc, nc


def _active_boundary_axes(dc: np.ndarray, nc: np.ndarray, axis_count: int) -> Tuple[bool, ...]:
    return tuple(
        bool(np.any(dc[2 * axis:2 * axis + 2]) or np.any(nc[2 * axis:2 * axis + 2]))
        for axis in range(axis_count)
    )


def _lift_boundary_matrix(
    matrix: csc_matrix,
    axis: int,
    dimensions: GridDimensions,
    active_axes: Tuple[bool, ...],
) -> csc_matrix:
    lifted: Optional[csc_matrix] = None
    for current_axis in reversed(range(len(dimensions))):
        if current_axis == axis:
            factor = matrix
        else:
            size = (
                dimensions[current_axis] + 2
                if active_axes[current_axis]
                else dimensions[current_axis]
            )
            factor = sparse.eye(size, format="lil", dtype=float)
            if current_axis > axis and active_axes[current_axis]:
                factor[0, 0] = 0.0
                factor[-1, -1] = 0.0
            factor = factor.tocsc()
        if lifted is None:
            lifted = factor
        else:
            lifted = sparse.kron(lifted, factor, format="csc")
    return csc_matrix(lifted)


def _axis_side_matrices(
    conditions: ScalarBoundaryConditions,
    assembly: _BoundaryAssembly,
    axis: int,
) -> _AxisBoundary:
    size = assembly.dimensions[axis]
    left_row = sparse.lil_matrix((1, size + 2), dtype=float)
    right_row = sparse.lil_matrix((1, size + 2), dtype=float)
    if assembly.dc[2 * axis] != 0.0:
        left_row[0, 0] = assembly.dc[2 * axis]
    if assembly.dc[2 * axis + 1] != 0.0:
        right_row[0, -1] = assembly.dc[2 * axis + 1]
    left_row = left_row.tocsc()
    right_row = right_row.tocsc()
    if assembly.nc[2 * axis] != 0.0 or assembly.nc[2 * axis + 1] != 0.0:
        gradient = MimeticGradient(size, assembly.spacings[axis], k=conditions.k).matrix
        left_row -= assembly.nc[2 * axis] * gradient.getrow(0)
        right_row += assembly.nc[2 * axis + 1] * gradient.getrow(size)
    left_matrix = sparse.vstack(
        (left_row, sparse.csc_matrix((size + 1, size + 2))), format="csc"
    )
    right_matrix = sparse.vstack(
        (sparse.csc_matrix((size + 1, size + 2)), right_row), format="csc"
    )
    return _AxisBoundary(
        _lift_boundary_matrix(
            left_matrix, axis, assembly.dimensions, assembly.active_axes
        ),
        _lift_boundary_matrix(
            right_matrix, axis, assembly.dimensions, assembly.active_axes
        ),
        axis,
        assembly.system_size,
        len(assembly.dimensions),
    )


def _build_side_matrices(
    conditions: ScalarBoundaryConditions,
    assembly: _BoundaryAssembly,
) -> List[Optional[_AxisBoundary]]:
    sides: List[Optional[_AxisBoundary]] = [None] * len(assembly.dimensions)
    for axis, is_active in enumerate(assembly.active_axes):
        if is_active:
            sides[axis] = _axis_side_matrices(conditions, assembly, axis)
    return sides


def _validate_system(matrix, rhs, system_shape: Tuple[int, ...]):
    system_size = int(np.prod(system_shape))
    sparse_matrix = csc_matrix(matrix, dtype=float)
    vector = np.asarray(rhs, dtype=float)
    expected_matrix_shape = (system_size, system_size)
    if sparse_matrix.shape != expected_matrix_shape:
        raise ValueError(
            f"matrix must have shape {expected_matrix_shape} for these boundary conditions"
        )
    if vector.shape != (system_size,):
        raise ValueError(f"rhs must have shape ({system_size},)")
    return sparse_matrix, vector.copy(), system_size


def _apply_axis_boundary(
    matrix: csc_matrix,
    rhs: np.ndarray,
    boundary: _AxisBoundary,
    left_values: np.ndarray,
    right_values: np.ndarray,
):
    left_rows = np.unique(boundary.left_matrix.nonzero()[0])
    right_rows = np.unique(boundary.right_matrix.nonzero()[0])
    if boundary.axis_count == 1:
        left_rows = np.array([0], dtype=int)
        right_rows = np.array([boundary.system_size - 1], dtype=int)
    affected_rows = np.union1d(left_rows, right_rows)
    if affected_rows.size:
        editable = matrix.tolil(copy=True)
        editable[affected_rows, :] = 0.0
        matrix = csc_matrix(
            editable.tocsc() + boundary.left_matrix + boundary.right_matrix
        )
        rhs[affected_rows] = 0.0

    for rows, values in (
        (left_rows, left_values),
        (right_rows, right_values),
    ):
        for index in range(min(rows.size, values.size)):
            rhs[rows[index]] = values[index]
    return matrix


def _prepare_boundary_application(matrix, rhs, n, h, conditions):
    dimensions, spacings = _normalize_grid(n, h)
    grid = _prepare_boundary_grid(dimensions, spacings, conditions)
    values = _normalize_boundary_values(conditions.values, len(dimensions))
    matrix, rhs, system_size = _validate_system(matrix, rhs, grid.system_shape)
    assembly = _BoundaryAssembly(
        grid.dc, grid.nc, dimensions, spacings, grid.active_axes, system_size
    )
    side_matrices = _build_side_matrices(conditions, assembly)
    return matrix, rhs, values, side_matrices


def _prepare_boundary_grid(dimensions, spacings, conditions):
    if any(not np.isfinite(step) or step <= 0.0 for step in spacings):
        raise ValueError("Grid spacings must be finite and positive")
    dc, nc = _boundary_coefficients(conditions, len(dimensions))
    active_axes = _active_boundary_axes(dc, nc, len(dimensions))
    system_shape = tuple(
        size + 2 if active_axes[axis] else size
        for axis, size in enumerate(dimensions)
    )
    return _BoundaryGrid(dc, nc, dimensions, spacings, active_axes, system_shape)


def create_scalar_bc_lhs(
    n: Union[int, GridDimensions],
    h: Union[float, Tuple[float, ...]],
    conditions: ScalarBoundaryConditions,
) -> Tuple[csc_matrix, ...]:
    """Build MOLE's side-specific scalar boundary-condition matrices."""
    dimensions, spacings = _normalize_grid(n, h)
    grid = _prepare_boundary_grid(dimensions, spacings, conditions)
    system_size = int(np.prod(grid.system_shape))
    assembly = _BoundaryAssembly(
        grid.dc, grid.nc, dimensions, spacings, grid.active_axes, system_size
    )
    side_matrices = _build_side_matrices(conditions, assembly)
    empty = csc_matrix((0, 0), dtype=float)
    result = []
    for boundary in side_matrices:
        result.extend(
            (csc_matrix((dimensions[0] + 2, dimensions[0] + 2)),) * 2
            if boundary is None and len(dimensions) == 1
            else (empty, empty)
            if boundary is None
            else (boundary.left_matrix, boundary.right_matrix)
        )
    return tuple(result)


def apply_scalar_bc_rhs(
    rhs,
    conditions: ScalarBoundaryConditions,
    side_rows: Sequence[Sequence[int]],
) -> np.ndarray:
    """Apply boundary values to indexed RHS faces and return an updated copy."""
    coefficient_count = np.asarray(conditions.dc).size
    if coefficient_count not in (2, 4, 6):
        raise ValueError("Boundary conditions must describe a 1D, 2D, or 3D grid")
    axis_count = coefficient_count // 2
    dc, nc = _boundary_coefficients(conditions, axis_count)
    active_axes = _active_boundary_axes(dc, nc, axis_count)
    values = _normalize_boundary_values(conditions.values, axis_count)
    if len(side_rows) != 2 * axis_count:
        raise ValueError(f"side_rows must contain {2 * axis_count} index vectors")
    result = np.asarray(rhs, dtype=float)
    if result.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    result = result.copy()

    for side, indices in enumerate(side_rows):
        if axis_count > 1 and not active_axes[side // 2]:
            continue
        rows = np.asarray(indices, dtype=np.intp).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= result.size):
            raise ValueError("side_rows contains an index outside rhs")
        for index in range(min(rows.size, values[side].size)):
            result[rows[index]] = values[side][index]
    return result


def add_scalar_bc(
    matrix,
    rhs,
    n: Union[int, GridDimensions],
    h: Union[float, Tuple[float, ...]],
    conditions: ScalarBoundaryConditions,
) -> Tuple[csc_matrix, np.ndarray]:
    """Apply scalar boundary conditions and return updated matrix and RHS copies.

    Boundary values follow MOLE face order: left/right, bottom/top, then
    front/back. An axis with zero Dirichlet and Neumann coefficients on both
    sides is periodic and has no ghost cells.
    """
    matrix, rhs, boundary_values, side_matrices = _prepare_boundary_application(
        matrix, rhs, n, h, conditions
    )
    for axis, boundary in enumerate(side_matrices):
        if boundary is None:
            continue
        matrix = _apply_axis_boundary(
            matrix,
            rhs,
            boundary,
            boundary_values[2 * axis],
            boundary_values[2 * axis + 1],
        )
    return matrix, rhs
