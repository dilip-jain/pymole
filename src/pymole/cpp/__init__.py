"""C++ implementation bindings for mimetic operators."""

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

try:
    # pylint: disable=import-outside-toplevel, import-self
    from . import _operators  # type: ignore[reportMissingModuleSource]
except ImportError as exc:
    raise ImportError(
        "C++ implementation not available. "
        "Please install pymole with C++ support using: pip install pymole[cpp]"
    ) from exc

from ..base import MimeticOperator
from ..pure.operators import _normalize_center_interpolator
from ..pure.boundary import (
    BC1D,
    BC2D,
    BC3D,
    ScalarBoundaryConditions,
    _boundary_coefficients,
    _normalize_boundary_values,
)


GridDimensions = Tuple[int, ...]
GridSpacings = Tuple[float, ...]
GridSize = Union[int, GridDimensions]
GridSpacing = Union[float, GridSpacings]
Boundary = Literal['periodic', 'nonperiodic']
AxisBoundaries = Union[Boundary, Tuple[Boundary, ...]]


def _normalize_grid(n: GridSize, h: GridSpacing) -> Tuple[GridDimensions, GridSpacings]:
    dimensions = (n,) if isinstance(n, int) else tuple(n)
    spacings = (float(h),) if isinstance(h, (int, float)) else tuple(float(step) for step in h)
    if not 1 <= len(dimensions) <= 3 or len(dimensions) != len(spacings):
        raise ValueError("Grid dimensions and spacings must have matching 1D, 2D, or 3D lengths")
    if any(size <= 0 for size in dimensions) or any(
        not np.isfinite(step) or step <= 0 for step in spacings
    ):
        raise ValueError("Grid dimensions and spacings must be positive and finite")
    return dimensions, spacings


def _native_grid_operator(
    operator: Any,
    k: int,
    dimensions: GridDimensions,
    spacings: GridSpacings
) -> Any:
    return operator(k, *dimensions, *spacings)


def _resolve_axis_coefficients(boundary, dimensions, dc, nc):
    if (dc is None) != (nc is None):
        raise ValueError("dc and nc must be supplied together")
    if dc is not None:
        dc_values = np.asarray(dc).reshape(-1)
        nc_values = np.asarray(nc).reshape(-1)
        if dc_values.size != 2 * len(dimensions) or nc_values.size != 2 * len(dimensions):
            raise ValueError("dc and nc must contain two coefficients per axis")
        dc_flags = np.not_equal(dc_values, 0).astype(np.int32)
        nc_flags = np.not_equal(nc_values, 0).astype(np.int32)
        return dc_flags.tolist(), nc_flags.tolist(), dc_flags, nc_flags

    modes = (boundary,) * len(dimensions) if isinstance(boundary, str) else tuple(boundary)
    if len(modes) != len(dimensions) or any(
        mode not in ("periodic", "nonperiodic") for mode in modes
    ):
        raise ValueError("boundary must select periodic or nonperiodic for each axis")
    dc_values = np.asarray([
        coefficient
        for mode in modes
        for coefficient in ((0, 0) if mode == "periodic" else (1, 1))
    ], dtype=np.int32)
    nc_values = np.zeros(2 * len(dimensions), dtype=np.int32)
    return dc_values.tolist(), nc_values.tolist(), dc_values, nc_values


def _native_matrix(operator: Optional[Any]) -> sparse.csc_matrix:
    if operator is None:
        raise RuntimeError("Native operator was not initialized")
    return sparse.csc_matrix(operator.to_scipy_sparse())


def _native_apply(operator: Any, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.asarray(operator @ x)
    return np.column_stack([
        operator @ np.ascontiguousarray(x[:, i])
        for i in range(x.shape[1])
    ])


def _boundary_type(boundary_conditions: Dict[str, Any], side: str) -> str:
    boundary_type = boundary_conditions.get(side)
    if not isinstance(boundary_type, str):
        raise ValueError(f"Boundary type for '{side}' must be a string")
    return boundary_type

# Export C++ operators directly
Gradient = _operators.Gradient
Divergence = _operators.Divergence
Laplacian = _operators.Laplacian
Interpol = _operators.Interpol
InterpolCtoF = _operators.InterpolCtoF
InterpolCtoN = _operators.InterpolCtoN
InterpolFtoC = _operators.InterpolFtoC
RobinBC = _operators.RobinBC
MixedBC = _operators.MixedBC

__all__ = [
    'add_scalar_bc',
    'ScalarBoundaryConditions',
    'BC1D',
    'BC2D',
    'BC3D',
    'MimeticOperator',
    'create_scalar_bc_lhs',
    'apply_scalar_bc_rhs',
    'apply_scalar_bc_rhs_1d',
    'Gradient',
    'Divergence', 
    'Laplacian',
    'Interpol',
    'InterpolCtoF',
    'InterpolCtoN',
    'InterpolFtoC',
    'RobinBC',
    'MixedBC',
    'MimeticGradient',
    'MimeticDivergence',
    'MimeticLaplacian',
    'MimeticInterpol',
    'MimeticInterpolCtoF',
    'MimeticInterpolCtoN',
    'MimeticInterpolFtoC',
    'MimeticRobinBC',
    'MimeticMixedBC',
]


def add_scalar_bc(matrix, rhs, n: GridSize, h: GridSpacing,
                  conditions: ScalarBoundaryConditions):
    """Apply scalar boundary conditions using MOLE's native implementation."""
    dimensions, spacings = _normalize_grid(n, h)
    dc, nc = _boundary_coefficients(conditions, len(dimensions))
    values = _normalize_boundary_values(conditions.values, len(dimensions))
    return _operators.add_scalar_bc(
        matrix,
        rhs,
        conditions.k,
        dimensions,
        spacings,
        dc.tolist(),
        nc.tolist(),
        [face_values.tolist() for face_values in values],
    )


def create_scalar_bc_lhs(
    n: GridSize,
    h: GridSpacing,
    conditions: ScalarBoundaryConditions,
):
    """Create side-specific scalar boundary matrices using native MOLE."""
    dimensions, spacings = _normalize_grid(n, h)
    dc, nc = _boundary_coefficients(conditions, len(dimensions))
    matrices = _operators.create_scalar_bc_lhs(
        conditions.k, dimensions, spacings, dc.tolist(), nc.tolist()
    )
    return tuple(sparse.csc_matrix(matrix) for matrix in matrices)


def apply_scalar_bc_rhs(rhs, conditions: ScalarBoundaryConditions, side_rows):
    """Apply boundary values to indexed RHS rows using native MOLE."""
    coefficient_count = len(conditions.dc)
    if coefficient_count not in (2, 4, 6):
        raise ValueError("Boundary conditions must describe a 1D, 2D, or 3D grid")
    axis_count = coefficient_count // 2
    dc, nc = _boundary_coefficients(conditions, axis_count)
    values = _normalize_boundary_values(conditions.values, axis_count)
    if len(side_rows) != 2 * axis_count:
        raise ValueError(f"side_rows must contain {2 * axis_count} index vectors")

    rhs_values = np.asarray(rhs, dtype=float)
    if rhs_values.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    normalized_rows = []
    for indices in side_rows:
        rows = np.asarray(indices, dtype=np.uintp).reshape(-1)
        if np.any(rows >= rhs_values.size):
            raise ValueError("side_rows contains an index outside rhs")
        normalized_rows.append(rows.tolist())

    return _operators.apply_scalar_bc_rhs(
        rhs_values,
        dc.tolist(),
        nc.tolist(),
        [face_values.tolist() for face_values in values],
        normalized_rows,
    )


def apply_scalar_bc_rhs_1d(rhs, values, indices):
    """Apply the exact MOLE 1D RHS overload through the native extension."""
    return _operators.apply_scalar_bc_rhs_1d(
        rhs,
        np.asarray(values, dtype=float).reshape(-1).tolist(),
        np.asarray(indices, dtype=np.uintp).reshape(-1).tolist(),
    )


class MimeticGradient(MimeticOperator):
    """1D/2D Mimetic gradient operator using C++ MOLE implementation.

    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ gradient operator
    """

    def __init__(self, n: GridSize, h: GridSpacing,
                 boundary: AxisBoundaries = 'nonperiodic',
                 k: int = 2, dc=None, nc=None):
        """Initialize the C++ gradient operator.

        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self._dimensions = dimensions
        self._spacings = spacings
        dc_values, nc_values, dc_array, nc_array = _resolve_axis_coefficients(
            boundary, dimensions, dc, nc
        )
        modes = tuple(
            'periodic'
            if not np.any(dc_array[2 * axis:2 * axis + 2])
            and not np.any(nc_array[2 * axis:2 * axis + 2])
            else 'nonperiodic'
            for axis in range(len(dimensions))
        )
        self.boundary = boundary if dc is None else modes
        self.k = k
        self._cpp_operator = _operators.Gradient(
            k, *dimensions, *spacings, dc_values, nc_values
        )

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply gradient operator: result = G @ x"""
        return _native_apply(self._cpp_operator, x)

    def getP(self) -> np.ndarray:
        return np.asarray(self._cpp_operator.getP(), dtype=float)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticDivergence(MimeticOperator):
    """1D/2D Mimetic divergence operator using C++ MOLE implementation.

    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ divergence operator
    """

    def __init__(self, n: GridSize, h: GridSpacing,
                 boundary: AxisBoundaries = 'nonperiodic',
                 k: int = 2, dc=None, nc=None):
        """Initialize the C++ divergence operator.

        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self._dimensions = dimensions
        self._spacings = spacings
        dc_values, nc_values, dc_array, nc_array = _resolve_axis_coefficients(
            boundary, dimensions, dc, nc
        )
        modes = tuple(
            'periodic'
            if not np.any(dc_array[2 * axis:2 * axis + 2])
            and not np.any(nc_array[2 * axis:2 * axis + 2])
            else 'nonperiodic'
            for axis in range(len(dimensions))
        )
        self.boundary = boundary if dc is None else modes
        self.k = k
        self._cpp_operator = _operators.Divergence(
            k, *dimensions, *spacings, dc_values, nc_values
        )

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply divergence operator: result = D @ x"""
        return _native_apply(self._cpp_operator, x)

    def getQ(self) -> np.ndarray:
        return np.asarray(self._cpp_operator.getQ(), dtype=float)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticLaplacian(MimeticOperator):
    """1D/2D/3D Mimetic Laplacian operator using C++ MOLE implementation.

    Attributes:
        n: Number of grid points (1D) or grid dimensions (2D/3D)
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ Laplacian operator
    """

    def __init__(self, n: GridSize, h: GridSpacing, k: int = 2):
        """Initialize the C++ Laplacian operator.

        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self._dimensions = dimensions
        self._spacings = spacings
        self.k = k
        self._cpp_operator = _native_grid_operator(
            _operators.Laplacian, k, dimensions, spacings)

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Laplacian operator: result = L @ x"""
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticInterpol(MimeticOperator):
    """1D/2D Mimetic interpolation operator using C++ MOLE implementation.

    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ interpolation operator
    """

    def __init__(self, n: GridSize, h: GridSpacing,
                 c: Union[float, Tuple[float, ...]] = 0.5):
        """Initialize the C++ interpolation operator.

        Args:
            n: Number of grid points
            h: Grid spacing
            k: Order of accuracy (default: 2)
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        weights = (c,) * len(dimensions) if isinstance(c, (int, float)) else tuple(c)
        if len(weights) != len(dimensions) or any(not 0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError("Interpolation weights must match the dimension and be in [0, 1]")
        self._dimensions = dimensions
        self._spacings = spacings
        self.c = c
        self._cpp_operator = _operators.Interpol(*dimensions, *weights)

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply interpolation operator: result = I @ x"""
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticInterpolCtoF(MimeticOperator):
    """MOLE center-to-face interpolation operator using the C++ backend."""

    def __init__(self, n: GridSize, boundary: AxisBoundaries = 'nonperiodic',
                 k: int = 2, dc: Optional[Sequence[int]] = None,
                 nc: Optional[Sequence[int]] = None):
        dimensions, dc_values, nc_values, modes = _normalize_center_interpolator(
            n, boundary, k, dc, nc
        )
        super().__init__(dimensions[0], 1.0)
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = 1.0
        self._dimensions = dimensions
        self.k = int(k)
        self.boundary = boundary if dc is None else modes
        self._cpp_operator = _operators.InterpolCtoF(
            self.k, *dimensions, dc_values, nc_values
        )

    def _build_matrix(self) -> sparse.csc_matrix:
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self @ x


class MimeticInterpolCtoN(MimeticOperator):
    """MOLE center-to-node interpolation operator using the C++ backend."""

    def __init__(self, n: GridSize, boundary: AxisBoundaries = 'nonperiodic',
                 k: int = 2, dc: Optional[Sequence[int]] = None,
                 nc: Optional[Sequence[int]] = None):
        dimensions, dc_values, nc_values, modes = _normalize_center_interpolator(
            n, boundary, k, dc, nc
        )
        super().__init__(dimensions[0], 1.0)
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = 1.0
        self._dimensions = dimensions
        self.k = int(k)
        self.boundary = boundary if dc is None else modes
        self._cpp_operator = _operators.InterpolCtoN(
            self.k, *dimensions, dc_values, nc_values
        )

    def _build_matrix(self) -> sparse.csc_matrix:
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self @ x


class MimeticInterpolFtoC(MimeticOperator):
    """MOLE face-to-center interpolation operator using the C++ backend."""

    def __init__(self, n: GridSize, boundary: AxisBoundaries = 'nonperiodic',
                 k: int = 2, dc: Optional[Sequence[int]] = None,
                 nc: Optional[Sequence[int]] = None):
        dimensions, dc_values, nc_values, modes = _normalize_center_interpolator(
            n, boundary, k, dc, nc
        )
        super().__init__(dimensions[0], 1.0)
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = 1.0
        self._dimensions = dimensions
        self.k = int(k)
        self.boundary = boundary if dc is None else modes
        self._cpp_operator = _operators.InterpolFtoC(
            self.k, *dimensions, dc_values, nc_values
        )

    def _build_matrix(self) -> sparse.csc_matrix:
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self @ x


class MimeticRobinBC(MimeticOperator):
    """Mimetic Robin Boundary Condition operator using C++ MOLE implementation.

    Robin BC: a*u + b*du/dn = f

    Attributes:
        n: Number of grid points or tuple for multi-dimensional
        h: Grid spacing or tuple for multi-dimensional
        k: Order of accuracy
        a: Coefficient of Dirichlet component
        b: Coefficient of Neumann component
        _cpp_operator: Underlying C++ RobinBC operator
    """

    def __init__(self, n: GridSize, h: GridSpacing, k: int = 2, a: float = 1.0, b: float = 0.0):
        """Initialize the C++ Robin BC operator.

        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
            a: Coefficient of Dirichlet component (default: 1.0)
            b: Coefficient of Neumann component (default: 0.0)
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.k = k
        self.a = a
        self.b = b

        # Create appropriate C++ operator based on dimensions
        if len(dimensions) == 1:
            self._cpp_operator = _operators.RobinBC(k, dimensions[0], spacings[0], a, b)
        elif len(dimensions) == 2:
                # 2D
            self._cpp_operator = _operators.RobinBC(
                k, dimensions[0], spacings[0], dimensions[1], spacings[1], a, b
            )
        elif len(dimensions) == 3:
                # 3D
            self._cpp_operator = _operators.RobinBC(
                k,
                dimensions[0], spacings[0],
                dimensions[1], spacings[1],
                dimensions[2], spacings[2],
                a, b,
            )
        else:
            raise ValueError("Unsupported dimensionality")

        self._matrix = None

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Robin BC operator: result = R @ x"""
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x


class MimeticMixedBC(MimeticOperator):
    """Mimetic Mixed Boundary Condition operator using C++ MOLE implementation.

    Supports mixed boundary conditions with different types on each boundary.
    Each boundary can be: 'Dirichlet', 'Neumann', or 'Robin'.

    Attributes:
        n: Number of grid points or tuple for multi-dimensional
        h: Grid spacing or tuple for multi-dimensional
        k: Order of accuracy
        bc_dict: Dictionary specifying boundary conditions and coefficients
        _cpp_operator: Underlying C++ MixedBC operator
    """

    def __init__(self, n: GridSize, h: GridSpacing, k: int = 2, **bc_dict):
        """Initialize the C++ Mixed BC operator.

        Args:
            n: Number of grid points or tuple of (m, n) for 2D or (m, n, o) for 3D
            h: Grid spacing or tuple of (dx, dy) for 2D or (dx, dy, dz) for 3D
            k: Order of accuracy (default: 2)
            **bc_dict: Boundary condition specifications as keyword arguments.
                       For 1D: left, coeffs_left, right, coeffs_right
                       For 2D: left, coeffs_left, right, coeffs_right, bottom, coeffs_bottom,
                               top, coeffs_top
                       For 3D: left, coeffs_left, right, coeffs_right, bottom, coeffs_bottom, 
                               top, coeffs_top, front, coeffs_front, back, coeffs_back
        """
        dimensions, spacings = _normalize_grid(n, h)
        super().__init__(dimensions[0], spacings[0])
        self.n = dimensions[0] if len(dimensions) == 1 else dimensions
        self.h = spacings[0] if len(spacings) == 1 else spacings
        self.k = k
        self._bc_dict = bc_dict

        # Create appropriate C++ operator based on dimensions
        if len(dimensions) == 1:
            self._cpp_operator = _operators.MixedBC(
                k, dimensions[0], spacings[0],
                _boundary_type(bc_dict, "left"), bc_dict.get('coeffs_left', []),
                _boundary_type(bc_dict, "right"), bc_dict.get('coeffs_right', [])
            )
        elif len(dimensions) == 2:
            self._cpp_operator = _operators.MixedBC(
                k, dimensions[0], spacings[0], dimensions[1], spacings[1],
                _boundary_type(bc_dict, "left"), bc_dict.get('coeffs_left', []),
                _boundary_type(bc_dict, "right"), bc_dict.get('coeffs_right', []),
                _boundary_type(bc_dict, "bottom"), bc_dict.get('coeffs_bottom', []),
                _boundary_type(bc_dict, "top"), bc_dict.get('coeffs_top', [])
            )
        elif len(dimensions) == 3:
            self._cpp_operator = _operators.MixedBC(
                k,
                dimensions[0], spacings[0],
                dimensions[1], spacings[1],
                dimensions[2], spacings[2],
                _boundary_type(bc_dict, "left"), bc_dict.get('coeffs_left', []),
                _boundary_type(bc_dict, "right"), bc_dict.get('coeffs_right', []),
                _boundary_type(bc_dict, "bottom"), bc_dict.get('coeffs_bottom', []),
                _boundary_type(bc_dict, "top"), bc_dict.get('coeffs_top', []),
                _boundary_type(bc_dict, "front"), bc_dict.get('coeffs_front', []),
                _boundary_type(bc_dict, "back"), bc_dict.get('coeffs_back', [])
            )
        else:
            raise ValueError("Unsupported dimensionality")

        self._matrix = None

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Mixed BC operator: result = M @ x"""
        return _native_apply(self._cpp_operator, x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to vector x."""
        return self @ x
