"""C++ implementation bindings for mimetic operators."""

from typing import Any, Dict, Literal, Optional, Tuple, Union

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
from ..pure.operators import (
    MimeticGradient as _PureGradient,
    MimeticDivergence as _PureDivergence,
    MimeticLaplacian as _PureLaplacian,
)


GridDimensions = Tuple[int, ...]
GridSpacings = Tuple[float, ...]
GridSize = Union[int, GridDimensions]
GridSpacing = Union[float, GridSpacings]


def _normalize_grid(n: GridSize, h: GridSpacing) -> Tuple[GridDimensions, GridSpacings]:
    dimensions = (n,) if isinstance(n, int) else tuple(n)
    spacings = (float(h),) if isinstance(h, (int, float)) else tuple(float(step) for step in h)
    if not 1 <= len(dimensions) <= 3 or len(dimensions) != len(spacings):
        raise ValueError("Grid dimensions and spacings must have matching 1D, 2D, or 3D lengths")
    if any(size <= 0 for size in dimensions) or any(step <= 0 for step in spacings):
        raise ValueError("Grid dimensions and spacings must be positive")
    return dimensions, spacings


def _native_grid_operator(
    operator: Any,
    k: int,
    dimensions: GridDimensions,
    spacings: GridSpacings
) -> Any:
    return operator(k, *dimensions, *spacings)


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
RobinBC = _operators.RobinBC
MixedBC = _operators.MixedBC

__all__ = [
    'Gradient',
    'Divergence', 
    'Laplacian',
    'Interpol',
    'RobinBC',
    'MixedBC',
    'MimeticGradient',
    'MimeticDivergence',
    'MimeticLaplacian',
    'MimeticInterpol',
    'MimeticRobinBC',
    'MimeticMixedBC',
]


class MimeticGradient(MimeticOperator):
    """1D/2D Mimetic gradient operator using C++ MOLE implementation.

    Attributes:
        n: Number of grid points
        h: Grid spacing
        k: Order of accuracy
        _cpp_operator: Underlying C++ gradient operator
    """

    def __init__(self, n: GridSize, h: GridSpacing,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic', k: int = 2):
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
        self.boundary = boundary
        self.k = k
        self._python_operator = None
        if boundary == 'periodic':
            self._python_operator = _PureGradient(n, h, boundary=boundary, k=k)
            self._cpp_operator = None
        elif boundary == 'nonperiodic':
            self._cpp_operator = _native_grid_operator(_operators.Gradient, k, dimensions, spacings)
        else:
            raise ValueError("boundary must be 'periodic' or 'nonperiodic'")

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        if self._python_operator is not None:
            return sparse.csc_matrix(self._python_operator.matrix)
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply gradient operator: result = G @ x"""
        if self._python_operator is not None:
            return self._python_operator @ x
        cpp_operator = self._cpp_operator
        if cpp_operator is None:
            raise RuntimeError("Native operator was not initialized")
        return _native_apply(cpp_operator, x)

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
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic', k: int = 2):
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
        self.boundary = boundary
        self.k = k
        self._python_operator = None
        if boundary == 'periodic':
            self._python_operator = _PureDivergence(n, h, boundary=boundary, k=k)
            self._cpp_operator = None
        elif boundary == 'nonperiodic':
            self._cpp_operator = _native_grid_operator(
                _operators.Divergence, k, dimensions, spacings)
        else:
            raise ValueError("boundary must be 'periodic' or 'nonperiodic'")

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        if self._python_operator is not None:
            return sparse.csc_matrix(self._python_operator.matrix)
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply divergence operator: result = D @ x"""
        if self._python_operator is not None:
            return self._python_operator @ x
        cpp_operator = self._cpp_operator
        if cpp_operator is None:
            raise RuntimeError("Native operator was not initialized")
        return _native_apply(cpp_operator, x)

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

    def __init__(self, n: GridSize, h: GridSpacing,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic', k: int = 2):
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
        self.boundary = boundary
        self.k = k
        self._python_operator = None
        if boundary == 'periodic':
            self._python_operator = _PureLaplacian(n, h, boundary=boundary, k=k)
            self._cpp_operator = None
        elif boundary == 'nonperiodic':
            self._cpp_operator = _native_grid_operator(
                _operators.Laplacian, k, dimensions, spacings)
        else:
            raise ValueError("boundary must be 'periodic' or 'nonperiodic'")

    def _build_matrix(self) -> sparse.csc_matrix:
        """Convert C++ sparse matrix to scipy format."""
        if self._python_operator is not None:
            return sparse.csc_matrix(self._python_operator.matrix)
        return _native_matrix(self._cpp_operator)

    @property
    def matrix(self) -> sparse.csc_matrix:
        """Get the sparse matrix representation."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Apply Laplacian operator: result = L @ x"""
        if self._python_operator is not None:
            return self._python_operator @ x
        cpp_operator = self._cpp_operator
        if cpp_operator is None:
            raise RuntimeError("Native operator was not initialized")
        return _native_apply(cpp_operator, x)

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
                 c: Union[float, Tuple[float, ...]] = 0.5,
                 boundary: Literal['periodic', 'nonperiodic'] = 'nonperiodic'):
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
        if boundary != 'nonperiodic':
            raise ValueError("MOLE interpolation supports only non-periodic boundaries")
        self._dimensions = dimensions
        self._spacings = spacings
        self.c = c
        self.boundary = boundary
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

    def __init__(self, n: GridSize, h: GridSpacing, k: int = 2, **bc_dict: Any):
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
