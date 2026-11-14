"""Base classes and interfaces for PyMOLE operators."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix

class MimeticOperator(ABC):
    """Abstract base class for all mimetic operators.
    
    Attributes:
        n (int): Number of grid points
        h (float): Grid spacing
        _matrix (scipy.sparse.spmatrix): Sparse matrix representation of the operator
    """

    def __init__(self, n: int, h: float):
        """Initialize the mimetic operator.
        
        Args:
            n: Number of grid points
            h: Grid spacing
        """
        if n <= 0:
            raise ValueError("Number of grid points must be positive")
        if h <= 0:
            raise ValueError("Grid spacing must be positive")

        self.n = n
        self.h = h
        self._matrix: Optional[csc_matrix] = None

    @abstractmethod
    def _build_matrix(self) -> csc_matrix:
        """Build the operator matrix.
        
        Returns:
            scipy.sparse.csc_matrix: Sparse matrix representation
        """
        # pylint: disable=unnecessary-pass
        pass

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Implement matrix multiplication (@) operator.
        
        Args:
            x: Input vector/matrix
            
        Returns:
            numpy.ndarray: Result of operator application
        """
        if self._matrix is None:
            self._matrix = self._build_matrix()
        result = self._matrix @ x
        if np.issubdtype(np.asarray(result).dtype, np.floating):
            result = np.asarray(result)
            result[np.abs(result) < 1e-13] = 0.0
        return result

    @property
    def matrix(self) -> csc_matrix:
        """Get the operator matrix.
        
        Returns:
            scipy.sparse.csc_matrix: Sparse matrix representation
        """
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix

    def to_scipy_sparse(self) -> csc_matrix:
        """Return the operator in SciPy CSC format."""
        return csc_matrix(self.matrix)

    def shape(self) -> Tuple[int, int]:
        """Return the matrix dimensions as ``(rows, columns)``."""
        return self.matrix.get_shape()

    def nnz(self) -> int:
        """Return the number of stored non-zero matrix entries."""
        return int(self.matrix.nnz)
