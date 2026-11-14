"""Tests for the optional native operator bindings."""

import importlib
import inspect
from typing import Any

import numpy as np
import pytest
from scipy import sparse

from pymole import pure
from pymole.pure.operators import (
    MimeticDivergence as PureDivergence,
    MimeticGradient as PureGradient,
    MimeticInterpol as PureInterpol,
    MimeticInterpolCtoF as PureInterpolCtoF,
    MimeticInterpolCtoN as PureInterpolCtoN,
    MimeticInterpolFtoC as PureInterpolFtoC,
    MimeticInterpolNtoC as PureInterpolNtoC,
)


def _native_backend() -> Any:
    try:
        return importlib.import_module("pymole.cpp")
    except ImportError as exc:
        cause = exc.__cause__
        cause_name = getattr(cause, "name", "")
        if isinstance(cause, ModuleNotFoundError) and cause_name.endswith("._operators"):
            pytest.skip(f"Native extension is not installed: {exc}")
        if isinstance(cause, ImportError) and "cannot import name '_operators'" in str(cause):
            pytest.skip(f"Native extension is not installed: {exc}")
        if isinstance(cause, ImportError) and "DLL load failed" in str(cause):
            pytest.skip(f"Native extension is not loadable: {exc}")
        raise


def test_gradient_matches_python_and_exports_csc():
    """Check native gradient parity and CSC matrix export."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    native_gradient = native.MimeticGradient(grid_size, spacing)
    python_gradient = PureGradient(grid_size, spacing)

    assert sparse.isspmatrix_csc(native_gradient.matrix)
    np.testing.assert_allclose(
        native_gradient.matrix.toarray(),
        sparse.csc_matrix(python_gradient.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )

    values = np.linspace(0.0, 1.0, grid_size + 2)
    np.testing.assert_allclose(
        native_gradient @ values,
        native_gradient.matrix @ values,
        rtol=0,
        atol=1e-12,
    )


def test_divergence_application_matches_matrix():
    """Check native divergence application against its sparse matrix."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    divergence = native.MimeticDivergence(grid_size, spacing)
    values = np.linspace(0.0, 1.0, grid_size + 1)

    assert divergence.matrix.shape == (grid_size + 2, grid_size + 1)
    np.testing.assert_allclose(
        divergence @ values,
        divergence.matrix @ values,
        rtol=0,
        atol=1e-12,
    )


def test_laplacian_is_divergence_composed_with_gradient():
    """Check the native Laplacian equals divergence composed with gradient."""
    native = _native_backend()
    grid_size, spacing = 20, 0.05
    gradient = native.MimeticGradient(grid_size, spacing)
    divergence = native.MimeticDivergence(grid_size, spacing)
    laplacian = native.MimeticLaplacian(grid_size, spacing)

    np.testing.assert_allclose(
        laplacian.matrix.toarray(),
        (divergence.matrix @ gradient.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )


def test_interpolation_matches_python_before_apply():
    """Check sparse interpolation export before any operator application."""
    native = _native_backend()
    grid_size, spacing, weight = 9, 0.125, 0.3
    native_interpol = native.MimeticInterpol(grid_size, spacing, weight)
    python_interpol = PureInterpol(grid_size, spacing, weight)

    np.testing.assert_allclose(
        native_interpol.matrix.toarray(),
        sparse.csc_matrix(python_interpol.matrix).toarray(),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("k", "m", "expected"),
    [
        (2, 5, (0.5, 0.5)),
        (4, 9, (-16 / 112, 70 / 112, 70 / 112, -14 / 112, 2 / 112)),
        (6, 13, (-768 / 8448, 4158 / 8448, 6930 / 8448, -2772 / 8448,
                 1188 / 8448, -330 / 8448, 42 / 8448)),
        (8, 17, (-1 / 15, 429 / 1024, 1001 / 1024, -3003 / 5120,
                 429 / 1024, -715 / 3072, 91 / 1024, -21 / 1024, 11 / 5120)),
    ],
)
def test_ctof_cpp_binding_matches_pinned_reference_and_python(k, m, expected):
    """Validate native CtoF against pinned MOLE coefficients and pure Python."""
    native = _native_backend()
    native_operator = native.MimeticInterpolCtoF(m, k=k)
    pure_operator = PureInterpolCtoF(m, k=k)

    assert native_operator.matrix.shape == (m + 1, m + 2)
    np.testing.assert_allclose(
        native_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(native_operator.matrix[1, :len(expected)].toarray()[0],
                               expected, rtol=0, atol=1e-14)
    raw_operator = native.InterpolCtoF(k, m, [1, 1], [0, 0])
    assert raw_operator.shape() == (m + 1, m + 2)
    np.testing.assert_allclose(
        raw_operator.to_scipy_sparse().toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("k", "m", "expected_entries"),
    [
        (2, 5, ((0, 0, 0.5), (0, 4, 0.5))),
        (4, 9, ((0, 0, 72 / 128), (0, 8, 72 / 128),
                (0, 7, -8 / 128), (0, 1, -8 / 128))),
        (6, 13, ((0, 0, 600 / 1024), (0, 12, 600 / 1024),
                 (0, 11, -100 / 1024), (0, 10, 12 / 1024),
                 (0, 2, 12 / 1024), (0, 1, -100 / 1024))),
        (8, 17, ((0, 0, 1225 / 2048), (0, 16, 1225 / 2048),
                 (0, 15, -245 / 2048), (0, 14, 49 / 2048),
                 (0, 13, -5 / 2048), (0, 3, -5 / 2048),
                 (0, 2, 49 / 2048), (0, 1, -245 / 2048))),
    ],
)
def test_ctof_cpp_periodic_and_multidimensional_shapes(k, m, expected_entries):
    """Check native periodic stencils and multidimensional MOLE assembly."""
    native = _native_backend()
    native_operator = native.MimeticInterpolCtoF(m, boundary="periodic", k=k)
    pure_operator = PureInterpolCtoF(m, boundary="periodic", k=k)
    assert native_operator.matrix.shape == (m, m)
    np.testing.assert_allclose(
        native_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )
    for row, column, expected in expected_entries:
        assert native_operator.matrix[row, column] == pytest.approx(expected)

    multidimensional = (
        ((9, 10), "nonperiodic", 2, (199, 264)),
        ((9, 10), "periodic", 2, (180, 180)),
        ((9, 10, 11), ("periodic", "nonperiodic", "periodic"), 2, (3069, 3564)),
    )
    for dimensions, boundary, order, expected_shape in multidimensional:
        native_grid = native.MimeticInterpolCtoF(
            dimensions, boundary=boundary, k=order
        )
        pure_grid = PureInterpolCtoF(dimensions, boundary=boundary, k=order)
        assert native_grid.matrix.shape == expected_shape
        difference = (native_grid.matrix - pure_grid.matrix).tocsc()
        np.testing.assert_allclose(
            difference.data,
            np.zeros_like(difference.data),
            rtol=0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            native_grid @ np.ones(expected_shape[1]),
            np.ones(expected_shape[0]),
            rtol=0,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("k", "m", "first_col", "expected"),
    [
        (2, 5, 1, (0.5, 0.5)),
        (4, 9, 0, (-16 / 112, 70 / 112, 70 / 112, -14 / 112, 2 / 112)),
        (6, 13, 0, (-768 / 8448, 4158 / 8448, 6930 / 8448, -2772 / 8448,
                    1188 / 8448, -330 / 8448, 42 / 8448)),
        (8, 17, 0, (-1 / 15, 429 / 1024, 1001 / 1024, -3003 / 5120,
                    429 / 1024, -715 / 3072, 91 / 1024, -21 / 1024, 11 / 5120)),
    ],
)
def test_cton_cpp_binding_matches_pinned_reference_and_python(
    k, m, first_col, expected
):
    """Validate native CtoN against pinned MOLE coefficients and pure Python."""
    native = _native_backend()
    native_operator = native.MimeticInterpolCtoN(m, k=k)
    pure_operator = PureInterpolCtoN(m, k=k)

    assert native_operator.matrix.shape == (m + 1, m + 2)
    np.testing.assert_allclose(
        native_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        native_operator.matrix[1, first_col:first_col + len(expected)].toarray()[0],
        expected,
        rtol=0,
        atol=1e-14,
    )
    raw_operator = native.InterpolCtoN(k, m, [1, 1], [0, 0])
    assert raw_operator.shape() == (m + 1, m + 2)
    np.testing.assert_allclose(
        raw_operator.to_scipy_sparse().toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )


def test_cton_cpp_periodic_and_multidimensional_shapes():
    """Check native CtoN periodic stencils and tensor-product assembly."""
    native = _native_backend()
    periodic = native.MimeticInterpolCtoN(9, boundary="periodic", k=4)
    assert periodic.matrix.shape == (9, 9)
    assert periodic.matrix[0, 0] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 8] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 7] == pytest.approx(-8 / 128)
    assert periodic.matrix[0, 1] == pytest.approx(-8 / 128)

    for dimensions, boundary, expected_shape in (
        ((9, 10), "nonperiodic", (110, 132)),
        ((9, 10), "periodic", (90, 90)),
        ((9, 10, 11), ("periodic", "nonperiodic", "periodic"), (1089, 1188)),
    ):
        native_grid = native.MimeticInterpolCtoN(dimensions, boundary=boundary, k=2)
        pure_grid = PureInterpolCtoN(dimensions, boundary=boundary, k=2)
        assert native_grid.matrix.shape == expected_shape
        difference = (native_grid.matrix - pure_grid.matrix).tocsc()
        np.testing.assert_allclose(difference.data, np.zeros_like(difference.data),
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            native_grid @ np.ones(expected_shape[1]),
            pure_grid @ np.ones(expected_shape[1]),
            rtol=0,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("k", "m", "expected"),
    [
        (2, 5, (0.5,)),
        (4, 9, (35 / 128, 140 / 128, -70 / 128, 28 / 128, -5 / 128)),
        (6, 13, (231 / 1024, 1386 / 1024, -1155 / 1024, 924 / 1024,
                 -495 / 1024, 154 / 1024, -21 / 1024)),
        (8, 17, (6435 / 32768, 6435 / 4096, -15015 / 8192, 9009 / 4096,
                 -32175 / 16384, 5005 / 4096, -4095 / 8192, 495 / 4096,
                 -429 / 32768)),
    ],
)
def test_ftoc_cpp_binding_matches_pinned_reference_and_python(k, m, expected):
    """Validate native FtoC against pinned MOLE coefficients and pure Python."""
    native = _native_backend()
    native_operator = native.MimeticInterpolFtoC(m, k=k)
    pure_operator = PureInterpolFtoC(m, k=k)

    assert native_operator.matrix.shape == (m + 2, m + 1)
    np.testing.assert_allclose(
        native_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        native_operator.matrix[1, :len(expected)].toarray()[0],
        expected,
        rtol=0,
        atol=1e-14,
    )
    raw_operator = native.InterpolFtoC(k, m, [1, 1], [0, 0])
    assert raw_operator.shape() == (m + 2, m + 1)
    np.testing.assert_allclose(
        raw_operator.to_scipy_sparse().toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )


def test_ftoc_cpp_periodic_and_multidimensional_shapes():
    """Check native FtoC periodic behavior and multidimensional assembly."""
    native = _native_backend()
    periodic = native.MimeticInterpolFtoC(9, boundary="periodic", k=4)
    assert periodic.matrix.shape == (9, 9)
    assert periodic.matrix[0, 0] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 1] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 2] == pytest.approx(-8 / 128)
    assert periodic.matrix[0, 8] == pytest.approx(-8 / 128)

    for dimensions, boundary, expected_shape in (
        ((9, 10), "nonperiodic", (264, 199)),
        ((9, 10), "periodic", (180, 180)),
        ((9, 10, 11), ("periodic", "nonperiodic", "periodic"), (3564, 3069)),
    ):
        native_grid = native.MimeticInterpolFtoC(dimensions, boundary=boundary, k=2)
        pure_grid = PureInterpolFtoC(dimensions, boundary=boundary, k=2)
        assert native_grid.matrix.shape == expected_shape
        difference = (native_grid.matrix - pure_grid.matrix).tocsc()
        np.testing.assert_allclose(difference.data, np.zeros_like(difference.data),
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            native_grid @ np.ones(expected_shape[1]),
            np.ones(expected_shape[0]),
            rtol=0,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("k", "m", "first_col", "expected"),
    [
        (2, 5, 0, (0.5, 0.5)),
        (4, 9, 0, (35 / 128, 140 / 128, -70 / 128, 28 / 128, -5 / 128)),
        (6, 13, 0, (231 / 1024, 1386 / 1024, -1155 / 1024, 924 / 1024,
                    -495 / 1024, 154 / 1024, -21 / 1024)),
        (8, 17, 0, (6435 / 32768, 6435 / 4096, -15015 / 8192, 9009 / 4096,
                    -32175 / 16384, 5005 / 4096, -4095 / 8192, 495 / 4096,
                    -429 / 32768)),
    ],
)
def test_ntoc_cpp_binding_matches_pinned_reference_and_python(
    k, m, first_col, expected
):
    """Validate native NtoC against pinned MOLE coefficients and pure Python."""
    native = _native_backend()
    native_operator = native.MimeticInterpolNtoC(m, k=k)
    pure_operator = PureInterpolNtoC(m, k=k)
    assert native_operator.matrix.shape == (m + 2, m + 1)
    np.testing.assert_allclose(
        native_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        native_operator.matrix[1, first_col:first_col + len(expected)].toarray()[0],
        expected,
        rtol=0,
        atol=1e-14,
    )
    raw_operator = native.InterpolNtoC(k, m, [1, 1], [0, 0])
    assert raw_operator.shape() == (m + 2, m + 1)
    np.testing.assert_allclose(
        raw_operator.to_scipy_sparse().toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-14,
    )


def test_ntoc_cpp_periodic_and_multidimensional_shapes():
    """Check native NtoC periodic coefficients and tensor-product assembly."""
    native = _native_backend()
    periodic = native.MimeticInterpolNtoC(9, boundary="periodic", k=4)
    assert periodic.matrix.shape == (9, 9)
    assert periodic.matrix[0, 0] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 1] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 2] == pytest.approx(-8 / 128)
    assert periodic.matrix[0, 8] == pytest.approx(-8 / 128)

    for dimensions, boundary, expected_shape in (
        ((9, 10), "nonperiodic", (132, 110)),
        ((9, 10), "periodic", (90, 90)),
        ((9, 10, 11), ("periodic", "nonperiodic", "periodic"), (1188, 1089)),
    ):
        native_grid = native.MimeticInterpolNtoC(dimensions, boundary=boundary, k=2)
        pure_grid = PureInterpolNtoC(dimensions, boundary=boundary, k=2)
        assert native_grid.matrix.shape == expected_shape
        difference = (native_grid.matrix - pure_grid.matrix).tocsc()
        np.testing.assert_allclose(difference.data, np.zeros_like(difference.data),
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            native_grid @ np.ones(expected_shape[1]),
            np.ones(expected_shape[0]),
            rtol=0,
            atol=1e-14,
        )


def test_public_operator_signatures_match_python():
    """Keep backend-selectable operator constructors API-compatible."""
    native = _native_backend()

    assert set(pure.__all__).issubset(native.__all__)
    for name in (
        "MimeticGradient",
        "MimeticDivergence",
        "MimeticLaplacian",
        "MimeticInterpol",
        "MimeticInterpolCtoF",
        "MimeticInterpolCtoN",
        "MimeticInterpolFtoC",
        "MimeticInterpolNtoC",
        "MimeticRobinBC",
        "MimeticMixedBC",
    ):
        assert inspect.signature(getattr(native, name)) == inspect.signature(
            getattr(pure, name)
        )

    assert "boundary" not in inspect.signature(pure.MimeticLaplacian).parameters
    assert "boundary" not in inspect.signature(pure.MimeticInterpol).parameters
    assert "boundary" not in inspect.signature(native.MimeticLaplacian).parameters
    assert "boundary" not in inspect.signature(native.MimeticInterpol).parameters


@pytest.mark.parametrize(
    ("size", "spacing", "kwargs"),
    [
        (8, 0.1, {}),
        ((8, 8), (0.1, 0.1), {"boundary": "periodic"}),
        ((8, 8), (0.1, 0.1), {"boundary": "nonperiodic"}),
        ((8, 8), (0.1, 0.1), {"boundary": ("nonperiodic", "periodic")}),
        (8, 0.1, {"dc": (0.5, 0.0), "nc": (0.0, 0.0)}),
    ],
)
@pytest.mark.parametrize(
    ("name", "python_operator"),
    [
        ("MimeticGradient", PureGradient),
        ("MimeticDivergence", PureDivergence),
    ],
)
def test_gradient_and_divergence_public_attributes_match_python(
    name, python_operator, size, spacing, kwargs
):
    """Match public boundary metadata and matrices for equivalent inputs."""
    native = _native_backend()
    cpp_operator = getattr(native, name)(size, spacing, **kwargs)
    pure_operator = python_operator(size, spacing, **kwargs)

    assert cpp_operator.boundary == pure_operator.boundary
    np.testing.assert_allclose(
        cpp_operator.matrix.toarray(),
        pure_operator.matrix.toarray(),
        rtol=0,
        atol=1e-12,
    )
