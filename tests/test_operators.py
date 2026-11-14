"""Tests for PyMOLE operators."""

import inspect
from typing import Any, Callable, Tuple

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import odeint
from pymole import (
    Gradient,
    Divergence,
    Laplacian,
    Interpol,
    InterpolCtoF,
    InterpolCtoN,
    InterpolFtoC,
    InterpolNtoC,
    RobinBC,
    MixedBC,
    create_gradient,
    create_divergence,
    create_laplacian,
    create_interpol,
    create_interpol_c_to_f,
    create_interpol_c_to_n,
    create_interpol_f_to_c,
    create_interpol_n_to_c,
    create_robin_bc,
    create_mixed_bc,
    add_scalar_bc,
    apply_scalar_bc_rhs,
    create_scalar_bc_lhs,
    ScalarBoundaryConditions,
    AddScalarBC,
    BC1D,
    BC2D,
    BC3D,
    get_backend,
    use_backend,
)
from pymole.pure.operators import (
    MimeticGradient,
    MimeticDivergence,
    MimeticLaplacian,
    MimeticInterpol,
    MimeticInterpolCtoF,
    MimeticInterpolCtoN,
    MimeticInterpolFtoC,
    MimeticInterpolNtoC,
    MimeticRobinBC as PureMimeticRobinBC,
    MimeticMixedBC as PureMimeticMixedBC,
)


def _load_native_backend() -> Any:
    try:
        # pylint: disable=import-outside-toplevel
        import pymole.cpp as native
    except ImportError as exc:
        cause = exc.__cause__
        cause_name = getattr(cause, "name", "")
        if isinstance(cause, ModuleNotFoundError) and cause_name.endswith("._operators"):
            pytest.skip(f"Native backend unavailable: {exc}")
        if isinstance(cause, ImportError) and "cannot import name '_operators'" in str(cause):
            pytest.skip(f"Native backend unavailable: {exc}")
        if isinstance(cause, ImportError) and "DLL load failed" in str(cause):
            pytest.skip(f"Native backend unavailable: {exc}")
        raise
    return native


def _call_with_unsupported_boundary(callable_: Callable[..., Any]) -> Any:
    return callable_(8, 0.125, **{"boundary": "periodic"})


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

def test_operator_validation():
    """Test operator input validation."""
    with pytest.raises(ValueError):
        MimeticGradient(0, 1.0)  # Invalid grid points
    with pytest.raises(ValueError):
        MimeticGradient(10, -1.0)  # Invalid grid spacing


def test_mole_shaped_operators_dispatch_to_the_selected_backend():
    """Keep MOLE's constructor forms available through backend selection."""
    previous_backend = get_backend()
    use_backend("python")
    try:
        operators = (
            (Gradient(2, 8, 0.125), create_gradient(8, 0.125)),
            (Divergence(2, 8, 0.125), create_divergence(8, 0.125)),
            (Laplacian(2, 8, 0.125), create_laplacian(8, 0.125)),
            (Interpol(9, 0.4), create_interpol(9, 1.0, 0.4)),
            (
                InterpolCtoF(2, 9, [1, 1], [0, 0]),
                create_interpol_c_to_f(9, k=2),
            ),
            (
                InterpolCtoN(2, 9, [1, 1], [0, 0]),
                create_interpol_c_to_n(9, k=2),
            ),
            (
                InterpolFtoC(2, 9, [1, 1], [0, 0]),
                create_interpol_f_to_c(9, k=2),
            ),
            (
                InterpolNtoC(2, 9, [1, 1], [0, 0]),
                create_interpol_n_to_c(9, k=2),
            ),
            (RobinBC(2, 9, 0.125, 1.0, 0.5),
             create_robin_bc(9, 0.125, k=2, a=1.0, b=0.5)),
            (
                MixedBC(
                    2, 9, 0.125, "Dirichlet", [1.0], "Neumann", [0.0]
                ),
                create_mixed_bc(
                    9, 0.125, k=2, left="Dirichlet", coeffs_left=[1.0],
                    right="Neumann", coeffs_right=[0.0],
                ),
            ),
        )
        for mole_operator, factory_operator in operators:
            np.testing.assert_allclose(
                mole_operator.to_scipy_sparse().toarray(),
                factory_operator.matrix.toarray(),
                rtol=0.0,
                atol=1e-12,
            )
            assert mole_operator.shape() == factory_operator.matrix.shape
            assert mole_operator.nnz() == factory_operator.matrix.nnz

        periodic_gradient = Gradient(
            k=2, m=8, dx=0.125, dc=[0, 0], nc=[0, 0]
        )
        assert periodic_gradient.shape() == (8, 8)
        assert isinstance(periodic_gradient.getP(), list)
        periodic_divergence = Divergence(
            k=2, m=8, dx=0.125, dc=[0, 0], nc=[0, 0]
        )
        assert isinstance(periodic_divergence.getQ(), list)
        assert Gradient(2, 4, 5, 0.25, 0.2).shape() == (49, 42)

        mixed_boundary = {
            "left": "Dirichlet", "coeffs_left": [1.0],
            "right": "Neumann", "coeffs_right": [0.0],
            "bottom": "Robin", "coeffs_bottom": [1.0, 0.5],
            "top": "Dirichlet", "coeffs_top": [1.0],
        }
        multidimensional = (
            (
                Gradient(2, 5, 6, 0.2, 0.15),
                create_gradient((5, 6), (0.2, 0.15)),
            ),
            (
                Divergence(2, 5, 6, 0.2, 0.15),
                create_divergence((5, 6), (0.2, 0.15)),
            ),
            (
                Laplacian(2, 5, 6, 0.2, 0.15),
                create_laplacian((5, 6), (0.2, 0.15)),
            ),
            (
                Interpol(5, 6, 0.4, 0.6),
                create_interpol((5, 6), (1.0, 1.0), (0.4, 0.6)),
            ),
            (
                InterpolCtoF(2, 5, 6, [1, 1, 1, 1], [0, 0, 0, 0]),
                create_interpol_c_to_f((5, 6), k=2),
            ),
            (
                InterpolCtoN(2, 5, 6, [1, 1, 1, 1], [0, 0, 0, 0]),
                create_interpol_c_to_n((5, 6), k=2),
            ),
            (
                InterpolFtoC(2, 5, 6, [1, 1, 1, 1], [0, 0, 0, 0]),
                create_interpol_f_to_c((5, 6), k=2),
            ),
            (
                InterpolNtoC(2, 5, 6, [1, 1, 1, 1], [0, 0, 0, 0]),
                create_interpol_n_to_c((5, 6), k=2),
            ),
            (
                RobinBC(2, 5, 0.2, 6, 0.15, 1.0, 0.5),
                create_robin_bc((5, 6), (0.2, 0.15), k=2, a=1.0, b=0.5),
            ),
            (
                MixedBC(
                    2, 5, 0.2, 6, 0.15,
                    "Dirichlet", [1.0], "Neumann", [0.0],
                    "Robin", [1.0, 0.5], "Dirichlet", [1.0],
                ),
                create_mixed_bc(
                    (5, 6), (0.2, 0.15), k=2, **mixed_boundary
                ),
            ),
        )
        for mole_operator, factory_operator in multidimensional:
            np.testing.assert_allclose(
                mole_operator.to_scipy_sparse().toarray(),
                factory_operator.matrix.toarray(),
                rtol=0.0,
                atol=1e-12,
            )
    finally:
        use_backend(previous_backend)


def test_add_scalar_bc_1d_replaces_boundary_equations():
    n, h = 8, 0.125
    laplacian = create_laplacian(n, h).matrix
    rhs = np.arange(n + 2, dtype=float)

    matrix, updated_rhs = add_scalar_bc(
        laplacian, rhs, n, h,
        ScalarBoundaryConditions(dc=[1.0, 1.0], nc=[0.0, 0.0], values=[2.5, -1.0]),
    )

    expected = laplacian.toarray()
    expected[0, :] = 0.0
    expected[0, 0] = 1.0
    expected[-1, :] = 0.0
    expected[-1, -1] = 1.0
    np.testing.assert_allclose(matrix.toarray(), expected)
    np.testing.assert_allclose(updated_rhs, [2.5, *rhs[1:-1], -1.0])
    np.testing.assert_array_equal(rhs, np.arange(n + 2, dtype=float))


def test_add_scalar_bc_2d_applies_side_values_without_double_counting_corners():
    dimensions, spacings = (4, 5), (0.25, 0.2)
    size = (dimensions[0] + 2) * (dimensions[1] + 2)
    operator = sparse.csc_matrix((size, size))
    rhs = np.zeros(size)
    dc = [1.0] * 4
    nc = [0.0] * 4
    values = ([1.0] * dimensions[1], [2.0] * dimensions[1],
              [3.0] * (dimensions[0] + 2), [4.0] * (dimensions[0] + 2))

    matrix, updated_rhs = add_scalar_bc(
        operator, rhs, dimensions, spacings,
        ScalarBoundaryConditions(dc=dc, nc=nc, values=values),
    )

    assert matrix.shape == (size, size)
    assert np.count_nonzero(updated_rhs == 1.0) == dimensions[1]
    assert np.count_nonzero(updated_rhs == 2.0) == dimensions[1]
    assert np.count_nonzero(updated_rhs == 3.0) == dimensions[0] + 2
    assert np.count_nonzero(updated_rhs == 4.0) == dimensions[0] + 2
    assert np.all(matrix.diagonal()[updated_rhs != 0.0] == 1.0)


def test_add_scalar_bc_periodic_axis_has_no_ghost_nodes():
    dimensions, spacings = (4, 5), (0.25, 0.2)
    size = dimensions[0] * (dimensions[1] + 2)
    operator = sparse.csc_matrix((size, size))
    rhs = np.zeros(size)

    matrix, updated_rhs = add_scalar_bc(
        operator,
        rhs,
        dimensions,
        spacings,
        ScalarBoundaryConditions(
            dc=[0.0, 0.0, 1.0, 1.0],
            nc=[0.0, 0.0, 0.0, 0.0],
            values=([], [], [5.0] * dimensions[0], [6.0] * dimensions[0]),
        ),
    )

    assert matrix.shape == (size, size)
    assert np.count_nonzero(updated_rhs == 5.0) == dimensions[0]
    assert np.count_nonzero(updated_rhs == 6.0) == dimensions[0]


def test_add_scalar_bc_1d_neumann_uses_outward_derivative_rows():
    n, h = 8, 0.125
    gradient = MimeticGradient(n, h).matrix
    operator = sparse.csc_matrix((n + 2, n + 2))

    matrix, updated_rhs = add_scalar_bc(
        operator, np.zeros(n + 2), n, h,
        ScalarBoundaryConditions(dc=[0.0, 0.0], nc=[1.0, 1.0], values=[3.0, 4.0]),
    )

    np.testing.assert_allclose(matrix.getrow(0).toarray(), -gradient.getrow(0).toarray())
    np.testing.assert_allclose(matrix.getrow(-1).toarray(), gradient.getrow(n).toarray())
    assert updated_rhs[0] == 3.0
    assert updated_rhs[-1] == 4.0


def test_add_scalar_bc_1d_updates_both_sides_of_active_boundary_pair():
    n, h = 8, 0.125
    operator = sparse.csc_matrix(np.ones((n + 2, n + 2)))

    matrix, updated_rhs = add_scalar_bc(
        operator, np.ones(n + 2), n, h,
        ScalarBoundaryConditions(dc=[1.0, 0.0], nc=[0.0, 0.0], values=[2.0, 3.0]),
    )

    assert matrix.getrow(0).nnz == 1
    assert matrix.getrow(-1).nnz == 0
    assert updated_rhs[0] == 2.0
    assert updated_rhs[-1] == 3.0


def test_add_scalar_bc_3d_applies_boundary_values_on_each_face():
    dimensions, spacings = (4, 5, 6), (0.25, 0.2, 0.125)
    effective = tuple(size + 2 for size in dimensions)
    size = int(np.prod(effective))
    operator = sparse.csc_matrix((size, size))
    values = np.asarray([
        [1.0] * (dimensions[1] * dimensions[2]),
        [2.0] * (dimensions[1] * dimensions[2]),
        [3.0] * ((dimensions[0] + 2) * dimensions[2]),
        [4.0] * ((dimensions[0] + 2) * dimensions[2]),
        [5.0] * ((dimensions[0] + 2) * (dimensions[1] + 2)),
        [6.0] * ((dimensions[0] + 2) * (dimensions[1] + 2)),
    ], dtype=object)

    matrix, updated_rhs = add_scalar_bc(
        operator, np.zeros(size), dimensions, spacings,
        ScalarBoundaryConditions(dc=[1.0] * 6, nc=[0.0] * 6, values=values),
    )

    assert matrix.shape == (size, size)
    for face, expected_count in enumerate((30, 30, 36, 36, 42, 42), start=1):
        assert np.count_nonzero(updated_rhs == face) == expected_count
    np.testing.assert_allclose(matrix.diagonal()[updated_rhs != 0.0], 1.0)


@pytest.mark.parametrize(("dimensions", "spacings", "dc", "nc", "k"), [
    ((5,), (0.2,), (1.0, 0.0), (0.25, 1.0), 2),
    ((9,), (0.125,), (1.0, 0.0), (0.25, 1.0), 4),
    ((13,), (0.075,), (1.0, 0.0), (0.25, 1.0), 6),
    ((17,), (0.05,), (1.0, 0.0), (0.25, 1.0), 8),
    ((9,), (0.125,), (1.0, 1.0), (0.0, 0.0), 0),
    ((4, 5), (0.25, 0.2), (1.0, 0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 0.0), 2),
    ((4, 5), (0.25, 0.2), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), 2),
    ((4, 5), (0.25, 0.2), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 2),
    ((4, 5, 6), (0.25, 0.2, 0.125),
     (1.0, 0.0, 0.0, 1.0, 1.0, 0.0),
     (0.0, 0.5, 1.0, 0.0, 0.25, 1.0), 2),
])
def test_add_scalar_bc_public_api_matches_selected_backends(
    dimensions: Tuple[int, ...],
    spacings: Tuple[float, ...],
    dc: Tuple[float, ...],
    nc: Tuple[float, ...],
    k: int,
):
    _load_native_backend()
    active_axes = tuple(
        any(value != 0.0 for value in (*dc[2 * axis:2 * axis + 2],
                                       *nc[2 * axis:2 * axis + 2]))
        for axis in range(len(dimensions))
    )
    face_sizes = []
    for axis in range(len(dimensions)):
        face_size = 1
        for other_axis, size in enumerate(dimensions):
            if other_axis == axis:
                continue
            if other_axis < axis and active_axes[other_axis]:
                face_size *= size + 2
            else:
                face_size *= size
        face_sizes.extend((face_size, face_size))
    if len(dimensions) == 1:
        values = (1.25, 2.25)
    else:
        values = tuple(
            np.linspace(side + 0.25, side + 1.25, face_size)
            for side, face_size in enumerate(face_sizes)
        )
    system_size = int(np.prod([
        size + 2 if active_axes[axis] else size
        for axis, size in enumerate(dimensions)
    ]))
    operator = sparse.diags(np.linspace(1.0, 2.0, system_size), format="csc")
    rhs = np.linspace(-2.0, 3.0, system_size)
    conditions = ScalarBoundaryConditions(
        dc=dc, nc=nc, values=values, k=k
    )
    args = (operator, rhs, dimensions, spacings, conditions)
    original_backend = get_backend()
    try:
        use_backend("python")
        python_result = add_scalar_bc(*args)
        use_backend("cpp")
        cpp_result = add_scalar_bc(*args)
    finally:
        use_backend(original_backend)

    np.testing.assert_allclose(
        cpp_result[0].toarray(), python_result[0].toarray(), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(cpp_result[1], python_result[1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(("dimensions", "spacings", "dc", "nc"), [
    ((9,), (0.125,), (1.0, 0.0), (0.0, 1.0)),
    ((5, 6), (0.2, 0.15), (1.0, 0.0, 0.0, 1.0), (0.0, 0.5, 1.0, 0.0)),
    ((5, 6), (0.2, 0.15), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((5, 6, 7), (0.2, 0.15, 0.1),
     (1.0, 0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 0.5, 1.0, 0.0, 0.25, 1.0)),
])
def test_scalar_bc_lhs_and_rhs_helpers_match_native(
    dimensions: Tuple[int, ...],
    spacings: Tuple[float, ...],
    dc: Tuple[float, ...],
    nc: Tuple[float, ...],
):
    _load_native_backend()
    axis_count = len(dimensions)
    active_axes = tuple(
        any(value != 0.0 for value in (*dc[2 * axis:2 * axis + 2],
                                       *nc[2 * axis:2 * axis + 2]))
        for axis in range(axis_count)
    )
    face_sizes = []
    for axis in range(axis_count):
        face_size = 1
        for other_axis, size in enumerate(dimensions):
            if other_axis != axis:
                face_size *= size + 2 if other_axis < axis and active_axes[other_axis] else size
        face_sizes.extend((face_size, face_size))
    values = (
        (1.0, 2.0)
        if axis_count == 1
        else tuple(
            np.full(face_sizes[side], side + 1.0)
            for side in range(2 * axis_count)
        )
    )
    conditions = ScalarBoundaryConditions(dc=dc, nc=nc, values=values)
    original_backend = get_backend()
    try:
        use_backend("python")
        python_matrices = create_scalar_bc_lhs(dimensions, spacings, conditions)
        use_backend("cpp")
        cpp_matrices = create_scalar_bc_lhs(dimensions, spacings, conditions)
    finally:
        use_backend(original_backend)

    assert len(python_matrices) == len(cpp_matrices) == 2 * axis_count
    for python_matrix, cpp_matrix in zip(python_matrices, cpp_matrices):
        assert python_matrix.shape == cpp_matrix.shape
        np.testing.assert_allclose(
            cpp_matrix.toarray(), python_matrix.toarray(), rtol=0.0, atol=1e-12
        )

    side_rows = tuple(
        np.unique(matrix.nonzero()[0]).astype(np.int64)
        for matrix in python_matrices
    )
    system_size = int(np.prod([
        size + 2 if np.any(dc[2 * axis:2 * axis + 2]) or
        np.any(nc[2 * axis:2 * axis + 2]) else size
        for axis, size in enumerate(dimensions)
    ]))
    rhs = np.linspace(-1.0, 1.0, system_size)
    try:
        use_backend("python")
        python_rhs = apply_scalar_bc_rhs(rhs, conditions, side_rows)
        use_backend("cpp")
        cpp_rhs = apply_scalar_bc_rhs(rhs, conditions, side_rows)
    finally:
        use_backend(original_backend)
    np.testing.assert_allclose(cpp_rhs, python_rhs, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("axis_count", [1, 2, 3])
def test_mole_shaped_add_scalar_bc_overloads_mutate_inputs(backend, axis_count):
    _load_native_backend()
    if axis_count == 1:
        dimensions, spacings = (9,), (0.125,)
        conditions = BC1D()
        conditions.dc[:] = [1.0, 0.0]
        conditions.nc[:] = [0.0, 0.5]
        conditions.v[:] = [2.0, -1.0]
    elif axis_count == 2:
        dimensions, spacings = (5, 6), (0.2, 0.15)
        conditions = BC2D()
        conditions.dc[:] = [1.0, 0.0, 0.0, 1.0]
        conditions.nc[:] = [0.0, 0.5, 1.0, 0.0]
        conditions.v = [
            np.linspace(1.0, 2.0, dimensions[1]),
            np.linspace(2.0, 3.0, dimensions[1]),
            np.linspace(3.0, 4.0, dimensions[0] + 2),
            np.linspace(4.0, 5.0, dimensions[0] + 2),
        ]
    else:
        dimensions, spacings = (4, 5, 6), (0.25, 0.2, 0.125)
        conditions = BC3D()
        conditions.dc[:] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        conditions.nc[:] = [0.0, 0.5, 1.0, 0.0, 0.25, 1.0]
        conditions.v = [
            np.ones(30), np.full(30, 2.0),
            np.full(36, 3.0), np.full(36, 4.0),
            np.full(42, 5.0), np.full(42, 6.0),
        ]

    dimension_values = list(dimensions)
    spacing_values = list(spacings)
    active_axes = tuple(
        np.any(conditions.dc[2 * axis:2 * axis + 2])
        or np.any(conditions.nc[2 * axis:2 * axis + 2])
        for axis in range(axis_count)
    )
    system_size = int(np.prod([
        size + 2 if active_axes[axis] else size
        for axis, size in enumerate(dimensions)
    ]))
    original_matrix = sparse.diags(np.linspace(1.0, 2.0, system_size), format="csc")
    original_rhs = np.linspace(-1.0, 1.0, system_size)
    expected_matrix, expected_rhs = add_scalar_bc(
        original_matrix,
        original_rhs,
        dimensions[0] if axis_count == 1 else dimensions,
        spacings[0] if axis_count == 1 else spacings,
        ScalarBoundaryConditions(
            dc=conditions.dc, nc=conditions.nc, values=conditions.v, k=2
        ),
    )
    matrix = original_matrix.copy()
    rhs = original_rhs.copy()
    if axis_count == 1:
        args = (
            matrix, rhs, 2, dimension_values[0], spacing_values[0], conditions
        )
    elif axis_count == 2:
        args = (
            matrix, rhs, 2, dimension_values[0], spacing_values[0],
            dimension_values[1], spacing_values[1], conditions,
        )
    else:
        args = (
            matrix, rhs, 2, dimension_values[0], spacing_values[0],
            dimension_values[1], spacing_values[1],
            dimension_values[2], spacing_values[2], conditions,
        )

    original_backend = get_backend()
    try:
        use_backend(backend)
        result = AddScalarBC.addScalarBC(*args)
    finally:
        use_backend(original_backend)

    assert result is None
    np.testing.assert_allclose(matrix.toarray(), expected_matrix.toarray(), atol=1e-12)
    np.testing.assert_allclose(rhs, expected_rhs, atol=1e-12)


def test_native_mole_shaped_add_scalar_bc_accepts_empty_sparse_operator():
    _load_native_backend()
    conditions = BC1D()
    conditions.dc[:] = [1.0, 1.0]
    conditions.v[:] = [2.0, 3.0]
    matrix = sparse.csc_matrix((11, 11))
    rhs = np.zeros(11)
    original_backend = get_backend()
    try:
        use_backend("cpp")
        result = AddScalarBC.addScalarBC(matrix, rhs, 2, 9, 0.125, conditions)
    finally:
        use_backend(original_backend)

    assert result is None
    assert matrix.nnz == 2
    np.testing.assert_allclose(matrix.diagonal()[[0, -1]], [1.0, 1.0])
    np.testing.assert_allclose(rhs[[0, -1]], [2.0, 3.0])


def test_mole_shaped_add_scalar_bc_validates_system_before_periodic_noop():
    conditions = BC1D()
    with pytest.raises(ValueError, match="square"):
        AddScalarBC.addScalarBC(
            sparse.csc_matrix((2, 3)), np.zeros(3), 2, 1, 1.0, conditions
        )


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_mole_shaped_add_scalar_bc_lhs_and_rhs_overloads(backend):
    _load_native_backend()
    conditions = BC1D()
    conditions.dc[:] = [1.0, 0.0]
    conditions.nc[:] = [0.0, 0.5]
    conditions.v[:] = [3.0, 4.0]
    original_backend = get_backend()
    left = sparse.csc_matrix((0, 0))
    right = sparse.csc_matrix((0, 0))
    try:
        use_backend(backend)
        lhs_result = AddScalarBC.addScalarBClhs(
            2, 9, 0.125, conditions.dc, conditions.nc, left, right
        )
        rhs = np.zeros(5)
        result = AddScalarBC.addScalarBCrhs(rhs, conditions.v, [0, 4])
    finally:
        use_backend(original_backend)

    assert left.shape == right.shape == (11, 11)
    assert lhs_result is None
    assert result is None
    np.testing.assert_array_equal(rhs, [3.0, 0.0, 0.0, 0.0, 4.0])


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("axis_count", [2, 3])
def test_mole_shaped_multidimensional_lhs_rhs_overloads(backend, axis_count):
    _load_native_backend()
    if axis_count == 2:
        dimensions, spacings = (5, 6), (0.2, 0.15)
        dc = np.asarray([1.0, 0.0, 0.0, 1.0])
        nc = np.asarray([0.0, 0.5, 1.0, 0.0])
    else:
        dimensions, spacings = (4, 5, 6), (0.25, 0.2, 0.125)
        dc = np.asarray([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        nc = np.asarray([0.0, 0.5, 1.0, 0.0, 0.25, 1.0])
    dimension_values = list(dimensions)
    spacing_values = list(spacings)
    conditions = ScalarBoundaryConditions(dc=dc, nc=nc, values=())
    expected_matrices = create_scalar_bc_lhs(dimensions, spacings, conditions)
    output_matrices = [sparse.csc_matrix((0, 0)) for _ in expected_matrices]
    if axis_count == 2:
        lhs_args = (
            2, dimension_values[0], spacing_values[0],
            dimension_values[1], spacing_values[1], dc, nc,
            *output_matrices,
        )
    else:
        lhs_args = (
            2, dimension_values[0], spacing_values[0],
            dimension_values[1], spacing_values[1],
            dimension_values[2], spacing_values[2], dc, nc, *output_matrices,
        )

    rows = tuple(np.unique(matrix.nonzero()[0]) for matrix in expected_matrices)
    values = [np.full(index.size, side + 1.0) for side, index in enumerate(rows)]
    boundary_conditions = ScalarBoundaryConditions(dc=dc, nc=nc, values=values)
    system_size = expected_matrices[0].get_shape()[0]
    rhs = np.zeros(system_size)
    expected_rhs = apply_scalar_bc_rhs(rhs, boundary_conditions, rows)
    actual_rhs = rhs.copy()
    if axis_count == 2:
        rhs_args = (actual_rhs, dc, nc, values, *rows)
    else:
        rhs_args = (actual_rhs, dc, nc, values, *rows)

    original_backend = get_backend()
    try:
        use_backend(backend)
        lhs_result = AddScalarBC.addScalarBClhs(*lhs_args)
        result = AddScalarBC.addScalarBCrhs(*rhs_args)
    finally:
        use_backend(original_backend)

    assert lhs_result is None
    assert result is None
    actual_matrices = output_matrices
    for expected, actual in zip(expected_matrices, actual_matrices):
        assert expected.shape == actual.shape
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-12)
    np.testing.assert_allclose(actual_rhs, expected_rhs, atol=1e-12)


# ============================================================================
# BASIC FUNCTIONALITY TESTS (ORIGINAL)
# ============================================================================

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_constant(n, h):
    """Test gradient of constant function is zero."""
    grad = MimeticGradient(n, h)
    x = np.ones(n + 2)
    np.testing.assert_allclose(grad @ x, np.zeros(n + 1), atol=1e-13)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_gradient_linear(n, h):
    """Test MOLE's interior gradient rows differentiate linear data."""
    grad = MimeticGradient(n, h)
    x = np.arange(n + 2) * h
    np.testing.assert_allclose((grad @ x)[1:-1], np.ones(n - 1), atol=1e-14)

@pytest.mark.parametrize("n,h", [(10, 0.1), (100, 0.01)])
def test_divergence_gradient_laplacian(n, h):
    """Test that div(grad(x)) equals Laplacian(x)."""
    grad = MimeticGradient(n, h)
    div = MimeticDivergence(n, h)
    lap = MimeticLaplacian(n, h)

    x = np.random.default_rng(0).random(n + 2)
    div_grad = div @ (grad @ x)
    lap_x = lap @ x

    np.testing.assert_allclose(div_grad, lap_x, atol=1e-14)

@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_mole_shapes_spacing_and_periodic_adjoint(k):
    """Test: MOLE operator shapes, spacing, and periodic adjoint property."""
    m = 2 * k + 1
    dx = 0.037
    grad = MimeticGradient(m, dx, k=k)
    div = MimeticDivergence(m, dx, k=k)
    lap = MimeticLaplacian(m, dx, k=k)

    assert grad.matrix.shape == (m + 1, m + 2)
    assert div.matrix.shape == (m + 2, m + 1)
    assert lap.matrix.shape == (m + 2, m + 2)
    assert grad.h == div.h == lap.h == dx
    composed_laplacian = sparse.csc_matrix(div.matrix @ grad.matrix)
    np.testing.assert_allclose(
        lap.matrix.toarray(),
        composed_laplacian.toarray(),
        atol=1e-12
    )

    grad_periodic = MimeticGradient(m, dx, boundary="periodic", k=k)
    div_periodic = MimeticDivergence(m, dx, boundary="periodic", k=k)
    assert grad_periodic.matrix.shape == div_periodic.matrix.shape == (m, m)
    np.testing.assert_allclose(
        div_periodic.matrix.toarray(),
        -grad_periodic.matrix.toarray().T,
        atol=1e-12
    )


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_native_matrix_parity_1d(k):
    """Test: Native matrix parity for 1D operators."""
    native = _load_native_backend()
    m, dx = 2 * k + 1, 0.037
    pure_operators = (
        MimeticGradient(m, dx, k=k),
        MimeticDivergence(m, dx, k=k),
        MimeticLaplacian(m, dx, k=k),
    )
    native_operators = (
        native.MimeticGradient(m, dx, k=k),
        native.MimeticDivergence(m, dx, k=k),
        native.MimeticLaplacian(m, dx, k=k),
    )
    for pure_operator, native_operator in zip(pure_operators, native_operators):
        np.testing.assert_allclose(
            pure_operator.matrix.toarray(), native_operator.matrix.toarray(), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("dimensions,spacings", [
    ((7, 8), (0.2, 0.15)), ((7, 8, 9), (0.2, 0.15, 0.1)),
])
def test_native_matrix_parity_multidimensional(dimensions, spacings):
    """Test: Native matrix parity for multi-dimensional operators"""
    native = _load_native_backend()
    k = 2
    pure_operators = (
        MimeticGradient(dimensions, spacings, k=k),
        MimeticDivergence(dimensions, spacings, k=k),
        MimeticLaplacian(dimensions, spacings, k=k),
    )
    native_operators = (
        native.Gradient(k, *dimensions, *spacings),
        native.Divergence(k, *dimensions, *spacings),
        native.Laplacian(k, *dimensions, *spacings),
    )
    for pure_operator, native_operator in zip(pure_operators, native_operators):
        np.testing.assert_allclose(
            pure_operator.matrix.toarray(),
            native_operator.to_scipy_sparse().toarray(),
            rtol=0,
            atol=1e-12,
        )


def test_native_matrix_parity_interpolation_and_boundaries():
    """Test: Native matrix parity for interpolation and boundary conditions."""
    native = _load_native_backend()
    m, dx, k, c = 9, 0.125, 2, 0.3
    pure_interpol = MimeticInterpol(m, dx, c)
    native_interpol = native.Interpol(m, c)
    np.testing.assert_allclose(
        pure_interpol.matrix.toarray(),
        native_interpol.to_scipy_sparse().toarray(),
        rtol=0, atol=1e-12
    )

    pure_robin = PureMimeticRobinBC(m, dx, k=k, a=2.0, b=0.5)
    native_robin = native.MimeticRobinBC(m, dx, k=k, a=2.0, b=0.5)
    np.testing.assert_allclose(
        pure_robin.matrix.toarray(), native_robin.matrix.toarray(), rtol=0, atol=1e-12
    )

    mixed_args = {
        "left": "Dirichlet", "coeffs_left": [2.0],
        "right": "Robin", "coeffs_right": [3.0, 0.5],
    }
    pure_mixed = PureMimeticMixedBC(m, dx, k=k, **mixed_args)
    native_mixed = native.MimeticMixedBC(m, dx, k=k, **mixed_args)
    np.testing.assert_allclose(
        pure_mixed.matrix.toarray(), native_mixed.matrix.toarray(), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dimensions,spacings", [
    ((7, 8), (0.2, 0.15)), ((7, 8, 9), (0.2, 0.15, 0.1)),
])
def test_mole_multidimensional_shapes_and_composition(dimensions, spacings):
    """Test: Multi-dimensional operator shapes and divergence and gradient equals Laplacian."""
    grad = MimeticGradient(dimensions, spacings)
    div = MimeticDivergence(dimensions, spacings)
    lap = MimeticLaplacian(dimensions, spacings)
    input_size = int(np.prod([size + 2 for size in dimensions]))

    gradient_rows, gradient_columns = grad.matrix.toarray().shape
    divergence_rows, divergence_columns = div.matrix.toarray().shape
    assert gradient_columns == input_size
    assert divergence_rows == input_size
    assert gradient_rows == divergence_columns
    assert lap.matrix.shape == (input_size, input_size)
    composed_laplacian = sparse.csc_matrix(div.matrix @ grad.matrix)
    np.testing.assert_allclose(lap.matrix.toarray(), composed_laplacian.toarray(), atol=1e-12)


def test_mole_interpolation_staggered_weights():
    """Test: Multi-dimensional interpolation operator with staggered weights."""
    m, dx, c = 7, 0.2, 0.3
    interp = MimeticInterpol(m, dx, c).matrix

    assert interp.shape == (m + 1, m + 2)
    assert interp[0, 0] == 1.0
    assert interp[m, m + 1] == 1.0
    assert interp[1, 1] == c
    assert interp[1, 2] == 1.0 - c

    dimensions = (7, 8, 9)
    spacings = (0.2, 0.15, 0.1)
    interp_3d = MimeticInterpol(dimensions, spacings, (0.3, 0.4, 0.5)).matrix
    expected_rows = (
        dimensions[2] * dimensions[1] * (dimensions[0] + 1)
        + dimensions[2] * (dimensions[1] + 1) * dimensions[0]
        + (dimensions[2] + 1) * dimensions[1] * dimensions[0]
    )
    assert interp_3d.shape == (expected_rows, int(np.prod([size + 2 for size in dimensions])))


def test_mole_robin_and_mixed_boundary_rows():
    """Test: Multi-dimensional Robin and Mixed boundary condition operators."""
    m, dx, k = 9, 0.125, 2
    gradient = MimeticGradient(m, dx, k=k).matrix
    robin = PureMimeticRobinBC(m, dx, k=k, a=2.0, b=0.5).matrix.toarray()
    mixed = PureMimeticMixedBC(
        m, dx, k=k,
        left="Dirichlet", coeffs_left=[2.0],
        right="Robin", coeffs_right=[3.0, 0.5],
    ).matrix.toarray()

    expected_robin = np.zeros((m + 2, m + 2))
    expected_robin[0, 0] = 2.0
    expected_robin[-1, -1] = 2.0
    expected_robin[0] -= 0.5 * gradient.toarray()[0]
    expected_robin[-1] += 0.5 * gradient.toarray()[m]
    np.testing.assert_allclose(robin, expected_robin, atol=1e-14)

    expected_mixed = np.zeros((m + 2, m + 2))
    expected_mixed[0, 0] = 2.0
    expected_mixed[-1, -1] = 3.0
    expected_mixed[-1] += 0.5 * gradient.toarray()[m]
    np.testing.assert_allclose(mixed, expected_mixed, atol=1e-14)

    dimensions, spacings = (7, 8), (0.2, 0.15)
    assert PureMimeticRobinBC(dimensions, spacings).matrix.shape == (90, 90)
    assert PureMimeticMixedBC(
        dimensions, spacings,
        left="Dirichlet", coeffs_left=[1.0], right="Dirichlet", coeffs_right=[1.0],
        bottom="Neumann", coeffs_bottom=[1.0], top="Neumann", coeffs_top=[1.0],
    ).matrix.shape == (90, 90)

@pytest.mark.parametrize("backend", ["python"])
def test_backends(backend):
    """Test backend selection."""
    use_backend(backend)
    n, h = 10, 0.1
    try:
        grad = create_gradient(n, h)
        x = np.ones(n + 2)
        result = grad @ x
        assert result.shape == (n + 1,)
    except ImportError:
        if backend == "python":
            raise
        pytest.skip("C++ backend not available")


def test_public_factories_forward_python_options():
    """Test public factory functions with Python backend and various options."""
    use_backend("python")
    dimensions, spacings = (7, 8), (0.2, 0.15)

    gradient = create_gradient(dimensions, spacings, boundary="periodic", k=2)
    divergence = create_divergence(dimensions, spacings, boundary="periodic", k=2)
    interpolation = create_interpol(dimensions, spacings, c=(0.3, 0.4))
    robin = create_robin_bc(9, 0.125, k=2, a=2.0, b=0.5)
    mixed_args = {
        "left": "Dirichlet", "coeffs_left": [2.0],
        "right": "Robin", "coeffs_right": [3.0, 0.5],
    }
    mixed = create_mixed_bc(9, 0.125, k=2, **mixed_args)

    assert gradient.boundary == divergence.boundary == "periodic"
    assert gradient.matrix.shape == (112, 56)
    assert divergence.matrix.shape == (56, 112)
    assert create_laplacian(dimensions, spacings, k=2).matrix.shape == (90, 90)
    np.testing.assert_allclose(
        interpolation.matrix.toarray(),
        MimeticInterpol(dimensions, spacings, c=(0.3, 0.4)).matrix.toarray(),
    )
    np.testing.assert_allclose(
        robin.matrix.toarray(),
        PureMimeticRobinBC(9, 0.125, k=2, a=2.0, b=0.5).matrix.toarray(),
    )
    np.testing.assert_allclose(
        mixed.matrix.toarray(),
        PureMimeticMixedBC(9, 0.125, k=2, **mixed_args).matrix.toarray(),
    )


def test_laplacian_and_interpolation_expose_only_mole_cpp_parameters():
    """Keep Python-only boundary selectors out of the shared API."""
    for callable_ in (
        create_laplacian,
        MimeticLaplacian,
        create_interpol,
        MimeticInterpol,
    ):
        assert "boundary" not in inspect.signature(callable_).parameters

    with pytest.raises(TypeError):
        _call_with_unsupported_boundary(create_laplacian)
    with pytest.raises(TypeError):
        _call_with_unsupported_boundary(MimeticLaplacian)
    with pytest.raises(TypeError):
        _call_with_unsupported_boundary(create_interpol)
    with pytest.raises(TypeError):
        _call_with_unsupported_boundary(MimeticInterpol)


@pytest.mark.parametrize(
    ("k", "m", "first_row", "first_col", "interior_row", "interior_col",
     "interior_values"),
    [
        (2, 5, (0.5, 0.5), 1, 1, 1, (0.5, 0.5)),
        (4, 9, (-16 / 112, 70 / 112, 70 / 112, -14 / 112, 2 / 112),
         0, 2, 1, (-7 / 112, 63 / 112, 63 / 112, -7 / 112)),
        (6, 13, (-768 / 8448, 4158 / 8448, 6930 / 8448, -2772 / 8448,
                 1188 / 8448, -330 / 8448, 42 / 8448),
         0, 3, 1, (99 / 8448, -825 / 8448, 4950 / 8448, 4950 / 8448,
                   -825 / 8448, 99 / 8448)),
        (8, 17, (-1 / 15, 429 / 1024, 1001 / 1024, -3003 / 5120,
                 429 / 1024, -715 / 3072, 91 / 1024, -21 / 1024, 11 / 5120),
         0, 4, 1, (-5 / 2048, 49 / 2048, -245 / 2048, 1225 / 2048,
                   1225 / 2048, -245 / 2048, 49 / 2048, -5 / 2048)),
    ],
)
def test_ctof_nonperiodic_1d_matches_pinned_mole_rows(
    k, m, first_row, first_col, interior_row, interior_col, interior_values
):
    """Check reference stencil rows copied from pinned MOLE C++."""
    operator = MimeticInterpolCtoF(m, k=k)
    matrix = operator.matrix.toarray()
    assert matrix.shape == (m + 1, m + 2)
    np.testing.assert_allclose(matrix[0, 0], 1.0)
    np.testing.assert_allclose(matrix[m, m + 1], 1.0)
    np.testing.assert_allclose(
        matrix[1, first_col:first_col + len(first_row)], first_row
    )
    np.testing.assert_allclose(
        matrix[
            interior_row,
            interior_col + np.arange(len(interior_values)),
        ],
        interior_values,
    )
    np.testing.assert_allclose(operator @ np.ones(m + 2), np.ones(m + 1))


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
def test_ctof_periodic_1d_matches_pinned_mole_rows(k, m, expected_entries):
    """Check periodic CtoF stencil coefficients from pinned MOLE C++."""
    operator = MimeticInterpolCtoF(m, boundary="periodic", k=k)
    matrix = operator.matrix
    assert matrix.shape == (m, m)
    for row, column, expected in expected_entries:
        assert matrix[row, column] == pytest.approx(expected)
    np.testing.assert_allclose(operator @ np.ones(m), np.ones(m))


def test_ctof_multidimensional_shapes_and_boundary_modes():
    """Exercise CtoF block assembly for each supported dimension."""
    nonperiodic_2d = MimeticInterpolCtoF((9, 10), k=2)
    periodic_2d = MimeticInterpolCtoF((9, 10), boundary="periodic", k=2)
    mixed_3d = MimeticInterpolCtoF(
        (9, 10, 11), boundary=("periodic", "nonperiodic", "periodic"), k=2
    )

    assert nonperiodic_2d.matrix.shape == (
        10 * 10 + 11 * 9,
        2 * 11 * 12,
    )
    assert periodic_2d.matrix.shape == (2 * 9 * 10, 2 * 9 * 10)
    assert mixed_3d.matrix.shape == (3069, 3564)
    np.testing.assert_allclose(
        nonperiodic_2d.matrix @ np.ones(nonperiodic_2d.matrix.get_shape()[1]),
        np.ones(nonperiodic_2d.matrix.get_shape()[0]),
    )
    np.testing.assert_allclose(
        mixed_3d.matrix @ np.ones(mixed_3d.matrix.get_shape()[1]),
        np.ones(mixed_3d.matrix.get_shape()[0]),
    )


def test_ctof_factory_and_validation():
    """Check public construction and MOLE's order/grid constraints."""
    previous_backend = get_backend()
    use_backend("python")
    try:
        assert inspect.signature(create_interpol_c_to_f) == inspect.signature(
            MimeticInterpolCtoF
        )
        assert create_interpol_c_to_f(9, k=2).matrix.shape == (10, 11)
        with pytest.raises(ValueError, match="m > 2\\*k"):
            MimeticInterpolCtoF(4, k=2)
        with pytest.raises(ValueError, match="supports k"):
            MimeticInterpolCtoF(9, k=3)
        with pytest.raises(ValueError, match="dc and nc"):
            MimeticInterpolCtoF(9, dc=[1, 1])
    finally:
        use_backend(previous_backend)


@pytest.mark.parametrize(
    ("k", "m", "first_col", "expected_row"),
    [
        (2, 5, 1, (0.5, 0.5)),
        (4, 9, 0, (-16 / 112, 70 / 112, 70 / 112, -14 / 112, 2 / 112)),
        (6, 13, 0, (-768 / 8448, 4158 / 8448, 6930 / 8448, -2772 / 8448,
                    1188 / 8448, -330 / 8448, 42 / 8448)),
        (8, 17, 0, (-1 / 15, 429 / 1024, 1001 / 1024, -3003 / 5120,
                    429 / 1024, -715 / 3072, 91 / 1024, -21 / 1024, 11 / 5120)),
    ],
)
def test_cton_nonperiodic_1d_matches_pinned_mole_rows(k, m, first_col, expected_row):
    """Check center-to-node boundary stencils from pinned MOLE C++."""
    operator = MimeticInterpolCtoN(m, k=k)
    matrix = operator.matrix.toarray()
    assert matrix.shape == (m + 1, m + 2)
    np.testing.assert_allclose(matrix[0, 0], 1.0)
    np.testing.assert_allclose(
        matrix[1, first_col:first_col + len(expected_row)], expected_row
    )
    np.testing.assert_allclose(matrix[m, m + 1], 1.0)
    np.testing.assert_allclose(operator @ np.ones(m + 2), np.ones(m + 1))


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
def test_cton_periodic_1d_matches_pinned_mole_rows(k, m, expected_entries):
    """Check periodic center-to-node coefficients from pinned MOLE C++."""
    operator = MimeticInterpolCtoN(m, boundary="periodic", k=k)
    assert operator.matrix.shape == (m, m)
    for row, column, expected in expected_entries:
        assert operator.matrix[row, column] == pytest.approx(expected)
    np.testing.assert_allclose(operator @ np.ones(m), np.ones(m))


def test_cton_multidimensional_shapes_and_boundary_modes():
    """Check tensor-product CtoN shapes and mixed-axis constant preservation."""
    nonperiodic_2d = MimeticInterpolCtoN((9, 10), k=2)
    periodic_2d = MimeticInterpolCtoN((9, 10), boundary="periodic", k=2)
    mixed_3d = MimeticInterpolCtoN(
        (9, 10, 11), boundary=("periodic", "nonperiodic", "periodic"), k=2
    )

    assert nonperiodic_2d.matrix.shape == (110, 132)
    assert periodic_2d.matrix.shape == (90, 90)
    assert mixed_3d.matrix.shape == (1089, 1188)
    for operator in (nonperiodic_2d, periodic_2d, mixed_3d):
        np.testing.assert_allclose(
            operator @ np.ones(operator.matrix.get_shape()[1]),
            np.ones(operator.matrix.get_shape()[0]),
        )


def test_cton_factory_and_validation():
    """Check public construction and MOLE's CtoN constraints."""
    previous_backend = get_backend()
    use_backend("python")
    try:
        assert inspect.signature(create_interpol_c_to_n) == inspect.signature(
            MimeticInterpolCtoN
        )
        assert create_interpol_c_to_n(9, k=2).matrix.shape == (10, 11)
        with pytest.raises(ValueError, match="m > 2\\*k"):
            MimeticInterpolCtoN(4, k=2)
        with pytest.raises(ValueError, match="supports k"):
            MimeticInterpolCtoN(9, k=3)
        with pytest.raises(ValueError, match="dc and nc"):
            MimeticInterpolCtoN(9, dc=[1, 1])
    finally:
        use_backend(previous_backend)


@pytest.mark.parametrize(
    ("k", "m", "first_row", "interior_row", "interior_values"),
    [
        (2, 5, (0.5,), 1, (0.5, 0.5)),
        (4, 9, (35 / 128, 140 / 128, -70 / 128, 28 / 128, -5 / 128),
         2, (-8 / 128, 72 / 128, 72 / 128, -8 / 128)),
        (6, 13, (231 / 1024, 1386 / 1024, -1155 / 1024, 924 / 1024,
                 -495 / 1024, 154 / 1024, -21 / 1024),
         3, (12 / 1024, -100 / 1024, 600 / 1024, 600 / 1024,
             -100 / 1024, 12 / 1024)),
        (8, 17, (6435 / 32768, 6435 / 4096, -15015 / 8192, 9009 / 4096,
                 -32175 / 16384, 5005 / 4096, -4095 / 8192, 495 / 4096,
                 -429 / 32768),
         4, (-5 / 2048, 49 / 2048, -245 / 2048, 1225 / 2048,
             1225 / 2048, -245 / 2048, 49 / 2048, -5 / 2048)),
    ],
)
def test_ftoc_nonperiodic_1d_matches_pinned_mole_rows(
    k, m, first_row, interior_row, interior_values
):
    """Check face-to-center stencils from pinned MOLE C++."""
    operator = MimeticInterpolFtoC(m, k=k)
    matrix = operator.matrix.toarray()
    assert matrix.shape == (m + 2, m + 1)
    np.testing.assert_allclose(matrix[0, 0], 1.0)
    np.testing.assert_allclose(matrix[1, :len(first_row)], first_row)
    np.testing.assert_allclose(
        matrix[interior_row, interior_row - (len(interior_values) // 2) +
               np.arange(len(interior_values))],
        interior_values,
    )
    np.testing.assert_allclose(matrix[m + 1, m], 1.0)
    np.testing.assert_allclose(operator @ np.ones(m + 1), np.ones(m + 2))


@pytest.mark.parametrize(
    ("k", "m", "expected_entries"),
    [
        (2, 5, ((0, 0, 0.5), (0, 1, 0.5))),
        (4, 9, ((0, 0, 72 / 128), (0, 1, 72 / 128),
                (0, 2, -8 / 128), (0, 8, -8 / 128))),
        (6, 13, ((0, 0, 600 / 1024), (0, 1, 600 / 1024),
                 (0, 2, -100 / 1024), (0, 3, 12 / 1024),
                 (0, 11, 12 / 1024), (0, 12, -100 / 1024))),
        (8, 17, ((0, 0, 1225 / 2048), (0, 1, 1225 / 2048),
                 (0, 2, -245 / 2048), (0, 3, 49 / 2048),
                 (0, 4, -5 / 2048), (0, 14, -5 / 2048),
                 (0, 15, 49 / 2048), (0, 16, -245 / 2048))),
    ],
)
def test_ftoc_periodic_1d_matches_pinned_mole_rows(k, m, expected_entries):
    """Check periodic face-to-center coefficients from pinned MOLE C++."""
    operator = MimeticInterpolFtoC(m, boundary="periodic", k=k)
    assert operator.matrix.shape == (m, m)
    for row, column, expected in expected_entries:
        assert operator.matrix[row, column] == pytest.approx(expected)
    np.testing.assert_allclose(operator @ np.ones(m), np.ones(m))


def test_ftoc_multidimensional_shapes_and_boundary_modes():
    """Check multidimensional FtoC blocks and preservation of constant fields."""
    nonperiodic_2d = MimeticInterpolFtoC((9, 10), k=2)
    periodic_2d = MimeticInterpolFtoC((9, 10), boundary="periodic", k=2)
    mixed_3d = MimeticInterpolFtoC(
        (9, 10, 11), boundary=("periodic", "nonperiodic", "periodic"), k=2
    )
    assert nonperiodic_2d.matrix.shape == (264, 199)
    assert periodic_2d.matrix.shape == (180, 180)
    assert mixed_3d.matrix.shape == (3564, 3069)
    expected_nonzero_rows = (218, 180, 3168)
    for operator, nonzero_rows in zip(
        (nonperiodic_2d, periodic_2d, mixed_3d), expected_nonzero_rows
    ):
        row_sums = operator.matrix @ np.ones(operator.matrix.get_shape()[1])
        expected_sums = (
            (0.0, 1.0)
            if nonzero_rows < operator.matrix.get_shape()[0]
            else (1.0,)
        )
        np.testing.assert_allclose(np.unique(row_sums), expected_sums)
        assert np.count_nonzero(row_sums) == nonzero_rows


def test_ftoc_factory_and_validation():
    """Check the public FtoC API and shared MOLE input constraints."""
    previous_backend = get_backend()
    use_backend("python")
    try:
        assert inspect.signature(create_interpol_f_to_c) == inspect.signature(
            MimeticInterpolFtoC
        )
        assert create_interpol_f_to_c(9, k=2).matrix.shape == (11, 10)
        with pytest.raises(ValueError, match="m > 2\\*k"):
            MimeticInterpolFtoC(4, k=2)
        with pytest.raises(ValueError, match="supports k"):
            MimeticInterpolFtoC(9, k=3)
    finally:
        use_backend(previous_backend)


@pytest.mark.parametrize(
    ("k", "m", "first_col", "expected_row"),
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
def test_ntoc_nonperiodic_1d_matches_pinned_mole_rows(k, m, first_col, expected_row):
    """Check node-to-center boundary stencils from pinned MOLE C++."""
    operator = MimeticInterpolNtoC(m, k=k)
    matrix = operator.matrix.toarray()
    assert matrix.shape == (m + 2, m + 1)
    np.testing.assert_allclose(matrix[0, 0], 1.0)
    np.testing.assert_allclose(
        matrix[1, first_col:first_col + len(expected_row)], expected_row
    )
    np.testing.assert_allclose(matrix[m + 1, m], 1.0)
    np.testing.assert_allclose(operator @ np.ones(m + 1), np.ones(m + 2))


def test_ntoc_periodic_multidimensional_shapes_and_factory():
    """Check periodic NtoC coefficients and tensor-product assembly."""
    periodic = MimeticInterpolNtoC(9, boundary="periodic", k=4)
    assert periodic.matrix.shape == (9, 9)
    assert periodic.matrix[0, 0] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 1] == pytest.approx(72 / 128)
    assert periodic.matrix[0, 2] == pytest.approx(-8 / 128)
    assert periodic.matrix[0, 8] == pytest.approx(-8 / 128)

    nonperiodic_2d = MimeticInterpolNtoC((9, 10), k=2)
    mixed_3d = MimeticInterpolNtoC(
        (9, 10, 11), boundary=("periodic", "nonperiodic", "periodic"), k=2
    )
    assert nonperiodic_2d.matrix.shape == (132, 110)
    assert mixed_3d.matrix.shape == (1188, 1089)
    for operator in (periodic, nonperiodic_2d, mixed_3d):
        np.testing.assert_allclose(
            operator @ np.ones(operator.matrix.get_shape()[1]),
            np.ones(operator.matrix.get_shape()[0]),
        )

    previous_backend = get_backend()
    use_backend("python")
    try:
        assert inspect.signature(create_interpol_n_to_c) == inspect.signature(
            MimeticInterpolNtoC
        )
        assert create_interpol_n_to_c(9, k=2).matrix.shape == (11, 10)
        with pytest.raises(ValueError, match="m > 2\\*k"):
            MimeticInterpolNtoC(4, k=2)
    finally:
        use_backend(previous_backend)


def test_public_factories_cpp_matrix_parity():
    """Test: Native matrix parity for C++ backend."""
    _load_native_backend()
    previous_backend = get_backend()
    try:
        use_backend("cpp")
        if get_backend() != "cpp":
            pytest.skip("C++ backend could not be selected")

        dimensions, spacings = (7, 8), (0.2, 0.15)
        mixed_args = {
            "left": "Dirichlet", "coeffs_left": [2.0],
            "right": "Robin", "coeffs_right": [3.0, 0.5],
        }
        pairs = ((
            create_gradient(dimensions, spacings, k=2),
            MimeticGradient(dimensions, spacings, k=2)
        ), (
            create_divergence(dimensions, spacings, k=2),
            MimeticDivergence(dimensions, spacings, k=2)
        ), (
            create_laplacian(dimensions, spacings, k=2),
            MimeticLaplacian(dimensions, spacings, k=2)
        ), (
            create_interpol(dimensions, spacings, c=(0.3, 0.4)),
            MimeticInterpol(dimensions, spacings, c=(0.3, 0.4))
        ), (
            create_interpol_c_to_f(9, k=2),
            MimeticInterpolCtoF(9, k=2)
        ), (
            create_interpol_c_to_n(9, k=2),
            MimeticInterpolCtoN(9, k=2)
        ), (
            create_interpol_f_to_c(9, k=2),
            MimeticInterpolFtoC(9, k=2)
        ), (
            create_interpol_n_to_c(9, k=2),
            MimeticInterpolNtoC(9, k=2)
        ), (
            create_robin_bc(9, 0.125, k=2, a=2.0, b=0.5),
            PureMimeticRobinBC(9, 0.125, k=2, a=2.0, b=0.5)
        ), (
            create_mixed_bc(9, 0.125, k=2, **mixed_args),
            PureMimeticMixedBC(9, 0.125, k=2, **mixed_args)
        ))
        for backend_operator, pure_operator in pairs:
            np.testing.assert_allclose(
                backend_operator.matrix.toarray(),
                pure_operator.matrix.toarray(),
                rtol=0, atol=1e-12
            )

        periodic = create_gradient(9, 0.1, boundary="periodic", k=2)
        pure_periodic = MimeticGradient(9, 0.1, boundary="periodic", k=2)
        np.testing.assert_allclose(periodic.matrix.toarray(), pure_periodic.matrix.toarray())
    finally:
        use_backend(previous_backend)


# ============================================================================
# ACCURACY ORDER TESTS
# ============================================================================

@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_gradient_accuracy_polynomial(k):
    """Test that gradient operator has correct order of accuracy on polynomials.

    For a k-th order accurate operator, the error should scale as O(h^k).
    """
    # Test on multiple grid sizes
    errors = []
    spacings = []

    for n in [32, 64, 128]:
        h = 1.0 / (n + 1)
        spacings.append(h)

        x = np.arange(n + 2) * h
        # Test function: f(x) = x^(k+1), f'(x) = (k+1)*x^k
        f = x**(k + 1)
        gradient_locations = (np.arange(n + 1) + 0.5) * h
        expected = (k + 1) * gradient_locations**k

        grad = MimeticGradient(n, h, k=k)
        computed = grad @ f

        # Compare interior points (excluding boundaries which have higher error)
        interior = slice(k // 2, n + 1 - k // 2)
        error = np.max(np.abs(computed[interior] - expected[interior]))
        errors.append(error)

    # Check convergence rate: error should decrease by factor of 2^k for spacing halving
    convergence_ratio = errors[0] / errors[1]
    expected_ratio = 2**k

    # Allow 50% tolerance in convergence rate
    assert convergence_ratio > expected_ratio * 0.5, \
        f"Convergence ratio {convergence_ratio:.2e} < expected {expected_ratio * 0.5:.2e}"


@pytest.mark.parametrize("k", [2, 4, 6])
def test_gradient_accuracy_trigonometry(k):
    """Test gradient accuracy on smooth trigonometric functions."""
    n = 128
    h = 2 * np.pi / (n + 1)
    x = np.arange(n + 2) * h
    f = np.sin(x)
    expected = np.cos((np.arange(n + 1) + 0.5) * h)

    grad = MimeticGradient(n, h, k=k)
    computed = grad @ f

    # Interior error (excluding boundaries)
    interior = slice(k // 2, n + 1 - k // 2)
    rel_error = np.linalg.norm(computed[interior] - expected[interior]) / \
                np.linalg.norm(expected[interior])

    # For smooth functions, error should be very small
    assert rel_error < 0.01, f"Relative error {rel_error:.4e} too large"


# ============================================================================
# MATRIX PROPERTY TESTS
# ============================================================================

def test_matrix_format():
    """Test that matrix is in scipy sparse CSC format."""
    grad = MimeticGradient(20, 0.1)
    mat = grad.matrix
    assert isinstance(mat, sparse.csc_matrix), "Matrix should be CSC sparse format"

def test_matrix_shape():
    """Test the staggered 1D matrix shapes used by MOLE."""
    n = 20
    h = 0.1

    grad = MimeticGradient(n, h)
    div = MimeticDivergence(n, h)
    lap = MimeticLaplacian(n, h)

    assert grad.matrix.shape == (n + 1, n + 2)
    assert div.matrix.shape == (n + 2, n + 1)
    assert lap.matrix.shape == (n + 2, n + 2)

def test_sparsity_pattern():
    """Test that matrices have expected sparsity (banded structure for 1D)."""
    grad = MimeticGradient(50, 0.1, k=2)
    mat = grad.matrix

    # 1D operators should be banded (tridiagonal for k=2)
    # Check that most entries are on the main diagonal and neighboring diagonals
    row_count, column_count = mat.toarray().shape
    data_density = mat.nnz / (row_count * column_count)
    assert data_density < 0.1, f"Matrix density {data_density:.2e} suggests not banded"

def test_matrix_consistency_with_operator():
    """Test that matrix and @ operator produce identical results."""
    n = 30
    h = 0.1
    grad = MimeticGradient(n, h)

    x = np.random.default_rng(0).random(n + 2)

    # Two ways to apply operator
    result_matmul = grad @ x
    result_matrix = grad.matrix @ x

    np.testing.assert_allclose(result_matmul, result_matrix, rtol=1e-14)

def test_operator_nnz():
    """Test non-zero count."""
    grad = MimeticGradient(20, 0.1)
    mat = grad.matrix

    # For 1D 2nd order operator, should have roughly 3*n non-zeros (banded)
    # Allow 50% variation
    expected_nnz = 2 * 20 + 6
    actual_nnz = mat.nnz

    assert actual_nnz > 0, "Should have non-zero elements"
    assert actual_nnz < expected_nnz * 2, "Sparsity should be maintained"


# ============================================================================
# SPARSE SOLVER INTEGRATION TESTS
# ============================================================================

def test_sparse_solver_laplacian():
    """Test that the MOLE Laplacian remains sparse and applicable."""
    n = 30
    h = 1.0 / (n + 1)
    lap = MimeticLaplacian(n, h)
    result = lap @ np.ones(n + 2)
    assert result.shape == (n + 2,)
    assert np.isfinite(result).all()

def test_eigenvalue_decomposition():
    """Test the k=2 MOLE Laplacian has non-positive spectrum."""
    n = 16
    h = 1.0 / (n + 1)
    lap = MimeticLaplacian(n, h)
    eigenvalues = np.linalg.eigvals(lap.matrix.toarray())
    assert np.max(eigenvalues.real) < 1e-10
    assert np.min(eigenvalues.real) < 0
    assert np.max(np.abs(eigenvalues.imag)) < 1e-10


# ============================================================================
# TIME INTEGRATION TESTS (Heat Equation)
# ============================================================================

def _solve_heat_equation_1d(backend):
    previous_backend = get_backend()
    try:
        if backend == "cpp":
            _load_native_backend()
        use_backend(backend)
        if get_backend() != backend:
            pytest.skip(f"{backend} backend could not be selected")

        diffusivity = 0.1
        t_final = 0.1
        n = 50
        h = 1.0 / (n + 1)

        lap_mat = create_laplacian(n, h).matrix
        x = np.linspace(0, 1, n + 2)
        u0 = np.sin(np.pi * x)

        def heat_ode(u, _):
            return diffusivity * (lap_mat @ u)

        t_eval = np.linspace(0, t_final, 20)
        solution = odeint(heat_ode, u0, t_eval)
        u_final_numerical = solution[-1, :]
        u_final_analytical = np.sin(np.pi * x) * np.exp(-np.pi**2 * diffusivity * t_final)

        interior = slice(2, -2)
        rel_error = np.linalg.norm(
            u_final_numerical[interior] - u_final_analytical[interior]
        ) / np.linalg.norm(u_final_analytical[interior])
        assert rel_error < 0.05, f"Heat equation error for {backend}: {rel_error:.4e}"
        return u_final_numerical
    finally:
        use_backend(previous_backend)


def test_heat_equation_1d():
    """Check the Python backend solves the 1D heat equation accurately."""
    _solve_heat_equation_1d("python")


def test_heat_equation_1d_backend_parity():
    """Check C++ and Python heat-equation solutions agree end to end."""
    python_solution = _solve_heat_equation_1d("python")
    cpp_solution = _solve_heat_equation_1d("cpp")
    np.testing.assert_allclose(cpp_solution, python_solution, rtol=0, atol=1e-10)

def test_wave_equation_1d():
    """Test solving 1D wave equation u_tt = c^2*u_xx using method of lines.

    Analytical solution: u(x,t) = sin(pi*x) * cos(pi*c*t)
    Initial condition: u(x,0) = sin(pi*x), u_t(x,0) = 0
    Boundary conditions: u(0,t) = u(1,t) = 0
    """
    # Parameters
    c = 1.0  # wave speed
    t_final = 0.25
    n = 60
    h = 1.0 / (n + 1)

    # Create Laplacian operator
    lap = MimeticLaplacian(n, h)
    lap_mat = c**2 * lap.matrix

    # Initial conditions
    x = np.linspace(0, 1, n + 2)
    u0 = np.sin(np.pi * x)
    v0 = np.zeros(n + 2)  # zero initial velocity

    # Combined state vector: [u, v]
    state0 = np.concatenate([u0, v0])

    # ODE system: d/dt[u, v] = [v, c^2*L@u]
    def wave_ode(state, _):
        u, v = state[:n + 2], state[n + 2:]
        dudt = v
        dvdt = lap_mat @ u
        return np.concatenate([dudt, dvdt])

    # Solve ODE
    t_eval = np.linspace(0, t_final, 30)
    solution = odeint(wave_ode, state0, t_eval)

    u_final_numerical = solution[-1, :n + 2]
    u_final_analytical = np.sin(np.pi * x) * np.cos(np.pi * c * t_final)

    # Relative error (excluding boundaries)
    interior = slice(2, -2)
    rel_error = np.linalg.norm(u_final_numerical[interior] - u_final_analytical[interior]) / \
                np.linalg.norm(u_final_analytical[interior])

    assert rel_error < 0.08, f"Wave equation solution error {rel_error:.4e} too large"


# ============================================================================
# NATIVE BOUNDARY AND 3D APPLICATION TESTS
# ============================================================================

@pytest.mark.parametrize(("a", "b"), [(1.0, 0.0), (0.0, 1.0)])
def test_native_robin_bc_limits_match_matrix(a, b):
    """Check Robin Dirichlet/Neumann limits and native matrix application."""
    native = _load_native_backend()
    n, h = 30, 1.0 / 29
    robin = native.MimeticRobinBC(n, h, k=2, a=a, b=b)
    pure_robin = PureMimeticRobinBC(n, h, k=2, a=a, b=b)
    values = np.linspace(0.0, 1.0, robin.matrix.shape[1])

    np.testing.assert_allclose(robin.matrix.toarray(), pure_robin.matrix.toarray(), atol=1e-12)
    np.testing.assert_allclose(robin @ values, robin.matrix @ values, atol=1e-12)


def test_native_mixed_bc_1d_application_matches_matrix():
    """Check 1D Dirichlet/Neumann mixed BC application against its matrix."""
    native = _load_native_backend()
    n, h = 30, 1.0 / 29
    boundary_conditions = {
        "left": "Dirichlet", "coeffs_left": [1.0],
        "right": "Neumann", "coeffs_right": [0.0],
    }
    mixed = native.MimeticMixedBC(n, h, k=2, **boundary_conditions)
    pure_mixed = PureMimeticMixedBC(n, h, k=2, **boundary_conditions)
    values = np.linspace(0.0, 1.0, mixed.matrix.shape[1])

    np.testing.assert_allclose(mixed.matrix.toarray(), pure_mixed.matrix.toarray(), atol=1e-12)
    np.testing.assert_allclose(mixed @ values, mixed.matrix @ values, atol=1e-12)


def test_native_mixed_bc_2d_application_matches_matrix():
    """Check 2D mixed BC application and its native/pure matrix parity."""
    native = _load_native_backend()
    dimensions, spacings = (7, 8), (0.2, 0.15)
    boundary_conditions = {
        "left": "Dirichlet", "coeffs_left": [1.0],
        "right": "Neumann", "coeffs_right": [0.0],
        "bottom": "Robin", "coeffs_bottom": [1.0, 0.5],
        "top": "Dirichlet", "coeffs_top": [1.0],
    }
    mixed = native.MimeticMixedBC(dimensions, spacings, k=2, **boundary_conditions)
    pure_mixed = PureMimeticMixedBC(dimensions, spacings, k=2, **boundary_conditions)
    values = np.linspace(-1.0, 1.0, mixed.matrix.shape[1])

    np.testing.assert_allclose(mixed.matrix.toarray(), pure_mixed.matrix.toarray(), atol=1e-12)
    np.testing.assert_allclose(mixed @ values, mixed.matrix @ values, atol=1e-12)


@pytest.mark.parametrize("operator_name", ["robin", "mixed"])
def test_native_boundary_metadata_matches_python(operator_name):
    """Check multidimensional boundary operators retain grid metadata."""
    native = _load_native_backend()
    dimensions, spacings = (7, 8), (0.2, 0.15)
    boundary_conditions = {
        "left": "Dirichlet", "coeffs_left": [1.0],
        "right": "Neumann", "coeffs_right": [0.0],
        "bottom": "Robin", "coeffs_bottom": [1.0, 0.5],
        "top": "Dirichlet", "coeffs_top": [1.0],
    }
    if operator_name == "robin":
        native_operator = native.MimeticRobinBC(dimensions, spacings, k=2)
        pure_operator = PureMimeticRobinBC(dimensions, spacings, k=2)
    else:
        native_operator = native.MimeticMixedBC(dimensions, spacings, k=2, **boundary_conditions)
        pure_operator = PureMimeticMixedBC(dimensions, spacings, k=2, **boundary_conditions)

    assert native_operator.n == pure_operator.n == dimensions
    assert native_operator.h == pure_operator.h == spacings


@pytest.mark.parametrize("operator_name", ["robin", "mixed"])
def test_native_boundary_matrix_rhs_matches_python(operator_name):
    """Check native boundary operators apply to multiple right-hand sides."""
    native = _load_native_backend()
    n, h = 9, 0.1
    boundary_conditions = {
        "left": "Dirichlet", "coeffs_left": [1.0],
        "right": "Neumann", "coeffs_right": [0.0],
    }
    if operator_name == "robin":
        native_operator = native.MimeticRobinBC(n, h, k=2, a=1.0, b=0.5)
        pure_operator = PureMimeticRobinBC(n, h, k=2, a=1.0, b=0.5)
    else:
        native_operator = native.MimeticMixedBC(n, h, k=2, **boundary_conditions)
        pure_operator = PureMimeticMixedBC(n, h, k=2, **boundary_conditions)
    values = np.linspace(-1.0, 1.0, native_operator.matrix.shape[1] * 2).reshape(-1, 2)

    result = native_operator @ values
    expected = pure_operator @ values
    assert result.shape == expected.shape == (native_operator.matrix.shape[0], 2)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_native_operator_multiple_rhs_matches_matrix():
    """Check every native wrapper handles multiple right-hand sides correctly."""
    native = _load_native_backend()
    n, h = 9, 0.1
    mixed_args = {
        "left": "Dirichlet", "coeffs_left": [1.0],
        "right": "Neumann", "coeffs_right": [0.0],
    }
    operators = (
        native.MimeticGradient(n, h),
        native.MimeticDivergence(n, h),
        native.MimeticLaplacian(n, h),
        native.MimeticInterpol(n, h),
        native.MimeticRobinBC(n, h),
        native.MimeticMixedBC(n, h, **mixed_args),
    )

    for operator in operators:
        matrix = operator.matrix
        values = np.arange(matrix.shape[1] * 2, dtype=float).reshape(-1, 2)
        result = operator @ values
        assert result.shape == (matrix.shape[0], 2)
        np.testing.assert_allclose(result, matrix @ values, atol=1e-12)


def test_native_3d_operator_application_matches_matrices():
    """Check 3D native operator applications against matrices and Python parity."""
    native = _load_native_backend()
    dimensions, spacings = (7, 8, 9), (0.2, 0.15, 0.1)
    native_operators = (
        native.MimeticGradient(dimensions, spacings, k=2),
        native.MimeticDivergence(dimensions, spacings, k=2),
        native.MimeticLaplacian(dimensions, spacings, k=2),
    )
    pure_operators = (
        MimeticGradient(dimensions, spacings, k=2),
        MimeticDivergence(dimensions, spacings, k=2),
        MimeticLaplacian(dimensions, spacings, k=2),
    )

    for native_operator, pure_operator in zip(native_operators, pure_operators):
        values = np.linspace(-1.0, 1.0, native_operator.matrix.shape[1])
        result = native_operator @ values
        assert result.shape == (native_operator.matrix.shape[0],)
        np.testing.assert_allclose(
            native_operator.matrix.toarray(),
            pure_operator.matrix.toarray(),
            atol=1e-12
        )

        np.testing.assert_allclose(result,
            native_operator.matrix @ values,
            atol=1e-12
        )
