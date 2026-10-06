"""Tests for PyMOLE operators."""

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import odeint
from pymole import (
    create_gradient,
    create_divergence,
    create_laplacian,
    create_interpol,
    create_robin_bc,
    create_mixed_bc,
    get_backend,
    use_backend,
)
from pymole.pure.operators import (
    MimeticGradient,
    MimeticDivergence,
    MimeticLaplacian,
    MimeticInterpol,
    MimeticRobinBC as PureMimeticRobinBC,
    MimeticMixedBC as PureMimeticMixedBC,
)


def _load_native_backend():
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
        raise
    return native


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

def test_operator_validation():
    """Test operator input validation."""
    with pytest.raises(ValueError):
        MimeticGradient(0, 1.0)  # Invalid grid points
    with pytest.raises(ValueError):
        MimeticGradient(10, -1.0)  # Invalid grid spacing


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
    laplacian = create_laplacian(dimensions, spacings, boundary="periodic", k=2)
    interpolation = create_interpol(dimensions, spacings, c=(0.3, 0.4))
    robin = create_robin_bc(9, 0.125, k=2, a=2.0, b=0.5)
    mixed_args = {
        "left": "Dirichlet", "coeffs_left": [2.0],
        "right": "Robin", "coeffs_right": [3.0, 0.5],
    }
    mixed = create_mixed_bc(9, 0.125, k=2, **mixed_args)

    assert gradient.boundary == divergence.boundary == laplacian.boundary == "periodic"
    assert gradient.matrix.shape == (112, 56)
    assert divergence.matrix.shape == (56, 112)
    assert laplacian.matrix.shape == (56, 56)
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
