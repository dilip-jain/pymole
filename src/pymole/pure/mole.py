"""MOLE-shaped constructor interfaces for the pure-Python backend."""

from numbers import Real
from typing import Any, Dict, List, Sequence, Tuple, overload

import numpy as np

from .operators import (
    MimeticDivergence,
    MimeticGradient,
    MimeticInterpol,
    MimeticInterpolCtoF,
    MimeticInterpolCtoN,
    MimeticInterpolFtoC,
    MimeticInterpolNtoC,
    MimeticLaplacian,
    MimeticMixedBC,
    MimeticRobinBC,
)


def _validate_order(k: int) -> int:
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer")
    if not 0 <= k <= 65535:
        raise ValueError("k must fit in an unsigned 16-bit integer")
    return int(k)


def _grid_dimensions(values: Sequence[Any]) -> Tuple[int, ...]:
    dimensions = []
    for value in values:
        if not isinstance(value, (int, np.integer)) or isinstance(
            value, (bool, np.bool_)
        ):
            raise TypeError("Grid dimensions must be integers")
        if not 0 < value <= 4294967295:
            raise ValueError(
                "Grid dimensions must fit in a positive unsigned 32-bit integer"
            )
        dimensions.append(int(value))
    return tuple(dimensions)


def _grid_spacings(values: Sequence[Any]) -> Tuple[float, ...]:
    spacings = []
    for value in values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("Grid spacings must be real numbers")
        spacings.append(float(value))
    return tuple(spacings)


def _bind_overload(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    overloads: Sequence[Tuple[str, ...]],
) -> Tuple[Any, ...]:
    for names in overloads:
        if len(args) > len(names) or not set(kwargs).issubset(names):
            continue
        values = dict(zip(names, args))
        if any(name in values for name in kwargs):
            continue
        values.update(kwargs)
        if len(values) == len(names):
            return tuple(values[name] for name in names)
    raise TypeError("Arguments do not match a MOLE constructor overload")


def _parse_derivative(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    bc_overloads = (
        ("m", "dx", "dc", "nc"),
        ("m", "n", "dx", "dy", "dc", "nc"),
        ("m", "n", "o", "dx", "dy", "dz", "dc", "nc"),
    )
    grid_overloads = (
        ("m", "dx"),
        ("m", "n", "dx", "dy"),
        ("m", "n", "o", "dx", "dy", "dz"),
    )
    has_vector_args = len(args) in (4, 6, 8) and all(
        isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes))
        for value in args[-2:]
    )
    bc_keywords = "dc" in kwargs or "nc" in kwargs
    names = bc_overloads if has_vector_args or bc_keywords else grid_overloads
    values = _bind_overload(args, kwargs, names)
    coefficient_count = 2 if names == bc_overloads else 0
    dimension_count = (len(values) - coefficient_count) // 2
    dimensions = _grid_dimensions(values[:dimension_count])
    spacings = _grid_spacings(values[dimension_count:2 * dimension_count])
    if coefficient_count:
        dc, nc = values[-2:]
        if dc is None or nc is None:
            raise TypeError("dc and nc must be coefficient sequences")
        return dimensions, spacings, dc, nc
    return dimensions, spacings, None, None


class Gradient(MimeticGradient):
    """Pure-Python implementation of MOLE's ``Gradient`` constructors."""

    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dx: float, dy: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, dx: float, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        dx: float,
        dy: float,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, spacings, dc, nc = _parse_derivative(args, kwargs)
        if dc is None:
            super().__init__(dimensions, spacings, k=k)
        else:
            super().__init__(dimensions, spacings, k=k, dc=dc, nc=nc)

    def getP(self) -> List[float]:  # type: ignore[reportIncompatibleMethodOverride]
        return super().getP().tolist()


class Divergence(MimeticDivergence):
    """Pure-Python implementation of MOLE's ``Divergence`` constructors."""

    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dx: float, dy: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, dx: float, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        dx: float,
        dy: float,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, spacings, dc, nc = _parse_derivative(args, kwargs)
        if dc is None:
            super().__init__(dimensions, spacings, k=k)
        else:
            super().__init__(dimensions, spacings, k=k, dc=dc, nc=nc)

    def getQ(self) -> List[float]:  # type: ignore[reportIncompatibleMethodOverride]
        return super().getQ().tolist()


class Laplacian(MimeticLaplacian):
    """Pure-Python implementation of MOLE's ``Laplacian`` constructors."""

    @overload
    def __init__(self, k: int, m: int, dx: float) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dx: float, dy: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        overloads = (
            ("m", "dx"),
            ("m", "n", "dx", "dy"),
            ("m", "n", "o", "dx", "dy", "dz"),
        )
        values = _bind_overload(tuple(args), kwargs, overloads)
        dimension_count = len(values) // 2
        dimensions = _grid_dimensions(values[:dimension_count])
        spacings = _grid_spacings(values[dimension_count:])
        super().__init__(dimensions, spacings, k=k)


class Interpol(MimeticInterpol):
    """Pure-Python implementation of MOLE's ``Interpol`` constructors."""

    @overload
    def __init__(self, m: int, c: float) -> None: ...
    @overload
    def __init__(self, m: int, n: int, c1: float, c2: float) -> None: ...
    @overload
    def __init__(
        self, m: int, n: int, o: int, c1: float, c2: float, c3: float
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        overloads = (
            ("m", "c"),
            ("m", "n", "c1", "c2"),
            ("m", "n", "o", "c1", "c2", "c3"),
        )
        values = _bind_overload(tuple(args), kwargs, overloads)
        dimension_count = len(values) // 2
        dimensions = _grid_dimensions(values[:dimension_count])
        weights = _grid_spacings(values[dimension_count:])
        super().__init__(dimensions, (1.0,) * len(dimensions), c=weights)


class InterpolCtoF(MimeticInterpolCtoF):
    """Pure-Python implementation of MOLE's ``InterpolCtoF`` constructors."""

    @overload
    def __init__(
        self, k: int, m: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, dc, nc = _parse_directional_arguments(args, kwargs)
        super().__init__(dimensions, k=k, dc=dc, nc=nc)


class InterpolCtoN(MimeticInterpolCtoN):
    """Pure-Python implementation of MOLE's ``InterpolCtoN`` constructors."""

    @overload
    def __init__(
        self, k: int, m: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, dc, nc = _parse_directional_arguments(args, kwargs)
        super().__init__(dimensions, k=k, dc=dc, nc=nc)


class InterpolFtoC(MimeticInterpolFtoC):
    """Pure-Python implementation of MOLE's ``InterpolFtoC`` constructors."""

    @overload
    def __init__(
        self, k: int, m: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, dc, nc = _parse_directional_arguments(args, kwargs)
        super().__init__(dimensions, k=k, dc=dc, nc=nc)


class InterpolNtoC(MimeticInterpolNtoC):
    """Pure-Python implementation of MOLE's ``InterpolNtoC`` constructors."""

    @overload
    def __init__(
        self, k: int, m: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, n: int, dc: Sequence[int], nc: Sequence[int]
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        n: int,
        o: int,
        dc: Sequence[int],
        nc: Sequence[int],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        dimensions, dc, nc = _parse_directional_arguments(args, kwargs)
        super().__init__(dimensions, k=k, dc=dc, nc=nc)


def _parse_directional_arguments(args, kwargs):
    overloads = (
        ("m", "dc", "nc"),
        ("m", "n", "dc", "nc"),
        ("m", "n", "o", "dc", "nc"),
    )
    values = _bind_overload(tuple(args), kwargs, overloads)
    return _grid_dimensions(values[:-2]), values[-2], values[-1]


class RobinBC(MimeticRobinBC):
    """Pure-Python implementation of MOLE's ``RobinBC`` constructors."""

    @overload
    def __init__(
        self, k: int, m: int, dx: float, a: float, b: float
    ) -> None: ...
    @overload
    def __init__(
        self, k: int, m: int, dx: float, n: int, dy: float, a: float, b: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        o: int,
        dz: float,
        a: float,
        b: float,
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        overloads = (
            ("m", "dx", "a", "b"),
            ("m", "dx", "n", "dy", "a", "b"),
            ("m", "dx", "n", "dy", "o", "dz", "a", "b"),
        )
        values = _bind_overload(tuple(args), kwargs, overloads)
        dimension_count = len(values) // 2 - 1
        dimensions = _grid_dimensions(
            tuple(values[2 * axis] for axis in range(dimension_count))
        )
        spacings = _grid_spacings(
            tuple(values[2 * axis + 1] for axis in range(dimension_count))
        )
        a, b = values[-2:]
        if not isinstance(a, Real) or not isinstance(b, Real):
            raise TypeError("Robin coefficients must be real numbers")
        super().__init__(dimensions, spacings, k=k, a=float(a), b=float(b))


def _mixed_boundary_names(axis_count: int) -> Tuple[str, ...]:
    return (
        ("left", "right")
        + ("bottom", "top") * (axis_count >= 2)
        + ("front", "back") * (axis_count >= 3)
    )


class MixedBC(MimeticMixedBC):
    """Pure-Python implementation of MOLE's ``MixedBC`` constructors."""

    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
        bottom: str,
        coeffs_bottom: Sequence[float],
        top: str,
        coeffs_top: Sequence[float],
    ) -> None: ...
    @overload
    def __init__(
        self,
        k: int,
        m: int,
        dx: float,
        n: int,
        dy: float,
        o: int,
        dz: float,
        left: str,
        coeffs_left: Sequence[float],
        right: str,
        coeffs_right: Sequence[float],
        bottom: str,
        coeffs_bottom: Sequence[float],
        top: str,
        coeffs_top: Sequence[float],
        front: str,
        coeffs_front: Sequence[float],
        back: str,
        coeffs_back: Sequence[float],
    ) -> None: ...
    def __init__(self, k: int, *args: Any, **kwargs: Any) -> None:
        k = _validate_order(k)
        overloads = (
            ("m", "dx", "left", "coeffs_left", "right", "coeffs_right"),
            (
                "m", "dx", "n", "dy",
                "left", "coeffs_left", "right", "coeffs_right",
                "bottom", "coeffs_bottom", "top", "coeffs_top",
            ),
            (
                "m", "dx", "n", "dy", "o", "dz",
                "left", "coeffs_left", "right", "coeffs_right",
                "bottom", "coeffs_bottom", "top", "coeffs_top",
                "front", "coeffs_front", "back", "coeffs_back",
            ),
        )
        values = _bind_overload(tuple(args), kwargs, overloads)
        axis_count = len(values) // 6
        dimensions = _grid_dimensions(
            tuple(values[2 * axis] for axis in range(axis_count))
        )
        spacings = _grid_spacings(
            tuple(values[2 * axis + 1] for axis in range(axis_count))
        )
        boundary_values = values[2 * axis_count:]
        boundary_args = {}
        for index, name in enumerate(_mixed_boundary_names(axis_count)):
            boundary_args[name] = boundary_values[2 * index]
            boundary_args[f"coeffs_{name}"] = boundary_values[2 * index + 1]
        super().__init__(dimensions, spacings, k=k, **boundary_args)
