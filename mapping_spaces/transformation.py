"""Transformations between the three reference frames.

Derivation: solutions/transformation.md.

Objects 1, 2, 3 live on the planes z = -1, +1, 0.  Every map between frames
is a z-dependent shear that leaves y and z alone:

    T_kl(x, y, z) = (x + beta_kl(z) * y,  y,  z),
    beta_kl(z)    = sigma_l(z) - sigma_k(z),

where sigma_k is frame k's shear potential (quadratic interpolant of the
requirement table).  A relationship ``y = f(x)`` on the plane z becomes
``y = g(x)`` with

    g = f o phi^{-1},     phi(x) = x + beta * f(x),   beta = beta_kl(z),

which is a differentiable function precisely when ``1 + beta*f'(x) != 0`` on
the domain of ``f``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

Func = Callable[[float], float]

#: Height of each object's plane.
OBJECT_Z = {1: -1.0, 2: 1.0, 3: 0.0}

FRAMES = (1, 2, 3)


def _check(a: float, *frames: int) -> None:
    if a == 0:
        raise ValueError("a must be non-zero")
    for k in frames:
        if k not in FRAMES:
            raise ValueError(f"frames are 1, 2 or 3, got {k!r}")


def shear_potential(k: int, a: float) -> Func:
    """sigma_k(z): shear of frame ``k`` relative to the base frame where all objects rest.

    sigma_k(z_i) is the slope of object i's locus in frame k (row k of the table).
    """
    _check(a, k)
    if k == 1:
        return lambda z: (1.0 - z * z) / a
    if k == 2:
        return lambda z: -(1.0 - z * z) / a
    return lambda z: z / a


def beta(src: int, dst: int, a: float, z: float) -> float:
    """Shear coefficient beta_{src,dst}(z) = sigma_dst(z) - sigma_src(z)."""
    _check(a, src, dst)
    return shear_potential(dst, a)(z) - shear_potential(src, a)(z)


@dataclass(frozen=True)
class FrameMap:
    """The restriction of T_{src,dst} to the plane at height ``z``: ``x -> x + beta*y``."""

    src: int
    dst: int
    z: float
    beta: float

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        return x + self.beta * y, y

    def inverse(self) -> "FrameMap":
        return FrameMap(self.dst, self.src, self.z, -self.beta)

    def compose(self, first: "FrameMap") -> "FrameMap":
        """Return ``self o first`` (apply ``first``, then ``self``)."""
        if first.dst != self.src or first.z != self.z:
            raise ValueError(f"cannot compose {first} then {self}")
        return FrameMap(first.src, self.dst, self.z, first.beta + self.beta)


def frame_map(src: int, dst: int, a: float, z: float) -> FrameMap:
    """The transformation from frame ``src`` to frame ``dst`` on the plane at height ``z``.

    Pass ``z=OBJECT_Z[i]`` for anything belonging to object ``i``.
    """
    return FrameMap(src, dst, z, beta(src, dst, a, z))


def transform_point(
    src: int, dst: int, a: float, x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """Coordinates in frame ``dst`` of the point ``(x, y, z)`` given in frame ``src``."""
    xp, yp = frame_map(src, dst, a, z)(x, y)
    return xp, yp, z


def transform_path(src: int, dst: int, a: float, obj: int, p: Func) -> Func:
    """Transform object ``obj``'s path ``x = p(y)`` from frame ``src`` to frame ``dst``.

    Because y and z are unchanged by every frame map this needs no inversion:
    ``p_new(y) = p(y) + beta_{src,dst}(z_obj) * y``.  With ``p == 0`` this is
    the requirement table.  Works for any relationship written as x = h(y).
    """
    b = beta(src, dst, a, OBJECT_Z[obj])
    return lambda y: p(y) + b * y


def transform_function(
    src: int,
    dst: int,
    a: float,
    f: Func,
    domain: Tuple[float, float],
    fprime: Optional[Func] = None,
    *,
    obj: Optional[int] = None,
    z: Optional[float] = None,
    samples: int = 1000,
    tol: float = 1e-12,
) -> Tuple[Func, Tuple[float, float], Optional[Func]]:
    """Apply ``Lambda_{src,dst}`` to the relationship ``y = f(x)`` on ``domain``.

    The relationship lives on the plane of object ``obj`` (or at explicit height
    ``z``; exactly one must be given).  Returns ``(g, g_domain, g_prime)``:
    ``g`` is the relationship in frame ``dst``, ``g_domain`` the interval on
    which it is defined, ``g_prime`` its derivative (``None`` unless ``fprime``
    was supplied).

    ``g`` is evaluated by inverting ``phi(x) = x + beta*f(x)`` with bisection,
    valid because ``phi`` is strictly monotone whenever ``g`` exists.  Raises
    ``ValueError`` if ``phi`` is found not to be monotone on ``domain`` (``f'``
    crosses the forbidden slope ``-1/beta``, so no differentiable ``g`` exists).
    """
    if (obj is None) == (z is None):
        raise ValueError("give exactly one of obj= or z=")
    height = OBJECT_Z[obj] if obj is not None else z
    b = beta(src, dst, a, height)
    lo, hi = domain
    if not lo < hi:
        raise ValueError("domain must be a non-degenerate interval (lo, hi)")

    def phi(x: float) -> float:
        return x + b * f(x)

    # Monotonicity check on a grid.  A fold narrower than the grid spacing can
    # be missed; pass a finer `samples` if needed.
    xs = [lo + (hi - lo) * i / samples for i in range(samples + 1)]
    vals = [phi(x) for x in xs]
    diffs = [q - p for q, p in zip(vals[1:], vals[:-1])]
    if any(d == 0 for d in diffs) or (min(diffs) < 0 < max(diffs)):
        raise ValueError(
            f"phi(x) = x + {b}*f(x) is not monotone on {domain}: f' crosses the "
            f"forbidden slope {-1 / b if b else 'n/a'}, so the image is not the "
            "graph of a differentiable function"
        )
    increasing = diffs[0] > 0
    g_lo, g_hi = (vals[0], vals[-1]) if increasing else (vals[-1], vals[0])

    def phi_inverse(u: float) -> float:
        if not g_lo - tol <= u <= g_hi + tol:
            raise ValueError(f"{u} is outside the transformed domain ({g_lo}, {g_hi})")
        left, right = lo, hi
        for _ in range(200):
            mid = 0.5 * (left + right)
            if (phi(mid) < u) == increasing:
                left = mid
            else:
                right = mid
            if right - left <= tol:
                break
        return 0.5 * (left + right)

    def g(u: float) -> float:
        return f(phi_inverse(u))

    g_prime: Optional[Func] = None
    if fprime is not None:

        def g_prime(u: float) -> float:  # noqa: F811 - intentional rebinding
            x = phi_inverse(u)
            denom = 1.0 + b * fprime(x)
            if denom == 0:
                raise ZeroDivisionError("g is not differentiable here: 1 + beta*f'(x) = 0")
            return fprime(x) / denom

    return g, (g_lo, g_hi), g_prime
