"""Compact C2 bumps sampled as FV cell averages by tensor Gauss quadrature."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss

from pde_cas.grid import Room2DFVGrid, normalize_mass


def compact_bump(
    grid: Room2DFVGrid,
    center: tuple[float, float],
    radius: float,
    *,
    quadrature_order: int = 6,
    normalize: bool = True,
) -> np.ndarray:
    """Sample beta(x) = (1 - |x-center|^2 / radius^2)_+^3.

    Return x-fast cell averages, optionally normalized to unit mass-weighted
    L2 norm. Bumps crossing a wall are simply restricted to the room. The
    averages are numerical approximations: refine the grid or quadrature for
    small supports or cells cut by the support boundary. A support missed by
    all quadrature nodes raises ValueError when normalization is requested.
    """
    location = np.asarray(center, dtype=float)
    if location.shape != (2,) or not np.all(np.isfinite(location)):
        raise ValueError("center must contain two finite coordinates")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, Integral)
        or quadrature_order < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")

    nodes, weights = leggauss(quadrature_order)
    x, y = grid.mesh()
    averages = np.zeros_like(x)
    # The reference-square Jacobian divided by the cell area is 1/4.
    # Loop over quadrature nodes to avoid a grid x order^2 temporary array.
    for nx, wx in zip(nodes, weights):
        dx2 = ((x + 0.5 * grid.hx * nx - location[0]) / radius) ** 2
        for ny, wy in zip(nodes, weights):
            dy2 = ((y + 0.5 * grid.hy * ny - location[1]) / radius) ** 2
            averages += 0.25 * wx * wy * np.maximum(1.0 - dx2 - dy2, 0.0) ** 3
    values = grid.flatten(averages)
    return normalize_mass(grid, values) if normalize else values
