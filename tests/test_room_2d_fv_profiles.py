from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import dblquad

from pde_cas.grid import Room2DFVGrid, mass_norm
from pde_cas.profiles import compact_bump


def test_compact_profiles_are_nonnegative_supported_and_mass_normalized() -> None:
    grid = Room2DFVGrid(num_x=24, num_y=17)
    center = (1.2, 0.8)
    radius = 0.4
    raw = compact_bump(grid, center, radius, normalize=False)
    profile = compact_bump(grid, center, radius)
    assert raw.shape == (grid.size,)
    assert np.all(raw >= 0.0)
    assert np.max(raw) <= 1.0
    assert np.any(raw > 0.0)
    x, y = grid.mesh()
    # Distance from the center to each closed cell rectangle.
    dx = np.maximum(np.abs(x - center[0]) - grid.hx / 2, 0.0)
    dy = np.maximum(np.abs(y - center[1]) - grid.hy / 2, 0.0)
    outside = grid.flatten(dx**2 + dy**2) >= radius**2
    assert_array_equal(raw[outside], 0.0)
    assert_allclose(profile, raw / mass_norm(grid, raw))
    assert_allclose(mass_norm(grid, profile), 1.0, atol=2e-15)


def test_sampling_is_a_cell_average_not_a_center_value() -> None:
    grid = Room2DFVGrid(length_x=1.0, length_y=0.6, num_x=1, num_y=1)
    center, radius = (0.5, 0.3), 1.0
    # Entire cell is inside the disk, so order 4 integrates the degree-6
    # polynomial exactly. The midpoint value alone would instead be one.
    values = compact_bump(grid, center, radius, quadrature_order=4, normalize=False)
    integral, _ = dblquad(
        lambda y, x: (1.0 - (x - center[0])**2 - (y - center[1])**2)**3,
        0.0, grid.length_x, lambda x: 0.0, lambda x: grid.length_y,
        epsabs=1e-13, epsrel=1e-13,
    )
    assert_allclose(values[0], integral / grid.cell_area, atol=2e-15)
    assert values[0] < 0.9


@pytest.mark.parametrize("center,fraction", [((1.5, 1.0), 1.0), ((0.0, 0.0), 0.25)])
def test_cell_average_integral_including_wall_clipping(center, fraction) -> None:
    grid = Room2DFVGrid(num_x=40, num_y=32)
    radius = 0.45
    values = compact_bump(grid, center, radius, quadrature_order=10, normalize=False)
    # Integral of (1-rho^2/r^2)^3 over the disk is pi r^2/4.
    expected = fraction * np.pi * radius**2 / 4.0
    assert_allclose(grid.cell_area * np.sum(values), expected, rtol=2e-7, atol=1e-10)


def test_zero_support_is_not_silently_normalized() -> None:
    grid = Room2DFVGrid(num_x=4, num_y=3)
    assert_array_equal(compact_bump(grid, (10.0, 10.0), 0.1, normalize=False), 0.0)
    with pytest.raises(ValueError, match="zero"):
        compact_bump(grid, (10.0, 10.0), 0.1)


@pytest.mark.parametrize("kwargs", [
    {"radius": 0.0}, {"radius": -1.0}, {"radius": np.nan}, {"radius": np.inf},
    {"center": (np.nan, 0.0)}, {"center": (0.0,)},
    {"quadrature_order": 0}, {"quadrature_order": 1.5}, {"quadrature_order": True},
])
def test_profile_rejects_invalid_inputs(kwargs) -> None:
    arguments = {"center": (1.0, 1.0), "radius": 0.2, **kwargs}
    with pytest.raises(ValueError):
        compact_bump(Room2DFVGrid(num_x=3, num_y=2), **arguments)
