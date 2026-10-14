from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pde_cas.grid import (
    Room2DFVGrid,
    mass_inner,
    mass_norm,
    normalize_mass,
)


def test_cell_centers_faces_and_x_fast_ordering() -> None:
    grid = Room2DFVGrid(length_x=6.0, length_y=2.0, num_x=3, num_y=2)
    assert grid.hx == 2.0
    assert grid.hy == 1.0
    assert grid.cell_area == 2.0
    assert grid.size == 6
    assert_array_equal(grid.x, [1.0, 3.0, 5.0])
    assert_array_equal(grid.y, [0.5, 1.5])
    assert_array_equal(grid.x_faces, [0.0, 2.0, 4.0, 6.0])
    assert_array_equal(grid.y_faces, [0.0, 1.0, 2.0])
    x, y = grid.mesh()
    assert_array_equal(grid.flatten(x), [1.0, 3.0, 5.0, 1.0, 3.0, 5.0])
    assert_array_equal(grid.flatten(y), [0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    assert_array_equal(grid.unflatten(grid.flatten(x)), x)
    assert_array_equal(grid.mass_matrix.toarray(), 2.0 * np.eye(6))


def test_mass_inner_norm_and_l2_normalization() -> None:
    grid = Room2DFVGrid(num_x=3, num_y=2)
    left = np.arange(grid.size, dtype=float)
    right = left[::-1]
    assert_allclose(mass_inner(grid, left, right), left @ grid.mass_matrix @ right)
    assert_allclose(mass_norm(grid, left) ** 2, mass_inner(grid, left, left))
    assert_allclose(mass_norm(grid, normalize_mass(grid, left)), 1.0)
    ones = np.ones(grid.size)
    assert_allclose(mass_inner(grid, ones, ones), grid.length_x * grid.length_y)
    for bad in (np.zeros(grid.size), np.full(grid.size, np.nan), np.full(grid.size, np.inf)):
        with pytest.raises(ValueError):
            normalize_mass(grid, bad)


def test_grid_rejects_shape_mismatches() -> None:
    grid = Room2DFVGrid(num_x=3, num_y=2)
    with pytest.raises(ValueError):
        grid.flatten(np.ones((3, 2)))
    with pytest.raises(ValueError):
        grid.unflatten(np.ones((2, 3)))
    with pytest.raises(ValueError):
        mass_inner(grid, np.ones(5), np.ones(5))


@pytest.mark.parametrize("kwargs", [
    {"length_x": 0.0}, {"length_y": -1.0}, {"length_x": np.inf}, {"length_y": np.nan},
    {"num_x": 0}, {"num_y": -1}, {"num_x": 2.5}, {"num_y": True},
])
def test_grid_rejects_invalid_geometry(kwargs) -> None:
    with pytest.raises(ValueError):
        Room2DFVGrid(**kwargs)
