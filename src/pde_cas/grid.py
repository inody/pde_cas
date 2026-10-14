"""Cell-centered FV geometry, independent of the existing nodal room grid."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Room2DFVGrid:
    """Uniform cells on [0, length_x] x [0, length_y], with x-fast ordering."""

    length_x: float = 3.0
    length_y: float = 2.0
    num_x: int = 95
    num_y: int = 63

    def __post_init__(self) -> None:
        for length in (self.length_x, self.length_y):
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError("domain lengths must be finite and positive")
        for count in (self.num_x, self.num_y):
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError("grid dimensions must be positive integers")

    @property
    def hx(self) -> float:
        return self.length_x / self.num_x

    @property
    def hy(self) -> float:
        return self.length_y / self.num_y

    @property
    def cell_area(self) -> float:
        return self.hx * self.hy

    @property
    def size(self) -> int:
        return self.num_x * self.num_y

    @property
    def x(self) -> np.ndarray:
        return self.hx * (np.arange(self.num_x) + 0.5)

    @property
    def y(self) -> np.ndarray:
        return self.hy * (np.arange(self.num_y) + 0.5)

    @property
    def x_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.length_x, self.num_x + 1)

    @property
    def y_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.length_y, self.num_y + 1)

    @property
    def mass_matrix(self) -> sparse.csr_matrix:
        return self.cell_area * sparse.eye(self.size, format="csr")

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="xy")

    def flatten(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (self.num_y, self.num_x):
            raise ValueError(f"expected shape {(self.num_y, self.num_x)}, got {array.shape}")
        return array.reshape(self.size)

    def unflatten(self, values: np.ndarray) -> np.ndarray:
        vector = _vector(self, values)
        return vector.reshape((self.num_y, self.num_x))


def _vector(grid: Room2DFVGrid, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (grid.size,):
        raise ValueError(f"expected length {grid.size}, got {vector.shape}")
    return vector


def mass_inner(grid: Room2DFVGrid, left: np.ndarray, right: np.ndarray) -> float:
    return float(grid.cell_area * np.dot(_vector(grid, left), _vector(grid, right)))


def mass_norm(grid: Room2DFVGrid, values: np.ndarray) -> float:
    return float(np.sqrt(mass_inner(grid, values, values)))


def normalize_mass(grid: Room2DFVGrid, values: np.ndarray) -> np.ndarray:
    """Normalize a cell-average vector in discrete L2, not to unit integral."""
    vector = _vector(grid, values)
    norm = mass_norm(grid, vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a zero or nonfinite profile")
    return vector / norm
