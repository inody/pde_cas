"""Conservative, linear donor-cell advection and no-flux FV diffusion.

Face fluxes are integrated normal velocities (velocity times face length),
not pointwise velocities. Every interior face contributes equal and opposite
mass changes to its two neighbors; wall faces contribute nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from pde_cas.grid import Room2DFVGrid


@dataclass(frozen=True)
class Room2DFVParameters:
    diffusivity: float = 0.03
    damping: float = 0.02
    velocity_scale: float = 0.15

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be finite and nonnegative")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        if not np.isfinite(self.velocity_scale):
            raise ValueError("velocity_scale must be finite")


@dataclass(frozen=True)
class Room2DFVFaceFluxes:
    """Integrated fluxes in the positive coordinate directions.

    x has shape (num_y, num_x + 1); y has shape (num_y + 1, num_x).
    """

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Room2DFVOperator:
    grid: Room2DFVGrid
    parameters: Room2DFVParameters
    face_fluxes: Room2DFVFaceFluxes
    laplacian: sparse.csr_matrix
    advection: sparse.csr_matrix
    generator: sparse.csr_matrix


def stream_function_vertices(grid: Room2DFVGrid, velocity_scale: float) -> np.ndarray:
    """Evaluate psi = V0 sin(pi x/Lx) sin(pi y/Ly) at cell vertices."""
    if not np.isfinite(velocity_scale):
        raise ValueError("velocity_scale must be finite")
    x, y = np.meshgrid(grid.x_faces, grid.y_faces, indexing="xy")
    psi = velocity_scale * np.sin(np.pi * x / grid.length_x) * np.sin(np.pi * y / grid.length_y)
    # Impose the analytic boundary value exactly, avoiding sin(pi) roundoff.
    psi[[0, -1], :] = 0.0
    psi[:, [0, -1]] = 0.0
    return psi


def build_face_fluxes(grid: Room2DFVGrid, velocity_scale: float = 0.15) -> Room2DFVFaceFluxes:
    """Discrete curl of vertex psi for v = (d_y psi, -d_x psi).

    These differences are exact face integrals for the chosen stream function.
    The cell flux balance telescopes, including next to walls.
    """
    psi = stream_function_vertices(grid, velocity_scale)
    return Room2DFVFaceFluxes(x=np.diff(psi, axis=0), y=-np.diff(psi, axis=1))


def discrete_divergence(grid: Room2DFVGrid, fluxes: Room2DFVFaceFluxes) -> np.ndarray:
    """Cell-average divergence, shaped (num_y, num_x)."""
    if fluxes.x.shape != (grid.num_y, grid.num_x + 1) or fluxes.y.shape != (
        grid.num_y + 1, grid.num_x
    ):
        raise ValueError("face-flux shapes do not match the grid")
    return (np.diff(fluxes.x, axis=1) + np.diff(fluxes.y, axis=0)) / grid.cell_area


def _face_matrix(
    size: int,
    left: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
) -> sparse.csr_matrix:
    """Transfer rates left -> right and right -> left on interior faces."""
    matrix = sparse.coo_matrix(
        (
            np.concatenate((-forward, backward, forward, -backward)),
            (np.concatenate((left, left, right, right)), np.concatenate((left, right, left, right))),
        ),
        shape=(size, size),
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def build_room_2d_fv_operator(
    grid: Room2DFVGrid,
    parameters: Room2DFVParameters | None = None,
) -> Room2DFVOperator:
    """Assemble kappa Delta_h + A_adv,h - gamma I, including kappa = 0.

    The advective state flux is F^+ u_L + F^- u_R. Positive flux removes
    mass from L and adds it to R. No limiter or state-dependent coefficients
    are used. Diffusion uses centered two-point fluxes on interior faces and
    zero wall flux, not zero ghost-cell values.
    """
    params = parameters if parameters is not None else Room2DFVParameters()
    fluxes = build_face_fluxes(grid, params.velocity_scale)
    indices = np.arange(grid.size).reshape(grid.num_y, grid.num_x)
    left = np.concatenate((indices[:, :-1].ravel(), indices[:-1, :].ravel()))
    right = np.concatenate((indices[:, 1:].ravel(), indices[1:, :].ravel()))
    flux = np.concatenate((fluxes.x[:, 1:-1].ravel(), fluxes.y[1:-1, :].ravel()))
    advection = _face_matrix(
        grid.size, left, right,
        np.maximum(flux, 0.0) / grid.cell_area,
        np.maximum(-flux, 0.0) / grid.cell_area,
    )

    # (face length / center distance) / cell area = 1 / spacing^2.
    diffusion_rates = np.concatenate((
        np.full(grid.num_y * (grid.num_x - 1), 1.0 / grid.hx**2),
        np.full((grid.num_y - 1) * grid.num_x, 1.0 / grid.hy**2),
    ))
    laplacian = _face_matrix(grid.size, left, right, diffusion_rates, diffusion_rates)
    generator = (
        params.diffusivity * laplacian + advection
        - params.damping * sparse.eye(grid.size, format="csr")
    ).tocsr()
    generator.eliminate_zeros()
    return Room2DFVOperator(grid, params, fluxes, laplacian, advection, generator)
