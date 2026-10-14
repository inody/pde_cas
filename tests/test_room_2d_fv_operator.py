from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from pde_cas.grid import Room2DFVGrid, mass_inner
from pde_cas.operator import (
    Room2DFVParameters,
    build_face_fluxes,
    build_room_2d_fv_operator,
    discrete_divergence,
    stream_function_vertices,
)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (8, 1), (7, 5), (47, 31)])
@pytest.mark.parametrize("scale", [-1.3, 0.0, 0.15])
def test_face_fluxes_have_zero_walls_and_discrete_divergence(shape, scale) -> None:
    grid = Room2DFVGrid(length_x=2.7, length_y=1.1, num_x=shape[0], num_y=shape[1])
    fluxes = build_face_fluxes(grid, scale)
    assert fluxes.x.shape == (grid.num_y, grid.num_x + 1)
    assert fluxes.y.shape == (grid.num_y + 1, grid.num_x)
    assert_array_equal(fluxes.x[:, [0, -1]], 0.0)
    assert_array_equal(fluxes.y[[0, -1], :], 0.0)
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(scale)) / grid.cell_area
    assert_allclose(discrete_divergence(grid, fluxes), 0.0, atol=tolerance, rtol=0.0)


def test_face_fluxes_are_oriented_stream_function_differences() -> None:
    grid = Room2DFVGrid(length_x=2.7, length_y=1.1, num_x=8, num_y=5)
    scale = 0.7
    psi = stream_function_vertices(grid, scale)
    fluxes = build_face_fluxes(grid, scale)
    x, y = np.meshgrid(grid.x_faces, grid.y_faces, indexing="xy")
    assert_allclose(psi, scale * np.sin(np.pi * x / grid.length_x) * np.sin(np.pi * y / grid.length_y), atol=1e-15)
    assert_array_equal(fluxes.x, psi[1:, :] - psi[:-1, :])
    assert_array_equal(fluxes.y, psi[:, :-1] - psi[:, 1:])
    # On the lower/left half of the room v_x > 0 and v_y < 0.
    assert fluxes.x[0, 1] > 0.0
    assert fluxes.y[1, 0] < 0.0


@pytest.mark.parametrize("scale", [-0.8, 0.0, 0.8])
@pytest.mark.parametrize("diffusivity", [0.0, 0.03])
def test_nonnegative_off_diagonals_and_mass_conservation(scale, diffusivity) -> None:
    grid = Room2DFVGrid(num_x=7, num_y=5)
    params = Room2DFVParameters(diffusivity=diffusivity, damping=0.12, velocity_scale=scale)
    op = build_room_2d_fv_operator(grid, params)
    for matrix in (op.advection, op.laplacian, op.generator):
        assert sparse.isspmatrix_csr(matrix)
        coo = matrix.tocoo()
        assert np.all(coo.data[coo.row != coo.col] >= 0.0)
    weights = np.full(grid.size, grid.cell_area)
    assert_allclose(weights @ op.advection, 0.0, atol=2e-15)
    assert_allclose(weights @ op.laplacian, 0.0, atol=2e-15)
    assert_allclose(weights @ op.generator, -params.damping * weights, atol=2e-15)
    if scale == 0.0:
        assert op.advection.nnz == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (7, 1), (7, 5)])
@pytest.mark.parametrize("diffusivity", [0.0, 0.03])
def test_constant_states_follow_damping_law(shape, diffusivity) -> None:
    grid = Room2DFVGrid(num_x=shape[0], num_y=shape[1])
    params = Room2DFVParameters(diffusivity=diffusivity, damping=0.12, velocity_scale=-0.7)
    op = build_room_2d_fv_operator(grid, params)
    ones = np.ones(grid.size)
    assert_allclose(op.laplacian @ ones, 0.0, atol=2e-14)
    assert_allclose(op.advection @ ones, 0.0, atol=2e-14)
    assert_allclose(op.generator @ ones, -params.damping * ones, atol=2e-14)
    time = 0.6
    assert_allclose(expm_multiply(time * op.generator, ones), np.exp(-params.damping * time) * ones, atol=2e-14)


@pytest.mark.parametrize("scale", [-0.8, 0.8])
def test_advection_matches_linear_donor_cell_flux_balance(scale) -> None:
    grid = Room2DFVGrid(length_x=2.7, length_y=1.1, num_x=8, num_y=5)
    op = build_room_2d_fv_operator(grid, Room2DFVParameters(velocity_scale=scale))
    state = np.random.default_rng(42).normal(size=(grid.num_y, grid.num_x))
    fx = np.zeros_like(op.face_fluxes.x)
    fy = np.zeros_like(op.face_fluxes.y)
    f = op.face_fluxes.x[:, 1:-1]
    fx[:, 1:-1] = np.maximum(f, 0.0) * state[:, :-1] + np.minimum(f, 0.0) * state[:, 1:]
    f = op.face_fluxes.y[1:-1, :]
    fy[1:-1, :] = np.maximum(f, 0.0) * state[:-1, :] + np.minimum(f, 0.0) * state[1:, :]
    expected = -(np.diff(fx, axis=1) + np.diff(fy, axis=0)) / grid.cell_area
    assert_allclose(op.advection @ grid.flatten(state), grid.flatten(expected), atol=2e-14)


def test_centered_diffusion_has_no_wall_sink() -> None:
    grid = Room2DFVGrid(length_x=6.0, length_y=2.0, num_x=3, num_y=2)
    laplacian = build_room_2d_fv_operator(grid).laplacian
    expected = np.array([
        [-1.25, 0.25, 0.0, 1.0, 0.0, 0.0],
        [0.25, -1.5, 0.25, 0.0, 1.0, 0.0],
        [0.0, 0.25, -1.25, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.25, 0.25, 0.0],
        [0.0, 1.0, 0.0, 0.25, -1.5, 0.25],
        [0.0, 0.0, 1.0, 0.0, 0.25, -1.25],
    ])
    assert_array_equal(laplacian.toarray(), expected)


@pytest.mark.parametrize("scale", [-0.8, 0.0, 0.8])
@pytest.mark.parametrize("diffusivity", [0.0, 0.03])
def test_mass_weighted_l2_dissipativity_and_face_energy_identity(scale, diffusivity) -> None:
    grid = Room2DFVGrid(length_x=2.7, length_y=1.1, num_x=7, num_y=5)
    params = Room2DFVParameters(diffusivity=diffusivity, damping=0.12, velocity_scale=scale)
    op = build_room_2d_fv_operator(grid, params)
    # Test the quadratic form for every state via its symmetric-part spectrum.
    for matrix, bound in ((op.advection, 0.0), (op.laplacian, 0.0), (op.generator, -params.damping)):
        weighted = grid.mass_matrix @ matrix
        symmetric = (weighted + weighted.T).toarray() / 2.0
        assert np.linalg.eigvalsh(symmetric)[-1] <= grid.cell_area * bound + 2e-14

    state = np.random.default_rng(123).normal(size=(grid.num_y, grid.num_x))
    vector = grid.flatten(state)
    jump_x = np.diff(state, axis=1) ** 2
    jump_y = np.diff(state, axis=0) ** 2
    adv_energy = -0.5 * (
        np.sum(np.abs(op.face_fluxes.x[:, 1:-1]) * jump_x)
        + np.sum(np.abs(op.face_fluxes.y[1:-1, :]) * jump_y)
    )
    diff_energy = -(grid.hy / grid.hx * np.sum(jump_x) + grid.hx / grid.hy * np.sum(jump_y))
    assert_allclose(mass_inner(grid, vector, op.advection @ vector), adv_energy, atol=2e-14)
    assert_allclose(mass_inner(grid, vector, op.laplacian @ vector), diff_energy, atol=2e-14)
    expected = adv_energy + diffusivity * diff_energy - params.damping * mass_inner(grid, vector, vector)
    assert_allclose(mass_inner(grid, vector, op.generator @ vector), expected, atol=2e-14)


@pytest.mark.parametrize("kwargs", [
    {"diffusivity": -0.01}, {"diffusivity": np.nan}, {"diffusivity": np.inf},
    {"damping": -0.01}, {"damping": np.nan}, {"damping": np.inf},
    {"velocity_scale": np.nan}, {"velocity_scale": np.inf},
])
def test_parameters_reject_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        Room2DFVParameters(**kwargs)
