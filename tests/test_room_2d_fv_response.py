from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.integrate import simpson
from scipy.linalg import expm

import pde_cas.response as fv_response
from pde_cas.grid import Room2DFVGrid, mass_inner
from pde_cas.operator import Room2DFVParameters, build_room_2d_fv_operator
from pde_cas.profiles import compact_bump
from pde_cas.response import (
    build_room_2d_fv_channel_gramians,
    channel_gramians_from_response_series,
    mass_adjoint,
    response_series_adjoint,
    response_series_forward,
    simpson_weights,
    target_matrix,
    whole_domain_channel_energies,
)


@pytest.fixture
def problem():
    grid = Room2DFVGrid(length_x=2.3, length_y=1.2, num_x=6, num_y=4)
    op = build_room_2d_fv_operator(grid, Room2DFVParameters(velocity_scale=0.7))
    b = np.column_stack([compact_bump(grid, z, 0.5) for z in [(0.3, 0.3), (1.1, 0.4), (1.9, 0.9)]])
    c = np.column_stack([compact_bump(grid, z, 0.6) for z in [(0.8, 0.7), (1.8, 0.4)]])
    return grid, op.generator, b, c, np.linspace(0.0, 0.9, 13)


@pytest.mark.parametrize("diffusivity", [0.0, 0.03])
@pytest.mark.parametrize("scale", [-0.7, 0.7])
def test_forward_and_mass_adjoint_agree_with_dense_exponential(problem, diffusivity, scale):
    grid, _, b, c, times = problem
    a = build_room_2d_fv_operator(
        grid, Room2DFVParameters(diffusivity=diffusivity, velocity_scale=scale),
    ).generator
    forward = response_series_forward(a, grid, b, c, times, chunk_size=3)
    adjoint = response_series_adjoint(a, grid, b, c, times, chunk_size=5)
    expected = np.array([c.T @ grid.mass_matrix @ expm(t * a.toarray()) @ b for t in times])
    assert forward.shape == (13, 2, 3)
    assert_allclose(forward, expected, rtol=2e-12, atol=2e-14)
    assert_allclose(adjoint, expected, rtol=2e-12, atol=2e-14)
    assert_allclose(forward[0], target_matrix(grid, c) @ b, atol=2e-15)
    u, v = b[:, 0], c[:, 0]
    assert_allclose(mass_inner(grid, u, a @ v), mass_inner(grid, mass_adjoint(a, grid) @ u, v), atol=2e-15)


@pytest.mark.parametrize("method", ["forward", "adjoint"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 12, 99, None])
def test_chunks_and_boundary_samples_match_unchunked_and_simpson(problem, method, chunk_size):
    grid, a, b, c, times = problem
    series = response_series_forward if method == "forward" else response_series_adjoint
    reference = series(a, grid, b, c, times, chunk_size=None)
    actual = series(a, grid, b, c, times, chunk_size=chunk_size)
    assert_allclose(actual, reference, rtol=2e-12, atol=2e-14)
    # This includes all shared endpoints, and last chunks of odd/even length.
    got_times, responses, grams = build_room_2d_fv_channel_gramians(
        a, grid, b, c, times[-1], num_time_points=times.size,
        method=method, chunk_size=chunk_size,
    )
    assert_array_equal(got_times, times)
    assert_allclose(responses, reference, rtol=2e-12, atol=2e-14)
    independent = np.stack([
        simpson(reference[:, :, i, None] * reference[:, None, :, i], x=times, axis=0)
        for i in range(b.shape[1])
    ])
    assert_allclose(grams, independent, rtol=2e-12, atol=2e-14)
    assert_allclose(channel_gramians_from_response_series(times, actual), grams, atol=2e-14)
    _, not_stored, streamed = build_room_2d_fv_channel_gramians(
        a, grid, b, c, times[-1], num_time_points=times.size,
        method=method, chunk_size=chunk_size, store_responses=False,
    )
    assert not_stored is None
    assert_allclose(streamed, grams, atol=2e-14)
    assert_array_equal(grams, grams.swapaxes(1, 2))
    assert np.min(np.linalg.eigvalsh(grams)) >= -2e-14


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 12, 99, None])
def test_whole_domain_fh_chunks_match_independent_full_history(problem, chunk_size):
    grid, a, b, _, times = problem
    states = np.array([expm(t * a.toarray()) @ b for t in times])
    expected = simpson(grid.cell_area * np.sum(states**2, axis=1), x=times, axis=0)
    actual = whole_domain_channel_energies(a, grid, b, times, chunk_size=chunk_size)
    reference = whole_domain_channel_energies(a, grid, b, times, chunk_size=None)
    assert actual.shape == (b.shape[1],)
    assert np.all(actual >= 0.0)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    assert_allclose(actual, reference, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7, None])
@pytest.mark.parametrize("method", ["forward", "adjoint"])
def test_simpson_exact_for_nonconstant_polynomial_gramians_and_fh(chunk_size, method):
    # Controlled nilpotent generator: A^2=0, hence exp(tA)B=B+tAB.
    # This is a quadrature fixture, not a conservative FV operator.
    grid = Room2DFVGrid(length_x=2.4, length_y=0.7, num_x=3, num_y=2)
    a = np.zeros((grid.size, grid.size))
    a[0, 1], a[2, 3], a[4, 5] = 0.7, -0.4, 1.3
    rng = np.random.default_rng(18)
    b, c = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    horizon = 1.7
    times, responses, grams = build_room_2d_fv_channel_gramians(
        a, grid, b, c, horizon, num_time_points=11, method=method, chunk_size=chunk_size,
    )
    q0, q1 = grid.cell_area * c.T @ b, grid.cell_area * c.T @ a @ b
    assert_allclose(responses, q0[None] + times[:, None, None] * q1[None], atol=3e-14)
    expected = np.stack([
        horizon * np.outer(q0[:, i], q0[:, i])
        + horizon**2 / 2 * (np.outer(q0[:, i], q1[:, i]) + np.outer(q1[:, i], q0[:, i]))
        + horizon**3 / 3 * np.outer(q1[:, i], q1[:, i])
        for i in range(b.shape[1])
    ])
    assert_allclose(grams, expected, rtol=3e-14, atol=3e-14)
    ab = a @ b
    expected_fh = grid.cell_area * (
        horizon * np.sum(b**2, axis=0) + horizon**2 * np.sum(b * ab, axis=0)
        + horizon**3 / 3 * np.sum(ab**2, axis=0)
    )
    assert_allclose(
        whole_domain_channel_energies(a, grid, b, times, chunk_size=chunk_size),
        expected_fh, rtol=3e-14, atol=3e-14,
    )


@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_simpson_weights_integrate_cubics_exactly(power):
    times = np.linspace(0.0, 1.7, 11)
    assert_allclose(simpson_weights(times) @ times**power, 1.7**(power + 1) / (power + 1), atol=2e-15)


def test_single_cell_damping_has_fourth_order_simpson_convergence():
    grid = Room2DFVGrid(length_x=2.0, length_y=3.0, num_x=1, num_y=1)
    gamma, horizon = 0.8, 1.3
    a = build_room_2d_fv_operator(grid, Room2DFVParameters(damping=gamma)).generator
    b, c = np.array([[2.0]]), np.array([[0.5]])
    factor = -np.expm1(-2 * gamma * horizon) / (2 * gamma)
    exact_gram = (grid.cell_area * 2 * 0.5)**2 * factor
    exact_fh = grid.cell_area * 2**2 * factor
    errors = []
    for num in (9, 17, 33):
        times, responses, grams = build_room_2d_fv_channel_gramians(
            a, grid, b, c, horizon, num_time_points=num, chunk_size=3,
        )
        fh = whole_domain_channel_energies(a, grid, b, times, chunk_size=5)
        assert responses.shape == (num, 1, 1)
        assert grams.shape == (1, 1, 1)
        assert_allclose(responses[:, 0, 0], 6 * np.exp(-gamma * times), atol=2e-14)
        errors.append([abs(grams[0, 0, 0] - exact_gram), abs(fh[0] - exact_fh)])
    errors = np.array(errors)
    assert np.all(errors[:-1] / errors[1:] > 15.0)


@pytest.mark.parametrize("method", ["forward", "adjoint", "fh"])
def test_propagation_calls_are_bounded_in_time_and_fh_columns(problem, monkeypatch, method):
    grid, a, b, c, times = problem
    original = fv_response.expm_multiply
    calls = []

    def spy(generator, initial, **kwargs):
        calls.append((initial.shape, kwargs["num"]))
        assert kwargs["num"] <= 5  # four intervals plus the shared endpoint
        return original(generator, initial, **kwargs)

    monkeypatch.setattr(fv_response, "expm_multiply", spy)
    if method == "fh":
        whole_domain_channel_energies(a, grid, b, times, chunk_size=4)
        assert len(calls) == 3 * b.shape[1]
        assert all(shape == (grid.size, 1) for shape, _ in calls)
    else:
        _, responses, _ = build_room_2d_fv_channel_gramians(
            a, grid, b, c, times[-1], num_time_points=times.size,
            method=method, chunk_size=4, store_responses=False,
        )
        assert responses is None
        assert len(calls) == 3
        columns = b.shape[1] if method == "forward" else c.shape[1]
        assert all(shape == (grid.size, columns) for shape, _ in calls)


@pytest.mark.parametrize("constructor", [
    sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix, sparse.dia_matrix,
    sparse.lil_matrix, sparse.dok_matrix, sparse.csr_array,
])
def test_sparse_formats_preserve_responses_and_mass_adjoint(problem, constructor):
    grid, a, b, c, times = problem
    converted = constructor(a)
    expected = response_series_forward(a, grid, b, c, times)
    assert_allclose(response_series_forward(converted, grid, b, c, times), expected, atol=2e-14)
    assert_array_equal(mass_adjoint(converted, grid).toarray(), a.T.toarray())


def test_zero_profiles_produce_zero_responses_gramians_and_energies(problem):
    grid, a, b, c, times = problem
    zero_b = np.zeros_like(b)
    for method in ("forward", "adjoint"):
        _, responses, grams = build_room_2d_fv_channel_gramians(
            a, grid, zero_b, c, times[-1], num_time_points=times.size,
            method=method, chunk_size=5,
        )
        assert_array_equal(responses, 0.0)
        assert_array_equal(grams, 0.0)
    assert_array_equal(whole_domain_channel_energies(a, grid, zero_b, times, chunk_size=5), 0.0)
    assert_array_equal(response_series_adjoint(a, grid, b, np.zeros_like(c), times), 0.0)


def _call(problem, route, **overrides):
    grid, a, b, c, times = problem
    args = {"generator": a, "grid": grid, "actuator_profiles": b, "target_profiles": c, "times": times}
    args.update(overrides)
    if route == "fh":
        args.pop("target_profiles")
        return whole_domain_channel_energies(**args)
    if route == "build":
        args.pop("times")
        return build_room_2d_fv_channel_gramians(**args, horizon=times[-1], num_time_points=times.size)
    function = response_series_forward if route == "forward" else response_series_adjoint
    return function(**args)


@pytest.mark.parametrize("route", ["forward", "adjoint", "fh", "build"])
@pytest.mark.parametrize("field,value", [
    ("actuator_profiles", np.ones(24)),
    ("actuator_profiles", np.ones((23, 2))),
    ("actuator_profiles", np.ones((24, 0))),
    ("actuator_profiles", np.full((24, 2), np.nan)),
    ("actuator_profiles", np.ones((24, 2), dtype=complex)),
    ("generator", np.eye(23)),
    ("generator", np.zeros(24)),
    ("generator", sparse.diags([np.inf] * 24)),
    ("generator", np.eye(24, dtype=complex)),
    ("generator", sparse.eye(24, dtype=complex)),
    ("chunk_size", 0), ("chunk_size", -2), ("chunk_size", 1.5), ("chunk_size", True),
])
def test_common_input_validation(problem, route, field, value):
    with pytest.raises(ValueError):
        _call(problem, route, **{field: value})


@pytest.mark.parametrize("route", ["forward", "adjoint", "build"])
@pytest.mark.parametrize("targets", [
    np.ones(24), np.ones((23, 2)), np.ones((24, 0)),
    np.full((24, 2), np.inf), np.ones((24, 2), dtype=complex),
])
def test_target_shape_and_value_validation(problem, route, targets):
    with pytest.raises(ValueError):
        _call(problem, route, target_profiles=targets)
    with pytest.raises(ValueError):
        target_matrix(problem[0], targets)


@pytest.mark.parametrize("times", [
    [], [0.0], [[0.0, 0.5, 1.0]], [0.1, 0.5, 0.9],
    [0.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.2, 1.0],
    [0.0, np.nan, 1.0], [0.0, 0.5, np.inf], [0.0, 0.5j, 1.0],
    [0.0, 1e-12, 3e-12],
])
@pytest.mark.parametrize("route", ["forward", "adjoint", "fh"])
def test_time_grid_validation(problem, times, route):
    with pytest.raises(ValueError):
        _call(problem, route, times=times)
    with pytest.raises(ValueError):
        simpson_weights(times)


@pytest.mark.parametrize("num", [2, 4, 6])
def test_even_time_samples_allowed_for_responses_but_not_simpson(problem, num):
    grid, a, b, c, _ = problem
    times = np.linspace(0.0, 0.9, num)
    assert response_series_forward(a, grid, b, c, times).shape == (num, 2, 3)
    assert response_series_adjoint(a, grid, b, c, times).shape == (num, 2, 3)
    with pytest.raises(ValueError):
        whole_domain_channel_energies(a, grid, b, times)
    with pytest.raises(ValueError):
        channel_gramians_from_response_series(times, np.ones((num, 2, 3)))


@pytest.mark.parametrize("responses", [
    np.ones((13, 2)), np.ones((12, 2, 3)), np.ones((13, 0, 3)), np.ones((13, 2, 0)),
    np.full((13, 2, 3), np.nan), np.ones((13, 2, 3), dtype=complex),
])
def test_response_series_shape_and_value_validation(problem, responses):
    with pytest.raises(ValueError):
        channel_gramians_from_response_series(problem[-1], responses)


@pytest.mark.parametrize("overrides", [
    {"horizon": 0.0}, {"horizon": -1.0}, {"horizon": np.nan}, {"horizon": np.inf},
    {"horizon": [1.0]}, {"horizon": 1j},
    {"num_time_points": 1}, {"num_time_points": 4}, {"num_time_points": 3.5},
    {"num_time_points": True}, {"method": "invalid"}, {"store_responses": 1},
])
def test_builder_control_validation(problem, overrides):
    grid, a, b, c, _ = problem
    kwargs = {"horizon": 0.9, "num_time_points": 13, **overrides}
    with pytest.raises(ValueError):
        build_room_2d_fv_channel_gramians(a, grid, b, c, **kwargs)
