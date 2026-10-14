"""Memory-bounded target responses and finite-horizon FV energy integrals.

Generators may be real dense or SciPy sparse matrices of shape (cells, cells).
Profiles are real cell-average columns: B has shape (cells, actuators), C has
shape (cells, targets). Q(t) = C.T M exp(t A) B has shape (targets, actuators).
On Room2DFVGrid, M = cell_area I, so the mass adjoint M^-1 A.T M is A.T.
No Euclidean renormalization or allocation weights are applied to profiles.

All time grids start at zero and are uniform. Integration uses composite
Simpson quadrature with an odd number of samples. ``chunk_size`` is the maximum
number of time *intervals* per exponential-action call (default 16); None uses
one full-time block for reference calculations. Full-state memory scales with
chunk_size, not with the total number of time samples. Reduced response series
may be retained, but Gramian-only calculations can disable even that storage.
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from pde_cas.grid import Room2DFVGrid


def _real_array(values, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _profiles(grid: Room2DFVGrid, values, name: str) -> np.ndarray:
    array = _real_array(values, name)
    if array.ndim != 2 or array.shape[0] != grid.size or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (grid.size, positive number of profiles)")
    return array


def _generator(generator, grid: Room2DFVGrid) -> sparse.csr_matrix:
    if sparse.issparse(generator):
        matrix = sparse.csr_matrix(generator)
        _real_array(matrix.data, "generator")
        matrix = matrix.astype(float, copy=False)
    else:
        array = _real_array(generator, "generator")
        if array.ndim != 2:
            raise ValueError("generator must be a square matrix")
        matrix = sparse.csr_matrix(array)
    if matrix.shape != (grid.size, grid.size):
        raise ValueError("generator must have shape (grid.size, grid.size)")
    return matrix


def _times(values, *, quadrature: bool = False) -> np.ndarray:
    times = _real_array(values, "times")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be one-dimensional with at least two samples")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must start at zero and be strictly increasing")
    expected = np.linspace(0.0, times[-1], times.size)
    tolerance = 64 * np.finfo(float).eps * times[-1]
    if not np.allclose(times, expected, rtol=0.0, atol=tolerance):
        raise ValueError("times must be uniformly spaced")
    if quadrature and (times.size < 3 or times.size % 2 == 0):
        raise ValueError("Simpson quadrature requires an odd number of at least three samples")
    return times


def _chunk_size(value: int | None, num_times: int) -> int:
    if value is None:
        return num_times - 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("chunk_size must be a positive integer or None")
    return min(int(value), num_times - 1)


def target_matrix(grid: Room2DFVGrid, target_profiles: np.ndarray) -> np.ndarray:
    """Return the observation matrix C.T M, shape (targets, cells)."""
    return grid.cell_area * _profiles(grid, target_profiles, "target_profiles").T


def mass_adjoint(generator, grid: Room2DFVGrid) -> sparse.csr_matrix:
    """Return M^-1 A.T M = A.T for this uniform cell-volume mass matrix."""
    return _generator(generator, grid).T.tocsr()


def simpson_weights(times: np.ndarray) -> np.ndarray:
    """Global composite-Simpson weights, independent of propagation chunks."""
    times = _times(times, quadrature=True)
    weights = np.ones(times.size)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * (times[-1] / (times.size - 1) / 3.0)


def _state_chunks(
    generator: sparse.csr_matrix,
    initial: np.ndarray,
    times: np.ndarray,
    chunk_size: int,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield disjoint global sample slices and (time, cells, columns) states.

    Each exponential call includes its local initial state, but that shared
    endpoint is emitted only by the preceding chunk. Copying the final state
    avoids retaining a view of an old full block while propagating the next.
    """
    state = initial
    trace = float(generator.diagonal().sum())
    dt = times[-1] / (times.size - 1)
    yield slice(0, 1), state[None, :, :]
    for start in range(0, times.size - 1, chunk_size):
        steps = min(chunk_size, times.size - 1 - start)
        evolved = expm_multiply(
            generator, state, start=0.0, stop=steps * dt,
            num=steps + 1, endpoint=True, traceA=trace,
        )
        if not np.all(np.isfinite(evolved)):
            raise FloatingPointError("nonfinite propagated states")
        state = evolved[-1].copy()
        yield slice(start + 1, start + steps + 1), evolved[1:]
        del evolved


def _response_inputs(generator, grid, actuators, targets, times, method, chunk_size):
    if method not in ("forward", "adjoint"):
        raise ValueError("method must be 'forward' or 'adjoint'")
    matrix = _generator(generator, grid)
    b = _profiles(grid, actuators, "actuator_profiles")
    c = _profiles(grid, targets, "target_profiles")
    t = _times(times)
    size = _chunk_size(chunk_size, t.size)
    return matrix, b, c, t, size


def _response_chunks(matrix, grid, b, c, times, method, chunk_size):
    if method == "forward":
        observer = grid.cell_area * c.T
        for indices, states in _state_chunks(matrix, b, times, chunk_size):
            yield indices, np.einsum("rn,tnm->trm", observer, states, optimize=True)
    else:
        # Evolve C with the mass adjoint, then form <exp(t A^dagger) C, B>_M.
        weighted_b = grid.cell_area * b
        for indices, states in _state_chunks(matrix.T.tocsr(), c, times, chunk_size):
            yield indices, np.einsum("tnr,nm->trm", states, weighted_b, optimize=True)


def _response_series(generator, grid, actuators, targets, times, method, chunk_size):
    matrix, b, c, times, size = _response_inputs(
        generator, grid, actuators, targets, times, method, chunk_size,
    )
    responses = np.empty((times.size, c.shape[1], b.shape[1]))
    for indices, values in _response_chunks(matrix, grid, b, c, times, method, size):
        responses[indices] = values
    return responses


def response_series_forward(
    generator,
    grid: Room2DFVGrid,
    actuator_profiles: np.ndarray,
    target_profiles: np.ndarray,
    times: np.ndarray,
    *,
    chunk_size: int | None = 16,
) -> np.ndarray:
    """Return Q(t), shape (time, targets, actuators), by evolving B."""
    return _response_series(
        generator, grid, actuator_profiles, target_profiles, times, "forward", chunk_size,
    )


def response_series_adjoint(
    generator,
    grid: Room2DFVGrid,
    actuator_profiles: np.ndarray,
    target_profiles: np.ndarray,
    times: np.ndarray,
    *,
    chunk_size: int | None = 16,
) -> np.ndarray:
    """Return Q(t) by evolving C with the mass adjoint (usually fewer columns)."""
    return _response_series(
        generator, grid, actuator_profiles, target_profiles, times, "adjoint", chunk_size,
    )


def _gramian_increment(weights: np.ndarray, responses: np.ndarray) -> np.ndarray:
    return np.einsum("t,trm,tsm->mrs", weights, responses, responses, optimize=True)


def channel_gramians_from_response_series(times: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """Integrate q_i(t) q_i(t).T; return (actuators, targets, targets)."""
    weights = simpson_weights(times)
    values = _real_array(responses, "responses")
    if values.ndim != 3 or values.shape[0] != weights.size or min(values.shape[1:]) == 0:
        raise ValueError("responses must have shape (len(times), positive targets, positive actuators)")
    grams = _gramian_increment(weights, values)
    return 0.5 * (grams + grams.swapaxes(1, 2))


def build_room_2d_fv_channel_gramians(
    generator,
    grid: Room2DFVGrid,
    actuator_profiles: np.ndarray,
    target_profiles: np.ndarray,
    horizon: float,
    *,
    num_time_points: int = 401,
    method: str = "adjoint",
    chunk_size: int | None = 16,
    store_responses: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return (times, responses or None, per-channel target Gramians).

    Simpson weights use global sample indices, so odd chunk lengths, shared
    endpoints, and a short last chunk need no special quadrature fallback.
    Set store_responses=False to accumulate Gramians without storing Q(t).
    """
    if np.ndim(horizon) != 0 or np.iscomplexobj(horizon) or not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if (
        isinstance(num_time_points, bool) or not isinstance(num_time_points, Integral)
        or num_time_points < 3 or num_time_points % 2 == 0
    ):
        raise ValueError("num_time_points must be an odd integer of at least three")
    if not isinstance(store_responses, bool):
        raise ValueError("store_responses must be a bool")
    times = np.linspace(0.0, float(horizon), num_time_points)
    matrix, b, c, times, size = _response_inputs(
        generator, grid, actuator_profiles, target_profiles, times, method, chunk_size,
    )
    weights = simpson_weights(times)
    responses = np.empty((times.size, c.shape[1], b.shape[1])) if store_responses else None
    grams = np.zeros((b.shape[1], c.shape[1], c.shape[1]))
    for indices, values in _response_chunks(matrix, grid, b, c, times, method, size):
        grams += _gramian_increment(weights[indices], values)
        if responses is not None:
            responses[indices] = values
    return times, responses, 0.5 * (grams + grams.swapaxes(1, 2))


def whole_domain_channel_energies(
    generator,
    grid: Room2DFVGrid,
    actuator_profiles: np.ndarray,
    times: np.ndarray,
    *,
    chunk_size: int | None = 16,
) -> np.ndarray:
    """Return FH_i = integral ||exp(t A) b_i||_M^2 dt, shape (actuators,).

    Propagate one actuator at a time and accumulate global Simpson weights.
    No full-state history across the entire horizon is retained by default.
    """
    matrix = _generator(generator, grid)
    b = _profiles(grid, actuator_profiles, "actuator_profiles")
    times = _times(times, quadrature=True)
    size = _chunk_size(chunk_size, times.size)
    weights = simpson_weights(times)
    energies = np.zeros(b.shape[1])
    for channel in range(b.shape[1]):
        for indices, states in _state_chunks(matrix, b[:, channel:channel + 1], times, size):
            squared_norms = grid.cell_area * np.einsum("tnm,tnm->t", states, states)
            energies[channel] += weights[indices] @ squared_norms
    return energies
