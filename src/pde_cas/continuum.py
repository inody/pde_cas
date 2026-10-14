"""Characteristic reference for the zero-diffusion room problem.

For analytically L2-normalized compact profiles, the undamped response is
    Q_ji(t) = integral c_j(x) b_i(Phi_{-t}(x)) dx.
The analytic divergence-free flow preserves area, so no Jacobian factor is
needed. Target-support quadrature uses Gauss-Legendre in squared radius and
periodic trapezoidal angles. Fixed-step RK4 approximates the characteristics;
it is NOT itself exactly area- or stream-function-preserving. Refine substeps
and quadrature orders independently before treating results as a reference.

No FV sampling, support-band masks, trajectory projection, or artificial
response thresholds are used. Only reduced responses are retained over time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss

from pde_cas.geometry import (
    ROOM_FV_ACTUATOR_CENTERS,
    ROOM_FV_PROFILE_RADIUS,
    ROOM_FV_TARGET_CENTERS,
    ROOM_FV_VELOCITY_SCALE,
)


def _real_array(values, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _scalar(value, name: str, *, positive: bool = False) -> float:
    array = _real_array(value, name)
    if array.ndim != 0 or (positive and array <= 0.0):
        raise ValueError(f"{name} must be a scalar" + (" and positive" if positive else ""))
    return float(array)


def _order(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(values) -> np.ndarray:
    points = _real_array(values, "points")
    if points.ndim < 1 or points.shape[-1] != 2 or points.size == 0:
        raise ValueError("points must have nonempty shape (..., 2)")
    return points


@dataclass(frozen=True)
class Room2DFlow:
    length_x: float = 3.0
    length_y: float = 2.0
    velocity_scale: float = ROOM_FV_VELOCITY_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_x", _scalar(self.length_x, "length_x", positive=True))
        object.__setattr__(self, "length_y", _scalar(self.length_y, "length_y", positive=True))
        object.__setattr__(self, "velocity_scale", _scalar(self.velocity_scale, "velocity_scale"))


@dataclass(frozen=True)
class CompactBump:
    """Full-disk C2 polynomial bump, with exact continuum L2 normalization.

    beta = (1 - |x-center|^2/radius^2)_+^3 has squared norm pi radius^2/7.
    The reference response requires disks contained in the room; clipped disks
    would need a different normalization and are explicitly rejected there.
    """

    center: tuple[float, float]
    radius: float = ROOM_FV_PROFILE_RADIUS

    def __post_init__(self) -> None:
        center = _real_array(self.center, "center")
        if center.shape != (2,):
            raise ValueError("center must have two coordinates")
        object.__setattr__(self, "center", tuple(center))
        object.__setattr__(self, "radius", _scalar(self.radius, "radius", positive=True))

    @property
    def normalization(self) -> float:
        return float(np.sqrt(7.0 / np.pi) / self.radius)

    def values(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the normalized profile; outside support return exact zero."""
        points = _points(points)
        distance_squared = np.sum(((points - self.center) / self.radius) ** 2, axis=-1)
        return self.normalization * np.maximum(1.0 - distance_squared, 0.0) ** 3


def adopted_profiles() -> tuple[tuple[CompactBump, ...], tuple[CompactBump, ...]]:
    """Return candidate D's six actuators and three targets, radius 0.15."""
    return (
        tuple(CompactBump(center) for center in ROOM_FV_ACTUATOR_CENTERS),
        tuple(CompactBump(center) for center in ROOM_FV_TARGET_CENTERS),
    )


def stream_function(points: np.ndarray, flow: Room2DFlow | None = None) -> np.ndarray:
    """psi(x,y) = V0 sin(pi x/Lx) sin(pi y/Ly), shape points.shape[:-1]."""
    flow = flow if flow is not None else Room2DFlow()
    points = _points(points)
    result = (
        flow.velocity_scale * np.sin(np.pi * points[..., 0] / flow.length_x)
        * np.sin(np.pi * points[..., 1] / flow.length_y)
    )
    wall = (
        (points[..., 0] == 0.0) | (points[..., 0] == flow.length_x)
        | (points[..., 1] == 0.0) | (points[..., 1] == flow.length_y)
    )
    return np.where(wall, 0.0, result)


def _velocity(points: np.ndarray, flow: Room2DFlow) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    ax, ay = np.pi / flow.length_x, np.pi / flow.length_y
    vx = flow.velocity_scale * ay * np.sin(ax * x) * np.cos(ay * y)
    vy = -flow.velocity_scale * ax * np.cos(ax * x) * np.sin(ay * y)
    # Enforce analytic wall values, avoiding residual sin(pi) evaluations.
    vx = np.where((x == 0.0) | (x == flow.length_x), 0.0, vx)
    vy = np.where((y == 0.0) | (y == flow.length_y), 0.0, vy)
    return np.stack((vx, vy), axis=-1)


def velocity_field(points: np.ndarray, flow: Room2DFlow | None = None) -> np.ndarray:
    """Analytic v = (d_y psi, -d_x psi), with vectorized shape (..., 2)."""
    return _velocity(_points(points), flow if flow is not None else Room2DFlow())


def polar_quadrature(
    profile: CompactBump,
    *,
    radial_order: int = 24,
    angular_order: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disk nodes (radial_order*angular_order, 2) and area weights.

    With s=(rho/radius)^2, dx dy = radius^2/2 ds dtheta. Gauss-Legendre
    integrates s in [0,1]; angles 2*pi*k/angular_order exclude the repeated
    endpoint. Weights integrate area alone, not the target profile.
    """
    radial_order = _order(radial_order, "radial_order")
    angular_order = _order(angular_order, "angular_order")
    nodes, weights = leggauss(radial_order)
    radii = profile.radius * np.sqrt((nodes + 1.0) / 2.0)
    angles = 2.0 * np.pi * np.arange(angular_order) / angular_order
    unit_vectors = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    points = np.asarray(profile.center) + radii[:, None, None] * unit_vectors[None, :, :]
    area_weights = np.repeat(weights * np.pi * profile.radius**2 / (2 * angular_order), angular_order)
    return points.reshape(-1, 2), area_weights


def _times(values) -> np.ndarray:
    times = _real_array(values, "times")
    if times.ndim != 1 or times.size == 0 or times[0] != 0.0:
        raise ValueError("times must be one-dimensional, nonempty, and start at zero")
    if times.size > 1:
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        tolerance = 64 * np.finfo(float).eps * times[-1]
        if not np.allclose(times, np.linspace(0.0, times[-1], times.size), rtol=0.0, atol=tolerance):
            raise ValueError("times must be uniformly spaced")
    return times


def _in_room(points: np.ndarray, flow: Room2DFlow, tolerance: float = 0.0) -> bool:
    return bool(np.all(points >= -tolerance) and np.all(points <= np.array([flow.length_x, flow.length_y]) + tolerance))


def _characteristic_steps(
    points: np.ndarray,
    times: np.ndarray,
    flow: Room2DFlow,
    substeps: int,
    backward: bool,
) -> Iterator[np.ndarray]:
    state = points.copy()
    yield state
    if times.size == 1:
        return
    step = times[-1] / (times.size - 1) / substeps * (-1.0 if backward else 1.0)
    tolerance = 64 * np.finfo(float).eps * max(flow.length_x, flow.length_y)
    for _ in range(times.size - 1):
        for _ in range(substeps):
            k1 = _velocity(state, flow)
            k2 = _velocity(state + 0.5 * step * k1, flow)
            k3 = _velocity(state + 0.5 * step * k2, flow)
            k4 = _velocity(state + step * k3, flow)
            state = state + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(state)) or not _in_room(state, flow, tolerance):
                raise FloatingPointError("characteristic left the room or became nonfinite; increase substeps")
        yield state


def characteristics(
    points: np.ndarray,
    times: np.ndarray,
    *,
    flow: Room2DFlow | None = None,
    substeps: int = 4,
    backward: bool = False,
) -> np.ndarray:
    """Return Phi_{+/-t}(points) with shape (time, *points.shape).

    Use backward=True for Phi_{-t}. Output times are uniform and start at zero;
    substeps RK4 steps are taken per output interval. Unlike response_series,
    this diagnostic helper deliberately retains the full trajectory history.
    """
    points, times = _points(points), _times(times)
    flow = flow if flow is not None else Room2DFlow()
    substeps = _order(substeps, "substeps")
    if not isinstance(backward, bool):
        raise ValueError("backward must be a bool")
    if not _in_room(points, flow):
        raise ValueError("initial points must lie in the room")
    return np.stack(list(_characteristic_steps(points, times, flow, substeps, backward)))


def _profiles(profiles: Sequence[CompactBump], flow: Room2DFlow, name: str) -> tuple[CompactBump, ...]:
    profiles = tuple(profiles)
    if not profiles or any(not isinstance(profile, CompactBump) for profile in profiles):
        raise ValueError(f"{name} must be a nonempty sequence of CompactBump objects")
    for profile in profiles:
        center = np.asarray(profile.center)
        if not _in_room(np.stack((center - profile.radius, center + profile.radius)), flow):
            raise ValueError(f"{name} disks must be contained in the room for analytic normalization")
    return profiles


def response_series(
    times: np.ndarray,
    actuator_profiles: Sequence[CompactBump] | None = None,
    target_profiles: Sequence[CompactBump] | None = None,
    *,
    flow: Room2DFlow | None = None,
    substeps: int = 4,
    radial_order: int = 24,
    angular_order: int = 96,
    damping: float = 0.0,
) -> np.ndarray:
    """Backward-characteristic target responses, shape (time, targets, actuators).

    Defaults use the adopted compact geometry. The default damping=0 returns
    exactly the undamped integral described above; pass damping=gamma to
    multiply by exp(-gamma*t) when comparing with a damped FV generator.
    Quadrature weights remain attached to the original target points, while
    actuator values are sampled at their backward images. Only the current
    point cloud, not its entire time history, is stored during propagation.
    """
    flow = flow if flow is not None else Room2DFlow()
    times = _times(times)
    substeps = _order(substeps, "substeps")
    damping = _scalar(damping, "damping")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    default_b, default_c = adopted_profiles()
    actuators = _profiles(default_b if actuator_profiles is None else actuator_profiles, flow, "actuators")
    targets = _profiles(default_c if target_profiles is None else target_profiles, flow, "targets")
    quadratures = [polar_quadrature(c, radial_order=radial_order, angular_order=angular_order) for c in targets]
    points = np.stack([nodes for nodes, _ in quadratures])
    weighted_targets = np.stack([c.values(nodes) * weights for c, (nodes, weights) in zip(targets, quadratures)])
    responses = np.empty((times.size, len(targets), len(actuators)))
    for index, feet in enumerate(_characteristic_steps(points, times, flow, substeps, True)):
        for channel, actuator in enumerate(actuators):
            responses[index, :, channel] = np.sum(weighted_targets * actuator.values(feet), axis=1)
        responses[index] *= np.exp(-damping * times[index])
    return responses
