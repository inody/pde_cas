from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import dblquad, solve_ivp

from pde_cas.continuum import (
    CompactBump,
    Room2DFlow,
    adopted_profiles,
    characteristics,
    polar_quadrature,
    response_series,
    stream_function,
    velocity_field,
)


@pytest.mark.parametrize("radius", [0.07, 0.15, 0.4])
def test_analytic_continuum_normalization_and_polar_area(radius):
    bump = CompactBump((1.1, 0.8), radius)
    nodes, weights = polar_quadrature(bump, radial_order=4, angular_order=12)
    assert nodes.shape == (48, 2)
    assert weights.shape == (48,)
    assert np.all(weights > 0.0)
    assert_allclose(weights.sum(), np.pi * radius**2, rtol=2e-15)
    assert_allclose(weights @ bump.values(nodes)**2, 1.0, atol=4e-15)
    assert_allclose(bump.normalization, np.sqrt(7 / (np.pi * radius**2)), rtol=2e-15)
    assert_allclose(weights @ bump.values(nodes), bump.normalization * np.pi * radius**2 / 4, atol=2e-15)
    # Area moments verify the Jacobian and periodic angular weights separately.
    centered = nodes - bump.center
    assert_allclose(weights @ centered, 0.0, atol=2e-16)
    assert_allclose(weights @ centered**2, np.full(2, np.pi * radius**4 / 4), atol=2e-16)
    assert_allclose(bump.values(np.array(bump.center)), bump.normalization)
    assert_array_equal(bump.values(np.array([[0.0, 0.0], [3.0, 2.0]])), 0.0)


def test_adopted_profiles_and_exact_initial_overlap():
    actuators, targets = adopted_profiles()
    assert len(actuators) == 6
    assert len(targets) == 3
    assert actuators[0].center == (0.34, 0.26)
    assert actuators[-1].center == (2.64, 1.74)
    assert targets[2].center == (2.40, 0.65)
    assert all(p.radius == 0.15 for p in (*actuators, *targets))
    q = response_series([0.0], radial_order=4, angular_order=12)
    assert q.shape == (1, 3, 6)
    assert_array_equal(q, 0.0)  # Adopted actuator and target disks are disjoint.
    for profiles in (actuators, targets):
        overlap = response_series([0.0], profiles, profiles, radial_order=4, angular_order=12)
        assert_allclose(overlap[0], np.eye(len(profiles)), atol=8e-15)


def test_initial_partial_overlap_against_independent_cartesian_integral():
    b, c = CompactBump((1.24, 0.92), 0.32), CompactBump((1.50, 0.80), 0.32)

    def value(x, y, bump):
        distance2 = ((x - bump.center[0])**2 + (y - bump.center[1])**2) / bump.radius**2
        return bump.normalization * max(1.0 - distance2, 0.0)**3

    def half_height(x):
        return np.sqrt(max(c.radius**2 - (x - c.center[0])**2, 0.0))

    reference, error = dblquad(
        lambda y, x: value(x, y, b) * value(x, y, c),
        c.center[0] - c.radius, c.center[0] + c.radius,
        lambda x: c.center[1] - half_height(x), lambda x: c.center[1] + half_height(x),
        epsabs=2e-11, epsrel=2e-11,
    )
    actual = response_series([0.0], [b], [c], radial_order=64, angular_order=256)
    assert 0.1 < reference < 0.3
    assert error < 1e-9
    assert_allclose(actual[0, 0, 0], reference, atol=3e-9, rtol=0.0)
    reverse = response_series([0.0], [c], [b], radial_order=64, angular_order=256)
    assert_allclose(reverse, actual, atol=3e-9, rtol=0.0)


def test_analytic_velocity_stream_function_and_wall_flux():
    flow = Room2DFlow(length_x=2.7, length_y=1.1, velocity_scale=-0.8)
    points = np.array([[0.4, 0.2], [1.1, 0.8], [2.2, 0.6]])
    x, y = points.T
    ax, ay = np.pi / flow.length_x, np.pi / flow.length_y
    gradient = flow.velocity_scale * np.column_stack((
        ax * np.cos(ax * x) * np.sin(ay * y), ay * np.sin(ax * x) * np.cos(ay * y),
    ))
    velocity = velocity_field(points, flow)
    assert_allclose(velocity, np.column_stack((gradient[:, 1], -gradient[:, 0])), atol=2e-15)
    assert_allclose(np.sum(gradient * velocity, axis=1), 0.0, atol=2e-15)
    assert_allclose(stream_function(points, flow), flow.velocity_scale * np.sin(ax * x) * np.sin(ay * y))
    walls = np.array([[0.0, 0.3], [flow.length_x, 0.3], [0.4, 0.0], [0.4, flow.length_y]])
    assert_array_equal(velocity_field(walls, flow)[[0, 1], 0], 0.0)
    assert_array_equal(velocity_field(walls, flow)[[2, 3], 1], 0.0)
    assert_array_equal(stream_function(walls, flow), 0.0)


@pytest.mark.parametrize("backward", [False, True])
def test_stream_function_invariance_and_forward_backward_reversibility(backward):
    flow = Room2DFlow(velocity_scale=0.8)
    points = np.array([[0.4, 0.35], [0.9, 1.3], [1.6, 0.6], [2.6, 1.6]])
    times = np.linspace(0.0, 2.0, 9)
    path = characteristics(points, times, flow=flow, substeps=16, backward=backward)
    assert path.shape == (9, 4, 2)
    assert_array_equal(path[0], points)
    assert np.max(np.abs(stream_function(path, flow) - stream_function(points, flow))) < 3e-10
    returned = characteristics(path[-1], times, flow=flow, substeps=16, backward=not backward)
    assert_allclose(returned[-1], points, rtol=0.0, atol=1e-10)
    # All leading point dimensions are vectorized, not treated as coordinates.
    shaped = characteristics(points.reshape(2, 2, 2), times, flow=flow, substeps=16, backward=backward)
    assert_allclose(shaped.reshape(path.shape), path, atol=2e-15)


def _enclosing_stream_band(profile, flow):
    # Exact bounds over the enclosing rectangle; therefore rigorous (possibly
    # loose) bounds over the disk. The room sine factors are nonnegative.
    lower, upper = [], []
    for center, length in zip(profile.center, (flow.length_x, flow.length_y)):
        lo, hi = center - profile.radius, center + profile.radius
        endpoints = np.sin(np.pi * np.array([lo, hi]) / length)
        lower.append(min(endpoints))
        upper.append(1.0 if lo <= length / 2 <= hi else max(endpoints))
    return flow.velocity_scale * np.prod(lower), flow.velocity_scale * np.prod(upper)


def test_disjoint_stream_function_bands_give_exact_zero_without_thresholding():
    flow = Room2DFlow()
    actuators, targets = adopted_profiles()
    isolated = tuple(actuators[i] for i in (0, 2, 3, 5))
    actuator_upper = max(_enclosing_stream_band(p, flow)[1] for p in isolated)
    target_lower = min(_enclosing_stream_band(p, flow)[0] for p in targets[:2])
    assert target_lower - actuator_upper > 0.016
    times = np.linspace(0.0, 12.0, 25)
    q = response_series(times, isolated, targets[:2], flow=flow, substeps=8, radial_order=8, angular_order=24)
    assert_array_equal(q, 0.0)
    # Check invariant drift is small compared to the proven support-band gap.
    nodes, _ = polar_quadrature(targets[0], radial_order=4, angular_order=12)
    feet = characteristics(nodes, times, flow=flow, substeps=8, backward=True)
    assert np.max(np.abs(stream_function(feet, flow) - stream_function(nodes, flow))) < 1e-9


@pytest.mark.parametrize("backward", [False, True])
def test_rk4_characteristic_temporal_convergence(backward):
    flow = Room2DFlow(velocity_scale=0.8)
    points = np.array([[0.4, 0.35], [0.9, 1.3], [1.6, 0.6], [2.6, 1.6]])
    times = np.linspace(0.0, 2.0, 9)
    sign = -1 if backward else 1
    independent = solve_ivp(
        lambda t, state: sign * velocity_field(state.reshape(-1, 2), flow).ravel(),
        (0.0, times[-1]), points.ravel(), method="DOP853", rtol=3e-14, atol=3e-15,
    )
    assert independent.success
    reference = independent.y[:, -1].reshape(points.shape)
    errors = []
    for substeps in (1, 2, 4, 8):
        actual = characteristics(points, times, flow=flow, substeps=substeps, backward=backward)[-1]
        errors.append(np.linalg.norm(actual - reference))
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((rates > 3.7) & (rates < 4.3))
    assert errors[-1] < 1e-7


@pytest.fixture
def overlap_problem():
    return (
        np.linspace(0.0, 0.8, 5), (CompactBump((1.24, 0.92), 0.32),),
        (CompactBump((1.50, 0.80), 0.32),), Room2DFlow(velocity_scale=0.8),
    )


def test_response_rk4_convergence_with_quadrature_held_fixed(overlap_problem):
    times, b, c, flow = overlap_problem
    kwargs = dict(flow=flow, radial_order=12, angular_order=48)
    reference = response_series(times, b, c, substeps=64, **kwargs)
    errors = [np.max(np.abs(response_series(times, b, c, substeps=s, **kwargs) - reference)) for s in (1, 2, 4, 8)]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((rates > 3.7) & (rates < 4.3))
    # This geometry has increasing overlap under forward transport; reversing
    # the characteristic direction would not reproduce this response.
    assert reference[-1, 0, 0] > 0.9
    assert reference[0, 0, 0] < 0.2


def test_radial_and_angular_quadrature_refine_independently(overlap_problem):
    times, b, c, flow = overlap_problem
    kwargs = dict(flow=flow, substeps=32)
    reference = response_series(times, b, c, radial_order=64, angular_order=256, **kwargs)
    finer = response_series(times, b, c, radial_order=96, angular_order=384, **kwargs)
    assert_allclose(reference, finer, rtol=0.0, atol=3e-9)
    for orders in (((6, 256), (12, 256), (24, 256)), ((64, 12), (64, 24), (64, 48))):
        errors = []
        for radial, angular in orders:
            actual = response_series(times, b, c, radial_order=radial, angular_order=angular, **kwargs)
            errors.append(np.max(np.abs(actual - reference)))
        assert errors[1] < errors[0] / 8
        assert errors[2] < errors[1] / 8
        assert errors[-1] < 3e-7


def test_zero_velocity_stationary_profiles_and_optional_damping():
    profile = CompactBump((1.5, 1.0))
    times = np.linspace(0.0, 3.0, 7)
    kwargs = dict(flow=Room2DFlow(velocity_scale=0.0), radial_order=4, angular_order=8)
    stationary = response_series(times, [profile], [profile], **kwargs)
    assert_allclose(stationary, 1.0, atol=3e-15)
    damped = response_series(times, [profile], [profile], damping=0.2, **kwargs)
    assert_allclose(damped[:, 0, 0], np.exp(-0.2 * times), atol=3e-15)
    points = np.array([1.4, 1.1])
    path = characteristics(points, times, flow=kwargs["flow"])
    assert path.shape == (7, 2)
    assert_array_equal(path, np.broadcast_to(points, path.shape))


@pytest.mark.parametrize("kwargs", [
    {"center": (np.nan, 0.2)}, {"center": (0.5,)}, {"center": (0.5, 1j)},
    {"radius": 0.0}, {"radius": -0.1}, {"radius": np.inf}, {"radius": [0.1]},
])
def test_invalid_bump_parameters(kwargs):
    with pytest.raises(ValueError):
        CompactBump(**{"center": (0.5, 0.5), **kwargs})


@pytest.mark.parametrize("kwargs", [
    {"length_x": 0.0}, {"length_y": -1.0}, {"length_x": np.nan},
    {"velocity_scale": np.inf}, {"velocity_scale": 1j}, {"length_y": [2.0]},
])
def test_invalid_flow_parameters(kwargs):
    with pytest.raises(ValueError):
        Room2DFlow(**kwargs)


@pytest.mark.parametrize("times", [
    [], [0.1], [[0.0, 1.0]], [0.0, 0.0], [0.0, -1.0], [0.0, 0.2, 1.0],
    [0.0, 1e-12, 3e-12], [0.0, np.nan], [0.0, np.inf], [0.0, 1j],
])
def test_invalid_output_times(times):
    with pytest.raises(ValueError):
        characteristics(np.array([1.0, 1.0]), times)
    with pytest.raises(ValueError):
        response_series(times)


@pytest.mark.parametrize("field,value", [
    ("substeps", 0), ("substeps", 1.5), ("substeps", True),
    ("radial_order", 0), ("radial_order", 2.5), ("radial_order", True),
    ("angular_order", -1), ("angular_order", 2.5), ("angular_order", True),
    ("damping", -1.0), ("damping", np.nan),
])
def test_invalid_response_controls(field, value):
    with pytest.raises(ValueError):
        response_series([0.0, 0.1], **{field: value})


@pytest.mark.parametrize("profiles", [[], [(0.5, 0.5)], [CompactBump((0.1, 0.5), 0.2)]])
@pytest.mark.parametrize("kind", ["actuator_profiles", "target_profiles"])
def test_response_rejects_missing_invalid_or_wall_clipped_profiles(profiles, kind):
    with pytest.raises(ValueError):
        response_series([0.0], **{kind: profiles})


@pytest.mark.parametrize("points", [[], [1.0], [[1.0, 1.0, 1.0]], [np.nan, 0.5], [1j, 0.5]])
def test_invalid_point_shapes_and_values(points):
    for evaluator in (velocity_field, stream_function, CompactBump((1.0, 1.0)).values):
        with pytest.raises(ValueError):
            evaluator(points)
    with pytest.raises(ValueError):
        characteristics(points, [0.0, 0.1])


def test_characteristic_controls_and_no_silent_domain_clipping():
    for substeps in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            characteristics([1.0, 1.0], [0.0, 0.1], substeps=substeps)
    with pytest.raises(ValueError):
        characteristics([1.0, 1.0], [0.0, 0.1], backward=1)
    with pytest.raises(ValueError):
        characteristics([-0.1, 1.0], [0.0, 0.1])
    with pytest.raises(FloatingPointError, match="increase substeps"):
        characteristics([0.5, 0.5], [0.0, 10.0], flow=Room2DFlow(velocity_scale=10.0), substeps=1)
