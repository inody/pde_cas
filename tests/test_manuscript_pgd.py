import numpy as np
import pytest

from pde_cas.pgd import (
    PGDSettings, TargetObjective, kkt_diagnostics, projected_trial,
    simplex_projection, solve_pgd,
)


def diagonal_problem():
    return np.array([np.diag([1., 0.]), np.diag([0., 4.])])


@pytest.mark.parametrize("criterion,expected", [("vcs", [.5, .5]), ("aecs", [2/3, 1/3])])
def test_closed_form_interior(criterion, expected):
    grams = diagonal_problem()
    settings = PGDSettings(epsilon_stop=2e-7, history_stride=1)
    result = solve_pgd(grams, criterion, settings, initial=np.array([.3, .7]))
    assert result["success"], result["reason"]
    p = np.array(result["allocation"])
    np.testing.assert_allclose(p, expected, atol=5e-8, rtol=0)
    _, grad = TargetObjective(grams, criterion).evaluate(p)
    alpha = result["accepted_step"]
    mapping = np.linalg.norm(p - projected_trial(p, grad, alpha)) / alpha
    assert mapping == result["gradient_mapping_norm_2"]
    assert mapping <= settings.epsilon_stop
    assert result["relative_kkt_residual"] < 2e-7
    assert all(r["armijo_margin"] >= 0 for r in result["history"])
    vals = [r["objective_value"] for r in result["history"]]
    assert np.all(np.diff(vals) <= 16 * np.finfo(float).eps * max(1, max(np.abs(vals))))
    assert all(r["objective_increment"] <= 0 for r in result["history"])


@pytest.mark.parametrize("criterion", ["vcs", "aecs"])
def test_boundary_optimum_and_inactive_inequalities(criterion):
    grams = np.array([[[1.]], [[4.]], [[2.]]])
    result = solve_pgd(grams, criterion, PGDSettings(epsilon_stop=1e-9))
    assert result["success"]
    np.testing.assert_allclose(result["allocation"], [0, 1, 0], atol=1e-12)
    assert result["inactive_kkt_violation"] == 0
    assert result["dual_gap"] == 0


@pytest.mark.parametrize("criterion", ["vcs", "aecs"])
def test_cholesky_gradient_and_values(criterion):
    rng = np.random.default_rng(72)
    roots = rng.normal(size=(6, 3, 3))
    grams = roots @ roots.swapaxes(1, 2)
    objective = TargetObjective(grams, criterion)
    p = np.full(6, 1/6)
    value, grad = objective.evaluate(p)
    g = np.einsum("i,ijk->jk", p, grams)
    inverse = np.linalg.inv(g)  # independent test only, not solver implementation
    expected = -np.linalg.slogdet(g)[1] if criterion == "vcs" else np.trace(inverse)
    assert value == pytest.approx(expected, rel=1e-13)
    exact_grad = (-np.einsum("ij,kji->k", inverse, grams) if criterion == "vcs"
                  else -np.einsum("ij,kjl,li->k", inverse, grams, inverse))
    np.testing.assert_allclose(grad, exact_grad, rtol=1e-12, atol=1e-13)
    for i in range(6):
        e = np.eye(6)[i] * 1e-6
        fd = (objective.evaluate(p+e)[0] - objective.evaluate(p-e)[0]) / 2e-6
        assert fd == pytest.approx(grad[i], rel=1e-7)


@pytest.mark.parametrize("criterion", ["vcs", "aecs"])
def test_stable_increment_is_same_objective_difference(criterion):
    objective = TargetObjective(diagonal_problem(), criterion)
    p = np.array([.3, .7])
    step = np.array([.01, -.01])
    direct = objective.evaluate(p+step)[0] - objective.evaluate(p)[0]
    assert objective.increment(p, step) == pytest.approx(direct, rel=1e-12, abs=1e-15)
    # At the exact VCS optimum the O(step^2) increase must not disappear when
    # direct subtraction of objective values rounds to zero.
    if criterion == "vcs":
        tiny = objective.increment(np.array([.5, .5]), np.array([1e-8, -1e-8]))
        assert tiny == pytest.approx(4e-16, rel=1e-7, abs=0)


def test_singular_trial_rejected_before_objective():
    objective = TargetObjective(diagonal_problem(), "vcs")
    assert objective.evaluate(np.array([1., 0.])) is None
    result = solve_pgd(diagonal_problem(), "vcs", PGDSettings(epsilon_stop=1e-6), initial=np.array([.1, .9]))
    assert result["success"]
    assert result["domain_rejections"] > 0


def test_line_search_failure_is_failure_without_kkt_fallback():
    result = solve_pgd(diagonal_problem(), "vcs", PGDSettings(max_backtracks=1), initial=np.array([.1, .9]))
    assert not result["success"]
    assert result["gradient_mapping_norm_2"] is None
    assert result["accepted_step"] is None
    assert result["domain_rejections"] == 1


def test_iteration_cap_retains_measured_residual():
    result = solve_pgd(diagonal_problem(), "aecs", PGDSettings(max_iterations=1, epsilon_stop=1e-12))
    assert not result["success"]
    assert result["iterations"] == 1
    assert result["gradient_mapping_norm_2"] > 0


def test_returns_current_not_trial_iterate():
    initial = np.array([.45, .55])
    result = solve_pgd(diagonal_problem(), "vcs", PGDSettings(epsilon_stop=1.), initial=initial)
    assert result["success"]
    assert result["iterations"] == 0
    np.testing.assert_array_equal(result["allocation"], initial)


def test_zero_step_underflow_is_not_success():
    result = solve_pgd(diagonal_problem(), "aecs", PGDSettings(alpha_bar=1e-30))
    assert not result["success"]
    assert "nonstationary" in result["reason"]


def test_initial_point_not_silently_projected():
    with pytest.raises(ValueError, match="simplex"):
        solve_pgd(diagonal_problem(), "vcs", initial=np.array([2., -1.]))
    with pytest.raises(ValueError, match="positive definite"):
        solve_pgd(diagonal_problem(), "vcs", initial=np.array([1., 0.]))


@pytest.mark.parametrize("field,value", [("beta", 1), ("armijo", 0), ("alpha_bar", np.inf), ("max_iterations", 1.5), ("history_stride", 0)])
def test_bad_settings(field, value):
    with pytest.raises(ValueError):
        PGDSettings(**{field: value})


def test_projection_translation_and_descent():
    z = np.array([.2, -.3, 1.])
    np.testing.assert_allclose(simplex_projection(z), simplex_projection(z + 5), atol=1e-15)
    p = np.full(3, 1/3)
    g = np.array([-1., 0., 2.])
    for alpha in [1., .5, .01]:
        trial = projected_trial(p, g, alpha)
        np.testing.assert_allclose(trial, simplex_projection(p-alpha*g), atol=1e-15)
        delta = trial-p
        assert g @ delta <= -(delta @ delta) / alpha + 1e-14


def test_kkt_diagnostic_detects_omitted_channel():
    d = kkt_diagnostics(np.array([1., 0.]), np.array([-1., -2.]))
    assert d["inactive_kkt_violation"] == 1
    assert d["dual_gap"] == 1


def test_rejects_indefinite_channel_instead_of_clipping():
    bad = diagonal_problem()
    bad[0, 1, 1] = -.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        TargetObjective(bad, "vcs")
