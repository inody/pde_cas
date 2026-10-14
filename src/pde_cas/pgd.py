"""Isolated implementation of manuscript Algorithm 1 (no solver fallback).

Only an accepted-step Euclidean gradient mapping can trigger convergence.
KKT diagnostics are independent checks, never an alternative stopping rule.
The historical optimizers and their cached numerical fingerprints are untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PGDSettings:
    alpha_bar: float = 1.0
    beta: float = 0.5
    armijo: float = 0.1
    epsilon_stop: float = 1e-8
    max_iterations: int = 100_000
    max_backtracks: int = 100
    history_stride: int = 100
    # A zero step with a large KKT violation indicates lost floating-point
    # resolution, not a mathematically stationary iterate.
    zero_step_kkt_guard: float = 1e-10

    def __post_init__(self):
        for name in ("alpha_bar", "epsilon_stop", "zero_step_kkt_guard"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta", "armijo"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0, 1)")
        for name in ("max_iterations", "max_backtracks", "history_stride"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


class TargetObjective:
    """Cholesky evaluations on the open positive-definite domain.

    No eigenvalue clipping, penalty objectives, or scaling of the objective.
    Trial points outside the domain return None *before* log/inverse evaluation.
    """
    def __init__(self, channel_gramians: np.ndarray, criterion: str):
        a = np.asarray(channel_gramians)
        if np.iscomplexobj(a):
            raise ValueError("real channel Gramians required")
        a = np.array(a, dtype=float, copy=True)
        if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] == 0 or a.shape[1] != a.shape[2]:
            raise ValueError("expected (m, r, r), m >= 2, r >= 1")
        if not np.isfinite(a).all() or criterion not in ("vcs", "aecs"):
            raise ValueError("invalid channel Gramians or criterion")
        for channel in a:
            scale = max(float(np.linalg.norm(channel, ord=2)), np.finfo(float).tiny)
            if np.max(np.abs(channel - channel.T)) > 64 * np.finfo(float).eps * scale:
                raise ValueError("channel Gramian is not symmetric to roundoff")
            if np.linalg.eigvalsh(channel)[0] < -64 * np.finfo(float).eps * scale:
                raise ValueError("channel Gramian is not positive semidefinite")
        self.grams = a
        self.criterion = criterion

    def evaluate(self, p: np.ndarray, *, gradient: bool = True):
        matrix = np.einsum("i,ijk->jk", p, self.grams)
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(factor).all():
            return None
        if self.criterion == "vcs":
            value = float(-2 * np.log(np.diag(factor)).sum())
        else:
            inverse_factor = np.linalg.solve(factor, np.eye(matrix.shape[0]))
            value = float(np.sum(inverse_factor ** 2))
        if not np.isfinite(value):
            return None
        if not gradient:
            return value
        # Z_i = L^{-1} G_i L^{-T}; no explicit allocated-Gramian inverse.
        left = np.linalg.solve(factor, self.grams)
        z = np.linalg.solve(factor, left.swapaxes(-1, -2)).swapaxes(-1, -2)
        if self.criterion == "vcs":
            grad = -np.trace(z, axis1=1, axis2=2)
        else:
            metric = inverse_factor @ inverse_factor.T
            grad = -np.einsum("ijk,kj->i", z, metric)
        if not np.isfinite(grad).all():
            return None
        return value, grad

    def increment(self, p: np.ndarray, step: np.ndarray):
        """Stable f(p+step)-f(p) on the simplex affine hull.

        Directly subtracting two O(1e4) AECS values loses the Armijo decrease
        close to a minimizer. With R=L^{-1} DeltaG L^{-T}, evaluate log1p(R)
        or (I+R)^{-1}-I spectrally instead. This is the same Armijo inequality,
        not an added tolerance. Centering channels removes roundoff in sum(step)
        (exactly zero for an exact simplex projection).
        """
        pivot = int(np.argmax(p))
        change = np.einsum("i,ijk->jk", step, self.grams - self.grams[pivot])
        matrix = np.einsum("i,ijk->jk", p, self.grams)
        factor = np.linalg.cholesky(matrix)
        left = np.linalg.solve(factor, change)
        relative = np.linalg.solve(factor, left.T).T
        eigenvalues, vectors = np.linalg.eigh((relative + relative.T) / 2)
        if np.min(eigenvalues) <= -1:
            return None
        if self.criterion == "vcs":
            return float(-np.log1p(eigenvalues).sum())
        inverse_factor = np.linalg.solve(factor, np.eye(matrix.shape[0]))
        weights = np.sum((vectors.T @ inverse_factor) ** 2, axis=1)
        return float(np.sum((-eigenvalues / (1 + eigenvalues)) * weights))


def simplex_projection(z: np.ndarray) -> np.ndarray:
    """Euclidean projection, using translation invariance for numerical stability."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all():
        raise ValueError("finite nonempty vector required")
    shifted = z - np.max(z)
    ordered = np.sort(shifted)[::-1]
    threshold = (np.cumsum(ordered) - 1) / np.arange(1, z.size + 1)
    active = np.flatnonzero(ordered > threshold)
    return np.maximum(shifted - threshold[active[-1]], 0.0)


def projected_trial(p: np.ndarray, gradient: np.ndarray, alpha: float) -> np.ndarray:
    # Adding a common constant to every projection argument changes no exact
    # simplex projection. Subtract it before alpha multiplication to avoid loss
    # of the small allocation offsets when the AECS gradient is large.
    return simplex_projection(p - alpha * (gradient - np.min(gradient)))


def kkt_diagnostics(p: np.ndarray, gradient: np.ndarray) -> dict:
    """Check all inequalities, plus a support-free Frank--Wolfe dual gap.

    These are floating-point diagnostics, not interval-certified bounds.
    The active set uses strictly positive weights; no small weight is discarded.
    """
    active = p > 0
    multiplier = float(np.mean(gradient[active]))
    deviation = float(np.max(np.abs(gradient[active] - multiplier)))
    violation = float(np.max(np.maximum(multiplier - gradient[~active], 0))) if (~active).any() else 0.0
    simplex = float(abs(p.sum() - 1))
    nonnegative = float(max(0, -p.min()))
    residual = max(deviation, violation, simplex, nonnegative)
    scale = max(1.0, float(np.linalg.norm(gradient, ord=np.inf)))
    gap = float(p @ (gradient - np.min(gradient)))
    return {"active_set": (np.flatnonzero(active) + 1).tolist(),
            "active_gradient_deviation": deviation, "inactive_kkt_violation": violation,
            "simplex_error": simplex, "nonnegativity_violation": nonnegative,
            "kkt_residual": residual, "relative_kkt_residual": residual / scale,
            "dual_gap": gap, "relative_dual_gap": gap / scale}


def solve_pgd(channel_gramians: np.ndarray, criterion: str,
              settings: PGDSettings | None = None, initial: np.ndarray | None = None) -> dict:
    settings = settings or PGDSettings()
    objective = TargetObjective(channel_gramians, criterion)
    m = objective.grams.shape[0]
    p = np.full(m, 1 / m) if initial is None else np.array(initial, dtype=float, copy=True)
    if (p.shape != (m,) or not np.isfinite(p).all() or np.any(p < 0)
            or abs(p.sum() - 1) > 64 * np.finfo(float).eps):
        raise ValueError("initial allocation must already lie on the simplex")
    evaluation = objective.evaluate(p)
    if evaluation is None:
        raise ValueError("initial allocated Gramian is not positive definite")
    start = p.copy()
    history = []
    backtracks_total = 0
    domain_rejections = 0
    alpha = None
    mapping = None
    # Explicitly record updates, not objective/line-search evaluations.
    updates = 0
    value, gradient = evaluation

    def finish(success: bool, reason: str):
        gramian = np.einsum("i,ijk->jk", p, objective.grams)
        return {"method": "manuscript Algorithm 1: domain-aware Armijo PGD",
                "criterion": criterion, "settings": asdict(settings),
                "initial_allocation": start.tolist(), "allocation": p.tolist(),
                "objective_value": value, "success": success, "reason": reason,
                "iterations": updates, "accepted_step": alpha,
                "gradient_mapping_norm_2": mapping,
                "backtracks": backtracks_total, "domain_rejections": domain_rejections,
                "lambda_min": float(np.linalg.eigvalsh(gramian)[0]),
                **kkt_diagnostics(p, gradient), "history": history}

    for iteration in range(settings.max_iterations + 1):
        # The last iteration evaluates the actual residual even at the update cap.
        alpha = settings.alpha_bar
        accepted = False
        mapping = None
        for backtrack in range(settings.max_backtracks):
            try:
                trial = projected_trial(p, gradient, alpha)
            except ValueError:
                return finish(False, "nonfinite projection argument")
            step = trial - p
            trial_value = objective.evaluate(trial, gradient=False)
            # Exact projected steps sum to zero. Remove the common gradient
            # component to avoid amplifying their floating-point sum error.
            slope = float((gradient - gradient[np.argmax(p)]) @ step)
            increment = None if trial_value is None else objective.increment(p, step)
            if increment is None:
                domain_rejections += 1
            elif increment <= settings.armijo * slope:
                accepted = True
                mapping = float(np.linalg.norm(step) / alpha)
                break
            alpha *= settings.beta
            backtracks_total += 1
            if alpha == 0:
                break
        if not accepted:
            alpha = None
            return finish(False, "line search exhausted; not convergence")
        record = {"iteration": iteration, "objective_value": value,
                  "allocation": p.tolist(), "accepted_step": alpha,
                  "gradient_mapping_norm_2": mapping, "backtracks": backtrack,
                  "objective_increment": increment,
                  "armijo_margin": float(settings.armijo * slope - increment)}
        if iteration % settings.history_stride == 0 or mapping <= settings.epsilon_stop or iteration == settings.max_iterations:
            history.append(record)
        if mapping == 0 and kkt_diagnostics(p, gradient)["relative_kkt_residual"] > settings.zero_step_kkt_guard:
            return finish(False, "zero projected step with nonstationary KKT diagnostic")
        if mapping <= settings.epsilon_stop:
            # Algorithm 1 returns p^(k), not the trial point.
            return finish(True, "accepted-step Euclidean mapping tolerance reached")
        if iteration == settings.max_iterations:
            return finish(False, "maximum iterations reached")
        p = trial
        updates += 1
        evaluation = objective.evaluate(p)
        if evaluation is None:
            raise FloatingPointError("accepted positive-definite trial lost its domain on reevaluation")
        value, gradient = evaluation
    raise AssertionError("unreachable")
