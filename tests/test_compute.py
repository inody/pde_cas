"""Recomputation, checkpoint integrity, and publication-data assembly."""

import copy
import numpy as np
import pytest
from pde_cas import compute as c
from pde_cas.dataio import load_data, write_json


@pytest.fixture(scope="module")
def adopted():
    return load_data()[0]


def test_full_plan_has_all_plotted_cases(adopted):
    plan = c.make_plan(adopted)
    assert len(plan) == 56
    assert sum(p["group"] == "apparent" for p in plan) == 4
    assert sum(p["group"] == "velocity" for p in plan) == 47
    assert sum(p["group"] == "refinement" for p in plan) == 5
    assert plan[-1]["grid"] == [767, 511]
    assert plan[-1]["samples"] == 801
    assert len(c.make_plan(adopted, True)) == 3


@pytest.fixture(scope="module")
def small_case(tmp_path_factory, adopted):
    directory = tmp_path_factory.mktemp("real-refinement")
    spec = next(p for p in c.make_plan(adopted, True) if p["group"] == "refinement")
    return directory, spec, c.case(directory, spec)


def test_real_fv_and_pgd_match_adopted_coarse_mesh(small_case, adopted):
    _, _, r = small_case
    expected = adopted["chi_refinement.json"]["rows"][0]["channel_gramians"]
    np.testing.assert_allclose(r["grams"], expected, rtol=2e-11, atol=1e-17)
    for name in c.SETTINGS:
        result = r["scores"][name]
        assert result["success"]
        np.testing.assert_allclose(
            result["initial_allocation"], np.ones(6) / 6, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            result["allocation"],
            adopted["refinement_pgd.json"]["rows"][0]["scores"][name]["allocation"],
            rtol=0,
            atol=2e-10,
        )


def test_checkpoint_resume_skips_propagation_and_solver(small_case, monkeypatch):
    directory, spec, original = small_case

    def forbidden(*args, **kwargs):
        raise AssertionError("completed case recomputed")

    monkeypatch.setattr(c, "build_room_2d_fv_channel_gramians", forbidden)
    monkeypatch.setattr(c, "solve_pgd", forbidden)
    result = c.case(directory, spec)
    np.testing.assert_array_equal(result["grams"], original["grams"])


@pytest.mark.parametrize("suffix", [".npz", "_scores.json"])
def test_corrupt_checkpoint_rejected(small_case, tmp_path, suffix):
    import shutil

    directory, spec, _ = small_case
    target = tmp_path / "copy"
    shutil.copytree(directory, target)
    path = target / (spec["key"] + suffix)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        c.case(target, spec)


def test_solver_failure_keeps_gramian_for_retry(small_case, tmp_path, monkeypatch):
    import shutil

    directory, spec, _ = small_case
    target = tmp_path / "retry"
    shutil.copytree(directory, target)
    (target / (spec["key"] + "_scores.json")).unlink()
    original = c.solve_pgd
    monkeypatch.setattr(
        c,
        "solve_pgd",
        lambda *args, **kwargs: {"success": False, "reason": "injected failure"},
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        c.case(target, spec)
    assert (target / (spec["key"] + ".npz")).exists()
    assert not (target / (spec["key"] + "_scores.json")).exists()
    monkeypatch.setattr(c, "solve_pgd", original)
    monkeypatch.setattr(
        c,
        "build_room_2d_fv_channel_gramians",
        lambda *a, **k: pytest.fail("propagation repeated"),
    )
    result = c.case(target, spec)
    assert all(s["success"] for s in result["scores"].values())


def test_limit_resume_and_settings_guard(tmp_path, monkeypatch):
    calls = []

    def fake_case(directory, spec):
        calls.append(spec["key"])
        write_json(
            directory
            / (spec["key"] + ("_scores.json" if spec["optimize"] else ".json")),
            {},
        )
        return {"spec": spec, "scores": {}}

    monkeypatch.setattr(c, "case", fake_case)
    output = tmp_path / "run"
    first = c.run(output, smoke=True, limit=1)
    assert first["complete"] is False and len(calls) == 1
    assert not (output / "data/manifest.json").exists()
    second = c.run(output, resume=True, smoke=True)
    assert second["smoke"] and second["complete"] and not second["paper_reproduction"]
    assert not (output / "data/manifest.json").exists()
    with pytest.raises(ValueError, match="Resume requires"):
        c.run(output, resume=True, smoke=False)
    with pytest.raises(FileExistsError):
        c.run(output, smoke=True)


def test_assembly_uses_new_arrays_and_allocations(adopted, small_case, monkeypatch):
    _, _, base = small_case
    results = []
    for spec in c.make_plan(adopted):
        # Small synthetic arrays test assembly only; they are not written as paper results.
        r = copy.deepcopy(base)
        r["spec"] = spec
        r["times"] = np.linspace(0.0, spec["horizon"], spec["samples"])
        factor = 1.0 + 1.0 / spec["grid"][0]
        r["responses"] = np.ones((spec["samples"], 3, 6)) * factor
        r["grams"] = base["grams"] * factor
        results.append(r)
    import pde_cas.plotting as plotting

    monkeypatch.setattr(
        plotting,
        "characteristic_history",
        lambda summary: (np.array([0.0, 5.0, 12.0]), np.ones((3, 3))),
    )
    out = c.assemble(
        adopted, results, {"python": "test", "numpy": "test", "scipy": "test"}
    )
    ref = out["refinement_pgd.json"]
    chi = out["chi_refinement.json"]
    assert all(v == 0 for v in ref["rows"][-1]["to_reference"].values())
    assert ref["rows"][0]["to_reference"]["response_l2_relative"] > 0
    assert len(out["velocity_pgd.json"]["grids"][0]["rows"]) == 47
    assert len(out["apparent_feasibility.json"]["rows"]) == 4
    np.testing.assert_array_equal(
        chi["rows"][0]["channel_gramians"], results[-5]["grams"]
    )
    assert out["characteristics.json"]["uniform_gramian_eigenvalues_at_T5"] == [
        1.0,
        1.0,
        1.0,
    ]


def test_complete_manifest_written_only_after_all_cases(tmp_path, monkeypatch, adopted):
    from pde_cas.dataio import FILES

    calls = []

    def fake_case(directory, spec):
        calls.append(spec["key"])
        return {"spec": spec}

    def fake_assemble(data, results, software):
        assert len(results) == 56
        return copy.deepcopy(adopted)

    monkeypatch.setattr(c, "case", fake_case)
    monkeypatch.setattr(c, "assemble", fake_assemble)
    output = tmp_path / "complete"
    result = c.run(output)
    assert result["complete"] and len(calls) == 56
    reports, manifest = load_data(output / "data")
    assert manifest["kind"] == "recomputed-paper-data"
    assert set(reports) == set(FILES)
