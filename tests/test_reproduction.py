"""Public CLI, input protection, and adopted plotting invariants."""

import json
import shutil
import subprocess
import sys
import numpy as np
import pytest
from pde_cas import dataio


@pytest.fixture
def bundle(tmp_path):
    target = tmp_path / "data"
    shutil.copytree(dataio.DATA_DIR, target)
    return target


def test_all_adopted_inputs_load():
    data, manifest = dataio.load_data()
    assert set(data) == set(dataio.FILES)
    assert manifest["kind"] == "adopted-paper-data"


@pytest.mark.parametrize("name", dataio.FILES)
def test_tampered_input_is_rejected(bundle, name):
    p = bundle / name
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        dataio.load_data(bundle)


@pytest.mark.parametrize("name", dataio.FILES)
def test_missing_input_is_not_replaced_by_legacy_data(bundle, name):
    (bundle / name).unlink()
    with pytest.raises(FileNotFoundError):
        dataio.load_data(bundle)


@pytest.mark.parametrize(
    "name,key",
    [("velocity_pgd.json", "metadata"), ("refinement_pgd.json", "configuration")],
)
def test_changed_hash_cannot_disguise_old_solver(bundle, name, key):
    path = bundle / name
    report = json.loads(path.read_text())
    report[key]["optimizer"] = "SLSQP"
    dataio.write_json(path, report)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"][name]["sha256"] = dataio.digest(path)
    dataio.write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Not an adopted PGD"):
        dataio.load_data(bundle)


@pytest.mark.parametrize("kind", ["incomplete", "smoke", None])
def test_incomplete_manifest_is_rejected(bundle, kind):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["kind"] = kind
    dataio.write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        dataio.load_data(bundle)


def test_protect_adopted_data(tmp_path):
    for path in (dataio.DATA_DIR, dataio.DATA_DIR.parent, dataio.DATA_DIR / "figures"):
        with pytest.raises(ValueError, match="overlap"):
            dataio.protect_inputs(path)
    dataio.protect_inputs(tmp_path / "figures")


def test_protect_custom_data_and_symlink(bundle, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="overlap"):
        dataio.protect_inputs(link, bundle)
    with pytest.raises(ValueError, match="overlap"):
        dataio.protect_inputs(bundle.parent, bundle)


@pytest.mark.parametrize("command", [[], ["figures", "--help"], ["compute", "--help"]])
def test_english_help(command):
    if not command:
        command = ["--help"]
    result = subprocess.run(
        [sys.executable, "-m", "pde_cas.cli", *command], text=True, capture_output=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert not any("\u3040" <= c <= "\u9fff" for c in result.stdout)


def test_invalid_case_limit():
    from pde_cas.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["compute", "--limit", "0"])
    assert exc.value.code == 2


def test_missing_tex_is_explained(monkeypatch):
    from pde_cas.figures import check_tex

    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="TeX prerequisites"):
        check_tex()


def test_refinement_triptych_preserves_data_and_square_panels():
    import matplotlib.pyplot as plt
    from pde_cas.refinement_plot import refinement_with_feasibility_figure
    from pde_cas.plotting import aggregate_gramian_minimum_eigenvalues

    data, _ = dataio.load_data()
    ref = data["refinement_pgd.json"]
    chi = data["chi_refinement.json"]
    fig = refinement_with_feasibility_figure(ref, chi)
    try:
        assert len(fig.axes) == 3
        assert fig.axes[1].get_title() == "(b) Target controllability"
        np.testing.assert_array_equal(
            fig.axes[1].lines[0].get_ydata(), aggregate_gramian_minimum_eigenvalues(chi)
        )
        np.testing.assert_array_equal(
            fig.axes[2].lines[0].get_ydata(), [r["chi"] for r in chi["rows"]]
        )
        for line, key in zip(
            fig.axes[0].lines,
            (
                "response_l2_relative",
                "gramian_fro_relative",
                "vcs_allocation_linf",
                "aecs_allocation_linf",
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(
                line.get_ydata(), [r["to_reference"][key] for r in ref["rows"][:-1]]
            )
        for axis in fig.axes:
            width = axis.get_position().width * fig.get_figwidth()
            height = axis.get_position().height * fig.get_figheight()
            assert width == pytest.approx(height, abs=1e-12)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "change", ["shape", "complex", "nan", "asymmetric", "negative"]
)
def test_invalid_gramians_rejected(change):
    from pde_cas.plotting import aggregate_gramian_minimum_eigenvalues

    grams = np.repeat(np.eye(3)[None], 6, axis=0)
    if change == "shape":
        grams = grams[:5]
    elif change == "complex":
        grams = grams.astype(complex) + 1j
    elif change == "nan":
        grams[0, 0, 0] = np.nan
    elif change == "asymmetric":
        grams[0, 0, 1] = 1
    elif change == "negative":
        grams[0, 0, 0] = -1
    with pytest.raises(ValueError):
        aggregate_gramian_minimum_eigenvalues({"rows": [{"channel_gramians": grams}]})
