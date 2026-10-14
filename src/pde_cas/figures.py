"""Regenerate the four paper figures from validated saved results."""

from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
from .dataio import DATA_DIR, load_data, write_json, digest, protect_inputs

STEMS = ("geometry", "gramian_eigenvalues", "velocity_allocations", "mesh_refinement")


def check_tex():
    missing = [
        name
        for name in ("pdflatex", "latex", "dvipng", "kpsewhich")
        if not shutil.which(name)
    ]
    if missing:
        raise RuntimeError(
            "Install the TeX prerequisites. Missing commands: " + ", ".join(missing)
        )
    for name in ("type1cm.sty", "type1ec.sty", "amsmath.sty", "amssymb.sty", "bm.sty"):
        if subprocess.run(
            ["kpsewhich", name], capture_output=True, text=True
        ).returncode:
            raise RuntimeError(f"Missing TeX package: {name}")


def render(data_dir: Path = DATA_DIR, output: Path = Path("outputs/figures")):
    protect_inputs(output, data_dir)
    reports, manifest = load_data(data_dir)
    check_tex()
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"
    os.environ["TZ"] = "UTC"
    from . import plotting as p
    from .refinement_plot import refinement_with_feasibility_figure

    p.apply_manuscript_style()
    builders = (
        p.geometry_figure,
        lambda: p.gramian_eigenvalues_figure(
            *p.characteristic_history(reports["characteristics.json"]),
            reports["apparent_feasibility.json"],
        ),
        lambda: p.velocity_figure(reports["velocity_pgd.json"], (383, 255)),
        lambda: refinement_with_feasibility_figure(
            reports["refinement_pgd.json"], reports["chi_refinement.json"]
        ),
    )
    for stem, build in zip(STEMS, builders, strict=True):
        p.save_figure(build(), stem, output)
    reference = manifest.get("reference_pdf_sha256", {})
    result = {
        "data_kind": manifest["kind"],
        "input_manifest_sha256": digest(data_dir / "manifest.json"),
        "files": {
            stem + ".pdf": {
                "sha256": digest(output / (stem + ".pdf")),
                "matches_reference_bytes": digest(output / (stem + ".pdf"))
                == reference[stem]
                if stem in reference
                else None,
            }
            for stem in STEMS
        },
    }
    write_json(output / "reproduction.json", result)
    return result
