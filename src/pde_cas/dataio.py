"""Validate inputs and protect the bundled adopted results."""

from pathlib import Path
import hashlib
import json

DATA_DIR = Path(__file__).parent / "data"
FILES = (
    "velocity_pgd.json",
    "refinement_pgd.json",
    "chi_refinement.json",
    "characteristics.json",
    "apparent_feasibility.json",
)
SOLVER = "manuscript Algorithm 1, uniform-start PGD; no polishing"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def protect_inputs(output, data_dir=DATA_DIR):
    output = Path(output).resolve()
    for source in (DATA_DIR.resolve(), Path(data_dir).resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(
                "The output directory must not overlap an input-data directory."
            )


def load_data(directory=DATA_DIR):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") not in (
        "adopted-paper-data",
        "recomputed-paper-data",
    ):
        raise ValueError("Unsupported or incomplete data manifest.")
    if set(manifest["files"]) != set(FILES):
        raise ValueError("All five input files are required for figure generation.")
    reports = {}
    for name in FILES:
        path = directory / name
        if digest(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        reports[name] = json.loads(path.read_text())
        if reports[name].get("schema_version") != 1:
            raise ValueError(f"Unsupported data schema: {name}")
    for name, container in [
        ("velocity_pgd.json", "metadata"),
        ("refinement_pgd.json", "configuration"),
    ]:
        if reports[name][container].get("optimizer") != SOLVER:
            raise ValueError(f"Not an adopted PGD result: {name}")
    return reports, manifest
