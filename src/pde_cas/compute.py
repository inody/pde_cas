"""Recompute FV responses and uniform-start PGD without modifying adopted data."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import json
import platform
import time
import numpy as np
import scipy
from scipy.linalg import null_space
from .dataio import (
    DATA_DIR,
    FILES,
    SOLVER,
    digest,
    load_data,
    protect_inputs,
    write_json,
)
from .benchmark import benchmark_profiles
from .grid import Room2DFVGrid
from .operator import Room2DFVParameters, build_room_2d_fv_operator
from .response import build_room_2d_fv_channel_gramians, simpson_weights
from .pgd import PGDSettings, solve_pgd

SETTINGS = {
    "vcs": PGDSettings(epsilon_stop=1e-12),
    "aecs": PGDSettings(epsilon_stop=1e-8),
}


def source_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def make_plan(data, smoke=False):
    apparent = data["apparent_feasibility.json"]
    velocity = data["velocity_pgd.json"]
    refinement = data["refinement_pgd.json"]
    plan = []
    for row in apparent["rows"][:1] if smoke else apparent["rows"]:
        plan.append(
            {
                "key": f"apparent_{row['num_x']}x{row['num_y']}",
                "group": "apparent",
                "grid": [row["num_x"], row["num_y"]],
                "diffusivity": 0.0,
                "velocity": 0.15,
                "horizon": 5.0,
                "samples": 501,
                "optimize": False,
            }
        )
    nodes = [0.15] if smoke else velocity["velocities"]
    for v in nodes:
        grid = [47, 31] if smoke else [383, 255]
        plan.append(
            {
                "key": f"velocity_{v:.14f}",
                "group": "velocity",
                "grid": grid,
                "diffusivity": 0.03,
                "velocity": v,
                "horizon": 6.0,
                "samples": 401,
                "optimize": True,
            }
        )
    for row in refinement["rows"][:1] if smoke else refinement["rows"]:
        plan.append(
            {
                "key": f"refinement_{row['num_x']}x{row['num_y']}",
                "group": "refinement",
                "grid": [row["num_x"], row["num_y"]],
                "diffusivity": 0.03,
                "velocity": 0.20,
                "horizon": 6.0,
                "samples": 801,
                "optimize": True,
            }
        )
    return plan


def validate_arrays(spec, times, responses, grams):
    if times.shape != (spec["samples"],) or not np.array_equal(
        times, np.linspace(0.0, spec["horizon"], spec["samples"])
    ):
        raise ValueError("Invalid checkpoint time grid.")
    if responses.shape != (spec["samples"], 3, 6) or grams.shape != (6, 3, 3):
        raise ValueError("Invalid checkpoint array shapes.")
    if not all(
        np.isfinite(x).all() and np.isrealobj(x) for x in (times, responses, grams)
    ):
        raise ValueError("Checkpoint arrays must be finite and real.")


def case(directory, spec):
    """Save Gramians before optimization; never switch solvers on failure."""
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / (spec["key"] + ".npz")
    record_path = directory / (spec["key"] + ".json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["case"] != spec or digest(array_path) != record["array_sha256"]:
            raise ValueError("Checkpoint parameters or hashes do not match.")
        with np.load(array_path, allow_pickle=False) as a:
            times, responses, grams = (
                a[k].copy() for k in ("times", "responses", "grams")
            )
    else:
        print(
            f"Computing: {spec['key']} / {spec['grid']} / {spec['samples']} time samples",
            flush=True,
        )
        grid = Room2DFVGrid(num_x=spec["grid"][0], num_y=spec["grid"][1])
        profiles = benchmark_profiles(grid, quadrature_order=6)
        op = build_room_2d_fv_operator(
            grid,
            Room2DFVParameters(
                diffusivity=spec["diffusivity"],
                velocity_scale=spec["velocity"],
                damping=0.02,
            ),
        )
        start = time.perf_counter()
        times, responses, grams = build_room_2d_fv_channel_gramians(
            op.generator,
            grid,
            *profiles,
            spec["horizon"],
            num_time_points=spec["samples"],
            method="adjoint",
            chunk_size=16,
            store_responses=True,
        )
        validate_arrays(spec, times, responses, grams)
        tmp = array_path.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, times=times, responses=responses, grams=grams)
        tmp.replace(array_path)
        record = {
            "case": spec,
            "array_sha256": digest(array_path),
            "propagation_seconds": time.perf_counter() - start,
        }
        write_json(record_path, record)
    validate_arrays(spec, times, responses, grams)
    scores = {}
    if spec["optimize"]:
        score_path = directory / (spec["key"] + "_scores.json")
        if score_path.exists():
            if digest(score_path) != record.get("scores_sha256"):
                raise ValueError(
                    "PGD checkpoint hash mismatch; do not reuse modified or incomplete results."
                )
            stored = json.loads(score_path.read_text())
            if stored["array_sha256"] != digest(array_path):
                raise ValueError("The PGD checkpoint does not match the Gramian.")
            scores = stored["scores"]
            for name, setting in SETTINGS.items():
                if (
                    scores[name]["settings"] != asdict(setting)
                    or not scores[name]["success"]
                ):
                    raise ValueError("Invalid PGD checkpoint.")
        else:
            for name, settings in SETTINGS.items():
                scores[name] = solve_pgd(grams, name, settings)
                if not scores[name]["success"]:
                    write_json(directory / (spec["key"] + "_failure.json"), scores)
                    raise RuntimeError(
                        f"PGD did not converge: {spec['key']} {name}: {scores[name]['reason']}"
                    )
            write_json(
                score_path, {"array_sha256": digest(array_path), "scores": scores}
            )
            record["scores_sha256"] = digest(score_path)
            write_json(record_path, record)
    return {
        "spec": spec,
        "times": times,
        "responses": responses,
        "grams": grams,
        "scores": scores,
    }


def compact_score(score):
    result = {
        k: score[k]
        for k in (
            "allocation",
            "objective_value",
            "success",
            "relative_kkt_residual",
            "simplex_error",
            "nonnegativity_violation",
            "gradient_mapping_norm_2",
            "iterations",
        )
    }
    result["active_set"] = (
        np.flatnonzero(np.asarray(score["allocation"]) > 1e-6) + 1
    ).tolist()
    result["optimization_method"] = score["method"]
    return result


def assemble(data, results, software):
    """Assemble plotting reports from new results, not adopted comparison values."""
    by_group = {
        g: [r for r in results if r["spec"]["group"] == g]
        for g in ("apparent", "velocity", "refinement")
    }
    apparent = {
        k: v for k, v in data["apparent_feasibility.json"].items() if k != "rows"
    }
    apparent["rows"] = []
    for r in by_group["apparent"]:
        nx, ny = r["spec"]["grid"]
        values = np.linalg.eigvalsh(r["grams"].mean(axis=0))
        apparent["rows"].append(
            {
                "num_x": nx,
                "num_y": ny,
                "state_dimension": nx * ny,
                "uniform_gramian_eigenvalues": values.tolist(),
                "practical_pd_threshold": 1e-14,
            }
        )
    velocity = {
        "schema_version": 1,
        "kind": "fv_adaptive_velocity_sweep",
        "metadata": copy.deepcopy(data["velocity_pgd.json"]["metadata"]),
        "velocities": [r["spec"]["velocity"] for r in by_group["velocity"]],
    }
    velocity["metadata"].pop("legacy_propagation_dependency_sha256", None)
    velocity["metadata"].update(
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        pgd_source_sha256=digest(Path(__file__).with_name("pgd.py")),
    )
    velocity["sampling_note"] = (
        "Recomputed at the 47 adopted velocity samples; adaptive sampling and 22 midpoint probes are not rerun."
    )
    velocity["grids"] = [
        {
            "num_x": 383,
            "num_y": 255,
            "role": "final_curve",
            "velocities": velocity["velocities"],
            "rows": [
                {
                    "velocity_scale": r["spec"]["velocity"],
                    **{k: compact_score(v) for k, v in r["scores"].items()},
                }
                for r in by_group["velocity"]
            ],
        }
    ]
    cfg = copy.deepcopy(data["refinement_pgd.json"]["configuration"])
    cfg.pop("legacy_propagation_dependency_sha256", None)
    cfg.update(
        optimizer=SOLVER,
        software=software,
        pgd_source_sha256=digest(Path(__file__).with_name("pgd.py")),
    )
    refinement = {
        "schema_version": 1,
        "configuration": cfg,
        "reference_grid": by_group["refinement"][-1]["spec"]["grid"],
        "rows": [],
    }
    chi = {
        "schema_version": 1,
        "kind": "fv_restricted_separation_mesh_sweep",
        "complete": True,
        "configuration": copy.deepcopy(cfg),
        "rows": [],
    }
    ref = by_group["refinement"][-1]
    weights = simpson_weights(ref["times"])
    ref_norm = float(np.sqrt(weights @ np.sum(ref["responses"] ** 2, axis=(1, 2))))
    directions = null_space(np.ones((1, 6)))
    for r in by_group["refinement"]:
        nx, ny = r["spec"]["grid"]
        h = max(3 / nx, 2 / ny)
        eg = np.linalg.norm(r["grams"] - ref["grams"], axis=(1, 2)) / np.linalg.norm(
            ref["grams"], axis=(1, 2)
        )
        eq = (
            float(
                np.sqrt(
                    weights
                    @ np.sum((r["responses"] - ref["responses"]) ** 2, axis=(1, 2))
                )
            )
            / ref_norm
        )
        errors = {"response_l2_relative": eq, "gramian_fro_relative": float(max(eg))}
        for name in SETTINGS:
            errors[name + "_allocation_linf"] = float(
                np.max(
                    np.abs(
                        np.asarray(r["scores"][name]["allocation"])
                        - ref["scores"][name]["allocation"]
                    )
                )
            )
        refinement["rows"].append(
            {
                "num_x": nx,
                "num_y": ny,
                "h": h,
                "is_reference": r is ref,
                "scores": {k: compact_score(v) for k, v in r["scores"].items()},
                "to_reference": errors,
            }
        )
        singular = np.linalg.svd(
            r["grams"].reshape(6, 9).T @ directions, compute_uv=False
        )
        chi["rows"].append(
            {
                "num_x": nx,
                "num_y": ny,
                "h": h,
                "chi": float(singular[-1]),
                "singular_values": singular.tolist(),
                "channel_gramians": r["grams"].tolist(),
            }
        )
    # Recompute characteristics and check the adopted endpoint values.
    from .plotting import characteristic_history

    characteristics = copy.deepcopy(data["characteristics.json"])
    times, eigenvalues = characteristic_history(characteristics)
    characteristics["uniform_gramian_eigenvalues_at_T5"] = eigenvalues[
        np.argmin(abs(times - 5))
    ].tolist()
    characteristics["uniform_gramian_eigenvalues_at_T12"] = eigenvalues[-1].tolist()
    return dict(
        zip(FILES, (velocity, refinement, chi, characteristics, apparent), strict=True)
    )


def run(output: Path, resume=False, smoke=False, limit=None):
    output = Path(output)
    protect_inputs(output)
    data, manifest = load_data()
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    metadata = {
        "schema_version": 1,
        "smoke": smoke,
        "cases": make_plan(data, smoke),
        "software": software,
        "source_sha256": source_hashes(),
        "input_manifest_sha256": digest(DATA_DIR / "manifest.json"),
        "solver_settings": {k: asdict(v) for k, v in SETTINGS.items()},
        "sampling": "Fixed adopted velocity locations; adaptive point selection is not rerun.",
    }
    if resume:
        if json.loads((output / "run.json").read_text()) != metadata:
            raise ValueError(
                "Resume requires matching code, settings, libraries, and inputs."
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run.json", metadata)
    results = []
    processed = 0
    for spec in metadata["cases"]:
        done = (
            output
            / "checkpoints"
            / (spec["key"] + ("_scores.json" if spec["optimize"] else ".json"))
        ).exists()
        if not done and limit is not None and processed >= limit:
            print(
                "Reached the requested case limit. Continue with --resume.", flush=True
            )
            return {"complete": False, "processed": processed}
        results.append(case(output / "checkpoints", spec))
        if not done:
            processed += 1
    if smoke:
        report = {
            "complete": True,
            "smoke": True,
            "paper_reproduction": False,
            "cases": [
                {
                    "case": r["spec"],
                    "scores": {k: compact_score(v) for k, v in r["scores"].items()},
                }
                for r in results
            ],
        }
        write_json(output / "smoke_result.json", report)
        return report
    reports = assemble(data, results, software)
    dest = output / "data"
    for name, payload in reports.items():
        write_json(dest / name, payload)
    write_json(
        dest / "manifest.json",
        {
            "schema_version": 1,
            "kind": "recomputed-paper-data",
            "files": {name: {"sha256": digest(dest / name)} for name in FILES},
            "run_sha256": digest(output / "run.json"),
        },
    )
    load_data(dest)
    return {"complete": True, "data_dir": str(dest)}
