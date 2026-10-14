# pde_cas

Reproducible numerical examples for *Controllability Allocation Scores for Linear Infinite-Dimensional Systems*.

**Private preparation version. No open-source license has been selected yet. Do not publish or redistribute this repository until the authors have approved its release and license.**

## Reproduce Figures 1–4

Install [uv](https://docs.astral.sh/uv/) and TeX Live first. The TeX installation must provide `pdflatex`, `latex`, `dvipng`, `kpsewhich`, and the packages `type1cm`, `type1ec` (cm-super), `amsmath`, `amssymb`, and `bm`. The code deliberately does not substitute fonts or resize panels to bypass missing dependencies.

For example, on Debian/Ubuntu the TeX dependencies can be installed with:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended texlive-pictures cm-super dvipng
```

A full MacTeX installation also provides the required tools. Python 3.13.11 is selected by `.python-version`; uv can install it if needed.

From the repository root, run:

```bash
uv run --locked pde-cas figures
```

uv creates the Python environment using the pinned dependencies in `uv.lock`. The command writes four PDF/PNG pairs and `reproduction.json` to `outputs/figures/`:

| Paper figure | Output stem |
|---|---|
| Figure 1: actuator and target geometry | `geometry` |
| Figure 2: apparent feasibility | `gramian_eigenvalues` |
| Figure 3: velocity-dependent allocations | `velocity_allocations` |
| Figure 4: refinement, feasibility, and separation | `mesh_refinement` |

This is a **figure-regeneration command**, not a full PDE/optimization rerun. It uses the bundled, adopted FV and PGD results. The short characteristic history in Figure 2 is recomputed using the recorded settings and checked against the saved endpoints at T=5 and T=12. All five input files are hash-checked, and the allocation data must identify the adopted uniform-start PGD solver. There is no fallback to historical optimization results.

`reproduction.json` records output hashes and comparison with the adopted PDF hashes. Exact PDF bytes also depend on TeX, fonts, and the operating system; byte equality is a stricter test than numerical or visual agreement.

## Recompute from PDE responses and optimization

```bash
uv run --locked pde-cas compute --output outputs/recomputed
uv run --locked pde-cas figures --data-dir outputs/recomputed/data --output outputs/recomputed/figures
```

The computation can be expensive: it covers the four FV meshes in Figure 2, the 47 plotted velocity samples in Figure 3, and the five refinement meshes in Figure 4, including 767×511 cells. Figure 3 uses the adopted velocity locations; this command does **not** repeat the historical adaptive point-selection process or its 22 midpoint probes. Table 1 is also outside this command's scope.

The FV generator, mass-normalized profiles, face-integrated fluxes, exponential-action propagation, composite Simpson quadrature, and PGD arithmetic follow the adopted implementation. PGD uses a uniform initial allocation, with accepted-step Euclidean mapping tolerances of 1e-12 for V-CAS and 1e-8 for AE-CAS. No polishing, regularization, clipping, or alternative solver is used.

Results are written to a new directory; the bundled adopted data are never overwritten. Each Gramian is checkpointed **before** optimization. To resume:

```bash
uv run --locked pde-cas compute --output outputs/recomputed --resume
```

Resume requires matching code, numerical-library versions, inputs, and solver settings. `--limit N` stops after at most N new cases; incomplete runs do not produce a complete plotting-data manifest.

For a small implementation check:

```bash
uv run --locked pde-cas compute --smoke --output outputs/smoke
```

The smoke run exercises three small cases. **It is not a reproduction of the full paper and cannot be used as its refinement study.**

## Tests

```bash
uv run --locked pytest
```

Tests cover the numerical kernels, input tampering and solver identification, checkpoint/resume behavior, failure handling, plotting arrays, and panel dimensions. Numerical positive definiteness and positive restricted-separation diagnostics are not certificates for the corresponding continuum properties.

## Repository layout

- `src/pde_cas/`: FV, characteristics, PGD, plotting, and command-line code.
- `src/pde_cas/data/`: adopted results and their hash manifest, treated as read-only.
- `tests/`: kernel and reproduction-path checks.
- `PROVENANCE.json`: source development commit, extracted source files, and hashes.

See [VALIDATION.md](VALIDATION.md) for the checks performed on this preparation version and their limits.

The development repository retains the historical heat/Gaussian experiments, alternative solvers, failed pilots, and audit history. This repository excludes those legacy workflows, private execution records, third-party literature, annotated manuscripts, and manuscript-diff archives.

The internal `vcs`/`aecs` keys are retained for compatibility with the saved results. Figure labels use V-CAS/AE-CAS; CAS denotes an optimizing allocation, not an objective value.
