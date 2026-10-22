# Validation of the distribution package

## Completed checks

- **374 tests passed**, including the adopted FV, characteristic, and PGD kernel tests and new input, CLI, checkpoint, failure, and plotting tests.
- `uv run --locked pde-cas figures` ran in this repository's newly created Python environment. All four PDF files matched the adopted manuscript PDFs byte for byte.
- A wheel was built and installed into a separate, isolated environment outside both source repositories. Its `pde-cas figures` command also generated all four PDFs with the same hashes. Bundled data are therefore available to an installed package; the command does not depend on the development repository.
- A real three-case smoke computation was paused after one case and resumed successfully. It exercised zero-diffusion FV propagation, positive-diffusion responses, and both uniform-start PGD objectives.
- A freshly computed 47×31 refinement case was compared with the adopted channel Gramians and allocations in the test suite.
- Hash checks reject modified inputs and checkpoints. A solver failure preserves the Gramian checkpoint without accepting a substitute solver. Incomplete and smoke runs do not create a complete paper-data manifest.
- The full 56-case plan and its report-assembly paths were tested, including a mocked completion-path test. This is **not** a full 56-case numerical rerun.
- The original development code, adopted data, manuscript figures, and manuscript sources were not modified by the repository extraction.

## Reference PDF hashes

| File | SHA-256 |
|---|---|
| `geometry.pdf` | `fedc24b3d1941077444533c31f703aaba0f14bc8cab0f4c639c1c16456967190` |
| `gramian_eigenvalues.pdf` | `81768a9f61878366b4392edbedfb03b94e8b7e486c296294db01de2934446a7e` |
| `velocity_allocations.pdf` | `85d583fb71ffd06e41e2106843649494942f82bd593d49d699ddb6a5c29be9bb` |
| `mesh_refinement.pdf` | `cf076c2e6b9a29421fcc2bd47c250296a7e2bcc4c81918df93070bd728fee3c3` |

## Tested environment

Python 3.13.11, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.0, and TeX Live 2026 on macOS. Python dependencies are locked in `uv.lock`; the system TeX installation is a separate prerequisite. Tests on other operating systems have not yet been performed.

## Limits and pending decisions

- The largest-grid and full velocity-sweep computations have **not** been rerun as part of this extraction. The bundled adopted results were generated and checked in the development project; provenance is preserved separately.
- The fresh computation command reproduces the plotted velocity locations, not the historical adaptive sample-selection decisions. Table 1 and the 22 midpoint probes are not included in its scope.
- Byte-identical PDFs on the tested setup do not guarantee identical bytes with another TeX/font installation.
- Discrete feasibility and separation diagnostics provide numerical evidence, not continuum certification.
