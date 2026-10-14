"""Adopted plotting functions without legacy command-line or optimization paths."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.integrate import cumulative_simpson
from .style import (BLACK, BLUE, GREEN, LIGHT_GRAY, PALETTE, PANEL_GAP_IN,
    PANEL_HEIGHT_IN, PANEL_WIDTH_IN, PURPLE, VERMILLION, apply_manuscript_style,
    manuscript_panels, quantitative_grid)
from .continuum import Room2DFlow, response_series, velocity_field
from .geometry import (ROOM_FV_ACTUATOR_CENTERS, ROOM_FV_DAMPING, ROOM_FV_PROFILE_RADIUS,
    ROOM_FV_SINGULAR_HORIZON, ROOM_FV_TARGET_CENTERS, ROOM_FV_VELOCITY_SCALE)
MIN_EIGENVALUE_TOLERANCE = 1.0e-14
RELATIVE_EIGENVALUE_TOLERANCE = 1.0e-12
def gramian_positive_definiteness_threshold(lambda_max: float) -> float:
    return max(
        MIN_EIGENVALUE_TOLERANCE,
        RELATIVE_EIGENVALUE_TOLERANCE * max(float(lambda_max), 0.0),
    )

CHANNEL_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
EIGENVALUE_LINEAR_THRESHOLD = 1.0e-20
EIGENVALUE_ENCODINGS = ((BLACK, "-", "o"), (BLUE, "--", "s"), (GREEN, "-.", "^"))


def geometry_figure() -> plt.Figure:
    fig, panels = manuscript_panels()
    axis = panels[0, 0]
    x, y = np.linspace(0.0, 3.0, 121), np.linspace(0.0, 2.0, 81)
    xx, yy = np.meshgrid(x, y)
    velocity = velocity_field(np.stack((xx, yy), axis=-1))
    speed = np.linalg.norm(velocity, axis=-1)
    # Avoid dense closed streamline traces at the vortex stagnation point.
    slow = speed < 0.04 * speed.max()
    axis.streamplot(
        x, y, np.ma.masked_where(slow, velocity[..., 0]), np.ma.masked_where(slow, velocity[..., 1]), color=LIGHT_GRAY,
        density=0.65, linewidth=0.5, arrowsize=0.6, zorder=1,
    )
    for centers, prefix, color, filled in (
        (ROOM_FV_ACTUATOR_CENTERS, "b", BLUE, True),
        (ROOM_FV_TARGET_CENTERS, "c", VERMILLION, False),
    ):
        for index, center in enumerate(centers, start=1):
            axis.add_patch(Circle(
                center, ROOM_FV_PROFILE_RADIUS, facecolor=color if filled else "white",
                edgecolor=color, alpha=0.22 if filled else 1.0,
                linestyle="-" if filled else "--", linewidth=1.0, zorder=3,
            ))
            axis.text(*center, rf"${prefix}_{index}$", ha="center", va="center", color=color, fontsize=8, zorder=4)
    axis.set(xlim=(0.0, 3.0), ylim=(0.0, 2.0), xlabel=r"$\xi_1$", ylabel=r"$\xi_2$")
    axis.set_xticks([0, 1, 2, 3])
    axis.set_yticks([0, 1, 2])
    axis.set_aspect("equal", adjustable="box")
    return fig

def characteristic_history(summary: dict) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, summary["horizon"], summary["num_time_points"])
    responses = response_series(
        times, flow=Room2DFlow(velocity_scale=ROOM_FV_VELOCITY_SCALE),
        damping=ROOM_FV_DAMPING, **summary["reference"],
    )
    products = np.einsum("tji,tki->tijk", responses, responses, optimize=True)
    grams = cumulative_simpson(products, x=times, axis=0, initial=0.0)
    eigenvalues = np.linalg.eigvalsh(np.mean(grams, axis=1))
    t5 = int(np.argmin(abs(times - ROOM_FV_SINGULAR_HORIZON)))
    if not np.isclose(times[t5], ROOM_FV_SINGULAR_HORIZON) or not np.isclose(times[-1], 12.0):
        raise ValueError("continuum history does not contain the validated horizons")
    for actual, key in (
        (eigenvalues[t5], "uniform_gramian_eigenvalues_at_T5"),
        (eigenvalues[-1], "uniform_gramian_eigenvalues_at_T12"),
    ):
        np.testing.assert_allclose(actual, summary[key], rtol=2e-9, atol=1e-22, err_msg="recomputed history differs from validated JSON")
    return times, eigenvalues

def gramian_eigenvalues_figure(times: np.ndarray, eigenvalues: np.ndarray, result: dict) -> plt.Figure:
    """Ordered continuum/discrete spectra, with common rank encodings and y scale.

    Zero and negative roundoff values are kept signed. The right-hand abscissa
    is the actual FV cell size, not a grid index or a nodal-grid spacing.
    """
    if result["schema_version"] != 1 or (
        result["diffusivity"], result["velocity_scale"], result["damping"], result["horizon"]
    ) != (0.0, ROOM_FV_VELOCITY_SCALE, ROOM_FV_DAMPING, ROOM_FV_SINGULAR_HORIZON):
        raise ValueError("unexpected apparent-feasibility parameters")
    times, eigenvalues = np.asarray(times, dtype=float), np.asarray(eigenvalues, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("horizons must be finite and strictly increasing")
    rows = result["rows"]
    if len(rows) != 4 or any(row["num_x"] <= 0 or row["num_y"] <= 0 for row in rows):
        raise ValueError("expected four positive rectangular grids")
    discrete = np.array([row["uniform_gramian_eigenvalues"] for row in rows], dtype=float)
    for values, count in ((eigenvalues, times.size), (discrete, len(rows))):
        if values.shape != (count, 3) or not np.all(np.isfinite(values)) or np.any(np.diff(values, axis=1) < 0.0):
            raise ValueError("expected three finite ordered eigenvalues per sample")
    flow = Room2DFlow()
    h = np.array([max(flow.length_x / row["num_x"], flow.length_y / row["num_y"]) for row in rows])
    if np.any(np.diff(h) >= 0.0):
        raise ValueError("FV grids must have decreasing cell sizes")
    thresholds = np.array([row["practical_pd_threshold"] for row in rows])
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError("practical thresholds must be finite and positive")

    fig, panels = manuscript_panels(ncols=2, sharey=True)
    continuum_axis, discrete_axis = panels[0]
    for index, (color, style, marker) in enumerate(EIGENVALUE_ENCODINGS):
        label = rf"$\lambda_{index + 1}$"
        continuum_axis.plot(times, eigenvalues[:, index], color=color, linestyle=style, label=label)
        discrete_axis.plot(h, discrete[:, index], color=color, linestyle=style, marker=marker,
                           markerfacecolor="none", label=label)
    continuum_thresholds = [gramian_positive_definiteness_threshold(value) for value in eigenvalues[:, -1]]
    continuum_axis.plot(times, continuum_thresholds, color=VERMILLION, linestyle=":", linewidth=1.2, label=r"$\tau_{\rm PD}$")
    discrete_axis.plot(h, thresholds, color=VERMILLION, linestyle=":", linewidth=1.2, label=r"$\tau_{\rm PD}$")
    lower = min(-EIGENVALUE_LINEAR_THRESHOLD, 1.2 * float(min(eigenvalues.min(), discrete.min())))
    upper = max(1e-2, 2.0 * float(max(eigenvalues.max(), discrete.max())))
    for axis in panels.flat:
        axis.set_yscale("symlog", linthresh=EIGENVALUE_LINEAR_THRESHOLD, linscale=0.55)
        axis.set_ylim(lower, upper)
        axis.set_yticks([0.0, 1e-16, 1e-12, 1e-8, 1e-3])
        quantitative_grid(axis)
    continuum_axis.set(xlabel=r"horizon $T$", ylabel=r"eigenvalues $\lambda_k$", title="(a) Horizon dependence", xlim=(times[0], times[-1]))
    continuum_axis.set_xticks([0, 3, 6, 9, 12])
    continuum_axis.axvline(ROOM_FV_SINGULAR_HORIZON, color=LIGHT_GRAY, linestyle="--", linewidth=0.9, zorder=0)
    # This legend identifies the same eigenvalue ranks/threshold in both panels.
    continuum_axis.legend(loc="upper left", handlelength=1.6, handletextpad=0.6, borderpad=0.3, fontsize=7.2)
    discrete_axis.set_xscale("log")
    discrete_axis.set(xlabel=r"cell size $h$", title="(b) Mesh refinement", xlim=(h[0] * 1.18, h[-1] / 1.18))
    discrete_axis.set_xticks(h, [f"{value:.3g}" for value in h])
    discrete_axis.tick_params(axis="x", labelsize=7.5)
    discrete_axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return fig

def velocity_figure(result: dict, curve_grid: tuple[int, int] = (191, 127)) -> plt.Figure:
    """Plot certified curve nodes from one requested grid, never midpoint probes."""
    if result["schema_version"] != 1 or result["kind"] != "fv_adaptive_velocity_sweep":
        raise ValueError("expected adaptive velocity-sweep schema")
    metadata = result["metadata"]
    if (metadata["diffusivity"], metadata["damping"], metadata["horizon"], metadata["geometry"]) != (0.03, 0.02, 6.0, "D"):
        raise ValueError("unexpected velocity-sweep parameters")
    tolerances = [metadata[key] for key in ("kkt_tolerance", "simplex_tolerance", "active_tolerance")]
    if not np.all(np.isfinite(tolerances)) or np.any(np.asarray(tolerances) <= 0.0):
        raise ValueError("invalid certification tolerances")
    kkt_tolerance, simplex_tolerance, active_tolerance = tolerances
    if len(curve_grid) != 2 or min(curve_grid) <= 0:
        raise ValueError("curve grid dimensions must be positive")
    selected = [grid for grid in result["grids"]
                if (grid["num_x"], grid["num_y"], grid["role"])
                == (curve_grid[0], curve_grid[1], "final_curve")]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {curve_grid[0]}x{curve_grid[1]} final_curve grid")
    grid = selected[0]
    rows = grid["rows"]
    velocities = np.array([row["velocity_scale"] for row in rows])
    if len(rows) < 2 or not np.all(np.isfinite(velocities)) or np.any(np.diff(velocities) <= 0.0):
        raise ValueError("velocity samples must be finite and strictly increasing")
    if not np.array_equal(velocities, grid["velocities"]) or not np.array_equal(velocities, result["velocities"]):
        raise ValueError("velocity rows must match the shared adaptive curve nodes")
    allocations = {}
    for name in ("vcs", "aecs"):
        if not all(row[name]["success"] for row in rows):
            raise ValueError("refusing to plot unsuccessful allocations as validated results")
        weights = np.array([row[name]["allocation"] for row in rows])
        if weights.shape != (len(rows), 6) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("invalid allocation array")
        if np.any(np.abs(weights.sum(axis=1) - 1.0) > simplex_tolerance):
            raise ValueError("allocations violate the simplex constraint")
        for row, allocation in zip(rows, weights):
            score = row[name]
            diagnostics = np.array([score[key] for key in ("relative_kkt_residual", "simplex_error", "nonnegativity_violation")])
            if (not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0)
                    or np.any(diagnostics > [kkt_tolerance, simplex_tolerance, 0.0])):
                raise ValueError("refusing to plot uncertified allocations")
            if score["active_set"] != (np.flatnonzero(allocation > active_tolerance) + 1).tolist():
                raise ValueError("allocation support differs from its certificate")
        allocations[name] = weights
    fig, panels = manuscript_panels(ncols=2, sharex=True, sharey=True)
    for axis, name, title in zip(panels[0], ("vcs", "aecs"), ("(a) V-CAS allocation", "(b) AE-CAS allocation")):
        weights = allocations[name]
        for channel in range(6):
            axis.plot(
                velocities, weights[:, channel], color=PALETTE[channel], linestyle=CHANNEL_STYLES[channel],
                linewidth=1.45, marker=None, label=rf"$p_{channel + 1}$",
            )
        axis.set(xlabel=r"velocity scale $V_0$", title=title, xlim=(velocities[0], velocities[-1]), ylim=(-0.025, 0.93))
        axis.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        axis.legend(loc="upper left", ncol=3, columnspacing=0.7, handlelength=2.2, borderpad=0.4)
        quantitative_grid(axis)
    panels[0, 0].set_ylabel(r"allocation $p_i$")
    return fig

def refinement_figure(result: dict) -> plt.Figure:
    """Plot saved finite-reference errors, without fitted/asymptotic rate claims.

    The zero-error reference is omitted, not replaced by a positive log floor.
    Exactly four curves are shown: response/Gramian relative differences and
    V-CAS/AE-CAS allocation Linf differences. No objective-value curves are used;
    optimizer stationarity residuals are not allocation-error bounds.
    """
    if result["schema_version"] != 1:
        raise ValueError("unsupported refinement schema")
    configuration = result["configuration"]
    physics = configuration["physics"]
    if (physics["diffusivity"], physics["velocity_scale"], configuration["time"]["horizon"]) != (0.03, 0.20, 6.0):
        raise ValueError("unexpected refinement parameters")
    rows = result["rows"]
    if len(rows) < 3 or [row["is_reference"] for row in rows] != [False] * (len(rows) - 1) + [True]:
        raise ValueError("expected at least two comparison grids followed by one reference")
    if result["reference_grid"] != [rows[-1]["num_x"], rows[-1]["num_y"]]:
        raise ValueError("reference grid does not match final row")
    geometry = configuration["geometry"]
    all_h = np.array([max(geometry["length_x"] / row["num_x"], geometry["length_y"] / row["num_y"]) for row in rows])
    if np.any(np.diff(all_h) >= 0.0) or not np.allclose(all_h, [row["h"] for row in rows], rtol=1e-12, atol=0.0):
        raise ValueError("refinement h must be the actual decreasing FV cell size")
    if not all(row["scores"][criterion]["success"] for row in rows for criterion in ("vcs", "aecs")):
        raise ValueError("refusing to plot unsuccessful optimizations as convergence data")
    h = all_h[:-1]

    def errors(key: str) -> np.ndarray:
        values = np.array([row["to_reference"][key] for row in rows[:-1]], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0) or rows[-1]["to_reference"][key] != 0:
            raise ValueError("expected positive finite comparison errors and zero reference error")
        return values

    fig, panels = manuscript_panels()
    axis = panels[0, 0]
    curves = (
        ("response_l2_relative", r"$E_Q$", BLUE, "-", "o"),
        ("gramian_fro_relative", r"$E_G$", GREEN, "--", "s"),
        ("vcs_allocation_linf", r"$E_p^{\rm V}$", BLACK, ":", "D"),
        ("aecs_allocation_linf", r"$E_p^{\rm A}$", VERMILLION, "-.", "^"),
    )
    for key, label, color, style, marker in curves:
        axis.loglog(h, errors(key), color=color, linestyle=style, marker=marker,
                    markerfacecolor="none", linewidth=1.45, label=label)
    axis.set(xlabel=r"cell size $h$", ylabel="error to reference", title="Mesh refinement", xlim=(h[0] * 1.18, h[-1] / 1.18))
    axis.set_xticks(h, [f"{value:.3g}" for value in h])
    axis.tick_params(axis="x", labelsize=7.5)
    axis.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=5))
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.legend(loc="center left", ncol=2, handlelength=2.0, columnspacing=0.7)
    quantitative_grid(axis)
    return fig

def validate_refinement_chi_pair(refinement: dict, chi_report: dict) -> tuple[np.ndarray, np.ndarray]:
    if chi_report.get("kind") != "fv_restricted_separation_mesh_sweep" or chi_report.get("schema_version") != 1:
        raise ValueError("unsupported chi report")
    if not chi_report.get("complete"):
        raise ValueError("refusing to plot an incomplete sweep as a final figure")
    for key in ("geometry", "physics", "time"):
        if refinement["configuration"][key] != chi_report["configuration"][key]:
            raise ValueError(f"refinement and chi configurations disagree: {key}")
    expected_grids = [(r["num_x"], r["num_y"]) for r in refinement["rows"]]
    rows = chi_report["rows"]
    if expected_grids != [(r["num_x"], r["num_y"]) for r in rows]:
        raise ValueError("refinement and chi meshes disagree")
    h = np.array([r["h"] for r in rows])
    chi = np.array([r["chi"] for r in rows])
    if not np.all(np.isfinite(h)) or np.any(h <= 0) or np.any(np.diff(h) >= 0):
        raise ValueError("chi mesh sizes must be finite, positive, and decreasing")
    if not np.allclose(h, [r["h"] for r in refinement["rows"]], rtol=1e-12, atol=0):
        raise ValueError("refinement and chi cell sizes disagree")
    if not np.all(np.isfinite(chi)) or np.any(chi < 0):
        raise ValueError("chi values must be finite and nonnegative")
    return h, chi

def refinement_mesh_axis(axis, h):
    axis.set_xscale("log")
    axis.set_xlim(h[0] * 1.18, h[-1] / 1.18)
    axis.set_xticks(h, [f"{value:.3g}" for value in h])
    axis.tick_params(axis="x", labelsize=7.0)
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_xlabel(r"cell size $h$")
    quantitative_grid(axis)

def aggregate_gramian_minimum_eigenvalues(chi_report: dict) -> np.ndarray:
    """Use the saved channel matrices, not an error norm or allocated Gramian.

    Symmetrization removes only arithmetic skew; no clipping, regularization,
    normalization, or positive-definiteness threshold is applied.
    """
    values = []
    for row in chi_report["rows"]:
        grams = np.asarray(row["channel_gramians"])
        if grams.shape != (6, 3, 3) or not np.isrealobj(grams) or not np.all(np.isfinite(grams)):
            raise ValueError("expected six finite real 3-by-3 channel Gramians")
        grams = grams.astype(float)
        tolerance = 64 * np.finfo(float).eps * np.max(np.abs(grams))
        if np.max(np.abs(grams - grams.swapaxes(1, 2))) > tolerance:
            raise ValueError("channel Gramians must be symmetric")
        symmetric = (grams + grams.swapaxes(1, 2)) / 2
        if np.min(np.linalg.eigvalsh(symmetric)) < -tolerance:
            raise ValueError("channel Gramians must be positive semidefinite up to roundoff")
        values.append(np.linalg.eigvalsh(symmetric.sum(axis=0))[0])
    return np.asarray(values)

def verify_layout(fig: plt.Figure) -> None:
    """Assert physical axes dimensions and unclipped text/legends."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    size = fig.get_size_inches()
    if fig._suptitle is not None:
        raise ValueError("candidate figures must not have figure-level titles")
    for axis in fig.axes:
        bounds = np.array(axis.get_position().bounds) * np.tile(size, 2)
        ncols = axis.get_subplotspec().get_gridspec().ncols
        panel_width = (2 * PANEL_WIDTH_IN - PANEL_GAP_IN) / 3 if ncols == 3 else PANEL_WIDTH_IN
        panel_height = panel_width if ncols == 3 else PANEL_HEIGHT_IN
        np.testing.assert_allclose(bounds[2:], [panel_width, panel_height], atol=1e-10)
        artists = [axis.title, axis.xaxis.label, axis.yaxis.label,
                   axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()]
        # Log locators also create ticks outside the view; Matplotlib does not
        # render those, so their off-canvas text boxes are not clipping errors.
        for locations, labels, limits in (
            (axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
            (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim()),
        ):
            low, high = sorted(limits)
            artists.extend(label for location, label in zip(locations, labels) if low <= location <= high)
        if axis.get_legend() is not None:
            legend = axis.get_legend()
            artists.append(legend)
            box = legend.get_window_extent(renderer)
            for line in axis.lines:
                if not line.get_visible():
                    continue
                xy = axis.transData.transform(line.get_xydata())
                covered = (xy[:, 0] > box.x0) & (xy[:, 0] < box.x1) & (xy[:, 1] > box.y0) & (xy[:, 1] < box.y1)
                if np.any(covered):
                    raise ValueError(f"legend obscures plotted samples: {line.get_label()}")
        for artist in artists:
            if not artist.get_visible():
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                raise ValueError(f"clipped label or legend: {artist}")

def save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    verify_layout(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    # PGF/pdfLaTeX preserves the Computer Modern minus glyph (native PDF in
    # Matplotlib 3.11 may lose the Type-1 slot-0 glyph). Do not crop or relayout.
    fig.savefig(output_dir / f"{stem}.pdf", backend="pgf", bbox_inches=None)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches=None)
    plt.close(fig)
    print(output_dir / f"{stem}.pdf")
    print(output_dir / f"{stem}.png")
