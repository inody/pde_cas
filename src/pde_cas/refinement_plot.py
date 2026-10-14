"""Draw the adopted square triptych without constructing the legacy two-panel figure."""

import matplotlib
import matplotlib.pyplot as plt
from .plotting import (
    validate_refinement_chi_pair,
    aggregate_gramian_minimum_eigenvalues,
    refinement_figure,
    refinement_mesh_axis,
)
from .style import BLUE, PURPLE, manuscript_panels


def refinement_with_feasibility_figure(refinement, chi_report):
    h, chi = validate_refinement_chi_pair(refinement, chi_report)
    eigenvalues = aggregate_gramian_minimum_eigenvalues(chi_report)
    source = refinement_figure(refinement)
    old = source.axes[0]
    fig, panels = manuscript_panels(ncols=3)
    error_axis, feasibility_axis, chi_axis = panels[0]
    for line in old.lines:
        error_axis.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            markerfacecolor=line.get_markerfacecolor(),
            linewidth=line.get_linewidth(),
            label=line.get_label(),
        )
    error_axis.set(
        yscale="log",
        ylim=old.get_ylim(),
        ylabel=old.get_ylabel(),
        title="(a) Reference differences",
    )
    plt.close(source)
    error_axis.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, numticks=5)
    )
    error_axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    error_axis.legend(loc="center left", ncol=2, handlelength=2.0, columnspacing=0.7)
    feasibility_axis.plot(
        h,
        eigenvalues,
        color=BLUE,
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        linewidth=1.45,
    )
    feasibility_axis.set(
        ylabel=r"$\lambda_{\min}(\Gamma_{T,h})$",
        title="(b) Target feasibility",
        ylim=(
            min(0.0, float(eigenvalues.min()) * 1.15),
            float(eigenvalues.max()) * 1.15 if eigenvalues.max() > 0 else 1,
        ),
    )
    chi_axis.plot(
        h,
        chi,
        color=PURPLE,
        linestyle="-",
        marker="v",
        markerfacecolor="none",
        linewidth=1.45,
    )
    chi_axis.set(
        ylabel=r"$\chi_{T,h}$",
        title="(c) Restricted separation",
        ylim=(0, float(chi.max()) * 1.15 if chi.max() > 0 else 1),
    )
    for axis in (feasibility_axis, chi_axis):
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        refinement_mesh_axis(axis, h)
        axis.set_xticklabels(
            [f"{value:.3g}" if index % 2 == 0 else "" for index, value in enumerate(h)]
        )
    refinement_mesh_axis(error_axis, h[:-1])
    for axis in panels[0]:
        axis.set_title(axis.get_title(), y=1.08)
    return fig
