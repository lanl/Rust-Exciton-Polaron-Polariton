#!/usr/bin/env python3
"""Plot dielectric-function data files written by the Rust solver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import re
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class DielectricDataset:
    path: Path
    omega: np.ndarray
    real_epsilon: np.ndarray
    imaginary_epsilon: np.ndarray
    real_alpha: np.ndarray | None
    imaginary_alpha: np.ndarray | None
    coupling: float | None
    reflectivity: float | None

    @property
    def label(self) -> str:
        labels: list[str] = []

        if self.coupling is not None:
            labels.append(f"g = {self.coupling:.4g}")

        if self.reflectivity is not None:
            labels.append(f"R = {self.reflectivity:.7g}")

        return ", ".join(labels) or self.path.stem


def theme_colors(dark_mode: bool) -> tuple[str, str]:
    if dark_mode:
        return "black", "#f2f2f2"

    return "white", "#111111"


def configure_matplotlib(dark_mode: bool) -> None:
    background_color, foreground_color = theme_colors(dark_mode)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 12.0,
            "axes.facecolor": background_color,
            "axes.edgecolor": foreground_color,
            "axes.labelcolor": foreground_color,
            "figure.facecolor": background_color,
            "savefig.facecolor": background_color,
            "text.color": foreground_color,
            "xtick.color": foreground_color,
            "ytick.color": foreground_color,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def parse_filename_value(path: Path, token: str) -> float | None:
    match = re.search(
        rf"(?:^|_){re.escape(token)}(?P<value>{FLOAT_PATTERN})(?=_|$)",
        path.stem,
    )

    if match is None:
        return None

    return float(match.group("value"))


def load_dataset(path: Path) -> DielectricDataset:
    values = np.loadtxt(path, comments="#", ndmin=2)

    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            f"{path} contains {values.shape[1] if values.ndim == 2 else 0} "
            "columns; expected at least omega, Re[eps], and Im[eps]"
        )

    real_alpha = values[:, 3] if values.shape[1] >= 5 else None
    imaginary_alpha = values[:, 4] if values.shape[1] >= 5 else None

    return DielectricDataset(
        path=path,
        omega=values[:, 0],
        real_epsilon=values[:, 1],
        imaginary_epsilon=values[:, 2],
        real_alpha=real_alpha,
        imaginary_alpha=imaginary_alpha,
        coupling=parse_filename_value(path, "lmcoup"),
        reflectivity=parse_filename_value(path, "refl"),
    )


def expand_inputs(
    inputs: Iterable[str],
    input_dir: str,
) -> list[Path]:
    items = list(inputs) or [input_dir]
    paths: dict[str, Path] = {}

    for item in items:
        item_path = Path(item)

        if any(character in item for character in "*?[]"):
            matches = [Path(match) for match in glob(item)]
        elif item_path.is_dir():
            matches = sorted(item_path.glob("dielectric_*.dat"))
        else:
            matches = [item_path]

        for path in matches:
            if path.is_file() and path.suffix == ".dat":
                paths[str(path)] = path

    if not paths:
        raise FileNotFoundError(
            "No dielectric .dat files were found. "
            f"Checked: {', '.join(items)}"
        )

    return sorted(paths.values(), key=lambda path: str(path))


def sort_datasets(datasets: list[DielectricDataset]) -> list[DielectricDataset]:
    return sorted(
        datasets,
        key=lambda dataset: (
            float("inf") if dataset.coupling is None else dataset.coupling,
            str(dataset.path),
        ),
    )


def make_colors(count: int, cmap_name: str) -> list[tuple[float, ...]]:
    color_map = mpl.colormaps[cmap_name]

    if count == 1:
        return [color_map(0.5)]

    positions = np.linspace(0.05, 0.95, count)
    return [color_map(float(position)) for position in positions]


def style_axis(axis: mpl.axes.Axes, dark_mode: bool) -> None:
    background_color, foreground_color = theme_colors(dark_mode)
    axis.set_facecolor(background_color)

    for spine in axis.spines.values():
        spine.set_color(foreground_color)

    axis.tick_params(which="both", colors=foreground_color)
    axis.xaxis.label.set_color(foreground_color)
    axis.yaxis.label.set_color(foreground_color)
    axis.title.set_color(foreground_color)


def add_legend(axis: mpl.axes.Axes, dark_mode: bool) -> None:
    background_color, foreground_color = theme_colors(dark_mode)
    legend = axis.legend(loc="best")

    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_facecolor(background_color)
    frame.set_edgecolor(foreground_color)

    for text in legend.get_texts():
        text.set_color(foreground_color)


def draw_series(
    axis: mpl.axes.Axes,
    datasets: list[DielectricDataset],
    colors: list[tuple[float, ...]],
    field_name: str,
    omega_min: float | None,
    omega_max: float | None,
    use_absolute_values: bool = False,
    include_labels: bool = True,
) -> np.ndarray:
    finite_values: list[np.ndarray] = []

    for dataset, color in zip(datasets, colors):
        values = getattr(dataset, field_name)
        plotted_values = np.abs(values) if use_absolute_values else values

        valid = np.isfinite(dataset.omega) & np.isfinite(plotted_values)

        if use_absolute_values:
            valid &= plotted_values > 0.0

        if omega_min is not None:
            valid &= dataset.omega >= omega_min

        if omega_max is not None:
            valid &= dataset.omega <= omega_max

        if not np.any(valid):
            continue

        axis.plot(
            dataset.omega[valid],
            plotted_values[valid],
            color=color,
            linewidth=1.6,
            label=dataset.label if include_labels else None,
        )
        finite_values.append(plotted_values[valid])

    if not finite_values:
        return np.empty(0, dtype=float)

    return np.concatenate(finite_values)


def set_real_scale(axis: mpl.axes.Axes, values: np.ndarray) -> None:
    if values.size == 0:
        return

    largest_value = float(np.max(np.abs(values)))
    axis.set_yscale(
        "symlog",
        linthresh=max(largest_value * 1e-3, 1e-8),
    )


def set_imaginary_scale(axis: mpl.axes.Axes, values: np.ndarray) -> None:
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("The imaginary data has no positive finite values")

    axis.set_yscale("log")


def prepare_figure(
    dark_mode: bool,
    figsize: tuple[float, float],
) -> tuple[mpl.figure.Figure, str]:
    background_color, _ = theme_colors(dark_mode)
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    figure.patch.set_facecolor(background_color)
    return figure, background_color


def plot_real(
    datasets: list[DielectricDataset],
    colors: list[tuple[float, ...]],
    args: argparse.Namespace,
) -> plt.Figure:
    figure, _ = prepare_figure(args.dark_mode, (7.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis, args.dark_mode)

    values = draw_series(
        axis,
        datasets,
        colors,
        "real_epsilon",
        args.omega_min,
        args.omega_max,
    )
    if values.size == 0:
        raise ValueError("No real dielectric data remains in the requested range")

    set_real_scale(axis, values)
    axis.set_xlabel("Frequency (a.u.)")
    axis.set_ylabel(r"Re[$\epsilon(\omega)$]")
    axis.set_title("Real part of dielectric function")
    add_legend(axis, args.dark_mode)
    return figure


def plot_imaginary(
    datasets: list[DielectricDataset],
    colors: list[tuple[float, ...]],
    args: argparse.Namespace,
) -> plt.Figure:
    figure, _ = prepare_figure(args.dark_mode, (7.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis, args.dark_mode)

    values = draw_series(
        axis,
        datasets,
        colors,
        "imaginary_epsilon",
        args.omega_min,
        args.omega_max,
        use_absolute_values=True,
    )
    if values.size == 0:
        raise ValueError(
            "No positive imaginary dielectric data remains in the requested range"
        )

    set_imaginary_scale(axis, values)
    axis.set_xlabel("Frequency (a.u.)")
    axis.set_ylabel(r"Im[$\epsilon(\omega)$]")
    axis.set_title("Imaginary part of dielectric function")
    add_legend(axis, args.dark_mode)
    return figure


def plot_both(
    datasets: list[DielectricDataset],
    colors: list[tuple[float, ...]],
    args: argparse.Namespace,
) -> plt.Figure:
    figure = plt.figure(figsize=(7.0, 7.5), constrained_layout=True)
    axes = figure.subplots(2, 1, sharex=True)
    background_color, _ = theme_colors(args.dark_mode)
    figure.patch.set_facecolor(background_color)

    real_axis, imaginary_axis = axes
    style_axis(real_axis, args.dark_mode)
    style_axis(imaginary_axis, args.dark_mode)

    real_values = draw_series(
        real_axis,
        datasets,
        colors,
        "real_epsilon",
        args.omega_min,
        args.omega_max,
    )
    imaginary_values = draw_series(
        imaginary_axis,
        datasets,
        colors,
        "imaginary_epsilon",
        args.omega_min,
        args.omega_max,
        use_absolute_values=True,
        include_labels=False,
    )

    if real_values.size == 0 or imaginary_values.size == 0:
        raise ValueError("No complete dielectric data remains in the requested range")

    set_real_scale(real_axis, real_values)
    set_imaginary_scale(imaginary_axis, imaginary_values)
    real_axis.set_ylabel(r"Re[$\epsilon(\omega)$]")
    imaginary_axis.set_xlabel("Frequency (a.u.)")
    imaginary_axis.set_ylabel(r"Im[$\epsilon(\omega)$]")
    real_axis.set_title("Dielectric function")
    add_legend(real_axis, args.dark_mode)
    return figure


def plot_alpha2d(
    datasets: list[DielectricDataset],
    colors: list[tuple[float, ...]],
    args: argparse.Namespace,
) -> plt.Figure:
    if any(
        dataset.real_alpha is None or dataset.imaginary_alpha is None
        for dataset in datasets
    ):
        raise ValueError("alpha2D plotting requires five-column dielectric data")

    figure = plt.figure(figsize=(7.0, 7.5), constrained_layout=True)
    axes = figure.subplots(2, 1, sharex=True)
    background_color, _ = theme_colors(args.dark_mode)
    figure.patch.set_facecolor(background_color)

    real_axis, imaginary_axis = axes
    style_axis(real_axis, args.dark_mode)
    style_axis(imaginary_axis, args.dark_mode)

    real_values = draw_series(
        real_axis,
        datasets,
        colors,
        "real_alpha",
        args.omega_min,
        args.omega_max,
    )
    imaginary_values = draw_series(
        imaginary_axis,
        datasets,
        colors,
        "imaginary_alpha",
        args.omega_min,
        args.omega_max,
        use_absolute_values=True,
        include_labels=False,
    )

    if real_values.size == 0 or imaginary_values.size == 0:
        raise ValueError("No complete alpha2D data remains in the requested range")

    set_imaginary_scale(imaginary_axis, imaginary_values)
    real_axis.set_ylabel(r"Re[$\alpha_{2D}(\omega)$]")
    imaginary_axis.set_xlabel("Frequency (a.u.)")
    imaginary_axis.set_ylabel(r"Im[$\alpha_{2D}(\omega)$]")
    real_axis.set_title("2D sheet polarizability")
    add_legend(real_axis, args.dark_mode)
    return figure


def save_figure(
    figure: plt.Figure,
    output_path: Path,
    dark_mode: bool,
    dpi: int,
    show: bool,
) -> None:
    background_color, _ = theme_colors(dark_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor=background_color,
        edgecolor=background_color,
        transparent=False,
    )
    print(f"Saved {output_path}")

    if show:
        plt.show()

    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dielectric-function .dat files written by the Rust "
            "light-matter solver."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Input .dat files, directories, or quoted globs. If omitted, "
            "--input-dir is scanned."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="dielectric_scan/effective_modes",
        help="Directory scanned when no positional inputs are provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/dielectric",
        help="Directory for generated plot files.",
    )
    parser.add_argument(
        "--name",
        default="dielectric",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--plot",
        choices=["all", "real", "imag", "both", "alpha2d"],
        default="all",
        help="Which dielectric plot to generate.",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "pdf", "png"],
        default="svg",
        help="Output file format.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap; defaults to viridis or plasma in dark mode.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=None,
        help="Lower frequency plotted.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=None,
        help="Upper frequency plotted.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization resolution for PNG output.",
    )
    parser.add_argument(
        "--dark-mode",
        "--dark",
        dest="dark_mode",
        action="store_true",
        help="Use black backgrounds and light annotations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated figures after saving.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.omega_min is not None and args.omega_max is not None:
        if args.omega_max <= args.omega_min:
            raise ValueError("--omega-max must exceed --omega-min")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args)
    configure_matplotlib(args.dark_mode)

    paths = expand_inputs(args.inputs, args.input_dir)
    datasets = sort_datasets([load_dataset(path) for path in paths])
    cmap_name = args.cmap or ("plasma" if args.dark_mode else "viridis")
    colors = make_colors(len(datasets), cmap_name)

    available_alpha = all(
        dataset.real_alpha is not None and dataset.imaginary_alpha is not None
        for dataset in datasets
    )
    if args.plot == "alpha2d" and not available_alpha:
        raise ValueError("The selected data does not contain alpha2D columns")

    plot_types = (
        ["real", "imag", "both"]
        if args.plot == "all"
        else [args.plot]
    )
    if args.plot == "all" and available_alpha:
        plot_types.append("alpha2d")

    plot_functions = {
        "real": plot_real,
        "imag": plot_imaginary,
        "both": plot_both,
        "alpha2d": plot_alpha2d,
    }

    for plot_type in plot_types:
        figure = plot_functions[plot_type](datasets, colors, args)
        output_path = (
            Path(args.output_dir)
            / f"{args.name}_{plot_type}.{args.format}"
        )
        save_figure(figure, output_path, args.dark_mode, args.dpi, args.show)

    print(f"Loaded {len(datasets)} dielectric data file(s)")


if __name__ == "__main__":
    main()
