#!/usr/bin/env python3
"""
Publication-quality band-structure plotting for Rust ColoredBandData files.

For photonic-character plots, colored line segments are sorted in decreasing
photonic character before rendering. Consequently:

    highly photonic segments are drawn first;
    low-photonic-character segments are drawn last/on top.

Highly photonic segments are also faded by default to suppress the dense sea
of photon modes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from band_data import (
    ColoredBandData,
    build_path_axis,
    load_band_datasets,
)


ColorKind = Literal["photonic", "n-expectation", "generic"]


@dataclass
class PreparedBandData:
    data: ColoredBandData
    x: np.ndarray
    energies: np.ndarray
    colors: np.ndarray | None
    symmetry_positions: np.ndarray | None
    symmetry_labels: list[str] | None
    colored_segments: np.ndarray | None
    segment_colors: np.ndarray | None


def theme_colors(dark_mode: bool) -> tuple[str, str]:
    """Return the background and foreground colors for the selected theme."""
    if dark_mode:
        return "black", "#f2f2f2"

    return "white", "black"


def configure_matplotlib(dark_mode: bool = False) -> None:
    """Set publication-oriented Matplotlib defaults."""
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
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.7,
            "axes.facecolor": background_color,
            "axes.edgecolor": foreground_color,
            "axes.labelcolor": foreground_color,
            "axes.unicode_minus": True,
            "figure.facecolor": background_color,
            "savefig.facecolor": background_color,
            "text.color": foreground_color,
            "xtick.color": foreground_color,
            "ytick.color": foreground_color,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def infer_color_kind(
    filenames: tuple[Path, ...],
    colors: np.ndarray | None,
) -> ColorKind:
    """Infer the meaning of color_values from filenames and values."""
    joined_name = " ".join(path.name.lower() for path in filenames)

    if "n_colored_band" in joined_name:
        return "n-expectation"

    if "color_band" in joined_name:
        return "photonic"

    if colors is not None:
        finite = colors[np.isfinite(colors)]

        if finite.size and finite.min() >= -1e-8 and finite.max() <= 1.0 + 1e-8:
            return "photonic"

    return "generic"


def default_colorbar_label(kind: ColorKind) -> str:
    if kind == "photonic":
        return "Photonic character"

    if kind == "n-expectation":
        return r"$\langle n\rangle$"

    return "Color value"


def make_line_segments(
    x: np.ndarray,
    energies: np.ndarray,
    energy_min: float | None,
    energy_max: float | None,
) -> np.ndarray:
    """
    Make two-point segments for all adjacent band points.

    The output has shape (n_segments, 2, 2), suitable for LineCollection.
    """
    if len(x) < 2:
        return np.empty((0, 2, 2), dtype=float)

    x0 = np.broadcast_to(
        x[:-1, None],
        energies[:-1].shape,
    )
    x1 = np.broadcast_to(
        x[1:, None],
        energies[1:].shape,
    )

    e0 = energies[:-1]
    e1 = energies[1:]

    segments = np.stack(
        [
            np.stack([x0, e0], axis=-1),
            np.stack([x1, e1], axis=-1),
        ],
        axis=-2,
    ).reshape(-1, 2, 2)

    valid = np.all(np.isfinite(segments), axis=(1, 2))

    # Discard segments fully outside the selected energy window. Segments
    # crossing a boundary are retained and clipped by the axes.
    if energy_min is not None:
        valid &= np.max(segments[:, :, 1], axis=1) >= energy_min

    if energy_max is not None:
        valid &= np.min(segments[:, :, 1], axis=1) <= energy_max

    return segments[valid]


def make_colored_line_segments(
    x: np.ndarray,
    energies: np.ndarray,
    colors: np.ndarray,
    energy_min: float | None,
    energy_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Make adjacent band segments and their midpoint color values."""
    if len(x) < 2:
        return (
            np.empty((0, 2, 2), dtype=float),
            np.empty(0, dtype=float),
        )

    x0 = np.broadcast_to(
        x[:-1, None],
        energies[:-1].shape,
    )
    x1 = np.broadcast_to(
        x[1:, None],
        energies[1:].shape,
    )

    segments = np.stack(
        [
            np.stack([x0, energies[:-1]], axis=-1),
            np.stack([x1, energies[1:]], axis=-1),
        ],
        axis=-2,
    ).reshape(-1, 2, 2)

    segment_colors = (
        0.5 * (colors[:-1] + colors[1:])
    ).reshape(-1)

    valid = (
        np.all(np.isfinite(segments), axis=(1, 2))
        & np.isfinite(segment_colors)
    )

    if energy_min is not None:
        valid &= np.max(segments[:, :, 1], axis=1) >= energy_min

    if energy_max is not None:
        valid &= np.min(segments[:, :, 1], axis=1) <= energy_max

    return segments[valid], segment_colors[valid]


def rendering_order(
    normalized_colors: np.ndarray,
    emphasis: str,
) -> np.ndarray:
    """
    Return the colored-segment rendering order.

    emphasis="low":
        High values first, low values last/on top.

    emphasis="high":
        Low values first, high values last/on top.

    emphasis="none":
        Retain the existing segment order.
    """
    if emphasis == "low":
        return np.argsort(
            -normalized_colors,
            kind="stable",
        )

    if emphasis == "high":
        return np.argsort(
            normalized_colors,
            kind="stable",
        )

    return np.arange(normalized_colors.size)


def point_alpha(
    normalized_colors: np.ndarray,
    emphasis: str,
    alpha_min: float,
    alpha_max: float,
    alpha_power: float,
    fade: bool,
) -> np.ndarray:
    """Calculate colored-segment opacity values for composition emphasis."""
    if not fade or emphasis == "none":
        return np.full(
            normalized_colors.shape,
            alpha_max,
            dtype=float,
        )

    if emphasis == "low":
        importance = 1.0 - normalized_colors
    else:
        importance = normalized_colors

    return alpha_min + (
        alpha_max - alpha_min
    ) * importance**alpha_power


def add_symmetry_guides(
    ax: mpl.axes.Axes,
    positions: np.ndarray,
    labels: list[str],
    color: str = "0.82",
) -> None:
    """Add high-symmetry ticks and unobtrusive vertical guides."""
    for position in positions:
        ax.axvline(
            position,
            color=color,
            linewidth=0.55,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def _normalized_band_stem(path: Path) -> str:
    stem = re.sub(
        r"_chunk\d+_of\d+(?:_zoomed)?$",
        "",
        path.stem,
    )

    if stem.startswith("color_band_chunk_"):
        stem = "color_band_" + stem[len("color_band_chunk_"):]

    return stem


def default_output_path(source_files: tuple[Path, ...]) -> Path:
    """Return a PDF path derived from the input data filename(s)."""
    if not source_files:
        return Path("band_structure.pdf")

    stems = [_normalized_band_stem(path) for path in source_files]

    if len(set(stems)) == 1:
        return Path(f"{stems[0]}.pdf")

    return Path(f"{stems[0]}_set.pdf")


def plot_bands(args: argparse.Namespace) -> None:
    datasets = load_band_datasets(
        args.inputs,
        input_dir=args.input_dir,
        require_complete_chunks=not args.allow_incomplete_chunks,
    )

    # Effective window limits used for both data filtering and the y-axis.
    lower_limit = args.energy_min
    energy_cutoffs = [data.energy_cutoff for data in datasets]
    if args.energy_max is not None:
        upper_limit = args.energy_max
    elif args.ignore_energy_cutoff or not all(
        cutoff is not None for cutoff in energy_cutoffs
    ):
        upper_limit = None
    else:
        upper_limit = max(
            cutoff for cutoff in energy_cutoffs if cutoff is not None
        )

    prepared: list[PreparedBandData] = []

    for data in datasets:
        # Energy-cutoff runs have a different number of states at each
        # k-point. Missing band indices are NaN-padded so lines break rather
        # than the band being truncated to the minimum count.
        k_points, energies, colors = data.padded_arrays(
            max_bands=args.bands,
        )

        x, symmetry_positions, symmetry_labels = build_path_axis(
            k_points,
            mode=args.path_mode,
        )

        colored_segments = None
        segment_colors = None

        if colors is not None:
            colored_segments, segment_colors = make_colored_line_segments(
                x,
                energies,
                colors,
                lower_limit,
                upper_limit,
            )

            if colored_segments.size == 0:
                source = data.source_files[0]
                raise ValueError(
                    "No finite colored line segments remain in the selected "
                    f"energy window for {source}"
                )

        prepared.append(
            PreparedBandData(
                data=data,
                x=x,
                energies=energies,
                colors=colors,
                symmetry_positions=symmetry_positions,
                symmetry_labels=symmetry_labels,
                colored_segments=colored_segments,
                segment_colors=segment_colors,
            )
        )

    configure_matplotlib(args.dark_mode)
    background_color, foreground_color = theme_colors(args.dark_mode)
    line_color = args.line_color or (
        "#888888" if args.dark_mode else "0.25"
    )
    uncolored_line_color = args.uncolored_line_color or (
        foreground_color if args.dark_mode else "#222222"
    )

    color_arrays = [
        item.colors
        for item in prepared
        if item.colors is not None
    ]
    has_colored_data = bool(color_arrays)

    if has_colored_data and len(color_arrays) != len(prepared):
        raise ValueError(
            "The input set mixes colored and uncolored band files"
        )

    if not has_colored_data and not args.uncolored:
        raise ValueError(
            "The input files contain empty color rows. Use --uncolored to "
            "make a conventional uncolored band plot."
        )

    all_color_values = None

    if has_colored_data:
        all_color_values = np.concatenate(
            [array.reshape(-1) for array in color_arrays]
        )

    source_files = tuple(
        path
        for item in prepared
        for path in item.data.source_files
    )

    if args.color_kind == "auto":
        color_kind = infer_color_kind(
            source_files,
            all_color_values,
        )
    else:
        color_kind = args.color_kind

    if args.emphasize == "auto":
        emphasis = "low" if color_kind == "photonic" else "none"
    else:
        emphasis = args.emphasize

    norm = None
    cmap = None

    if has_colored_data:
        segment_color_arrays = [
            item.segment_colors
            for item in prepared
            if item.segment_colors is not None
        ]
        finite_colors = np.concatenate(
            [values[np.isfinite(values)] for values in segment_color_arrays]
        )

        if finite_colors.size == 0:
            raise ValueError(
                "No finite colored line segments remain in the selected "
                "energy window"
            )

        if color_kind == "photonic":
            default_min = 0.0
            default_max = 1.0
        else:
            default_min = float(np.min(finite_colors))
            default_max = float(np.max(finite_colors))

        color_min = (
            args.color_min
            if args.color_min is not None
            else default_min
        )
        color_max = (
            args.color_max
            if args.color_max is not None
            else default_max
        )

        if color_max <= color_min:
            color_max = color_min + 1.0

        norm = mpl.colors.Normalize(
            vmin=color_min,
            vmax=color_max,
            clip=True,
        )
        cmap_name = args.cmap or (
            "bwr" if args.dark_mode else "Spectral_r"
        )
        cmap = mpl.colormaps[cmap_name]

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    rasterize_dense_layers = not args.vector_points

    for item in prepared:
        if not args.hide_lines:
            segments = make_line_segments(
                item.x,
                item.energies,
                lower_limit,
                upper_limit,
            )

            if segments.size:
                line_collection = LineCollection(
                    segments,
                    colors=line_color,
                    linewidths=args.line_width,
                    alpha=args.line_alpha,
                    zorder=1,
                    rasterized=rasterize_dense_layers,
                )
                ax.add_collection(line_collection)

        if has_colored_data:
            if (
                item.colored_segments is None
                or item.segment_colors is None
                or norm is None
                or cmap is None
            ):
                raise RuntimeError(
                    "Colored band data was not prepared for rendering"
                )

            normalized = norm(item.segment_colors)
            order = rendering_order(normalized, emphasis)

            colored_segments = item.colored_segments[order]
            normalized = normalized[order]

            rgba = cmap(normalized)
            rgba[:, 3] = point_alpha(
                normalized,
                emphasis=emphasis,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                alpha_power=args.alpha_power,
                fade=not args.no_fade,
            )

            gradient_collection = LineCollection(
                colored_segments,
                colors=rgba,
                linewidths=args.gradient_line_width,
                zorder=2,
                rasterized=rasterize_dense_layers,
            )
            ax.add_collection(gradient_collection)

        elif args.uncolored:
            # NaN entries break the line automatically.
            for band_index in range(item.energies.shape[1]):
                ax.plot(
                    item.x,
                    item.energies[:, band_index],
                    color=uncolored_line_color,
                    linewidth=args.uncolored_line_width,
                    zorder=2,
                    rasterized=rasterize_dense_layers,
                )

    if has_colored_data:
        if norm is None or cmap is None:
            raise RuntimeError(
                "Colored band data was not prepared for the colorbar"
            )
        scalar_mappable = mpl.cm.ScalarMappable(
            norm=norm,
            cmap=cmap,
        )
        scalar_mappable.set_array([])

        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            pad=0.025,
            fraction=0.055,
            aspect=25,
        )

        colorbar.set_label(
            args.colorbar_label
            or default_colorbar_label(color_kind)
        )
        colorbar.ax.set_facecolor(background_color)
        colorbar.ax.yaxis.label.set_color(foreground_color)
        colorbar.outline.set_linewidth(0.6)
        colorbar.outline.set_edgecolor(foreground_color)
        colorbar.ax.tick_params(
            width=0.6,
            length=2.5,
            colors=foreground_color,
        )

        if color_kind == "photonic" and (
            args.color_min is None
            and args.color_max is None
        ):
            colorbar.set_ticks(
                np.linspace(0.0, 1.0, 6)
            )

    symmetry_positions = prepared[0].symmetry_positions
    symmetry_labels = prepared[0].symmetry_labels

    if symmetry_positions is not None and symmetry_labels is not None:
        add_symmetry_guides(
            ax,
            symmetry_positions,
            symmetry_labels,
            color="#4c4c4c" if args.dark_mode else "0.82",
        )

    ax.set_xlim(
        min(float(item.x[0]) for item in prepared),
        max(float(item.x[-1]) for item in prepared),
    )

    finite_energies = np.concatenate(
        [item.energies[np.isfinite(item.energies)] for item in prepared]
    )

    if finite_energies.size == 0:
        raise ValueError("The energy array has no finite values")

    data_min = float(np.min(finite_energies))
    data_max = float(np.max(finite_energies))

    energy_min = lower_limit if lower_limit is not None else data_min
    energy_max = upper_limit if upper_limit is not None else data_max

    if energy_max <= energy_min:
        raise ValueError(
            f"The upper energy limit ({energy_max:.6g}) must exceed the "
            f"lower energy limit ({energy_min:.6g})"
        )

    padding = 0.025 * (energy_max - energy_min)

    if padding == 0.0:
        padding = 1e-3

    # Pad only the sides that were inferred from the data. An explicit
    # --energy-min/--energy-max or the run's energy cutoff is used exactly.
    if lower_limit is None:
        energy_min -= padding

    if upper_limit is None:
        energy_max += padding

    ax.set_ylim(energy_min, energy_max)

    if args.xlabel is not None:
        ax.set_xlabel(args.xlabel, color=foreground_color)
    elif symmetry_positions is None:
        ax.set_xlabel(r"Wave vector", color=foreground_color)
    else:
        ax.set_xlabel("", color=foreground_color)

    ax.set_ylabel(args.ylabel, color=foreground_color)

    if args.title:
        ax.set_title(args.title, pad=5.0, color=foreground_color)

    for spine in ax.spines.values():
        spine.set_color(foreground_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        which="both",
        top=False,
        right=False,
        colors=foreground_color,
    )

    if args.y_minor_ticks:
        ax.yaxis.set_minor_locator(
            mpl.ticker.AutoMinorLocator()
        )

    output = (
        Path(args.output)
        if args.output is not None
        else default_output_path(source_files)
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=args.dpi,
        transparent=args.transparent,
    )

    print(f"Saved {output}")
    print(
        f"Plotted {len(prepared)} dataset(s), up to "
        f"{max(item.energies.shape[1] for item in prepared)} bands "
        f"(states per k-point: "
        f"{min(item.data.min_band_count for item in prepared)}.."
        f"{max(item.data.max_band_count for item in prepared)})"
    )

    finite_cutoffs = [
        cutoff for cutoff in energy_cutoffs if cutoff is not None
    ]

    if finite_cutoffs and not args.ignore_energy_cutoff:
        source = (
            "overridden by --energy-max"
            if args.energy_max is not None
            else "used as y-max"
        )
        if len(set(finite_cutoffs)) == 1:
            print(
                "Energy cutoff from filename: "
                f"{finite_cutoffs[0]:.6e} ({source})"
            )
        else:
            print(
                "Energy cutoffs from filenames: "
                + ", ".join(f"{value:.6e}" for value in finite_cutoffs)
            )

    print(f"y-range: [{energy_min:.6g}, {energy_max:.6g}]")

    if has_colored_data:
        print(
            f"Color kind: {color_kind}; "
            f"emphasis: {emphasis}"
        )

        if emphasis == "low":
            print(
                "High color values were drawn first; "
                "low values were drawn last/on top."
            )

    if args.show:
        plt.show()

    plt.close(fig)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality band-structure figures directly "
            "from Rust ColoredBandData .bin files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Band .bin files, directories, or quoted globs. Distributed "
            "chunk sets are combined; independent files are overlaid."
        ),
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory scanned for .bin files when no positional inputs are "
            "provided."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output PDF, SVG, or PNG filename. If omitted, derives a PDF "
            "name from the input data filename."
        ),
    )

    parser.add_argument(
        "--allow-incomplete-chunks",
        action="store_true",
        help="Allow a partial distributed chunk set.",
    )

    parser.add_argument(
        "--bands",
        type=int,
        default=None,
        help=(
            "Plot only the first N eigenvalue indices. By default every "
            "saved band index is kept (for energy-cutoff data, up to the "
            "largest per-k-point count; missing indices break the line)."
        ),
    )

    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="Lower plotted energy limit.",
    )

    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help=(
            "Upper plotted energy limit. If omitted and the input filename "
            "contains _ecut<E> (Rust --energy-cutoff), E is used."
        ),
    )

    parser.add_argument(
        "--ignore-energy-cutoff",
        action="store_true",
        help=(
            "Do not use the _ecut<E> value from the filename as the "
            "default upper energy limit."
        ),
    )

    parser.add_argument(
        "--path-mode",
        choices=["auto", "square", "distance"],
        default="auto",
        help=(
            "Use automatic square-path detection, force Γ-X-M-Γ ticks, "
            "or use raw cumulative path distance."
        ),
    )

    parser.add_argument(
        "--color-kind",
        choices=[
            "auto",
            "photonic",
            "n-expectation",
            "generic",
        ],
        default="auto",
        help="Interpretation of color_values.",
    )

    parser.add_argument(
        "--emphasize",
        choices=["auto", "low", "high", "none"],
        default="auto",
        help=(
            "Which color values should be drawn last and kept most opaque. "
            "Photonic-character plots default to low."
        ),
    )

    parser.add_argument(
        "--cmap",
        default=None,
        help=(
            "Matplotlib colormap name. Defaults to Spectral_r in light "
            "mode and bwr in dark mode."
        ),
    )

    parser.add_argument(
        "--color-min",
        type=float,
        default=None,
        help="Color normalization minimum.",
    )

    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="Color normalization maximum.",
    )

    parser.add_argument(
        "--colorbar-label",
        default=None,
        help="Override the automatic colorbar label.",
    )

    parser.add_argument(
        "--gradient-line-width",
        "--point-size",
        dest="gradient_line_width",
        type=float,
        default=0.8,
        help=(
            "Width of colored gradient band lines in points. The legacy "
            "--point-size spelling is retained as an alias."
        ),
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.05,
        help="Minimum opacity for de-emphasized points.",
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1.0,
        help="Maximum point opacity.",
    )

    parser.add_argument(
        "--alpha-power",
        type=float,
        default=1.5,
        help="Nonlinearity of composition-dependent opacity.",
    )

    parser.add_argument(
        "--no-fade",
        action="store_true",
        help=(
            "Disable composition-dependent opacity while retaining the "
            "requested rendering order."
        ),
    )

    parser.add_argument(
        "--hide-lines",
        action="store_true",
        help="Do not draw the faint connectivity layer.",
    )

    parser.add_argument(
        "--line-color",
        default=None,
        help=(
            "Color of faint connectivity lines. Defaults to dark gray in "
            "light mode and light gray in dark mode."
        ),
    )

    parser.add_argument(
        "--line-width",
        type=float,
        default=0.30,
        help="Width of faint connectivity lines.",
    )

    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.08,
        help="Opacity of faint connectivity lines.",
    )

    parser.add_argument(
        "--uncolored",
        action="store_true",
        help="Allow plotting files whose color rows are empty.",
    )

    parser.add_argument(
        "--uncolored-line-color",
        default=None,
        help=(
            "Band color for an uncolored plot. Defaults to near-black in "
            "light mode and near-white in dark mode."
        ),
    )

    parser.add_argument(
        "--uncolored-line-width",
        type=float,
        default=0.65,
        help="Line width for an uncolored plot.",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=3.45,
        help="Figure width in inches.",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=2.75,
        help="Figure height in inches.",
    )

    parser.add_argument(
        "--xlabel",
        default=None,
        help="Override the x-axis label.",
    )

    parser.add_argument(
        "--ylabel",
        default="Energy (a.u.)",
        help="Y-axis label.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help=(
            "Optional title. Publication figures commonly omit the title and "
            "put parameter information in the caption."
        ),
    )

    parser.add_argument(
        "--y-minor-ticks",
        action="store_true",
        help="Show minor ticks on the energy axis.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help=(
            "Resolution for PNG output and rasterized layers embedded in "
            "PDF/SVG output."
        ),
    )

    parser.add_argument(
        "--vector-lines",
        "--vector-points",
        dest="vector_points",
        action="store_true",
        help=(
            "Keep every colored line segment as a vector object. This can "
            "create very large PDFs for dense photon-mode calculations."
        ),
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save with a transparent figure background.",
    )

    parser.add_argument(
        "--dark-mode",
        "--dark",
        dest="dark_mode",
        action="store_true",
        help="Use a black figure and axes background with light annotations.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving.",
    )

    return parser

def validate_arguments(args: argparse.Namespace) -> None:
    if args.bands is not None and args.bands <= 0:
        raise ValueError("--bands must be positive")

    if (
        args.energy_min is not None
        and args.energy_max is not None
        and args.energy_max <= args.energy_min
    ):
        raise ValueError("--energy-max must exceed --energy-min")

    if args.gradient_line_width <= 0.0:
        raise ValueError("--gradient-line-width must be positive")

    if not 0.0 <= args.alpha_min <= 1.0:
        raise ValueError("--alpha-min must lie in [0,1]")

    if not 0.0 <= args.alpha_max <= 1.0:
        raise ValueError("--alpha-max must lie in [0,1]")

    if args.alpha_min > args.alpha_max:
        raise ValueError("--alpha-min cannot exceed --alpha-max")

    if args.alpha_power <= 0.0:
        raise ValueError("--alpha-power must be positive")

    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("Figure dimensions must be positive")

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args)
    plot_bands(args)


if __name__ == "__main__":
    main()
