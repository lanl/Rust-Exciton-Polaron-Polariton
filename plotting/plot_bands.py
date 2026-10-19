#!/usr/bin/env python3
"""
Publication-quality band-structure plotting for Rust ColoredBandData files.

For photonic-character plots, points are sorted in decreasing photonic
character before rendering. Consequently:

    highly photonic points are drawn first;
    low-photonic-character points are drawn last/on top.

Highly photonic points are also faded by default to suppress the dense sea of
photon modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from band_data import build_path_axis, load_band_inputs


ColorKind = Literal["photonic", "n-expectation", "generic"]


def configure_matplotlib() -> None:
    """Set publication-oriented Matplotlib defaults."""
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
            "axes.unicode_minus": True,
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


def prepare_point_data(
    x: np.ndarray,
    energies: np.ndarray,
    colors: np.ndarray,
    energy_min: float | None,
    energy_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and filter the colored band points."""
    x_values = np.broadcast_to(
        x[:, None],
        energies.shape,
    ).reshape(-1)

    energy_values = energies.reshape(-1)
    color_values = colors.reshape(-1)

    valid = (
        np.isfinite(x_values)
        & np.isfinite(energy_values)
        & np.isfinite(color_values)
    )

    if energy_min is not None:
        valid &= energy_values >= energy_min

    if energy_max is not None:
        valid &= energy_values <= energy_max

    return (
        x_values[valid],
        energy_values[valid],
        color_values[valid],
    )


def rendering_order(
    normalized_colors: np.ndarray,
    emphasis: str,
) -> np.ndarray:
    """
    Return the point rendering order.

    emphasis="low":
        High values first, low values last/on top.

    emphasis="high":
        Low values first, high values last/on top.

    emphasis="none":
        Retain the existing flattened order.
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
    """Calculate opacity values used for composition emphasis."""
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
) -> None:
    """Add high-symmetry ticks and unobtrusive vertical guides."""
    for position in positions:
        ax.axvline(
            position,
            color="0.82",
            linewidth=0.55,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def plot_bands(args: argparse.Namespace) -> None:
    data = load_band_inputs(
        args.inputs,
        require_complete_chunks=not args.allow_incomplete_chunks,
    )

    # Energy-cutoff runs have a different number of states at each k-point.
    # Missing band indices are NaN-padded so lines break rather than the
    # band being truncated to the minimum count.
    k_points, energies, colors = data.padded_arrays(
        max_bands=args.bands,
    )

    energy_cutoff = (
        None if args.ignore_energy_cutoff else data.energy_cutoff
    )

    # Effective window limits used for both data filtering and the y-axis.
    lower_limit = args.energy_min
    upper_limit = (
        args.energy_max
        if args.energy_max is not None
        else energy_cutoff
    )

    x, symmetry_positions, symmetry_labels = build_path_axis(
        k_points,
        mode=args.path_mode,
    )

    configure_matplotlib()

    if args.color_kind == "auto":
        color_kind = infer_color_kind(
            data.source_files,
            colors,
        )
    else:
        color_kind = args.color_kind

    if args.emphasize == "auto":
        emphasis = "low" if color_kind == "photonic" else "none"
    else:
        emphasis = args.emphasize

    if colors is None and not args.uncolored:
        raise ValueError(
            "This file contains empty color rows. Use --uncolored to make "
            "a conventional uncolored band plot."
        )

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        constrained_layout=True,
    )

    rasterize_dense_layers = not args.vector_points

    if not args.hide_lines:
        segments = make_line_segments(
            x,
            energies,
            lower_limit,
            upper_limit,
        )

        if segments.size:
            line_collection = LineCollection(
                segments,
                colors=args.line_color,
                linewidths=args.line_width,
                alpha=args.line_alpha,
                zorder=1,
                rasterized=rasterize_dense_layers,
            )
            ax.add_collection(line_collection)

    if colors is not None:
        x_points, energy_points, color_points = prepare_point_data(
            x,
            energies,
            colors,
            lower_limit,
            upper_limit,
        )

        if x_points.size == 0:
            raise ValueError(
                "No finite colored points remain in the selected energy window"
            )

        finite_colors = color_points[np.isfinite(color_points)]

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
        cmap = mpl.colormaps[args.cmap]

        normalized = norm(color_points)
        order = rendering_order(normalized, emphasis)

        x_points = x_points[order]
        energy_points = energy_points[order]
        color_points = color_points[order]
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

        ax.scatter(
            x_points,
            energy_points,
            s=args.point_size,
            c=rgba,
            marker="o",
            edgecolors="none",
            linewidths=0.0,
            zorder=2,
            rasterized=rasterize_dense_layers,
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
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(
            width=0.6,
            length=2.5,
        )

        if color_kind == "photonic" and (
            args.color_min is None
            and args.color_max is None
        ):
            colorbar.set_ticks(
                np.linspace(0.0, 1.0, 6)
            )

    elif args.uncolored:
        # NaN entries break the line automatically.
        for band_index in range(energies.shape[1]):
            ax.plot(
                x,
                energies[:, band_index],
                color=args.uncolored_line_color,
                linewidth=args.uncolored_line_width,
                zorder=2,
                rasterized=rasterize_dense_layers,
            )

    if symmetry_positions is not None and symmetry_labels is not None:
        add_symmetry_guides(
            ax,
            symmetry_positions,
            symmetry_labels,
        )

    ax.set_xlim(float(x[0]), float(x[-1]))

    finite_energies = energies[np.isfinite(energies)]

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
        ax.set_xlabel(args.xlabel)
    elif symmetry_positions is None:
        ax.set_xlabel(r"Wave vector")
    else:
        ax.set_xlabel("")

    ax.set_ylabel(args.ylabel)

    if args.title:
        ax.set_title(args.title, pad=5.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        which="both",
        top=False,
        right=False,
    )

    if args.y_minor_ticks:
        ax.yaxis.set_minor_locator(
            mpl.ticker.AutoMinorLocator()
        )

    output = Path(args.output)
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
        f"Plotted up to {energies.shape[1]} bands at "
        f"{energies.shape[0]} k-points "
        f"(states per k-point: {data.min_band_count}..{data.max_band_count})"
    )

    if energy_cutoff is not None:
        source = (
            "overridden by --energy-max"
            if args.energy_max is not None
            else "used as y-max"
        )
        print(f"Energy cutoff from filename: {energy_cutoff:.6e} ({source})")

    print(f"y-range: [{energy_min:.6g}, {energy_max:.6g}]")

    if colors is not None:
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
        nargs="+",
        help=(
            "One gathered .bin file or a set/glob of distributed chunk files. "
            "Quoted wildcard patterns are supported."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default="band_structure.pdf",
        help="Output PDF, SVG, or PNG filename.",
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
        default="Spectral_r",
        help="Matplotlib colormap name.",
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
        "--point-size",
        type=float,
        default=3.0,
        help="Scatter marker area in points squared.",
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
        default="0.25",
        help="Color of faint connectivity lines.",
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
        default="#222222",
        help="Band color for an uncolored plot.",
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
        "--vector-points",
        action="store_true",
        help=(
            "Keep every point and line segment as a vector object. This can "
            "create very large PDFs for dense photon-mode calculations."
        ),
    )

    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save with a transparent figure background.",
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

    if args.point_size <= 0.0:
        raise ValueError("--point-size must be positive")

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