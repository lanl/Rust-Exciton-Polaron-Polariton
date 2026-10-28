"""Inspect one or more ColoredBandData files or collections of chunks."""

from __future__ import annotations

import argparse

import numpy as np

from band_data import ColoredBandData, load_band_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Rust ColoredBandData bincode files. Inputs may be "
            "files, directories, quoted globs, or distributed chunk sets."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Input .bin files, directories, or quoted globs."
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
        "--allow-incomplete-chunks",
        action="store_true",
        help="Permit inspection of only part of a chunk set.",
    )

    return parser


def _concat_finite(rows: list[np.ndarray]) -> np.ndarray:
    parts = [row[np.isfinite(row)] for row in rows if row.size]
    if not parts:
        return np.empty(0, dtype=float)
    return np.concatenate(parts)


def inspect_dataset(
    data: ColoredBandData,
    dataset_index: int,
    dataset_count: int,
) -> None:
    if dataset_count > 1:
        print(f"Dataset {dataset_index}/{dataset_count}")
        print()

    print("Decoded ColoredBandData successfully")
    print()

    print("Source files:")
    for path in data.source_files:
        print(f"  {path}")

    energy_cutoff = data.energy_cutoff

    print()
    print(f"k-points:             {data.n_k_points}")
    print(
        "eigenstate selection: "
        + (
            f"all states with E <= {energy_cutoff:.6e} (energy cutoff)"
            if energy_cutoff is not None
            else "lowest n_eig states (no energy cutoff in filename)"
        )
    )
    print(f"minimum band count:   {data.min_band_count}")
    print(f"maximum band count:   {data.max_band_count}")
    print(f"ragged rows:          {data.is_ragged}")
    print(f"complete color data:  {data.has_complete_colors}")

    counts = np.array([row.size for row in data.eigenvalues])
    n_empty = int(np.count_nonzero(counts == 0))

    if n_empty:
        print(f"warning: {n_empty} k-point(s) have no stored states")

    if data.is_ragged:
        print(
            "states per k-point:   "
            f"mean {counts.mean():.1f}, median {np.median(counts):.0f}"
        )

    k_points = data.k_array()

    print(f"k-point array shape:  {k_points.shape}")
    print(f"first k-point:        {k_points[0]}")
    print(f"last k-point:         {k_points[-1]}")

    finite_energies = _concat_finite(data.eigenvalues)

    if finite_energies.size:
        print(
            "energy range:          "
            f"[{finite_energies.min():.12g}, "
            f"{finite_energies.max():.12g}]"
        )

        if energy_cutoff is not None and finite_energies.max() > energy_cutoff:
            print(
                "warning: some stored energies exceed the filename cutoff "
                f"({finite_energies.max():.6e} > {energy_cutoff:.6e})"
            )
    else:
        print("energy range:          (no finite energies)")

    if data.has_complete_colors:
        finite_colors = _concat_finite(data.color_values)

        if finite_colors.size:
            print(
                "color-value range:     "
                f"[{finite_colors.min():.12g}, "
                f"{finite_colors.max():.12g}]"
            )

            outside_photonic_range = np.count_nonzero(
                (finite_colors < -1e-10)
                | (finite_colors > 1.0 + 1e-10)
            )

            if outside_photonic_range:
                print(
                    "note: color values are not restricted to [0,1]; "
                    "this may be an n-expectation file rather than a "
                    "photonic-character file."
                )

    row_lengths = sorted({row.size for row in data.eigenvalues})
    if len(row_lengths) > 12:
        print(
            f"eigenvalue row sizes: {len(row_lengths)} distinct values, "
            f"{row_lengths[0]}..{row_lengths[-1]}"
        )
    else:
        print(f"eigenvalue row sizes: {row_lengths}")

    color_lengths = sorted({row.size for row in data.color_values})
    if len(color_lengths) > 12:
        print(
            f"color row sizes:      {len(color_lengths)} distinct values, "
            f"{color_lengths[0]}..{color_lengths[-1]}"
        )
    else:
        print(f"color row sizes:      {color_lengths}")


def main() -> None:
    args = build_parser().parse_args()

    datasets = load_band_datasets(
        args.inputs,
        input_dir=args.input_dir,
        require_complete_chunks=not args.allow_incomplete_chunks,
    )

    for index, data in enumerate(datasets, start=1):
        inspect_dataset(data, index, len(datasets))

        if index < len(datasets):
            print()

    if len(datasets) > 1:
        print(f"Inspected {len(datasets)} band datasets")


if __name__ == "__main__":
    main()
