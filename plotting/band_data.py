#!/usr/bin/env python3
"""
Reader for band-structure files written by:

    sim_common::data::ColoredBandData

Rust schema:

    #[derive(Serialize, Deserialize)]
    pub struct ColoredBandData {
        pub k_points: Vec<Vec<f64>>,
        pub eigenvalues: Vec<Vec<f64>>,
        pub color_values: Vec<Vec<f64>>,
    }

The files are written with bincode 1.3 using:

    bincode::serialize(value)

or equivalently through:

    sim_common::cache::save_bincode_atomic(...)

For bincode 1.3's top-level serialize/deserialize helpers, the relevant
encoding is:

    - little endian
    - fixed-width integer encoding
    - Vec lengths represented as u64
    - f64 represented as IEEE-754 little-endian values

This reader is specifically for ColoredBandData files. It is not a generic
bincode reader and should not be used for matrix-element caches or dielectric
cache files.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
import re
import struct
from typing import Iterable, Sequence

import numpy as np


class BincodeDecodeError(RuntimeError):
    """The file does not match the expected ColoredBandData schema."""


@dataclass
class ColoredBandData:
    """Python representation of sim_common::data::ColoredBandData."""

    k_points: list[np.ndarray]
    eigenvalues: list[np.ndarray]
    color_values: list[np.ndarray]
    source_files: tuple[Path, ...] = ()

    def validate(self) -> None:
        """Check the structural consistency of the decoded data."""
        n_k = len(self.k_points)

        if n_k == 0:
            raise BincodeDecodeError("The file contains no k-points")

        if len(self.eigenvalues) != n_k:
            raise BincodeDecodeError(
                f"Found {n_k} k-points but "
                f"{len(self.eigenvalues)} eigenvalue rows"
            )

        if len(self.color_values) != n_k:
            raise BincodeDecodeError(
                f"Found {n_k} k-points but "
                f"{len(self.color_values)} color-value rows"
            )

        for k_index, point in enumerate(self.k_points):
            if point.ndim != 1:
                raise BincodeDecodeError(
                    f"k-point {k_index} is not one-dimensional"
                )

            if point.size < 2:
                raise BincodeDecodeError(
                    f"k-point {k_index} has dimension {point.size}; "
                    "expected at least two components"
                )

            if not np.all(np.isfinite(point)):
                raise BincodeDecodeError(
                    f"k-point {k_index} contains a non-finite value"
                )

        for k_index, (energies, colors) in enumerate(
            zip(self.eigenvalues, self.color_values)
        ):
            if energies.ndim != 1 or colors.ndim != 1:
                raise BincodeDecodeError(
                    f"Data row {k_index} is not one-dimensional"
                )

            if colors.size not in (0, energies.size):
                raise BincodeDecodeError(
                    f"k-point {k_index} has {energies.size} eigenvalues "
                    f"but {colors.size} color values"
                )

    @property
    def n_k_points(self) -> int:
        return len(self.k_points)

    @property
    def min_band_count(self) -> int:
        if not self.eigenvalues:
            return 0
        return min(row.size for row in self.eigenvalues)

    @property
    def max_band_count(self) -> int:
        if not self.eigenvalues:
            return 0
        return max(row.size for row in self.eigenvalues)

    @property
    def is_ragged(self) -> bool:
        """True when the number of states differs between k-points."""
        return self.min_band_count != self.max_band_count

    @property
    def has_complete_colors(self) -> bool:
        return bool(self.eigenvalues) and all(
            colors.size == energies.size
            for energies, colors in zip(
                self.eigenvalues,
                self.color_values,
            )
        )

    @property
    def energy_cutoff(self) -> float | None:
        """
        Energy cutoff encoded in the source filenames (``_ecut...``).

        Returns None for ``_neigN`` files. Raises if the inputs disagree,
        which would indicate that two different runs were mixed.
        """
        if not self.source_files:
            return None

        cutoffs = [parse_energy_cutoff(path) for path in self.source_files]
        unique = set(cutoffs)

        if len(unique) > 1:
            raise ValueError(
                "Input files disagree on the energy cutoff: "
                f"{sorted(str(value) for value in unique)}"
            )

        return cutoffs[0]

    def k_array(self) -> np.ndarray:
        """
        Return k-points as a dense array.

        Raises ValueError if different k-points have different dimensions.
        """
        dimensions = {point.size for point in self.k_points}

        if len(dimensions) != 1:
            raise ValueError(
                f"Ragged k-point dimensions: {sorted(dimensions)}"
            )

        return np.stack(self.k_points, axis=0)

    def dense_arrays(
        self,
        max_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Return dense arrays truncated to the band count available at EVERY
        k-point.

        For energy-cutoff data (ragged rows) this can discard most states;
        use padded_arrays() instead.
        """
        self.validate()

        n_bands = self.min_band_count

        if max_bands is not None:
            if max_bands <= 0:
                raise ValueError("max_bands must be positive")
            n_bands = min(n_bands, max_bands)

        if n_bands == 0:
            raise ValueError(
                "No band index is available at every k-point"
                + (" (ragged energy-cutoff data: use padded_arrays())"
                   if self.is_ragged else "")
            )

        k_points = self.k_array()
        energies = np.stack(
            [row[:n_bands] for row in self.eigenvalues],
            axis=0,
        )

        if self.has_complete_colors:
            colors = np.stack(
                [row[:n_bands] for row in self.color_values],
                axis=0,
            )
        else:
            colors = None

        return k_points, energies, colors

    def padded_arrays(
        self,
        max_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Return arrays of shape (n_k, n_bands) with n_bands = max band count.

        Band indices that do not exist at a k-point (fewer states below the
        energy cutoff) are filled with NaN, so plotted lines break there
        instead of the whole band being dropped.
        """
        self.validate()

        n_bands = self.max_band_count

        if max_bands is not None:
            if max_bands <= 0:
                raise ValueError("max_bands must be positive")
            n_bands = min(n_bands, max_bands)

        if n_bands == 0:
            raise ValueError("No bands are available at any k-point")

        k_points = self.k_array()
        n_k = len(self.eigenvalues)

        energies = np.full((n_k, n_bands), np.nan, dtype=float)

        for k_index, row in enumerate(self.eigenvalues):
            n_keep = min(row.size, n_bands)
            energies[k_index, :n_keep] = row[:n_keep]

        colors: np.ndarray | None = None

        if self.has_complete_colors:
            colors = np.full((n_k, n_bands), np.nan, dtype=float)

            for k_index, row in enumerate(self.color_values):
                n_keep = min(row.size, n_bands)
                colors[k_index, :n_keep] = row[:n_keep]

        return k_points, energies, colors

class _FixedBincodeReader:
    """Minimal fixed-integer bincode 1.3 reader."""

    def __init__(
        self,
        data: bytes,
        *,
        max_outer_rows: int = 10_000_000,
        max_inner_values: int = 100_000_000,
    ):
        self.data = data
        self.offset = 0
        self.max_outer_rows = max_outer_rows
        self.max_inner_values = max_inner_values

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def read_u64(self) -> int:
        if self.remaining < 8:
            raise BincodeDecodeError(
                "Unexpected end of file while reading a u64 "
                f"at byte offset {self.offset}"
            )

        value = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return value

    def read_vec_f64(self) -> np.ndarray:
        length = self.read_u64()

        if length > self.max_inner_values:
            raise BincodeDecodeError(
                f"Unreasonable inner vector length {length} "
                f"at byte offset {self.offset - 8}"
            )

        n_bytes = length * 8

        if n_bytes > self.remaining:
            raise BincodeDecodeError(
                f"Vector claims to contain {length} f64 values "
                f"({n_bytes} bytes), but only {self.remaining} bytes remain"
            )

        values = np.frombuffer(
            self.data,
            dtype="<f8",
            count=length,
            offset=self.offset,
        ).copy()

        self.offset += n_bytes
        return values

    def read_nested_vec_f64(self) -> list[np.ndarray]:
        outer_length = self.read_u64()

        if outer_length > self.max_outer_rows:
            raise BincodeDecodeError(
                f"Unreasonable outer vector length {outer_length}"
            )

        # Each inner Vec needs at least an eight-byte length field.
        if outer_length > self.remaining // 8:
            raise BincodeDecodeError(
                f"Outer vector claims to contain {outer_length} rows, "
                f"but only {self.remaining} bytes remain"
            )

        return [
            self.read_vec_f64()
            for _ in range(outer_length)
        ]


def load_colored_band_file(
    filename: str | Path,
    *,
    allow_trailing_bytes: bool = False,
) -> ColoredBandData:
    """
    Load one ColoredBandData bincode file.

    This handles both:

    - light-matter photonic-character band files;
    - exciton n-colored band files;
    - uncolored exciton band files saved with empty color rows.
    """
    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(filename)

    reader = _FixedBincodeReader(filename.read_bytes())

    result = ColoredBandData(
        k_points=reader.read_nested_vec_f64(),
        eigenvalues=reader.read_nested_vec_f64(),
        color_values=reader.read_nested_vec_f64(),
        source_files=(filename,),
    )

    if reader.remaining != 0 and not allow_trailing_bytes:
        raise BincodeDecodeError(
            f"{filename} contains {reader.remaining} unparsed trailing bytes. "
            "It may have a different schema or bincode configuration."
        )

    result.validate()
    return result


_CHUNK_PATTERN = re.compile(
    r"_chunk(?P<index>\d+)_of(?P<total>\d+)(?:_zoomed)?\.bin$"
)


def _band_stem_without_chunk(path: str | Path) -> str:
    path = Path(path)
    match = _CHUNK_PATTERN.search(path.name)

    if match is None:
        return path.stem

    stem = path.name[:match.start()]

    if stem.startswith("color_band_chunk_"):
        stem = "color_band_" + stem[len("color_band_chunk_"):]

    return stem


def chunk_information(path: str | Path) -> tuple[int, int] | None:
    """Return (chunk_index, total_chunks) from a chunk filename."""
    path = Path(path)
    match = _CHUNK_PATTERN.search(path.name)

    if match is None:
        return None

    return (
        int(match.group("index")),
        int(match.group("total")),
    )

# Rust writes the cutoff tag as  _ecut{:.6e}  e.g. "_ecut-1.000000e-2"
# followed by "_zoomed", "_chunkN_ofM", or ".bin".
_ECUT_PATTERN = re.compile(
    r"_ecut(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?=_|\.bin$)"
)


def parse_energy_cutoff(path: str | Path) -> float | None:
    """
    Return the energy cutoff encoded in a band filename, or None if the
    file was produced with plain --n-eig (``_neigN``).
    """
    match = _ECUT_PATTERN.search(Path(path).name)

    if match is None:
        return None

    return float(match.group("value"))

def concatenate_band_data(
    datasets: Sequence[ColoredBandData],
) -> ColoredBandData:
    """Concatenate band datasets along the k-point dimension."""
    if not datasets:
        raise ValueError("No datasets were provided")

    k_points: list[np.ndarray] = []
    eigenvalues: list[np.ndarray] = []
    color_values: list[np.ndarray] = []
    source_files: list[Path] = []

    for data in datasets:
        data.validate()
        k_points.extend(data.k_points)
        eigenvalues.extend(data.eigenvalues)
        color_values.extend(data.color_values)
        source_files.extend(data.source_files)

    result = ColoredBandData(
        k_points=k_points,
        eigenvalues=eigenvalues,
        color_values=color_values,
        source_files=tuple(source_files),
    )
    result.validate()
    return result


def load_chunk_files(
    filenames: Sequence[str | Path],
    *,
    require_complete: bool = True,
) -> ColoredBandData:
    """
    Load and combine light-matter distributed chunk files.

    Chunk files are sorted numerically by `_chunkN_ofM`, not lexicographically.
    """
    paths = [Path(path) for path in filenames]

    if not paths:
        raise ValueError("No chunk files were provided")

    information = []

    for path in paths:
        info = chunk_information(path)

        if info is None:
            raise ValueError(
                f"{path} does not look like a distributed chunk file"
            )

        information.append((path, info[0], info[1]))

    totals = {total for _, _, total in information}

    if len(totals) != 1:
        raise ValueError(
            f"Chunk files disagree about total chunk count: {sorted(totals)}"
        )

    total_chunks = totals.pop()
    indices = [index for _, index, _ in information]

    if len(indices) != len(set(indices)):
        raise ValueError(
            "Duplicate chunk indices were supplied. The glob may include "
            "multiple parameter sets."
        )

    if require_complete:
        expected = set(range(total_chunks))
        actual = set(indices)

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)

            raise ValueError(
                f"Incomplete chunk set. Missing={missing}, extra={extra}"
            )

    information.sort(key=lambda item: item[1])

    datasets = [
        load_colored_band_file(path)
        for path, _, _ in information
    ]

    return concatenate_band_data(datasets)


def expand_input_patterns(
    inputs: Iterable[str | Path],
    input_dir: str | Path | None = None,
) -> list[Path]:
    """
    Expand files, directories, and shell-style wildcard patterns.

    Quoted globs are supported, for example:

        "pp_data/color_band_chunk_*_chunk*_of16.bin"
    """
    items = list(inputs)

    if not items and input_dir is not None:
        items = [input_dir]

    paths: list[Path] = []

    for item in items:
        text = str(item)
        item_path = Path(item)

        if any(character in text for character in "*?[]"):
            matches = [Path(path) for path in glob(text)]

            if not matches:
                raise FileNotFoundError(
                    f"Input pattern matched no files: {text}"
                )

            paths.extend(matches)
        elif item_path.is_dir():
            matches = (
                set(item_path.glob("color_band_*.bin"))
                | set(item_path.glob("n_colored_band_*.bin"))
            )

            if not matches:
                matches = set(item_path.glob("*.bin"))

            paths.extend(sorted(matches))
        else:
            paths.append(item_path)

    # Resolve exact duplicates while retaining deterministic order.
    unique: dict[str, Path] = {}

    for path in paths:
        if path.is_file() and path.suffix == ".bin":
            unique[str(path)] = path

    if not unique:
        checked = ", ".join(str(item) for item in items) or "(none)"
        raise FileNotFoundError(
            "No band .bin files were found. "
            f"Checked: {checked}"
        )

    return sorted(unique.values(), key=lambda path: str(path))


def load_band_datasets(
    inputs: Sequence[str | Path],
    *,
    input_dir: str | Path | None = None,
    require_complete_chunks: bool = True,
) -> list[ColoredBandData]:
    """
    Load one or more independent band datasets.

    A complete set of distributed chunk files is combined into one dataset.
    Multiple non-chunk files are returned as separate datasets so callers can
    overlay parameter scans without connecting one path to the next.
    """
    paths = expand_input_patterns(inputs, input_dir=input_dir)

    chunk_paths = [
        path
        for path in paths
        if chunk_information(path) is not None
    ]

    nonchunk_paths = [
        path
        for path in paths
        if chunk_information(path) is None
    ]
    datasets = [load_colored_band_file(path) for path in nonchunk_paths]

    gathered_keys = {
        (str(path.parent), path.stem)
        for path in nonchunk_paths
    }

    chunk_groups: dict[tuple[str, str], list[Path]] = {}

    for path in chunk_paths:
        key = (
            str(path.parent),
            _band_stem_without_chunk(path),
        )
        chunk_groups.setdefault(key, []).append(path)

    for key, group in chunk_groups.items():
        if key in gathered_keys:
            continue

        datasets.append(
            load_chunk_files(
                sorted(group),
                require_complete=require_complete_chunks,
            )
        )

    return sorted(
        datasets,
        key=lambda data: str(data.source_files[0]),
    )


def load_band_inputs(
    inputs: Sequence[str | Path],
    *,
    input_dir: str | Path | None = None,
    require_complete_chunks: bool = True,
) -> ColoredBandData:
    """
    Load one gathered file or a collection of distributed chunk files.

    Multiple non-chunk files are concatenated in the supplied order. Normally,
    publication plots should use either one gathered file or one complete chunk
    set.
    """
    datasets = load_band_datasets(
        inputs,
        input_dir=input_dir,
        require_complete_chunks=require_complete_chunks,
    )

    if len(datasets) == 1:
        return datasets[0]

    return concatenate_band_data(datasets)


def cumulative_path_distances(k_points: np.ndarray) -> np.ndarray:
    """Reproduce sim_common::kpath::path_distances."""
    if k_points.ndim != 2 or k_points.shape[1] < 2:
        raise ValueError("k_points must have shape (n_k, at least 2)")

    if len(k_points) == 0:
        return np.empty(0, dtype=float)

    differences = np.diff(k_points[:, :2], axis=0)
    increments = np.linalg.norm(differences, axis=1)

    return np.concatenate(
        [np.array([0.0]), np.cumsum(increments)]
    )


def looks_like_closed_path(k_points: np.ndarray) -> bool:
    """Determine whether the path returns to its initial k-point."""
    if len(k_points) < 4:
        return False

    extent = np.max(
        np.linalg.norm(k_points[:, :2], axis=1),
        initial=0.0,
    )
    tolerance = 1e-8 * max(1.0, extent)

    return (
        np.linalg.norm(k_points[-1, :2] - k_points[0, :2])
        <= tolerance
    )


def square_path_axis(
    k_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Construct the intended Γ-X-M-Γ path coordinate.

    Rust's closed_polyline_path uses equal numbers of interpolation intervals
    for the three segments, even though M->Γ is geometrically longer. If a
    corner is not sampled exactly, ordinary cumulative point-to-point distance
    cuts slightly across the corner. This routine reconstructs the intended
    piecewise path coordinate from the interpolation index.

    Returns
    -------
    x:
        Coordinate for every sampled k-point.
    tick_positions:
        Exact Γ, X, M, Γ positions.
    tick_labels:
        Matplotlib labels.
    """
    if len(k_points) < 2:
        raise ValueError("At least two k-points are required")

    half_bz = float(np.max(np.abs(k_points[:, :2])))

    if half_bz <= 0.0:
        raise ValueError(
            "Cannot infer the square-lattice scale from zero k-points"
        )

    n_k = len(k_points)
    u = np.linspace(0.0, 3.0, n_k)

    segment = np.minimum(np.floor(u).astype(int), 2)
    local_t = u - segment

    segment_lengths = np.array(
        [half_bz, half_bz, np.sqrt(2.0) * half_bz]
    )
    cumulative = np.array(
        [
            0.0,
            segment_lengths[0],
            segment_lengths[0] + segment_lengths[1],
        ]
    )

    x = cumulative[segment] + local_t * segment_lengths[segment]

    ticks = np.array(
        [
            0.0,
            half_bz,
            2.0 * half_bz,
            (2.0 + np.sqrt(2.0)) * half_bz,
        ]
    )

    labels = [
        r"$\Gamma$",
        r"$X$",
        r"$M$",
        r"$\Gamma$",
    ]

    return x, ticks, labels


def build_path_axis(
    k_points: np.ndarray,
    mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    """
    Build a band-plot x-axis.

    mode="auto":
        Use the reconstructed Γ-X-M-Γ coordinate for a closed path, otherwise
        use cumulative distance.

    mode="square":
        Force the reconstructed Γ-X-M-Γ coordinate.

    mode="distance":
        Always use cumulative distance and do not add symmetry labels.
    """
    if mode not in {"auto", "square", "distance"}:
        raise ValueError(f"Unknown path-axis mode: {mode}")

    if mode == "square" or (
        mode == "auto" and looks_like_closed_path(k_points)
    ):
        return square_path_axis(k_points)

    return cumulative_path_distances(k_points), None, None
