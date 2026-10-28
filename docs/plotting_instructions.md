# Plotting instructions

This guide shows how to make plots from data that already exists on disk. It
does not recalculate band structures or dielectric functions. Run the commands
from the repository root unless a command says otherwise.

The examples use:

- `plotting/plot_bands.py` for exciton and polariton `.bin` band-data files;
- `plotting/inspect_band_file.py` to check a band-data file before plotting;
- `plotting/plot_dielectric.py` for dielectric `.dat` files.

## Prerequisites

The Python band plotter needs NumPy and Matplotlib:

```bash
python3 -c 'import numpy, matplotlib; print("NumPy and Matplotlib are available")'
```

For headless machines, configure Matplotlib and create a directory for plots:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/rust-exciton-mplconfig}"
mkdir -p "$MPLCONFIGDIR" plots
```

The band inspection and plotting scripts accept individual `.bin` files,
directories, and quoted globs. A complete distributed chunk set is combined
into one band dataset; independent files are inspected or plotted as separate
overlaid datasets. The plotter writes PDF, SVG, and PNG output; the format is
selected by the output filename.

When `--output` is omitted, `plot_bands.py` derives a PDF filename from the
input data filename. For example, `color_band_a0..._phcoup0.100000.bin`
becomes `color_band_a0..._phcoup0.100000.pdf`. Distributed chunk suffixes are
removed from the default name. When several independent files are supplied,
the first input stem receives a `_set` suffix.

The Python dielectric plotter uses the same NumPy and Matplotlib environment as
the band plotter and does not require MPI, PETSc, or SLEPc.

Dark mode is enabled with `--dark-mode` (or `--dark`). It uses a black
background, light annotations, and defaults to the `bwr` colormap. Pass
`--cmap NAME` to override the colormap.

## Exciton colored bands

The exciton colored-band files are named like:

```text
n_colored_band_a0..._V0..._nrel..._nbands..._npoints....bin
```

These files contain exciton eigenvalues colored by the principal-quantum-number
expectation value. The following command selects the first matching file in
`$EXCITON_DATA_DIR`; if the directory contains several parameter sets, list
the files first and replace `EXCITON_FILE` with the desired exact path.

```bash
EXCITON_DATA_DIR="${EXCITON_DATA_DIR:-$PWD/saved_data_ex}"
EXCITON_FILE="$({
  find "$EXCITON_DATA_DIR" -maxdepth 1 -type f \
    -name 'n_colored_band_*.bin' -print
} | sort | head -n 1)"
test -n "$EXCITON_FILE" || {
  echo "No n_colored_band_*.bin file found in $EXCITON_DATA_DIR" >&2
  exit 1
}

python3 plotting/inspect_band_file.py "$EXCITON_FILE"
python3 plotting/plot_bands.py "$EXCITON_FILE" \
  --color-kind n-expectation \
  --dark-mode \
  --output plots/exciton_colored_bands.pdf
```

To inspect or overlay every `.bin` file in a directory instead, pass the
directory directly. The default plot name is derived from the first sorted
input filename when `--output` is omitted:

```bash
python3 plotting/inspect_band_file.py "$EXCITON_DATA_DIR"
python3 plotting/plot_bands.py "$EXCITON_DATA_DIR" \
  --color-kind n-expectation \
  --dark-mode
```

For a PNG preview, change the final output option to:

```bash
python3 plotting/plot_bands.py "$EXCITON_FILE" \
  --color-kind n-expectation \
  --dark-mode \
  --dpi 300 \
  --output plots/exciton_colored_bands.png
```

If the filename contains an `_ecut...` tag, the Python plotter uses that value
as the default upper energy limit. Use `--ignore-energy-cutoff` or provide
`--energy-max VALUE` when a different window is needed.

## Polariton band structure

The light-matter band-data files are named like:

```text
color_band_a0..._V0..._nrel..._nbands..._npoints..._phcoup..._nmodes...bin
```

After chunk gathering, prefer the non-chunk gathered file. The command below
selects one gathered file and explicitly identifies its color values as
photonic character:

```bash
POLARITON_DATA_DIR="${POLARITON_DATA_DIR:-$PWD/pp_data}"
POLARITON_FILE="$({
  find "$POLARITON_DATA_DIR" -maxdepth 1 -type f \
    -name 'color_band_*.bin' \
    ! -name 'color_band_chunk_*.bin' -print
} | sort | head -n 1)"
test -n "$POLARITON_FILE" || {
  echo "No gathered color_band_*.bin file found in $POLARITON_DATA_DIR" >&2
  exit 1
}

python3 plotting/inspect_band_file.py "$POLARITON_FILE"
python3 plotting/plot_bands.py "$POLARITON_FILE" \
  --color-kind photonic \
  --dark-mode \
  --output plots/polariton_bands.pdf
```

The same scripts can inspect or plot a directory of independent gathered
files. Each file is drawn on the same axes, while a directory containing one
complete chunk set is combined before plotting:

```bash
python3 plotting/inspect_band_file.py --input-dir "$POLARITON_DATA_DIR"
python3 plotting/plot_bands.py --input-dir "$POLARITON_DATA_DIR" \
  --color-kind photonic \
  --dark-mode \
  --output plots/polariton_bands_all.pdf
```

If only distributed chunks are available, construct a run-specific glob from
one chunk. This avoids accidentally combining chunks from different parameter
sets:

```bash
FIRST_CHUNK="$({
  find "$POLARITON_DATA_DIR" -maxdepth 1 -type f \
    -name 'color_band_chunk_*.bin' -print
} | sort | head -n 1)"
test -n "$FIRST_CHUNK" || {
  echo "No color-band chunk found in $POLARITON_DATA_DIR" >&2
  exit 1
}

CHUNK_PREFIX="${FIRST_CHUNK%_chunk*_of*.bin}"
POLARITON_CHUNKS="${CHUNK_PREFIX}_chunk*_of*.bin"

python3 plotting/plot_bands.py "$POLARITON_CHUNKS" \
  --color-kind photonic \
  --dark-mode \
  --output plots/polariton_bands_from_chunks.pdf
```

The default behavior requires every chunk in the set. Add
`--allow-incomplete-chunks` only when intentionally plotting a partial set.

For a band file produced by a pipeline run, the gathered path is recorded in
`pipeline_runs/<run-id>/gathered_band_file.txt`:

```bash
RUN_ID="<run-id>"
POLARITON_FILE="$(cat "pipeline_runs/${RUN_ID}/gathered_band_file.txt")"
python3 plotting/plot_bands.py "$POLARITON_FILE" \
  --color-kind photonic \
  --dark-mode \
  --output plots/polariton_bands.pdf
```

Replace `<run-id>` with the directory name for the run being plotted.

## Dielectric function

The dielectric calculation writes text data files named `dielectric_*.dat`.
Each data row contains at least `omega`, `Re[eps]`, and `Im[eps]`; slab and
sheet2d runs also include `Re[alpha2D]` and `Im[alpha2D]`. The Python plotter
scans the existing files and can generate real, imaginary, combined, and
alpha2D plots without rerunning the Rust calculation.

First inspect the existing data directory:

```bash
DIELECTRIC_DIR="${DIELECTRIC_DIR:-$PWD/dielectric_scan/sheet_lwa}"
find "$DIELECTRIC_DIR" -maxdepth 1 -type f \
  -name 'dielectric_*.dat' -print | sort
```

Generate all available dielectric plots in dark mode:

```bash
python3 plotting/plot_dielectric.py \
  --input-dir "$DIELECTRIC_DIR" \
  --output-dir plots/dielectric \
  --plot all \
  --dark-mode \
  --format svg
```

This writes `dielectric_real.svg`, `dielectric_imag.svg`,
`dielectric_both.svg`, and `dielectric_alpha2d.svg` when the input files have
the five required columns. Dark mode uses a black background and a `plasma`
line colormap by default; override it with `--cmap NAME`.

To generate one plot in a different format or frequency window:

```bash
python3 plotting/plot_dielectric.py \
  --input-dir "$DIELECTRIC_DIR" \
  --output-dir plots/dielectric \
  --plot both \
  --omega-min 0.15 \
  --omega-max 0.21 \
  --format pdf
```

The script also accepts explicit files or a quoted glob. Use this when a
directory contains several unrelated scans:

```bash
python3 plotting/plot_dielectric.py \
  "$DIELECTRIC_DIR/dielectric_*.dat" \
  --output-dir plots/dielectric-selected \
  --plot all \
  --dark-mode \
  --format png \
  --dpi 300
```

The positional glob must describe one compatible set of files. If a plot is
empty, first check that the selected files belong to the same scan and that
their filename parameters match the intended dataset.
