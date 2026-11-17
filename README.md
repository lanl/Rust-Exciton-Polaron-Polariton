# rust_exciton_pp

Rust code for simulating excitons in 2D materials with a periodic potential, and
for coupling those excitons to photon modes (light–matter) and phonon/Fock modes
(polaron-polariton). It computes exciton and polariton band structures and
the dielectric response of the resulting system.

## Workspace layout

| Crate / directory | Purpose |
|---|---|
| `exciton_2d/` | Library and binary for 2D excitons: relative-coordinate Hamiltonian (Laguerre basis), center-of-mass band structure, plotting. See [`exciton_2d/README.md`](exciton_2d/README.md). |
| `light-matter/` | Main application (`light-matter` binary). Builds the light–matter and polaron-polariton Hamiltonians, solves for band structures, and computes dielectric functions. |
| `sim-common/` | Shared utilities: k-paths, square-lattice symmetry, atomic file writes, MPI environment helpers, vector types. |
| `plotting/` | Python (NumPy/Matplotlib) scripts for plotting saved band and dielectric data. |
| `scripts/slurm/` | Slurm pipelines for cluster runs (`pipeline/` is current, `legacy/` is old). |
| `scripts/dev/` | Helper scripts (build environment, line counts, source collection). |
| `docs/` | Documentation, currently [`plotting_instructions.md`](docs/plotting_instructions.md). |

## Requirements

- Rust toolchain with edition 2024 support (workspace uses resolver 3).
- PETSc/SLEPc (with complex scalars) and MPI for the sparse eigensolvers.
- Python 3 with NumPy and Matplotlib for the plotting scripts.

## Building

```bash
cargo build --release
```

`scripts/dev/light_matter_build.sh` wraps `cargo` with the ARPACK library paths
used on the original cluster; edit the paths for your machine before using it.
The `profiling` profile (release plus debug symbols) is also available:
`cargo build --profile profiling`.

## Usage

The `light-matter` binary has three subcommands:

```bash
cargo run --release -p light-matter -- <subcommand> [options]
```

| Subcommand | What it does |
|---|---|
| `bandstructure` | Exciton–photon (and optional polaron) band structure along a k-path. |
| `dielectric` | Dielectric function of the coupled system (`legacy`, `slab`, or `sheet2d` methods). |
| `matrix-elem-calc` | Precompute and cache the light–matter matrix elements. |

Run `light-matter <subcommand> --help` for the full option list. Commonly used
options:

- Exciton model: `--n-kappa`, `--a0`, `--v0`, `--m-e`, `--m-h`, `--grid-size`, `--box-size`
- Solver: `--method sparse|dense|linop`, `--n-rel`, `--n-ex-eig`, `--n-eig`, `--energy-cutoff`
- Photons: `--photon-modes cartesian|effective`, `--n-modes`, `--q-max`, `--total-coupling`, `--long-wavelength`
- Caching: `--save-dir`, `--use-cached`, `--recalculate`
- Threads: `--rayon-threads` (defaults to `OMP_NUM_THREADS`, else 1)

Note that `--save-dir` defaults to a cluster scratch path, so set it explicitly
when running elsewhere. Diagonalization and dielectric calculations write
checkpoints and can be relaunched to resume.

The standalone exciton binary and library API are described in
[`exciton_2d/README.md`](exciton_2d/README.md).

## Running on a cluster

Band structures can be split across a Slurm array: each array task solves a
chunk of k-points (`--parallel distributed-chunks`, using `K_CHUNK_INDEX` /
`K_N_CHUNKS` or the Slurm array variables), and a gather step merges the chunk
files (`--parallel gather-chunks`). The scripts in `scripts/slurm/pipeline/`
(`band_chunks`, `band_gather`, `band_matcalc`, `dielectric_calc`,
`dielectric_gather`, `submit_pipelines.sh`) implement this. They contain
site-specific paths and resource requests, so edit `pipeline_params.sh` and the
`#SBATCH` headers first.

## Plotting

Saved band `.bin` files and dielectric `.dat` files can be plotted without
recomputation:

```bash
python3 plotting/inspect_band_file.py <file-or-dir-or-glob>
python3 plotting/plot_bands.py <file-or-dir-or-glob> --output bands.pdf
python3 plotting/plot_dielectric.py <file>.dat
```

See [`docs/plotting_instructions.md`](docs/plotting_instructions.md) for details.

##

LANL's Richard P. Feynman Center for Innovation assigned O number: O5171.
