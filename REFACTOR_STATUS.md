# Refactor Status

## Completed

### Workspace / shared infrastructure

- Added `sim-common` crate.
- Moved generic cache helpers to `sim-common::cache`.
- Moved MPI/chunking helpers to `sim-common::mpi`.
- Moved generic colored band serialization to `sim-common::data`.
- Moved k-path and square-lattice symmetry helpers to `sim-common::kpath`.
- Preserved absolute paths for:
  - `/vast/home/mike_tay/matrix_ops`
  - `/vast/home/mike_tay/slepc_rust`

### `exciton_2d`

- Replaced duplicate `ColorBandData` with shared `sim_common::data::ColoredBandData`.
- Reused shared k-path generation for square-lattice band calculations.
- Reused shared path-distance helper in plotting.
- Split Hamiltonian code into:
  - `hamiltonian/mod.rs`
  - `hamiltonian/full.rs`
  - `hamiltonian/prepared.rs`
  - `relative/hamiltonian.rs`
  - `relative/laguerre.rs`
- Added named result structs in `results.rs`.
- Refactored visualization into:
  - `visualization/mod.rs`
  - `visualization/eigenstate.rs`
  - `visualization/spectrum.rs`
  - `visualization/bands.rs`
  - `visualization/colored_bands.rs`
- Fixed square-lattice path generation to return the requested number of points.
- Fixed basic band-structure cache loading/saving for plain exciton bands.

### `light-matter`

- Renamed `hamiltonians` module to `hamiltonian`.
- Renamed cryptic Hamiltonian files:
  - `boson.rs` → `boson_basis.rs`
  - `exciton.rs` → `exciton_projected.rs`
  - `lm.rs` → `light_matter.rs`
  - `pp.rs` → `polaron_polariton/mod.rs`
- Moved light-matter band-structure code into `bands`.
- Split band plotting into `bands/plot.rs`.
- Moved dielectric computation code into `dielectric`.
- Split dielectric cache/filename helpers into `dielectric/cache.rs`.
- Split CLI/app orchestration into:
  - `app/mod.rs`
  - `app/config.rs`
  - `app/bandstructure.rs`
  - `app/dielectric.rs`
  - `app/matrix_elem.rs`
- Moved `PPConfig` out of `utils.rs`.
- Moved `Vector3D`, `BosonMode`, and `Polarization` out of `utils.rs`.
- Reduced `utils.rs` to K-point hash helpers only.
- Fixed `timed_println` macros to use shared MPI helpers.
- Removed duplicate matrix-element serialization definition from builder code.
- Renamed `populate_tay_mat_elements_with_cache` to `populate_taylor_matrix_elements_with_cache`.
- Split product-Fock indexing helpers into `hamiltonian/polaron_polariton/fock.rs`.
- Refactored visualization into:
  - `visualization/mod.rs`
  - `visualization/data.rs`
  - `visualization/dielectric.rs`
  - `visualization/bands.rs`
  - `visualization/stacked.rs`
- Removed old compatibility module names from main crate exports.
- Fixed basic visualization indexing issues found during review.

### Tests

- Added smoke tests for:
  - k-path generation,
  - path-distance monotonicity,
  - MPI chunk bounds,
  - colored-band data conversion,
  - exciton flat/unflat indexing,
  - `KappaKey::add`,
  - product-Fock index/occupation round trips.

## Remaining To Do

### API cleanup

- Replace tuple-heavy public APIs with named result structs throughout.
- Convert more library-level `unwrap`, `expect`, and `panic!` paths into `Result`.
- Replace remaining `Vec<Vec<f64>>` K-point APIs with typed `Vec2`/`KPoint2` aliases or structs.

### Further decomposition

- Further split `light-matter/src/hamiltonian/polaron_polariton/mod.rs` into:
  - `builder.rs`
  - `interactions.rs`
  - `sparse_kronsum.rs`
- Consider moving generic gnuplot generation into `sim-common::plot`.

### Plot quality

- Implement a real heatmap in `exciton_2d::visualization::eigenstate`.
- Remove hard-coded plot energy windows where possible.
- Make plotting options configurable instead of embedded constants.
