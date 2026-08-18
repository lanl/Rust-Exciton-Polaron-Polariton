# Refactor Status

## Completed

- Added `sim-common` crate.
- Moved generic cache helpers to `sim-common::cache`.
- Moved MPI/chunking helpers to `sim-common::mpi`.
- Moved generic colored band serialization to `sim-common::data`.
- Moved k-path and square-lattice symmetry helpers to `sim-common::kpath`.
- Replaced duplicate `ColorBandData` definitions with shared `ColoredBandData`.
- Renamed `light-matter/src/hamiltonians` to `light-matter/src/hamiltonian`.
- Moved light-matter band structure code into `light-matter/src/bands`.
- Moved dielectric computation code into `light-matter/src/dielectric`.
- Moved `PPConfig` out of `utils.rs`.
- Moved `Vector3D`, `BosonMode`, and `Polarization` out of `utils.rs`.
- Fixed `timed_println` macros to use shared MPI helpers.
- Preserved absolute paths for `matrix-ops` and `slepc-rs`.

## Still To Do

- Fully split `exciton_2d/src/hamiltonian/mod.rs` into relative/full/prepared modules.
- Fully split `light-matter/src/app/mod.rs`.
- Split `light-matter/src/hamiltonian/polaron_polariton.rs`.
- Replace tuple-heavy APIs with named result structs throughout.
- Replace remaining `Vec<Vec<f64>>` K-point usage with typed aliases or structs.
- Remove temporary compatibility aliases:
  - `band_structure`
  - `dielectric_function`
  - `hamiltonians`
- Reduce `unwrap`, `expect`, and `panic!` in library code.
- Move plotting code out of physics modules.
- Add smoke tests for:
  - k-path generation
  - chunk bounds
  - Fock index conversion
  - cache round-trips
  - exciton flat/unflat index
