# Exciton 2D

A Rust library for simulating excitons in 2D materials with periodic potentials.

## Features

- Calculation of exciton states in 2D materials
- Band structure calculations along high-symmetry paths
- Visualization of eigenstates and band structures
- Support for different effective masses and potential strengths
- Memory-efficient implementation for large grids

## Usage

```rust
use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};

fn main() {
    // Create configuration
    let config = ExcitonConfig {
        n_kappa: 11,
        a0: 3.0,
        v0: 0.05,
        k: vec![0.0, 0.0],
        m_e: 1.0,
        m_h: 1.0,
        verbose: true,
        grid_size: 64,
        box_size: 3.0,
        save_dir: "saved_data".into(),
        recalculate: false,
        n_jobs: -1,
        use_gpu: false,
        batch_size: 32,
    };
    
    // Create the 2D exciton system
    let system = Exciton2DGridSystem::from_config(config);
    
    // Solve for the relative coordinate eigenstates
    let (rel_eigvals, rel_eigvecs, rel_pars) = system.solve_rel_hamiltonian(16);
    
    // Calculate band structure
    let (k_points, eigvals_list, parities_list) = system.calculate_band_structure(
        60,  // n_points
        9,   // n_rel_states
        16,  // n_bands
        true, // recalculate_bands
        0,    // select_parity (0: both, 1: even, -1: odd)
    );
    
    // Plot band structure
    system.plot_band_structure(&k_points, &eigvals_list, &parities_list, Some("band_structure.png"));
}
```

## Command-line Interface

The crate also provides a command-line interface:

```bash
exciton_2d --a0 3.0 --v0 0.05 --grid_size 128 --box_size 3.0 --n_rel_states 9 --n_bands 16
```

Run with `--help` to see all available options.

## License

This project is licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.