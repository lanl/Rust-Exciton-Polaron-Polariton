# Matrix Element Caching Implementation

## Overview

Implemented a disk-based caching system for expensive photonic matrix element calculations in `LMMatrixElementArray::populate_tay_mat_elements`. The cache stores computed matrices keyed by unique parameter combinations (boson modes, mu, n_com, n_rel, exciton eigenvectors).

## Implementation Details

### Core Features

1. **Two-Level Caching Strategy**
   - **Relative-basis cache**: Stores matrices computed in relative coordinates (independent of exciton eigenvectors)
   - **Exciton-basis cache**: Stores final transformed matrices (depends on eigenvectors, but keyed by config parameters)
   - Relative-basis cache can be reused across different eigenvector sets

2. **Cache File Naming**
   - **Exciton-basis**: `matrix_elements_mu{:.6}_ncom{}_nrel{}_confighash{:016x}.bin`
   - **Relative-basis**: `matrix_elements_rel_mu{:.6}_nrel{}_confighash{:016x}.bin`
   - Default location: `saved_data_lm/matrix_element_cache/`

3. **Deterministic Hashing**
   - `get_config_hash()` (in PPConfig): Hashes all exciton and interaction parameters (a0, v0, mu, n_rel, n_ex_eig, lm_coupling, pn_coupling, q_values, pn_k, etc.)
   - **Important**: Uses config parameter hash instead of eigenvector hash to avoid issues with gauge freedom (global phases in eigenvectors)
   - **Simplified**: No separate mode hash needed since PPConfig already includes all mode q-values
   - Uses `DefaultHasher` with `f64::to_bits()` for reproducible hashes

4. **Serialization Format**
   - Uses `bincode` crate for efficient binary serialization
   - Custom `SerializableMatrixElements` struct converts HashMap<Vec<i32>, Array2<c64>> to serializable format
   - Stores matrix shape separately for reconstruction

## API Changes

### New Methods in `LMMatrixElementArray`

```rust
// Cache filename generators
pub fn get_cache_filename(cache_dir: &str, mu: f64, n_com: usize, 
                          n_rel: usize, config_hash: u64) -> String

pub fn get_cache_filename_rel(cache_dir: &str, mu: f64, 
                               n_rel: usize, config_hash: u64) -> String

// Hashing functions (in PPConfig)
pub fn hash_exciton_config(a0: f64, v0: f64, mu: f64, n_rel: usize, 
                           n_ex_eig: usize, n_com: usize) -> u64

// Serialization methods
pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>>
pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>>
pub fn save_rel_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>>
pub fn load_rel_from_file(filename: &str) -> Result<(HashMap, HashMap, HashMap), Box<dyn std::error::Error>>

// Main computation method with caching
pub fn populate_tay_mat_elements_with_cache(
    &mut self,
    all_q_ind: &Vec<Vec<i32>>,
    e_vecs: &Vec<Array1<c64>>,
    modes: &Vec<BosonMode>,
    n_com: usize,
    mu: f64,
    cache_dir: Option<&str>,
    force_recompute: bool,
    config_hash: u64,
)

// Backward-compatible wrapper (no caching)
pub fn populate_tay_mat_elements(...) // Calls populate_tay_mat_elements_with_cache with cache_dir=None, config_hash=0
```

### PPConfig Changes

**PPConfig** (`light-matter/src/utils.rs`):
```rust
pub struct PPConfig {
    // ... existing fields ...
    pub cache_dir: Option<String>,  // Defaults to Some("saved_data_lm/matrix_element_cache")
}

impl PPConfig {
    // Generate hash from all configuration parameters
    pub fn get_config_hash(&self) -> u64
}
```

**LMHamiltonian** (`light-matter/src/hamiltonians.rs`):
```rust
pub struct LMHamiltonian<M, B> {
    // ... existing fields ...
    pub cache_dir: Option<String>,
    pub config_hash: u64,  // Hash from PPConfig
}

// New constructor
pub fn new_with_cache(..., cache_dir: Option<String>, config_hash: u64) -> Self
```

**PPOffHamiltonian** (`light-matter/src/sing_ex.rs`):
```rust
pub struct PPOffHamiltonian {
    // ... existing fields ...
    pub cache_dir: Option<String>,
    pub config_hash: u64,  // Hash from configuration
}

// New constructor
pub fn new_with_cache(..., cache_dir: Option<String>, config_hash: u64) -> Self
```

## Usage

### Automatic Caching (via PPConfig)

```rust
let pp_config = PPConfig::new(
    e_config,
    n_rel,
    n_ex_eig,
    lm_coupling,
    omega_0,
    q_values,
    pn_coupling,
    pn_omega,
    pn_k,
    pn_n_fock,
    n_k_points,
    n_bands,
);
// cache_dir automatically set to "saved_data_lm/matrix_element_cache"

let pp_ham = PPHamiltonian::from_config(&pp_config);
pp_ham.initialize(Some(k_vec));  // Will use cache automatically
```

### Manual Caching Control

```rust
// Disable caching
let mut pp_config = PPConfig::new(...);
pp_config.cache_dir = None;

// Custom cache directory
pp_config.cache_dir = Some("/path/to/custom/cache".to_string());

// Get config hash for manual use
let config_hash = pp_config.get_config_hash();

// Force recomputation (bypass cache)
matrix_elements.populate_tay_mat_elements_with_cache(
    &all_q_ind, 
    e_vecs, 
    modes, 
    n_com, 
    mu,
    Some("saved_data_lm/matrix_element_cache"),
    true,  // force_recompute
    config_hash
);
```

## Design Rationale

### Why Hash Config Parameters Instead of Eigenvectors?

The implementation hashes `PPConfig` parameters rather than eigenvector components because:

1. **Gauge Freedom**: Eigenvectors have arbitrary global phases (e^(iφ)) that don't affect physics but change hash values
2. **Numerical Stability**: Small numerical differences in diagonalization can produce different eigenvectors for the same Hamiltonian
3. **Physical Equivalence**: Same physical parameters → same eigenvectors (up to gauge), so config hash is sufficient
4. **Determinism**: Config parameters are exactly specified, while eigenvectors are computed

The config hash includes all parameters that determine the Hamiltonian:
- Exciton parameters: a0, v0, mu, n_rel, n_ex_eig
- Light-matter coupling: lm_coupling, omega_0, q_values (all x,y,z components)
- Polaron coupling: pn_coupling, pn_omega, pn_k (all components), pn_n_fock

This single hash captures all the physics, including mode information, so no separate mode hash is needed. This ensures cache hits only occur when the physics is truly identical.

## Cache Behavior

### Cache Hit
- Logs: `"Successfully loaded matrix elements from cache"`
- Skips expensive computation
- Loads both relative-basis and exciton-basis caches if available

### Cache Miss
- Logs: Cache file doesn't exist (no message, proceeds to computation)
- Computes matrix elements
- Saves results to cache for future runs
- Logs: `"Saved matrix elements to cache: {filename}"`

### Cache Corruption
- Logs: `"Failed to load cache file {}: {}. Recomputing..."`
- Gracefully falls back to computation
- Overwrites corrupted cache with fresh computation

## Performance Benefits

- **First run**: Computes and saves to cache (~same time as before)
- **Subsequent runs**: Loads from cache (expected speedup: 10-100x depending on problem size)
- **Relative-basis reuse**: If only eigenvectors change but modes/mu/n_rel stay same, relative-basis cache is reused

## Files Modified

1. **light-matter/src/matrix_elements.rs** (~250 lines added)
   - Core caching implementation
   - Serialization/deserialization
   - Hash functions

2. **light-matter/src/utils.rs**
   - Added `cache_dir` field to `PPConfig`
   - Updated constructor to set default cache directory

3. **light-matter/src/hamiltonians.rs**
   - Added `cache_dir` field to `LMHamiltonian`
   - Added `new_with_cache()` constructor
   - Updated `initialize()` to use caching
   - Updated `PPHamiltonian::from_config()` to pass cache_dir

4. **light-matter/src/sing_ex.rs**
   - Added `cache_dir` field to `PPOffHamiltonian`
   - Added `new_with_cache()` constructor
   - Updated `calc_matrix_element_arrays()` to use caching

## Future Enhancements

### High Priority
- [ ] Add cache file version marker to detect incompatible formats
- [ ] Add dimension validation on load (verify matrix shapes match expected)
- [ ] Add checksum validation for data integrity

### Medium Priority
- [ ] Cache statistics (track hit/miss rates, cache size)
- [ ] Cache cleanup utilities (remove old/unused caches)
- [ ] Partial cache invalidation (separate photon/phonon caches)

### Low Priority
- [ ] Compression for large cache files
- [ ] Multi-threaded cache access (currently single-threaded)
- [ ] Cache prewarming for batch computations

## Testing Checklist

- [ ] Verify compilation succeeds
- [ ] Test first run creates cache files in correct location
- [ ] Test second run loads from cache (verify timing improvement)
- [ ] Test results match between cached and non-cached runs (numerical accuracy)
- [ ] Test cache_dir=None disables caching
- [ ] Test custom cache directory works
- [ ] Test force_recompute bypasses cache
- [ ] Test graceful handling of corrupted cache files
- [ ] Test parameter changes create new cache files (not reusing wrong cache)
- [ ] Profile performance improvement on realistic workloads

## Migration Guide

### For Existing Code

**No changes required** - the implementation maintains backward compatibility. Existing code using `populate_tay_mat_elements()` will continue to work without caching.

**To enable caching**, either:
1. Use `PPConfig` (automatic caching enabled by default)
2. Use `new_with_cache()` constructors and pass a cache directory
3. Call `populate_tay_mat_elements_with_cache()` directly with `cache_dir` parameter

### Cache Management

```bash
# View cache files
ls -lh saved_data_lm/matrix_element_cache/

# Clear all caches
rm -rf saved_data_lm/matrix_element_cache/*

# Clear caches for specific parameters (use glob patterns)
rm saved_data_lm/matrix_element_cache/matrix_elements_mu0.500000_*
```

## Logging

The implementation uses Rust's `log` crate with `info!()` and `warn!()` macros. To see cache activity:

```rust
env_logger::init();  // Enable logging in your application
```

Log messages:
- `info!`: Cache hits and successful saves
- `warn!`: Cache misses, corrupted files, save failures
