/// Quantization scale for K-point cache hashes.
///
/// This should be much smaller than the spacing between physically distinct
/// K points in your scans. It prevents tiny floating-point differences from
/// changing cache filenames.
pub const K_HASH_QUANTUM: f64 = 1e-10;

pub fn quantize_f64_for_hash(x: f64) -> i64 {
    if !x.is_finite() {
        panic!("Cannot hash non-finite floating-point value: {}", x);
    }

    let q = (x / K_HASH_QUANTUM).round();

    if q.abs() > i64::MAX as f64 {
        panic!(
            "Floating-point value {} is too large to quantize with quantum {}",
            x, K_HASH_QUANTUM
        );
    }

    if q == 0.0 { 0 } else { q as i64 }
}

pub fn hash_quantized_f64<H: std::hash::Hasher>(x: f64, hasher: &mut H) {
    use std::hash::Hash;

    quantize_f64_for_hash(x).hash(hasher);
}

/// Filename tag describing how the eigenstates were selected.
/// `None`  -> "neig{n_eig}"   (unchanged from before, old caches stay valid)
/// `Some`  -> "ecut{E:.6e}"   (n_eig is ignored when a cutoff is given)
pub fn eig_selection_tag(n_eig: usize, energy_cutoff: Option<f64>) -> String {
    match energy_cutoff {
        Some(cutoff) => format!("ecut{:.6e}", cutoff),
        None => format!("neig{}", n_eig),
    }
}
