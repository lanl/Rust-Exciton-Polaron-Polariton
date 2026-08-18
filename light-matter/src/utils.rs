use bincode::{self};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::Path;

// Define a serializable structure for band structure data storage
#[derive(Serialize, Deserialize)]
struct SerializableData {
    matrix: Vec<Vec<f64>>, // Stores k-points as a 2D matrix
    arrays: Vec<Vec<f64>>, // Stores eigenvalues as vectors
}

/// Saves band structure data to a binary file using bincode serialization
pub fn save_band_data<P: AsRef<Path>>(
    path: P,
    data: (Vec<Vec<f64>>, Vec<Array1<f64>>),
) -> Result<(), Box<dyn std::error::Error>> {
    let (matrix, arrays) = data;
    let converted_arrays: Vec<Vec<f64>> = arrays.into_iter().map(|arr| arr.to_vec()).collect();

    let serializable_data = SerializableData {
        matrix,
        arrays: converted_arrays,
    };

    let encoded: Vec<u8> = bincode::serialize(&serializable_data)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

/// Loads band structure data from a binary file
pub fn load_band_data<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let serializable_data: SerializableData = bincode::deserialize(&bytes)?;

    let arrays: Vec<Array1<f64>> = serializable_data
        .arrays
        .into_iter()
        .map(|vec| Array1::from_vec(vec))
        .collect();

    Ok((serializable_data.matrix, arrays))
}

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
