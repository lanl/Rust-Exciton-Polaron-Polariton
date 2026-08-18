use crate::cache;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Generic colored band data.
///
/// `color_values` can represent:
/// - n expectation
/// - photonic character
/// - phonon number
/// - any scalar band color observable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColoredBandData {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Vec<f64>>,
    pub color_values: Vec<Vec<f64>>,
}

impl ColoredBandData {
    pub fn from_results(
        k_points: Vec<Vec<f64>>,
        eigenvalues: Vec<Array1<f64>>,
        color_values: Vec<Vec<f64>>,
    ) -> Self {
        let eigenvalues = eigenvalues.iter().map(|arr| arr.to_vec()).collect();

        Self {
            k_points,
            eigenvalues,
            color_values,
        }
    }

    pub fn to_results(&self) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>) {
        let eigenvalues = self
            .eigenvalues
            .iter()
            .map(|vec| Array1::from_vec(vec.clone()))
            .collect();

        (
            self.k_points.clone(),
            eigenvalues,
            self.color_values.clone(),
        )
    }

    pub fn save_to_file(&self, filename: impl AsRef<Path>) -> cache::CacheResult<()> {
        cache::save_bincode_atomic(filename, self)
    }

    pub fn load_from_file(filename: impl AsRef<Path>) -> cache::CacheResult<Self> {
        cache::load_bincode(filename)
    }
}
