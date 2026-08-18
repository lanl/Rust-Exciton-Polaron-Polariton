use ndarray::{Array1, Array2};
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct RelativeSolution {
    pub energies: Array1<f64>,
    pub states: Array2<f64>,
    pub parities: Array1<i32>,
}

impl RelativeSolution {
    pub fn into_tuple(self) -> (Array1<f64>, Array2<f64>, Array1<i32>) {
        (self.energies, self.states, self.parities)
    }
}

#[derive(Debug, Clone)]
pub struct ExcitonDiagonalization {
    pub energies: Array1<f64>,
    pub eigenvectors: Array2<Complex64>,
    pub relative_energies: Array1<f64>,
    pub relative_eigenvectors: Array2<f64>,
}

impl ExcitonDiagonalization {
    pub fn into_tuple(self) -> (Array1<f64>, Array2<Complex64>, Array1<f64>, Array2<f64>) {
        (
            self.energies,
            self.eigenvectors,
            self.relative_energies,
            self.relative_eigenvectors,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BandResult {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Array1<f64>>,
}

#[derive(Debug, Clone)]
pub struct ColoredBandResult {
    pub k_points: Vec<Vec<f64>>,
    pub eigenvalues: Vec<Array1<f64>>,
    pub colors: Vec<Vec<f64>>,
}
