// src/config.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcitonConfig {
    pub n_kappa: usize,
    pub a0: f64,
    pub v0: f64,
    pub k: Vec<f64>,
    pub m_e: f64,
    pub m_h: f64,
    pub verbose: bool,
    pub grid_size: usize,
    pub box_size: f64,
    pub save_dir: PathBuf,
    pub recalculate: bool,
    pub n_jobs: i32,
    pub use_gpu: bool,
    pub batch_size: usize,
}

impl Default for ExcitonConfig {
    fn default() -> Self {
        Self {
            n_kappa: 31,
            a0: 1.0,
            v0: 1.0,
            k: vec![0.0, 0.0],
            m_e: 1.0,
            m_h: 1.0,
            verbose: true,
            grid_size: 64,
            box_size: 20.0,
            save_dir: PathBuf::from("saved_data"),
            recalculate: false,
            n_jobs: -1,
            use_gpu: false,
            batch_size: 32,
        }
    }
}

impl ExcitonConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_kappa < 1 {
            return Err("N_kappa must be a positive integer".to_string());
        }
        if self.a0 <= 0.0 {
            return Err("Lattice constant a0 must be positive".to_string());
        }
        if self.v0 < 0.0 {
            return Err("Potential strength V0 must be non-negative".to_string());
        }
        if self.m_e <= 0.0 || self.m_h <= 0.0 {
            return Err("Effective masses must be positive".to_string());
        }
        if self.grid_size < 4 {
            return Err("grid_size must be a positive integer >= 4".to_string());
        }
        if self.box_size <= 0.0 {
            return Err("box_size must be positive".to_string());
        }
        Ok(())
    }
}
