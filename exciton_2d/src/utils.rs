// src/utils.rs
use log::info;
use ndarray::Array2;
use num_complex::Complex64;
use std::process;

pub fn print_memory_usage(message: &str) {
    // This is platform-specific, so we'll use a simpler approach
    // For a more accurate implementation, consider using the 'psutil' crate
    let usage = process::id(); // Just a placeholder
    info!("Memory usage {}: {} bytes (estimated)", message, usage);
}

pub fn apply_symmetry(k_point: &[f64], symmetry_op: usize) -> Vec<f64> {
    let kx = k_point[0];
    let ky = k_point[1];
    
    match symmetry_op {
        0 => vec![kx, ky],                // Identity
        1 => vec![-ky, kx],               // 90° rotation
        2 => vec![-kx, -ky],              // 180° rotation
        3 => vec![ky, -kx],               // 270° rotation
        4 => vec![kx, -ky],               // Reflection across x-axis
        5 => vec![-kx, ky],               // Reflection across y-axis
        6 => vec![ky, kx],                // Reflection across y=x
        7 => vec![-ky, -kx],              // Reflection across y=-x
        _ => panic!("Invalid symmetry operation: {}", symmetry_op),
    }
}

pub fn fast_matrix_element_calc(
    psi_i_real: &Array2<f64>,
    psi_i_imag: &Array2<f64>,
    psi_j_real: &Array2<f64>,
    psi_j_imag: &Array2<f64>,
    cos_term: &Array2<f64>,
    dx: f64,
) -> Complex64 {
    // Calculate the conjugate of psi_i
    let psi_i_conj_real = psi_i_real;
    let psi_i_conj_imag = &(-psi_i_imag);
    
    // Calculate the product (psi_i_conj * cos_term * psi_j)
    let real_part = (psi_i_conj_real * cos_term) * psi_j_real - 
                   (psi_i_conj_imag * cos_term) * psi_j_imag;
    let imag_part = (psi_i_conj_real * cos_term) * psi_j_imag + 
                   (psi_i_conj_imag * cos_term) * psi_j_real;
    
    // Sum and scale by dx^2
    let real_sum = real_part.sum() * dx * dx;
    let imag_sum = imag_part.sum() * dx * dx;
    
    Complex64::new(real_sum, imag_sum)
}

// Add this to utils.rs:

pub struct ThreadLimiter {
    previous_value: Option<String>,
    env_var: String,
}

impl ThreadLimiter {
    pub fn new(env_var: &str, value: &str) -> Self {
        let previous_value = std::env::var(env_var).ok();
        std::env::set_var(env_var, value);
        
        Self {
            previous_value,
            env_var: env_var.to_string(),
        }
    }
}

impl Drop for ThreadLimiter {
    fn drop(&mut self) {
        // Restore the previous value when this object is dropped
        if let Some(ref value) = self.previous_value {
            std::env::set_var(&self.env_var, value);
        } else {
            std::env::remove_var(&self.env_var);
        }
    }
}

// Usage example:
// {
//     let _limiter = ThreadLimiter::new("OPENBLAS_NUM_THREADS", "1");
//     // Code in this block will run with OPENBLAS_NUM_THREADS=1
// } // Original value is restored when _limiter goes out of scope

