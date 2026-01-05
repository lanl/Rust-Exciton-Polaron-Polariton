use log::warn;
use ndarray::{Array1, Array2, array, linalg::kron};
use serde::{Serialize, Deserialize};
use std::path::Path;
use bincode::{self};
use exciton_2d::{ExcitonConfig, FullHamiltonian, utils::fast_matrix_element_calc};
use ndarray_linalg::{Norm, c64};
use scilib::math::basic::factorial;
use std::{collections::HashMap, f64::consts::PI};
use rayon::prelude::*;

use crate::timed_println;

// Speed of light in atomic units
const C: f64 = 137.036;

// Define a serializable structure for band structure data storage
#[derive(Serialize, Deserialize)]
struct SerializableData {
    matrix: Vec<Vec<f64>>,  // Stores k-points as a 2D matrix
    arrays: Vec<Vec<f64>>,  // Stores eigenvalues as vectors
}

/// Saves band structure data to a binary file using bincode serialization
pub fn save_band_data<P: AsRef<Path>>(
    path: P,
    data: (Vec<Vec<f64>>, Vec<Array1<f64>>)
) -> Result<(), Box<dyn std::error::Error>> {
    let (matrix, arrays) = data;
    let converted_arrays: Vec<Vec<f64>> = arrays
        .into_iter()
        .map(|arr| arr.to_vec())
        .collect();

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
    path: P
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let serializable_data: SerializableData = bincode::deserialize(&bytes)?;

    let arrays: Vec<Array1<f64>> = serializable_data.arrays
        .into_iter()
        .map(|vec| Array1::from_vec(vec))
        .collect();

    Ok((serializable_data.matrix, arrays))
}

/// Represents a 3D vector with x, y, z components
#[derive(Debug, Clone, Copy)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3D { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        f64::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return *self;
        }
        Vector3D {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    pub fn cross(&self, other: &Vector3D) -> Self {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn _dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Represents a photon mode with polarization, wavevector, frequency, and Fock space dimension
#[derive(Debug, Clone)]
pub struct PhotonMode {
    pub eps: Array1<f64>, // Polarization vector (x,y components)
    pub q: Array1<f64>,   // Wavevector (x,y components)
    pub omega: f64,       // Frequency
    pub n_fock: usize,    // Dimension of the Fock space
}

impl PhotonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: usize) -> Self {
        let omega = q.magnitude() * C;
        let normal = Vector3D::new(0.0, 0.0, 1.0);
        let eps = pol.get_pol(q, normal);
        let q = array![q.x, q.y];
        let eps = array![eps.0, eps.1];

        Self {
            eps: eps,
            q: q,
            omega: omega,
            n_fock: n_fock,
        }
    }
}

/// Enum representing different polarization types for photons
pub enum Polarization {
    TE, // Transverse Electric
    TM, // Transverse Magnetic
}

impl Polarization {
    pub fn get_pol(&self, q: Vector3D, normal: Vector3D) -> (f64, f64) {
        let te: Vector3D;
        let tm: Vector3D;

        if q.cross(&normal).normalize().magnitude() < 0.5 {
            te = Vector3D::new(0.0, 1.0, 0.0);
            tm = Vector3D::new(1.0, 0.0, 0.0);
        } else {
            te = q.cross(&normal).normalize();
            tm = te.cross(&q).normalize();
        }

        match self {
            Polarization::TE => (te.x, te.y),
            Polarization::TM => (tm.x, tm.y),
        }
    }
}

/// Stores matrix elements for light-matter interaction calculations
pub struct LMMatrixElement {
    pub cos_mat: HashMap<(usize, usize, Vec<i32>), f64>, // Cosine matrix elements
    pub p_sin_mat: HashMap<(usize, usize, Vec<i32>), f64>, // Momentum-sine matrix elements
    pub all_q_ind: Option<Vec<Vec<i32>>>,                // All mode indices
}

impl LMMatrixElement {
    pub fn populate_mat_elements(
        &self,
        rel_vecs: &Array2<f64>,
        modes: &Vec<PhotonMode>,
        exciton: &FullHamiltonian,
    ) -> Self {
        let n_rel = rel_vecs.shape()[1];
        let n_modes = modes.len();

        // Use capacity hints for collections
        let mut sing_q_ind = Vec::with_capacity(n_modes);
        let mut doub_q_ind = Vec::with_capacity(n_modes * n_modes);

        // Generate single-mode indices
        for ind in 0..n_modes {
            sing_q_ind.push(vec![ind as i32]);
        }

        // Generate two-mode indices
        for ind1 in 0..n_modes {
            for ind2 in 0..n_modes {
                doub_q_ind.push(vec![ind1 as i32, ind2 as i32]);
            }
        }

        // Combine all indices
        let mut all_q_ind = Vec::with_capacity(sing_q_ind.len() + doub_q_ind.len());
        all_q_ind.extend(sing_q_ind);
        all_q_ind.extend(doub_q_ind);

        // Initialize matrix element storage with capacity hints
        let estimated_elements = n_rel * n_rel * all_q_ind.len();
        let mut cos_mat = HashMap::with_capacity(estimated_elements);
        let mut p_sin_mat = HashMap::with_capacity(n_rel * n_rel * n_modes);

        // Cache q_diff and q_sum calculations
        let mut q_diff_cache = HashMap::with_capacity(n_modes * n_modes);
        let mut q_sum_cache = HashMap::with_capacity(n_modes * n_modes);

        for ind1 in 0..n_modes {
            for ind2 in 0..n_modes {
                let q_1 = &modes[ind1].q;
                let q_2 = &modes[ind2].q;
                q_diff_cache.insert((ind1, ind2), q_1 - q_2);
                q_sum_cache.insert((ind1, ind2), q_1 + q_2);
            }
        }

        // Define helper functions to calculate matrix elements
        let get_cos = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 {
            self.get_cos_elements(exciton, rel_vecs, i1, i2, q)
        };

        let get_p_sin = |i1: usize, i2: usize, q: &Array1<f64>, eps: &Array1<f64>| -> f64 {
            self.get_psin_elements(exciton, rel_vecs, i1, i2, q, eps)
        };

        // Use iterators for the main computation
        for q_inds in &all_q_ind {
            for i1 in 0..n_rel {
                for i2 in 0..n_rel {
                    if q_inds.len() == 1 {
                        // Single-mode case
                        let mode_idx = q_inds[0] as usize;
                        let q = &modes[mode_idx].q;
                        let eps = &modes[mode_idx].eps;

                        // Calculate and store cosine matrix element
                        cos_mat
                            .entry((i1, i2, q_inds.clone()))
                            .or_insert_with(|| get_cos(i1, i2, q));

                        // Calculate and store momentum-sine matrix element
                        p_sin_mat
                            .entry((i1, i2, q_inds.clone()))
                            .or_insert_with(|| get_p_sin(i1, i2, q, eps));
                    } else if q_inds.len() == 2 {
                        // Two-mode case
                        let q_ind1 = q_inds[0] as usize;
                        let q_ind2 = q_inds[1] as usize;
                        let max_ind = q_ind1.max(q_ind2);
                        let min_ind = q_ind1.min(q_ind2);

                        // Use cached difference and sum
                        let q_diff = &q_diff_cache[&(max_ind, min_ind)];
                        let q_sum = &q_sum_cache[&(q_ind1, q_ind2)];

                        // Calculate and store difference matrix element
                        cos_mat
                            .entry((i1, i2, vec![max_ind as i32, -(min_ind as i32)]))
                            .or_insert_with(|| get_cos(i1, i2, q_diff));

                        // Calculate and store sum matrix element
                        cos_mat
                            .entry((i1, i2, vec![q_ind1 as i32, q_ind2 as i32]))
                            .or_insert_with(|| get_cos(i1, i2, q_sum));
                    }
                }
            }
        }

        Self {
            cos_mat,
            p_sin_mat,
            all_q_ind: Some(all_q_ind),
        }
    }

    pub fn new() -> Self {
        let cos_mat: HashMap<(usize, usize, Vec<i32>), f64> = HashMap::new();
        let p_sin_mat: HashMap<(usize, usize, Vec<i32>), f64> = HashMap::new();

        Self {
            cos_mat: cos_mat,
            p_sin_mat: p_sin_mat,
            all_q_ind: None,
        }
    }

    pub fn get_cos_elements(
        &self,
        exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>,
        i1: usize,
        i2: usize,
        q: &Array1<f64>,
    ) -> f64 {
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let grid_size = exciton.rel_ham.grid.grid_size;
        let dx = exciton.rel_ham.grid.dx;

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = rel_vecs
            .column(i1)
            .to_shape((grid_size, grid_size))
            .unwrap()
            .to_owned();
        let psi_j = rel_vecs
            .column(i2)
            .to_shape((grid_size, grid_size))
            .unwrap()
            .to_owned();

        // Precompute constants for efficiency
        let half_q0 = q[0] / 2.0;
        let half_q1 = q[1] / 2.0;

        // Calculate the cosine term more efficiently
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));

        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * half_q0;
            q_y[[0, i]] = y_grid[i] * half_q1;
        }

        let cos_term = (q_x + q_y).cos();

        // Calculate matrix element using fast implementation
        fast_matrix_element_calc(&psi_i, &psi_j, &cos_term, dx)
    }

    fn get_psin_elements(
        &self,
        exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>,
        i1: usize,
        i2: usize,
        q: &Array1<f64>,
        eps: &Array1<f64>,
    ) -> f64 {
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let eigvecs = rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;
        let dx = exciton.rel_ham.grid.dx;

        // Get the eigenvectors for states i and j
        let psi_i = eigvecs.column(i1);
        let psi_j = eigvecs.column(i2);

        // Calculate the sine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));

        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }

        let sin_term = (q_x + q_y).sin();
        let sin = sin_term.flatten();

        // Calculate momentum operator in sinc-DVR basis
        let p_1d = self.compute_sinc_dvr_rel(exciton);

        let p = kron(&(eps[0] * &p_1d), &Array2::eye(grid_size))
            + kron(&Array2::eye(grid_size), &(eps[1] * &p_1d));

        // Calculate matrix element
        (psi_i.dot(&p.dot(&(&sin * &psi_j))) + psi_i.dot(&(&sin * &p.dot(&psi_j)))) * dx * dx
    }

    fn compute_sinc_dvr_rel(&self, exciton: &FullHamiltonian) -> Array2<f64> {
        let dx = exciton.rel_ham.grid.dx;
        let nx = exciton.rel_ham.grid.grid_size;

        // Create momentum operator matrix using sinc-DVR formula
        Array2::from_shape_fn((nx, nx), |(i, j)| {
            if i == j {
                // Diagonal elements
                PI.powi(2) / 3.0 / dx * dx
            } else {
                // Off-diagonal elements
                let diff: f64 = i as f64 - j as f64;
                (-1.0 as f64).powf(diff) / (diff + f64::EPSILON) / dx
            }
        })
    }
}

#[derive(Clone)]
pub struct LMMatrixElementArray {
    pub p_cos_mat: HashMap<Vec<i32>, Array2<f64>>, // Momentum Cosine matrix elements
    pub cos_mat: HashMap<Vec<i32>, Array2<f64>>, // Cosine matrix elements
    pub sin_mat: HashMap<Vec<i32>, Array2<f64>>, // Sine matrix elements
}

impl LMMatrixElementArray {
    pub fn new() -> Self {
        let   cos_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();
        let p_cos_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();
        let   sin_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();

        Self {
            cos_mat: cos_mat,
            sin_mat: sin_mat,
            p_cos_mat: p_cos_mat,
        }
    }

    // pub fn populate_mat_elements(
    //     &self,
    //     mat_els: &LMMatrixElement,
    //     e_vecs: &Vec<Array1<f64>>,
    //     parities: &Array1<i32>,
    //     n_com: usize,
    // ) -> Self {
    //     // Find the number of even and odd parity states more efficiently
    //     let mut n_rel_even = 0;
    //     let mut n_rel_odd = 0;

    //     assert!(
    //         &mat_els.all_q_ind.is_some(),
    //         "all_q_ind mus be computed first!"
    //     );

    //     for (idx, &parity) in parities.iter().enumerate() {
    //         if parity == 1 && n_rel_even == 0 {
    //             n_rel_even = e_vecs[idx].len() / n_com;
    //         } else if parity == -1 && n_rel_odd == 0 {
    //             n_rel_odd = e_vecs[idx].len() / n_com;
    //         }

    //         // Break early if we've found both
    //         if n_rel_even > 0 && n_rel_odd > 0 {
    //             break;
    //         }
    //     }

    //     let n_rel_par = [n_rel_even, n_rel_odd];
    //     let n_eigs = e_vecs.len();

    //     // Use capacity hints for HashMaps
    //     let estimated_capacity = if let Some(all_q_ind) = &mat_els.all_q_ind {
    //         all_q_ind.len()
    //     } else {
    //         16 // Default capacity if all_q_ind is None
    //     };

    //     let mut cos_mat = HashMap::with_capacity(estimated_capacity);
    //     let mut p_sin_mat = HashMap::with_capacity(estimated_capacity);

    //     // Pre-calculate parity indices for all eigenvectors
    //     let parity_indices: Vec<usize> = parities.iter().map(|&p| (p == -1) as usize).collect();

    //     if let Some(all_q_ind) = &mat_els.all_q_ind {
    //         for q_inds in all_q_ind {
    //             let mut matrix_1 = Array2::zeros((n_eigs, n_eigs));
    //             let mut matrix_2 = Array2::zeros((n_eigs, n_eigs));

    //             // Keep using iterators for the main computation
    //             for idx1 in 0..n_eigs {
    //                 for idx2 in 0..n_eigs {
    //                     let vec1 = &e_vecs[idx1];
    //                     let vec2 = &e_vecs[idx2];

    //                     let idx1_par = parity_indices[idx1];
    //                     let idx2_par = parity_indices[idx2];

    //                     // Calculate the matrix elements using iterators
    //                     let (cos_sum, sin_sum) = (0..n_rel_par[idx1_par])
    //                         .flat_map(|rel_ind1| {
    //                             (0..n_rel_par[idx2_par]).flat_map(move |rel_ind2| {
    //                                 let r_ind1 = rel_ind1 + idx1_par * n_rel_even;
    //                                 let r_ind2 = rel_ind2 + idx2_par * n_rel_even;

    //                                 // Get matrix elements
    //                                 let (mat_element1, mat_element2) =
    //                                     self.get_matrix_elements(mat_els, r_ind1, r_ind2, q_inds);

    //                                 (0..n_com).map(move |com_ind| {
    //                                     let i1 = n_com * rel_ind1 + com_ind;
    //                                     let i2 = n_com * rel_ind2 + com_ind;
    //                                     let a1 = vec1[i1];
    //                                     let a2 = vec2[i2];

    //                                     (a1 * a2 * mat_element1, a1 * a2 * mat_element2)
    //                                 })
    //                             })
    //                         })
    //                         .fold((0.0, 0.0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

    //                     matrix_1[[idx1, idx2]] = cos_sum;
    //                     matrix_2[[idx1, idx2]] = sin_sum;
    //                 }
    //             }

    //             // Insert into the appropriate HashMaps
    //             if q_inds.len() == 1 {
    //                 cos_mat.insert(q_inds.clone(), matrix_1);
    //                 p_sin_mat.insert(q_inds.clone(), matrix_2);
    //             } else if q_inds.len() == 2 {
    //                 let q_ind1 = q_inds[0].max(q_inds[1]);
    //                 let q_ind2 = q_inds[0].min(q_inds[1]);

    //                 cos_mat.insert(vec![q_ind1, -q_ind2], matrix_1);
    //                 cos_mat.insert(vec![q_ind1, q_ind2], matrix_2);
    //             }
    //         }
    //     }

    //     // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
    //     // timed_println!("cos_mat in populate_mat_elements {:?}",test);

    //     Self { cos_mat, p_sin_mat }
    // }

    // // Helper function to get matrix elements
    // fn get_matrix_elements(
    //     &self,
    //     mat_els: &LMMatrixElement,
    //     r_ind1: usize,
    //     r_ind2: usize,
    //     q_inds: &Vec<i32>,
    // ) -> (f64, f64) {
    //     if q_inds.len() == 1 {
    //         // Use pattern matching to avoid redundant lookups
    //         let key = (r_ind1, r_ind2, q_inds.clone());

    //         match (mat_els.cos_mat.get(&key), mat_els.p_sin_mat.get(&key)) {
    //             (Some(&mat_element1), Some(&mat_element2)) => (mat_element1, mat_element2),
    //             _ => panic!(
    //                 "Matrix element missing for i1 = {}, i2 = {}, q_ind = {:?}",
    //                 r_ind1, r_ind2, q_inds
    //             ),
    //         }
    //     } else {
    //         let q_ind1 = q_inds[0].max(q_inds[1]);
    //         let q_ind2 = q_inds[0].min(q_inds[1]);

    //         let key1 = (r_ind1, r_ind2, vec![q_ind1, -q_ind2]);
    //         let key2 = (r_ind1, r_ind2, vec![q_ind1, q_ind2]);

    //         match (mat_els.cos_mat.get(&key1), mat_els.cos_mat.get(&key2)) {
    //             (Some(&mat_element1), Some(&mat_element2)) => (mat_element1, mat_element2),
    //             _ => panic!(
    //                 "Matrix element missing for i1 = {}, i2 = {}, q_inds = {:?}/{:?}",
    //                 r_ind1,
    //                 r_ind2,
    //                 vec![q_ind1, -q_ind2],
    //                 vec![q_ind1, q_ind2]
    //             ),
    //         }
    //     }
    // }

    pub fn populate_tay_mat_elements(
        &self,
        all_q_ind: &Vec<Vec<i32>>,
        e_vecs: &Vec<Array1<f64>>,
        parities: &Array1<i32>,
        modes: &Vec<PhotonMode>,
        n_com: usize,
        mu: f64,
    ) -> Self {
        // Find the number of even and odd parity states more efficiently
        let mut n_rel_even = 0;
        let mut n_rel_odd = 0;

        // timed_println!("All q_ind = {:?}", all_q_ind);

        for (idx, &parity) in parities.iter().enumerate() {
            if parity == 1 && n_rel_even == 0 {
                n_rel_even = e_vecs[idx].len() / n_com;
            } else if parity == -1 && n_rel_odd == 0 {
                n_rel_odd = e_vecs[idx].len() / n_com;
            }

            // Break early if we've found both
            if n_rel_even > 0 && n_rel_odd > 0 {
                break;
            }
        }

        let n_rel_par = [n_rel_even, n_rel_odd];
        let n_eigs = e_vecs.len();

        // Use capacity hints for HashMaps
        let mut cos_mat = HashMap::with_capacity(all_q_ind.len() - modes.len());
        let mut sin_mat = HashMap::with_capacity(modes.len());
        let mut p_cos_mat = HashMap::with_capacity(modes.len());

        // Pre-calculate parity indices for all eigenvectors
        let parity_indices: Vec<usize> = parities.iter().map(|&p| (p == -1) as usize).collect();

        // Get Laguerre coefficients for radial integrals
        let (lag_coeffs, states) = &self.get_laguerre_coeffs(n_rel_even + n_rel_odd);
        // let dr_coeffs = &self.get_dr_radial_poly(states);

        // Use parallel map to compute matrices for each q_inds value
        let results: Vec<(Vec<i32>, Array2<f64>, Array2<f64>)> = all_q_ind.par_iter().enumerate().map(|(q_idx, q_inds)| {
            let mut matrix_1 = Array2::zeros((n_eigs, n_eigs));
            let mut matrix_2 = Array2::zeros((n_eigs, n_eigs));

            // Keep using iterators for the main computation
            for idx1 in 0..n_eigs {
                for idx2 in 0..n_eigs {
                    let vec1 = &e_vecs[idx1];
                    let vec2 = &e_vecs[idx2];

                    let idx1_par = parity_indices[idx1];
                    let idx2_par = parity_indices[idx2];

                    // Calculate the matrix elements using iterators
                    let (element_1, element_2) = (0..n_rel_par[idx1_par])
                        .flat_map(|rel_ind1| {
                            let r_ind1 = rel_ind1 + idx1_par * n_rel_even;
                            
                            (0..n_rel_par[idx2_par]).flat_map(move |rel_ind2| {
                                let r_ind2 = rel_ind2 + idx2_par * n_rel_even;

                                let (mat_element1, mat_element2): (f64,f64);

                                // Get matrix elements
                                if q_inds.len() == 1 {
                                    let mode_idx = q_inds[0] as usize;
                                    let q = &modes[mode_idx].q;
                                    let eps = &modes[mode_idx].eps;
                                    if q_idx < modes.len(){  
                                        mat_element1 = self.get_p_cos_tayl(&states[r_ind1], &states[r_ind2], q, eps,  mu, &lag_coeffs);
                                    }
                                    else {
                                        mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], q,  mu, &lag_coeffs);
                                    }
                                    mat_element2 = self.get_sin_tayl(&states[r_ind1], &states[r_ind2], q,  mu, &lag_coeffs);
                                }
                                else {
                                    let q_diff = &modes[q_inds[0] as usize].q - &modes[q_inds[1] as usize].q;
                                    let q_sum = &modes[q_inds[0] as usize].q + &modes[q_inds[1] as usize].q;
                                    mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_diff, mu, &lag_coeffs);
                                    mat_element2 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_sum, mu, &lag_coeffs);
                                }

                                (0..n_com).map(move |com_ind| {
                                    let i1 = n_com * rel_ind1 + com_ind;
                                    let i2 = n_com * rel_ind2 + com_ind;
                                    let a1 = vec1[i1];
                                    let a2 = vec2[i2];

                                    (a1 * a2 * mat_element1, a1 * a2 * mat_element2)
                                })
                            })
                        })
                        .fold((0.0, 0.0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

                    matrix_1[[idx1, idx2]] = element_1;
                    matrix_2[[idx1, idx2]] = element_2;
                }
            }
            
            // Return tuple of q_inds and computed matrices
            (q_inds.clone(), matrix_1, matrix_2)
        }).collect();

        // Process results and populate HashMaps
        results.iter().enumerate().for_each(|(idx, (q_inds, matrix_1, matrix_2))|{
            if q_inds.len() == 1 {
                if idx < modes.len(){  
                    p_cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                else {
                    cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                sin_mat.insert(q_inds.clone(), matrix_2.clone());
            } else if q_inds.len() == 2 {
                let q_ind1 = q_inds[0].max(q_inds[1]);
                let q_ind2 = q_inds[0].min(q_inds[1]);

                cos_mat.insert(vec![q_ind1, -q_ind2], matrix_1.clone());
                cos_mat.insert(vec![q_ind1, q_ind2], matrix_2.clone());
            }
        });

        Self { cos_mat, sin_mat, p_cos_mat }
    }

    /// Returns the polynomial coefficients for ∂/∂r R_{n,m}(rho)
    /// where R_{n,m}(rho) is the radial part of the 2D hydrogenic wavefunction
    /// expressed in terms of associated Laguerre polynomials for rho = beta_n r.
    /// The coefficients are returned in ascending order of powers of r.
    /// e^{-rho/2} is factored out.
    fn _get_dr_radial_poly(&self, states: &Vec<(i32, i32)>) -> HashMap<(i32, i32), Vec<(i32, f64)>>{

        let mut dr_coeffs: HashMap<(i32, i32), Vec<(i32, f64)>> = HashMap::with_capacity(states.len());

        states.iter().for_each(|&(n,m)| {
            let n = n as usize;
            let m = m.abs() as usize;

            match (n,m) {
                (0,0) => {
                    // ∂/∂r R_{0,0}(rho) = -1/2 e^{-rho/2}
                    dr_coeffs.insert((0,0), vec![(0,-0.5)]);
                }
                (1,0) => {
                    // ∂/∂r R_{1,0}(rho) = (-3/2 + r/2) e^{-rho/2}
                    dr_coeffs.insert((1,0), vec![(0,-1.5), (1,0.5)]);
                }
                (_,0) => {
                    // ∂/∂r R_{n,0}(rho) = -0.5 (L_n(rho) + 2 L^(1)_(n-1) (rho) ) e^{-rho/2}
                    let l_n = self.laguerre(n, 0.0);
                    let l_n_minus_1 = self.laguerre(n-1, 1.0);

                    let mut coeff: Vec<(i32,f64)> = Vec::with_capacity(l_n.len());
                    for (i, c) in l_n.iter() {
                        let c1 = -0.5 * c;
                        let i1 = *i as usize;
                        if i1 < l_n_minus_1.len() {
                            let c2 = -l_n_minus_1[i1].1;
                            coeff.push((*i, c1 + c2));
                        } else {
                            coeff.push((*i, c1));
                        }
                    }

                    dr_coeffs.insert((n as i32, 0), coeff);
                }
                (n,m) if n == m as usize => {
                    // ∂/∂r R_{n,n}(rho) = 0.5 rho^(n-1) (2n - rho) e^{-rho/2}
                    dr_coeffs.insert((n as i32, m as i32), 
                    vec![
                        (n as i32 - 1, n as f64),
                        (n as i32, -1.0)
                    ]);
                }
                (_,_) => {
                    // ∂/∂r R_{n,m}(rho) = 0.5 rho^(|m|-1) ((2|m| - rho) L^(2|m|)_(n-|m|)(rho) 
                    //                                      - 2 rho L^(2|m|)_(n-|m|-1)(rho) ) e^{-rho/2}
                    let l_n_m = self.laguerre(n - m, 2.0 * m as f64);
                    let l_n_m_minus_1 = self.laguerre(n - m - 1, 2.0 * m as f64 + 1.0);

                    // (n-m) + 1 + (m - 1) = n
                    let max_order = n;
                    // From rho^(|m|-1) prefactor
                    let min_order = m - 1;

                    let mut coeff: Vec<(i32,f64)> = Vec::with_capacity(max_order - min_order + 1);
                    for (ind, order) in (min_order..=max_order).enumerate() {
                        if ind == 0 {
                            coeff.push((order as i32, m as f64 * l_n_m[0].1));
                        }
                        else if ind < l_n_m.len() {
                            let c1 = m as f64 * l_n_m[ind].1 - 0.5 * l_n_m[ind - 1].1;
                            let c2 = -l_n_m_minus_1[ind - 1].1;
                            coeff.push((order as i32, c1 + c2));
                        }
                        else if ind == l_n_m.len() {
                            coeff.push((order as i32, - 0.5 * l_n_m[ind - 1].1));

                        }
                    }

                    dr_coeffs.insert((n as i32, m as i32), coeff);
                }
            }
        });

        dr_coeffs
    }


    /// Matrix element: < n1,m1 | (epsilon·∇) sin(q·r/2) | n2,m2 >
    /// ≈ < n1,m1 | (epsilon·∇) (q·r/2) | n2,m2 > (First order Taylor expansion)
    /// = < n1,m1 | (|q|r) cos(θ - θ_q) (1/r) sin(θ-θ_e) ∂/∂θ | n2,m2 > <- ∂θ term
    /// + < n1,m1 | (|q|r) cos(θ - θ_q) cos(θ-θ_e) ∂/∂r | n2,m2 > <- ∂r term
    pub fn _get_p_sin_tayl(
        &self,
        states: &Vec<(i32, i32)>,
        i1: usize,
        i2: usize,
        q: &Array1<f64>,
        epsilon: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
        dr_lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        let (n1, m1) = states[i1];
        let (n2, m2) = states[i2];

        // Test for q == 0
        let q_mag = q.norm();
        if q_mag < 1e-10 {
            return 0.0;
        }
        
        // Validate quantum numbers
        debug_assert!(n1 >= m1.abs(), "Invalid state: n1 must be >= |m1|");
        debug_assert!(n2 >= m2.abs(), "Invalid state: n2 must be >= |m2|");

        // Common constants
        const SQRT_2: f64 = 1.4142135623730951;
        
        // Calculate angular component
        let m_sign = (m1 * m2).signum();
        let theta_q = f64::atan2(q[1], q[0]);
        let theta_eps = f64::atan2(epsilon[1], epsilon[0]);
        let cos_theta = (theta_q + theta_eps).cos();
        let sin_theta = (theta_q + theta_eps).sin();

        let (ang_dtheta, ang_dr) = match m_sign {
            // Case: At least one of m1 or m2 is zero
            0 => {
                // Both m1 and m2 are zero: matrix element is zero since q.eps = 0
                if m1 == 0 && m2 == 0 {
                    (0.0, 0.0)
                }
                // One of m1 or m2 is zero, the other is +/-2
                else if m1 + m2 == 2 {
                    // m2 = 2, m1 = 0
                    let m2_term_dtheta = if m2 != 0 {
                        -PI * SQRT_2 * sin_theta * m2.signum() as f64
                    } else {
                        0.0
                    };
                    let m2_term_dr = if m2 != 0 {
                        PI / SQRT_2 * cos_theta * m2.signum() as f64
                    } else {
                        0.0
                    };
                    // m1 = 2, m2 = 0 -> ∂/∂θ |n,0> = 0= 0, only ∂/∂r term contributes
                    let m1_term_dr = if m1 != 0 {
                        PI * SQRT_2 * cos_theta * m1.signum() as f64
                    } else {
                        0.0
                    };

                    (m2_term_dtheta, m2_term_dr + m1_term_dr)
                } else if m1 + m2 == -2 {
                    // m2 = -2, m1 = 0
                    let m2_term_dtheta = if m2 != 0 {
                        -PI * SQRT_2 * sin_theta * m2.signum() as f64
                    } else {
                        0.0
                    };
                    let m2_term_dr = if m2 != 0 {
                        PI / SQRT_2 * sin_theta * m2.signum() as f64
                    } else {
                        0.0
                    };
                    // m1 = 2, m2 = 0 -> ∂/∂θ |n,0> = 0, only ∂/∂r term contributes
                    let m1_term_dr = if m1 != 0 {
                        PI * SQRT_2 * cos_theta * m1.signum() as f64
                    } else {
                        0.0
                    };

                    (m2_term_dtheta, m2_term_dr + m1_term_dr)
                } else {
                    (0.0, 0.0)
                }
            }

            // Case: m1 and m2 have the same sign
            1 => {
                let mut result = (0.0, 0.0);

                // Check difference of m values
                if (m2 - m1).abs() == 2 {
                    result.1 += PI / 2.0 * cos_theta;
                    result.0 += m2 as f64 * -PI / 2.0 * sin_theta * (m2 - m1).signum() as f64;
                }

                // Check sum of m values
                if (m1 + m2).abs() == 2 {
                    result.1 += PI / 2.0 * cos_theta * m2.signum() as f64;
                    result.0 += m2.abs() as f64 * -PI / 2.0 * sin_theta * (m2 - m1).signum() as f64;
                }

                result
            }

            // Case: m1 and m2 have opposite signs
            -1 => {
                let mut result = (0.0, 0.0);

                // Check difference of m values
                if (m2 + m1).abs() == 2 {
                    result.1 += -PI / 2.0 * sin_theta * (m2 - m1).signum() as f64;
                    result.0 += m2 as f64 * PI / 2.0 * sin_theta;
                }

                // Check sum of m values
                if (m1 - m2).abs() == 2 {
                    result.1 += PI / 2.0 * sin_theta * (m2.signum() * (m2 - m1).signum()) as f64;
                    result.0 += -m2.abs() as f64 * PI / 2.0 * sin_theta;
                }

                result
            }

            // This should never happen
            _ => panic!("Unexpected sign value"),
        };

        // Calculate radial component
        let beta_1 = 2.0 * mu / (n1 as f64 + 0.5);
        let beta_2 = 2.0 * mu / (n2 as f64 + 0.5);
        
        // Precompute normalization factors
        let norm_1 = beta_1
            * (factorial(n1 as usize - m1.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n1 as usize + m1.abs() as usize) as f64
                    * (2. * n1 as f64 + 1.)))
                .sqrt();
        let norm_2 = beta_2
            * (factorial(n2 as usize - m2.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n2 as usize + m2.abs() as usize) as f64
                    * (2. * n2 as f64 + 1.)))
                .sqrt();

        let prefactor = norm_1 * norm_2 * q_mag / 2.0;
        
        // Helper function for radial integrals
        let calculate_radial_integral = |i: i32, j: i32, add_power: i32| -> f64 {
            let a = (beta_1 + beta_2) / 2.0;
            let b = m1.abs() + m2.abs() + i + j + add_power;
            
            // Use logarithmic factorials for large values to prevent overflow
            let factorial_term = if b > 170 {
                (0..=b).map(|i| (i as f64).ln()).sum::<f64>().exp()
            } else {
                factorial(b as usize) as f64
            };
            
            let denom_term = a.powi(b + 1);
            
            factorial_term / denom_term * beta_1.powi(m1.abs() + i) * beta_2.powi(m2.abs() + j)
        };

        // Calculate radial components for dtheta and dr terms
        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];
        let dr_lag_coeff = &dr_lag_coeffs[&(n2, m2.abs())];

        // < n1,m1 | (|q|r) (1/r) | n2,m2 > <- ∂θ term
        let rad_dtheta = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                acc2 + c1 * c2 * calculate_radial_integral(*i, *j, 1)
            })
        }) * prefactor;

        // < n1,m1 | (|q|r) ∂/∂r | n2,m2 > <- ∂r term
        let rad_dr = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + dr_lag_coeff.iter().fold(0.0, |acc2, (j, c2)| {
                acc2 + c1 * c2 * calculate_radial_integral(*i, *j, 2)
            })
        }) * prefactor;

        // if n1 == 0 {
        //     timed_println!("States (n1={},m1={}), (n2={},m2={}): Radial dtheta = {}, Radial dr = {}", n1, m1, n2, m2, rad_dtheta, rad_dr);
        // }

        ang_dtheta * rad_dtheta + ang_dr * rad_dr
    }


    /// Evaluates overlap integral of radial wavefunctions for two states (n1,m1) and (n2,m2)
    /// multiplied by r^l 
    /// O(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) R_{n2,m2}(r)
    pub fn get_overlap(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];

        let radial = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                // \int e^{-a r} r^b dr = b! / a^(b+1)

                // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                let a = (beta_1 + beta_2) / 2.0;

                // From r^|m1| * r^i * r^|m2| * r^j * r^l dr
                let b = m1.abs() + m2.abs() + i + j + l;

                let factorial_term = factorial(b as usize) as f64;
                let denom_term = (a).powi(b + 1);

                acc2 + c1 * c2 * factorial_term / denom_term
                    * beta_1.powi(m1.abs() + i)
                    * beta_2.powi(m2.abs() + j)
            })
        });

        radial
    }

    /// Evaluates overlap integral of radial wavefunctions for two states |n1,m1> and ∂/∂r |n2,m2>
    /// multiplied by r^l 
    /// O'(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) ∂/∂r R_{n2,m2}(r)
    pub fn get_dr_overlap(
        &self,
        i1: &(i32,i32),
        i2: &(i32,i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n2 = i2.0;   
        let m2 = i2.1;
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        let mut radial = (m2.abs() as f64)/beta_2 * self.get_overlap(i1, i2, l - 1, mu, lag_coeffs);
        
        radial += self.get_overlap(i1, i2, l, mu, lag_coeffs);

        if n2 - m2.abs() - 1 >= 0 {
            let n1 = i1.0;
            let m1 = i1.1;
            let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);

            let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
            let lag_coeff2 = self.laguerre((n2 - m2.abs() - 1) as usize, (2 * m2.abs() + 1) as f64);

            radial += lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
                acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                    // \int e^{-a r} r^b dr = b! / a^(b+1)

                    // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                    let a = (beta_1 + beta_2) / 2.0;

                    // From r^|m1| * r^i * r^|m2| * r^j * r^l dr
                    let b = m1.abs() + m2.abs() + i + j + l;

                    let factorial_term = factorial(b as usize) as f64;
                    let denom_term = (a).powi(b + 1);

                    acc2 + c1 * c2 * factorial_term / denom_term
                        * beta_1.powi(m1.abs() + i)
                        * beta_2.powi(m2.abs() + j)
                })
            });
        }

        radial
    }

    /// Calculate the sine term using analytic integrals of leading Taylor series terms
    /// < n1,m1 | sin(q.r/2) | n2,m2 > ~  < n1,m1 | (q.r/2) | n2,m2 >
    pub fn get_sin_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    1  => c64::from_polar(1.0,  theta_q) * weight1 * weight2,
                    -1 => c64::from_polar(1.0, -theta_q) * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.im > 1e-10 {
            warn!("Imaginary part of angular integral is non-zero in get_sin_tayl(): {}", angular.im);
        }

        let angular = angular.re;

        if angular.abs() < 1e-10 {
            return 0.0;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        let rad_integral = self.get_overlap(i1, i2, 2, mu, lag_coeffs);

        norms * rad_integral * PI * q.norm() / 2.0 * angular
    }

    /// Calculate the cosine term using analytic integrals of leading Taylor series terms
    /// < n1,m1 | cos(q.r/2) | n2,m2 > ~  < n1,m1 | 1 - (q.r/2)^2/2 | n2,m2 >
    pub fn get_cos_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let overlap = if i1.0 == i2.0 && i1.1 == i2.1 {
            1.0
        } else {
            0.0
        };

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    0  => c64::new(1.0, 0.0) * weight1 * weight2,
                    2  => c64::from_polar(1.0, -2.0 * theta_q) * weight1 * weight2 / 2.0,
                    -2 => c64::from_polar(1.0,  2.0 * theta_q) * weight1 * weight2 / 2.0,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.im > 1e-10 {
            warn!("Imaginary part of angular integral is non-zero in get_cos_tayl(): {}", angular.im);
        }

        let angular = angular.re;

        if angular.abs() < 1e-10 {
            return overlap;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        let rad_integral = self.get_overlap(i1, i2, 3, mu, lag_coeffs);

        overlap - angular * norms * rad_integral
    }

    /// Matrix element: i < n1,m1 | (epsilon·∇) cos(q·r/2) | n2,m2 >
    /// ≈ < n1,m1 | (epsilon·∇) (1 - (q·r/2)^2/2) | n2,m2 > (Second order Taylor expansion)
    pub fn get_p_cos_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        eps: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let q2 = q.map(|val| val*val).sum();
        let theta_q = f64::atan2(q[1], q[0]);
        let theta_eps = f64::atan2(eps[1], eps[0]);

        let integral = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;

                let j1 = & (i1.0, *m1);
                let j2 = & (i2.0, *m2);
                
                match diff {
                    3  => c64::from_polar(1.0,  theta_eps + 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * ( (*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    -3 => c64::from_polar(1.0, -theta_eps - 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * (-(*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    1  => c64::from_polar(1.0,  theta_eps)
                            * (
                                -(*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                - q2 / 32.0 * (
                                    -(*m2 as f64) * (2.0 - c64::from_polar(1.0,  2.0 * (theta_q - theta_eps))) 
                                    * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    + (2.0 + c64::from_polar(1.0,  2.0 * (theta_q - theta_eps)))
                                    * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                                )
                            )
                            * weight1 * weight2,
                    -1 => c64::from_polar(1.0,  theta_eps)
                            * (
                                 (*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                - q2 / 32.0 * (
                                     (*m2 as f64) * (2.0 - c64::from_polar(1.0, -2.0 * (theta_q - theta_eps))) 
                                    * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    + (2.0 + c64::from_polar(1.0, -2.0 * (theta_q - theta_eps)))
                                    * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                                )
                            )
                            * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if integral.im > 1e-10 {
            warn!("Imaginary part of integral is non-zero in get_p_cos_tayl(): {}", integral.im);
        }

        let integral = integral.re;

        if integral.abs() < 1e-10 {
            return 0.0;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        norms * integral
    }

    fn real_l_to_complex_m(&self, l: i32) -> Vec<(i32, c64)>{

        match l {
            0 => vec![(0, c64::new(1.0, 0.0))],
            l if l > 0 => vec![
                ( l, c64::new(1.0 / 2_f64.sqrt(), 0.0)), 
                (-l, c64::new(1.0 / 2_f64.sqrt(), 0.0))],
            l if l < 0 => vec![
                ( l, c64::new(0.0,  1.0 / 2_f64.sqrt())), 
                (-l, c64::new(0.0, -1.0 / 2_f64.sqrt()))],
            _ => panic!("Invalid l value in real_l_to_complex_m")
        }
    }

    fn norm_constant(&self, (n,m): &(i32,i32), mu: f64) -> f64 {

        let beta = mu * 2.0 / (*n as f64 + 0.5);

        beta
        * (factorial(*n as usize - m.abs() as usize) as f64
            / (2. * PI
                * factorial(*n as usize + m.abs() as usize) as f64
                * (2. * *n as f64 + 1.)))
            .sqrt()
    }

    /// Calculate the cosine term using analytic integrals of leading Taylor series terms
    /// < n1,m1 | cos(q.r/2) | n2,m2 > ~  < n1,m1 | 1 - (q.r/2)^2/2 | n2,m2 >
    pub fn _get_cos_tayl(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let overlap = if (n1,m1) == (n2,m2) { 1.0 } else { 0.0 };

        // Test for q == 0
        let q_2 = q[0].powi(2) + q[1].powi(2);
        if q_2 < 1e-10 {
            return overlap;
        }

        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        // Calculate angular component
        let m_sign = (m1 * m2).signum();
        let theta_q = f64::atan2(q[1], q[0]);
        let cos_2theta = (theta_q * 2.0).cos();
        let sin_2theta = (theta_q * 2.0).sin();

        let angular = match m_sign {
            // Case: At least one of m1 or m2 is zero
            0 => {
                if m1 == m2 {
                    PI
                } else if m1 + m2 == 2 {
                    PI / 2.0_f64.sqrt() * cos_2theta
                } else if m1 + m2 == -2 {
                    PI / 2.0_f64.sqrt() * sin_2theta
                } else {
                    0.0
                }
            }

            // Case: m1 and m2 have the same sign
            1 => {
                let mut result = 0.0;

                // Check difference of m values
                if m1 - m2 == 0 {
                    result = PI;
                } else if (m1 - m2).abs() == 2 {
                    result = PI / 2.0 * cos_2theta;
                }

                // Check sum of m values
                if (m1 + m2).abs() == 2 {
                    result += PI / 2.0 * cos_2theta * m1.signum() as f64;
                }

                result
            }

            // Case: m1 and m2 have opposite signs
            -1 => {
                let mut result = 0.0;

                // Check sum of m values
                if m1 + m2 == 2 {
                    result = -PI / 2.0 * sin_2theta;
                } else if m1 + m2 == -2 {
                    result = PI / 2.0 * sin_2theta;
                }

                // Check sum of absolute values
                if m1.abs() + m2.abs() == 2 {
                    result += PI / 2.0 * sin_2theta;
                } else if m1.abs() + m2.abs() == -2 {
                    result += -PI / 2.0 * sin_2theta;
                }

                result
            }

            // This should never happen
            _ => panic!("Unexpected sign value"),
        };

        // If angular part is negligible, return overlap only
        if angular.abs() < 1e-10 || q.norm() < 1e-10 {
            return overlap;
        };

        // Calculate radial component
        let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);
        let norm_1 = beta_1
            * (factorial(n1 as usize - m1.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n1 as usize + m1.abs() as usize) as f64
                    * (2. * n1 as f64 + 1.)))
                .sqrt();
        let norm_2 = beta_2
            * (factorial(n2 as usize - m2.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n2 as usize + m2.abs() as usize) as f64
                    * (2. * n2 as f64 + 1.)))
                .sqrt();

        let prefactor = norm_1 * norm_2 * q_2 / 8.0;

        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];

        let radial = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                // \int e^{-a r} r^b dr = b! / a^(b+1)

                // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                let a = (beta_1 + beta_2) / 2.0;

                // From r^|m1| * r^i * r^|m2| * r^j * (|q| r)^2 * r dr
                let b = m1.abs() + m2.abs() + i + j + 3;

                let factorial_term = factorial(b as usize) as f64;
                let denom_term = (a).powi(b + 1);

                acc2 + c1 * c2 * factorial_term / denom_term
                    * beta_1.powi(m1.abs() + i)
                    * beta_2.powi(m2.abs() + j)
            })
        }) * prefactor;

        // if n1 == 0 {
            // timed_println!("States (n1={},m1={}), (n2={},m2={}): Angular = {}, Radial = {}, Overlap = {}", n1, m1, n2, m2, angular, radial, overlap);
        // }

        overlap - angular * radial
    }

    fn _test_psi_norm(&self, states: &Vec<(i32, i32)>, mu: f64) {
        states.iter().for_each(|&(n,m)| {
            let n = n;
            let m = m.abs();

            let beta = mu / 8. / PI / PI / (n as f64 + 0.5);
            let norm = beta
                * (factorial(n as usize - m as usize) as f64
                    / (2.
                        * PI
                        * factorial((n + m) as usize) as f64
                        * (2. * n as f64 + 1.)))
                    .sqrt();

            let lag_coeffs = self.laguerre((n - m) as usize, 2. * m as f64);

            // Calculate integral of |R_{n,m}(rho)|^2 rho d rho
            let integral = lag_coeffs.iter().fold(0.0, |acc1, (i, c1)| {
                acc1 + lag_coeffs.iter().fold(0.0, |acc2, (j, c2)| {
                    // \int e^{-a r} r^b dr = b! / a^(b+1)

                    // From e^{-beta x/2} * e^{-beta x/2} = e^{-beta x}
                    let a = beta;

                    // From r^|m| * r^i * r^|m| * r^j * r dr
                    let b = 2 * m + i + j + 1;

                    let factorial_term = factorial(b as usize) as f64;
                    let denom_term = (a).powi(b + 1);

                    acc2 + c1 * c2 * factorial_term / denom_term
                        * beta.powi(2 * m + i + j)
                })
            });

            let psi_norm = 2. * PI * norm.powi(2) * integral;

            timed_println!("State (n={},m={}): Norm = {}", n, m, psi_norm);
        });
    }

    pub fn get_x(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        // Calculate angular component
        let diff = (m1 -m2).abs();
        let sum = (m1 + m2).abs();

        // Only jumps in m by ±1 allowed
        if diff & sum != 1 {
            return 0.0
        }
        
        let m_sign = (m1 * m2).signum();

        if m_sign == 0 && (m1 + m2) < 0 {
            return 0.0;
        }

        let angular = match m_sign {
            0  => PI * 2.0_f64.sqrt(),
            1  => {
                let mut result = 0.0;
                if diff == 1 { result += PI};
                if sum  == 1 { result += PI * m1.signum() as f64};
                result
            },
            _ => 0.0
        };

        // Calculate radial component
        let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);
        let norm_1 = beta_1
            * (factorial(n1 as usize - m1.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n1 as usize + m1.abs() as usize) as f64
                    * (2. * n1 as f64 + 1.)))
                .sqrt();
        let norm_2 = beta_2
            * (factorial(n2 as usize - m2.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n2 as usize + m2.abs() as usize) as f64
                    * (2. * n2 as f64 + 1.)))
                .sqrt();

        let prefactor = norm_1 * norm_2;

        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];

        // timed_println!("n1 = {}, m1 = {}, n2 = {}, m2 = {}", n1,m1,n2,m2);
        // timed_println!("Lag1 = {:?}", lag_coeff1);
        // timed_println!("Lag2 = {:?}", lag_coeff2);

        let radial = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                // \int e^{-a r} r^b dr = b! / a^(b+1)
                // timed_println!("j = {}", j);

                // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                let a = (beta_1 + beta_2) / 2.0;

                // From r^|m1| * r^i * r^|m2| * r^j * (x.r) * r dr
                let b = m1.abs() + m2.abs() + i + j + 2;

                let factorial_term = factorial(b as usize) as f64;
                let denom_term = (a).powi(b + 1);

                acc2 + c1 * c2 * factorial_term / denom_term
                    * beta_1.powi(m1.abs() + i)
                    * beta_2.powi(m2.abs() + j)
            })
        }) * prefactor;

        angular * radial
    }

    pub fn get_laguerre_coeffs(&self, n_rel: usize) -> (HashMap<(i32, i32), Vec<(i32, f64)>>, Vec<(i32, i32)>) {
        // Generate all states (n,m) and sort by energy
        let mut states = Vec::new();
        let mut n: i32 = 0;
        while states.len() < n_rel {
            for m in -n..=n {
                if n >= m.abs() {
                    // Valid state condition - also check Laguerre order
                    if (n - m.abs()) as usize <= 7 {
                        states.push((n, m));
                    } else {
                        warn!("Skipping state (n={},m={}) because Laguerre polynomials of order >7 are not implemented", n, m);
                    }
                }
            }
            n += 1;
        }

        // Keep only the first n_rel states
        states.truncate(n_rel);

        let mut lag_coeffs: HashMap<(i32, i32), Vec<(i32, f64)>> = HashMap::new();

        for &(n, m) in &states {
            lag_coeffs.insert((n, m), self.laguerre((n - m.abs()) as usize, 2. * m.abs() as f64));
        }

            // Ensure all necessary keys are present
        for &(n1, m1) in &states {
            if !lag_coeffs.contains_key(&(n1, m1.abs())) {
                panic!("Missing key ({},{})", n1, m1.abs());
            }
        }

        (lag_coeffs, states)
    }

    fn laguerre(&self, order: usize, alpha: f64) -> Vec<(i32, f64)> {
        let coeffs: Vec<(i32, f64)>;

        match order {
            0 => {
                coeffs = vec![(0, 1.0)];
            }
            1 => {
                coeffs = vec![(0, alpha + 1.0), (1, -1.0)];
            }
            2 => {
                coeffs = vec![
                    (0, 0.5 * (alpha + 2.0) * (alpha + 1.0)),
                    (1,      -(alpha + 2.0)),
                    (2, 0.5),
                ];
            }
            3 => {
                coeffs = vec![
                    (0,(1.0 / 6.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -0.5       * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  0.5       * (alpha + 3.0)),
                    (3,-(1.0 / 6.0)),
                ];
            }
            4 => {
                coeffs = vec![
                    (0, (1.0 / 24.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 6.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,         0.25 * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 6.0) * (alpha + 4.0)),
                    (4, 1.0 / 24.0),
                ];
            }
            5 => {
                coeffs = vec![
                    (0, (1.0 / 120.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 24.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 24.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0)),
                ];
            }
            6 => {
                coeffs = vec![
                    (0,  (1.0 / 720.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 120.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3,  -(1.0 / 36.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0) * (alpha + 6.0)),
                    (6, 1.0 / 720.0),
                ];
            }
            7 => {
                coeffs = vec![
                    (0, (1.0 / 5040.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 720.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0)),
                    (6,  (1.0 / 720.0) * (alpha + 7.0)),
                    (7, -(1.0 / 5040.0)),
                ];
            }
            _ => {
                panic!("Laguerre polynomials of order >7 not implemented yet!")
            }
        }

        coeffs
    }

}

#[derive(Clone)]
pub struct PPConfig {
    // Exciton Parameters
    pub e_config: ExcitonConfig,
    pub n_rel: usize,
    pub n_ex_eig: usize,

    // Polariton Parameters
    pub lm_coupling: f64,
    pub omega_0: f64,
    pub q_values: Vec<Vector3D>,

    // Polaron Parameters
    pub pn_coupling: f64,
    pub pn_omega: f64,
    pub pn_k: Array1<f64>,
    pub pn_n_fock: usize,

    // Band Parameters
    pub n_k_points: usize,
    pub n_bands: usize,
}

impl PPConfig {
    pub fn new(
        // Exciton Parameters
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,

        // Polariton Parameters
        lm_coupling: f64,
        omega_0: f64,
        q_values: Vec<Vector3D>,

        // Polaron Parameters
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,

        // Band Parameters
        n_k_points: usize,
        n_bands: usize,
    ) -> Self {
        Self {
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
        }
    }
}
