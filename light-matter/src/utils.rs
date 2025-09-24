use std::{usize};
use ndarray::{array, Array1, Array2};
use rayon::vec;
use serde::{Serialize, Deserialize};
use std::path::Path;
use bincode::{self};
use exciton_2d::{utils::fast_matrix_element_calc, FullHamiltonian};
use std::{collections::HashMap, f64::consts::PI};

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
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3D { x, y, z }
    }
    
    fn magnitude(&self) -> f64 {
        f64::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }
    
    fn normalize(&self) -> Self {
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
    
    fn cross(&self, other: &Vector3D) -> Self {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    
    fn _dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Represents a photon mode with polarization, wavevector, frequency, and Fock space dimension
#[derive(Debug, Clone)]
pub struct PhotonMode {
    pub eps: Array1<f64>, // Polarization vector (x,y components)
    pub q: Array1<f64>,   // Wavevector (x,y components)
    pub omega: f64,       // Frequency
    pub n_fock: usize     // Dimension of the Fock space
}

impl PhotonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: usize) -> Self{
        let omega = q.magnitude() * C;
        let normal = Vector3D::new(0.0, 0.0, 1.0);
        let eps = pol.get_pol(q, normal);
        let q = array![q.x, q.y];
        let eps = array![eps.0, eps.1];

        Self { eps: eps, q: q, omega: omega, n_fock: n_fock }
    }
}

/// Enum representing different polarization types for photons
pub enum Polarization {
    TE, // Transverse Electric
    TM  // Transverse Magnetic
}

impl Polarization {
    pub fn get_pol(&self, q: Vector3D, normal: Vector3D) -> (f64,f64) {
        let te:Vector3D;
        let tm:Vector3D;

        if q.cross(&normal).normalize().magnitude() < 0.5 {
            te = Vector3D::new(0.0, 1.0, 0.0);
            tm = Vector3D::new(1.0, 0.0, 0.0);
        }
        else {
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
    pub   cos_mat: HashMap<(usize,usize, Vec<i32>), f64>, // Cosine matrix elements
    pub p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64>, // Momentum-sine matrix elements
    pub all_q_ind: Option<Vec<Vec<i32>>>, // All mode indices
}

impl LMMatrixElement {
    pub fn populate_mat_elements(&self, rel_vecs: &Array2<f64>, modes: &Vec<PhotonMode>, exciton: &FullHamiltonian) -> Self{
        let n_rel = rel_vecs.shape()[1];
        
        // Generate single-mode indices
        let sing_q_ind: Vec<Vec<i32>> = (0 .. modes.len()).map(|ind| vec![ind as i32]).collect();

        // Generate two-mode indices
        let doub_q_ind: Vec<Vec<i32>> = (0..modes.len()).flat_map(|ind1| {
            (0..modes.len()).map(|ind2| vec![ind1.clone() as i32, ind2 as i32]).collect::<Vec<Vec<i32>>>()
        }).collect();

        // Combine all indices
        let all_q_ind = [sing_q_ind,doub_q_ind].concat();

        // Initialize matrix element storage
        let mut  cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let mut p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        // Define helper functions to calculate matrix elements
        let get_cos = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 { 
            self.get_cos_elements(exciton, rel_vecs, i1, i2, q)
        };

        let get_p_sin = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 {
            self.get_psin_elements(exciton, rel_vecs, i1, i2, q)
        };

        // Populate matrix elements for all mode combinations
        all_q_ind.iter().for_each(|q_inds| 
            for i1 in 0 .. n_rel {
                for i2 in 0 .. n_rel {
                    if q_inds.len() == 1 {
                        // Single-mode case
                        let q = &modes[q_inds[0] as usize].q;

                        // Calculate and store cosine matrix element
                        cos_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_cos(i1, i2, q));

                        // Calculate and store momentum-sine matrix element
                        p_sin_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_p_sin(i1, i2, q));
                        
                    }
                    else if q_inds.len() == 2 {
                        // Two-mode case
                        let q_ind1 = q_inds[0].max(q_inds[1]);
                        let q_ind2 = q_inds[0].min(q_inds[1]);
                        let q_1 = &modes[q_ind1 as usize].q;
                        let q_2 = &modes[q_ind2 as usize].q;
                        let q_diff = q_1 - q_2;
                        let q_sum = q_1 + q_2;

                        // Calculate and store difference matrix element
                        cos_mat.entry((i1,i2,vec![ q_ind1, -q_ind2]))
                            .or_insert_with(|| get_cos(i1, i2, &q_diff));

                        // Calculate and store sum matrix element
                        cos_mat.entry((i1,i2,vec![ q_ind1,  q_ind2]))
                            .or_insert_with(|| get_cos(i1, i2, &q_sum));
                    }
                }
            }
        );

        Self{cos_mat: cos_mat, p_sin_mat: p_sin_mat, all_q_ind: Some(all_q_ind)  }
    }

    pub fn new() -> Self{
        let   cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        Self { cos_mat: cos_mat, p_sin_mat: p_sin_mat, all_q_ind: None  }
    }

    pub fn get_cos_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let eigvecs = &rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let cos_term = (q_x + q_y).cos();
        
        // Calculate matrix element using fast implementation
        fast_matrix_element_calc(
            &psi_i, 
            &psi_j, 
            &cos_term, exciton.rel_ham.grid.dx
        )
    }

    fn get_psin_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        let x_grid =  &exciton.rel_ham.grid.x;
        let y_grid =  &exciton.rel_ham.grid.y;
        let eigvecs = rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;
        let dx = exciton.rel_ham.grid.dx;

        // Get the eigenvectors for states i and j
        let psi_i = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the sine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let sin_term = (q_x + q_y).sin();
        
        // Calculate momentum operator in sinc-DVR basis
        let p = self.compute_sinc_dvr_rel(exciton);
        
        // Calculate p·sin + sin·p term
        let p_sin = &p.dot(&sin_term) + &sin_term.dot(&p);
        
        // Calculate matrix element
        (psi_i * p_sin * psi_j).sum() * dx * dx
    }

    fn compute_sinc_dvr_rel(&self, exciton: &FullHamiltonian) -> Array2<f64> {
        let dx = exciton.rel_ham.grid.dx;
        let nx = exciton.rel_ham.grid.grid_size;

        // Create momentum operator matrix using sinc-DVR formula
        Array2::from_shape_fn((nx,nx), |(i, j)| {
            if i == j {
                // Diagonal elements
                PI.powi(2) / 3.0 / dx*dx
            }
            else {
                // Off-diagonal elements
                let diff: f64 = i as f64 - j as f64;
                (-1.0 as f64).powf(diff) / (diff + f64::EPSILON) / dx 
            }
        })
    }
}

pub struct LMMatrixElementArray {
    pub   cos_mat: HashMap<Vec<i32>, Array2<f64>>, // Cosine matrix elements
    pub p_sin_mat: HashMap<Vec<i32>, Array2<f64>>, // Momentum-sine matrix elements
}

impl LMMatrixElementArray {
    pub fn new(mat_els: &LMMatrixElement, e_vecs: &Vec<Array1<f64>>, parities: Array1<i32>, n_com: usize) -> Self {
        let mut cos_mat:   HashMap<Vec<i32>, Array2<f64>> = HashMap::new();
        let mut p_sin_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();

        let n_eigs = e_vecs.len();

        // Order of the rel_vecs is even then odd
        let mut n_rel_even: Option<usize> = None;
        let mut n_rel_odd: Option<usize> = None;
        let mut unique_parities = 0;

        for (ind, &p) in parities.iter().enumerate() {
            if p == 1 && n_rel_even.is_none() {
                n_rel_even = Some(e_vecs[ind].len() / n_com);
                unique_parities += 1;
            }
            else if p == -1 && n_rel_odd.is_none() {
                n_rel_odd = Some(e_vecs[ind].len() / n_com);
                unique_parities += 1;
            }

            if unique_parities == 2 {
                break;
            }
        }

        // Now unwrap the Options to get the values
        let n_rel_par = vec![n_rel_even.unwrap_or(0), n_rel_odd.unwrap_or(0)];

        if let Some(all_q_ind) = &mat_els.all_q_ind {
            all_q_ind.iter().for_each(|q_inds| {
                                
                let mut mat_1 = Array2::zeros((n_eigs, n_eigs));
                let mut mat_2 = Array2::zeros((n_eigs, n_eigs));

                for idx1 in 0 .. n_eigs {
                    for idx2 in 0 .. n_eigs {
                    let vec1 = &e_vecs[idx1];
                    let vec2 = &e_vecs[idx2];
                    
                    let idx1_par = (parities[idx1] as i32 - 1) / -2; // 0 for even, 1 for odd
                    let idx2_par = (parities[idx2] as i32 - 1) / -2; // 0 for even, 1 for odd

                    let mat_ij = (0..n_rel_par[idx1_par as usize]).flat_map(|rel_ind1| {
                        (0 .. n_rel_par[idx2_par as usize]).flat_map(move |rel_ind2| {

                            let r_ind1 = rel_ind1 + idx1_par as usize * n_rel_even.unwrap();
                            let r_ind2 = rel_ind2 + idx2_par as usize * n_rel_even.unwrap();

                            let mat_element1: &f64;
                            let mat_element2: &f64;

                            if q_inds.len() == 1 {
                                mat_element1 = mat_els.cos_mat.get(&(r_ind1, r_ind2, q_inds.clone()))
                                            .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                            i2 = {}, q_ind = {:?}",r_ind1,r_ind2, q_inds));

                                mat_element2 = mat_els.p_sin_mat.get(&(r_ind1, r_ind2, q_inds.clone()))
                                            .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                            i2 = {}, q_ind = {:?}",r_ind1,r_ind2, q_inds));
                            }
                            else {
                                let q_ind1 = q_inds[0].max(q_inds[1]);
                                let q_ind2 = q_inds[0].min(q_inds[1]);

                                mat_element1 = mat_els.cos_mat.get(&(r_ind1, r_ind2, vec![ q_ind1, -q_ind2]))
                                            .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                            i2 = {}, q_inds = {:?}",r_ind1,r_ind2, vec![ q_ind1, -q_ind2]));

                                mat_element2 = mat_els.cos_mat.get(&(r_ind1, r_ind2, vec![ q_ind1,  q_ind2]))
                                            .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                            i2 = {}, q_inds = {:?}",r_ind1,r_ind2, vec![ q_ind1,  q_ind2]));
                            }
                            
                            (0 .. n_com).map(move |com_ind| {
                                let i1 = n_com * rel_ind1 + com_ind;
                                let i2 = n_com * rel_ind2 + com_ind;
                                let a1 = vec1[i1];
                                let a2 = vec2[i2];

                                (a1 * a2 * mat_element1, a1 * a2 * mat_element2)
                            })

                        })
                    }).collect::<Vec<(f64,f64)>>();

                    (mat_1[[idx1,idx2]], mat_2[[idx1,idx2]]) = mat_ij.iter().fold((0.0, 0.0), |a, b| {
                        (a.0 + b.0, a.1 + b.1)
                    });

                }};

                if q_inds.len() == 1 {
                    cos_mat  .insert(q_inds.clone(), mat_1);
                    p_sin_mat.insert(q_inds.clone(), mat_2);
                }
                else if q_inds.len() == 2 {
                    // Two-mode case
                    let q_ind1 = q_inds[0].max(q_inds[1]);
                    let q_ind2 = q_inds[0].min(q_inds[1]);

                    cos_mat.insert(vec![ q_ind1, -q_ind2], mat_1);
                    cos_mat.insert(vec![ q_ind1,  q_ind2], mat_2);
                }
            });
        }

        Self{cos_mat: cos_mat, p_sin_mat: p_sin_mat}
    }


    pub fn get_cos_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let eigvecs = &rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let cos_term = (q_x + q_y).cos();
        
        // Calculate matrix element using fast implementation
        fast_matrix_element_calc(
            &psi_i, 
            &psi_j, 
            &cos_term, exciton.rel_ham.grid.dx
        )
    }

    fn get_psin_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        let x_grid =  &exciton.rel_ham.grid.x;
        let y_grid =  &exciton.rel_ham.grid.y;
        let eigvecs = rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;
        let dx = exciton.rel_ham.grid.dx;

        // Get the eigenvectors for states i and j
        let psi_i = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the sine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let sin_term = (q_x + q_y).sin();
        
        // Calculate momentum operator in sinc-DVR basis
        let p = self.compute_sinc_dvr_rel(exciton);
        
        // Calculate p·sin + sin·p term
        let p_sin = &p.dot(&sin_term) + &sin_term.dot(&p);
        
        // Calculate matrix element
        (psi_i * p_sin * psi_j).sum() * dx * dx
    }

    fn compute_sinc_dvr_rel(&self, exciton: &FullHamiltonian) -> Array2<f64> {
        let dx = exciton.rel_ham.grid.dx;
        let nx = exciton.rel_ham.grid.grid_size;

        // Create momentum operator matrix using sinc-DVR formula
        Array2::from_shape_fn((nx,nx), |(i, j)| {
            if i == j {
                // Diagonal elements
                PI.powi(2) / 3.0 / dx*dx
            }
            else {
                // Off-diagonal elements
                let diff: f64 = i as f64 - j as f64;
                (-1.0 as f64).powf(diff) / (diff + f64::EPSILON) / dx 
            }
        })
    }
}