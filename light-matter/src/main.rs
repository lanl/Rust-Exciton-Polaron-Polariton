use exciton_2d::{utils::fast_matrix_element_calc, Exciton2DGridSystem, ExcitonConfig, FullHamiltonian};
use std::{collections::HashMap, f64::consts::PI, path::PathBuf, usize};
use ndarray::{arr1, array, Array1, Array2, s};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::path::Path;
use bincode::{self};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use plotters::prelude::*;

mod main_dense;

// Add sprs and arpack dependencies
use sprs::{kronecker_product as kron, smmp::mul_csr_csr, CsMat, CsMatBase, TriMat};
use arpack::{Config, Operator}; 

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

// Speed of light in atomic units
const C: f64 = 137.036;

/// Represents a 3D vector with x, y, z components
#[derive(Debug, Clone, Copy)]
pub struct Vector3D {
    x: f64,
    y: f64,
    z: f64,
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
                        cos_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_cos(i1, i2, &q_sum));
                    }
                }
            }
        );

        Self{cos_mat: cos_mat, p_sin_mat: p_sin_mat}
    }

    pub fn new() -> Self{
        let   cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        Self { cos_mat: cos_mat, p_sin_mat: p_sin_mat }
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

/// Represents the photonic part of the Hamiltonian
pub struct PhotonicHamiltonian {
    pub photon_modes: Vec<PhotonMode>,
}

impl PhotonicHamiltonian {
    pub fn new(photon_modes: Vec<PhotonMode>) -> Self {
        Self { photon_modes: photon_modes }
    }

    /// Creates a single-mode photonic Hamiltonian matrix
    pub fn get_h_ph_sing_sprs(&self, mode_i: usize) -> CsMat<f64> {
        let n_fock = self.photon_modes[mode_i].n_fock;
        let n_max = (n_fock - 1) as f64;

        let mode = &self.photon_modes[mode_i];
        
        let dense = Array2::from_diag(
            &Array1::linspace(0.0, mode.omega * n_max, mode.n_fock)
        );

        sprs::CsMatI::csr_from_dense(dense.view(), 1e-10)
    } 

    pub fn get_h_ph_sing_full_sprs(&self, mode_i: usize) -> CsMat<f64> {
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // For the selected mode, create energy diagonal
                self.get_h_ph_sing_sprs(mode_i)
            }
            else {
                // For other modes, use identity matrix
                CsMat::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(acc.view(), sing_mat.view()))
        .unwrap()
    } 

    /// Creates a sparse photonic Hamiltonian
    pub fn get_h_ph_sparse(&self) -> CsMat<f64> {
        if self.photon_modes.is_empty() {
            // Return 1x1 identity matrix
            return CsMat::eye(1);
        }

        // Sum Hamiltonians for each mode
        self.photon_modes.iter().enumerate().map(|(mode_i, _)| self.get_h_ph_sing_full_sprs(mode_i))
            .reduce(|acc, h_ph_sing| &acc + &h_ph_sing)
            .unwrap()
    }
}

/// Represents the light-matter interaction Hamiltonian
#[derive(Clone)]
pub struct LMIntHamiltonian {
    pub exciton: FullHamiltonian,        // Exciton Hamiltonian
    pub photon_modes: Vec<PhotonMode>,    // Photon modes
    pub coupling: f64,                    // Light-matter coupling strength
    pub exciton_vecs: Vec<Array1<f64>>,   // Exciton eigenvectors
    pub rel_vecs: Array2<f64>             // Relative coordinate eigenvectors
}

impl LMIntHamiltonian {
    pub fn new (exciton: FullHamiltonian, 
                modes:  Vec<PhotonMode>, 
                coupling: f64, 
                e_vecs: Vec<Array1<f64>>,
                r_vecs: Array2<f64>) -> Self {
        Self {  exciton: exciton, 
                photon_modes: modes, 
                coupling: coupling,
                exciton_vecs: e_vecs,
                rel_vecs: r_vecs}
    }

    /// Constructs the sparse light-matter interaction Hamiltonian
    pub fn get_int_h_sparse(&self) -> CsMat<f64> {
        // Calculate matrix elements for the interaction
        let matrix_elements = LMMatrixElement::new()
            .populate_mat_elements(&self.rel_vecs, &self.photon_modes, &self.exciton);

        let kappas = &self.exciton.kappa_vectors;
        let k = &self.exciton.k;

        // Calculate dimensions of the Hilbert space
        let n_ph = self.photon_modes.iter()
            .fold(1, |acc, mode| acc * mode.n_fock);
        let n_ex = self.exciton_vecs.len();
        let total_dim = n_ex * n_ph;
        
        let i_ex: CsMat<f64> = CsMat::eye(n_ex);

        // Initialize interaction Hamiltonian as a triplet matrix (for building sparse matrix)
        let mut h_int: CsMat<f64> = CsMat::zero((total_dim, total_dim));

        let dummy_photon = PhotonicHamiltonian::new(self.photon_modes.clone());


        // Define matrix elements HashMaps
        let cos_x = &matrix_elements.cos_mat;
        let p_sin_x = &matrix_elements.p_sin_mat;
        
        // Iterate over all photon modes
        for (mode_i, mode1) in self.photon_modes.iter().enumerate() {
            let m_com = self.exciton.m;
            let q = &mode1.q;
            let q_ind = vec![mode_i as i32];

            // Define photonic operators
            let a1 = self.get_a_sing_full(mode_i);
            let a1_dag = a1.clone().transpose_into().to_csr();
            let aa_dag1 = self.get_h_ph_sing_full(mode_i, &dummy_photon);
            let x_cav_1 = &a1 + &a1_dag;
            let p_cav_1 = &a1_dag - &a1;

            // Define relative operators
            let cos_x_mat = self.get_matrix_from_hash_sparse(&q_ind, &cos_x);
            let p_sin_x_mat = self.get_matrix_from_hash_sparse(&q_ind, &p_sin_x);

            // Define P operators in COM basis
            let p_dot_eps = self.get_p_com_sparse(kappas, k, &mode1.eps);
            let p_dot_q = self.get_p_com_sparse(kappas, k, &mode1.q);

            // P·q (a† a) term
            let delta: CsMat<f64> = kron(p_dot_q.view(), aa_dag1.view());
            h_int = &h_int + &delta;

            // -P·A cos(q·r) (a† + a) term
            let p_cos_factor = - 2.0 * self.coupling / m_com;
            let p_cos_matrix: CsMat<f64> = &mul_csr_csr(p_dot_eps.view(), cos_x_mat.view()) * p_cos_factor;
            let delta: CsMat<f64> = kron(p_cos_matrix.view(), x_cav_1.view());
            h_int = &h_int + &delta;
            
            // ∇·A sin(q·r) (a† - a) term
            let p_sin_factor = self.coupling * 2.0 / self.exciton.rel_ham.mu;
            let delta = kron((&p_sin_x_mat * p_sin_factor).view(), p_cav_1.view());
            h_int = &h_int + &delta;
            
            // Iterate over all photon modes again for quadratic terms
            for (mode_j, mode2) in self.photon_modes.iter().enumerate() {
                let q_2 = &mode2.q;
                let qq = q.dot(q_2);

                // q_i·q_j a_i†a_i a_j†a_j term (A²)
                let aa_2 = &(mul_csr_csr(
                    aa_dag1.view(), 
                    self.get_h_ph_sing_full(mode_j, &dummy_photon).view()))
                    * (qq / 2.0 / m_com);
                let delta = kron(i_ex.view(), aa_2.view());
                h_int = &h_int + &delta;

                // Diamagnetic terms (A²)
                let a2 = self.get_a_sing_full(mode_j);
                let a2_dag = a2.clone().transpose_into().to_csr();
                let q_ind1 = mode_i.max(mode_j) as i32;
                let q_ind2 = mode_i.min(mode_j) as i32;
                
                // cos((q_i + q_j)·r) (a_i a_j + a_i† a_j†) term
                let cos_sum_matrix = self.get_matrix_from_hash_sparse(&vec![q_ind1, q_ind2], cos_x);
                let diamag_factor = self.coupling.powi(2);

                let delta = kron(
                    (&cos_sum_matrix * diamag_factor).view(), 
                    (&(mul_csr_csr(a1.view(), a2.view()) as CsMat<f64>) 
                        + &(mul_csr_csr(a1_dag.view(), a2_dag.view()) as CsMat<f64>)
                ).view());

                h_int = &h_int + &delta;

                // cos((q_i - q_j)·r) (a_i† a_j + a_i a_j†) term
                let cos_diff_matrix = self.get_matrix_from_hash_sparse(&vec![q_ind1, -q_ind2], cos_x);
                
                let delta = kron(
                    (&cos_diff_matrix * diamag_factor).view(), 
                    (&(mul_csr_csr(a1_dag.view(), a2.view()) as CsMat<f64>) 
                        + &(mul_csr_csr(a1.view(), a2_dag.view()) as CsMat<f64>)
                ).view());

                h_int = &h_int + &delta;
            }
        }
        
        // Convert to CSR format
        h_int.to_csr()
    }

    /// Creates the annihilation operator for a single mode, tensored with identity for other modes
    fn get_a_sing_full(&self, mode_i: usize)  -> CsMat<f64> {
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // Create annihilation operator for this mode
                let dim = mode.n_fock;
                let mut a = TriMat::new((dim, dim));
                for n in 1..dim {
                    a.add_triplet(n-1, n, (n as f64).sqrt());
                }
                a.to_csr()
            }
            else {
                // Identity for other modes
                CsMat::<f64>::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(acc.view(), sing_mat.view()))
        .unwrap()
    }

    /// Creates the photon number operator for a single mode, tensored with identity for other modes
    fn get_h_ph_sing_full(&self, mode_i: usize, dummy_photon: &PhotonicHamiltonian) -> CsMat<f64>{
        dummy_photon.get_h_ph_sing_full_sprs(mode_i)
    }

    /// Constructs a sparse matrix from matrix elements stored in a hash map
    pub fn get_matrix_from_hash_sparse(&self, q_inds: &Vec<i32>, mat_hash: &HashMap<(usize,usize, Vec<i32>), f64>) -> CsMat<f64> {
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;
        
        // Initialize triplet matrix for building sparse matrix
        let mut mat = TriMat::new((n_eig, n_eig));
        
        // Build matrix for relative coordinate operators
        for idx1 in 0..n_eig {
            let vec1 = &e_vecs[idx1];
            
            for idx2 in 0..n_eig {
                let vec2 = &e_vecs[idx2];
                
                    // Calculate matrix element in the full basis
                let mat_ij: f64 = (0..n_rel).flat_map(|rel_ind1| {
                        (0 .. n_rel).flat_map(move |rel_ind2| {
                            // Get the matrix element for these relative states
                            let mat_element = mat_hash.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                        .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                        i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds));
                            
                            // Sum over center-of-mass states (diagonal in CoM)
                            (0..n_com).map(move |com_ind| {
                                let i1 = n_com * rel_ind1 + com_ind;
                                let i2 = n_com * rel_ind2 + com_ind;

                                let a1 = vec1[i1];
                                let a2 = vec2[i2];

                                a1 * a2 * mat_element
                            })
                        })
                    }).sum();
                
                // Only add non-zero elements to sparse matrix
                if mat_ij.abs() > 1e-10 {
                    mat.add_triplet(idx1, idx2, mat_ij);
                }
            }
        }
        
        mat.to_csr()
    }

    /// Calculates the sparse center-of-mass momentum matrix element P·field_vec
    pub fn get_p_com_sparse(&self, kappas: &Vec<Vec<f64>>, k: &Vec<f64>, field_vec: &Array1<f64>) -> CsMat<f64> {
        let e_vecs = &self.exciton_vecs;
        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;
        
        // Convert k to Array1 once for efficiency
        let k_arr = arr1(&k);
        
        // Precompute p_dot_field for each component
        let p_dot_field: Vec<f64> = kappas.iter()
            .map(|kappa| {
                // Calculate (k + kappa)·field_vec for each kappa point
                (k_arr.clone() + arr1(kappa)).dot(field_vec)
            })
            .collect();
        
        // Initialize triplet matrix for building sparse matrix
        let mut mat = TriMat::new((n_eig, n_eig));
        
        // Build matrix for momentum operator
        for idx1 in 0..n_eig {
            let vec1 = &e_vecs[idx1];
            
            for idx2 in 0..n_eig {
                let vec2 = &e_vecs[idx2];
                
                // This is a purely diagonal operator
                let p_ij: f64 = (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_field[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        let a2 = vec2[i];
                        a1 * a2 * p
                    })
                }).sum();
                
                // Only add non-zero elements to sparse matrix
                if p_ij.abs() > 1e-10 {
                    mat.add_triplet(idx1, idx2, p_ij);
                }
            }
        }
        
        mat.to_csr()
    }
}

/// Main light-matter coupled system class
pub struct LightMatter {
    pub photon: PhotonicHamiltonian,      // Photonic part of the Hamiltonian
    pub interaction: LMIntHamiltonian,    // Light-matter interaction Hamiltonian
    pub exciton_energies: Array1<f64>,    // Exciton energies
    n_rel: usize,                         // Number of relative coordinate states
    n_ex_eig: usize,                      // Number of exciton eigenstates
    parity: i32                           // Parity of the states to consider
}

impl LightMatter {
    /// Creates a new light-matter coupled system
    pub fn new( exciton_ham: FullHamiltonian, 
                photon_modes: Vec<PhotonMode>, 
                coupling: f64,
                n_rel:usize, 
                n_ex_eig:usize, 
                parity:i32) -> Self {

        // Diagonalize the exciton Hamiltonian
        let (exciton_energies, e_vecs, _, r_vecs, _) 
            = exciton_ham.diagonalize_full_hamiltonian(n_rel, n_ex_eig, parity);

        // Create photonic Hamiltonian
        let photon = PhotonicHamiltonian::new(photon_modes.clone());
        
        // Create interaction Hamiltonian
        let interaction = LMIntHamiltonian::new(exciton_ham, photon_modes.clone(), coupling, e_vecs, r_vecs);
        
        Self { photon: photon, interaction: interaction, exciton_energies: exciton_energies, n_rel: n_rel,
            n_ex_eig: n_ex_eig, parity: parity}
    }

    /// Builds the complete light-matter Hamiltonian as a sparse matrix
    pub fn build_lm_hamiltonian_sparse(&self) -> CsMat<f64> {
        // Get dimensions
        let n_ex = self.exciton_energies.len();
        let n_ph = self.photon.photon_modes.iter()
            .fold(1, |acc, mode| acc * mode.n_fock);
                
        // Add exciton Hamiltonian terms (diagonal in exciton basis)
        let h_ex: CsMat<f64> = CsMatBase::csr_from_dense(Array2::from_diag(&self.exciton_energies).view(), 1e-10);
        
        // Get photonic Hamiltonian (sparse)
        let h_ph = self.photon.get_h_ph_sparse();
                
        // Get interaction Hamiltonian (sparse)
        let h_int = self.interaction.get_int_h_sparse();
                
        // Add together to form total Hamiltonian
        &(&kron(h_ex.view(), CsMat::eye(n_ph).view()) 
        + &kron(CsMat::eye(n_ex).view(), h_ph.view())) 
        + &h_int
    }

    /// Diagonalizes the light-matter Hamiltonian using ARPACK
    pub fn diagonalize_lm_ham_sparse(&self, n_eigenvalues: usize) -> (Array1<f64>, Array2<f64>) {
        // Build the sparse Hamiltonian
        let h_lm = self.build_lm_hamiltonian_sparse();
        let (n, _) = h_lm.shape();
        
        // Create ARPACK operator
        let op = ArpackOperator::new(h_lm);
        
        // Configure ARPACK
        let config = Config::new()
            .which(arpack::Which::SA)  // Smallest algebraic eigenvalues
            .tol(1e-8)                // Tolerance
            .max_iter(1000);          // Maximum iterations
        
        // Solve eigenvalue problem
        let result = arpack::eigs(op, n,n_eigenvalues, config).unwrap();
        
        // Convert to ndarray format
        let eigenvalues = Array1::from_vec(result.values);
        
        // Convert eigenvectors to ndarray format
        let n = result.vectors.len() / n_eigenvalues;
        let mut eigenvectors = Array2::zeros((n, n_eigenvalues));
        
        for i in 0..n_eigenvalues {
            for j in 0..n {
                eigenvectors[[j, i]] = result.vectors[i * n + j];
            }
        }
        
        (eigenvalues, eigenvectors)
    }

    /// Creates a new LightMatter object with updated k-vector
    pub fn update_k(&self, k: Vec<f64>) -> Self {
        // Clone the exciton Hamiltonian
        let mut exciton = self.interaction.exciton.clone();
        
        // Update the k-vector
        exciton.k = k;

        // Create a new LightMatter object with the updated exciton
        LightMatter::new(exciton, self.photon.photon_modes.clone(), self.interaction.coupling,self.n_rel,
            self.n_ex_eig, self.parity)
    }
}

/// ARPACK operator for sparse matrix eigenvalue problems
struct ArpackOperator {
    matrix: CsMat<f64>,
}

impl ArpackOperator {
    fn new(matrix: CsMat<f64>) -> Self {
        Self { matrix }
    }
}

impl Operator<f64> for ArpackOperator {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        
        // Apply matrix-vector product
        let mut result = vec![0.0; y.len()];
        sprs::prod::mul_acc_mat_vec_csr(self.matrix.view(), x, &mut result);
        y.copy_from_slice(&result);
    }
    
}

/// Calculator for band structure of light-matter coupled systems
pub struct BandStructureCalculatorLM {
    pub light_matter: LightMatter,      // Light-matter system
    pub a0: f64,                        // Lattice constant
    pub v0: f64,                        // Potential strength
    pub save_dir: String,               // Directory to save results
    pub sym_points: Vec<Vec<f64>>       // High-symmetry points in k-space
}

impl BandStructureCalculatorLM {
    /// Creates a new band structure calculator
    pub fn new(light_matter: LightMatter, a0: f64, v0: f64, save_dir: String, sym_points: Vec<Vec<f64>>) -> Self {
        Self { light_matter: light_matter, a0: a0, v0: v0, save_dir: save_dir, sym_points: sym_points}
    }

    /// Calculates eigenvalues and eigenvectors at a specific k-point using sparse matrices
    pub fn calculate_at_k_point_lm_sparse(&self, k: Vec<f64>, n_eigenvalues: usize) -> (Array1<f64>, Array2<f64>) {
        // Update the light-matter system with the new k-point
        let light_matter = self.light_matter.update_k(k);

        // Diagonalize the Hamiltonian at this k-point using ARPACK
        light_matter.diagonalize_lm_ham_sparse(n_eigenvalues)
    }

    /// Generates a filename for the band structure data
    pub fn get_lm_band_structure_filename(
        &self,
        n_points: usize,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}_par{}_coup{}_nmodes{}.bin",
            self.save_dir,
            self.a0,
            self.v0,
            self.light_matter.interaction.exciton.rel_ham.grid.grid_size,
            self.light_matter.interaction.exciton.rel_ham.grid.box_size,
            self.light_matter.n_rel,
            self.light_matter.interaction.exciton.kappa_vectors.len(),
            n_points,
            self.light_matter.parity,
            self.light_matter.interaction.coupling,
            self.light_matter.photon.photon_modes.len(),
        )
    }
    
    /// Calculates the full band structure along a path in k-space using sparse matrices
    pub fn calculate_lm_band_structure_sparse(&self, n_points: usize, n_eigenvalues: usize) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        // Generate filename for saving results
        let filename = self.get_lm_band_structure_filename(n_points);

        // Calculate points per segment of the path
        let points_per_seg = n_points / self.sym_points.len();

        // Generate k-points along the path
        let k_points: Vec<Vec<f64>> = self.sym_points.iter().enumerate().flat_map(|(sym_ind,sym_point)| {
            let sym_ind2 = (sym_ind + 1) % self.sym_points.len();
            (0..points_per_seg).map(move |t| {
                // Interpolate between symmetry points
                let t_frac = t as f64 / (points_per_seg - 1) as f64;
                let x = sym_point[0] + t_frac * (self.sym_points[sym_ind2][0] - sym_point[0]);
                let y = sym_point[1] + t_frac * (self.sym_points[sym_ind2][1] - sym_point[1]);

                vec![x,y]
            })
        }).collect();

        // Set up progress bar
        let total = k_points.len() as u64;
        let pb = ProgressBar::new(total);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} k-points ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        // Calculate eigenvalues at each k-point
        let eigvals_list: Vec<Array1<f64>> = k_points.iter().map(|k| {
            let (eigvals, _) = self.calculate_at_k_point_lm_sparse(k.clone(), n_eigenvalues);
            pb.inc(1);
            eigvals
        }).collect();
        pb.finish_with_message("Bandstructure calculation complete");

        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(&filename).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        // Save the band structure data
        save_band_data(filename, (k_points.clone(), eigvals_list.clone())).unwrap();

        (k_points, eigvals_list)
    }
}

/// Represents a polaron-polariton system (exciton-polariton coupled to phonons)
pub struct PolaronPolariton {
    pub phonon_modes: Vec<PhotonMode>,    // Phonon modes (using PhotonMode structure)
    pub polariton: LightMatter,           // Exciton-polariton system
    pub coupling: Vec<f64>,               // Coupling strengths to phonon modes
    pub pp_int: PPIntHamiltonian,         // Polaron-polariton interaction Hamiltonian
}

impl PolaronPolariton {
    /// Creates a new polaron-polariton system
    pub fn new(phonon_modes: Vec<PhotonMode>, polariton: LightMatter, coupling: Vec<f64>) -> Self{
        // Create the interaction Hamiltonian
        let pp_int = PPIntHamiltonian::new(polariton.interaction.clone(), 
                                           polariton.photon.photon_modes.clone(), 
                                           coupling.clone());

        Self { phonon_modes: phonon_modes, polariton: polariton, coupling: coupling, pp_int: pp_int}
    }

    /// Builds the complete polaron-polariton Hamiltonian as a sparse matrix
    pub fn build_pp_hamiltonian_sparse(&self) -> CsMat<f64> {
        // Get dimensions
        let n_lm = self.polariton.exciton_energies.len() * 
                   self.polariton.photon.photon_modes.iter()
                       .fold(1, |acc, mode| acc * mode.n_fock);
        let n_pn = self.phonon_modes.iter()
                       .fold(1, |acc, mode| acc * mode.n_fock);
        let total_dim = n_lm * n_pn;
        
        // Create sparse matrix
        let mut h_pp = TriMat::new((total_dim, total_dim));
        
        // Get the light-matter Hamiltonian (sparse)
        let h_lm = self.polariton.build_lm_hamiltonian_sparse();
        
        // Add light-matter terms (tensor with identity for phonons)
        for (&val, (i, j)) in h_lm.iter() {
            for k in 0..n_pn {
                let row = i * n_pn + k;
                let col = j * n_pn + k;
                h_pp.add_triplet(row, col, val);
            }
        }
        
        // Get the phonon Hamiltonian (sparse)
        let h_pn = self.get_h_pn_sparse();
        
        // Add phonon terms (tensor with identity for light-matter)
        for i in 0..n_lm {
            for (&val, (j, k)) in h_pn.iter() {
                let row = i * n_pn + j;
                let col = i * n_pn + k;
                h_pp.add_triplet(row, col, val);
            }
        }
        
        // Get the interaction Hamiltonian (sparse)
        let h_pp_int = self.pp_int.get_pp_int_h_sparse();
        
        // Add interaction terms
        for (&val, (i, j)) in h_pp_int.iter() {
            h_pp.add_triplet(i, j, val);
        }
        
        // Convert to CSR format
        h_pp.to_csr()
    }

    /// Gets the phonon Hamiltonian as a sparse matrix
    pub fn get_h_pn_sparse(&self) -> CsMat<f64> {
        // Reuse PhotonicHamiltonian for phonons (same structure)
        let phonon = PhotonicHamiltonian::new(self.phonon_modes.clone());
        phonon.get_h_ph_sparse()
    }
    
    /// Diagonalizes the polaron-polariton Hamiltonian using ARPACK
    pub fn diagonalize_pp_ham_sparse(&self, n_eigenvalues: usize) -> (Array1<f64>, Array2<f64>) {
        // Build the sparse Hamiltonian
        let h_pp = self.build_pp_hamiltonian_sparse();
        let (n, _) = h_pp.shape();
        
        // Create ARPACK operator
        let op = ArpackOperator::new(h_pp);
        
        // Configure ARPACK
        let config = Config::new()
            .which(arpack::Which::SA)  // Smallest algebraic eigenvalues
            .tol(1e-8)                // Tolerance
            .max_iter(1000);          // Maximum iterations
        
        // Solve eigenvalue problem
        let result = arpack::eigs(op, n, n_eigenvalues, config).unwrap();
        
        // Convert to ndarray format
        let eigenvalues = Array1::from_vec(result.values);
        
        // Convert eigenvectors to ndarray format
        let n = result.vectors.len() / n_eigenvalues;
        let mut eigenvectors = Array2::zeros((n, n_eigenvalues));
        
        for i in 0..n_eigenvalues {
            for j in 0..n {
                eigenvectors[[j, i]] = result.vectors[i * n + j];
            }
        }
        
        (eigenvalues, eigenvectors)
    }
}

/// Represents the polaron-polariton interaction Hamiltonian
pub struct PPIntHamiltonian {
    pub polariton_int: LMIntHamiltonian,  // Light-matter interaction Hamiltonian
    pub phonon_modes: Vec<PhotonMode>,    // Phonon modes
    pub coupling: Vec<f64>,               // Coupling strengths
}

impl PPIntHamiltonian {
    /// Creates a new polaron-polariton interaction Hamiltonian
    pub fn new (polariton_int: LMIntHamiltonian,
                phonon_modes: Vec<PhotonMode>,
                coupling: Vec<f64>, ) -> Self {
        Self {  polariton_int: polariton_int,
                phonon_modes: phonon_modes,
                coupling: coupling}
    }

    /// Constructs the polaron-polariton interaction Hamiltonian as a sparse matrix
    pub fn get_pp_int_h_sparse(&self) -> CsMat<f64> {
        // Calculate matrix elements
        let cos_x = self.get_pp_matrix_elements();

        // Get dimensions
        let n_ex = self.polariton_int.exciton_vecs.len();
        let n_ph = self.polariton_int.photon_modes.iter()
                       .fold(1, |acc, mode| acc * mode.n_fock);
        let n_pn = self.phonon_modes.iter()
                       .fold(1, |acc, mode| acc * mode.n_fock);
        let total_dim = n_ex * n_ph * n_pn;
        
        // Create sparse matrix
        let mut h_int = TriMat::new((total_dim, total_dim));

        // Create identity matrix for photons
        let i_ph: sprs::CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<_>> = CsMat::eye(n_ph);

        // Iterate over all phonon modes
        for (mode_i, _) in self.phonon_modes.iter().enumerate() {
            // 2 gamma cos(k·r) (b† + b) term
            let coupling = self.coupling[mode_i];
            let k_ind = vec![mode_i as i32];
            
            // Get matrix elements
            let g_cos_matrix = self.polariton_int.get_matrix_from_hash_sparse(&k_ind, &cos_x);
            let g_cos_factor = coupling * 2.0;
            
            // Get phonon creation/annihilation operators
            let b = self.get_b_sing_full_sparse(mode_i);
            let b_dag = b.transpose_view();
            
            // Add interaction terms
            for (&val, (i, j)) in g_cos_matrix.iter() {
                // Skip very small elements
                if val.abs() < 1e-10 { continue; }
                
                // For each photon state
                for ph_idx in 0..n_ph {
                    // For each phonon state
                    for (&pn_val, (pn_i, pn_j)) in b.iter() {
                        // b term
                        let row = (i * n_ph + ph_idx) * n_pn + pn_i;
                        let col = (j * n_ph + ph_idx) * n_pn + pn_j;
                        h_int.add_triplet(row, col, val * g_cos_factor * pn_val);
                    }
                    
                    // b† term
                    for (&pn_val, (pn_i, pn_j)) in b_dag.iter() {
                        let row = (i * n_ph + ph_idx) * n_pn + pn_i;
                        let col = (j * n_ph + ph_idx) * n_pn + pn_j;
                        h_int.add_triplet(row, col, val * g_cos_factor * pn_val);
                    }
                }
            }
        }

        // Convert to CSR format
        h_int.to_csr()
    }

    /// Creates the phonon annihilation operator for a single mode as a sparse matrix
    pub fn get_b_sing_full_sparse(&self, mode_i: usize) -> CsMat<f64> {
        let n_modes = self.phonon_modes.len();
        let n_fock_total = self.phonon_modes.iter()
                               .fold(1, |acc, mode| acc * mode.n_fock);
        
        // Create sparse matrix
        let mut b_mat = TriMat::new((n_fock_total, n_fock_total));
        
        // Add non-zero elements
        for state_idx in 0..n_fock_total {
            let mut indices = Vec::with_capacity(n_modes);
            let mut temp_idx = state_idx;
            
            // Calculate indices for this state
            for mode in &self.phonon_modes {
                indices.push(temp_idx % mode.n_fock);
                temp_idx /= mode.n_fock;
            }
            
            // Check if annihilation is possible
            if indices[mode_i] > 0 {
                // Create new state with one less phonon
                let mut new_indices = indices.clone();
                new_indices[mode_i] -= 1;
                
                // Calculate new state index
                let mut new_state_idx = 0;
                let mut multiplier = 1;
                for (i, &idx) in new_indices.iter().enumerate() {
                    new_state_idx += idx * multiplier;
                    multiplier *= self.phonon_modes[i].n_fock;
                }
                
                // Add matrix element
                let factor = (indices[mode_i] as f64).sqrt();
                b_mat.add_triplet(new_state_idx, state_idx, factor);
            }
        }
        
        // Convert to CSR format
        b_mat.to_csr()
    }

    /// Calculates matrix elements for polaron-polariton interaction
    pub fn get_pp_matrix_elements(&self) -> HashMap<(usize,usize, Vec<i32>), f64>{
        // Create matrix element calculator
        let mat_calc = LMMatrixElement::new();

        // Get relevant data
        let rel_vecs = &self.polariton_int.rel_vecs;
        let modes = &self.phonon_modes;
        let exciton = &self.polariton_int.exciton;

        let n_rel = rel_vecs.shape()[1];

        // Define helper function to calculate cosine elements
        let get_cos = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 { 
            mat_calc.get_cos_elements(exciton, rel_vecs, i1, i2, q)
        };

        // Initialize matrix element storage
        let mut cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        // Calculate matrix elements for all phonon modes
        modes.iter().enumerate().for_each(|(k_ind,mode)| {
            for i1 in 0 .. n_rel {
                for i2 in 0 .. n_rel {
                    let k = &mode.q;

                    // Calculate and store cosine matrix element
                    cos_mat.entry((i1,i2,vec![k_ind as i32]))
                        .or_insert_with(|| get_cos(i1, i2, k));
                }
            }
        });

        cos_mat
    }
}

/// Plots the band structure
pub fn plot_band_structure(
    k_points: &Vec<Vec<f64>>,           // k-points along the path
    sym_points: Vec<Vec<f64>>,          // High-symmetry points
    sym_labels: Vec<String>,            // Labels for high-symmetry points
    eigvals_list_full: &Vec<Array1<f64>>, // Eigenvalues at each k-point
    n_bands: usize,                     // Number of bands to plot
    a0: f64,                            // Lattice constant
    save_path: Option<&str>,            // Path to save the plot
    _show: bool,                        // Whether to show the plot (unused)
    v0: f64,                            // Potential strength
    coupling: f64,                      // Light-matter coupling
) {
    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);
        
        // Create a larger drawing area with better resolution
        let root = BitMapBackend::new(path, (1200, 900)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Create x-axis for the plot (path length)
        let mut x = vec![0.0; k_points.len()];
        for i in 1..k_points.len() {
            let dx = k_points[i][0] - k_points[i-1][0];
            let dy = k_points[i][1] - k_points[i-1][1];
            x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
        }
        
        let dx = k_points[1][0] - k_points[0][0];

        // Define the high-symmetry points
        let points_per_segment = k_points.len() / 3;
        let n_sym = sym_points.len();
        
        // Calculate positions of symmetry points on the x-axis
        let mut sym_positions: Vec<f64> = sym_points.iter().enumerate().map(|(ind, _)| {
            x[ind * points_per_segment]
        }).collect();
        sym_positions.push(x[x.len() - 1]);

        // Find min/max energy values with some padding
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;

        // Limit the number of bands to plot
        let n_bands = n_bands.min(eigvals_list_full[0].len());

        // Extract the bands to plot
        let eigvals_list: &Vec<Array1<f64>> = &eigvals_list_full.iter().map(|eigvals| {
            eigvals.slice(s![0..n_bands-1]).to_owned()
        }).collect();
        
        // Find energy range
        for eigvals in eigvals_list {
            min_energy = min_energy.min(eigvals.fold(f64::INFINITY, |a, &b| a.min(b)));
            max_energy = max_energy.max(eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
        
        // Add padding to the energy range (5%)
        let energy_padding = (max_energy - min_energy) * 0.05;
        min_energy -= energy_padding;
        max_energy += energy_padding;
        
        // Add a title with parameters
        let title = format!("Exciton-Polariton Band Structure (a₀={:.1}, V₀={:.1}, g={:.3})", 
                           a0, v0, coupling);

        // Set up chart with better styling
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30).into_font())
            .margin(10)
            .margin_top(20)
            .margin_bottom(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                x[0]..x[x.len() - 1],
                min_energy..max_energy
            )
            .unwrap();
        
        // Configure mesh with custom x-ticks for high-symmetry points
        chart.configure_mesh()
            .x_labels(n_sym + 1)
            .x_label_formatter(&|v| {
                for (ind, &x_pos) in sym_positions.iter().enumerate() {
                    if (v - x_pos).abs() < dx {
                        if ind < n_sym {
                            return sym_labels[ind].clone();
                        } else {
                            return sym_labels[0].clone(); // Last point wraps to first label
                        }
                    }
                }
                "".to_string()
            })
            .y_desc("Energy (a.u.)")
            .axis_desc_style(("sans-serif", 20))
            .label_style(("sans-serif", 18))
            .disable_mesh()
            .draw()
            .unwrap();
        
        // Draw vertical lines at high-symmetry points
        let line_style = ShapeStyle {
            color: RGBColor(80, 80, 80).mix(0.8),
            filled: false,
            stroke_width: 1,
        };
        
        for &x_pos in &sym_positions {
            chart.draw_series(LineSeries::new(
                vec![(x_pos, min_energy), (x_pos, max_energy)],
                line_style,
            )).unwrap();
        }
        
        // Draw each band with the same color (blue)
        let num_bands = (&eigvals_list[0]).len();
        println!("Number of bands: {}", num_bands );
        
        for band_idx in 0..num_bands {
            let color = BLUE;
            let band_points: Vec<(f64, f64)> = x.iter()
                .zip(eigvals_list.iter())
                .map(|(&x_val, eigvals)| (x_val, eigvals[band_idx]))
                .collect();
            
            chart.draw_series(LineSeries::new(
                band_points,
                ShapeStyle {
                    color: color.mix(0.9),
                    filled: false,
                    stroke_width: 2,
                }
            )).unwrap();
        }
        
        // Present the final plot
        root.present().unwrap();
        info!("Band structure plot saved to {}", path);
    }
}

/// Main function demonstrating the use of the light-matter coupling framework
fn main() {
    // Create exciton configuration
    let e_config = ExcitonConfig {
        n_kappa: 11,                           // Number of k-points in Brillouin zone
        a0: 9.0,                               // Lattice constant
        v0: 0.01,                              // Potential strength
        k: vec![0.0, 0.0],                     // Initial k-vector
        m_e: 0.3,                              // Electron effective mass
        m_h: 0.3,                              // Hole effective mass
        verbose: false,                        // Verbose output flag
        grid_size: 128,                        // Grid size for real-space calculations
        box_size: 3.0,                         // Real-space box size
        save_dir: PathBuf::from("saved_data_lm"), // Directory to save data
        recalculate: false,                    // Whether to recalculate existing data
        n_jobs: 2,                             // Number of parallel jobs
        use_gpu: false,                        // Whether to use GPU acceleration
        batch_size: 32,                        // Batch size for calculations
    };

    // Create the 2D exciton system
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;
    let exciton = Exciton2DGridSystem::from_config(e_config);
    
    // Parameters for diagonalization
    let n_rel = 16;      // Number of relative coordinate states
    let n_ex_eig = 100;  // Number of exciton eigenstates
    let parity = 0;      // Parity of states to consider
    
    // Light-matter coupling parameters
    let coupling = 0.00;  // Coupling strength
    let omega_0 = 0.0;    // Photon frequency
    
    // Define photon modes
    let qx_values = vec![0.0];
    let qz = omega_0 / C;
    let n_fock = 2;      // Dimension of Fock space
    
    // Create photon modes
    let photon_modes: Vec<PhotonMode> = qx_values.iter().map(|qx| {
            PhotonMode::new(Vector3D { x: *qx, y: 0.0, z: qz }, Polarization::TE, n_fock)
        }).collect();
    
    // Create light-matter system
    let light_matter = LightMatter::new(exciton.full_ham, photon_modes, coupling,
                                    n_rel, n_ex_eig, parity);

    // Define high-symmetry points in k-space for band structure calculation
    // Here we use Γ to X path (center to edge of Brillouin zone)
    let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0]];
    let sym_labels = vec!["Γ".to_string(), "X".to_string()];

    // Start timing the band structure calculation
    let start_time = Instant::now();
    
    // Create band structure calculator
    let band_structure = BandStructureCalculatorLM::new(
        light_matter, a0, v0, "lm_data".to_string(), sym_points.clone());
    
    // Calculate band structure with 60 points along the path
    // Using sparse matrices and ARPACK to calculate 40 lowest eigenvalues
    let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure_sparse(60, 40);
    
    // End timing and print duration
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Calc dispersion in {:?}", duration);

    // Plot the band structure, showing 40 bands
    let n_bands = 40;
    plot_band_structure(
        &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
        Some("disp_plot.png"), false, v0, coupling);
    
    // // Define phonon modes for polaron-polariton calculation
    // let phonon_freq = 0.01;  // Phonon frequency in a.u.
    // let phonon_q = phonon_freq / C;
    // let phonon_modes: Vec<PhotonMode> = vec![
    //     PhotonMode::new(Vector3D { x: phonon_q, y: 0.0, z: 0.0 }, Polarization::TE, 2),
    //     PhotonMode::new(Vector3D { x: 0.0, y: phonon_q, z: 0.0 }, Polarization::TE, 2)
    // ];
    
    // // Phonon coupling strengths
    // let phonon_coupling = vec![0.005, 0.005];
    
    // // Create polaron-polariton system
    // let polaron_polariton = PolaronPolariton::new(
    //     phonon_modes, light_matter, phonon_coupling);
    
    // // Calculate polaron-polariton eigenvalues at Γ point
    // let start_time = Instant::now();
    // let (eigvals, _) = polaron_polariton.diagonalize_pp_ham_sparse(20);
    // let end_time = Instant::now();
    
    // // Print results
    // println!("Polaron-polariton eigenvalues at Γ point:");
    // println!("{:?}", eigvals);
    // println!("Calculation time: {:?}", end_time.duration_since(start_time));
}
