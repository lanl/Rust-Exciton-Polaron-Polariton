// Import statements for the various libraries used in this code
use exciton_2d::{utils::fast_matrix_element_calc, FullHamiltonian};
use ndarray_linalg::{Eigh};
use std::{collections::HashMap, f64::consts::PI, usize};
use ndarray::{arr1, linalg::kron, Array1, Array2};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use crate::utils::{PhotonMode, save_band_data};

/// Stores matrix elements for light-matter interaction calculations
pub struct LMMatrixElement {
    pub   cos_mat: HashMap<(usize,usize, Vec<i32>), f64>, // Cosine matrix elements
    pub p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64>, // Momentum-sine matrix elements
}

impl LMMatrixElement {
    /// Populates matrix elements for light-matter interaction
    /// 
    /// # Arguments
    /// * `rel_vecs` - Relative coordinate eigenvectors
    /// * `modes` - Photon modes
    /// * `exciton` - Exciton Hamiltonian
    /// 
    /// # Returns
    /// * New LMMatrixElement with populated matrix elements
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
                        // Using only one direction since cos(q) = cos(-q)
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

    /// Creates a new empty LMMatrixElement
    pub fn new() -> Self{
        let   cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        Self { cos_mat: cos_mat, p_sin_mat: p_sin_mat }
    }

    /// Calculates cosine matrix elements between two exciton states
    pub fn get_cos_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let eigvecs = &rel_vecs;
        let _test = eigvecs.shape();
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

    /// Calculates momentum-sine matrix elements between two exciton states
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

    /// Computes the momentum operator in the sinc-DVR basis
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
    /// Creates a new photonic Hamiltonian with the given modes
    pub fn new(photon_modes: Vec<PhotonMode>) -> Self {
        Self { photon_modes: photon_modes }
    }

    /// Creates a single-mode photonic Hamiltonian matrix
    fn _get_h_ph_sing(&self, omega:f64, n_fock: usize) -> Array2<f64> {
        let n_max = (n_fock - 1) as f64;
        // Create diagonal matrix with energies 0, ω, 2ω, ..., (n_max)ω
        let diags = Array1::linspace(0.0, omega * n_max, n_fock);
        return Array2::from_diag(&diags);
    }

    /// Creates a full photonic Hamiltonian matrix for a single mode, tensored with identity for other modes
    fn get_h_ph_sing_full(&self, mode_i: usize) -> Array2<f64>{
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // For the selected mode, create energy diagonal
                let n_max = (mode.n_fock - 1) as f64;
                let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);
                Array2::from_diag(&diags)
            }
            else {
                // For other modes, use identity matrix
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    /// Gets the complete photonic Hamiltonian
    pub fn get_h_ph(&self) -> Array2<f64> {
        let photon_modes = &self.photon_modes;

        // Handle empty vector case
        if photon_modes.is_empty() {
            return Array2::eye(1);
        }

        // Sum Hamiltonians for each mode
        photon_modes.iter().enumerate().map(|(mode_i, _)| self.get_h_ph_sing_full(mode_i))
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
    /// Creates a new light-matter interaction Hamiltonian
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

    /// Constructs the full light-matter interaction Hamiltonian
    pub fn get_int_h(&self) -> Array2<f64> {
        // Calculate matrix elements for the interaction
        let matrix_elements = LMMatrixElement::new()
            .populate_mat_elements(&self.rel_vecs, &self.photon_modes, &self.exciton);

        let kappas = &self.exciton.kappa_vectors;
        let k = &self.exciton.k;

        // Calculate dimensions of the Hilbert space
        let n_ph = self.photon_modes[0].n_fock.pow(self.photon_modes.len() as u32);
        let n_ex = self.exciton_vecs.len();
        
        // Initialize interaction Hamiltonian
        let mut h_int = Array2::<f64>::zeros((n_ex * n_ph, n_ex * n_ph));
        
        // Iterate over all photon modes
        self.photon_modes.iter().enumerate().for_each(|(mode_i,mode1)|{
            let m_com = self.exciton.m;
            let q = &mode1.q;

            // Get photon number operator for this mode
            let aa_dag1 = self.get_h_ph_sing_full(mode_i);
            let p_dot_q = self.get_p_com(kappas, k, &mode1.q);
            h_int = &h_int + kron(&p_dot_q, &aa_dag1);

            
            // Calculate P·A term (linear coupling)
            let p_dot_eps = self.get_p_com(kappas, k, &mode1.eps);
            let cos_x = &matrix_elements.cos_mat;

            let q_ind = vec![mode_i as i32];

            // -P·A cos(q·r) (a† + a) term
            let p_cos = - 2.0 * self.coupling / m_com * self.get_matrix_from_hash(&q_ind, &cos_x).dot(&p_dot_eps);
            let a1 = self.get_a_sing_full(mode_i);
            h_int = &h_int + &kron(&p_cos, &(&a1 + &a1.t()));

            // -∇·A sin(q·r) (a† - a) term
            let p_sin_x = &matrix_elements.p_sin_mat;
            let p_sin = self.get_matrix_from_hash(&q_ind, &p_sin_x) * self.coupling * -2.0 / self.exciton.rel_ham.mu;
            h_int = &h_int + &kron(&p_sin, &(&a1.t() - &a1));

            // Iterate over all photon modes again for quadratic terms
            self.photon_modes.iter().enumerate().for_each(|(mode_j,mode2)| {
                let q_2 = &mode2.q;
                let qq = q.dot(q_2);

                // q_i·q_j a_i†a_i a_j†a_j term (A²)
                let ex_eye: Array2<f64> = Array2::eye(n_ex);
                let aa_2 = &aa_dag1.dot(&self.get_h_ph_sing_full(mode_j)) * qq / 2.0 / m_com;
                h_int = &h_int + &kron(&ex_eye, &aa_2);

                // Diamagnetic terms (A²)
                let a2 = self.get_a_sing_full(mode_j);
                let q_ind1 = mode_i.max(mode_j) as i32;
                let q_ind2 = mode_i.min(mode_j) as i32;
                
                // cos((q_i + q_j)·r) (a_i a_j + a_i† a_j†) term
                let cos_sum = self.get_matrix_from_hash(&vec![q_ind1, q_ind2], &cos_x) * self.coupling.powi(2);
                h_int = &h_int + kron(&cos_sum, &(&a1.dot(&a2) + &a1.t().dot(&a2.t())));

                // cos((q_i - q_j)·r) (a_i† a_j + a_i a_j†) term
                let cos_diff = self.get_matrix_from_hash(&vec![q_ind1, -q_ind2], &cos_x) * self.coupling.powi(2);
                h_int = &h_int + kron(&cos_diff, &(&a1.t().dot(&a2) + &a1.dot(&a2.t())));
            });
        });
        
        let _sum = h_int.sum();

        h_int
    }

    /// Creates the annihilation operator for a single mode, tensored with identity for other modes
    fn get_a_sing_full(&self, mode_i: usize)  -> Array2<f64> {
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // Create annihilation operator for this mode
                let dim = mode.n_fock;
                let mut a = Array2::<f64>::zeros((dim, dim));
                for n in 1..dim {
                    a[[n-1,n]] = (n as f64).sqrt();
                }
                a
            }
            else {
                // Identity for other modes
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    /// Creates the photon number operator for a single mode, tensored with identity for other modes
    fn get_h_ph_sing_full(&self, mode_i: usize) -> Array2<f64>{
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // Create number operator (diagonal with 0,1,2,...,n_max)
                let n_max = (mode.n_fock - 1) as f64;
                let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);
                Array2::from_diag(&diags)
            }
            else {
                // Identity for other modes
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    /// Constructs a matrix from matrix elements stored in a hash map
    pub fn get_matrix_from_hash(&self, q_inds: &Vec<i32>, mat_hash: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64> {
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;

        // Build matrix using functional operations for relative coordinate operators
        // Note that these are all diagonal in the CoM picture
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
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

            mat_ij
        })
    }

    /// Alternative implementation of get_matrix_from_hash (not used in current code)
    fn _get_cos_x(&self, q_inds: &Vec<i32>, cos_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();

        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];
            
            // Compute p_ij using iterator operations
            let cos_ij: f64 = vec1.iter().enumerate()
                .flat_map(|(i1, &a1)| {
                    let com_ind1 = i1 % n_com;
                    let rel_ind1 = i1 / n_com;
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 % n_com;
                            let rel_ind2 = i2 / n_com;
                            if com_ind1 == com_ind2 {
                                                                Some(a1 * a2 * cos_mat.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                    .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                    i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds)))
                            } else {
                                None
                            }
                        })
                })
                .sum();

            cos_ij
        })
    }

    /// Alternative implementation of get_matrix_from_hash (not used in current code)
    fn _get_p_sin_x(&self, q_inds: &Vec<i32>, p_sin_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();

        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];
            
            // Compute p_ij using iterator operations
            let sin_ij: f64 = vec1.iter().enumerate()
                .flat_map(|(i1, &a1)| {
                    let com_ind1 = i1 % n_com;
                    let rel_ind1 = i1 / n_com;
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 % n_com;
                            let rel_ind2 = i2 / n_com;
                            if com_ind1 == com_ind2 {
                                Some(a1 * a2 * p_sin_mat.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                    .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                    i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds)))
                            } else {
                                None
                            }
                        })
                })
                .sum();

            sin_ij
        })
    }

    /// Calculates the center-of-mass momentum matrix element P·field_vec
    pub fn get_p_com(&self, kappas: &Vec<Vec<f64>>, k: &Vec<f64>, field_vec: &Array1<f64>) -> Array2<f64> {
        let e_vecs = &self.exciton_vecs;
        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;
        
        // Convert k to Array1 once for efficiency
        let k_arr = arr1(&k);
        
        // Precompute p_dot_unit for each component
        let p_dot_unit_cache: Vec<f64> = kappas.iter()
            .map(|kappa| {
                let _test = kappa.clone();
                let _test1 = [field_vec[0], field_vec[1]];
                let _test2 = [k_arr[0], k_arr[1]];
                // Calculate (k + kappa)·field_vec for each kappa point
                (k_arr.clone() + arr1(kappa)).dot(field_vec)
            })
            .collect();
        
        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];

            // Compute P_ij using iterator operations
            // This is a purely diagonal operator
            (0 .. n_com).flat_map(|com_ind| {
                let p = p_dot_unit_cache[com_ind];
                
                (0 .. n_rel).map(move |rel_ind| {
                    let i = n_com * rel_ind + com_ind;
                    let a1 = vec1[i];
                    let a2 = vec2[i];
                    a1 * a2 * p
                })
            }).sum()
        })
    }

    /// Alternative implementation for calculating a²a² terms (not used in current code)
    fn _get_aa_2(&self, mode_i: usize, mode_j: usize, mode1: &PhotonMode, mode2: &PhotonMode) -> Array2<f64>{
        let q2 = mode1.q.dot(&mode2.q);

        if mode_i == mode_j {
            // For same mode: create n² operator
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i {
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);
                    Array2::from_diag(&(&diags * &diags))
                }
                else {
                    Array2::eye(mode.n_fock)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
        else {
            // For different modes: create n_i × n_j operator
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i || idx == mode_j{
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);
                    Array2::from_diag(&diags)
                }
                else {
                     Array2::eye(mode.n_fock)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
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

    /// Builds the complete light-matter Hamiltonian
    pub fn build_lm_hamiltonian(&self) -> Array2<f64>{
        // Get exciton Hamiltonian (diagonal matrix of energies)
        let h_ex = Array2::from_diag(&self.exciton_energies);
        
        // Get photonic Hamiltonian
        let h_ph = self.photon.get_h_ph();
        
        // Get interaction Hamiltonian
        let h_int = self.interaction.get_int_h();

        // Combine all parts: H_ex ⊗ I_ph + I_ex ⊗ H_ph + H_int
        kron(&h_ex, &Array2::eye(h_ph.shape()[0]))
        + kron(&Array2::eye(h_ex.shape()[0]), &h_ph)
        + h_int
    }

    /// Diagonalizes the light-matter Hamiltonian
    pub fn diagonalize_lm_ham(&self) -> (Array1<f64> , Array2<f64>){
        // Build the full Hamiltonian
        let h_lm = self.build_lm_hamiltonian();

        // Diagonalize using LAPACK (upper triangle is used)
        h_lm.eigh(ndarray_linalg::UPLO::Upper).unwrap()
    }

    /// Creates a new LightMatter object with updated k-vector
    pub fn update_k(&self, k: Vec<f64>) -> Self {
        // Clone the exciton Hamiltonian
        let mut exciton = self.interaction.exciton.clone();
        
        // Update the k-vector
        exciton.k = k;

        // Create a new LightMatter object with the updated exciton
        LightMatter::new( exciton, self.photon.photon_modes.clone(), self.interaction.coupling,self.n_rel,
            self.n_ex_eig, self.parity)
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

    /// Calculates eigenvalues and eigenvectors at a specific k-point
    pub fn calculate_at_k_point_lm(&self, k: Vec<f64>) -> (Array1<f64> , Array2<f64>) {
        // Update the light-matter system with the new k-point
        let light_matter = self.light_matter.update_k(k);

        // Diagonalize the Hamiltonian at this k-point
        light_matter.diagonalize_lm_ham()
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
    
    /// Calculates the full band structure along a path in k-space
    pub fn calculate_lm_band_structure(&self, n_points: usize) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        // Generate filename for saving results
        let filename = self.get_lm_band_structure_filename(n_points);

        // Calculate points per segment of the path
        let points_per_seg = n_points / self.sym_points.len();

        // Generate k-points along the path
        let k_points: Vec<Vec<f64>> = self.sym_points.iter().enumerate().flat_map( | (sym_ind,sym_point) | {
            let sym_ind2 = (sym_ind + 1)%&self.sym_points.len();
            (0 .. points_per_seg).map(move | t | {
                // Interpolate between symmetry points
                let t_frac = t as f64  / (points_per_seg - 1) as f64;  // Changed denominator to include endpoints
                let x = sym_point[0] + t_frac * (&self.sym_points[sym_ind2][0] - sym_point[0]);
                let y = sym_point[1] + t_frac * (&self.sym_points[sym_ind2][1] - sym_point[1]);

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
            let (eigvals, _) =self.calculate_at_k_point_lm(k.clone());
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

    /// Builds the complete polaron-polariton Hamiltonian
    pub fn build_pp_hamiltonian(&self) -> Array2<f64>{
        // Get the light-matter Hamiltonian
        let h_lm = self.polariton.build_lm_hamiltonian();
        
        // Get the phonon Hamiltonian
        let h_pn = self.get_h_pn();
        
        // Get the interaction Hamiltonian
        let h_pp_int = self.pp_int.get_pp_int_h();

        // Create identity matrices for tensor products
        let i_lm = Array2::eye(h_lm.shape()[0]);
        let i_pn = Array2::eye(h_pn.shape()[0]);

        // Combine all parts: H_lm ⊗ I_pn + I_lm ⊗ H_pn + H_pp_int
        kron(&h_lm, &i_pn) + kron(&i_lm, &h_pn) + h_pp_int
    }

    /// Gets the phonon Hamiltonian
    pub fn get_h_pn(&self) -> Array2<f64> {
        // Reuse PhotonicHamiltonian for phonons (same structure)
        let phonon = PhotonicHamiltonian::new(self.phonon_modes.clone());
        phonon.get_h_ph()
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

    /// Constructs the polaron-polariton interaction Hamiltonian
    pub fn get_pp_int_h(&self) -> Array2<f64>{
        // Calculate matrix elements
        let cos_x = self.get_pp_matrix_elements();

        // Get dimensions
        let n_ex = self.polariton_int.exciton_vecs.len();
        let n_ph = self.polariton_int.photon_modes[0].n_fock.pow(self.polariton_int.photon_modes.len() as u32);
        let n_pn = self.phonon_modes[0].n_fock.pow(self.phonon_modes.len() as u32);

        // Create identity matrix for photons
        let i_ph = Array2::eye(n_ph);

        // Initialize interaction Hamiltonian
        let mut h_int = Array2::<f64>::zeros((n_ex*n_ph*n_pn, n_ex*n_ph*n_pn));

        // Iterate over all phonon modes
        self.phonon_modes.iter().enumerate().for_each(|(mode_i, _)| {
            // 2 gamma cos(k·r) (b† + b) term
            let coupling = self.coupling[mode_i];
            let k_ind = vec![mode_i as i32];
            
            // Get matrix elements
            let g_cos = coupling * 2.0 * self.polariton_int.get_matrix_from_hash(&k_ind, &cos_x);
            
            // Get phonon operator
            let b = self.get_b_sing_full(mode_i);

            // Add to interaction Hamiltonian
            h_int = &h_int + &kron(&kron(&g_cos, &i_ph), &(&b.t() + &b))
        });

        h_int
    }

    /// Creates the phonon annihilation operator for a single mode
    pub fn get_b_sing_full(&self, mode_i:usize) -> Array2<f64> {
        self.phonon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // Create annihilation operator for this mode
                let dim = mode.n_fock;
                let mut a = Array2::<f64>::zeros((dim, dim));
                for n in 1..dim {
                    a[[n-1,n]] = (n as f64).sqrt();
                }
                a
            }
            else {
                // Identity for other modes
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
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

// /// Main function demonstrating the use of the light-matter coupling framework
// fn main() {
//     // Create exciton configuration
//     let e_config = ExcitonConfig {
//         n_kappa: 11,                           // Number of k-points in Brillouin zone
//         a0: 9.0,                               // Lattice constant
//         v0: 0.01,                              // Potential strength
//         k: vec![0.0, 0.0],                     // Initial k-vector
//         m_e: 0.3,                              // Electron effective mass
//         m_h: 0.3,                              // Hole effective mass
//         verbose: false,                        // Verbose output flag
//         grid_size: 128,                        // Grid size for real-space calculations
//         box_size: 3.0,                         // Real-space box size
//         save_dir: PathBuf::from("saved_data_lm"), // Directory to save data
//         recalculate: false,                    // Whether to recalculate existing data
//         n_jobs: 2,                             // Number of parallel jobs
//         use_gpu: false,                        // Whether to use GPU acceleration
//         batch_size: 32,                        // Batch size for calculations
//     };

//     // Create the 2D exciton system
//     let a0 = (&e_config).a0;
//     let v0 = (&e_config).v0;
//     let exciton = Exciton2DGridSystem::from_config(e_config);
    
//     // Parameters for diagonalization
//     let n_rel = 16;      // Number of relative coordinate states
//     let n_ex_eig = 20;  // Number of exciton eigenstates
//     let parity = 0;      // Parity of states to consider
    
//     // Light-matter coupling parameters
//     let coupling = 0.03;  // Coupling strength
//     let omega_0 = 0.25;    // Photon frequency
    
//     // Define photon modes
//     let qx_values = vec![0.0,0.01,0.02,0.03, 0.04,0.05,0.06,0.07];
//     let qz = omega_0 / C;
//     let n_fock = 2;      // Dimension of Fock space
    
//     // Create photon modes
//     let photon_modes: Vec<PhotonMode> = qx_values.iter().map(|qx| {
//             PhotonMode::new(Vector3D { x: *qx, y: 0.0, z: qz }, Polarization::TE, n_fock)
//         }).collect();
    
//     // Create light-matter system
//     let light_matter = LightMatter::new(exciton.full_ham, photon_modes, coupling,
//                                     n_rel, n_ex_eig, parity);

//     // Define high-symmetry points in k-space for band structure calculation
//     // Here we use Γ to X path (center to edge of Brillouin zone)
//     let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0]];
//     let sym_labels = vec!["Γ".to_string(), "X".to_string()];

//     // Start timing the band structure calculation
//     let start_time = Instant::now();
    
//     // Create band structure calculator
//     let band_structure = BandStructureCalculatorLM::new(
//         light_matter, a0, v0, "lm_data".to_string(), sym_points.clone());
    
//     // Calculate band structure with 60 points along the path
//     let n_k_points = 6;  // Number of k-points along the path
//     let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure(n_k_points);
    
//     // End timing and print duration
//     let end_time = Instant::now();
//     let duration = end_time.duration_since(start_time);
//     println!("Calc dispersion in {:?}", duration);

//     // Plot the band structure, showing 40 bands
//     let n_bands = 40;
//     plot_band_structure(
//         &k_points, sym_points, sym_labels, &eigvals_list, n_bands, a0, 
//         Some("disp_plot.png"), false, v0, coupling);

//     // The following code is commented out but shows examples of building
//     // and timing individual components of the Hamiltonian
    
//     // // Build and time exciton Hamiltonian
//     // let start_time = Instant::now();
//     // let (sorted_eigvals, _, _, _, _) 
//     //     = exciton.diagonalize_full_hamiltonian(n_rel, n_ex_eig, parity);
//     // let h_ex = Array2::from_diag(&sorted_eigvals);
//     // println!("Exciton Hamiltonian of size: {:?}", h_ex.shape());
//     // let end_time = Instant::now();
//     // let duration = end_time.duration_since(start_time);
//     // println!("Built Exciton Hamiltonian in {:?}", duration);
    
//     // // Build and time photonic Hamiltonian
//     // let start_time = Instant::now();
//     // let h_ph = light_matter.photon.get_h_ph();
//     // println!("Photonic Hamiltonian of size: {:?}", h_ph.shape());
//     // let end_time = Instant::now();
//     // let duration = end_time.duration_since(start_time);
//     // println!("Built Photonic Hamiltonian in {:?}", duration);

//     // // Build and time interaction Hamiltonian
//     // let start_time = Instant::now();
//     // let h_int = light_matter.interaction.get_int_h();
//     // println!("Interaction Hamiltonian of size: {:?}", h_int.shape());
//     // let end_time = Instant::now();
//     // let duration = end_time.duration_since(start_time);
//     // println!("Built Interaction Hamiltonian in {:?}", duration);

//     // // Build and time full Hamiltonian
//     // let start_time = Instant::now();
//     // let h = light_matter.build_lm_hamiltonian();
//     // println!("Full Hamiltonian of size: {:?}", h.shape());
//     // let end_time = Instant::now();
//     // let duration = end_time.duration_since(start_time);
//     // println!("Built Full Hamiltonian in {:?}", duration);
// }

