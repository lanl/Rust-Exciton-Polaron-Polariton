use exciton_2d:: FullHamiltonian;
use std::{collections::HashMap, usize};
use ndarray::{arr1, Array1, Array2};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};

use sprs::{kronecker_product as kron, smmp::mul_csr_csr, CsMat, CsMatBase, TriMat};
use arpack::{Config, Operator}; 

use crate::utils::{PhotonMode, LMMatrixElement};//, save_band_data};




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
            
            // -∇·A sin(q·r) (a† - a) term
            let p_sin_factor = -self.coupling * 2.0 / self.exciton.rel_ham.mu;
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

        // CsMatBase::csr_from_dense(Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
        //     // For each pair of eigenvectors
        //     let vec1 = &e_vecs[idx1];
        //     let vec2 = &e_vecs[idx2];

        //     // Calculate matrix element in the full basis
        //     let mat_ij: f64 = (0..n_rel).flat_map(|rel_ind1| {
        //             (0 .. n_rel).flat_map(move |rel_ind2| {
        //                 // Get the matrix element for these relative states
        //                 let mat_element = mat_hash.get(&(rel_ind1,rel_ind2,q_inds.clone()))
        //                             .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
        //                             i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds));
                        
        //                 // Sum over center-of-mass states (diagonal in CoM)
        //                 (0..n_com).map(move |com_ind| {
        //                     let i1 = n_com * rel_ind1 + com_ind;
        //                     let i2 = n_com * rel_ind2 + com_ind;

        //                     let a1 = vec1[i1];
        //                     let a2 = vec2[i2];

        //                     a1 * a2 * mat_element
        //                 })
        //             })
        //         }).sum();

        //     mat_ij
        // }).view(), 1e-10)
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

        // println!("Sparse Hamiltonian of density {}% built.", h_lm.density() * 100.0);
        let (n, _) = h_lm.shape();
        
        // Create ARPACK operator
        let op = ArpackOperator::new(h_lm);
        
        // Configure ARPACK
        let config = Config::new()
            .which(arpack::Which::SA)  // Smallest algebraic eigenvalues
            .tol(1e-10)                // Tolerance
            .max_iter(1000);          // Maximum iterations
        
        // Solve eigenvalue problem
        let result = arpack::eigs(&op, n,n_eigenvalues, config).unwrap();
        
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

        // // Save the band structure data
        // save_band_data(filename, (k_points.clone(), eigvals_list.clone())).unwrap();

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
            .which(arpack::Which::SR)  // Smallest real-part eigenvalues
            .tol(1e-10)                // Tolerance
            .max_iter(1000);          // Maximum iterations
        
        // Solve eigenvalue problem
        let result = arpack::eigs(&op, n, n_eigenvalues, config).unwrap();
        
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

