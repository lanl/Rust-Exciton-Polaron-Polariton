use ndarray::{Array1, Array2};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};

use sprs::CsMat;
use arpack::{Config, Operator}; 

use crate::{hamiltonians::{LMHamiltonian, PPHamiltonian}, utils::SolverOperator};

use rust_slepc_bindings::EPSType;

pub fn diagonalize_sparse(
    matrix: CsMat<f64>, 
    n_eig: usize,
    block_size: usize, 
    solve_type: Option<EPSType>,
    initial_guess: Option<&Array2<f64>>) 
    -> (Array1<f64> , Array2<f64>){

    use crate::slepc::diagonalize_slepc;

    let op = CsMatOperator{matrix: matrix};
    
    diagonalize_slepc(&op, n_eig, block_size, solve_type, initial_guess)
}

fn _diagonalize_sparse_arpack(
    matrix: CsMat<f64>,
    n_eig: usize,) 
    -> (Array1<f64> , Array2<f64>){

    let n = matrix.shape().0;

    // Create SolverOperator
    let op = ArpackOperator::new(matrix);
    
    // Configure ARPACK
    let config = Config::new()
        .which(arpack::Which::SA)  // Smallest algebraic eigenvalues
        .tol(1e-10)                // Tolerance
        .max_iter(1000);          // Maximum iterations
    
    // Solve eigenvalue problem
    let result = arpack::eigs(&op, n,n_eig, config).unwrap();
    
    // Convert to ndarray format
    let eigenvalues = Array1::from_vec(result.values);
    
    // Convert eigenvectors to ndarray format
    let n = result.vectors.len() / n_eig;
    let mut eigenvectors = Array2::zeros((n, n_eig));
    
    for i in 0..n_eig {
        for j in 0..n {
            eigenvectors[[j, i]] = result.vectors[i * n + j];
        }
    }
    
    (eigenvalues, eigenvectors)
}

pub fn diagonalize_lm_ham_sparse(
    light_matter: &mut LMHamiltonian<CsMat<f64>>, 
    n_eigenvalues: usize,
    initial_guess: Option<&Array2<f64>>) 
    -> (Array1<f64> , Array2<f64>){

    let h_lm = light_matter.build_lm_hamiltonian();

    diagonalize_sparse(h_lm, n_eigenvalues, 128, None, initial_guess) 
}

pub fn diagonalize_pp_ham_sparse(
    polaron_polariton: &mut PPHamiltonian<CsMat<f64>>, 
    n_eigenvalues: usize,
    initial_guess: Option<&Array2<f64>>) 
    -> (Array1<f64> , Array2<f64>){
    let h_pp = polaron_polariton.build_pp_hamiltonian();
    
    diagonalize_sparse(h_pp, n_eigenvalues, 128, None, initial_guess) 
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

struct CsMatOperator {
    matrix: CsMat<f64>,
}

impl SolverOperator<f64> for CsMatOperator {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        // Apply matrix-vector product
        let mut result = vec![0.0; y.len()];
        sprs::prod::mul_acc_mat_vec_csr(self.matrix.view(), x, &mut result);
        y.copy_from_slice(&result);
    }

    fn get_diagonal(&self) -> Vec<f64> {
        let mut diags = vec![0.0; self.matrix.shape().0];

        self.matrix.diag().iter().for_each(|(i, val)| {
            diags[i] = *val;
        });

        diags
    }

    fn dimension(&self) -> usize {
        self.matrix.shape().0
    }
}

/// Calculator for band structure of light-matter coupled systems
pub struct BandStructureCalculatorLM {
    pub light_matter: LMHamiltonian<CsMat<f64>>,      // Light-matter system
    pub a0: f64,                        // Lattice constant
    pub v0: f64,                        // Potential strength
    pub save_dir: String,               // Directory to save results
    pub sym_points: Vec<Vec<f64>>       // High-symmetry points in k-space
}

impl BandStructureCalculatorLM {
    /// Creates a new band structure calculator
    pub fn new(light_matter: LMHamiltonian<CsMat<f64>>, a0: f64, v0: f64, save_dir: String, sym_points: Vec<Vec<f64>>) -> Self {
        Self { light_matter: light_matter, a0: a0, v0: v0, save_dir: save_dir, sym_points: sym_points}
    }

    /// Calculates eigenvalues and eigenvectors at a specific k-point using sparse matrices
    pub fn calculate_at_k_point_lm_sparse(&mut self, k: Vec<f64>, n_eigenvalues: usize) -> (Array1<f64>, Array2<f64>) {
        // Update the light-matter system with the new k-point
        self.light_matter.update_k(k);

        // Diagonalize the Hamiltonian at this k-point using ARPACK
        diagonalize_lm_ham_sparse(&mut self.light_matter, n_eigenvalues, None)
    }

    /// Generates a filename for the band structure data
    pub fn get_lm_band_structure_filename(
        &self,
        n_points: usize,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}_coup{}_nmodes{}_sparse.bin",
            self.save_dir,
            self.a0,
            self.v0,
            self.light_matter.exciton.exciton_ham.rel_ham.grid.grid_size,
            self.light_matter.exciton.exciton_ham.rel_ham.grid.box_size,
            self.light_matter.exciton.n_rel,
            self.light_matter.exciton.exciton_ham.kappa_vectors.len(),
            n_points,
            self.light_matter.coupling,
            self.light_matter.photon.bosonic_modes.len(),
        )
    }
    
    /// Calculates the full band structure along a path in k-space using sparse matrices
    pub fn calculate_lm_band_structure_sparse(&mut self, n_points: usize, n_eigenvalues: usize) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        // Generate filename for saving results
        let filename = self.get_lm_band_structure_filename(n_points);

        // Calculate points per segment of the path
        let points_per_seg = n_points / self.sym_points.len();

        // Generate k-points along the path
        let sym_points = &self.sym_points.clone();
        let k_points: Vec<Vec<f64>> = sym_points.iter().enumerate().flat_map( | (sym_ind,sym_point) | {
            let sym_ind2 = (sym_ind + 1)% &sym_points.len();
            (0 .. points_per_seg).map(move | t | {
                // Interpolate between symmetry points
                let t_frac = t as f64  / (points_per_seg - 1) as f64;  // Changed denominator to include endpoints
                let x = sym_point[0] + t_frac * (&sym_points[sym_ind2][0] - sym_point[0]);
                let y = sym_point[1] + t_frac * (&sym_points[sym_ind2][1] - sym_point[1]);

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