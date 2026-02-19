// Import statements for the various libraries used in this code
use ndarray_linalg::{Eigh};
use ndarray::{Array1, Array2};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray_linalg::c64;
use crate::hamiltonians::{Boson, LMHamiltonian, PPHamiltonian};

pub fn diagonalize_lm_ham_dense<B: Boson<Array2<c64>>>(light_matter: &mut LMHamiltonian<Array2<c64>, B>) 
    -> (Array1<f64> , Array2<c64>){
    let h_lm = light_matter.build_lm_hamiltonian(None);

    h_lm.eigh(ndarray_linalg::UPLO::Upper).unwrap()
}

pub fn diagonalize_pp_ham_dense<B: Boson<Array2<c64>>>(polaron_polariton: &mut PPHamiltonian<Array2<c64>, B>)
    -> (Array1<f64> , Array2<c64>){
    let h_pp = polaron_polariton.build_pp_hamiltonian(None);

    h_pp.eigh(ndarray_linalg::UPLO::Upper).unwrap()
}


/// Calculator for band structure of light-matter coupled systems
pub struct BandStructureCalculatorLM<B: Boson<Array2<c64>>> {
    pub light_matter: LMHamiltonian<Array2<c64>, B>,      // Light-matter system
    pub a0: f64,                        // Lattice constant
    pub v0: f64,                        // Potential strength
    pub save_dir: String,               // Directory to save results
    pub sym_points: Vec<Vec<f64>>       // High-symmetry points in k-space
}

impl <B: Boson<Array2<c64>>> BandStructureCalculatorLM<B> {
    /// Creates a new band structure calculator
    pub fn new(light_matter: LMHamiltonian<Array2<c64>, B>, a0: f64, v0: f64, save_dir: String, sym_points: Vec<Vec<f64>>) -> Self {
        Self { light_matter: light_matter, a0: a0, v0: v0, save_dir: save_dir, sym_points: sym_points}
    }

    /// Calculates eigenvalues and eigenvectors at a specific k-point
    pub fn calculate_at_k_point_lm(&mut self, k: Vec<f64>) -> (Array1<f64> , Array2<c64>) {
        // Update the light-matter system with the new k-point
        self.light_matter.update_k(k);

        // Diagonalize the Hamiltonian at this k-point
        diagonalize_lm_ham_dense(&mut self.light_matter)
    }

    /// Generates a filename for the band structure data
    pub fn get_lm_band_structure_filename(
        &self,
        n_points: usize,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}_coup{}_nmodes{}.bin",
            self.save_dir,
            self.a0,
            self.v0,
            self.light_matter.exciton.exciton_ham.rel_ham.grid.grid_size,
            self.light_matter.exciton.exciton_ham.rel_ham.grid.box_size,
            self.light_matter.exciton.n_rel,
            self.light_matter.exciton.exciton_ham.kappa_vectors.len(),
            n_points,
            self.light_matter.coupling,
            self.light_matter.photon.boson.get_modes().len(),
        )
    }
    
    /// Calculates the full band structure along a path in k-space
    pub fn calculate_lm_band_structure(&mut self, n_points: usize) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
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
            let (eigvals, _) =self.calculate_at_k_point_lm(k.clone());
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

