// Import statements for the various libraries used in this code
use ndarray_linalg::{Eigh};
use ndarray::{linalg::kron, Array1, Array2};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use crate::sing_ex::ExcitonicHamiltonian;
use crate::utils::{BosonMode};//, save_band_data};
use crate::matrix_elements::LMMatrixElementArray;

/// Represents the photonic part of the Hamiltonian
#[derive(Clone)]
pub struct BosonicHamiltonian {
    pub bosonic_modes: Vec<BosonMode>,
}

impl BosonicHamiltonian {
    /// Creates a new photonic Hamiltonian with the given modes
    pub fn new(bosonic_modes: Vec<BosonMode>) -> Self {
        Self { bosonic_modes: bosonic_modes }
    }

    /// Creates annihilation operator
    pub fn get_a(&self, mode_i: usize) -> Array2<f64> {
        self.bosonic_modes.iter().enumerate().map(|(idx,mode)| {
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

    pub fn get_adag_a(&self, mode_i: usize) -> Array2<f64> {
        let a = self.get_a(mode_i);
        a.t().dot(&a)
    }

    /// Gets the complete photonic Hamiltonian
    pub fn get_h(&self) -> Array2<f64> {
        let bosonic_modes = &self.bosonic_modes;

        // Handle empty vector case
        if bosonic_modes.is_empty() {
            return Array2::eye(1);
        }

        // Sum Hamiltonians for each mode
        bosonic_modes.iter().enumerate().map(|(mode_i, mode)| {
            mode.omega * self.get_adag_a(mode_i)
            })
            .reduce(|acc, h_ph_sing| &acc + &h_ph_sing)
            .unwrap()
    }

    pub fn get_all_q_ind(&self) -> Vec<Vec<i32>> {
        let mut all_q_ind: Vec<Vec<i32>> = Vec::new();
        let n_ph = self.bosonic_modes.len();

        // Single photon modes
        for i in 0..n_ph {
            all_q_ind.push(vec![i as i32]);
        }

        // Pairs of bosonic_modes modes for A^2 terms
        for i in 0..n_ph {
            for j in i..n_ph {
                if i != j {
                    all_q_ind.push(vec![i as i32, j as i32]);
                }
            }
        }

        all_q_ind
    }
}

pub struct LightMatter {
    exciton: ExcitonicHamiltonian,
    photon: BosonicHamiltonian,
    matrix_elements: LMMatrixElementArray,
    coupling: f64,
    p_c_sin: Option<Vec<Array2<f64>>>,
    p_q: Option<Vec<Array2<f64>>>,
}

impl LightMatter {
    pub fn new( exciton: ExcitonicHamiltonian, 
                photon: BosonicHamiltonian, 
                coupling: f64) -> Self {

        let matrix_elements = LMMatrixElementArray::new();

        Self { exciton, photon, matrix_elements, coupling, p_c_sin: None, p_q: None }
    }

    fn initialize(&mut self) {

        self.exciton.compute_eigenstates();

        let all_q_ind = self.photon.get_all_q_ind();
        self.matrix_elements = self.matrix_elements.populate_tay_mat_elements(
            &all_q_ind, 
            self.exciton.e_vecs.as_ref().unwrap(), 
            self.exciton.parities.as_ref().unwrap(),
            &self.photon.bosonic_modes, 
            self.exciton.exciton_ham.kappa_vectors.len(),
            self.exciton.exciton_ham.rel_ham.mu
        );

        self.calc_p_c_sin_matrix();
    }

    fn calc_p_c_sin_matrix(&mut self) {
        
        let n_ph = self.photon.bosonic_modes.len() + 1; // +1 for ground state

        let sin_mat = &self.matrix_elements.sin_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // timed_println!("{:?}",test);
        
        let mut p_c_sin: Vec<Array2<f64>> = Vec::new();
        let mut p_q: Vec<Array2<f64>> = Vec::new();
        (0 .. n_ph-1).for_each(|i| {
            let eps = &self.photon.bosonic_modes[i].eps;
            let q = &self.photon.bosonic_modes[i].q;
            let q_ind_key = i as i32;

            p_c_sin.push( 
                    self.exciton.get_p_com(eps, false)// P·eps 
                    .dot(sin_mat.get(&vec![q_ind_key]).unwrap()) // cos(qx/2) matrix element
                );
            
            p_q.push( 
                    self.exciton.get_p_com(q, false)// P·q 
                );

        });

        self.p_c_sin = Some(p_c_sin);
        self.p_q = Some(p_q);
    }

    pub fn build_lm_hamiltonian(&mut self) -> Array2<f64>{

        // Initialize
        self.initialize();
        
        // Get matrix dimensions
        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let n_ph: usize = self.photon.bosonic_modes
            .iter().map(|mode| mode.n_fock).product();
        let mut ham: Array2<f64> = Array2::zeros([n_ex*n_ph, n_ex*n_ph]);
        let ex_eye: Array2<f64> = Array2::eye(n_ex);
        let ph_eye: Array2<f64> = Array2::eye(n_ph);

        // Exciton Hamiltonian
        ham += &kron(&Array2::from_diag(&self.exciton.e_vals.as_ref().unwrap()), &ph_eye);

        // Photonic Hamiltonian
        ham += &kron(&ex_eye, &self.photon.get_h());

        // Interaction Hamiltonian
        self.get_interaction_hamiltonian(&mut ham);

        ham
    }

    fn get_interaction_hamiltonian(&self, ham: &mut Array2<f64>) {
        // Iterate over photon modes to build interaction terms
        self.photon.bosonic_modes.iter().enumerate().for_each(|(i_ph, mode_i)| {

            let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
            let ex_eye: Array2<f64> = Array2::eye(n_ex);

            // p.A terms
            let a_i = &self.photon.get_a(i_ph);
            let p_c_sin = &(&self.p_c_sin.as_ref().unwrap()[i_ph] * 2.0 / self.exciton.exciton_ham.m * self.coupling);
            let p_q = &(&self.p_q.as_ref().unwrap()[i_ph] / 2.0 / self.exciton.exciton_ham.m);

            let p_cos = self.matrix_elements.p_cos_mat.get(&vec![i_ph as i32]).unwrap() * self.coupling;
            
            // P•A sin(x) term
            *ham += &kron(p_c_sin, &(a_i + &a_i.t()));

            // p•A cos(x) term
            *ham += &kron(&p_cos, &(-a_i + &a_i.t()));

            // P•q a†a term
            *ham += &kron(p_q, &(&a_i.t().dot(a_i)));

            // A² terms
            self.photon.bosonic_modes.iter().enumerate().for_each(|(j_ph, mode_j)| {
                let q_ind1 = (i_ph as i32).max(j_ph as i32);
                let q_ind2 = (i_ph as i32).min(j_ph as i32);
                let q_diff_key = vec![q_ind1, -q_ind2];
                let q_sum_key = vec![q_ind1, q_ind2];

                let a_j = &self.photon.get_a(j_ph);

                let coupling_2 = self.coupling * self.coupling * mode_i.eps.dot(&mode_j.eps) / 2. / self.exciton.exciton_ham.rel_ham.mu;

                let cos_diff = self.matrix_elements.cos_mat.get(&q_diff_key).unwrap();
                let cos_sum  = self.matrix_elements.cos_mat.get(&q_sum_key ).unwrap();

                let a_diff = &( a_i.dot(&a_j.t()) + a_i.t().dot(a_j) );
                let a_sum  = &( a_i.t().dot(&a_j.t()) + a_i.dot(a_j) );

                // cos(q + q') term
                *ham += &(coupling_2 * kron(cos_diff, a_diff));

                // cos(q - q') term
                *ham -= &(coupling_2 * kron(cos_sum, a_sum));

                // (q a†a)^2 term
                let q_q = mode_i.q.dot(&mode_j.q);
                *ham += &(q_q * kron(&ex_eye, &(a_i.t().dot(a_i).dot(&a_j.t()).dot(a_j)) ));
            });
        });
    }

    /// Diagonalizes the light-matter Hamiltonian
    pub fn diagonalize_lm_ham(&mut self) -> (Array1<f64> , Array2<f64>){
        // Build the full Hamiltonian
        let h_lm = self.build_lm_hamiltonian();

        // Diagonalize using LAPACK (upper triangle is used)
        h_lm.eigh(ndarray_linalg::UPLO::Upper).unwrap()
    }

    pub fn update_k(&mut self, k: Vec<f64>) {
        self.exciton.exciton_ham.k = k;
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
    pub fn calculate_at_k_point_lm(&mut self, k: Vec<f64>) -> (Array1<f64> , Array2<f64>) {
        // Update the light-matter system with the new k-point
        self.light_matter.update_k(k);

        // Diagonalize the Hamiltonian at this k-point
        self.light_matter.diagonalize_lm_ham()
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
            self.light_matter.photon.bosonic_modes.len(),
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


// /// Represents a polaron-polariton system (exciton-polariton coupled to phonons)
// pub struct PolaronPolariton {
//     pub phonon_modes: Vec<BosonMode>,    // Phonon modes (using PhotonMode structure)
//     pub polariton: LightMatter,           // Exciton-polariton system
//     pub coupling: Vec<f64>,               // Coupling strengths to phonon modes
//     pub pp_int: PPIntHamiltonian,         // Polaron-polariton interaction Hamiltonian
// }

// impl PolaronPolariton {
//     /// Creates a new polaron-polariton system
//     pub fn new(phonon_modes: Vec<BosonMode>, polariton: LightMatter, coupling: Vec<f64>) -> Self{
//         // Create the interaction Hamiltonian
//         let pp_int = PPIntHamiltonian::new(polariton.interaction.clone(), 
//                                            polariton.photon.bosonic_modes.clone(), 
//                                            coupling.clone());

//         Self { phonon_modes: phonon_modes, polariton: polariton, coupling: coupling, pp_int: pp_int}
//     }

//     /// Builds the complete polaron-polariton Hamiltonian
//     pub fn build_pp_hamiltonian(&self) -> Array2<f64>{
//         // Get the light-matter Hamiltonian
//         let h_lm = self.polariton.build_lm_hamiltonian();
        
//         // Get the phonon Hamiltonian
//         let h_pn = self.get_h_pn();
        
//         // Get the interaction Hamiltonian
//         let h_pp_int = self.pp_int.get_pp_int_h();

//         // Create identity matrices for tensor products
//         let i_lm = Array2::eye(h_lm.shape()[0]);
//         let i_pn = Array2::eye(h_pn.shape()[0]);

//         // Combine all parts: H_lm ⊗ I_pn + I_lm ⊗ H_pn + H_pp_int
//         kron(&h_lm, &i_pn) + kron(&i_lm, &h_pn) + h_pp_int
//     }

//     /// Gets the phonon Hamiltonian
//     pub fn get_h_pn(&self) -> Array2<f64> {
//         // Reuse BosonicHamiltonian for phonons (same structure)
//         let phonon = BosonicHamiltonian::new(self.phonon_modes.clone());
//         phonon.get_h()
//     }
// }

// /// Represents the polaron-polariton interaction Hamiltonian
// pub struct PPIntHamiltonian {
//     pub polariton_int: LMIntHamiltonian,  // Light-matter interaction Hamiltonian
//     pub phonon_modes: Vec<BosonMode>,    // Phonon modes
//     pub coupling: Vec<f64>,               // Coupling strengths
// }

// impl PPIntHamiltonian {
//     /// Creates a new polaron-polariton interaction Hamiltonian
//     pub fn new (polariton_int: LMIntHamiltonian,
//                 phonon_modes: Vec<BosonMode>,
//                 coupling: Vec<f64>, ) -> Self {
//         Self {  polariton_int: polariton_int,
//                 phonon_modes: phonon_modes,
//                 coupling: coupling}
//     }

//     /// Constructs the polaron-polariton interaction Hamiltonian
//     pub fn get_pp_int_h(&self) -> Array2<f64>{
//         // Calculate matrix elements
//         let cos_x = self.get_pp_matrix_elements();

//         // Get dimensions
//         let n_ex = self.polariton_int.exciton_vecs.len();
//         let n_ph = self.polariton_int.photon.bosonic_modes[0].n_fock.pow(self.polariton_int.photon.bosonic_modes.len() as u32);
//         let n_pn = self.phonon_modes[0].n_fock.pow(self.phonon_modes.len() as u32);

//         // Create identity matrix for photons
//         let i_ph = Array2::eye(n_ph);

//         // Initialize interaction Hamiltonian
//         let mut h_int = Array2::<f64>::zeros((n_ex*n_ph*n_pn, n_ex*n_ph*n_pn));

//         // Iterate over all phonon modes
//         self.phonon_modes.iter().enumerate().for_each(|(mode_i, _)| {
//             // 2 gamma cos(k·r) (b† + b) term
//             let coupling = self.coupling[mode_i];
//             let k_ind = vec![mode_i as i32];
            
//             // Get matrix elements
//             let g_cos = coupling * 2.0 * self.polariton_int.get_matrix_from_hash(&k_ind, &cos_x);
            
//             // Get phonon operator
//             let b = self.get_b_sing_full(mode_i);

//             // Add to interaction Hamiltonian
//             h_int = &h_int + &kron(&kron(&g_cos, &i_ph), &(&b.t() + &b))
//         });

//         h_int
//     }

//     /// Creates the phonon annihilation operator for a single mode
//     pub fn get_b_sing_full(&self, mode_i:usize) -> Array2<f64> {
//         self.phonon_modes.iter().enumerate().map(|(idx,mode)| {
//             if idx == mode_i {
//                 // Create annihilation operator for this mode
//                 let dim = mode.n_fock;
//                 let mut a = Array2::<f64>::zeros((dim, dim));
//                 for n in 1..dim {
//                     a[[n-1,n]] = (n as f64).sqrt();
//                 }
//                 a
//             }
//             else {
//                 // Identity for other modes
//                 Array2::eye(mode.n_fock)
//             }
//         }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
//         .unwrap()
//     }

//     /// Calculates matrix elements for polaron-polariton interaction
//     pub fn get_pp_matrix_elements(&self) -> HashMap<(usize,usize, Vec<i32>), f64>{
//         // Create matrix element calculator
//         let mat_calc = LMMatrixElement::new();

//         // Get relevant data
//         let rel_vecs = &self.polariton_int.rel_vecs;
//         let modes = &self.phonon_modes;
//         let exciton = &self.polariton_int.exciton;

//         let n_rel = rel_vecs.shape()[1];

//         // Define helper function to calculate cosine elements
//         let get_cos = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 { 
//             mat_calc.get_cos_elements(exciton, rel_vecs, i1, i2, q)
//         };

//         // Initialize matrix element storage
//         let mut cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

//         // Calculate matrix elements for all phonon modes
//         modes.iter().enumerate().for_each(|(k_ind,mode)| {
//             for i1 in 0 .. n_rel {
//                 for i2 in 0 .. n_rel {
//                     let k = &mode.q;

//                     // Calculate and store cosine matrix element
//                     cos_mat.entry((i1,i2,vec![k_ind as i32]))
//                         .or_insert_with(|| get_cos(i1, i2, k));
//                 }
//             }
//         });

//         cos_mat
//     }
// }
