use exciton_2d::{Exciton2DGridSystem, FullHamiltonian};
use ndarray::{arr1, Array1, Array2, s};
use num_traits::{One, Zero};
use std::f64::consts::PI;
use std::sync::Arc;
use ndarray_linalg::c64;
use rayon::{prelude::*};
use indicatif::{ProgressBar, ProgressStyle};
use crate::timed_println;

use crate::utils::{ArrayOperator, BosonMode, DiagOperator, Operator, PPConfig, Polarization, PrettyArray2, SolverOperator, save_band_data};
use crate::matrix_elements::LMMatrixElementArray;

/// Represents the photonic part of the Hamiltonian
pub struct PhotonicHamiltonian {
    pub photon_modes: Vec<BosonMode>,
}

impl PhotonicHamiltonian {
    pub fn new(photon_modes: Vec<BosonMode>) -> Self {
        Self { photon_modes: photon_modes }
    }
}

impl ArrayOperator<f64> for PhotonicHamiltonian {
    fn apply(&self, x: &Array1<f64>) -> Vec<f64> {

        assert_eq!(x.len(), self.photon_modes.len() + 1, "Input vector length must match number of photon modes + 1.");

       x.iter().enumerate().map(|(i, &val)| {
            if i == 0 {
                0.0  // Ground state has zero photon energy
            } else {
                let ph_energy = self.photon_modes[i - 1].omega;
                ph_energy * val
            }
        }).collect::<Vec<f64>>()
    }
}

impl DiagOperator<f64> for PhotonicHamiltonian {
    fn form_diagonal(&self) -> Array1<f64> {
        // Diagonal elements are 0 (ground state) and then the photon mode energies
        (0 .. self.photon_modes.len() + 1).map(|i| {
            if i == 0 {
                0.0
            } else {
                self.photon_modes[i - 1].omega
            }
        }).collect::<Array1<f64>>()
    }
    
}

pub struct ExcitonicHamiltonian {
    pub exciton_ham: FullHamiltonian,
    pub n_ex_eig: usize,
    pub n_rel: usize,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Vec<Array1<num_complex::Complex64>>>,
    pub r_vecs: Option<Array2<f64>>,
    pub parities: Option<Array1<i32>>,
}

impl ExcitonicHamiltonian {
    pub fn new(exciton_ham: FullHamiltonian, n_ex_eig: usize, n_rel: usize) -> Self {
        Self { exciton_ham, n_ex_eig, n_rel, e_vals: None, e_vecs: None, r_vecs: None, parities: None}
    }

    pub fn is_empty(&self) -> bool{
        self.e_vals.is_none() || self.e_vals.is_none() || self.r_vecs.is_none() || self.parities.is_none()
    }

    pub fn compute_eigenstates(&mut self) {
        let (exciton_energies, e_vecs, _, r_vecs, parities) 
            = self.exciton_ham.diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig);
        
        self.e_vals = Some(exciton_energies.slice(s![..self.n_ex_eig]).to_owned());
        // timed_println!("Exciton energies: {:?}", self.e_vals);

        self.e_vecs = Some(e_vecs[0..self.n_ex_eig].to_vec());
        self.r_vecs = Some(r_vecs);
        self.parities = Some(parities.slice(s![..self.n_ex_eig]).to_owned());
    }

    /// Calculates the center-of-mass momentum matrix element P·field_vec
    /// 
    /// Note: Eigenvectors are complex, so we compute <ψ_i|P|ψ_j> = Σ ψ_i* P ψ_j
    pub fn get_p_com(&self, field_vec: &Array1<f64>, diag: bool) -> Array2<c64> {
        let e_vecs = self.e_vecs.as_ref().expect("Eigenvectors not computed. Call compute_eigenstates() first.");
        let kappas= &self.exciton_ham.kappa_vectors;
        let k = &self.exciton_ham.k;

        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        
        // Convert k to Array1 once for efficiency
        let k_arr = arr1(&k);
        
        // Precompute p_dot_unit for each component - avoid unnecessary cloning
        let p_dot_unit_cache: Vec<f64> = kappas.iter()
            .map(|kappa| {
                // Calculate (k + kappa)·field_vec for each kappa point more efficiently
                (k_arr[0] + kappa[0]) * field_vec[0] + (k_arr[1] + kappa[1]) * field_vec[1]
            })
            .collect();
            
        if diag {
            Array2::from_shape_fn((n_eig,1), |(idx1,_)| {
                // For each eigenvector
                let vec1 = &e_vecs[idx1];
                let n_rel = vec1.len() / n_com;

                // Compute P_ii using iterator operations
                // For complex eigenvectors: <ψ|P|ψ> = Σ |ψ|² P
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        a1.conj()* a1 * p  // |ψ|² for complex numbers
                    })
                }).sum()
            })
        }
        else {
            // Build matrix using functional operations
            Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
                // For each pair of eigenvectors
                let vec1 = &e_vecs[idx1];
                let vec2 = &e_vecs[idx2];

                let n_rel = vec1.len() / n_com;

                // Compute P_ij using iterator operations
                // For complex eigenvectors: <ψ_i|P|ψ_j> = Σ ψ_i* P ψ_j
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        let a2 = vec2[i];
                        (a1.conj() * a2) * p  
                    })
                }).sum()
            })
        }
    }
}

impl DiagOperator<f64> for ExcitonicHamiltonian {
    fn form_diagonal(&self) -> Array1<f64> {
        assert!(self.e_vals.is_some(), "Excitonic eigenstates not computed. Call compute_eigenstates() first.");
        self.e_vals.as_ref().unwrap().clone()
    }
}

/// Phonon Hamiltonian --- Limited to a single phonon mode for simplicity
pub struct PhononHamiltonian {
    pub phonon_mode: BosonMode,
}
impl PhononHamiltonian {
    pub fn new(phonon_mode: BosonMode) -> Self {
        Self { phonon_mode }
    }
}

impl DiagOperator<f64> for PhononHamiltonian {
    fn form_diagonal(&self) -> Array1<f64> {
        // Diagonal elements are n * omega for n = 0, 1, ..., n_fock - 1
        (0..self.phonon_mode.n_fock).map(|n| n as f64 * self.phonon_mode.omega).collect::<Array1<f64>>()
    }
    
}

#[derive(Clone)]
pub struct PPDiagHamiltonian {
    pub exciton: Arc<ExcitonicHamiltonian>,
    pub photon: Arc<PhotonicHamiltonian>,
    pub phonon: Option<Arc<PhononHamiltonian>>,
    pub k: Vec<f64>,
    pub int_array: Option<Array1<c64>>,
    pub lm_coupling: f64
}

impl PPDiagHamiltonian{
    pub fn new(
        exciton: Arc<ExcitonicHamiltonian>, 
        photon: Arc<PhotonicHamiltonian>,
        phonon: Option<Arc<PhononHamiltonian>>, 
        k: Vec<f64>,
        lm_coupling: f64
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            k, 
            int_array: None,
            lm_coupling
        }
    }

    pub fn calculate_interaction_array(&mut self) {

        let lm_array_len = self.exciton.n_ex_eig * (self.photon.photon_modes.len() + 1);
        let n_ph = self.photon.photon_modes.len();

        let int_array = (0..lm_array_len).map(|i| {
            let ph_ind = i % (n_ph + 1);    
            let ex_ind = i / (n_ph + 1);

            if ph_ind == 0 {
                c64::zero()  // No photons, no interaction terms
            } else {
                // P·q (a† a) / 2 M term (diagonal terms only)
                let q = &self.photon.photon_modes[ph_ind - 1].q;
                let p_com = self.exciton.get_p_com(q, true);
                p_com[[ex_ind,0]] / 2.0 / self.exciton.exciton_ham.m // (a† a) = 1 for single photon states

                // Diamagnetic A^2 term (diagonal terms only)
                + self.lm_coupling * self.lm_coupling / 2.0 / self.exciton.exciton_ham.rel_ham.mu  // A^2 term scales with coupling squared

                // q^2 (a† a)^2 term
                + q.dot(q) / 2.0 / self.exciton.exciton_ham.m   // (a† a)^2 = 1 for single photon states
            }
        }).collect::<Array1<c64>>();

        self.int_array = Some(int_array);
    }
    
}

impl ArrayOperator<c64> for PPDiagHamiltonian {
    fn apply(&self, x: &Array1<c64>) -> Vec<c64> {
        assert!(self.int_array.is_some(), "Interaction array not computed. Call calculate_interaction_array() first.");
        let int_array = self.int_array.as_ref().unwrap();

        let exciton_part = self.exciton.form_diagonal();
        let photon_part = self.photon.form_diagonal();

        // let n_ex = exciton_part.len();
        let n_ph = photon_part.len();

        if self.phonon.is_none() {
            assert!(x.len() == self.exciton.n_ex_eig * (self.photon.photon_modes.len() + 1), 
                "Input vector length must match LM Hamiltonian dimension.");

            let mut result = Vec::with_capacity(x.len());

            result.extend(exciton_part.iter().enumerate().flat_map(|(i, &e_val)| {
                    photon_part.iter().enumerate().map(move |(j, &ph_val)| {
                        let idx = i * n_ph + j;
                        (e_val + ph_val + int_array[idx]) * x[idx]
                    })
                }));

            result
        }
        else {
            assert!(x.len() == self.exciton.n_ex_eig * (self.photon.photon_modes.len() + 1)
                * self.phonon.as_ref().unwrap().phonon_mode.n_fock, 
                "Input vector length must match PP Hamiltonian dimension.");

            let phonon_part = self.phonon.as_ref().unwrap().form_diagonal();
            let n_pn = phonon_part.len();

            let mut result = Vec::with_capacity(x.len());
            result.extend(x.iter().enumerate().map(|(idx, &val)| {
                    let ex_idx = idx / (n_ph * n_pn);
                    let ph_idx = (idx / n_pn) % n_ph;
                    let pn_idx = idx % n_pn;
                    let lm_idx = ex_idx * n_ph + ph_idx;

                    let e_val = exciton_part[ex_idx];
                    let ph_val = photon_part[ph_idx];
                    let pn_val = phonon_part[pn_idx];

                (e_val + ph_val + pn_val + int_array[lm_idx]) * val
            }));

            result

        }
    }
}

#[derive(Clone)]
pub struct PPOffHamiltonian {
    pub exciton: Arc<ExcitonicHamiltonian>,
    pub photon: Arc<PhotonicHamiltonian>,
    pub phonon: Option<Arc<PhononHamiltonian>>, 
    pub lm_coupling: f64,
    pub pn_coupling: f64,
    pub matrix_element_arrays: Option<LMMatrixElementArray>,
    pub p_c_sin: Option<Vec<Array2<c64>>>,
    pub p_q: Option<Vec<Array2<c64>>>,
}

impl PPOffHamiltonian {
    pub fn new(
        exciton: Arc<ExcitonicHamiltonian>, 
        photon: Arc<PhotonicHamiltonian>,
        phonon: Option<Arc<PhononHamiltonian>>, 
        lm_coupling: f64,
        pn_coupling: f64,
        matrix_element_arrays: Option<LMMatrixElementArray>,
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            lm_coupling,
            pn_coupling,
            matrix_element_arrays,
            p_c_sin: None,
            p_q: None,
        }
    }

    pub fn calc_p_c_sin_matrix(&mut self) {
        
        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state

        let sin_mat = &self.matrix_element_arrays.as_ref().unwrap().sin_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // timed_println!("{:?}",test);
        
        let mut p_c_sin: Vec<Array2<c64>> = Vec::new();
        let mut p_q: Vec<Array2<c64>> = Vec::new();
        (0 .. n_ph-1).for_each(|i| {
            let eps = &self.photon.photon_modes[i].eps;
            let q = &self.photon.photon_modes[i].q;
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

    fn calc_all_q_ind(&self) -> Vec<Vec<i32>> {
        let mut all_q_ind: Vec<Vec<i32>> = Vec::new();
        let n_ph = self.photon.photon_modes.len();

        // Single photon modes
        for i in 0..n_ph {
            all_q_ind.push(vec![i as i32]);
        }

        // Pairs of photon modes for A^2 terms
        for i in 0..n_ph {
            for j in i..n_ph {
                if i != j {
                    all_q_ind.push(vec![i as i32, j as i32]);
                }
            }
        }

        // Single phonon mode if present
        if self.phonon.is_some() {
            let phonon_mode_ind = n_ph as i32;
            all_q_ind.push(vec![phonon_mode_ind]);
        }

        all_q_ind
    }

    pub fn calc_matrix_element_arrays(&mut self) {

        // if taylor {
            // Using Taylor expansion for small q with analytic integrals
            let all_q_ind = self.calc_all_q_ind();

            let mut bosonic_modes = self.photon.photon_modes.clone();
            if self.phonon.is_some() {
                bosonic_modes.push(self.phonon.as_ref().unwrap().phonon_mode.clone());
            }

            let mut matrix_element_arrays = LMMatrixElementArray::new();
            matrix_element_arrays = matrix_element_arrays.populate_tay_mat_elements(
                &all_q_ind, 
                self.exciton.e_vecs.as_ref().unwrap(), 
                &bosonic_modes, 
                self.exciton.exciton_ham.kappa_vectors.len(),
                self.exciton.exciton_ham.rel_ham.mu
            );

            self.matrix_element_arrays = Some(matrix_element_arrays);
    }

}

impl Operator<c64> for PPOffHamiltonian {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        
        assert!(self.matrix_element_arrays.is_some(), "Matrix element arrays not computed. Call calc_matrix_element_arrays() first.");
        assert!(self.p_c_sin.is_some(), "P sin(x) element arrays not computed. Call calc_p_cos_matrix() first.");
        assert!(self.p_q.is_some(), "P q element arrays not computed. Call calc_p_cos_matrix() first.");

        let p_c_sin_ref = self.p_c_sin.as_ref().unwrap();
        let matrix_element_arrays_ref = self.matrix_element_arrays.as_ref().unwrap();
        let p_q_ref = self.p_q.as_ref().unwrap();

        let n_ex = self.exciton.n_ex_eig;
        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state
        let n_pn = if self.phonon.is_some() { self.phonon.as_ref().unwrap().phonon_mode.n_fock } else { 1 };
        assert!(x.len() == n_ex * n_ph * n_pn, 
            "Input vector length must match PP Hamiltonian dimension.");

        let x_vec = ndarray::ArrayView1::from(x);
        let stride = n_ph * n_pn;
        
        // Pre-compute some constants to avoid repeated calculations
        let lm_coupling_2_over_m = self.lm_coupling * 2.0 / self.exciton.exciton_ham.m;
        let lm_coupling_2_over_mu = self.lm_coupling / self.exciton.exciton_ham.rel_ham.mu;
        let one_over_2m = 1.0 / (2.0 * self.exciton.exciton_ham.m);

        // Implement all a and a† terms
        let y_ph_list: Vec<(usize, c64)> = self.photon.photon_modes.par_iter().enumerate().flat_map(|(mode_i, photon_mode_i)| {
            let p_c_sin = &p_c_sin_ref[mode_i];
            let p_cos = matrix_element_arrays_ref.p_cos_mat.get(&vec![mode_i as i32]).unwrap();
            let p_q = &p_q_ref[mode_i];

            let mut a_term: Array1<c64>;
            let mut a_dag_term: Array1<c64>;
            let mut aa_term: Array1<c64>;

            let mut local_y: Vec<(usize, c64)> = Vec::new();

            // Sum over all phonon states
            for pn_ind in 0..n_pn {
                let mode_i_start = (mode_i + 1) * n_pn + pn_ind;
                let ground_start = pn_ind;
                
                // Get slice for mode_i at this phonon level
                let x_ph_mode_i = x_vec.slice(s![mode_i_start..;stride]);

                // Combine a† and P a† a terms - more efficient!
                {
                    // Get slices for ground state and mode_i at this phonon level
                    let x_ph_ground = x_vec.slice(s![ground_start..;stride]);

                    // Calculate terms using BLAS-powered dot products
                    // ( P sin(q•x/2) + i p cos(q•x/2) ) â† 
                    a_dag_term = &(&p_c_sin.dot(&x_ph_ground) * lm_coupling_2_over_m) + 
                                 &(&p_cos.dot(&x_ph_ground) * lm_coupling_2_over_mu);
                    aa_term = &p_q.dot(&x_ph_mode_i) * one_over_2m;
                    

                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + (mode_i + 1) * n_pn + ex_ind * stride).collect();
                    let values: Vec<c64> = (a_dag_term + &aa_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
                    
                // For a terms - Process mode_i to ground state transition
                {                    
                    // ( P sin(q•x/2) - i p cos(q•x/2) ) â† 
                    a_term = &(&p_c_sin.dot(&x_ph_mode_i) * lm_coupling_2_over_m) - 
                             &(&p_cos.dot(&x_ph_mode_i) * lm_coupling_2_over_mu);
                    
                    // Add to y at photon ground state and phonon in pn_ind
                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + ex_ind * stride).collect();
                    let values: Vec<c64> = (a_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
            }

            // Diamagnetic A^2 terms
            for (mode_j, photon_mode_j) in self.photon.photon_modes.iter().enumerate() {
                let q_ind1 = (mode_i as i32).max(mode_j as i32);
                let q_ind2 = (mode_i as i32).min(mode_j as i32);
                let q_ind_key = vec![q_ind1, -q_ind2];

                // Since we are in sing-ex for photons a^† a^† and a a terms are 0
                // Only a^† a terms contribute for off-diagonal terms
                if let Some(cos_matrix) = matrix_element_arrays_ref.cos_mat.get(&q_ind_key) 
                    && mode_i != mode_j 
                    {
                    let eps_i = &self.photon.photon_modes[mode_i].eps;
                    let eps_j = &self.photon.photon_modes[mode_j].eps;
                    
                    // Calculate dot product directly
                    let eps_dot = eps_j[0] * eps_i[0] + eps_j[1] * eps_i[1];
                    let coupling_factor: f64 = self.lm_coupling * self.lm_coupling * eps_dot / 2.0 / self.exciton.exciton_ham.rel_ham.mu;

                    for pn_ind in 0..n_pn {
                        let mode_j_start = (mode_j + 1) * n_pn + pn_ind;
                        
                        // Get slice for mode_j at this phonon level
                        let x_ph_mode_j = x_vec.slice(s![mode_j_start..;stride]);
                        
                        // Calculate aa_term using BLAS-powered dot product
                        aa_term = &cos_matrix.dot(&x_ph_mode_j) * coupling_factor;

                        // Add to y at photon mode_i and phonon in pn_ind
                        let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + (mode_i + 1) * n_pn + ex_ind * stride).collect();
                        let values: Vec<c64> = (aa_term).to_vec();
                        local_y.extend(indices.into_iter().zip(values.into_iter()));
                    }
                }
                else if let None = matrix_element_arrays_ref.cos_mat.get(&q_ind_key) && mode_i != mode_j  {
                    panic!("No key found for q = {:?} and q' = {:?}", photon_mode_i.q, photon_mode_j.q);
                }
            }

            local_y
        }).collect();

        // Implement all phonon b and b† terms if n_pn > 1
        let y_pn_list: Vec<(usize, c64)>;

        if self.phonon.is_some() && self.pn_coupling.abs() > 1e-12 {
            let phonon_mode_ind: i32 = self.photon.photon_modes.len() as i32;
            let _cos_mat_phonon = matrix_element_arrays_ref.cos_mat.get(&vec![phonon_mode_ind]).unwrap();

            //// TEST: Using Sin(x_r) instead of Cos(x_r)
            let sin_mat_phonon = matrix_element_arrays_ref.sin_mat.get(&vec![phonon_mode_ind]).unwrap();
                            
            // let disp_mat = PrettyArray2(&sin_mat_phonon, 4);
            // timed_println!("Sine Matrix for phonon = ");
            // println!("{}", disp_mat);

            y_pn_list = (0..n_pn-1).into_par_iter().flat_map(|pn_ind| {
                let sqrt_pn_plus_1 = (c64::one() * (pn_ind + 1) as f64).sqrt();
                let pn_coupling_factor = self.pn_coupling * sqrt_pn_plus_1;

                let mut local_y: Vec<(usize, c64)> = Vec::new();

                for ph_ind in 0..n_ph {
                    let pn_dex_start = ph_ind * n_pn + pn_ind;
                    let pn_exc_start = pn_dex_start + 1;
                    
                    // For b† term - Process pn_ind to pn_ind+1 transition
                    {
                        // Get slice for phonon pn_ind at this photon level
                        let x_ph_pn = x_vec.slice(s![pn_dex_start..;stride]);
                        
                        // Calculate b_dag_term using BLAS-powered dot product
                        let b_dag_term = &sin_mat_phonon.dot(&x_ph_pn) * pn_coupling_factor;
                        
                        // Add to y at phonon pn_ind+1 and photon in ph_ind
                        let indices: Vec<usize> = (0..n_ex).map(|ex_ind| (pn_ind + 1) + ph_ind * n_pn + ex_ind * stride).collect();
                        let values: Vec<c64> = (b_dag_term).to_vec();
                        local_y.extend(indices.into_iter().zip(values.into_iter()));
                    }
                    
                    // For b term - Process pn_ind+1 to pn_ind transition
                    {
                        // Get slice for phonon pn_ind+1 at this photon level
                        let x_ph_pn_plus_1 = x_vec.slice(s![pn_exc_start..;stride]);
                        
                        // Calculate b_term using BLAS-powered dot product
                        let b_term = &sin_mat_phonon.dot(&x_ph_pn_plus_1) * pn_coupling_factor;
                        
                        // Add to y at phonon pn_ind and photon in ph_ind
                        let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + ph_ind * n_pn + ex_ind * stride).collect();
                        let values: Vec<c64> = (b_term).to_vec();
                        local_y.extend(indices.into_iter().zip(values.into_iter()));
                    }
                }

                local_y
            }).collect();
        }
        else {
            y_pn_list = vec![(0,c64::zero())]; // No phonon contributions
        }

        // Combine results from photon and phonon contributions
        for (idx, val) in y_ph_list.into_iter().chain(y_pn_list.into_iter()) {
            y[idx] += val;
        }

    }
}

#[derive(Clone)]
pub struct PolaronPolariton {
    pub k: Vec<f64>,
    pub pp_off_ham:  PPOffHamiltonian,
    pub pp_diag_ham: PPDiagHamiltonian,
    pub exciton: Arc<ExcitonicHamiltonian>,
}

impl PolaronPolariton {
    pub fn new(
        exciton: Arc<ExcitonicHamiltonian>,
        photon: Arc<PhotonicHamiltonian>,
        phonon: Option<Arc<PhononHamiltonian>>, 
        k: Vec<f64>,
        lm_coupling: f64,
        pn_coupling: f64,
    ) -> Self {

        let mut pp_off_ham:PPOffHamiltonian;  
        let mut pp_diag_ham:PPDiagHamiltonian;  

        if let Some(pn) = phonon{
            pp_off_ham = PPOffHamiltonian::new(Arc::clone(&exciton), photon.clone(), Some(pn.clone()), lm_coupling, pn_coupling, None);
            pp_off_ham.calc_matrix_element_arrays();
            pp_off_ham.calc_p_c_sin_matrix();

            pp_diag_ham = PPDiagHamiltonian::new(Arc::clone(&exciton), photon, Some(pn), k.clone(), lm_coupling);
            pp_diag_ham.calculate_interaction_array();
        }
        else {
            pp_off_ham = PPOffHamiltonian::new(Arc::clone(&exciton), photon.clone(), None, lm_coupling, pn_coupling, None);
            pp_off_ham.calc_matrix_element_arrays();
            pp_off_ham.calc_p_c_sin_matrix();

            pp_diag_ham = PPDiagHamiltonian::new(Arc::clone(&exciton), photon, None, k.clone(), lm_coupling);
            pp_diag_ham.calculate_interaction_array();
        }

        Self { 
            k,
            pp_off_ham:  pp_off_ham,
            pp_diag_ham: pp_diag_ham,
            exciton
        }
    }


    pub fn update_k(&mut self, new_k: Vec<f64>) {
        self.k = new_k.clone();

        // Create a new ExcitonicHamiltonian
        let mut ex_ham = self.exciton.exciton_ham.clone();
        ex_ham.k = new_k.clone();
        let mut new_exciton = ExcitonicHamiltonian::new(
            ex_ham, 
            self.exciton.n_ex_eig, 
            self.exciton.n_rel
        );
        new_exciton.compute_eigenstates();
        
        // Update the exciton
        self.exciton = Arc::new(new_exciton);
        
        // Update Mat-Vec Hamiltonians with references to the updated exciton
        self.pp_diag_ham.exciton = self.exciton.clone();
        self.pp_diag_ham.k = new_k;
        self.pp_diag_ham.calculate_interaction_array();


        self.pp_off_ham.exciton = self.exciton.clone();
        self.pp_off_ham.calc_p_c_sin_matrix();
    }

}


impl SolverOperator<c64> for PolaronPolariton {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Avoid unnecessary allocation and copying
        let x_arr = Array1::from_shape_fn(x.len(), |i| x[i]);

        // timed_println!("Applying Mat-Vec");
        // timed_println!("Applying Diag");
        // Apply diagonal part directly to y
        let diag_part = self.pp_diag_ham.apply(&x_arr);
        for i in 0..y.len() {
            y[i] = diag_part[i];
        }
        
        // timed_println!("Applying Off-Diag");
        // Apply off-diagonal part and add to y
        self.pp_off_ham.apply(x, y);
    }

    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        // Since Hamiltonians are hermitian
        self.apply(x, y)
    }
    
    fn get_diagonal(&self) -> Vec<c64> {
        // Get the diagonal elements for preconditioning
        let exciton_diag = self.exciton.form_diagonal();
        let photon_diag = self.pp_diag_ham.photon.form_diagonal();
        let n_ex = exciton_diag.len();
        let n_ph = photon_diag.len();
        
        let mut diag = Vec::with_capacity(n_ex * n_ph);
        
        if let Some(phonon) = &self.pp_diag_ham.phonon {
            let phonon_diag = phonon.form_diagonal();
            let n_pn = phonon_diag.len();
            
            for ex_idx in 0..n_ex {
                for ph_idx in 0..n_ph {
                    for pn_idx in 0..n_pn {
                        let lm_idx = ex_idx * n_ph + ph_idx;
                        let e_val = exciton_diag[ex_idx];
                        let ph_val = photon_diag[ph_idx];
                        let pn_val = phonon_diag[pn_idx];
                        let int_val = self.pp_diag_ham.int_array.as_ref().unwrap()[lm_idx];
                        
                        diag.push(e_val + ph_val + pn_val + int_val);
                    }
                }
            }
        } else {
            for ex_idx in 0..n_ex {
                for ph_idx in 0..n_ph {
                    let lm_idx = ex_idx * n_ph + ph_idx;
                    let e_val = exciton_diag[ex_idx];
                    let ph_val = photon_diag[ph_idx];
                    let int_val = self.pp_diag_ham.int_array.as_ref().unwrap()[lm_idx];
                    
                    diag.push(e_val + ph_val + int_val);
                }
            }
        }
        
        diag
    }
    
    fn dimension(&self) -> usize {
        let n_ex = self.exciton.n_ex_eig;
        let n_ph = self.pp_diag_ham.photon.photon_modes.len() + 1;
        
        if let Some(phonon) = &self.pp_diag_ham.phonon {
            n_ex * n_ph * phonon.phonon_mode.n_fock
        } else {
            n_ex * n_ph
        }
    }

    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<rust_slepc_bindings::EPSType>, initial_guess: Option<&Array2<c64>>) -> (Array1<f64>, Array2<c64>) {
        
        let block_size = 128;
        
        use crate::slepc::diagonalize_slepc;

        if let Some(n) = n_eig {                   
            diagonalize_slepc(self, n, block_size, solve_type, initial_guess)
        }
        else {
            timed_println!("Warning: No eigenvalue count specified, using default of 128");
            diagonalize_slepc(self, block_size, 128, solve_type, initial_guess)
        }
    }
}

pub struct SingExWrapper {
    pp_config: PPConfig,
    pub polaron_polariton: Option<PolaronPolariton>
}

impl SingExWrapper {
    pub fn new(pp_config: PPConfig) -> Self {
        Self { pp_config, polaron_polariton: None }
    }

    pub fn update_k(&mut self, new_k:Vec<f64> ) {
        self.polaron_polariton.as_mut().unwrap().update_k(new_k);
    }

    pub fn generate_pp(&mut self) {
        let config = &self.pp_config;

        // Initialize Exciton System
        let ex_system = Exciton2DGridSystem::from_config(config.e_config.clone());
        let mut exciton = ExcitonicHamiltonian::new(ex_system.full_ham, 
            config.n_ex_eig, config.n_rel);
        
        exciton.compute_eigenstates();

        // Initialize Photonic System
        let photon_modes: Vec<BosonMode> = config.q_values.iter().map(|q| {
                BosonMode::new(*q, Polarization::TE, 2)
            }).collect();
        let photon = PhotonicHamiltonian::new(photon_modes);

        if config.pn_n_fock > 1 {
            assert!(config.pn_n_fock > 1, "Phonon Fock space must be greater than 1 when phonon coupling is non-zero.");

            // Initialize Phononic System
            let phonon_mode = BosonMode { eps: arr1(&[0.0,0.0]), q: config.pn_k.clone(), 
                omega: config.pn_omega, n_fock: config.pn_n_fock };
            let phonon = PhononHamiltonian::new(phonon_mode);

            self.polaron_polariton = Some(PolaronPolariton::new(Arc::new(exciton), Arc::new(photon), Some(Arc::new(phonon)), vec![0.,0.], 
                config.lm_coupling, config.pn_coupling));
        }
        else {
            self.polaron_polariton = Some(PolaronPolariton::new(Arc::new(exciton), Arc::new(photon), None, vec![0.,0.], 
                config.lm_coupling, config.pn_coupling));
        }


    }


}

/// Generates a filename for the cached diagonalization data (pp_vals and x_pp)
pub fn get_dielectric_cache_filename(
    pp: &PolaronPolariton,
    pp_config: &PPConfig,
    save_dir: &str,
    n_eig: usize
) -> String {
    format!(
        "{}/dielectric_cache_a0{}_V0{}_nrel{}_lmcoup{}_nmodes{}_pncoup{}_npn{}_neig{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    )
}

/// Generates a filename for the dielectric function data
pub fn get_dielectric_function_filename(
    pp: &PolaronPolariton,
    pp_config: &PPConfig,
    save_dir: &str,
    n_eig: usize,
    gamma: f64
) -> String {
    format!(
        "{}/dielectric_sin_a0{}_V0{}_nrel{}_lmcoup{}_nmodes{}_pncoup{}_npn{}_neig{}_gamma{:.1e}.dat",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig,
        gamma
    )
}

/// Save the cached diagonalization data (pp_vals and x_pp)
pub fn save_dielectric_cache(
    filename: &str,
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(filename)?;
    
    // Write dimensions
    let n_eig = pp_vals.len();
    file.write_all(&(n_eig as u64).to_le_bytes())?;
    
    // Write pp_vals
    for &val in pp_vals.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    
    // Write x_pp
    for &val in x_pp.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    
    Ok(())
}

/// Load the cached diagonalization data (pp_vals and x_pp)
pub fn load_dielectric_cache(
    filename: &str
) -> std::io::Result<(Array1<f64>, Array1<f64>)> {
    use std::fs::File;
    use std::io::Read;
    
    let mut file = File::open(filename)?;
    
    // Read dimensions
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    let n_eig = u64::from_le_bytes(buf) as usize;
    
    // Read pp_vals
    let mut pp_vals = Array1::<f64>::zeros(n_eig);
    for i in 0..n_eig {
        file.read_exact(&mut buf)?;
        pp_vals[i] = f64::from_le_bytes(buf);
    }
    
    // Read x_pp
    let mut x_pp = Array1::<f64>::zeros(n_eig);
    for i in 0..n_eig {
        file.read_exact(&mut buf)?;
        x_pp[i] = f64::from_le_bytes(buf);
    }
    
    Ok((pp_vals, x_pp))
}

/// Calculate the dielectric function from cached or newly computed pp_vals and x_pp
pub fn get_dielectric_function(
    pp_config: &PPConfig, 
    pp: &PolaronPolariton, 
    n_eig: usize, 
    omegas: Array1<f64>, 
    gamma: f64, 
    save_dir: &str,
    use_cached: bool
) -> (Array1<f64>, Array1<f64>) {
    timed_println!("Calculating dielectric function with {} eigenstates, gamma={:.1e}", n_eig, gamma);

    let cache_filename = get_dielectric_cache_filename(pp, pp_config, save_dir, n_eig);
    
    // Try to load cached data if requested
    let (pp_vals, x_pp) = if use_cached && std::path::Path::new(&cache_filename).exists() {
        timed_println!("Loading cached diagonalization data from {}", cache_filename);
        match load_dielectric_cache(&cache_filename) {
            Ok((vals, x)) => {
                timed_println!("Successfully loaded cached data");
                (vals, x)
            },
            Err(e) => {
                timed_println!("Failed to load cache ({}), recalculating...", e);
                let (vals, x) = compute_dielectric_cache(pp_config, pp, n_eig);
                // Save the newly computed cache
                if let Err(e) = save_dielectric_cache(&cache_filename, &vals, &x) {
                    timed_println!("Warning: Failed to save cache: {}", e);
                }
                (vals, x)
            }
        }
    } else {
        timed_println!("Computing diagonalization data...");
        let (vals, x) = compute_dielectric_cache(pp_config, pp, n_eig);
        // Save the cache
        timed_println!("Saving diagonalization cache to {}", cache_filename);
        if let Err(e) = save_dielectric_cache(&cache_filename, &vals, &x) {
            timed_println!("Warning: Failed to save cache: {}", e);
        }
        (vals, x)
    };

    timed_println!("Relative energies = {:?}", pp_vals.slice(s![0..50.min(pp_vals.len())]).to_owned() - pp_vals[0]);

    // Compute dielectric function from cached data
    let dielectric_function: Array1<c64> = Array1::from_shape_fn(omegas.len(), |idx| {
        let omega = omegas[idx];

        let result: c64 = x_pp.iter().enumerate().map(|(pp_ind, x_pp_val)| {
            x_pp_val * x_pp_val / (omega - (pp_vals[pp_ind] - pp_vals[0]) + c64::i() * gamma)
        }).sum();

        1.0 + 4.0 * PI * result
    });

    (dielectric_function.map(|&x| x.re), dielectric_function.map(|&x| x.im))
}

/// Helper function to compute the expensive diagonalization and matrix elements
fn compute_dielectric_cache(
    pp_config: &PPConfig,
    pp: &PolaronPolariton,
    n_eig: usize,
) -> (Array1<f64>, Array1<f64>) {
    // Lag Coeffs for Relative coordinates
    let (lag_coeffs, states) = &pp.pp_off_ham.matrix_element_arrays.as_ref().unwrap().get_laguerre_coeffs(pp.exciton.n_rel);
    let mu = pp.exciton.exciton_ham.rel_ham.mu;

    // Get <0|x|n> matrix elements in the exciton basis
    
    // Calculate the matrix elements using iterators
    let parities = pp.exciton.parities.as_ref().unwrap();
    let e_vecs = pp.exciton.e_vecs.as_ref().unwrap();
    let n_com = pp.exciton.exciton_ham.kappa_vectors.len();
    let n_rel = pp_config.n_rel;
    let n_ex_eig = pp_config.n_ex_eig;

    let mut n_rel_even = 0;
    let mut n_rel_odd = 0;
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

    // Pre-calculate parity indices for all eigenvectors -> Odd = 1, Even = 0
    let parity_indices: Vec<usize> = parities.iter().map(|&p| (p == -1) as usize).collect();

    let x_rel = &Array2::from_shape_fn((n_rel,n_rel), |(r_ind1, r_ind2)| {
        pp.pp_off_ham.matrix_element_arrays.as_ref().unwrap().get_x(&states[r_ind1], &states[r_ind2], mu, &lag_coeffs)
    });

    timed_println!("<x> for Relative Coordinate:");
    println!("{}", PrettyArray2(&x_rel, 4));

    // Get n, m for even and odd states
    let mut odd_par_states = Vec::new();
    let mut even_par_states = Vec::new();

    states.iter().for_each(|(n,m)| { 
        if m.abs() % 2 == 1 {
            odd_par_states.push((*n, *m));
        }
        else if m.abs() % 2 == 0 {
            even_par_states.push((*n, *m));
        }
     });

     println!("odd_par_states = {:?}", odd_par_states);
     println!("even_par_states = {:?}", even_par_states);



    let rel_idx_converter = |(rel_par_idx, par): (usize,usize)| {

        if par == 1 {
            let state = odd_par_states[rel_par_idx];

            let r_idx = states.iter().position(|&s| s == state).unwrap();
            return r_idx
        }
        else {
            let state = even_par_states[rel_par_idx];

            let r_idx = states.iter().position(|&s| s == state).unwrap();
            return r_idx
        }
    };

    // Get <x> in the Exciton basis
    let x_exc = &Array2::from_shape_fn((n_ex_eig,n_ex_eig) , |(idx1, idx2)| {
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];

            let idx1_par = parity_indices[idx1];
            let idx2_par = parity_indices[idx2];

            // if idx1_par == idx2_par {
            //     return 0.0
            // }

            // Calculate the matrix elements using iterators
            (0..n_rel_par[idx1_par]).flat_map(|rel_ind1| {
                let r_ind1 = rel_idx_converter((rel_ind1,idx1_par));
                
                (0..n_rel_par[idx2_par]).flat_map(move |rel_ind2| {
                    let r_ind2 = rel_idx_converter((rel_ind2,idx2_par));

                    (0..n_com).map(move |com_ind| {
                        let i1 = n_com * rel_ind1 + com_ind;
                        let i2 = n_com * rel_ind2 + com_ind;
                        let a1 = vec1[i1];
                        let a2 = vec2[i2];

                        // For complex eigenvectors: <ψ_1|x|ψ_2> = Σ ψ_1* x ψ_2
                        (a1.conj() * a2).re * x_rel[[r_ind1,r_ind2]]
                    })
                })
            })
            .fold(0.0, |acc, val| acc + val)
    });


    // // Debug: Get the Relative Coordinate EV distribution in Exciton EV

    // let e_vecs_rel = &Array2::from_shape_fn( (n_rel, n_ex_eig), |(rel_idx, ex_idx)| {

    //     let e_vec = &e_vecs[ex_idx];
    //     let vec_par = parity_indices[ex_idx];

    //     let r_ind = rel_idx + vec_par * n_rel_odd;

    //     (0..n_com).map(|com_idx| {
    //         let idx = n_com * r_ind + com_idx;

    //         e_vec[idx]
    //     }).sum::<f64>()
    // });
    // timed_println!("Relative Coordinate EV distribution in Exciton EV:");
    // println!("{}", PrettyArray2(&e_vecs_rel, 4));


    timed_println!("<x> for exciton:");
    println!("{}", PrettyArray2(&x_exc, 4));

    timed_println!("Exciton eigenvalues transitions");
    let e_vals_transition = pp.exciton.e_vals.as_ref().unwrap() - pp.exciton.e_vals.as_ref().unwrap()[0];
    println!("{}", e_vals_transition);
    

    // Diagonalize at the Gamma point
    let (pp_vals, pp_vecs) = diagonalize_pp_sing_ex(pp, n_eig, None);

    // get <x> in the PP basis
    let n_ph = pp_config.q_values.len() + 1;
    let n_pn = pp_config.pn_n_fock;

    let vec0 = pp_vecs.column(0);

    let x_pp = Array1::from_shape_fn(n_eig, |idx| {
        let vec1 = pp_vecs.column(idx);

        (0.. n_ex_eig).flat_map(|ex_ind1| {
            (0 .. n_ex_eig).flat_map(move |ex_ind2|{
                let x_element = x_exc[[ex_ind1,ex_ind2]];
                (0 .. n_ph*n_pn).map(move | bos_ind | {
                    let pp_ind1 = ex_ind1 * n_ph*n_pn + bos_ind;
                    let pp_ind2 = ex_ind2 * n_ph*n_pn + bos_ind;

                    vec0[pp_ind1] * vec1[pp_ind2] * x_element
                })
            })
        }).fold(c64::zero(), |acc, val| acc + val).norm()
    });

    (pp_vals, x_pp)
}

/// Generates a filename for the band structure data
pub fn get_lm_band_structure_filename(
    pp: &PolaronPolariton,
    pp_config: &PPConfig,
    save_dir: String,
    n_points: usize,
    n_eig: usize
) -> String {
    format!(
        "{}/band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{}_nmodes{}_pncoup{}_npn{}_neig{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_points,
        pp_config.lm_coupling,
        pp.pp_off_ham.photon.photon_modes.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    )
}

// Then replace all diagonalization functions with a single unified function
pub fn diagonalize_pp_sing_ex(
    pp: &PolaronPolariton, 
    n_eig: usize,
    initial_vec: Option<&Array2<c64>>
) -> (Array1<f64>, Array2<c64>) {
    
    let block_size = 128;
    
    use crate::slepc::diagonalize_slepc;
    use rust_slepc_bindings::{EPSType};

    diagonalize_slepc(pp, n_eig, block_size, Some(EPSType::KrylovSchur), initial_vec)           
}

// Replace the specific band structure calculation functions with a unified one
pub fn calculate_band_structure_sing_ex(
    pp: &mut PolaronPolariton, 
    n_k_points: usize, 
    n_eig: usize, 
    sym_points: &Vec<Vec<f64>>,
    pp_config: &PPConfig
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {   
    let save_dir = "pp_data".to_string();
    let filename = get_lm_band_structure_filename(&pp, &pp_config, save_dir, n_k_points, n_eig);

    // Calculate points per segment of the path
    let points_per_seg = n_k_points / sym_points.len();

    // Generate k-points along the path
    let k_points: Vec<Vec<f64>> = sym_points.iter().enumerate().flat_map(|(sym_ind,sym_point)| {
        let sym_ind2 = (sym_ind + 1) % sym_points.len();
        (0..points_per_seg).map(move |t| {
            // Interpolate between symmetry points
            let t_frac = t as f64 / (points_per_seg - 1) as f64;
            let x = sym_point[0] + t_frac * (sym_points[sym_ind2][0] - sym_point[0]);
            let y = sym_point[1] + t_frac * (sym_points[sym_ind2][1] - sym_point[1]);

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

    // Create a clone of the PolaronPolariton for each thread
    let pp_arc = Arc::new(pp.clone());
    
    // Calculate eigenvalues at each k-point in parallel
    let eigvals_list: Vec<(usize, Array1<f64>, Array2<c64>)> = k_points.par_iter().enumerate().map(|(idx, k)| {
        if k.iter().sum::<f64>().is_nan() {
            panic!("Too few k-points must be at least 2x number of high symmetry points");
        }
        
        // Create a thread-local copy of the PolaronPolariton
        let mut local_pp = (*pp_arc).clone();
        local_pp.update_k(k.clone());
        
        // Diagonalize without initial guess in parallel mode
        let (eigvals, eig_vec) = diagonalize_pp_sing_ex(&local_pp, n_eig, None);
        
        // Update progress bar
        pb.inc(1);
        
        // Return index with results to maintain order
        (idx, eigvals, eig_vec)
    }).collect();
    
    pb.finish_with_message("Bandstructure calculation complete");

    // Sort results by index to maintain the original order
    let mut sorted_eigvals = vec![Array1::<f64>::zeros(0); k_points.len()];
    for (idx, eigvals, _) in eigvals_list {
        sorted_eigvals[idx] = eigvals;
    }

    // Save the band structure data
    save_band_data(filename, (k_points.clone(), sorted_eigvals.clone())).unwrap();

    (k_points, sorted_eigvals)
}

