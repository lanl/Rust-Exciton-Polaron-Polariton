use exciton_2d::{Exciton2DGridSystem,  FullHamiltonian};
use ndarray::{arr1, Array1, Array2, s};
use arpack::{Config, Operator}; 
use std::sync::Arc;
use rayon::{prelude::*};


use crate::utils::{LMMatrixElement, LMMatrixElementArray, PPConfig, PhotonMode, Polarization};

pub fn kron_1d(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    a.iter().flat_map(|&x| b.iter().map(move |&y| x * y)).collect::<Array1<f64>>()
}

pub trait ArrayOperator<T> {
    fn apply(&self, x: &Array1<T>) -> Vec<T>;
}

pub trait DiagOperator<T> {
    fn form_diagonal(&self) -> Array1<T>;
}

/// Represents the photonic part of the Hamiltonian
pub struct PhotonicHamiltonian {
    pub photon_modes: Vec<PhotonMode>,
}

impl PhotonicHamiltonian {
    pub fn new(photon_modes: Vec<PhotonMode>) -> Self {
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
    pub parity: i32,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Vec<Array1<f64>>>,
    pub r_vecs: Option<Array2<f64>>,
    pub parities: Option<Array1<i32>>,
}

impl ExcitonicHamiltonian {
    pub fn new(exciton_ham: FullHamiltonian, n_ex_eig: usize, n_rel: usize, parity: i32) -> Self {
        Self { exciton_ham, n_ex_eig, n_rel, parity, e_vals: None, e_vecs: None, r_vecs: None, parities: None}
    }

    pub fn compute_eigenstates(&mut self) {
        let (exciton_energies, e_vecs, _, r_vecs, parities) 
            = self.exciton_ham.diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig, self.parity);
        
        self.e_vals = Some(exciton_energies.slice(s![..self.n_ex_eig]).to_owned());
        // println!("Exciton energies: {:?}", self.e_vals);

        self.e_vecs = Some(e_vecs[0..self.n_ex_eig].to_vec());
        self.r_vecs = Some(r_vecs);
        self.parities = Some(parities.slice(s![..self.n_ex_eig]).to_owned());
    }

    /// Calculates the center-of-mass momentum matrix element P·field_vec
    pub fn get_p_com(&self, field_vec: &Array1<f64>, diag: bool) -> Array2<f64> {
        let e_vecs = self.e_vecs.as_ref().expect("Eigenvectors not computed. Call compute_eigenstates() first.");
        let kappas= &self.exciton_ham.kappa_vectors;
        let k = &self.exciton_ham.k;

        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;
        
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

                // Compute P_ii using iterator operations
                // This is a purely diagonal operator
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        a1 * a1 * p
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
    pub phonon_mode: PhotonMode,
}
impl PhononHamiltonian {
    pub fn new(phonon_mode: PhotonMode) -> Self {
        Self { phonon_mode }
    }
}

impl DiagOperator<f64> for PhononHamiltonian {
    fn form_diagonal(&self) -> Array1<f64> {
        // Diagonal elements are n * omega for n = 0, 1, ..., n_fock - 1
        (0..self.phonon_mode.n_fock).map(|n| n as f64 * self.phonon_mode.omega).collect::<Array1<f64>>()
    }
    
}

pub struct PPDiagHamiltonian {
    pub exciton: Arc<ExcitonicHamiltonian>,
    pub photon: Arc<PhotonicHamiltonian>,
    pub phonon: Option<Arc<PhononHamiltonian>>,
    pub k: Vec<f64>,
    pub int_array: Option<Array1<f64>>,
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
                0.0  // No photons, no interaction terms
            } else {
                // P·q (a† a) / 2 M term (diagonal terms only)
                let q = &self.photon.photon_modes[ph_ind - 1].q;
                let p_com = self.exciton.get_p_com(q, true);
                p_com[[ex_ind,0]] / 2.0 / self.exciton.exciton_ham.m // (a† a) = 1 for single photon states

                // Diamagnetic A^2 term (diagonal terms only)
                + 2.0 * self.lm_coupling * self.lm_coupling  // A^2 term scales with coupling squared

                // q^2 (a† a)^2 term
                + q.dot(q) / 2.0 / self.exciton.exciton_ham.m   // (a† a)^2 = 1 for single photon states
            }
        }).collect::<Array1<f64>>();

        self.int_array = Some(int_array);
    }
    
}

impl ArrayOperator<f64> for PPDiagHamiltonian {
    fn apply(&self, x: &Array1<f64>) -> Vec<f64> {
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

pub struct PPOffHamiltonian {
    pub exciton: Arc<ExcitonicHamiltonian>,
    pub photon: Arc<PhotonicHamiltonian>,
    pub phonon: Option<Arc<PhononHamiltonian>>, 
    pub lm_coupling: f64,
    pub pn_coupling: f64,
    pub matrix_element_arrays: Option<LMMatrixElementArray>,
    pub p_cos: Option<Vec<Array2<f64>>>,
    pub p_q: Option<Vec<Array2<f64>>>,
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
            p_cos: None,
            p_q: None,
        }
    }

    pub fn calculate_matrix_elements(&self) -> LMMatrixElement {
        let mut bosonic_modes = self.photon.photon_modes.clone();
        if self.phonon.is_some() {
            bosonic_modes.push(self.phonon.as_ref().unwrap().phonon_mode.clone());
        }

        LMMatrixElement::new()
            .populate_mat_elements(&self.exciton.r_vecs.as_ref().unwrap(), &bosonic_modes, &self.exciton.exciton_ham)

    }

    pub fn calc_p_cos_matrix(&mut self) {
        
        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state

        let cos_mat = &self.matrix_element_arrays.as_ref().unwrap().cos_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // println!("{:?}",test);
        
        let mut p_cos: Vec<Array2<f64>> = Vec::new();
        let mut p_q: Vec<Array2<f64>> = Vec::new();
        (0 .. n_ph-1).for_each(|i| {
            let eps = &self.photon.photon_modes[i].eps;
            let q = &self.photon.photon_modes[i].q;
            let q_ind_key = i as i32;

            p_cos.push( 
                    self.exciton.get_p_com(eps, false)// P·eps 
                    .dot(cos_mat.get(&vec![q_ind_key]).unwrap()) // cos(qx/2) matrix element
                );
            
            p_q.push( 
                    self.exciton.get_p_com(q, false)// P·q 
                );

        });

        self.p_cos = Some(p_cos);
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

    pub fn calc_matrix_element_arrays(&mut self, taylor: bool) {

        if taylor {
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
                self.exciton.parities.as_ref().unwrap(),
                &bosonic_modes, 
                self.exciton.exciton_ham.kappa_vectors.len(),
                self.exciton.exciton_ham.rel_ham.mu
            );

            self.matrix_element_arrays = Some(matrix_element_arrays);
        }
        else {
            // Using grid wavefunctions and photon/phonon modes, compute all necessary matrix elements
            let matrix_elements = &self.calculate_matrix_elements();

            let mut matrix_element_arrays = LMMatrixElementArray::new();
            matrix_element_arrays = matrix_element_arrays.populate_mat_elements(matrix_elements, 
                self.exciton.e_vecs.as_ref().unwrap(), 
                self.exciton.parities.as_ref().unwrap(), 
                self.exciton.exciton_ham.kappa_vectors.len());

            self.matrix_element_arrays = Some(matrix_element_arrays);
        }

    }

}

impl Operator<f64> for PPOffHamiltonian {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        
        assert!(self.matrix_element_arrays.is_some(), "Matrix element arrays not computed. Call calc_matrix_element_arrays() first.");
        assert!(self.p_cos.is_some(), "P cos(x) element arrays not computed. Call calc_p_cos_matrix() first.");
        assert!(self.p_q.is_some(), "P q element arrays not computed. Call calc_p_cos_matrix() first.");

        let p_cos_ref = self.p_cos.as_ref().unwrap();
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
        let lm_coupling_2_over_mu = self.lm_coupling * 2.0 / self.exciton.exciton_ham.rel_ham.mu;
        let one_over_2m = 1.0 / (2.0 * self.exciton.exciton_ham.m);

        // Implement all a and a† terms
        let y_ph_list: Vec<(usize, f64)> = self.photon.photon_modes.par_iter().enumerate().flat_map(|(mode_i, _)| {
            let p_sin = matrix_element_arrays_ref.p_sin_mat.get(&vec![mode_i as i32]).unwrap();
            let p_cos = &p_cos_ref[mode_i];
            let p_q = &p_q_ref[mode_i];

            let mut a_term: Array1<f64>;
            let mut a_dag_term: Array1<f64>;
            let mut aa_term: Array1<f64>;

            let mut local_y: Vec<(usize, f64)> = Vec::new();

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
                    a_dag_term = &(&p_cos.dot(&x_ph_ground) * lm_coupling_2_over_m) + 
                                 &(&p_sin.dot(&x_ph_ground) * lm_coupling_2_over_mu);
                    aa_term = &p_q.dot(&x_ph_mode_i) * one_over_2m;
                    

                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + (mode_i + 1) * n_pn + ex_ind * stride).collect();
                    let values: Vec<f64> = (a_dag_term + &aa_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
                    
                // For a terms - Process mode_i to ground state transition
                {                    
                    // Calculate a_term using BLAS-powered dot products
                    a_term = &(&p_cos.dot(&x_ph_mode_i) * lm_coupling_2_over_m) - 
                             &(&p_sin.dot(&x_ph_mode_i) * lm_coupling_2_over_mu);
                    
                    // Add to y at photon ground state and phonon in pn_ind
                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + ex_ind * stride).collect();
                    let values: Vec<f64> = (a_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
            }

            // Diamagnetic A^2 terms
            for (mode_j, _) in self.photon.photon_modes.iter().enumerate() {
                let q_ind1 = (mode_i as i32).max(mode_j as i32);
                let q_ind2 = (mode_i as i32).min(mode_j as i32);
                let q_ind_key = vec![q_ind1, -q_ind2];

                if let Some(cos_matrix) = matrix_element_arrays_ref.cos_mat.get(&q_ind_key) {
                    let eps_i = &self.photon.photon_modes[mode_i].eps;
                    let eps_j = &self.photon.photon_modes[mode_j].eps;
                    
                    // Calculate dot product directly
                    let eps_dot = eps_j[0] * eps_i[0] + eps_j[1] * eps_i[1];
                    let coupling_factor: f64 = self.lm_coupling * self.lm_coupling * eps_dot;

                    for pn_ind in 0..n_pn {
                        let mode_j_start = (mode_j + 1) * n_pn + pn_ind;
                        
                        // Get slice for mode_j at this phonon level
                        let x_ph_mode_j = x_vec.slice(s![mode_j_start..;stride]);
                        
                        // Calculate aa_term using BLAS-powered dot product
                        aa_term = &cos_matrix.dot(&x_ph_mode_j) * coupling_factor;

                        // Add to y at photon mode_i and phonon in pn_ind
                        let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + (mode_i + 1) * n_pn + ex_ind * stride).collect();
                        let values: Vec<f64> = (aa_term).to_vec();
                        local_y.extend(indices.into_iter().zip(values.into_iter()));
                    }
                }
            }

            local_y
        }).collect();

        // Implement all phonon b and b† terms if n_pn > 1
        let y_pn_list: Vec<(usize, f64)>;

        if self.phonon.is_some() && self.pn_coupling.abs() > 1e-12 {
        let phonon_mode_ind: i32 = self.photon.photon_modes.len() as i32;
        let cos_mat_phonon = matrix_element_arrays_ref.cos_mat.get(&vec![phonon_mode_ind]).unwrap();

        y_pn_list = (0..n_pn-1).into_par_iter().flat_map(|pn_ind| {
            let sqrt_pn_plus_1 = ((pn_ind + 1) as f64).sqrt();
            let pn_coupling_factor = self.pn_coupling * sqrt_pn_plus_1;

            let mut local_y: Vec<(usize, f64)> = Vec::new();

            for ph_ind in 0..n_ph {
                let pn_dex_start = ph_ind * n_pn + pn_ind;
                let pn_exc_start = pn_dex_start + 1;
                
                // For b† term - Process pn_ind to pn_ind+1 transition
                {
                    // Get slice for phonon pn_ind at this photon level
                    let x_ph_pn = x_vec.slice(s![pn_dex_start..;stride]);
                    
                    // Calculate b_dag_term using BLAS-powered dot product
                    let b_dag_term = &cos_mat_phonon.dot(&x_ph_pn) * pn_coupling_factor;
                    
                    // Add to y at phonon pn_ind+1 and photon in ph_ind
                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| (pn_ind + 1) + ph_ind * n_pn + ex_ind * stride).collect();
                    let values: Vec<f64> = (b_dag_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
                
                // For b term - Process pn_ind+1 to pn_ind transition
                {
                    // Get slice for phonon pn_ind+1 at this photon level
                    let x_ph_pn_plus_1 = x_vec.slice(s![pn_exc_start..;stride]);
                    
                    // Calculate b_term using BLAS-powered dot product
                    let b_term = &cos_mat_phonon.dot(&x_ph_pn_plus_1) * pn_coupling_factor;
                    
                    // Add to y at phonon pn_ind and photon in ph_ind
                    let indices: Vec<usize> = (0..n_ex).map(|ex_ind| pn_ind + ph_ind * n_pn + ex_ind * stride).collect();
                    let values: Vec<f64> = (b_term).to_vec();
                    local_y.extend(indices.into_iter().zip(values.into_iter()));
                }
            }

            local_y
        }).collect();
        }
        else {
            y_pn_list = vec![(0,0.0)]; // No phonon contributions
        }

        // Combine results from photon and phonon contributions
        for (idx, val) in y_ph_list.into_iter().chain(y_pn_list.into_iter()) {
            y[idx] += val;
        }

    }
}
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

        // Calculate matrix elements using taylor expansion
        let taylor = true;

        if let Some(pn) = phonon{
            pp_off_ham = PPOffHamiltonian::new(Arc::clone(&exciton), photon.clone(), Some(pn.clone()), lm_coupling, pn_coupling, None);
            pp_off_ham.calc_matrix_element_arrays(taylor);
            pp_off_ham.calc_p_cos_matrix();

            pp_diag_ham = PPDiagHamiltonian::new(Arc::clone(&exciton), photon, Some(pn), k.clone(), lm_coupling);
            pp_diag_ham.calculate_interaction_array();
        }
        else {
            pp_off_ham = PPOffHamiltonian::new(Arc::clone(&exciton), photon.clone(), None, lm_coupling, pn_coupling, None);
            pp_off_ham.calc_matrix_element_arrays(taylor);
            pp_off_ham.calc_p_cos_matrix();

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
            self.exciton.n_rel, 
            self.exciton.parity
        );
        new_exciton.compute_eigenstates();
        
        // Update the exciton
        self.exciton = Arc::new(new_exciton);
        
        // Update Mat-Vec Hamiltonians with references to the updated exciton
        self.pp_diag_ham.exciton = self.exciton.clone();
        self.pp_diag_ham.k = new_k;
        self.pp_diag_ham.calculate_interaction_array();


        self.pp_off_ham.exciton = self.exciton.clone();
        self.pp_off_ham.calc_p_cos_matrix();
    }

}

impl Operator<f64> for PolaronPolariton {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        // Avoid unnecessary allocation and copying
        let x_arr = Array1::from_shape_fn(x.len(), |i| x[i]);

        // println!("Applying Mat-Vec");
        // println!("Applying Diag");
        // Apply diagonal part directly to y
        let diag_part = self.pp_diag_ham.apply(&x_arr);
        for i in 0..y.len() {
            y[i] = diag_part[i];
        }
        
        // println!("Applying Off-Diag");
        // Apply off-diagonal part and add to y
        self.pp_off_ham.apply(x, y);
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
            config.n_ex_eig, config.n_rel, 0);
        
        exciton.compute_eigenstates();

        // Initialize Photonic System
        let photon_modes: Vec<PhotonMode> = config.q_values.iter().map(|q| {
                PhotonMode::new(*q, Polarization::TE, 2)
            }).collect();
        let photon = PhotonicHamiltonian::new(photon_modes);

        if config.pn_n_fock > 1 {
            assert!(config.pn_n_fock > 1, "Phonon Fock space must be greater than 1 when phonon coupling is non-zero.");

            // Initialize Phononic System
            let phonon_mode = PhotonMode { eps: arr1(&[0.0,0.0]), q: config.pn_k.clone(), 
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


pub fn diagonalize_pp_sing_ex(pp_config: &PPConfig, pp: PolaronPolariton, n_eig: usize) -> (Array1<f64>, Array2<f64>) {

    // Configure ARPACK
    let config = Config::new()
        .which(arpack::Which::SA)  // Smallest algebraic eigenvalues
        .tol(1e-6)                // Tolerance
        .max_iter(1000);          // Maximum iterations
    
    let n: usize;
    let n_ph = pp_config.q_values.len() + 1;
    // Ensure phonon Fock space is valid if phonon coupling is present
    if pp.pp_off_ham.phonon.is_some() {
        assert!(pp_config.pn_n_fock > 1, "Phonon Fock space must be greater than 1 when phonon coupling is non-zero.");

        let n_pn = if pp.pp_diag_ham.phonon.is_some() {
                pp_config.pn_n_fock
            } else {1};
        n = pp.exciton.n_ex_eig * n_ph * n_pn;
    }
    else {
        n = pp.exciton.n_ex_eig * n_ph;
    }

    // Solve eigenvalue problem
    let result = arpack::eigs(pp, n,n_eig, config).unwrap();
        
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