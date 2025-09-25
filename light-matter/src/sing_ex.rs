use exciton_2d:: FullHamiltonian;
use std::{collections::HashMap, usize};
use ndarray::{arr1, Array1, Array2, s};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use plotters::prelude::*;
use arpack::{Config, Operator}; 

use crate::utils::{LMMatrixElement, LMMatrixElementArray, PhotonMode};

pub fn kron_1d(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    a.iter().flat_map(|&x| b.iter().map(move |&y| x * y)).collect::<Array1<f64>>()
}

pub trait ArrayOperator<T> {
    fn apply(&self, x: &Array1<T>) -> Array1<T>;
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
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {

        assert_eq!(x.len(), self.photon_modes.len() + 1, "Input vector length must match number of photon modes + 1.");

       x.iter().enumerate().map(|(i, &val)| {
            if i == 0 {
                0.0  // Ground state has zero photon energy
            } else {
                let ph_energy = self.photon_modes[i - 1].omega;
                ph_energy * val
            }
        }).collect::<Array1<f64>>()
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
        
        self.e_vals = Some(exciton_energies);
        self.e_vecs = Some(e_vecs);
        self.r_vecs = Some(r_vecs);
        self.parities = Some(parities);
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

pub struct PPDiagHamiltonian<'a> {
    pub exciton: &'a ExcitonicHamiltonian,
    pub photon: &'a PhotonicHamiltonian,
    pub phonon: Option<&'a PhononHamiltonian>,
    pub k: f64,
    pub int_array: Option<Array1<f64>>,
    pub coupling: f64
}

impl<'a> PPDiagHamiltonian<'a> {
    pub fn new(
        exciton: &'a ExcitonicHamiltonian, 
        photon: &'a PhotonicHamiltonian,
        phonon: Option<&'a PhononHamiltonian>, 
        k: f64,
        coupling: f64
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            k, 
            int_array: None,
            coupling
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
                + 2.0 * self.coupling * self.coupling  // A^2 term scales with coupling squared

                // q^2 (a† a)^2 term
                + q.dot(q) / 2.0 / self.exciton.exciton_ham.m   // (a† a)^2 = 1 for single photon states
            }
        }).collect::<Array1<f64>>();

        self.int_array = Some(int_array);
    }
    
}

impl ArrayOperator<f64> for PPDiagHamiltonian<'_> {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        assert!(self.int_array.is_some(), "Interaction array not computed. Call calculate_interaction_array() first.");
        let int_array = self.int_array.as_ref().unwrap();

        let exciton_part = self.exciton.form_diagonal();
        let photon_part = self.photon.form_diagonal();

        // let n_ex = exciton_part.len();
        let n_ph = photon_part.len();

        if self.phonon.is_none() {
            assert!(x.len() == self.exciton.n_ex_eig * (self.photon.photon_modes.len() + 1), 
                "Input vector length must match LM Hamiltonian dimension.");

            let result = Array1::from_iter(
                exciton_part.iter().enumerate().flat_map(|(i, &e_val)| {
                    photon_part.iter().enumerate().map(move |(j, &ph_val)| {
                        let idx = i * n_ph + j;
                        (e_val + ph_val + int_array[idx]) * x[idx]
                    })
                })
            );

            result
        }
        else {
            assert!(x.len() == self.exciton.n_ex_eig * (self.photon.photon_modes.len() + 1)
                * self.phonon.as_ref().unwrap().phonon_mode.n_fock, 
                "Input vector length must match PP Hamiltonian dimension.");

            let phonon_part = self.phonon.as_ref().unwrap().form_diagonal();
            let n_pn = phonon_part.len();

            let result = Array1::from_iter(
                x.iter().enumerate().map(|(idx, &val)| {
                    let ex_idx = idx / (n_ph * n_pn);
                    let ph_idx = (idx / n_pn) % n_ph;
                    let pn_idx = idx % n_pn;
                    let lm_idx = ex_idx * n_ph + ph_idx;

                    let e_val = exciton_part[ex_idx];
                    let ph_val = photon_part[ph_idx];
                    let pn_val = phonon_part[pn_idx];

                    (e_val + ph_val + pn_val + int_array[lm_idx]) * val
                })
            );

            result

        }
    }
}

pub struct PPOffHamiltonian<'a> {
    pub exciton: &'a ExcitonicHamiltonian,
    pub photon: &'a PhotonicHamiltonian,
    pub phonon: Option<&'a PhononHamiltonian>,
    pub lm_coupling: f64,
    pub pn_coupling: f64,
    pub matrix_element_arrays: Option<LMMatrixElementArray>,
    pub p_cos: Option<HashMap<i32, Array2<f64>>>,
}

impl<'a> PPOffHamiltonian<'a> {
    pub fn new(
        exciton: &'a ExcitonicHamiltonian, 
        photon: &'a PhotonicHamiltonian,
        phonon: Option<&'a PhononHamiltonian>, 
        lm_coupling: f64,
        pn_coupling: f64
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            lm_coupling,
            pn_coupling,
            matrix_element_arrays: None,
            p_cos: None
        }
    }

    pub fn calculate_matrix_elements(&self) -> LMMatrixElement {
        let mut bosonic_modes = self.photon.photon_modes.clone();
        if self.phonon.is_some() {
            bosonic_modes.push(self.phonon.unwrap().phonon_mode.clone());
        }

        LMMatrixElement::new()
            .populate_mat_elements(&self.exciton.r_vecs.as_ref().unwrap(), &bosonic_modes, &self.exciton.exciton_ham)

    }

    pub fn calc_matrix_element_arrays(&mut self) {
        let matrix_elements = &self.calculate_matrix_elements();

        let matrix_element_arrays = LMMatrixElementArray::new();
        matrix_element_arrays.populate_mat_elements(matrix_elements, 
            self.exciton.e_vecs.as_ref().unwrap(), 
            self.exciton.parities.as_ref().unwrap(), 
            self.exciton.exciton_ham.kappa_vectors.len());


        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state

        let mut p_cos: HashMap<i32, Array2<f64>> = HashMap::new();
        (0 .. n_ph - 1).for_each(|i| {
            let eps = &self.photon.photon_modes[i].eps;
            let q_ind_key = i as i32;

            if !p_cos.contains_key(&q_ind_key) {
                p_cos.insert(q_ind_key, 
                    self.exciton.get_p_com(eps, false)// P·q 
                    .dot(matrix_element_arrays.cos_mat.get(&vec![q_ind_key]).unwrap()) // cos(qx/2) matrix element
                );
            }
        });

        self.p_cos = Some(p_cos);
        self.matrix_element_arrays = Some(matrix_element_arrays);
    }

    fn apply_lm_int(&self, x_ph_ind: usize, r_ph_ind: usize, x_ex_ind: usize, r_ex_ind: usize, val: f64) -> f64 {
        let mut res = 0.0;
        
        // Cache these references to avoid repeated unwrapping
        let p_cos_ref = self.p_cos.as_ref().unwrap();
        let matrix_element_arrays_ref = self.matrix_element_arrays.as_ref().unwrap();
        let m = self.exciton.exciton_ham.m;
        let mu = self.exciton.exciton_ham.rel_ham.mu;
        let lm_coupling = self.lm_coupling;

        if x_ph_ind == 0 && r_ph_ind > 0 {
            // 2 P cos(qx/2) a† / M term
            let r_ph_ind_i32 = r_ph_ind as i32 - 1;
            if let Some(p_cos_matrix) = p_cos_ref.get(&r_ph_ind_i32) {
                res += p_cos_matrix[[r_ex_ind, x_ex_ind]] * lm_coupling * 2.0 / m * val;
            }

            // 2 p sin(qx/2) a† / mu term
            if let Some(p_sin_matrix) = matrix_element_arrays_ref.p_sin_mat.get(&vec![r_ph_ind_i32]) {
                res += p_sin_matrix[[r_ex_ind, x_ex_ind]] * lm_coupling * 2.0 / mu * val;
            }
        }
        else if x_ph_ind > 0 && r_ph_ind == 0 {
            // 2 P cos(qx/2) a / M term
            let x_ph_ind_i32 = x_ph_ind as i32 - 1;
            if let Some(p_cos_matrix) = p_cos_ref.get(&x_ph_ind_i32) {
                res += p_cos_matrix[[r_ex_ind, x_ex_ind]] * lm_coupling * 2.0 / m * val;
            }

            // 2 p sin(qx/2) a / mu term
            if let Some(p_sin_matrix) = matrix_element_arrays_ref.p_sin_mat.get(&vec![x_ph_ind_i32]) {
                res += p_sin_matrix[[r_ex_ind, x_ex_ind]] * lm_coupling / 2.0 / mu * val;
            }
        }
        else if x_ph_ind == r_ph_ind && r_ph_ind > 0 && r_ex_ind != x_ex_ind {
            // P·q (a† a) / 2 M term (off-diagonal terms only)
            let q = &self.photon.photon_modes[r_ph_ind - 1].q;
            let p_com = self.exciton.get_p_com(q, false);
            res += p_com[[r_ex_ind, x_ex_ind]] / 2.0 / m * val;
        }
        else if x_ph_ind != r_ph_ind && x_ph_ind > 0 && r_ph_ind > 0 {
            // A A cos((q - q') x / 2) a†_q a_q'
            let q_x = x_ph_ind as i32 - 1;
            let q_r = r_ph_ind as i32 - 1;
            let q_ind1 = q_x.max(q_r);
            let q_ind2 = q_x.min(q_r);
            let q_ind_key = vec![q_ind1, -q_ind2];

            if let Some(cos_matrix) = matrix_element_arrays_ref.cos_mat.get(&q_ind_key) {
                let eps_x = &self.photon.photon_modes[x_ph_ind - 1].eps;
                let eps_r = &self.photon.photon_modes[r_ph_ind - 1].eps;
                
                // Calculate dot product directly
                let eps_dot = eps_r[0] * eps_x[0] + eps_r[1] * eps_x[1];
                
                res += cos_matrix[[r_ex_ind, x_ex_ind]] * lm_coupling * lm_coupling * eps_dot * val;
            }
        }

        res
    }


    
}

impl ArrayOperator<f64> for PPOffHamiltonian<'_> {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {

        assert!(self.matrix_element_arrays.is_some(), "Matrix element arrays not computed. Call calc_matrix_element_arrays() first.");

        let n_ex = self.exciton.n_ex_eig;
        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state

        if self.phonon.is_none() {
            assert!(x.len() == n_ex * n_ph, 
                "Input vector length must match LM Hamiltonian dimension.");

            Array1::from_iter((0 .. x.len()).map( |r_ind| {
                let r_ex_ind = r_ind / n_ph;
                let r_ph_ind = r_ind % n_ph;

                x.iter().enumerate().map( |(x_ind, &val)| {
                    let x_ex_ind = x_ind / n_ph;
                    let x_ph_ind = x_ind % n_ph;

                    self.apply_lm_int(x_ph_ind, r_ph_ind, x_ex_ind, r_ex_ind, val)
                    
                }).sum::<f64>()
            } ))
        }
        else {
            assert!(x.len() == n_ex * n_ph * self.phonon.as_ref().unwrap().phonon_mode.n_fock, 
                "Input vector length must match PP Hamiltonian dimension.");

            let n_pn = self.phonon.as_ref().unwrap().phonon_mode.n_fock;
            let phonon_mode_ind: i32 = self.photon.photon_modes.len() as i32; // Phonon mode is last in the list

            Array1::from_iter( (0 .. x.len()).map( |r_ind| {
                let r_ex_ind =  r_ind / (n_ph * n_pn);
                let r_ph_ind = (r_ind / n_pn) % n_ph;
                let r_pn_ind =  r_ind % n_pn;

                x.iter().enumerate().map(|(x_ind, &val)| {
                    let x_ex_ind = x_ind / (n_ph * n_pn);
                    let x_ph_ind = (x_ind / n_pn) % n_ph;
                    let x_pn_ind = x_ind % n_pn;

                    if x_pn_ind == r_pn_ind {
                        if x_ph_ind == r_ph_ind && r_ex_ind != x_ex_ind{

                            // P·k (b† b) / 2 M term (off-diagonal terms only)
                            let k = &self.phonon.unwrap().phonon_mode.q;
                            let p_com = self.exciton.get_p_com(k, false);

                            p_com[[r_ex_ind, x_ex_ind]] / 2.0 / self.exciton.exciton_ham.m * val * x_pn_ind as f64  // (b† b) = n for phonon states

                            // All other LM interaction terms
                            + self.apply_lm_int(x_ph_ind, r_ph_ind, x_ex_ind, r_ex_ind, val)

                        }
                        else {
                            // No phonon off-diagonal operators, only LM interaction
                            self.apply_lm_int(x_ph_ind, r_ph_ind, x_ex_ind, r_ex_ind, val)
                        }
                    }
                    else if x_pn_ind == r_pn_ind + 1 {
                        // g (b† + b) term
                        &self.matrix_element_arrays.as_ref().unwrap().cos_mat.get(&vec![phonon_mode_ind]).unwrap()[[r_ex_ind, x_ex_ind]] *
                        self.pn_coupling * ((x_pn_ind) as f64).sqrt() * val  // b† term
                    }
                    else if x_pn_ind + 1 == r_pn_ind {
                        // sqrt(n) g (b† + b) term
                        &self.matrix_element_arrays.as_ref().unwrap().cos_mat.get(&vec![phonon_mode_ind]).unwrap()[[r_ex_ind, x_ex_ind]] *
                        self.pn_coupling * ((x_pn_ind + 1) as f64).sqrt() * val  // b term
                    }
                    else {
                        0.0
                    }

                }).sum::<f64>()
            }))
        }
    }
}



pub struct PolaronPolariton<'a> {
    pub exciton: &'a ExcitonicHamiltonian,
    pub photon: &'a PhotonicHamiltonian,
    pub phonon: Option<&'a PhononHamiltonian>,
    pub coupling: f64,
    pub n_rel: usize,
    pub n_ex_eig: usize,
    pub parity: i32,
    pub matrix_elements: Option<LMMatrixElement>,
}

impl<'a> PolaronPolariton<'a> {
    pub fn new(
        exciton: &'a ExcitonicHamiltonian, 
        photon: &'a PhotonicHamiltonian,
        phonon: Option<&'a PhononHamiltonian>, 
        coupling: f64,
        n_rel: usize,
        n_ex_eig: usize,
        parity: i32
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            coupling, 
            n_rel, 
            n_ex_eig, 
            parity,
            matrix_elements: None  // Also fixed this to use the parameter
        }
    }

}