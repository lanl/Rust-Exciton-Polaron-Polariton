use exciton_2d:: FullHamiltonian;
use std::{collections::HashMap, usize};
use ndarray::{arr1, Array1, Array2, s};
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use plotters::prelude::*;
use arpack::{Config, Operator}; 

use crate::utils::{PhotonMode, LMMatrixElement};

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
}

impl ExcitonicHamiltonian {
    pub fn new(exciton_ham: FullHamiltonian, n_ex_eig: usize, n_rel: usize, parity: i32) -> Self {
        Self { exciton_ham, n_ex_eig, n_rel, parity, e_vals: None, e_vecs: None, r_vecs: None}
    }

    pub fn compute_eigenstates(&mut self) {
        let (exciton_energies, e_vecs, _, r_vecs, _) 
            = self.exciton_ham.diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig, self.parity);
        
        self.e_vals = Some(exciton_energies);
        self.e_vecs = Some(e_vecs);
        self.r_vecs = Some(r_vecs);
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
    pub coupling: f64,
    pub matrix_elements: Option<LMMatrixElement>,
}

impl<'a> PPOffHamiltonian<'a> {
    pub fn new(
        exciton: &'a ExcitonicHamiltonian, 
        photon: &'a PhotonicHamiltonian,
        phonon: Option<&'a PhononHamiltonian>, 
        coupling: f64
    ) -> Self {
        Self { 
            exciton, 
            photon, 
            phonon, 
            coupling,
            matrix_elements: None  // Also fixed this to use the parameter
        }
    }

    pub fn calculate_matrix_elements(&mut self) {
        let mut bosonic_modes = self.photon.photon_modes.clone();
        if self.phonon.is_some() {
            bosonic_modes.push(self.phonon.unwrap().phonon_mode.clone());
        }

        let matrix_calc = LMMatrixElement::new()
            .populate_mat_elements(&self.exciton.r_vecs.as_ref().unwrap(), &bosonic_modes, &self.exciton.exciton_ham);

        self.matrix_elements = Some(matrix_calc);
    }


    
}

impl ArrayOperator<f64> for PPOffHamiltonian<'_> {
    fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        assert!(self.matrix_elements.is_some(), "Matrix elements not computed. Call calculate_matrix_elements() first.");
        let matrix_elements = self.matrix_elements.as_ref().unwrap();

        let n_ex = self.exciton.n_ex_eig;
        let n_ph = self.photon.photon_modes.len() + 1; // +1 for ground state

        if self.phonon.is_none() {
            assert!(x.len() == n_ex * n_ph, 
                "Input vector length must match LM Hamiltonian dimension.");

            let result = Array1::from_iter((0 .. x.len()).map( |r_ind| {
                let r_ex_ind = r_ind / n_ph;
                let r_ph_ind = r_ind % n_ph;

                x.iter().enumerate().map( |(x_ind, &val)| {
                    let x_ex_ind = x_ind / n_ph;
                    let x_ph_ind = x_ind % n_ph;

                    let x_mode = if x_ph_ind == 0 { None } else { Some(&self.photon.photon_modes[x_ph_ind - 1]) };

                    let mut res = 0.0;

                    if x_ph_ind == 0 && r_ph_ind > 0 {

                        // 2 P cos(qx/2) a† / 2 M term
                        // res += {
                        //     let q = &self.photon.photon_modes[r_ph_ind - 1].q;
                        //     let p_com = self.exciton.get_p_com(q, false);
                        //     let mat_val = matrix_elements.cos_mat.get(&(r_ex_ind, x_ex_ind, r_ph_ind - 1)).unwrap_or(&0.0);
                        //     p_com[[r_ex_ind, x_ex_ind]] * (q[0] * self.exciton.exciton_ham.a0 / 2.0).cos() 
                        //         / 2.0 / self.exciton.exciton_ham.m * val // a† = 1 for single photon states
                        // }


                    }
                    else if x_ph_ind == r_ph_ind && r_ph_ind > 0 {
                        // P·q (a† a) / 2 M term (off-diagonal terms only)
                        res += if r_ex_ind != x_ex_ind {
                            let q = &self.photon.photon_modes[r_ph_ind - 1].q;
                            let p_com = self.exciton.get_p_com(q, false);
                            p_com[[r_ex_ind, x_ex_ind]] / 2.0 / self.exciton.exciton_ham.m * val // (a† a) = 1 for single photon states
                        } else { 0.0 };
                    }
                    else {
                        res = 0.0;
                    }



                    res
                    
                }).sum::<f64>()
            } ));

            result
        }
        else {
        //     assert!(x.len() == n_ex * n_ph * self.phonon.as_ref().unwrap().phonon_mode.n_fock, 
        //         "Input vector length must match PP Hamiltonian dimension.");

        //     let n_pn = self.phonon.as_ref().unwrap().phonon_mode.n_fock;

        //     let result = Array1::from_iter(
        //         x.iter().enumerate().map(|(idx, &val)| {
        //             let ex_idx = idx / (n_ph * n_pn);
        //             let ph_idx = (idx / n_pn) % n_ph;
        //             let pn_idx = idx % n_pn;

        //             (0..n_ex).flat_map(|k| {
        //                 (0..n_ph).flat_map(move |l| {
        //                     (0..n_pn).map(move |m| {
        //                         let j_op = if ph_idx == 0 { 0 } else { ph_idx - 1 }; // Photon annihilation
        //                         let l_op = if l == 0 { 0 } else { l - 1 }; // Photon creation
        //                         let me = matrix_elements.get_element(ex_idx, j_op, k, l_op);
        //                         let pn_factor = if pn_idx == m + 1 {
        //                             (m + 1) as f64.sqrt()  // Phonon annihilation
        //                         } else if pn_idx + 1 == m {
        //                             (pn_idx + 1) as f64.sqrt()  // Phonon creation
        //                         } else {
        //                             0.0
        //                         };
        //                         me * pn_factor * val
        //                     })
        //                 })
        //             }).sum()
        //         })
        //     );
            panic!("Off-diagonal operator with phonons not implemented yet.");
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

    pub fn calculate_matrix_elements(&mut self) {
        let mut bosonic_modes = self.photon.photon_modes.clone();
        if self.phonon.is_some() {
            bosonic_modes.push(self.phonon.unwrap().phonon_mode.clone());
        }

        let matrix_calc = LMMatrixElement::new()
            .populate_mat_elements(&self.exciton.r_vecs.as_ref().unwrap(), &bosonic_modes, &self.exciton.exciton_ham);

        self.matrix_elements = Some(matrix_calc);
    }
}