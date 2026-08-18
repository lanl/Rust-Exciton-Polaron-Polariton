use super::builder::LMMatrixElementArray;
use super::keys::{ModeKey, ModeTask};

use crate::utils::BosonMode;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, c64};
use num_traits::Zero;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

impl LMMatrixElementArray {
    /// ------------------------------------------------------------------------
    /// NEW: Compute Taylor‑expanded matrix elements in the *relative* basis only.
    /// Returns three HashMaps keyed by the q‑index vector:
    ///   - cos_mat   : <n₁,m₁|cos(q·r/2)|n₂,m₂>
    ///   - sin_mat   : <n₁,m₁|sin(q·r/2)|n₂,m₂>
    ///   - p_cos_mat : i< n₁,m₁ | (ε·∇)cos(q·r/2) | n₂,m₂ >
    /// Each matrix has shape (n_rel, n_rel) where n_rel = number of relative states.
    /// ------------------------------------------------------------------------
    pub fn compute_relative_tay_mat_elements(
        &self,
        all_q_ind: &Vec<ModeTask>,
        modes: &Vec<BosonMode>,
        n_rel: usize,
        mu: f64,
    ) -> (
        HashMap<ModeKey, Array2<c64>>,
        HashMap<ModeKey, Array2<c64>>,
        HashMap<ModeKey, Array2<c64>>,
    ) {
        let (lag_coeffs, states) = self.get_laguerre_coeffs(n_rel);

        let from_real = false;

        let mut cos_mat = HashMap::with_capacity(all_q_ind.len());
        let mut sin_mat = HashMap::with_capacity(modes.len());
        let mut p_cos_mat = HashMap::with_capacity(modes.len());

        let results: Vec<(ModeTask, Array2<c64>, Array2<c64>)> = all_q_ind
            .par_iter()
            .map(|task| {
                let mut matrix1 = Array2::zeros((n_rel, n_rel));
                let mut matrix2 = Array2::zeros((n_rel, n_rel));

                for (i1, &(n1, m1)) in states.iter().enumerate() {
                    for (i2, &(n2, m2)) in states.iter().enumerate() {
                        let (elem1, elem2) = match *task {
                            ModeTask::PhotonSingle { mode } => {
                                let q = &modes[mode].q;
                                let eps = &modes[mode].eps;

                                let p_cos = self.get_p_cos_tayl(
                                    &(n1, m1),
                                    &(n2, m2),
                                    q,
                                    eps,
                                    mu,
                                    from_real,
                                    &lag_coeffs,
                                );

                                let sin = self.get_sin_tayl(
                                    &(n1, m1),
                                    &(n2, m2),
                                    q,
                                    mu,
                                    from_real,
                                    &lag_coeffs,
                                );

                                (p_cos, sin)
                            }

                            ModeTask::PhotonPair { q1, q2 } => {
                                let q_vec1 = &modes[q1].q;
                                let q_vec2 = &modes[q2].q;

                                let q_dif = q_vec1 - q_vec2;
                                let q_sum = q_vec1 + q_vec2;

                                let cos_dif = self.get_cos_tayl(
                                    &(n1, m1),
                                    &(n2, m2),
                                    &q_dif,
                                    mu,
                                    from_real,
                                    &lag_coeffs,
                                );

                                let cos_sum = self.get_cos_tayl(
                                    &(n1, m1),
                                    &(n2, m2),
                                    &q_sum,
                                    mu,
                                    from_real,
                                    &lag_coeffs,
                                );

                                (cos_dif, cos_sum)
                            }
                        };

                        matrix1[[i1, i2]] = elem1;
                        matrix2[[i1, i2]] = elem2;
                    }
                }

                let real_mat1 = self.to_real_rel_basis(&matrix1);
                let real_mat2 = self.to_real_rel_basis(&matrix2);

                (*task, real_mat1, real_mat2)
            })
            .collect();

        for (task, mat_1, mat_2) in results {
            match task {
                ModeTask::PhotonSingle { mode } => {
                    let key = ModeKey::single(mode);

                    p_cos_mat.insert(key, mat_1);
                    sin_mat.insert(key, mat_2);
                }

                ModeTask::PhotonPair { q1, q2 } => {
                    cos_mat.insert(ModeKey::diff(q1, q2), mat_1);
                    cos_mat.insert(ModeKey::sum(q1, q2), mat_2);
                }
            }
        }

        (cos_mat, sin_mat, p_cos_mat)
    }

    // Only returns the sine term for phonon modes
    pub fn phonon_mat_elements(
        &self,
        e_vecs: &Array2<c64>,
        modes: &Vec<BosonMode>,
        mu: f64,
        n_com: usize,
    ) -> Vec<Array2<c64>> {
        // let n_eigs = e_vecs.len();
        let n_rel = e_vecs.nrows() / n_com;

        // Get Laguerre coefficients for radial integrals
        let (lag_coeffs, states) = &self.get_laguerre_coeffs(n_rel);

        // Use parallel map to compute matrices for each q_inds value
        modes
            .par_iter()
            .map(|mode| {
                let mut rel_matrix = Array2::zeros((n_rel, n_rel));

                for (i1, &(n1, m1)) in states.iter().enumerate() {
                    for (i2, &(n2, m2)) in states.iter().enumerate() {
                        rel_matrix[[i1, i2]] = self.get_sin_tayl(
                            &(n1, m1),
                            &(n2, m2),
                            &mode.q,
                            mu,
                            false,
                            &lag_coeffs,
                        );
                    }
                }

                self.to_exciton_basis(&rel_matrix, e_vecs, n_com)

                // let mut matrix = Array2::zeros((n_eigs, n_eigs));

                // // Keep using iterators for the main computation
                // for idx1 in 0..n_eigs {
                //     for idx2 in 0..n_eigs {
                //         let vec1 = &e_vecs[idx1];
                //         let vec2 = &e_vecs[idx2];

                //         // Calculate the matrix elements using iterators
                //         let element = (0..n_rel)
                //             .flat_map(|r_ind1| {
                //                 (0..n_rel).flat_map(move |r_ind2| {
                //                     let q = &mode.q;
                //                     let mat_element = self.get_sin_tayl(&states[r_ind1], &states[r_ind2], q,  mu, true, &lag_coeffs);

                //                     (0..n_com).map(move |com_ind| {
                //                         let i1 = n_com * r_ind1 + com_ind;
                //                         let i2 = n_com * r_ind2 + com_ind;
                //                         let a1 = vec1[i1];
                //                         let a2 = vec2[i2];

                //                         a1.conj() * a2 * mat_element
                //                     })
                //                 })
                //             })
                //             .sum();

                //         matrix[[idx1, idx2]] = element;
                //     }
                // }

                // // Return tuple of q_inds and computed matrices
                // matrix
            })
            .collect()
    }

    /// Calculate the sine term using analytic integrals of leading Taylor series terms
    ///  n1,m1 | sin(q.r/2) | n2,m2 > ~   n1,m1 | (q.r/2) | n2,m2 >
    pub fn get_sin_tayl(
        &self,
        i1: &(i32, i32),
        i2: &(i32, i32),
        q: &Array1<f64>,
        mu: f64,
        from_real: bool, // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {
        // Choose angular basis according to the flag
        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec
            .iter()
            .flat_map(|(m1, weight1)| {
                m2_vec.iter().map(move |(m2, weight2)| {
                    let diff = m1 - m2;

                    match diff {
                        1 => c64::from_polar(1.0, theta_q) * weight1 * weight2,
                        -1 => c64::from_polar(1.0, -theta_q) * weight1 * weight2,
                        _ => c64::new(0.0, 0.0),
                    }
                })
            })
            .sum::<c64>();

        if angular.norm() < 1e-10 {
            return c64::zero();
        }

        let norms = self.norm_constant(i1, mu) * self.norm_constant(i2, mu);
        let rad_integral = self.get_overlap(i1, i2, 2, mu, lag_coeffs);

        c64::new(norms * rad_integral * PI * q.norm() / 2.0, 0.0) * angular
    }

    /// Calculate the cosine term using analytic integrals of leading Taylor series terms
    ///  n1,m1 | cos(q.r/2) | n2,m2 > ~   n1,m1 | 1 - (q.r/2)^2/2 | n2,m2 >
    pub fn get_cos_tayl(
        &self,
        i1: &(i32, i32),
        i2: &(i32, i32),
        q: &Array1<f64>,
        mu: f64,
        from_real: bool, // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {
        let overlap = if i1.0 == i2.0 && i1.1 == i2.1 {
            1.0
        } else {
            0.0
        };

        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec
            .iter()
            .flat_map(|(m1, weight1)| {
                m2_vec.iter().map(move |(m2, weight2)| {
                    let diff = m1 - m2;

                    match diff {
                        0 => c64::new(1.0, 0.0) * weight1 * weight2,
                        2 => c64::from_polar(1.0, -2.0 * theta_q) * weight1 * weight2 / 2.0,
                        -2 => c64::from_polar(1.0, 2.0 * theta_q) * weight1 * weight2 / 2.0,
                        _ => c64::new(0.0, 0.0),
                    }
                })
            })
            .sum::<c64>();

        if angular.norm() < 1e-10 {
            return c64::new(overlap, 0.0);
        }

        let norms = self.norm_constant(i1, mu) * self.norm_constant(i2, mu);
        let rad_integral = self.get_overlap(i1, i2, 3, mu, lag_coeffs);
        let q2 = q.dot(q);

        // The angular factor computed above is normalized so that the explicit
        // angular integral contributes a factor of π. Therefore the Taylor term is
        //
        //   - (|q|^2 / 8) * [π * angular] * radial_overlap,
        //
        // which gives the prefactor π |q|^2 / 8 below.
        c64::new(overlap, 0.0) - angular * c64::new(norms * rad_integral * PI * q2 / 8.0, 0.0)
    }

    /// Matrix element: i  n1,m1 | (epsilon·∇) cos(q·r/2) | n2,m2 >
    /// ≈  n1,m1 | (epsilon·∇) (1 - (q·r/2)^2/2) | n2,m2 > (Second order Taylor expansion)
    pub fn get_p_cos_tayl(
        &self,
        i1: &(i32, i32),
        i2: &(i32, i32),
        q: &Array1<f64>,
        eps: &Array1<f64>,
        mu: f64,
        from_real: bool, // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {
        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let q2 = q.map(|val| val * val).sum();
        let theta_q = f64::atan2(q[1], q[0]);
        let theta_eps = f64::atan2(eps[1], eps[0]);

        let integral = m1_vec
            .iter()
            .flat_map(|(m1, weight1)| {
                m2_vec.iter().map(move |(m2, weight2)| {
                    let diff = m2 - m1;

                    let j1 = &(i1.0, *m1);
                    let j2 = &(i2.0, *m2);

                    match diff {
                        3 => {
                            c64::from_polar(1.0, theta_eps + 2.0 * theta_q) * q2 * PI / 32.0
                                * (-(*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                                * weight1
                                * weight2
                        }
                        -3 => {
                            c64::from_polar(1.0, -theta_eps - 2.0 * theta_q) * q2 * PI / 32.0
                                * ((*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                                * weight1
                                * weight2
                        }
                        1 => {
                            c64::from_polar(PI, theta_eps)
                                * ((*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                    + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                    - q2 / 32.0
                                        * ((*m2 as f64)
                                            * (2.0
                                                - c64::from_polar(
                                                    1.0,
                                                    2.0 * (theta_q - theta_eps),
                                                ))
                                            * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                            + (2.0
                                                + c64::from_polar(
                                                    1.0,
                                                    2.0 * (theta_q - theta_eps),
                                                ))
                                                * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)))
                                * weight1
                                * weight2
                        }
                        -1 => {
                            c64::from_polar(PI, -theta_eps)
                                * (-(*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                    + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                    - q2 / 32.0
                                        * (-(*m2 as f64)
                                            * (2.0
                                                - c64::from_polar(
                                                    1.0,
                                                    -2.0 * (theta_q - theta_eps),
                                                ))
                                            * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                            + (2.0
                                                + c64::from_polar(
                                                    1.0,
                                                    -2.0 * (theta_q - theta_eps),
                                                ))
                                                * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)))
                                * weight1
                                * weight2
                        }
                        _ => c64::new(0.0, 0.0),
                    }
                })
            })
            .sum::<c64>();

        if integral.norm() < 1e-10 {
            return c64::zero();
        }

        let norms = self.norm_constant(i1, mu) * self.norm_constant(i2, mu);
        c64::new(norms, 0.0) * integral
    }
}
