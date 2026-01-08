use ndarray_linalg::{Norm, c64};
use scilib::math::basic::factorial;
use std::{collections::HashMap, f64::consts::PI};
use rayon::prelude::*;
use log::warn;
use ndarray::{Array1, Array2};
use crate::utils::BosonMode;

use crate::timed_println;

#[derive(Clone)]
pub struct LMMatrixElementArray {
    pub p_cos_mat: HashMap<Vec<i32>, Array2<f64>>, // Momentum Cosine matrix elements
    pub cos_mat: HashMap<Vec<i32>, Array2<f64>>, // Cosine matrix elements
    pub sin_mat: HashMap<Vec<i32>, Array2<f64>>, // Sine matrix elements
}

impl LMMatrixElementArray {
    pub fn new() -> Self {
        let   cos_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();
        let p_cos_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();
        let   sin_mat: HashMap<Vec<i32>, Array2<f64>> = HashMap::new();

        Self {
            cos_mat: cos_mat,
            sin_mat: sin_mat,
            p_cos_mat: p_cos_mat,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_mat.is_empty() && self.sin_mat.is_empty() && self.p_cos_mat.is_empty()
    }

    pub fn populate_tay_mat_elements(
        &self,
        all_q_ind: &Vec<Vec<i32>>,
        e_vecs: &Vec<Array1<f64>>,
        parities: &Array1<i32>,
        modes: &Vec<BosonMode>,
        n_com: usize,
        mu: f64,
    ) -> Self {
        // Find the number of even and odd parity states more efficiently
        let mut n_rel_even = 0;
        let mut n_rel_odd = 0;

        // timed_println!("All q_ind = {:?}", all_q_ind);

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
        let n_eigs = e_vecs.len();

        // Use capacity hints for HashMaps
        let mut cos_mat = HashMap::with_capacity(all_q_ind.len() - modes.len());
        let mut sin_mat = HashMap::with_capacity(modes.len());
        let mut p_cos_mat = HashMap::with_capacity(modes.len());

        // Pre-calculate parity indices for all eigenvectors
        let parity_indices: Vec<usize> = parities.iter().map(|&p| (p == -1) as usize).collect();

        // Get Laguerre coefficients for radial integrals
        let (lag_coeffs, states) = &self.get_laguerre_coeffs(n_rel_even + n_rel_odd);
        // let dr_coeffs = &self.get_dr_radial_poly(states);

        // Use parallel map to compute matrices for each q_inds value
        let results: Vec<(Vec<i32>, Array2<f64>, Array2<f64>)> = all_q_ind.par_iter().enumerate().map(|(q_idx, q_inds)| {
            let mut matrix_1 = Array2::zeros((n_eigs, n_eigs));
            let mut matrix_2 = Array2::zeros((n_eigs, n_eigs));

            // Keep using iterators for the main computation
            for idx1 in 0..n_eigs {
                for idx2 in 0..n_eigs {
                    let vec1 = &e_vecs[idx1];
                    let vec2 = &e_vecs[idx2];

                    let idx1_par = parity_indices[idx1];
                    let idx2_par = parity_indices[idx2];

                    // Calculate the matrix elements using iterators
                    let (element_1, element_2) = (0..n_rel_par[idx1_par])
                        .flat_map(|rel_ind1| {
                            let r_ind1 = rel_ind1 + idx1_par * n_rel_even;
                            
                            (0..n_rel_par[idx2_par]).flat_map(move |rel_ind2| {
                                let r_ind2 = rel_ind2 + idx2_par * n_rel_even;

                                let (mat_element1, mat_element2): (f64,f64);

                                // Get matrix elements
                                if q_inds.len() == 1 {
                                    let mode_idx = q_inds[0] as usize;
                                    let q = &modes[mode_idx].q;
                                    let eps = &modes[mode_idx].eps;
                                    if q_idx < modes.len(){  
                                        mat_element1 = self.get_p_cos_tayl(&states[r_ind1], &states[r_ind2], q, eps,  mu, &lag_coeffs);
                                    }
                                    else {
                                        mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], q,  mu, &lag_coeffs);
                                    }
                                    mat_element2 = self.get_sin_tayl(&states[r_ind1], &states[r_ind2], q,  mu, &lag_coeffs);
                                }
                                else {
                                    let q_diff = &modes[q_inds[0] as usize].q - &modes[q_inds[1] as usize].q;
                                    let q_sum = &modes[q_inds[0] as usize].q + &modes[q_inds[1] as usize].q;
                                    mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_diff, mu, &lag_coeffs);
                                    mat_element2 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_sum, mu, &lag_coeffs);
                                }

                                (0..n_com).map(move |com_ind| {
                                    let i1 = n_com * rel_ind1 + com_ind;
                                    let i2 = n_com * rel_ind2 + com_ind;
                                    let a1 = vec1[i1];
                                    let a2 = vec2[i2];

                                    (a1 * a2 * mat_element1, a1 * a2 * mat_element2)
                                })
                            })
                        })
                        .fold((0.0, 0.0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

                    matrix_1[[idx1, idx2]] = element_1;
                    matrix_2[[idx1, idx2]] = element_2;
                }
            }
            
            // Return tuple of q_inds and computed matrices
            (q_inds.clone(), matrix_1, matrix_2)
        }).collect();

        // Process results and populate HashMaps
        results.iter().enumerate().for_each(|(idx, (q_inds, matrix_1, matrix_2))|{
            if q_inds.len() == 1 {
                if idx < modes.len(){  
                    p_cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                else {
                    cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                sin_mat.insert(q_inds.clone(), matrix_2.clone());
            } else if q_inds.len() == 2 {
                let q_ind1 = q_inds[0].max(q_inds[1]);
                let q_ind2 = q_inds[0].min(q_inds[1]);

                cos_mat.insert(vec![q_ind1, -q_ind2], matrix_1.clone());
                cos_mat.insert(vec![q_ind1, q_ind2], matrix_2.clone());
            }
        });

        Self { cos_mat, sin_mat, p_cos_mat }
    }

    /// Evaluates overlap integral of radial wavefunctions for two states (n1,m1) and (n2,m2)
    /// multiplied by r^l 
    /// O(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) R_{n2,m2}(r)
    pub fn get_overlap(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];

        let radial = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                // \int e^{-a r} r^b dr = b! / a^(b+1)

                // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                let a = (beta_1 + beta_2) / 2.0;

                // From r^|m1| * r^i * r^|m2| * r^j * r^l dr
                let b = m1.abs() + m2.abs() + i + j + l;

                let factorial_term = factorial(b as usize) as f64;
                let denom_term = (a).powi(b + 1);

                acc2 + c1 * c2 * factorial_term / denom_term
                    * beta_1.powi(m1.abs() + i)
                    * beta_2.powi(m2.abs() + j)
            })
        });

        radial
    }

    /// Evaluates overlap integral of radial wavefunctions for two states |n1,m1> and ∂/∂r |n2,m2>
    /// multiplied by r^l 
    /// O'(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) ∂/∂r R_{n2,m2}(r)
    pub fn get_dr_overlap(
        &self,
        i1: &(i32,i32),
        i2: &(i32,i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n2 = i2.0;   
        let m2 = i2.1;
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        let mut radial = (m2.abs() as f64)/beta_2 * self.get_overlap(i1, i2, l - 1, mu, lag_coeffs);
        
        radial += self.get_overlap(i1, i2, l, mu, lag_coeffs);

        if n2 - m2.abs() - 1 >= 0 {
            let n1 = i1.0;
            let m1 = i1.1;
            let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);

            let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
            let lag_coeff2 = self.laguerre((n2 - m2.abs() - 1) as usize, (2 * m2.abs() + 1) as f64);

            radial += lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
                acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                    // \int e^{-a r} r^b dr = b! / a^(b+1)

                    // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                    let a = (beta_1 + beta_2) / 2.0;

                    // From r^|m1| * r^i * r^|m2| * r^j * r^l dr
                    let b = m1.abs() + m2.abs() + i + j + l;

                    let factorial_term = factorial(b as usize) as f64;
                    let denom_term = (a).powi(b + 1);

                    acc2 + c1 * c2 * factorial_term / denom_term
                        * beta_1.powi(m1.abs() + i)
                        * beta_2.powi(m2.abs() + j)
                })
            });
        }

        radial
    }

    /// Calculate the sine term using analytic integrals of leading Taylor series terms
    /// < n1,m1 | sin(q.r/2) | n2,m2 > ~  < n1,m1 | (q.r/2) | n2,m2 >
    pub fn get_sin_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    1  => c64::from_polar(1.0,  theta_q) * weight1 * weight2,
                    -1 => c64::from_polar(1.0, -theta_q) * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.im > 1e-10 {
            warn!("Imaginary part of angular integral is non-zero in get_sin_tayl(): {}", angular.im);
        }

        let angular = angular.re;

        if angular.abs() < 1e-10 {
            return 0.0;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        let rad_integral = self.get_overlap(i1, i2, 2, mu, lag_coeffs);

        norms * rad_integral * PI * q.norm() / 2.0 * angular
    }

    /// Calculate the cosine term using analytic integrals of leading Taylor series terms
    /// < n1,m1 | cos(q.r/2) | n2,m2 > ~  < n1,m1 | 1 - (q.r/2)^2/2 | n2,m2 >
    pub fn get_cos_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let overlap = if i1.0 == i2.0 && i1.1 == i2.1 {
            1.0
        } else {
            0.0
        };

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    0  => c64::new(1.0, 0.0) * weight1 * weight2,
                    2  => c64::from_polar(1.0, -2.0 * theta_q) * weight1 * weight2 / 2.0,
                    -2 => c64::from_polar(1.0,  2.0 * theta_q) * weight1 * weight2 / 2.0,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.im > 1e-10 {
            warn!("Imaginary part of angular integral is non-zero in get_cos_tayl(): {}", angular.im);
        }

        let angular = angular.re;

        if angular.abs() < 1e-10 {
            return overlap;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        let rad_integral = self.get_overlap(i1, i2, 3, mu, lag_coeffs);

        overlap - angular * norms * rad_integral
    }

    /// Matrix element: i < n1,m1 | (epsilon·∇) cos(q·r/2) | n2,m2 >
    /// ≈ < n1,m1 | (epsilon·∇) (1 - (q·r/2)^2/2) | n2,m2 > (Second order Taylor expansion)
    pub fn get_p_cos_tayl(
        &self,
        i1: &(i32,i32), // (n, l) of real wavefunction |l| = |m|
        i2: &(i32,i32),
        q: &Array1<f64>,
        eps: &Array1<f64>,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let m1_vec = self.real_l_to_complex_m(i1.1);
        let m2_vec = self.real_l_to_complex_m(i2.1);

        let q2 = q.map(|val| val*val).sum();
        let theta_q = f64::atan2(q[1], q[0]);
        let theta_eps = f64::atan2(eps[1], eps[0]);

        let integral = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;

                let j1 = & (i1.0, *m1);
                let j2 = & (i2.0, *m2);
                
                match diff {
                    3  => c64::from_polar(1.0,  theta_eps + 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * ( (*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    -3 => c64::from_polar(1.0, -theta_eps - 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * (-(*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    1  => c64::from_polar(1.0,  theta_eps)
                            * (
                                -(*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                - q2 / 32.0 * (
                                    -(*m2 as f64) * (2.0 - c64::from_polar(1.0,  2.0 * (theta_q - theta_eps))) 
                                    * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    + (2.0 + c64::from_polar(1.0,  2.0 * (theta_q - theta_eps)))
                                    * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                                )
                            )
                            * weight1 * weight2,
                    -1 => c64::from_polar(1.0,  theta_eps)
                            * (
                                 (*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                                + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                                - q2 / 32.0 * (
                                     (*m2 as f64) * (2.0 - c64::from_polar(1.0, -2.0 * (theta_q - theta_eps))) 
                                    * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                    + (2.0 + c64::from_polar(1.0, -2.0 * (theta_q - theta_eps)))
                                    * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                                )
                            )
                            * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if integral.im > 1e-10 {
            warn!("Imaginary part of integral is non-zero in get_p_cos_tayl(): {}", integral.im);
        }

        let integral = integral.re;

        if integral.abs() < 1e-10 {
            return 0.0;
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);

        norms * integral
    }

    fn real_l_to_complex_m(&self, l: i32) -> Vec<(i32, c64)>{

        match l {
            0 => vec![(0, c64::new(1.0, 0.0))],
            l if l > 0 => vec![
                ( l, c64::new(1.0 / 2_f64.sqrt(), 0.0)), 
                (-l, c64::new(1.0 / 2_f64.sqrt(), 0.0))],
            l if l < 0 => vec![
                ( l, c64::new(0.0,  1.0 / 2_f64.sqrt())), 
                (-l, c64::new(0.0, -1.0 / 2_f64.sqrt()))],
            _ => panic!("Invalid l value in real_l_to_complex_m")
        }
    }

    fn norm_constant(&self, (n,m): &(i32,i32), mu: f64) -> f64 {

        let beta = mu * 2.0 / (*n as f64 + 0.5);

        beta
        * (factorial(*n as usize - m.abs() as usize) as f64
            / (2. * PI
                * factorial(*n as usize + m.abs() as usize) as f64
                * (2. * *n as f64 + 1.)))
            .sqrt()
    }

    fn _test_psi_norm(&self, states: &Vec<(i32, i32)>, mu: f64) {
        states.iter().for_each(|&(n,m)| {
            let n = n;
            let m = m.abs();

            let beta = mu / 8. / PI / PI / (n as f64 + 0.5);
            let norm = beta
                * (factorial(n as usize - m as usize) as f64
                    / (2.
                        * PI
                        * factorial((n + m) as usize) as f64
                        * (2. * n as f64 + 1.)))
                    .sqrt();

            let lag_coeffs = self.laguerre((n - m) as usize, 2. * m as f64);

            // Calculate integral of |R_{n,m}(rho)|^2 rho d rho
            let integral = lag_coeffs.iter().fold(0.0, |acc1, (i, c1)| {
                acc1 + lag_coeffs.iter().fold(0.0, |acc2, (j, c2)| {
                    // \int e^{-a r} r^b dr = b! / a^(b+1)

                    // From e^{-beta x/2} * e^{-beta x/2} = e^{-beta x}
                    let a = beta;

                    // From r^|m| * r^i * r^|m| * r^j * r dr
                    let b = 2 * m + i + j + 1;

                    let factorial_term = factorial(b as usize) as f64;
                    let denom_term = (a).powi(b + 1);

                    acc2 + c1 * c2 * factorial_term / denom_term
                        * beta.powi(2 * m + i + j)
                })
            });

            let psi_norm = 2. * PI * norm.powi(2) * integral;

            timed_println!("State (n={},m={}): Norm = {}", n, m, psi_norm);
        });
    }

    pub fn get_x(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        // Calculate angular component
        let diff = (m1 -m2).abs();
        let sum = (m1 + m2).abs();

        // Only jumps in m by ±1 allowed
        if diff & sum != 1 {
            return 0.0
        }
        
        let m_sign = (m1 * m2).signum();

        if m_sign == 0 && (m1 + m2) < 0 {
            return 0.0;
        }

        let angular = match m_sign {
            0  => PI * 2.0_f64.sqrt(),
            1  => {
                let mut result = 0.0;
                if diff == 1 { result += PI};
                if sum  == 1 { result += PI * m1.signum() as f64};
                result
            },
            _ => 0.0
        };

        // Calculate radial component
        let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);
        let norm_1 = beta_1
            * (factorial(n1 as usize - m1.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n1 as usize + m1.abs() as usize) as f64
                    * (2. * n1 as f64 + 1.)))
                .sqrt();
        let norm_2 = beta_2
            * (factorial(n2 as usize - m2.abs() as usize) as f64
                / (2.
                    * PI
                    * factorial(n2 as usize + m2.abs() as usize) as f64
                    * (2. * n2 as f64 + 1.)))
                .sqrt();

        let prefactor = norm_1 * norm_2;

        let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
        let lag_coeff2 = &lag_coeffs[&(n2, m2.abs())];

        // timed_println!("n1 = {}, m1 = {}, n2 = {}, m2 = {}", n1,m1,n2,m2);
        // timed_println!("Lag1 = {:?}", lag_coeff1);
        // timed_println!("Lag2 = {:?}", lag_coeff2);

        let radial = lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
            acc1 + lag_coeff2.iter().fold(0.0, |acc2, (j, c2)| {
                // \int e^{-a r} r^b dr = b! / a^(b+1)
                // timed_println!("j = {}", j);

                // From e^{-beta1 x/2} * e^{-beta2 x/2} = e^{-(beta1 + beta2) x/2}
                let a = (beta_1 + beta_2) / 2.0;

                // From r^|m1| * r^i * r^|m2| * r^j * (x.r) * r dr
                let b = m1.abs() + m2.abs() + i + j + 2;

                let factorial_term = factorial(b as usize) as f64;
                let denom_term = (a).powi(b + 1);

                acc2 + c1 * c2 * factorial_term / denom_term
                    * beta_1.powi(m1.abs() + i)
                    * beta_2.powi(m2.abs() + j)
            })
        }) * prefactor;

        angular * radial
    }

    pub fn get_laguerre_coeffs(&self, n_rel: usize) -> (HashMap<(i32, i32), Vec<(i32, f64)>>, Vec<(i32, i32)>) {
        // Generate all states (n,m) and sort by energy
        let mut states = Vec::new();
        let mut n: i32 = 0;
        while states.len() < n_rel {
            for m in -n..=n {
                if n >= m.abs() {
                    // Valid state condition - also check Laguerre order
                    if (n - m.abs()) as usize <= 7 {
                        states.push((n, m));
                    } else {
                        warn!("Skipping state (n={},m={}) because Laguerre polynomials of order >7 are not implemented", n, m);
                    }
                }
            }
            n += 1;
        }

        // Keep only the first n_rel states
        states.truncate(n_rel);

        let mut lag_coeffs: HashMap<(i32, i32), Vec<(i32, f64)>> = HashMap::new();

        for &(n, m) in &states {
            lag_coeffs.insert((n, m), self.laguerre((n - m.abs()) as usize, 2. * m.abs() as f64));
        }

            // Ensure all necessary keys are present
        for &(n1, m1) in &states {
            if !lag_coeffs.contains_key(&(n1, m1.abs())) {
                panic!("Missing key ({},{})", n1, m1.abs());
            }
        }

        (lag_coeffs, states)
    }

    fn laguerre(&self, order: usize, alpha: f64) -> Vec<(i32, f64)> {
        let coeffs: Vec<(i32, f64)>;

        match order {
            0 => {
                coeffs = vec![(0, 1.0)];
            }
            1 => {
                coeffs = vec![(0, alpha + 1.0), (1, -1.0)];
            }
            2 => {
                coeffs = vec![
                    (0, 0.5 * (alpha + 2.0) * (alpha + 1.0)),
                    (1,      -(alpha + 2.0)),
                    (2, 0.5),
                ];
            }
            3 => {
                coeffs = vec![
                    (0,(1.0 / 6.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -0.5       * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  0.5       * (alpha + 3.0)),
                    (3,-(1.0 / 6.0)),
                ];
            }
            4 => {
                coeffs = vec![
                    (0, (1.0 / 24.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 6.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,         0.25 * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 6.0) * (alpha + 4.0)),
                    (4, 1.0 / 24.0),
                ];
            }
            5 => {
                coeffs = vec![
                    (0, (1.0 / 120.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 24.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 24.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0)),
                ];
            }
            6 => {
                coeffs = vec![
                    (0,  (1.0 / 720.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 120.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3,  -(1.0 / 36.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0) * (alpha + 6.0)),
                    (6, 1.0 / 720.0),
                ];
            }
            7 => {
                coeffs = vec![
                    (0, (1.0 / 5040.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 720.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0)),
                    (6,  (1.0 / 720.0) * (alpha + 7.0)),
                    (7, -(1.0 / 5040.0)),
                ];
            }
            _ => {
                panic!("Laguerre polynomials of order >7 not implemented yet!")
            }
        }

        coeffs
    }

}