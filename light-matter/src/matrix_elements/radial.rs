use super::builder::LMMatrixElementArray;

use crate::timed_println;
use log::warn;
use ndarray_linalg::c64;
use scilib::math::basic::factorial;
use std::collections::HashMap;
use std::f64::consts::PI;

impl LMMatrixElementArray {
    /// Evaluates overlap integral of radial wavefunctions for two states (n1,m1) and (n2,m2)
    /// multiplied by r^l
    /// O(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) R_{n2,m2}(r)
    pub fn get_overlap(
        &self,
        (n1, m1): &(i32, i32),
        (n2, m2): &(i32, i32),
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
        i1: &(i32, i32),
        i2: &(i32, i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n2 = i2.0;
        let m2 = i2.1;
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        // We factor out a final beta_2 at the end. Therefore the first two terms
        // are written as |m|/beta_2 * O(l-1) - 1/2 * O(l), and multiplication by
        // beta_2 gives |m| O(l-1) - beta_2/2 O(l), as required by
        // dR/dr = [|m|/r - beta_2/2]R - beta_2 * L' contribution.

        let mut radial =
            (m2.abs() as f64) / beta_2 * self.get_overlap(i1, i2, l - 1, mu, lag_coeffs);

        radial -= 0.5 * self.get_overlap(i1, i2, l, mu, lag_coeffs);

        if n2 - m2.abs() - 1 >= 0 {
            let n1 = i1.0;
            let m1 = i1.1;
            let beta_1 = mu * 2.0 / (n1 as f64 + 0.5);

            let lag_coeff1 = &lag_coeffs[&(n1, m1.abs())];
            let lag_coeff2 = self.laguerre((n2 - m2.abs() - 1) as usize, (2 * m2.abs() + 1) as f64);

            radial -= lag_coeff1.iter().fold(0.0, |acc1, (i, c1)| {
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

        radial * beta_2
    }

    pub(crate) fn real_l_to_complex_m(&self, l: i32) -> Vec<(i32, c64)> {
        let inv_sqrt2 = 1.0 / 2_f64.sqrt();

        match l {
            0 => vec![(0, c64::new(1.0, 0.0))],

            l if l > 0 => vec![
                (l, c64::new(inv_sqrt2, 0.0)),
                (-l, c64::new(inv_sqrt2, 0.0)),
            ],

            l if l < 0 => {
                let a = -l;
                vec![(a, -c64::i() * inv_sqrt2), (-a, c64::i() * inv_sqrt2)]
            }

            _ => unreachable!(),
        }
    }

    pub(crate) fn norm_constant(&self, (n, m): &(i32, i32), mu: f64) -> f64 {
        let beta = mu * 2.0 / (*n as f64 + 0.5);

        beta * (factorial(*n as usize - m.abs() as usize) as f64
            / (2. * PI * factorial(*n as usize + m.abs() as usize) as f64 * (2. * *n as f64 + 1.)))
            .sqrt()
    }

    pub(crate) fn _test_psi_norm(&self, states: &Vec<(i32, i32)>, mu: f64) {
        states.iter().for_each(|&(n, m)| {
            let n = n;
            let m = m.abs();

            let beta = mu / 8. / PI / PI / (n as f64 + 0.5);
            let norm = beta
                * (factorial(n as usize - m as usize) as f64
                    / (2. * PI * factorial((n + m) as usize) as f64 * (2. * n as f64 + 1.)))
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

                    acc2 + c1 * c2 * factorial_term / denom_term * beta.powi(2 * m + i + j)
                })
            });

            let psi_norm = 2. * PI * norm.powi(2) * integral;

            timed_println!("State (n={},m={}): Norm = {}", n, m, psi_norm);
        });
    }

    pub fn get_x(
        &self,
        (n1, m1): &(i32, i32),
        (n2, m2): &(i32, i32),
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;
        let m2 = *m2;

        // Calculate angular component
        let diff = (m1 - m2).abs();
        let sum = (m1 + m2).abs();

        // Only jumps in m by ±1 allowed
        if diff != 1 && sum != 1 {
            return 0.0;
        }

        let m_sign = (m1 * m2).signum();

        if m_sign == 0 && (m1 + m2) < 0 {
            return 0.0;
        }

        let angular = match m_sign {
            0 => PI * 2.0_f64.sqrt(),
            1 => {
                let mut result = 0.0;
                if diff == 1 {
                    result += PI
                };
                if sum == 1 {
                    result += PI * m1.signum() as f64
                };
                result
            }
            _ => 0.0,
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

    pub fn get_laguerre_coeffs(
        &self,
        n_rel: usize,
    ) -> (HashMap<(i32, i32), Vec<(i32, f64)>>, Vec<(i32, i32)>) {
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
                        warn!(
                            "Skipping state (n={},m={}) because Laguerre polynomials of order >7 are not implemented",
                            n, m
                        );
                    }
                }
            }
            n += 1;
        }

        // Keep only the first n_rel states
        states.truncate(n_rel);

        let mut lag_coeffs: HashMap<(i32, i32), Vec<(i32, f64)>> = HashMap::new();

        for &(n, m) in &states {
            lag_coeffs.insert(
                (n, m),
                self.laguerre((n - m.abs()) as usize, 2. * m.abs() as f64),
            );
        }

        // Ensure all necessary keys are present
        for &(n1, m1) in &states {
            if !lag_coeffs.contains_key(&(n1, m1.abs())) {
                panic!("Missing key ({},{})", n1, m1.abs());
            }
        }

        (lag_coeffs, states)
    }

    pub(crate) fn laguerre(&self, order: usize, alpha: f64) -> Vec<(i32, f64)> {
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
                    (1, -(alpha + 2.0)),
                    (2, 0.5),
                ];
            }
            3 => {
                coeffs = vec![
                    (
                        0,
                        (1.0 / 6.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0),
                    ),
                    (1, -0.5 * (alpha + 3.0) * (alpha + 2.0)),
                    (2, 0.5 * (alpha + 3.0)),
                    (3, -(1.0 / 6.0)),
                ];
            }
            4 => {
                coeffs = vec![
                    (
                        0,
                        (1.0 / 24.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0)
                            * (alpha + 1.0),
                    ),
                    (
                        1,
                        -(1.0 / 6.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0),
                    ),
                    (2, 0.25 * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 6.0) * (alpha + 4.0)),
                    (4, 1.0 / 24.0),
                ];
            }
            5 => {
                coeffs = vec![
                    (
                        0,
                        (1.0 / 120.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0)
                            * (alpha + 1.0),
                    ),
                    (
                        1,
                        -(1.0 / 24.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0),
                    ),
                    (
                        2,
                        (1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0),
                    ),
                    (3, -(1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4, (1.0 / 24.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0)),
                ];
            }
            6 => {
                coeffs = vec![
                    (
                        0,
                        (1.0 / 720.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0)
                            * (alpha + 1.0),
                    ),
                    (
                        1,
                        -(1.0 / 120.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0),
                    ),
                    (
                        2,
                        (1.0 / 48.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0),
                    ),
                    (
                        3,
                        -(1.0 / 36.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0),
                    ),
                    (4, (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0) * (alpha + 6.0)),
                    (6, 1.0 / 720.0),
                ];
            }
            7 => {
                coeffs = vec![
                    (
                        0,
                        (1.0 / 5040.0)
                            * (alpha + 7.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0)
                            * (alpha + 1.0),
                    ),
                    (
                        1,
                        -(1.0 / 720.0)
                            * (alpha + 7.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0)
                            * (alpha + 2.0),
                    ),
                    (
                        2,
                        (1.0 / 240.0)
                            * (alpha + 7.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0)
                            * (alpha + 3.0),
                    ),
                    (
                        3,
                        -(1.0 / 144.0)
                            * (alpha + 7.0)
                            * (alpha + 6.0)
                            * (alpha + 5.0)
                            * (alpha + 4.0),
                    ),
                    (
                        4,
                        (1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0),
                    ),
                    (5, -(1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0)),
                    (6, (1.0 / 720.0) * (alpha + 7.0)),
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
