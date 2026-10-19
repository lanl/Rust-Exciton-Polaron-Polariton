use ndarray::Array1;
use ndarray_linalg::c64;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DielectricNormalization {
    /// Old formula: eps = 1 - 4*pi*S. Kept for regression only.
    Legacy,
    /// eps = 1 + 4*pi*alpha_2D/d, alpha_2D = -n_x*S
    Slab { n_x: f64, d: f64 },
    /// eps = 1 + 2*pi*q*alpha_2D, alpha_2D = -n_x*S
    Sheet2D { n_x: f64, q: f64 },
}

pub fn method_tag(norm: DielectricNormalization, antiresonant: bool, lwa: bool) -> String {
    let mut tag = match norm {
        DielectricNormalization::Legacy if !antiresonant && !lwa => String::new(),
        DielectricNormalization::Legacy => "_methlegacy".to_string(),
        DielectricNormalization::Slab { n_x, d } => {
            format!("_methslab_nx{:.3e}_deff{:.3}", n_x, d)
        }
        DielectricNormalization::Sheet2D { n_x, q } => {
            format!("_methsheet_nx{:.3e}_qpr{:.4e}", n_x, q)
        }
    };

    if antiresonant {
        tag.push_str("_ar");
    }
    if lwa {
        tag.push_str("_lwa");
    }

    tag
}

/// Compute the Lehmann sum for transitions `(D_n, |x_n0|^2)`.
pub fn lehmann_sum(
    omega: f64,
    gamma: f64,
    transitions: &[(f64, f64)],
    antiresonant: bool,
) -> c64 {
    let mut acc = c64::new(0.0, 0.0);

    for &(delta, strength) in transitions {
        acc += c64::new(strength, 0.0) / c64::new(omega - delta, gamma);
        if antiresonant {
            acc -= c64::new(strength, 0.0) / c64::new(omega + delta, gamma);
        }
    }

    acc
}

/// Return `(epsilon, alpha_2D)`. The polarizability is absent for `Legacy`.
pub fn epsilon_and_alpha(
    sum: c64,
    norm: DielectricNormalization,
) -> (c64, Option<c64>) {
    let one = c64::new(1.0, 0.0);

    match norm {
        DielectricNormalization::Legacy => {
            (one - c64::new(4.0 * PI, 0.0) * sum, None)
        }
        DielectricNormalization::Slab { n_x, d } => {
            let alpha = -n_x * sum;
            (one + c64::new(4.0 * PI, 0.0) * alpha / d, Some(alpha))
        }
        DielectricNormalization::Sheet2D { n_x, q } => {
            let alpha = -n_x * sum;
            (one + c64::new(2.0 * PI * q, 0.0) * alpha, Some(alpha))
        }
    }
}

/// Build transitions from cached energies and dipole magnitudes.
pub fn transitions_from_cache(
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>,
    ground_energy: f64,
    skip_index: Option<usize>,
) -> Vec<(f64, f64)> {
    pp_vals
        .iter()
        .zip(x_pp.iter())
        .enumerate()
        .filter(|(index, _)| Some(*index) != skip_index)
        .map(|(_, (&energy, &x))| (energy - ground_energy, x * x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_static_response_has_positive_alpha() {
        let transitions = [(0.1, 1.0)];
        let sum = lehmann_sum(0.0, 1e-6, &transitions, true);
        let (eps, alpha) = epsilon_and_alpha(
            sum,
            DielectricNormalization::Slab {
                n_x: 1e-3,
                d: 10.0,
            },
        );

        let alpha = alpha.expect("slab response must provide alpha");
        assert!((alpha.re - 0.02).abs() < 1e-8);
        assert!((eps.re - (1.0 + 4.0 * PI * 0.02 / 10.0)).abs() < 1e-8);
    }

    #[test]
    fn resonant_alpha_has_positive_imaginary_part() {
        let sum = lehmann_sum(0.1, 1e-6, &[(0.1, 1.0)], true);
        let (_, alpha) = epsilon_and_alpha(
            sum,
            DielectricNormalization::Slab {
                n_x: 1e-3,
                d: 10.0,
            },
        );

        assert!(alpha.expect("slab response must provide alpha").im > 0.0);
    }

    #[test]
    fn legacy_response_matches_original_formula() {
        let omega = 0.13;
        let delta = 0.1;
        let gamma = 1e-6;
        let sum = lehmann_sum(omega, gamma, &[(delta, 1.0)], false);
        let (actual, alpha) = epsilon_and_alpha(sum, DielectricNormalization::Legacy);
        let expected = c64::new(1.0, 0.0)
            - c64::new(4.0 * PI, 0.0)
                * (c64::new(1.0, 0.0) / c64::new(omega - delta, gamma));

        assert_eq!(actual, expected);
        assert_eq!(alpha, None);
    }

    #[test]
    fn slab_and_sheet_share_alpha() {
        let sum = lehmann_sum(0.13, 1e-6, &[(0.1, 1.0)], true);
        let (_, slab_alpha) = epsilon_and_alpha(
            sum,
            DielectricNormalization::Slab {
                n_x: 1e-3,
                d: 10.0,
            },
        );
        let (_, sheet_alpha) = epsilon_and_alpha(
            sum,
            DielectricNormalization::Sheet2D {
                n_x: 1e-3,
                q: 0.02,
            },
        );

        assert_eq!(slab_alpha, sheet_alpha);
    }
}
