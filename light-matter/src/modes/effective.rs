use crate::types::Vector3D;

// Speed of light in atomic units
const C: f64 = 137.036;
const PI: f64 = std::f64::consts::PI;

/// Generate q_values and coupling strengths for photon modes
///
/// # Arguments
/// * `d_omega` - Frequency spacing between modes
/// * `n_modes` - Number of modes (will generate n_modes x n_modes grid)
/// * `omega_0` - Central photon frequency
///
/// # Returns
/// A tuple containing:
/// * `Vec<Vector3D>` - Vector of q-values for each mode
/// * `Array1<f64>` - Array of coupling strengths per mode
pub fn generate_modes(
    d_omega: f64,
    n_modes: usize,
    omega_0: f64,
    omega_max: f64,
    reflectivity: f64,
) -> (Vec<Vector3D>, Vec<f64>) {
    let n_omega = ((omega_max - omega_0) / d_omega).floor() as usize;

    if omega_0 > omega_max {
        panic!("Central frequency omega_0 must be less than or equal to omega_max");
    }

    if reflectivity < 0.0 || reflectivity >= 1.0 {
        panic!("Reflectivity must be between 0 and 1 (exclusive)");
    }

    let target_modes = n_modes * n_modes;
    let (d_q, total_modes) = find_d_q_for_target_modes(target_modes, d_omega, omega_0, omega_max);

    let (q_values, couplings): (Vec<Vector3D>, Vec<f64>);
    if reflectivity > f64::EPSILON {
        (q_values, couplings) = (0..n_omega)
            .flat_map(|i| {
                let omega_n = omega_0 + (i as f64) * d_omega;
                let q_n = ((omega_n * omega_n - omega_0 * omega_0).sqrt()) / C;

                let qs = get_q_per_frequency(d_q, q_n, omega_0);
                let c = vec![get_coupling(d_omega, d_q, omega_n, omega_0, reflectivity); qs.len()];
                qs.into_iter().zip(c.into_iter())
            })
            .unzip();
    } else {
        q_values = (0..n_omega)
            .flat_map(|i| {
                let omega_n = omega_0 + (i as f64) * d_omega;
                let q_n = ((omega_n * omega_n - omega_0 * omega_0).sqrt()) / C;

                get_q_per_frequency(d_q, q_n, omega_0)
            })
            .collect();
        couplings = vec![0.0; q_values.len()];
    }

    if q_values.len() != total_modes {
        println!(
            "Warning: Generated {} modes, but expected {}",
            q_values.len(),
            total_modes
        );
    }

    (q_values, couplings)
}

fn get_coupling(d_omega: f64, d_q: f64, omega_n: f64, omega_0: f64, reflectivity: f64) -> f64 {
    let loss = 1.0 - reflectivity;

    let q_n = (omega_n * omega_n - omega_0 * omega_0).sqrt() / C;
    let l_c = 2.0 * PI / (omega_0 / C);

    let n_modes_per_freq = modes_per_frequency(q_n, d_q);
    let d_theta = 2.0 * PI / (n_modes_per_freq as f64);

    // A_0 = sqrt(hbar ell_n / (2 * epsilon_0 * omega * V))
    // where ell_n / V is from effective mode density:
    //        ell_n / V = omega * d_omega * d_theta / (8 * pi^2 * c^2 * loss * l_c)
    (d_omega * d_theta / (8.0 * PI * PI * C * C * loss * l_c) / (8.0 * PI)).sqrt()
}

fn get_q_per_frequency(d_q: f64, q_n: f64, omega_0: f64) -> Vec<Vector3D> {
    let n_modes = modes_per_frequency(q_n, d_q);

    let q_z = omega_0 / C;

    let d_theta = 2.0 * PI / (n_modes as f64);

    (0..n_modes)
        .map(|m| {
            let theta = m as f64 * d_theta;
            Vector3D {
                x: q_n * theta.cos(),
                y: q_n * theta.sin(),
                z: q_z,
            }
        })
        .collect::<Vec<Vector3D>>()
}

fn modes_per_frequency(q_n: f64, d_q: f64) -> usize {
    if q_n <= f64::EPSILON {
        return 1;
    }

    if d_q / q_n < 2.0 {
        (PI / (d_q / 2.0 / q_n).asin()).floor() as usize
    } else {
        1 // Only one mode if d_q >= q_n
    }
}

/// Find d_q that gives approximately the target number of modes
///
/// # Arguments
/// * `n_modes_target` - Target total number of modes
/// * `d_omega` - Frequency spacing between modes
/// * `omega_0` - Central photon frequency
/// * `omega_max` - Maximum photon frequency
///
/// # Returns
/// The d_q value that achieves approximately n_modes_target total modes
pub fn find_d_q_for_target_modes(
    n_modes_target: usize,
    d_omega: f64,
    omega_0: f64,
    omega_max: f64,
) -> (f64, usize) {
    // Helper function to compute total modes for given d_q
    fn compute_total_modes(d_q: f64, d_omega: f64, omega_0: f64, omega_max: f64) -> usize {
        let n_freq = ((omega_max - omega_0) / d_omega).floor() as usize;
        let mut total = 0;

        for i in 0..n_freq {
            let omega_n = omega_0 + (i as f64) * d_omega;
            let q_n = ((omega_n * omega_n - omega_0 * omega_0).sqrt()) / C;
            total += modes_per_frequency(q_n, d_q);
        }
        total
    }

    let n_freq = ((omega_max - omega_0) / d_omega).floor() as usize;

    // If target modes equals or is less than number of frequencies,
    // we're in the regime where d_q >= 2*q_n for most/all frequencies
    if n_modes_target <= n_freq {
        // Start with d_q large enough to give 1 mode per frequency
        // Find the largest q_n (at omega_max)
        let omega_max_actual = omega_0 + (n_freq as f64 - 1.0) * d_omega;
        let q_max = ((omega_max_actual * omega_max_actual - omega_0 * omega_0).sqrt()) / C;

        // Set d_q = 2*q_max to ensure 1 mode per frequency
        return (2.0 * q_max, n_freq);
    }

    // Otherwise, we need multiple modes per frequency
    // Initial guess: assume we need modes_per_freq modes at each frequency
    let avg_modes_per_freq = (n_modes_target as f64) / (n_freq as f64);

    // Use average q_n for initial guess
    let mut sum_q = 0.0;
    for i in 0..n_freq {
        let omega_n = omega_0 + (i as f64) * d_omega;
        sum_q += ((omega_n * omega_n - omega_0 * omega_0).sqrt()) / C;
    }
    let avg_q = sum_q / (n_freq as f64);

    // From modes_per_frequency ≈ PI / asin(d_q / (2*q)) for small d_q/q
    // We want: avg_modes_per_freq ≈ PI / asin(d_q / (2*avg_q))
    // So: d_q ≈ 2*avg_q * sin(PI / avg_modes_per_freq)
    let d_q_guess = 2.0 * avg_q * (PI / avg_modes_per_freq).sin();

    // Binary search bounds around the guess
    // d_q inversely related to number of modes, so:
    let mut d_q_min = d_q_guess * 0.1; // Smaller d_q -> more modes
    let mut d_q_max = d_q_guess * 10.0; // Larger d_q -> fewer modes

    // Ensure bounds bracket the target
    let mut max_iterations = 50;
    while compute_total_modes(d_q_min, d_omega, omega_0, omega_max) < n_modes_target
        && max_iterations > 0
    {
        d_q_min *= 0.5;
        max_iterations -= 1;
    }
    max_iterations = 50;
    while compute_total_modes(d_q_max, d_omega, omega_0, omega_max) > n_modes_target
        && max_iterations > 0
    {
        d_q_max *= 2.0;
        max_iterations -= 1;
    }

    // Binary search
    let tolerance = 1e-8;
    while d_q_max - d_q_min > tolerance {
        let d_q_mid = (d_q_min + d_q_max) / 2.0;
        let total = compute_total_modes(d_q_mid, d_omega, omega_0, omega_max);

        if total > n_modes_target {
            d_q_min = d_q_mid; // Increase d_q to reduce modes
        } else if total < n_modes_target {
            d_q_max = d_q_mid; // Decrease d_q to increase modes
        } else {
            println!(
                "Exact match found: d_q = {}, total modes = {}",
                d_q_mid, total
            );
            return (d_q_mid, total); // Exact match
        }
    }

    // Return the value that gets closest to target
    let d_q_result = (d_q_min + d_q_max) / 2.0;
    let total_achieved = compute_total_modes(d_q_result, d_omega, omega_0, omega_max);

    println!(
        "Target modes: {}, Achieved: {}, d_q: {}",
        n_modes_target, total_achieved, d_q_result
    );

    (d_q_result, total_achieved)
}
