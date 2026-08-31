use super::cache::{get_dielectric_cache_filename, load_dielectric_cache, save_dielectric_cache};
use crate::{bands, timed_println};
use matrix_ops::{
    ConvertFrom, DiagonalizationOptions, Diagonalize, HermitianConjugate, MatrixOps,
    MatrixOpsError, Norm, Sqrt,
};
use ndarray::{Array1, Array2, s};
use ndarray_linalg::c64;
use num_traits::Zero;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::iter::Sum;

use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};

/// Result of dielectric‑function calculation including composition data.
/// The real and imaginary parts of the dielectric function are returned as
/// separate `Array1<f64>` values.  In addition, the photonic character and the
/// average phonon number for each eigenstate are provided as `Vec<f64>`.
pub struct DielectricResult {
    pub real: Array1<f64>,
    pub imag: Array1<f64>,
    pub photonic_chars: Vec<f64>,
    pub phonon_numbers: Vec<f64>,
}

/// Calculate the dielectric function from cached or newly computed pp_vals and x_pp.
pub fn get_dielectric_function<M, B>(
    pp_config: &PPConfig,
    pp: &mut PPHamiltonian<M, B>,
    n_eig: usize,
    omegas: Array1<f64>,
    gamma: f64,
    save_dir: &str,
    use_cached: bool,
) -> Result<DielectricResult, MatrixOpsError>
where
    M: MatrixOps + Diagonalize,
    M::Scalar: Sqrt
        + Sum
        + ConvertFrom<f64>
        + ConvertFrom<c64>
        + Norm
        + HermitianConjugate<Output = M::Scalar>,
    B: Boson<M>,
{
    timed_println!(
        "Calculating dielectric function with {} eigenstates, gamma={:.1e}",
        n_eig,
        gamma
    );

    let cache_filename = get_dielectric_cache_filename(pp, pp_config, save_dir, n_eig);
    timed_println!("Cache filename: {}", cache_filename);

    // Try to load cached data if requested
    // Load cached eigenvalues/x_pp if possible, otherwise compute everything
    let (pp_vals, x_pp, photonic_chars, phonon_numbers, ground_state_index) =
        if use_cached && std::path::Path::new(&cache_filename).exists() {
            timed_println!(
                "Cache file exists, loading cached diagonalization data from {}",
                cache_filename
            );
            match load_dielectric_cache(&cache_filename) {
                Ok((vals, x, photonic_chars, phonon_numbers, gs_idx)) => {
                    timed_println!("Successfully loaded cached data");
                    (vals, x, photonic_chars, phonon_numbers, gs_idx)
                }
                Err(e) => {
                    timed_println!("Failed to load cache ({}), recalculating...", e);
                    compute_dielectric_cache(pp_config, pp, n_eig)?
                }
            }
        } else {
            if !use_cached {
                timed_println!("use_cached=false, computing diagonalization data...");
            } else if !std::path::Path::new(&cache_filename).exists() {
                timed_println!("Cache file does not exist, computing diagonalization data...");
            } else {
                timed_println!("Computing diagonalization data...");
            }
            let cache_data = compute_dielectric_cache(pp_config, pp, n_eig)?;

            if sim_common::mpi::is_mpi_rank0_env() {
                timed_println!("Saving cache to {}", cache_filename);

                if let Err(e) = save_dielectric_cache(
                    &cache_filename,
                    &cache_data.0,
                    &cache_data.1,
                    &cache_data.2,
                    &cache_data.3,
                    cache_data.4,
                ) {
                    timed_println!("Warning: Failed to save cache: {}", e);
                } else {
                    timed_println!("Successfully saved cache");
                }
            }

            cache_data
        };

    timed_println!(
        "Relative energies = {:?}",
        pp_vals.slice(s![0..50.min(pp_vals.len())]).to_owned() - pp_vals[ground_state_index]
    );

    // Precompute transition energies and scalar oscillator strengths.
    // x_pp[n] stores |<0|x|n>|, so the numerator is x_pp[n]^2.
    let ground_energy = pp_vals[ground_state_index];

    let transitions: Vec<(f64, f64)> = pp_vals
        .iter()
        .zip(x_pp.iter())
        .enumerate()
        .filter(|(idx, _)| *idx != ground_state_index)
        .map(|(_, (&energy, &x))| (energy - ground_energy, x * x))
        .collect();

    let dielectric_vec: Vec<c64> = omegas
        .as_slice()
        .expect("omegas should be contiguous")
        .par_iter()
        .map(|&omega| {
            let result: c64 = transitions
                .iter()
                .map(|&(delta_e, strength)| {
                    c64::new(strength, 0.0) / (c64::new(omega - delta_e, gamma))
                })
                .sum();

            c64::new(1.0, 0.0) - c64::new(4.0 * PI, 0.0) * result
        })
        .collect();

    let dielectric_function = Array1::from_vec(dielectric_vec);

    Ok(DielectricResult {
        real: dielectric_function.map(|&x| x.re),
        imag: dielectric_function.map(|&x| x.im),
        photonic_chars,
        phonon_numbers,
    })
}

/// Helper function to compute the expensive diagonalization and matrix elements
fn compute_dielectric_cache<M, B>(
    pp_config: &PPConfig,
    pp: &mut PPHamiltonian<M, B>,
    n_eig: usize,
) -> Result<(Array1<f64>, Array1<f64>, Vec<f64>, Vec<f64>, usize), MatrixOpsError>
where
    M: MatrixOps + Diagonalize,
    M::Scalar: Sqrt
        + Sum
        + ConvertFrom<f64>
        + ConvertFrom<c64>
        + HermitianConjugate<Output = M::Scalar>
        + Norm,
    B: Boson<M>,
{
    // Build Hamiltonian
    let h_pp: M = pp.build_pp_hamiltonian(None)?;

    // In the ComposedOperator path, the KronSum now owns Arc references to any
    // dense matrix-element blocks it needs. Drop the HashMap owners before
    // diagonalization to reduce resident memory.
    pp.polariton.matrix_elements.clear_relative_cache();
    pp.polariton.matrix_elements.clear_full_cache_maps();

    // Lag Coeffs for Relative coordinates
    let (lag_coeffs, states) = &pp
        .polariton
        .matrix_elements
        .get_laguerre_coeffs(pp.polariton.exciton.n_rel);
    let mu = pp.polariton.exciton.exciton_ham.rel_ham.mu;

    // Get <0|x|n> matrix elements in the exciton basis

    // Calculate the matrix elements using iterators
    let e_vecs = pp.polariton.exciton.e_vecs.as_ref().unwrap();
    let n_com = pp.polariton.exciton.exciton_ham.kappa_vectors.len();
    let n_rel = pp_config.n_rel;
    let n_ex_eig = pp_config.n_ex_eig;

    let x_rel = &Array2::from_shape_fn((n_rel, n_rel), |(r_ind1, r_ind2)| {
        pp.polariton
            .matrix_elements
            .get_x(&states[r_ind1], &states[r_ind2], mu, &lag_coeffs)
    });

    // timed_println!("<x> for Relative Coordinate:");
    // println!("{}", PrettyArray2(&x_rel, 4));

    // Get <x> in the Exciton basis
    let x_exc = &Array2::from_shape_fn((n_ex_eig, n_ex_eig), |(idx1, idx2)| {
        let vec1 = &e_vecs.column(idx1);
        let vec2 = &e_vecs.column(idx2);

        // Calculate the matrix elements using iterators
        (0..n_rel)
            .flat_map(|rel_ind1| {
                (0..n_rel).flat_map(move |rel_ind2| {
                    (0..n_com).map(move |com_ind| {
                        let i1 = com_ind * n_rel + rel_ind1;
                        let i2 = com_ind * n_rel + rel_ind2;
                        let a1 = vec1[i1];
                        let a2 = vec2[i2];

                        a1.conj() * a2 * x_rel[[rel_ind1, rel_ind2]]
                    })
                })
            })
            .fold(c64::zero(), |acc, val| acc + val)
    });

    // timed_println!("|<x>| for exciton:");
    // let x_exc_mag: Array2<f64> = x_exc.mapv(|val| val.norm());
    // println!("{}", PrettyArray2(&x_exc_mag, 4));

    // timed_println!("Exciton eigenvalues transitions");
    // let e_vals_transition = pp.polariton.exciton.e_vals.as_ref().unwrap()
    //     - pp.polariton.exciton.e_vals.as_ref().unwrap()[0];
    // println!("{}", e_vals_transition);

    // Diagonalize at the Gamma point
    let (pp_vals, pp_vecs) = h_pp
        .diagonalize(
            DiagonalizationOptions::smallest_real(n_eig)
                .backend(matrix_ops::DiagonalizationBackend::Auto)
                .hermitian()
                .max_iterations(50_000)
                .deflation_batch_size(512)
                .tolerance(1e-10),
        )?
        .into_arrays_hermitian();

    // If the phonon coupling is effectively zero we *must* use a single
    // phonon vacuum state (dimension = 1).  This is independent of the value
    // stored in `pp_config.pn_n_fock`.
    let n_ph = pp_config.q_values.len() + 1; // photon modes (+ the q=0 mode)
    let n_pn = if pp_config.pn_coupling.abs() < f64::EPSILON {
        1
    } else {
        // Guard against a user‑provided zero here (which would make the loop
        // empty and break the indexing logic).
        if pp_config.pn_n_fock == 0 {
            timed_println!(
                "Warning: phonon coupling non-zero but pn_n_fock == 0 -> forcing n_pn = 1"
            );
            1
        } else {
            pp_config.pn_n_fock
        }
    };

    // Sanity check: the size of the ground‑state vector must equal
    // n_ex_eig * n_ph * n_pn (or be a multiple thereof if extra blocks are
    // present). If it does not, abort with a clear message.
    let mut vec0 = pp_vecs.column(0);
    let expected_len = pp.layout.pp_dim(n_ex_eig, n_ph, n_pn)?;
    if vec0.len() != expected_len {
        panic!(
            "Inconsistent basis dimensions: vec0.len() = {}, expected {} (n_ex_eig={} * n_ph={} * n_pn={})",
            vec0.len(),
            expected_len,
            n_ex_eig,
            n_ph,
            n_pn
        );
    }

    let mut photon_character = bands::calculate_photonic_character(pp, &vec0.to_owned())?;
    let mut ground_state_index = 0;

    while photon_character > 0.1 {
        ground_state_index += 1;
        if ground_state_index >= pp_vals.len() {
            panic!(
                "No ground state with photonic character <= 0.1 found among the computed eigenstates"
            );
        }
        vec0 = pp_vecs.column(ground_state_index);
        photon_character = bands::calculate_photonic_character(pp, &vec0.to_owned())?;
    }

    // We store the magnitude of the scalar transition dipole,
    //     x_pp[n] = |<0|x|n>|.
    // This is sufficient for the scalar resonant dielectric function, where the
    // numerator is |<0|x|n>|^2. For a full tensor dielectric calculation,
    // one should store complex transition dipoles for each Cartesian component
    // and form <n|x_i|0><0|x_j|n> explicitly.
    let mut x_pp_values = Vec::with_capacity(n_eig);

    for idx in 0..n_eig {
        let vec1 = pp_vecs.column(idx);
        let mut acc = c64::zero();

        for ex_ind1 in 0..n_ex_eig {
            for ex_ind2 in 0..n_ex_eig {
                let x_element = x_exc[[ex_ind1, ex_ind2]];

                for ph_idx in 0..n_ph {
                    for pn_idx in 0..n_pn {
                        let pp_ind1 = pp
                            .layout
                            .pp_index(ex_ind1, ph_idx, pn_idx, n_ex_eig, n_ph, n_pn)?;

                        let pp_ind2 = pp
                            .layout
                            .pp_index(ex_ind2, ph_idx, pn_idx, n_ex_eig, n_ph, n_pn)?;

                        acc += vec0[pp_ind1].conj() * vec1[pp_ind2] * x_element;
                    }
                }
            }
        }

        x_pp_values.push(acc.norm());
    }

    let x_pp = Array1::from_vec(x_pp_values);

    // -----------------------------------------------------------------
    // Compute composition data: photonic character and average phonon number
    // -----------------------------------------------------------------

    let mut photonic_chars: Vec<f64> = Vec::with_capacity(n_eig);
    let mut phonon_numbers: Vec<f64> = Vec::with_capacity(n_eig);

    // Check if the Hamiltonian actually includes phonon degrees of freedom
    // When pn_coupling == 0, build_pp_hamiltonian returns early without phonon space
    let has_phonon_dof = pp_config.pn_coupling.abs() >= f64::EPSILON && n_pn > 1;

    // Only create phonon number operator if phonons are actually present in the Hamiltonian
    let phonon_number_op = if has_phonon_dof {
        let lm_dim = pp.layout.lm.lm_dim(n_ex_eig, n_ph)?;
        let lm_eye = M::eye(lm_dim);
        let pn_number = pp.phonon.boson.adag_a(0)?;

        Some(pp.layout.kron_lm_pn(&lm_eye, &pn_number))
    } else {
        None
    };

    for idx in 0..n_eig {
        let vec: Array1<M::Scalar> = pp_vecs
            .column(idx)
            .iter()
            .map(|val| M::Scalar::convert_from(*val))
            .collect::<Array1<M::Scalar>>();
        let vec_dag: Array1<M::Scalar> = vec.iter().map(|x| x.dag()).collect();

        // Photonic character: sum_i <ψ|a_i† a_i|ψ>
        photonic_chars.push(bands::calculate_photonic_character(
            pp,
            &pp_vecs.column(idx).to_owned(),
        )?);

        // Phonon number expectation
        // Only compute if phonons are actually present in the Hamiltonian
        if let Some(ref op) = phonon_number_op {
            let mut tmp: Array1<M::Scalar> = Array1::zeros(vec.len()); // zero vector of same shape
            op.apply(vec.as_slice().unwrap(), tmp.as_slice_mut().unwrap())?;
            let val: M::Scalar = vec_dag.dot(&tmp);
            phonon_numbers.push(<M as MatrixOps>::Scalar::norm(val));
        } else {
            // No phonon degrees of freedom, phonon number is always zero
            phonon_numbers.push(0.0);
        }
    }

    Ok((
        pp_vals,
        x_pp,
        photonic_chars,
        phonon_numbers,
        ground_state_index,
    ))
}
