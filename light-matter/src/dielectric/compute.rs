use super::cache::{get_dielectric_cache_filename, load_dielectric_cache, save_dielectric_cache};
use super::response::{
    DielectricNormalization, epsilon_and_alpha, lehmann_sum, transitions_from_cache,
};
use super::sheet::exciton_dipole_cross;
use crate::{bands, timed_println_rank0};
use matrix_ops::{
    ConvertFrom, DiagonalizationOptions, Diagonalize, HermitianConjugate, MatrixOps,
    MatrixOpsError, Norm, Sqrt, RealCutoff, BatchPolicy
};
use ndarray::{Array1, Array2, ArrayView1, s};
use ndarray_linalg::c64;
use num_traits::Zero;
use rayon::prelude::*;
use std::iter::Sum;

use crate::checkpoint::DiagCheckpoint;
use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPBasisLayout, PPHamiltonian};

pub(crate) fn pp_dipole(
    bra: ArrayView1<'_, c64>,
    ket: ArrayView1<'_, c64>,
    x_ex: &Array2<c64>,
    layout: PPBasisLayout,
    n_ex: usize,
    n_ph: usize,
    n_pn: usize,
) -> Result<c64, MatrixOpsError> {
    let mut acc = c64::zero();

    for ex_ind1 in 0..n_ex {
        for ex_ind2 in 0..n_ex {
            let x_element = x_ex[[ex_ind1, ex_ind2]];

            for ph_idx in 0..n_ph {
                for pn_idx in 0..n_pn {
                    let pp_ind1 = layout.pp_index(ex_ind1, ph_idx, pn_idx, n_ex, n_ph, n_pn)?;
                    let pp_ind2 = layout.pp_index(ex_ind2, ph_idx, pn_idx, n_ex, n_ph, n_pn)?;
                    acc += bra[pp_ind1].conj() * ket[pp_ind2] * x_element;
                }
            }
        }
    }

    Ok(acc)
}

pub(crate) fn n_ph_n_pn(pp_config: &PPConfig) -> (usize, usize) {
    let n_ph = pp_config.q_values.len() + 1;
    let n_pn = if pp_config.pn_coupling.abs() < f64::EPSILON {
        1
    } else if pp_config.pn_n_fock == 0 {
        timed_println_rank0!(
            "Warning: phonon coupling non-zero but pn_n_fock == 0 -> forcing n_pn = 1"
        );
        1
    } else {
        pp_config.pn_n_fock
    };

    (n_ph, n_pn)
}

pub(crate) fn select_ground_state<M, B>(
    pp: &PPHamiltonian<M, B>,
    vecs: &Array2<c64>,
) -> Result<usize, MatrixOpsError>
where
    M: MatrixOps,
    B: Boson<M>,
{
    let mut ground_state_index = 0;
    let mut photon_character = bands::calculate_photonic_character(
        pp,
        &vecs.column(ground_state_index).to_owned(),
    )?;

    while photon_character > 0.1 {
        ground_state_index += 1;
        if ground_state_index >= vecs.ncols() {
            return Err(MatrixOpsError::Other {
                message: format!(
                    "No ground state with photonic character <= 0.1 found among the {} computed eigenstates",
                    vecs.ncols()
                ),
            });
        }
        photon_character = bands::calculate_photonic_character(
            pp,
            &vecs.column(ground_state_index).to_owned(),
        )?;
    }

    Ok(ground_state_index)
}

pub(crate) fn exciton_x_rel<M, B>(pp: &PPHamiltonian<M, B>, n_rel: usize) -> Array2<f64>
where
    M: MatrixOps,
    B: Boson<M>,
{
    let (lag_coeffs, states) = pp
        .polariton
        .matrix_elements
        .get_laguerre_coeffs(n_rel);
    let mu = pp.polariton.exciton.exciton_ham.rel_ham.mu;

    Array2::from_shape_fn((n_rel, n_rel), |(r_ind1, r_ind2)| {
        pp.polariton
            .matrix_elements
            .get_x(&states[r_ind1], &states[r_ind2], mu, &lag_coeffs)
    })
}

pub(crate) fn dielectric_diagonalization_options(
    n_eig: usize,
    energy_cutoff: Option<f64>,
) -> DiagonalizationOptions {
    match energy_cutoff {
        Some(cutoff) => DiagonalizationOptions::all_real_below(cutoff).real_cutoff(RealCutoff {
            batch_policy: BatchPolicy::DensityEstimate {
                initial: 500,
                safety: 1.25,
                min_batch: 128,
                max_batch: 512,
            },
            ..RealCutoff::new(cutoff)
        }),
        None => DiagonalizationOptions::smallest_real(n_eig),
    }
}

/// Result of dielectric‑function calculation including composition data.
/// The real and imaginary parts of the dielectric function are returned as
/// separate `Array1<f64>` values.  In addition, the photonic character and the
/// average phonon number for each eigenstate are provided as `Vec<f64>`.
pub struct DielectricResult {
    pub real: Array1<f64>,
    pub imag: Array1<f64>,
    pub alpha_re: Option<Array1<f64>>,
    pub alpha_im: Option<Array1<f64>>,
    pub photonic_chars: Vec<f64>,
    pub phonon_numbers: Vec<f64>,
}

/// Calculate the dielectric function from cached or newly computed pp_vals and x_pp.
pub fn get_dielectric_function<M, B>(
    pp_config: &PPConfig,
    pp: &mut PPHamiltonian<M, B>,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    omegas: Array1<f64>,
    gamma: f64,
    save_dir: &str,
    use_cached: bool,
    norm: DielectricNormalization,
    antiresonant: bool,
    checkpoint: Option<&DiagCheckpoint>,
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
    match energy_cutoff {
        Some(cutoff) => timed_println_rank0!(
            "Calculating dielectric function with ALL eigenstates below E = {:.6e} (overrides n_eig = {}), gamma={:.1e}",
            cutoff,
            n_eig,
            gamma
        ),
        None => timed_println_rank0!(
            "Calculating dielectric function with {} eigenstates, gamma={:.1e}",
            n_eig,
            gamma
        ),
    }

    let cache_filename =
        get_dielectric_cache_filename(pp, pp_config, save_dir, n_eig, energy_cutoff);
    timed_println_rank0!("Cache filename: {}", cache_filename);

    let (pp_vals, x_pp, photonic_chars, phonon_numbers, ground_state_index) =
        if use_cached && std::path::Path::new(&cache_filename).exists() {
            timed_println_rank0!(
                "Cache file exists, loading cached diagonalization data from {}",
                cache_filename
            );
            match load_dielectric_cache(&cache_filename) {
                Ok((vals, x, photonic_chars, phonon_numbers, gs_idx))
                    if x.len() == vals.len() && gs_idx < vals.len() =>
                {
                    timed_println_rank0!("Successfully loaded cached data ({} states)", vals.len());
                    (vals, x, photonic_chars, phonon_numbers, gs_idx)
                }
                Ok((vals, x, _, _, gs_idx)) => {
                    timed_println_rank0!(
                        "Cache is inconsistent (pp_vals={}, x_pp={}, gs_idx={}), recalculating...",
                        vals.len(),
                        x.len(),
                        gs_idx
                    );
                    compute_dielectric_cache(pp_config, pp, n_eig, energy_cutoff, checkpoint)?
                }
                Err(e) => {
                    timed_println_rank0!("Failed to load cache ({}), recalculating...", e);
                    compute_dielectric_cache(pp_config, pp, n_eig, energy_cutoff, checkpoint)?
                }
            }
        } else {
            if !use_cached {
                timed_println_rank0!("use_cached=false, computing diagonalization data...");
            } else {
                timed_println_rank0!("Cache file does not exist, computing diagonalization data...");
            }
            let cache_data =
                compute_dielectric_cache(pp_config, pp, n_eig, energy_cutoff, checkpoint)?;

            if sim_common::mpi::is_mpi_rank0_env() {
                timed_println_rank0!("Saving cache to {}", cache_filename);

                if let Err(e) = save_dielectric_cache(
                    &cache_filename,
                    &cache_data.0,
                    &cache_data.1,
                    &cache_data.2,
                    &cache_data.3,
                    cache_data.4,
                ) {
                    timed_println_rank0!("Warning: Failed to save cache: {}", e);
                } else {
                    timed_println_rank0!("Successfully saved cache");
                }
            }

            cache_data
        };

    if pp_vals.is_empty() {
        return Err(MatrixOpsError::Other {
            message: format!(
                "No eigenstates available for dielectric function (energy_cutoff = {:?})",
                energy_cutoff
            ),
        });
    }

    timed_println_rank0!(
        "Relative energies = {:?}",
        pp_vals.slice(s![0..50.min(pp_vals.len())]).to_owned() - pp_vals[ground_state_index]
    );

    let ground_energy = pp_vals[ground_state_index];

    let transitions = transitions_from_cache(
        &pp_vals,
        &x_pp,
        ground_energy,
        Some(ground_state_index),
    );

    let f_sum: f64 = transitions
        .iter()
        .map(|&(delta, strength)| 2.0 * delta * strength)
        .sum();
    let inverse_mu = 1.0 / pp.polariton.exciton.exciton_ham.rel_ham.mu;
    timed_println_rank0!(
        "f_sum = {:.6e}, 1/mu = {:.6e}, ratio = {:.6e}",
        f_sum,
        inverse_mu,
        f_sum / inverse_mu
    );

    let dielectric_vec: Vec<(c64, Option<c64>)> = omegas
        .as_slice()
        .expect("omegas should be contiguous")
        .par_iter()
        .map(|&omega| {
            let sum = lehmann_sum(omega, gamma, &transitions, antiresonant);
            epsilon_and_alpha(sum, norm)
        })
        .collect();

    let dielectric_function = Array1::from_iter(dielectric_vec.iter().map(|(eps, _)| eps.re));
    let dielectric_imag = Array1::from_iter(dielectric_vec.iter().map(|(eps, _)| eps.im));
    let has_alpha = dielectric_vec.iter().any(|(_, alpha)| alpha.is_some());
    let alpha_re = has_alpha.then(|| {
        Array1::from_iter(
            dielectric_vec
                .iter()
                .map(|(_, alpha)| alpha.expect("normalization alpha must be present").re),
        )
    });
    let alpha_im = has_alpha.then(|| {
        Array1::from_iter(
            dielectric_vec
                .iter()
                .map(|(_, alpha)| alpha.expect("normalization alpha must be present").im),
        )
    });

    Ok(DielectricResult {
        real: dielectric_function,
        imag: dielectric_imag,
        alpha_re,
        alpha_im,
        photonic_chars,
        phonon_numbers,
    })
}

/// Helper function to compute the expensive diagonalization and matrix elements
fn compute_dielectric_cache<M, B>(
    pp_config: &PPConfig,
    pp: &mut PPHamiltonian<M, B>,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    checkpoint: Option<&DiagCheckpoint>,
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
    // Must happen BEFORE build_pp_hamiltonian (which triggers compute_eigenstates).
    let ckpt_cfg = match (checkpoint, energy_cutoff) {
        (Some(c), Some(cut)) => Some(c.prepare(pp, pp_config, "dielectric_k0", cut)),
        _ => None,
    };

    let h_pp: M = pp.build_pp_hamiltonian(None)?;

    pp.polariton.matrix_elements.clear_relative_cache();
    pp.polariton.matrix_elements.clear_full_cache_maps();

    let e_vecs = pp.polariton.exciton.e_vecs.as_ref().unwrap();
    let n_com = pp.polariton.exciton.exciton_ham.kappa_vectors.len();
    let n_rel = pp_config.n_rel;
    let n_ex_eig = pp_config.n_ex_eig;
    let x_rel = exciton_x_rel(pp, n_rel);
    let x_exc = exciton_dipole_cross(e_vecs, e_vecs, &x_rel, n_com);

    if let Some(cutoff) = energy_cutoff {
        timed_println_rank0!(
            "Using energy cutoff of {:.6e}: computing ALL eigenvalues below it (n_eig = {} is ignored)",
            cutoff,
            n_eig
        );
    }

    let mut diag_options = dielectric_diagonalization_options(n_eig, energy_cutoff)
        .backend(matrix_ops::DiagonalizationBackend::Auto)
        .hermitian()
        .max_iterations(50_000)
        .deflation_batch_size(512)
        .tolerance(1e-10)
        .eigenvectors_rank0_only();
    if let Some(cfg) = ckpt_cfg {
        diag_options = diag_options.checkpoint(cfg);
    }

    timed_println_rank0!("Diagonalizing Hamiltonian of size {}", h_pp.shape().0);
    let (pp_vals, pp_vecs) = h_pp.diagonalize(diag_options)?.into_arrays_hermitian();

    let n_found = pp_vals.len();

    if n_found == 0 {
        return Err(MatrixOpsError::Other {
            message: format!(
                "Diagonalization returned no eigenvalues (energy_cutoff = {:?}, n_eig = {})",
                energy_cutoff, n_eig
            ),
        });
    }

    let e_min = pp_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let e_max = pp_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    timed_println_rank0!(
        "Diagonalization complete: {} eigenpairs, min energy = {:.6e}, max energy = {:.6e}",
        n_found,
        e_min,
        e_max
    );

    if energy_cutoff.is_none() && n_found != n_eig {
        timed_println_rank0!(
            "Warning: requested n_eig = {} but diagonalization returned {} eigenpairs",
            n_eig,
            n_found
        );
    }

    if !sim_common::mpi::is_mpi_rank0_env() {
        return Ok((pp_vals, Array1::<f64>::zeros(0), Vec::new(), Vec::new(), 0));
    }

    if pp_vecs.ncols() != n_found {
        return Err(MatrixOpsError::DimensionMismatch {
            expected: n_found,
            got: pp_vecs.ncols(),
            context: "dielectric: number of eigenvectors vs eigenvalues on rank 0".to_string(),
        });
    }

    let (n_ph, n_pn) = n_ph_n_pn(pp_config);
    let expected_len = pp.layout.pp_dim(n_ex_eig, n_ph, n_pn)?;
    if pp_vecs.nrows() != expected_len {
        panic!(
            "Inconsistent basis dimensions: pp_vecs.nrows() = {}, expected {} (n_ex_eig={} * n_ph={} * n_pn={})",
            pp_vecs.nrows(),
            expected_len,
            n_ex_eig,
            n_ph,
            n_pn
        );
    }

    let ground_state_index = select_ground_state(pp, &pp_vecs)?;
    let vec0 = pp_vecs.column(ground_state_index);

    let mut x_pp_values = Vec::with_capacity(n_found);
    for idx in 0..n_found {
        x_pp_values.push(
            pp_dipole(vec0, pp_vecs.column(idx), &x_exc, pp.layout, n_ex_eig, n_ph, n_pn)?.norm(),
        );
    }
    let x_pp = Array1::from_vec(x_pp_values);

    let mut photonic_chars: Vec<f64> = Vec::with_capacity(n_found);
    let mut phonon_numbers: Vec<f64> = Vec::with_capacity(n_found);

    let has_phonon_dof = pp_config.pn_coupling.abs() >= f64::EPSILON && n_pn > 1;
    let phonon_number_op = if has_phonon_dof {
        let lm_dim = pp.layout.lm.lm_dim(n_ex_eig, n_ph)?;
        let lm_eye = M::eye(lm_dim);
        let pn_number = pp.phonon.boson.adag_a(0)?;
        Some(pp.layout.kron_lm_pn(&lm_eye, &pn_number))
    } else {
        None
    };

    for idx in 0..n_found {
        photonic_chars.push(bands::calculate_photonic_character(
            pp,
            &pp_vecs.column(idx).to_owned(),
        )?);

        if let Some(ref op) = phonon_number_op {
            let vec: Array1<M::Scalar> = pp_vecs
                .column(idx)
                .iter()
                .map(|val| M::Scalar::convert_from(*val))
                .collect();
            let vec_dag: Array1<M::Scalar> = vec.iter().map(|x| x.dag()).collect();
            let mut tmp: Array1<M::Scalar> = Array1::zeros(vec.len());
            op.apply(vec.as_slice().unwrap(), tmp.as_slice_mut().unwrap())?;
            phonon_numbers.push(<M as MatrixOps>::Scalar::norm(vec_dag.dot(&tmp)));
        } else {
            phonon_numbers.push(0.0);
        }
    }

    Ok((pp_vals, x_pp, photonic_chars, phonon_numbers, ground_state_index))
}