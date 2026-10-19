use super::compute::{
    DielectricResult, dielectric_diagonalization_options, exciton_x_rel, n_ph_n_pn, pp_dipole,
    select_ground_state,
};
use super::response::{DielectricNormalization, epsilon_and_alpha, lehmann_sum};
use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};
use crate::utils::eig_selection_tag;
use crate::timed_println_rank0;
use matrix_ops::{
    ConvertFrom, Diagonalize, DiagonalizationOptions, HermitianConjugate, MatrixOps,
    MatrixOpsError, Norm, Sqrt,
};
use ndarray::{Array1, Array2, s};
use ndarray_linalg::c64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// X[i,j] = <e_bra_i|x_rel|e_ket_j>, with the same relative-coordinate
/// row/column convention used by the existing dielectric calculation.
pub fn exciton_dipole_cross(
    e_bra: &Array2<c64>,
    e_ket: &Array2<c64>,
    x_rel: &Array2<f64>,
    n_com: usize,
) -> Array2<c64> {
    let n_rel = x_rel.nrows();
    assert_eq!(x_rel.ncols(), n_rel);
    assert_eq!(e_ket.nrows(), n_com * n_rel);
    assert_eq!(e_bra.nrows(), n_com * n_rel);

    let x_c = x_rel.mapv(|value| c64::new(value, 0.0));
    let mut transformed_ket = Array2::<c64>::zeros(e_ket.raw_dim());

    for com in 0..n_com {
        let rows = com * n_rel..(com + 1) * n_rel;
        let ket_block = e_ket.slice(s![rows.clone(), ..]).to_owned();
        let block = ndarray::linalg::Dot::dot(&x_c, &ket_block);
        transformed_ket
            .slice_mut(s![rows, ..])
            .assign(&block);
    }

    let bra_conjugate = e_bra.t().mapv(|value| value.conj());
    ndarray::linalg::Dot::dot(&bra_conjugate, &transformed_ket)
}

pub const SHEET_CACHE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SheetDielectricCache {
    pub version: u32,
    pub q_probe: f64,
    pub include_photon_recoil: bool,
    pub e0: f64,
    /// Index into energies_q of the K=q boosted ground state.
    pub ground_state_index_q: usize,
    pub energies_q: Vec<f64>,
    pub x_q: Vec<f64>,
    pub photonic_chars_q: Vec<f64>,
    pub phonon_numbers_q: Vec<f64>,
}

pub fn get_sheet_dielectric_cache_filename<M, B>(
    pp: &PPHamiltonian<M, B>,
    pp_config: &PPConfig,
    save_dir: &str,
    q_probe: f64,
    n_eig: usize,
    energy_cutoff: Option<f64>,
) -> String
where
    M: MatrixOps,
    B: Boson<M>,
{
    let base = super::cache::get_dielectric_cache_filename(
        pp,
        pp_config,
        save_dir,
        n_eig,
        energy_cutoff,
    );
    let prefix = format!("{}/dielectric_cache_", save_dir);
    let body = base
        .strip_prefix(&prefix)
        .unwrap_or_else(|| panic!("unexpected dielectric cache filename: {}", base))
        .strip_suffix(".bin")
        .expect("dielectric cache filename must end in .bin");
    let lwa_tag = if !pp_config.include_photon_recoil {
        "_lwa"
    } else {
        ""
    };
    let body = body.strip_suffix(lwa_tag).unwrap_or(body);
    let eig_tag = eig_selection_tag(n_eig, energy_cutoff);
    let body = body
        .strip_suffix(&eig_tag)
        .expect("dielectric cache filename must contain eigenvalue tag")
        .trim_end_matches('_');

    format!(
        "{}/dielectric_sheet_cache_{}_qpr{:.4e}_{}{}.bin",
        save_dir, body, q_probe, eig_tag, lwa_tag
    )
}

pub fn compute_sheet_dielectric_cache<M, B>(
    pp_config: &PPConfig,
    q_probe: f64,
    n_eig: usize,
    energy_cutoff: Option<f64>,
) -> Result<SheetDielectricCache, MatrixOpsError>
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
    let mut cfg0 = pp_config.clone();
    cfg0.e_config.k = vec![0.0, 0.0];
    let mut pp0 = PPHamiltonian::<M, B>::from_config(&cfg0);
    let h0 = pp0.build_pp_hamiltonian(None)?;
    pp0.polariton.matrix_elements.clear_relative_cache();
    pp0.polariton.matrix_elements.clear_full_cache_maps();
    let e_vecs0 = pp0
        .polariton
        .exciton
        .e_vecs
        .clone()
        .ok_or_else(|| MatrixOpsError::Other {
            message: "K=0 exciton eigenvectors were not computed".to_string(),
        })?;
    let x_rel = exciton_x_rel(&pp0, pp_config.n_rel);
    let n_com = pp0.polariton.exciton.exciton_ham.kappa_vectors.len();
    let layout = pp0.layout;
    let (n_ph, n_pn) = n_ph_n_pn(pp_config);

    let (energies0, vecs0) = h0
        .diagonalize(
            DiagonalizationOptions::smallest_real(16)
                .backend(matrix_ops::DiagonalizationBackend::Auto)
                .hermitian()
                .max_iterations(50_000)
                .deflation_batch_size(512)
                .tolerance(1e-10)
                .eigenvectors_rank0_only(),
        )?
        .into_arrays_hermitian();

    let (e0, psi0) = if sim_common::mpi::is_mpi_rank0_env() {
        if energies0.is_empty() || vecs0.ncols() == 0 {
            return Err(MatrixOpsError::Other {
                message: "K=0 sheet diagonalization returned no eigenvectors".to_string(),
            });
        }
        let ground_state_index = select_ground_state(&pp0, &vecs0)?;
        (
            energies0[ground_state_index],
            Some(vecs0.column(ground_state_index).to_owned()),
        )
    } else {
        (0.0, None)
    };

    drop(h0);
    drop(pp0);

    let mut cfgq = pp_config.clone();
    cfgq.e_config.k = vec![q_probe, 0.0];
    let mut ppq = PPHamiltonian::<M, B>::from_config(&cfgq);
    let hq = ppq.build_pp_hamiltonian(None)?;
    ppq.polariton.matrix_elements.clear_relative_cache();
    ppq.polariton.matrix_elements.clear_full_cache_maps();
    let e_vecs_q = ppq
        .polariton
        .exciton
        .e_vecs
        .clone()
        .ok_or_else(|| MatrixOpsError::Other {
            message: "K=q exciton eigenvectors were not computed".to_string(),
        })?;

    if e_vecs0.ncols() != e_vecs_q.ncols() || e_vecs0.ncols() != pp_config.n_ex_eig {
        return Err(MatrixOpsError::DimensionMismatch {
            expected: pp_config.n_ex_eig,
            got: e_vecs_q.ncols(),
            context: format!(
                "sheet2d exciton eigenbasis columns at K=0 ({}) and K=q ({})",
                e_vecs0.ncols(),
                e_vecs_q.ncols()
            ),
        });
    }

    let diag_options = dielectric_diagonalization_options(n_eig, energy_cutoff);
    let (energies_q, vecs_q) = hq
        .diagonalize(
            diag_options
                .backend(matrix_ops::DiagonalizationBackend::Auto)
                .hermitian()
                .max_iterations(50_000)
                .deflation_batch_size(512)
                .tolerance(1e-10)
                .eigenvectors_rank0_only(),
        )?
        .into_arrays_hermitian();

    if !sim_common::mpi::is_mpi_rank0_env() {
        return Ok(SheetDielectricCache {
            version: SHEET_CACHE_VERSION,
            q_probe,
            include_photon_recoil: pp_config.include_photon_recoil,
            e0,
            ground_state_index_q: 0,
            energies_q: Vec::new(),
            x_q: Vec::new(),
            photonic_chars_q: Vec::new(),
            phonon_numbers_q: Vec::new(),
        });
    }

    if energies_q.is_empty() || vecs_q.ncols() != energies_q.len() {
        return Err(MatrixOpsError::Other {
            message: "K=q sheet diagonalization returned inconsistent eigenpairs".to_string(),
        });
    }

    let ground_state_index_q = select_ground_state(&ppq, &vecs_q)?;
    let boost_shift = energies_q[ground_state_index_q] - e0;
    let m_total = pp_config.e_config.m_e + pp_config.e_config.m_h;
    let expected_shift = q_probe * q_probe / (2.0 * m_total);
    timed_println_rank0!(
        "sheet2d: E_gs(q) - E_gs(0) = {:.6e} (free-COM estimate q^2/2M = {:.6e})",
        boost_shift,
        expected_shift
    );
    if boost_shift < -1e-10 {
        timed_println_rank0!(
            "Warning: boosted ground state lies BELOW the K=0 ground state by {:.3e}; check K handling",
            -boost_shift
        );
    }

    let psi0 = psi0.expect("rank 0 must retain the K=0 ground state");
    let x_ex = exciton_dipole_cross(&e_vecs_q, &e_vecs0, &x_rel, n_com);
    let mut x_q = Vec::with_capacity(energies_q.len());
    let mut photonic_chars_q = Vec::with_capacity(energies_q.len());
    let mut phonon_numbers_q = Vec::with_capacity(energies_q.len());

    let has_phonon_dof = pp_config.pn_coupling.abs() >= f64::EPSILON && n_pn > 1;
    let phonon_number_op = if has_phonon_dof {
        let lm_dim = ppq.layout.lm.lm_dim(pp_config.n_ex_eig, n_ph)?;
        let lm_eye = M::eye(lm_dim);
        let pn_number = ppq.phonon.boson.adag_a(0)?;
        Some(ppq.layout.kron_lm_pn(&lm_eye, &pn_number))
    } else {
        None
    };

    for idx in 0..energies_q.len() {
        let vec = vecs_q.column(idx);
        x_q.push(
            pp_dipole(
                vec,
                psi0.view(),
                &x_ex,
                layout,
                pp_config.n_ex_eig,
                n_ph,
                n_pn,
            )?
            .norm(),
        );
        photonic_chars_q.push(crate::bands::calculate_photonic_character(
            &ppq,
            &vec.to_owned(),
        )?);

        if let Some(ref op) = phonon_number_op {
            let converted: Array1<M::Scalar> = vec
                .iter()
                .map(|value| M::Scalar::convert_from(*value))
                .collect();
            let dag: Array1<M::Scalar> = converted.iter().map(|value| value.dag()).collect();
            let mut tmp = Array1::<M::Scalar>::zeros(converted.len());
            op.apply(
                converted.as_slice().expect("converted eigenvector must be contiguous"),
                tmp.as_slice_mut().expect("temporary vector must be contiguous"),
            )?;
            phonon_numbers_q.push(<M as MatrixOps>::Scalar::norm(dag.dot(&tmp)));
        } else {
            phonon_numbers_q.push(0.0);
        }
    }

    Ok(SheetDielectricCache {
        version: SHEET_CACHE_VERSION,
        q_probe,
        include_photon_recoil: pp_config.include_photon_recoil,
        e0,
        ground_state_index_q,
        energies_q: energies_q.to_vec(),
        x_q,
        photonic_chars_q,
        phonon_numbers_q,
    })
}

pub fn sheet_dielectric_result(
    cache: &SheetDielectricCache,
    omegas: &Array1<f64>,
    gamma: f64,
    n_x: f64,
    q_probe: f64,
    antiresonant: bool,
) -> DielectricResult {
    let gs = cache.ground_state_index_q;
    let transitions: Vec<(f64, f64)> = cache
        .energies_q
        .iter()
        .zip(cache.x_q.iter())
        .enumerate()
        .filter(|(index, _)| *index != gs)
        .map(|(_, (&energy, &x))| (energy - cache.e0, x * x))
        .collect();

    let n_nonpositive = transitions.iter().filter(|(delta, _)| *delta <= 0.0).count();
    if n_nonpositive > 0 {
        timed_println_rank0!(
            "Warning: sheet2d has {} transitions with Delta <= 0 after removing the boosted ground state",
            n_nonpositive
        );
    }
    timed_println_rank0!(
        "sheet2d: skipped boosted ground state index {} (|X_00| = {:.3e})",
        gs,
        cache.x_q.get(gs).copied().unwrap_or(0.0)
    );

    let values: Vec<(c64, Option<c64>)> = omegas
        .as_slice()
        .expect("omegas should be contiguous")
        .par_iter()
        .map(|&omega| {
            epsilon_and_alpha(
                lehmann_sum(omega, gamma, &transitions, antiresonant),
                DielectricNormalization::Sheet2D { n_x, q: q_probe },
            )
        })
        .collect();

    DielectricResult {
        real: Array1::from_iter(values.iter().map(|(eps, _)| eps.re)),
        imag: Array1::from_iter(values.iter().map(|(eps, _)| eps.im)),
        alpha_re: Some(Array1::from_iter(
            values
                .iter()
                .map(|(_, alpha)| alpha.expect("sheet response must provide alpha").re),
        )),
        alpha_im: Some(Array1::from_iter(
            values
                .iter()
                .map(|(_, alpha)| alpha.expect("sheet response must provide alpha").im),
        )),
        photonic_chars: cache.photonic_chars_q.clone(),
        phonon_numbers: cache.phonon_numbers_q.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn cross_dipole_matches_naive_triple_loop() {
        let n_com = 3;
        let n_rel = 4;
        let n_ex = 5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let e_bra = Array2::from_shape_fn((n_com * n_rel, n_ex), |_| {
            c64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        });
        let e_ket = Array2::from_shape_fn((n_com * n_rel, n_ex), |_| {
            c64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        });
        let x_rel = Array2::from_shape_fn((n_rel, n_rel), |_| rng.random_range(-1.0..1.0));

        let actual = exciton_dipole_cross(&e_bra, &e_ket, &x_rel, n_com);
        let mut expected = Array2::<c64>::zeros((n_ex, n_ex));
        for i in 0..n_ex {
            for j in 0..n_ex {
                for com in 0..n_com {
                    for bra_rel in 0..n_rel {
                        for ket_rel in 0..n_rel {
                            let bra_index = com * n_rel + bra_rel;
                            let ket_index = com * n_rel + ket_rel;
                            expected[[i, j]] += e_bra[[bra_index, i]].conj()
                                * x_rel[[bra_rel, ket_rel]]
                                * e_ket[[ket_index, j]];
                        }
                    }
                }
            }
        }

        let max_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(max_error < 1e-12, "max error = {max_error:e}");
    }
}
