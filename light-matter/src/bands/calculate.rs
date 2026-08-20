use matrix_ops::{
    ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError, Sqrt,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::c64;
use std::f64::EPSILON;
use std::iter::Sum;
use std::path::Path;

use crate::timed_println;

use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPBasisLayout, PPHamiltonian};

/// Calculate the photonic character of an eigenstate.
///
/// Returns a value between 0, purely excitonic, and 1, photonic.
pub fn calculate_photonic_character<M, B>(
    pp: &PPHamiltonian<M, B>,
    eigenvector: &Array1<c64>,
) -> Result<f64, MatrixOpsError>
where
    M: MatrixOps,
    B: Boson<M>,
{
    let n_ex = pp.polariton.exciton.n_ex_eig;
    let n_ph = pp.polariton.photon.dimension();

    let n_pn = if pp.pn_coupling < EPSILON || pp.phonon.dimension() == 1 {
        1
    } else {
        pp.phonon.dimension()
    };

    calculate_photonic_character_with_layout(eigenvector, pp.layout, n_ex, n_ph, n_pn)
}

pub fn calculate_photonic_character_with_layout(
    eigenvector: &Array1<c64>,
    layout: PPBasisLayout,
    n_ex: usize,
    n_ph: usize,
    n_pn: usize,
) -> Result<f64, MatrixOpsError> {
    let expected_len = layout.pp_dim(n_ex, n_ph, n_pn)?;

    if eigenvector.len() != expected_len {
        return Err(MatrixOpsError::DimensionMismatch {
            expected: expected_len,
            got: eigenvector.len(),
            context: "photonic character eigenvector length".to_string(),
        });
    }

    let mut photonic_prob = 0.0;
    let mut total_prob = 0.0;

    for ex_idx in 0..n_ex {
        for ph_idx in 0..n_ph {
            for pn_idx in 0..n_pn {
                let idx = layout.pp_index(ex_idx, ph_idx, pn_idx, n_ex, n_ph, n_pn)?;

                let prob = eigenvector[idx].norm_sqr();

                if ph_idx > 0 {
                    photonic_prob += prob;
                }

                total_prob += prob;
            }
        }
    }

    if total_prob > 0.0 {
        Ok(photonic_prob / total_prob)
    } else {
        Ok(0.0)
    }
}

pub type ColorBandData = sim_common::data::ColoredBandData;

pub fn load_color_band_structure<M, B: Boson<M>>(
    pp: &PPHamiltonian<M, B>,
    n_k_points: usize,
    n_eig: usize,
    pp_config: &PPConfig,
    zoomed: bool,
) -> (Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>)
where
    M: MatrixOps,
{
    let save_dir = "pp_data".to_string();
    std::fs::create_dir_all(&save_dir).unwrap_or_else(|_| {});

    // Generate filename for the color band structure data
    let mut filename = format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{}_nmodes{}_pncoup{}_npn{}_neig{}",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        pp_config.lm_couplings.iter().sum::<f64>(),
        pp.polariton.photon.boson.get_modes().len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig
    );

    filename += if zoomed { "_zoomed.bin" } else { ".bin" };

    timed_println!("Loading cached color band structure data from {}", filename);
    match ColorBandData::load_from_file(&filename) {
        Ok(data) => return data.to_results(),
        Err(e) => panic!("Failed to load cached data: {}, recalculating...", e),
    }
}

pub fn generate_band_k_points(
    n_k_points: usize,
    sym_points: &[Vec<f64>],
    a0: f64,
    zoomed: bool,
) -> Vec<Vec<f64>> {
    sim_common::kpath::generic_closed_path(n_k_points, sym_points, a0, zoomed)
}

pub fn color_band_filename(
    pp_config: &PPConfig,
    save_dir: &str,
    n_k_points: usize,
    n_eig: usize,
    zoomed: bool,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        total_coupling,
        pp_config.q_values.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig,
        if zoomed { "_zoomed" } else { "" },
    )
}

pub fn color_band_chunk_filename(
    pp_config: &PPConfig,
    save_dir: &str,
    n_k_points: usize,
    n_eig: usize,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_chunk_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_neig{}_chunk{}_of{}{}.bin",
        save_dir,
        pp_config.e_config.a0,
        pp_config.e_config.v0,
        pp_config.n_rel,
        pp_config.n_bands,
        n_k_points,
        total_coupling,
        pp_config.q_values.len(),
        pp_config.pn_coupling,
        pp_config.pn_n_fock,
        n_eig,
        chunk_index,
        n_chunks,
        if zoomed { "_zoomed" } else { "" },
    )
}

pub fn calculate_color_band_structure_distributed_chunk<M, B>(
    pp: &mut PPHamiltonian<M, B>,
    n_k_points: usize,
    n_eig: usize,
    sym_points: &[Vec<f64>],
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError>
where
    M: MatrixOps + Diagonalize,
    M::Scalar: Sqrt + Sum + ConvertFrom<f64> + ConvertFrom<c64>,
    B: Boson<M>,
{
    let rank = sim_common::mpi::mpi_rank_env_or_zero();
    let world_size = sim_common::mpi::mpi_size_env_or_one();

    let save_dir = "pp_data";
    std::fs::create_dir_all(save_dir).ok();

    let k_points = generate_band_k_points(n_k_points, sym_points, pp_config.e_config.a0, zoomed);

    let total_k_points = k_points.len();

    let (start_idx, end_idx) = sim_common::mpi::chunk_bounds(total_k_points, chunk_index, n_chunks);

    let chunk_filename = color_band_chunk_filename(
        pp_config,
        save_dir,
        n_k_points,
        n_eig,
        zoomed,
        chunk_index,
        n_chunks,
    );

    if rank == 0 {
        timed_println!(
            "Bandstructure distributed chunk {}/{}: K indices {}..{} of {} using {} MPI ranks",
            chunk_index,
            n_chunks,
            start_idx,
            end_idx,
            total_k_points,
            world_size,
        );
    }

    // Important: every rank checks the same cache path and exits consistently.
    if use_cached && Path::new(&chunk_filename).exists() {
        if rank == 0 {
            timed_println!("Loading cached band chunk from {}", chunk_filename);
            let data = ColorBandData::load_from_file(&chunk_filename).map_err(|e| {
                MatrixOpsError::Other {
                    message: format!("Failed to load cached chunk {}: {}", chunk_filename, e),
                }
            })?;
            return Ok(data.to_results());
        } else {
            return Ok((vec![], vec![], vec![]));
        }
    }

    let mut chunk_k_points = Vec::new();
    let mut chunk_eigvals = Vec::new();
    let mut chunk_photonic = Vec::new();

    let mut prev_eigvecs: Option<Array2<c64>> = None;

    for global_idx in start_idx..end_idx {
        let k = &k_points[global_idx];

        if rank == 0 {
            timed_println!(
                "Chunk {}/{}: solving K index {} / {}: {:?}",
                chunk_index,
                n_chunks,
                global_idx + 1,
                total_k_points,
                k,
            );
        }

        // All ranks in this Slurm array task build the same K Hamiltonian
        // and enter the same collective SLEPc solve.
        let h_pp = pp.build_pp_hamiltonian(Some(k.clone()))?;

        let mut options = DiagonalizationOptions::smallest_real(n_eig)
            .backend(matrix_ops::DiagonalizationBackend::Auto)
            .hermitian()
            .max_iterations(50_000)
            .tolerance(1e-10);

        options.initial_guess = prev_eigvecs.clone();

        let result = h_pp.diagonalize(options)?;
        let (eigvals, eigvecs) = result.into_arrays_hermitian();

        let n_returned = eigvals.len();

        let mut photonic_character = Vec::with_capacity(n_returned);

        for i in 0..n_returned {
            photonic_character.push(calculate_photonic_character(
                pp,
                &eigvecs.column(i).to_owned(),
            )?);
        }

        prev_eigvecs = Some(eigvecs);

        if rank == 0 {
            chunk_k_points.push(k.clone());
            chunk_eigvals.push(eigvals);
            chunk_photonic.push(photonic_character);
        }
    }

    if rank == 0 {
        let chunk_data = ColorBandData::from_results(
            chunk_k_points.clone(),
            chunk_eigvals.clone(),
            chunk_photonic.clone(),
        );

        chunk_data
            .save_to_file(&chunk_filename)
            .map_err(|e| MatrixOpsError::Other {
                message: format!("Failed to save band chunk {}: {}", chunk_filename, e),
            })?;

        timed_println!("Saved band chunk to {}", chunk_filename);

        Ok((chunk_k_points, chunk_eigvals, chunk_photonic))
    } else {
        Ok((vec![], vec![], vec![]))
    }
}

pub fn gather_color_band_structure_chunks(
    n_k_points: usize,
    n_eig: usize,
    pp_config: &PPConfig,
    zoomed: bool,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError> {
    let save_dir = "pp_data";
    std::fs::create_dir_all(save_dir).ok();

    let mut all_k_points = Vec::new();
    let mut all_eigvals = Vec::new();
    let mut all_photonic = Vec::new();

    for chunk_index in 0..n_chunks {
        let chunk_filename = color_band_chunk_filename(
            pp_config,
            save_dir,
            n_k_points,
            n_eig,
            zoomed,
            chunk_index,
            n_chunks,
        );

        timed_println!("Loading band chunk {}", chunk_filename);

        let data =
            ColorBandData::load_from_file(&chunk_filename).map_err(|e| MatrixOpsError::Other {
                message: format!("Failed to load chunk file {}: {}", chunk_filename, e),
            })?;

        let (k, eig, phot) = data.to_results();

        all_k_points.extend(k);
        all_eigvals.extend(eig);
        all_photonic.extend(phot);
    }

    let final_filename = color_band_filename(pp_config, save_dir, n_k_points, n_eig, zoomed);

    let final_data = ColorBandData::from_results(
        all_k_points.clone(),
        all_eigvals.clone(),
        all_photonic.clone(),
    );

    final_data
        .save_to_file(&final_filename)
        .map_err(|e| MatrixOpsError::Other {
            message: format!(
                "Failed to save gathered bandstructure {}: {}",
                final_filename, e
            ),
        })?;

    timed_println!("Saved gathered bandstructure to {}", final_filename);

    Ok((all_k_points, all_eigvals, all_photonic))
}
