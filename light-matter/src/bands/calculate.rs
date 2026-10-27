use matrix_ops::{
    BatchPolicy, ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError,
    RealCutoff, Sqrt,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::c64;
use std::f64::EPSILON;
use std::iter::Sum;
use std::path::Path;

use crate::{timed_println, timed_println_rank0};
use crate::utils::eig_selection_tag;
use crate::checkpoint::DiagCheckpoint;

use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPBasisLayout, PPHamiltonian};

const BAND_SAVE_DIR: &str = "pp_data";

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

/// Load a gathered color band structure written by
/// `gather_color_band_structure_chunks`.
pub fn load_color_band_structure(
    pp_config: &PPConfig,
    n_k_points: usize,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    zoomed: bool,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError> {
    std::fs::create_dir_all(BAND_SAVE_DIR).ok();

    let filename = color_band_filename(
        pp_config,
        BAND_SAVE_DIR,
        n_k_points,
        n_eig,
        energy_cutoff,
        zoomed,
    );

    timed_println!("Loading cached color band structure data from {}", filename);

    let data = ColorBandData::load_from_file(&filename).map_err(|e| MatrixOpsError::Other {
        message: format!(
            "Failed to load gathered band structure {}: {} \
             (run with --calculate true, or check --n-eig / --energy-cutoff)",
            filename, e
        ),
    })?;

    Ok(data.to_results())
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
    energy_cutoff: Option<f64>,
    zoomed: bool,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_{}{}{}.bin",
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
        eig_selection_tag(n_eig, energy_cutoff),
        if !pp_config.include_photon_recoil { "_lwa" } else { "" },
        if zoomed { "_zoomed" } else { "" },
    )
}

pub fn color_band_chunk_filename(
    pp_config: &PPConfig,
    save_dir: &str,
    n_k_points: usize,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
) -> String {
    let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

    format!(
        "{}/color_band_chunk_a0{}_V0{}_nrel{}_nbands{}_npoints{}_phcoup{:.6}_nmodes{}_pncoup{}_npn{}_{}_chunk{}_of{}{}{}.bin",
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
        eig_selection_tag(n_eig, energy_cutoff),
        chunk_index,
        n_chunks,
        if !pp_config.include_photon_recoil { "_lwa" } else { "" },
        if zoomed { "_zoomed" } else { "" },
    )
}

/// Eigensolver selection for one K point.
///
/// With a cutoff, all eigenvalues <= cutoff are computed and `n_eig` only
/// seeds the batch size of the density estimate. Eigenvectors are gathered
/// to rank 0 only (same as the dielectric path); other ranks receive
/// eigenvalues but an empty eigenvector matrix.
fn band_diagonalization_options(
    n_eig: usize,
    energy_cutoff: Option<f64>,
) -> DiagonalizationOptions {
    let options = match energy_cutoff {
        Some(cutoff) => DiagonalizationOptions::all_real_below(cutoff)
            .real_cutoff(RealCutoff {
                batch_policy: BatchPolicy::DensityEstimate {
                    initial: n_eig.clamp(32, 1024) as _,
                    safety: 1.25,
                    min_batch: 32,
                    max_batch: 1024,
                },
                ..RealCutoff::new(cutoff)
            })
            .deflation_batch_size(512),
        None => DiagonalizationOptions::smallest_real(n_eig),
    };

    options
        .backend(matrix_ops::DiagonalizationBackend::Auto)
        .hermitian()
        .max_iterations(50_000)
        .tolerance(1e-10)
}

pub fn calculate_color_band_structure_distributed_chunk<M, B>(
    pp: &mut PPHamiltonian<M, B>,
    n_k_points: usize,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    sym_points: &[Vec<f64>],
    pp_config: &PPConfig,
    use_cached: bool,
    zoomed: bool,
    chunk_index: usize,
    n_chunks: usize,
    checkpoint: Option<&DiagCheckpoint>,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError>
where
    M: MatrixOps + Diagonalize,
    M::Scalar: Sqrt + Sum + ConvertFrom<f64> + ConvertFrom<c64>,
    B: Boson<M>,
{
    let rank = sim_common::mpi::mpi_rank_env_or_zero();
    let world_size = sim_common::mpi::mpi_size_env_or_one();
    let is_root = rank == 0;

    let warm_start = energy_cutoff.is_none();
    let ckpt = checkpoint.zip(energy_cutoff);

    std::fs::create_dir_all(BAND_SAVE_DIR).ok();

    let k_points = generate_band_k_points(n_k_points, sym_points, pp_config.e_config.a0, zoomed);
    let total_k_points = k_points.len();
    let (start_idx, end_idx) = sim_common::mpi::chunk_bounds(total_k_points, chunk_index, n_chunks);

    let chunk_filename = color_band_chunk_filename(
        pp_config,
        BAND_SAVE_DIR,
        n_k_points,
        n_eig,
        energy_cutoff,
        zoomed,
        chunk_index,
        n_chunks,
    );

    timed_println_rank0!(
        "Bandstructure distributed chunk {}/{}: K indices {}..{} of {} using {} MPI ranks",
        chunk_index,
        n_chunks,
        start_idx,
        end_idx,
        total_k_points,
        world_size,
    );

    match energy_cutoff {
        Some(cutoff) => timed_println_rank0!(
            "Eigenstate selection: ALL states with E <= {:.6e} (n_eig = {} used only as batch hint); \
             eigenvectors on rank 0 only, warm start disabled, checkpoint = {}",
            cutoff,
            n_eig,
            ckpt.is_some()
        ),
        None => timed_println_rank0!(
            "Eigenstate selection: lowest {} states (warm start enabled)",
            n_eig
        ),
    }

    if use_cached && Path::new(&chunk_filename).exists() {
        if is_root {
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

        timed_println_rank0!(
            "Chunk {}/{}: solving K index {} / {}: {:?}",
            chunk_index,
            n_chunks,
            global_idx + 1,
            total_k_points,
            k,
        );

        // Per-K checkpoint config. Must be prepared BEFORE build_pp_hamiltonian,
        // which recomputes the exciton basis when K changes.
        let ckpt_cfg = ckpt.map(|(c, cut)| {
            let mut cfg_k = pp_config.clone();
            cfg_k.e_config.k = k.clone();
            c.prepare(pp, &cfg_k, &format!("band_k{global_idx}"), cut)
        });

        let (eigvals, eigvecs) = {
            let h_pp = pp.build_pp_hamiltonian(Some(k.clone()))?;

            pp.polariton.matrix_elements.clear_relative_cache();
            pp.polariton.matrix_elements.clear_full_cache_maps();

            let mut options = band_diagonalization_options(n_eig, energy_cutoff);
            if energy_cutoff.is_some() {
                options = options.eigenvectors_rank0_only();
            }
            if let Some(cfg) = ckpt_cfg {
                options = options.checkpoint(cfg);
            }
            options.initial_guess = if warm_start { prev_eigvecs.take() } else { None };

            h_pp.diagonalize(options)?.into_arrays_hermitian()
        };

        let n_returned = eigvals.len();

        if !is_root {
            if warm_start {
                prev_eigvecs = (eigvecs.ncols() == n_returned && n_returned > 0).then_some(eigvecs);
            }
            continue;
        }

        if eigvecs.ncols() != n_returned {
            return Err(MatrixOpsError::DimensionMismatch {
                expected: n_returned,
                got: eigvecs.ncols(),
                context: format!(
                    "bandstructure (rank 0): eigensolver returned {} eigenvalues but {} eigenvectors \
                     at K index {} (cutoff = {:?}); eigenvectors were not gathered to rank 0",
                    n_returned,
                    eigvecs.ncols(),
                    global_idx,
                    energy_cutoff
                ),
            });
        }

        let mut order: Vec<usize> = (0..n_returned)
            .filter(|&i| {
                let e = eigvals[i];
                e.is_finite() && energy_cutoff.map_or(true, |cutoff| e <= cutoff)
            })
            .collect();
        order.sort_by(|&a, &b| eigvals[a].total_cmp(&eigvals[b]));

        let mut kept_eigvals = Vec::with_capacity(order.len());
        let mut photonic_character = Vec::with_capacity(order.len());
        for &i in &order {
            kept_eigvals.push(eigvals[i]);
            photonic_character.push(calculate_photonic_character(
                pp,
                &eigvecs.column(i).to_owned(),
            )?);
        }

        let kept = Array1::from_vec(kept_eigvals);
        let n_dropped = n_returned - kept.len();

        if kept.is_empty() {
            timed_println!(
                "  Warning: no eigenstates kept at K index {} (returned {}, cutoff = {:?})",
                global_idx,
                n_returned,
                energy_cutoff
            );
        } else {
            timed_println!(
                "  Kept {} eigenpairs (dropped {}), E in [{:.6e}, {:.6e}]",
                kept.len(),
                n_dropped,
                kept[0],
                kept[kept.len() - 1]
            );
        }

        if warm_start {
            prev_eigvecs = (eigvecs.ncols() > 0).then_some(eigvecs);
        }

        chunk_k_points.push(k.clone());
        chunk_eigvals.push(kept);
        chunk_photonic.push(photonic_character);
    }

    if is_root {
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

        // Chunk is on disk: per-K eigenvector checkpoints can go.
        if let Some(c) = checkpoint {
            c.finish();
        }

        Ok((chunk_k_points, chunk_eigvals, chunk_photonic))
    } else {
        Ok((vec![], vec![], vec![]))
    }
}

pub fn gather_color_band_structure_chunks(
    n_k_points: usize,
    n_eig: usize,
    energy_cutoff: Option<f64>,
    pp_config: &PPConfig,
    zoomed: bool,
    n_chunks: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>, Vec<Vec<f64>>), MatrixOpsError> {
    std::fs::create_dir_all(BAND_SAVE_DIR).ok();

    let mut all_k_points = Vec::new();
    let mut all_eigvals = Vec::new();
    let mut all_photonic = Vec::new();

    for chunk_index in 0..n_chunks {
        let chunk_filename = color_band_chunk_filename(
            pp_config,
            BAND_SAVE_DIR,
            n_k_points,
            n_eig,
            energy_cutoff,
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

    if let (Some(min_count), Some(max_count)) = (
        all_eigvals.iter().map(|e| e.len()).min(),
        all_eigvals.iter().map(|e| e.len()).max(),
    ) {
        timed_println!(
            "Gathered {} K points; states per K point: min = {}, max = {}",
            all_k_points.len(),
            min_count,
            max_count
        );
    }

    let final_filename = color_band_filename(
        pp_config,
        BAND_SAVE_DIR,
        n_k_points,
        n_eig,
        energy_cutoff,
        zoomed,
    );

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
