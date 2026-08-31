use super::keys::{ModeKey, ModeTask};
use crate::modes::types::BosonMode;
use log::{info, warn};
use ndarray::Array2;
use ndarray_linalg::c64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub type SharedC64Matrix = Arc<Array2<c64>>;
pub type MatrixElementMap = HashMap<ModeKey, SharedC64Matrix>;

#[derive(Clone)]
pub struct LMMatrixElementArray {
    pub p_cos_mat: MatrixElementMap, // Momentum-cosine matrix elements
    pub cos_mat: MatrixElementMap,   // Cosine matrix elements
    pub sin_mat: MatrixElementMap,   // Sine matrix elements

    pub p_cos_mat_rel: Option<MatrixElementMap>,
    pub cos_mat_rel: Option<MatrixElementMap>,
    pub sin_mat_rel: Option<MatrixElementMap>,
}

impl LMMatrixElementArray {
    pub fn new() -> Self {
        Self {
            cos_mat: HashMap::new(),
            sin_mat: HashMap::new(),
            p_cos_mat: HashMap::new(),
            cos_mat_rel: None,
            p_cos_mat_rel: None,
            sin_mat_rel: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_mat.is_empty() && self.sin_mat.is_empty() && self.p_cos_mat.is_empty()
    }

    pub fn clear_relative_cache(&mut self) {
        self.cos_mat_rel = None;
        self.sin_mat_rel = None;
        self.p_cos_mat_rel = None;
    }

    pub fn clear_full_cache_maps(&mut self) {
        self.cos_mat.clear();
        self.sin_mat.clear();
        self.p_cos_mat.clear();
    }

    pub fn populate_taylor_matrix_elements_with_cache(
        &mut self,
        all_q_ind: &Vec<ModeTask>,
        e_vecs: &Array2<c64>,
        modes: &Vec<BosonMode>,
        n_com: usize,
        mu: f64,
        cache_dir: Option<&str>,
        force_recompute: bool,
        relative_config_hash: u64,
        exciton_config_hash: u64,
    ) {
        let n_rel = e_vecs.nrows() / n_com;

        // Try to load full exciton-basis cache.
        // If this succeeds, do NOT load relative-basis cache into memory.
        if let Some(dir) = cache_dir {
            if !force_recompute {
                let cache_file =
                    Self::get_cache_filename(dir, mu, n_com, n_rel, exciton_config_hash);

                if Path::new(&cache_file).exists() {
                    match Self::load_from_file(&cache_file) {
                        Ok(cached) => {
                            info!("Successfully loaded matrix elements from cache");

                            self.cos_mat = cached.cos_mat;
                            self.sin_mat = cached.sin_mat;
                            self.p_cos_mat = cached.p_cos_mat;

                            // Critical memory fix:
                            // relative cache is only needed to construct the full cache.
                            self.clear_relative_cache();

                            return;
                        }
                        Err(e) => {
                            warn!(
                                "Failed to load cache file {}: {}. Recomputing...",
                                cache_file, e
                            );
                        }
                    }
                }
            }
        }

        // Compute or load relative-basis matrices only because the full cache was
        // not available.
        if self.cos_mat_rel.is_none() || self.sin_mat_rel.is_none() || self.p_cos_mat_rel.is_none()
        {
            let mut loaded_from_cache = false;

            if let Some(dir) = cache_dir {
                if !force_recompute {
                    let cache_file_rel =
                        Self::get_cache_filename_rel(dir, mu, n_rel, relative_config_hash);

                    if Path::new(&cache_file_rel).exists() {
                        match Self::load_rel_from_file(&cache_file_rel) {
                            Ok((cos_rel, sin_rel, p_cos_rel)) => {
                                if Self::has_all_required_relative_keys(
                                    &cos_rel, &sin_rel, &p_cos_rel, all_q_ind,
                                ) {
                                    info!(
                                        "Successfully loaded relative-basis matrix elements from cache"
                                    );
                                    self.cos_mat_rel = Some(cos_rel);
                                    self.sin_mat_rel = Some(sin_rel);
                                    self.p_cos_mat_rel = Some(p_cos_rel);
                                    loaded_from_cache = true;
                                } else {
                                    warn!(
                                        "Relative-basis cache {} is incomplete for this mode set; recomputing",
                                        cache_file_rel
                                    );
                                }
                            }
                            Err(e) => {
                                warn!("Failed to load relative-basis cache: {}. Recomputing...", e);
                            }
                        }
                    }
                }
            }

            if !loaded_from_cache {
                let (cos_mat_rel, sin_mat_rel, p_cos_mat_rel) =
                    self.compute_relative_tay_mat_elements(all_q_ind, modes, n_rel, mu);

                self.cos_mat_rel = Some(cos_mat_rel);
                self.sin_mat_rel = Some(sin_mat_rel);
                self.p_cos_mat_rel = Some(p_cos_mat_rel);

                if sim_common::mpi::is_mpi_rank0_env() {
                    if let Some(dir) = cache_dir {
                        let cache_file_rel =
                            Self::get_cache_filename_rel(dir, mu, n_rel, relative_config_hash);
                        if let Err(e) = self.save_rel_to_file(&cache_file_rel) {
                            warn!("Failed to save relative-basis cache: {}", e);
                        }
                    }
                }
            }
        }

        let cos_rel = self.cos_mat_rel.as_ref().unwrap();
        let sin_rel = self.sin_mat_rel.as_ref().unwrap();
        let p_cos_rel = self.p_cos_mat_rel.as_ref().unwrap();

        let task_chunk_size = 2048usize;

        for chunk in all_q_ind.chunks(task_chunk_size) {
            let results: Vec<(ModeTask, Array2<c64>, Array2<c64>)> = chunk
                .par_iter()
                .map(|task| match *task {
                    ModeTask::PhotonSingle { mode } => {
                        let key = ModeKey::single(mode);

                        let sin = sin_rel.get(&key).unwrap_or_else(|| {
                            panic!("Missing relative sine matrix for {:?}", key)
                        });

                        let p_cos = p_cos_rel.get(&key).unwrap_or_else(|| {
                            panic!("Missing relative p_cos matrix for {:?}", key)
                        });

                        (
                            *task,
                            self.to_exciton_basis(p_cos.as_ref(), e_vecs, n_com),
                            self.to_exciton_basis(sin.as_ref(), e_vecs, n_com),
                        )
                    }

                    ModeTask::PhotonPair { q1, q2 } => {
                        let diff_key = ModeKey::diff(q1, q2);
                        let sum_key = ModeKey::sum(q1, q2);

                        let cos_diff = cos_rel.get(&diff_key).unwrap_or_else(|| {
                            panic!("Missing relative cos-diff matrix for {:?}", diff_key)
                        });

                        let cos_sum = cos_rel.get(&sum_key).unwrap_or_else(|| {
                            panic!("Missing relative cos-sum matrix for {:?}", sum_key)
                        });

                        (
                            *task,
                            self.to_exciton_basis(cos_diff.as_ref(), e_vecs, n_com),
                            self.to_exciton_basis(cos_sum.as_ref(), e_vecs, n_com),
                        )
                    }
                })
                .collect();

            for (task, mat_1, mat_2) in results {
                match task {
                    ModeTask::PhotonSingle { mode } => {
                        let key = ModeKey::single(mode);

                        self.p_cos_mat.insert(key, Arc::new(mat_1));
                        self.sin_mat.insert(key, Arc::new(mat_2));
                    }

                    ModeTask::PhotonPair { q1, q2 } => {
                        self.cos_mat.insert(ModeKey::diff(q1, q2), Arc::new(mat_1));
                        self.cos_mat.insert(ModeKey::sum(q1, q2), Arc::new(mat_2));
                    }
                }
            }
        }

        if sim_common::mpi::is_mpi_rank0_env() {
            if let Some(dir) = cache_dir {
                let cache_file =
                    Self::get_cache_filename(dir, mu, n_com, n_rel, exciton_config_hash);
                if let Err(e) = self.save_to_file(&cache_file) {
                    warn!("Failed to save matrix elements to cache: {}", e);
                }
            }
        }

        // Critical memory fix:
        // after full exciton-basis matrices exist, relative-basis matrices are no
        // longer needed in memory.
        self.clear_relative_cache();
    }
}
