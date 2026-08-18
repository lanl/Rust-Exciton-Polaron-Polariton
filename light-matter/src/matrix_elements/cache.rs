use super::builder::LMMatrixElementArray;
use super::keys::{ModeKey, ModeTask};

use log::info;
use ndarray::Array2;
use ndarray_linalg::c64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::Path;

/// Serializable version of matrix element arrays for caching.
#[derive(Serialize, Deserialize)]
struct SerializableMatrixElements {
    cos_mat: Vec<(ModeKey, Vec<c64>)>,
    sin_mat: Vec<(ModeKey, Vec<c64>)>,
    p_cos_mat: Vec<(ModeKey, Vec<c64>)>,
    matrix_shape: (usize, usize),
}

impl LMMatrixElementArray {
    /// Generate a cache filename based on parameters
    /// Uses config_hash which includes all mode information from PPConfig
    pub fn get_cache_filename(
        cache_dir: &str,
        mu: f64,
        n_com: usize,
        n_rel: usize,
        config_hash: u64,
    ) -> String {
        format!(
            "{}/matrix_elements_v2_mu{:.6}_ncom{}_nrel{}_confighash{:016x}.bin",
            cache_dir, mu, n_com, n_rel, config_hash
        )
    }

    /// Generate a cache filename for relative basis matrices (independent of eigenvectors)
    /// Uses config_hash which includes all mode information from PPConfig
    pub fn get_cache_filename_rel(
        cache_dir: &str,
        mu: f64,
        n_rel: usize,
        config_hash: u64,
    ) -> String {
        format!(
            "{}/matrix_elements_rel_v2_mu{:.6}_nrel{}_confighash{:016x}.bin",
            cache_dir, mu, n_rel, config_hash
        )
    }

    /// Create a deterministic hash from exciton configuration parameters
    /// This hashes the physical parameters that determine the eigenvectors,
    /// avoiding issues with gauge freedom (global phases) in the eigenvectors themselves
    pub fn hash_exciton_config(
        a0: f64,
        v0: f64,
        mu: f64,
        n_rel: usize,
        n_ex_eig: usize,
        n_com: usize,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash exciton parameters
        a0.to_bits().hash(&mut hasher);
        v0.to_bits().hash(&mut hasher);
        mu.to_bits().hash(&mut hasher);
        n_rel.hash(&mut hasher);
        n_ex_eig.hash(&mut hasher);
        n_com.hash(&mut hasher);

        hasher.finish()
    }

    /// Save matrix elements to cache file
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(filename).parent() {
            fs::create_dir_all(parent)?;
        }

        // Convert HashMaps to serializable format
        let mut cos_vec = Vec::new();
        let mut sin_vec = Vec::new();
        let mut p_cos_vec = Vec::new();

        let shape = self
            .cos_mat
            .values()
            .chain(self.sin_mat.values())
            .chain(self.p_cos_mat.values())
            .next()
            .map(|m| (m.nrows(), m.ncols()))
            .unwrap_or((0, 0));

        for (key, mat) in &self.cos_mat {
            cos_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }
        for (key, mat) in &self.sin_mat {
            sin_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }
        for (key, mat) in &self.p_cos_mat {
            p_cos_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }

        let serializable = SerializableMatrixElements {
            cos_mat: cos_vec,
            sin_mat: sin_vec,
            p_cos_mat: p_cos_vec,
            matrix_shape: shape,
        };

        let encoded = bincode::serialize(&serializable)?;
        crate::utils::atomic_write(filename, &encoded)?;

        info!("Saved matrix elements to cache: {}", filename);
        Ok(())
    }

    /// Load matrix elements from cache file
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let serializable: SerializableMatrixElements = bincode::deserialize(&buffer)?;

        let mut cos_mat = HashMap::new();
        let mut sin_mat = HashMap::new();
        let mut p_cos_mat = HashMap::new();

        let (nrows, ncols) = serializable.matrix_shape;

        for (key, data) in serializable.cos_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            cos_mat.insert(key, mat);
        }
        for (key, data) in serializable.sin_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            sin_mat.insert(key, mat);
        }
        for (key, data) in serializable.p_cos_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            p_cos_mat.insert(key, mat);
        }

        info!("Loaded matrix elements from cache: {}", filename);
        Ok(Self {
            cos_mat,
            sin_mat,
            p_cos_mat,
            cos_mat_rel: None,
            sin_mat_rel: None,
            p_cos_mat_rel: None,
        })
    }

    /// Save relative basis matrix elements to cache file
    pub fn save_rel_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.cos_mat_rel.is_none() || self.sin_mat_rel.is_none() || self.p_cos_mat_rel.is_none()
        {
            return Err("Relative basis matrices not computed".into());
        }

        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(filename).parent() {
            fs::create_dir_all(parent)?;
        }

        let cos_mat_rel = self.cos_mat_rel.as_ref().unwrap();
        let sin_mat_rel = self.sin_mat_rel.as_ref().unwrap();
        let p_cos_mat_rel = self.p_cos_mat_rel.as_ref().unwrap();

        let mut cos_vec = Vec::new();
        let mut sin_vec = Vec::new();
        let mut p_cos_vec = Vec::new();

        let shape = cos_mat_rel
            .values()
            .chain(sin_mat_rel.values())
            .chain(p_cos_mat_rel.values())
            .next()
            .map(|m| (m.nrows(), m.ncols()))
            .unwrap_or((0, 0));

        for (key, mat) in cos_mat_rel {
            cos_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }
        for (key, mat) in sin_mat_rel {
            sin_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }
        for (key, mat) in p_cos_mat_rel {
            p_cos_vec.push((key.clone(), mat.iter().cloned().collect::<Vec<_>>()));
        }

        let serializable = SerializableMatrixElements {
            cos_mat: cos_vec,
            sin_mat: sin_vec,
            p_cos_mat: p_cos_vec,
            matrix_shape: shape,
        };

        let encoded = bincode::serialize(&serializable)?;
        crate::utils::atomic_write(filename, &encoded)?;

        info!(
            "Saved relative basis matrix elements to cache: {}",
            filename
        );
        Ok(())
    }

    /// Load relative basis matrix elements from cache file
    pub fn load_rel_from_file(
        filename: &str,
    ) -> Result<
        (
            HashMap<ModeKey, Array2<c64>>,
            HashMap<ModeKey, Array2<c64>>,
            HashMap<ModeKey, Array2<c64>>,
        ),
        Box<dyn std::error::Error>,
    > {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let serializable: SerializableMatrixElements = bincode::deserialize(&buffer)?;

        let mut cos_mat = HashMap::new();
        let mut sin_mat = HashMap::new();
        let mut p_cos_mat = HashMap::new();

        let (nrows, ncols) = serializable.matrix_shape;

        for (key, data) in serializable.cos_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            cos_mat.insert(key, mat);
        }
        for (key, data) in serializable.sin_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            sin_mat.insert(key, mat);
        }
        for (key, data) in serializable.p_cos_mat {
            let mat = Array2::from_shape_vec((nrows, ncols), data)?;
            p_cos_mat.insert(key, mat);
        }

        info!(
            "Loaded relative basis matrix elements from cache: {}",
            filename
        );
        Ok((cos_mat, sin_mat, p_cos_mat))
    }

    pub(crate) fn has_all_required_relative_keys(
        cos_rel: &HashMap<ModeKey, Array2<c64>>,
        sin_rel: &HashMap<ModeKey, Array2<c64>>,
        p_cos_rel: &HashMap<ModeKey, Array2<c64>>,
        all_q_ind: &[ModeTask],
    ) -> bool {
        for task in all_q_ind {
            match *task {
                ModeTask::PhotonSingle { mode } => {
                    let key = ModeKey::single(mode);

                    if !sin_rel.contains_key(&key) || !p_cos_rel.contains_key(&key) {
                        return false;
                    }
                }

                ModeTask::PhotonPair { q1, q2 } => {
                    let diff_key = ModeKey::diff(q1, q2);
                    let sum_key = ModeKey::sum(q1, q2);

                    if !cos_rel.contains_key(&diff_key) || !cos_rel.contains_key(&sum_key) {
                        return false;
                    }
                }
            }
        }

        true
    }
}
