use ndarray_linalg::{Norm, c64};
use num_traits::{One, Zero};
use scilib::math::basic::factorial;
use std::{collections::HashMap, f64::consts::PI};
use rayon::prelude::*;
use log::{warn, info};
use ndarray::{Array1, Array2};
use crate::utils::{BosonMode};
use matrix_ops::{HermitianConjugate};
use std::fs::{self, File};
use std::io::{Write, Read};
use std::path::Path;
use serde::{Serialize, Deserialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::timed_println;

/// Serializable version of matrix element arrays for caching
#[derive(Serialize, Deserialize)]
struct SerializableMatrixElements {
    cos_mat: Vec<(Vec<i32>, Vec<c64>)>,
    sin_mat: Vec<(Vec<i32>, Vec<c64>)>,
    p_cos_mat: Vec<(Vec<i32>, Vec<c64>)>,
    matrix_shape: (usize, usize),
}

#[derive(Clone)]
pub struct LMMatrixElementArray {
    pub p_cos_mat: HashMap<Vec<i32>, Array2<c64>>, // Momentum Cosine matrix elements
    pub cos_mat: HashMap<Vec<i32>, Array2<c64>>, // Cosine matrix elements
    pub sin_mat: HashMap<Vec<i32>, Array2<c64>>, // Sine matrix elements
    pub p_cos_mat_rel: Option<HashMap<Vec<i32>, Array2<c64>>>, // Momentum Cosine matrix elements in relative basis
    pub cos_mat_rel:   Option<HashMap<Vec<i32>, Array2<c64>>>, // Cosine matrix elements in relative basis
    pub sin_mat_rel:   Option<HashMap<Vec<i32>, Array2<c64>>>, // Sine matrix elements in relative basis
}

impl LMMatrixElementArray {
    pub fn new() -> Self {
        let   cos_mat:     HashMap<Vec<i32>, Array2<c64>> = HashMap::new();
        let p_cos_mat:     HashMap<Vec<i32>, Array2<c64>> = HashMap::new();
        let   sin_mat:     HashMap<Vec<i32>, Array2<c64>> = HashMap::new();

        Self {
            cos_mat:       cos_mat,
            sin_mat:       sin_mat,
            p_cos_mat:     p_cos_mat,
            cos_mat_rel:   None,
            p_cos_mat_rel: None,
            sin_mat_rel:   None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_mat.is_empty() && self.sin_mat.is_empty() && self.p_cos_mat.is_empty()
    }

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
            "{}/matrix_elements_mu{:.6}_ncom{}_nrel{}_confighash{:016x}.bin",
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
            "{}/matrix_elements_rel_mu{:.6}_nrel{}_confighash{:016x}.bin",
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
        let mut shape = (0, 0);

        for (key, mat) in &self.cos_mat {
            shape = (mat.nrows(), mat.ncols());
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
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        
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
        if self.cos_mat_rel.is_none() || self.sin_mat_rel.is_none() || self.p_cos_mat_rel.is_none() {
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
        let mut shape = (0, 0);

        for (key, mat) in cos_mat_rel {
            shape = (mat.nrows(), mat.ncols());
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
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        
        info!("Saved relative basis matrix elements to cache: {}", filename);
        Ok(())
    }

    /// Load relative basis matrix elements from cache file
    pub fn load_rel_from_file(filename: &str) -> Result<(HashMap<Vec<i32>, Array2<c64>>, HashMap<Vec<i32>, Array2<c64>>, HashMap<Vec<i32>, Array2<c64>>), Box<dyn std::error::Error>> {
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
        
        info!("Loaded relative basis matrix elements from cache: {}", filename);
        Ok((cos_mat, sin_mat, p_cos_mat))
    }

    pub fn populate_tay_mat_elements(
        &mut self,
        all_q_ind: &Vec<Vec<i32>>,
        e_vecs: &Array2<c64>,
        modes: &Vec<BosonMode>,
        n_com: usize,
        mu: f64,
    ) {
        self.populate_tay_mat_elements_with_cache(all_q_ind, e_vecs, modes, n_com, mu, None, false, 0);
    }

    pub fn populate_tay_mat_elements_with_cache(
        &mut self,
        all_q_ind: &Vec<Vec<i32>>,
        e_vecs: &Array2<c64>,
        modes: &Vec<BosonMode>,
        n_com: usize,
        mu: f64,
        cache_dir: Option<&str>,
        force_recompute: bool,
        config_hash: u64,
    ) {
        let n_rel = e_vecs.nrows() / n_com;
        
        // Try to load from cache if cache_dir is provided and not forcing recompute
        if let Some(dir) = cache_dir {
            if !force_recompute {
                let cache_file = Self::get_cache_filename(dir, mu, n_com, n_rel, config_hash);
                
                if Path::new(&cache_file).exists() {
                    match Self::load_from_file(&cache_file) {
                        Ok(cached) => {
                            info!("Successfully loaded matrix elements from cache");
                            self.cos_mat = cached.cos_mat;
                            self.sin_mat = cached.sin_mat;
                            self.p_cos_mat = cached.p_cos_mat;
                            
                            // Also try to load relative basis cache
                            let cache_file_rel = Self::get_cache_filename_rel(dir, mu, n_rel, config_hash);
                            if Path::new(&cache_file_rel).exists() {
                                if let Ok((cos_rel, sin_rel, p_cos_rel)) = Self::load_rel_from_file(&cache_file_rel) {
                                    self.cos_mat_rel = Some(cos_rel);
                                    self.sin_mat_rel = Some(sin_rel);
                                    self.p_cos_mat_rel = Some(p_cos_rel);
                                }
                            }
                            return;
                        }
                        Err(e) => {
                            warn!("Failed to load cache file {}: {}. Recomputing...", cache_file, e);
                        }
                    }
                }
            }
        }
        
        // Compute relative basis matrices (with caching)
        if self.cos_mat_rel.is_none() || self.sin_mat_rel.is_none() || self.p_cos_mat_rel.is_none() {
            // Try to load relative basis from cache first
            let mut loaded_from_cache = false;
            if let Some(dir) = cache_dir {
                if !force_recompute {
                    let cache_file_rel = Self::get_cache_filename_rel(dir, mu, n_rel, config_hash);
                    if Path::new(&cache_file_rel).exists() {
                        match Self::load_rel_from_file(&cache_file_rel) {
                            Ok((cos_rel, sin_rel, p_cos_rel)) => {
                                info!("Successfully loaded relative basis matrix elements from cache");
                                self.cos_mat_rel = Some(cos_rel);
                                self.sin_mat_rel = Some(sin_rel);
                                self.p_cos_mat_rel = Some(p_cos_rel);
                                loaded_from_cache = true;
                            }
                            Err(e) => {
                                warn!("Failed to load relative basis cache: {}. Recomputing...", e);
                            }
                        }
                    }
                }
            }
            
            // Compute if not loaded from cache
            if !loaded_from_cache {
                let (cos_mat_rel, sin_mat_rel, p_cos_mat_rel) = self.compute_relative_tay_mat_elements(all_q_ind, modes, n_rel, mu);

                self.cos_mat_rel = Some(cos_mat_rel);
                self.sin_mat_rel = Some(sin_mat_rel);
                self.p_cos_mat_rel = Some(p_cos_mat_rel);
                
                // Save relative basis to cache
                if let Some(dir) = cache_dir {
                    let cache_file_rel = Self::get_cache_filename_rel(dir, mu, n_rel, config_hash);
                    if let Err(e) = self.save_rel_to_file(&cache_file_rel) {
                        warn!("Failed to save relative basis cache: {}", e);
                    }
                }
            }
        }

                // Parallel computation over all q‑index vectors
        let results: Vec<(Vec<i32>, Array2<c64>, Array2<c64>)> = all_q_ind
            .par_iter()
            .enumerate()
            .map(|(q_idx, q_inds)| {
                
                let (matrix1, matrix2) = if q_inds.len() == 1 {
                    // Single‑boson case
                    let sin = self.sin_mat_rel.as_ref().unwrap().get(q_inds).unwrap();
                    if q_idx < modes.len() {
                        // Single Photon
                        let p_cos = self.p_cos_mat_rel.as_ref().unwrap().get(q_inds).unwrap();
                        (self.to_exciton_basis(p_cos, e_vecs, n_com), self.to_exciton_basis(sin, e_vecs, n_com))
                    } else {
                        // Single Phonon
                        let cos = self.cos_mat_rel.as_ref().unwrap().get(q_inds).unwrap();
                        (self.to_exciton_basis(cos, e_vecs, n_com), self.to_exciton_basis(sin, e_vecs, n_com))
                    }
                } else {
                    // Two‑phonon (q₁±q₂) case – only cosine elements are needed
                    let q_ind1 = q_inds[0].max(q_inds[1]);
                    let q_ind2 = q_inds[0].min(q_inds[1]);
                    let cos_dif = self.cos_mat_rel.as_ref().unwrap().get(&vec![q_ind1, -q_ind2]).unwrap();
                    let cos_sum = self.cos_mat_rel.as_ref().unwrap().get(&vec![q_ind1,  q_ind2]).unwrap();
                    (self.to_exciton_basis(cos_dif, e_vecs, n_com), self.to_exciton_basis(cos_sum, e_vecs, n_com))
                };

                // Return the key and the two matrices (the caller will decide where each goes)
                (q_inds.clone(), matrix1, matrix2)
            })
            .collect();

        // Populate the three hash‑maps according to the length of q_inds
        results.iter().enumerate().for_each(|(idx, (q_inds, mat_1, mat_2))| {
            if q_inds.len() == 1 {
                // Single‑phonon entry – decide whether it belongs to p_cos or ordinary cos
                if idx < modes.len() {
                    self.p_cos_mat.insert(q_inds.clone(), mat_1.clone());
                } else {
                    self.cos_mat.insert(q_inds.clone(), mat_1.clone());
                }
                self.sin_mat.insert(q_inds.clone(), mat_2.clone());
            } else if q_inds.len() == 2 {
                // Two‑phonon entry – store both combinations in the cosine map
                let q_ind1 = q_inds[0].max(q_inds[1]);
                let q_ind2 = q_inds[0].min(q_inds[1]);

                self.cos_mat.insert(vec![q_ind1, -q_ind2], mat_1.clone()); // q₁−q₂
                self.cos_mat.insert(vec![q_ind1,  q_ind2], mat_2.clone()); // q₁+q₂
            }
        });
        
        // Save to cache if cache_dir is provided
        if let Some(dir) = cache_dir {
            let cache_file = Self::get_cache_filename(dir, mu, n_com, n_rel, config_hash);
            if let Err(e) = self.save_to_file(&cache_file) {
                warn!("Failed to save matrix elements to cache: {}", e);
            }
        }
    }

    /// Old matrix element population
    pub fn _populate_tay_mat_elements(
        &self,
        all_q_ind: &Vec<Vec<i32>>,
        e_vecs: &Vec<Array1<c64>>,
        modes: &Vec<BosonMode>,
        n_com: usize,
        mu: f64,
    ) -> Self {
        
        let n_eigs = e_vecs.len();
        let n_rel = e_vecs[0].len() / n_com;

        // Use capacity hints for HashMaps
        let mut cos_mat = HashMap::with_capacity(all_q_ind.len() - modes.len());
        let mut sin_mat = HashMap::with_capacity(modes.len());
        let mut p_cos_mat = HashMap::with_capacity(modes.len());

        // Get Laguerre coefficients for radial integrals
        let (lag_coeffs, states) = &self.get_laguerre_coeffs(n_rel);
        // let dr_coeffs = &self.get_dr_radial_poly(states);

        // Use parallel map to compute matrices for each q_inds value
        let results: Vec<(Vec<i32>, Array2<c64>, Array2<c64>)> = all_q_ind.par_iter().enumerate().map(|(q_idx, q_inds)| {
            let mut matrix_1 = Array2::zeros((n_eigs, n_eigs));
            let mut matrix_2 = Array2::zeros((n_eigs, n_eigs));

            // Keep using iterators for the main computation
            for idx1 in 0..n_eigs {
                for idx2 in 0..n_eigs {
                    let vec1 = &e_vecs[idx1];
                    let vec2 = &e_vecs[idx2];

                    // Calculate the matrix elements using iterators
                    let (element_1, element_2) = (0..n_rel)
                        .flat_map(|r_ind1| {                            
                            (0..n_rel).flat_map(move |r_ind2| {
                                let (mat_element1, mat_element2): (c64,c64);

                                // Get matrix elements
                                if q_inds.len() == 1 {
                                    let mode_idx = q_inds[0] as usize;
                                    let q = &modes[mode_idx].q;
                                    let eps = &modes[mode_idx].eps;
                                    if q_idx < modes.len(){  
                                        mat_element1 = self.get_p_cos_tayl(&states[r_ind1], &states[r_ind2], q, eps,  mu, true, &lag_coeffs);
                                    }
                                    else {
                                        mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], q,  mu, true, &lag_coeffs);
                                    }
                                    mat_element2 = self.get_sin_tayl(&states[r_ind1], &states[r_ind2], q,  mu, true, &lag_coeffs);
                                }
                                else {
                                    let q_diff = &modes[q_inds[0] as usize].q - &modes[q_inds[1] as usize].q;
                                    let q_sum = &modes[q_inds[0] as usize].q + &modes[q_inds[1] as usize].q;
                                    mat_element1 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_diff, mu, true, &lag_coeffs);
                                    mat_element2 = self.get_cos_tayl(&states[r_ind1], &states[r_ind2], &q_sum, mu, true, &lag_coeffs);
                                }

                                (0..n_com).map(move |com_ind| {
                                    let i1 = n_com * r_ind1 + com_ind;
                                    let i2 = n_com * r_ind2 + com_ind;
                                    let a1 = vec1[i1];
                                    let a2 = vec2[i2];

                                    (a1.conj() * a2 * mat_element1, a1.conj() * a2 * mat_element2)
                                })
                            })
                        })
                        .fold((c64::zero(), c64::zero()), |acc, val| (acc.0 + val.0, acc.1 + val.1));

                    matrix_1[[idx1, idx2]] = element_1;
                    matrix_2[[idx1, idx2]] = element_2;
                }
            }

            // timed_println!("q_idx = {}", q_idx);
            // timed_println!("q_inds = {:?}", q_inds);
            
            // if q_idx >  modes.len() && q_inds.len() == 1{
            //     let disp_mat = PrettyArray2(&matrix_2, 4);
            //     timed_println!("Sine Matrix for phonon = ");
            //     println!("{}", disp_mat);
            // }

            // Return tuple of q_inds and computed matrices
            (q_inds.clone(), matrix_1, matrix_2)
        }).collect();

        // Process results and populate HashMaps
        results.iter().enumerate().for_each(|(idx, (q_inds, matrix_1, matrix_2))|{
            if q_inds.len() == 1 {
                if idx < modes.len(){  
                    p_cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                else {
                    cos_mat.insert(q_inds.clone(), matrix_1.clone());
                }
                sin_mat.insert(q_inds.clone(), matrix_2.clone());
            } else if q_inds.len() == 2 {
                let q_ind1 = q_inds[0].max(q_inds[1]);
                let q_ind2 = q_inds[0].min(q_inds[1]);

                cos_mat.insert(vec![q_ind1, -q_ind2], matrix_1.clone());
                cos_mat.insert(vec![q_ind1, q_ind2], matrix_2.clone());
            }
        });

        Self { cos_mat, sin_mat, p_cos_mat, cos_mat_rel: None, sin_mat_rel: None, p_cos_mat_rel: None}
    }

    pub fn to_exciton_basis(
        &self,
        rel_matrix: &Array2<c64>,
        e_vecs: &Array2<c64>,
        n_com: usize,
    ) -> Array2<c64>{
        use ndarray::linalg::kron;

        let com_identity = Array2::eye(n_com);

        let matrix = kron(&rel_matrix, &com_identity);

        let transform = e_vecs;

        let transform_dag = &transform.dag();

        // println!("Unitary test: \n{}", PrettyC64Array2(&(transform_dag.dot(&transform)), 2));

        transform_dag.dot(&matrix).dot(transform)
    }

    pub fn to_real_rel_basis(
        &self,
        rel_matrix: &Array2<c64>,
    ) -> Array2<c64>{
        let transform = Array2::from_shape_fn([rel_matrix.shape()[0], rel_matrix.shape()[1]], |(i,j)|{

            let n1 = (i as f64).sqrt().floor() as i32;
            let n2 = (j as f64).sqrt().floor() as i32;
            
            let m1 = i as i32 - n1*n1 - n1; 
            let m2 = j as i32 - n2*n2 - n2; 
            
            if i == j && m1 == 0 {
                c64::one()
            }
            else if m1.abs() == m2.abs() && n1 == n2 {
                if m1 < 0{
                    - c64::i() / 2_f64.sqrt() * m2.signum() as f64
                }
                else {
                    c64::one() / 2_f64.sqrt()
                }
            }
            else{
                c64::zero()
            }
        });

        // println!("Basis transform: \n{}", PrettyC64Array2(&transform, 2));
        // println!("Unitary test: \n{}", PrettyC64Array2(&transform.dot(&transform.dag()), 2));
            
        transform.dot(rel_matrix).dot(&transform.dag())
    }

    /// ------------------------------------------------------------------------
    /// NEW: Compute Taylor‑expanded matrix elements in the *relative* basis only.
    /// Returns three HashMaps keyed by the q‑index vector:
    ///   - cos_mat   : <n₁,m₁|cos(q·r/2)|n₂,m₂>
    ///   - sin_mat   : <n₁,m₁|sin(q·r/2)|n₂,m₂>
    ///   - p_cos_mat : i< n₁,m₁ | (ε·∇)cos(q·r/2) | n₂,m₂ >
    /// Each matrix has shape (n_rel, n_rel) where n_rel = number of relative states.
    /// ------------------------------------------------------------------------
    pub fn compute_relative_tay_mat_elements(
        &self,
        all_q_ind: &Vec<Vec<i32>>,
        modes: &Vec<BosonMode>,
        n_rel: usize,
        mu: f64,
    ) -> (
        HashMap<Vec<i32>, Array2<c64>>, // cos_mat
        HashMap<Vec<i32>, Array2<c64>>, // sin_mat
        HashMap<Vec<i32>, Array2<c64>>, // p_cos_mat
    ) {
        // Laguerre coefficients and the list of (n,m) states for the relative basis
        let (lag_coeffs, states) = self.get_laguerre_coeffs(n_rel);

        //
        let from_real = false;

        // Prepare hash‑maps with sensible capacity hints
        let mut cos_mat   = HashMap::with_capacity(all_q_ind.len());
        let mut sin_mat   = HashMap::with_capacity(modes.len());
        let mut p_cos_mat = HashMap::with_capacity(modes.len());

        // Parallel computation over all q‑index vectors
        let results: Vec<(Vec<i32>, Array2<c64>, Array2<c64>)> = all_q_ind
            .par_iter()
            .enumerate()
            .map(|(q_idx, q_inds)| {
                // Relative‑basis matrices (n_rel × n_rel)
                let mut matrix1 = Array2::zeros((n_rel, n_rel));
                let mut matrix2 = Array2::zeros((n_rel, n_rel));

                // Loop over the relative states
                for (i1, &(n1, m1)) in states.iter().enumerate() {
                    for (i2, &(n2, m2)) in states.iter().enumerate() {
                        // ----------------------------------------------------------------
                        // Choose the appropriate analytic element depending on q_inds
                        // ----------------------------------------------------------------
                        let (elem_cos, elem_sin) = if q_inds.len() == 1 {
                            // Single‑phonon (or momentum‑cosine) case
                            let mode_idx = q_inds[0] as usize;
                            let q   = &modes[mode_idx].q;   
                            let eps = &modes[mode_idx].eps; 

                            // NOTE: In the relative‑basis version we do **not** have `n_com`,
                            // so we directly use the (n,m) pair as the state index.
                            if q_idx < modes.len() {
                                // momentum‑cosine (p_cos) element
                                let pc = self.get_p_cos_tayl(&(n1, m1), &(n2, m2), q, eps, mu, from_real, &lag_coeffs);
                                (pc, self.get_sin_tayl(&(n1, m1), &(n2, m2), q, mu, from_real, &lag_coeffs))
                            } else {
                                // ordinary cosine element
                                let c = self.get_cos_tayl(&(n1, m1), &(n2, m2), q, mu, from_real, &lag_coeffs);
                                (c, self.get_sin_tayl(&(n1, m1), &(n2, m2), q, mu, from_real, &lag_coeffs))
                            }
                        } else {
                            // Two‑phonon (q₁±q₂) case – only cosine elements are needed
                            let q1 = &modes[q_inds[0] as usize].q;
                            let q2 = &modes[q_inds[1] as usize].q;
                            let q_dif = q1 - q2;
                            let q_sum  = q1 + q2;
                            let cos_dif = self.get_cos_tayl(&(n1, m1), &(n2, m2), &q_dif, mu, from_real, &lag_coeffs);
                            let cos_sum = self.get_cos_tayl(&(n1, m1), &(n2, m2), &q_sum,  mu, from_real, &lag_coeffs);
                            (cos_dif, cos_sum) // we will store c1 in `matrix_cos` and c2 in `matrix_sin`
                        };

                        // Fill the matrices
                        matrix1[[i1, i2]] = elem_cos;
                        matrix2[[i1, i2]] = elem_sin;
                    }
                }

                let real_mat1 = self.to_real_rel_basis(&matrix1);
                let real_mat2 = self.to_real_rel_basis(&matrix2);

                // Return the key and the two matrices (the caller will decide where each goes)
                (q_inds.clone(), real_mat1, real_mat2)
            })
            .collect();

        // Populate the three hash‑maps according to the length of q_inds
        results.iter().enumerate().for_each(|(idx, (q_inds, mat_1, mat_2))| {
            if q_inds.len() == 1 {
                // Single‑phonon entry – decide whether it belongs to p_cos or ordinary cos
                if idx < modes.len() {
                    p_cos_mat.insert(q_inds.clone(), mat_1.clone());
                } else {
                    cos_mat.insert(q_inds.clone(), mat_1.clone());
                }
                sin_mat.insert(q_inds.clone(), mat_2.clone());
            } else if q_inds.len() == 2 {
                // Two‑phonon entry – store both combinations in the cosine map
                let q_ind1 = q_inds[0].max(q_inds[1]);
                let q_ind2 = q_inds[0].min(q_inds[1]);

                cos_mat.insert(vec![q_ind1, -q_ind2], mat_1.clone()); // q₁−q₂
                cos_mat.insert(vec![q_ind1,  q_ind2], mat_2.clone()); // q₁+q₂
            }
        });

        (cos_mat, sin_mat, p_cos_mat)
    }

    // Only returns the sine term for phonon modes
    pub fn phonon_mat_elements(
        &self,
        e_vecs: &Array2<c64>,
        modes: &Vec<BosonMode>,
        mu: f64,
        n_com: usize,
    ) -> Vec<Array2<c64>>{

        // let n_eigs = e_vecs.len();
        let n_rel = e_vecs.nrows() / n_com;

        // Get Laguerre coefficients for radial integrals
        let (lag_coeffs, states) = &self.get_laguerre_coeffs(n_rel);

        // Use parallel map to compute matrices for each q_inds value
        modes.par_iter().map(|mode| {

            let mut rel_matrix = Array2::zeros((n_rel,n_rel));

            for (i1, &(n1, m1)) in states.iter().enumerate() {
                for (i2, &(n2, m2)) in states.iter().enumerate() {
                    rel_matrix[[i1, i2]] = self.get_sin_tayl(&(n1, m1), &(n2, m2), &mode.q, mu, false, &lag_coeffs);
                }
            }

            self.to_exciton_basis(&rel_matrix, e_vecs, n_com)

            // let mut matrix = Array2::zeros((n_eigs, n_eigs));

            // // Keep using iterators for the main computation
            // for idx1 in 0..n_eigs {
            //     for idx2 in 0..n_eigs {
            //         let vec1 = &e_vecs[idx1];
            //         let vec2 = &e_vecs[idx2];

            //         // Calculate the matrix elements using iterators
            //         let element = (0..n_rel)
            //             .flat_map(|r_ind1| {                            
            //                 (0..n_rel).flat_map(move |r_ind2| {
            //                     let q = &mode.q;
            //                     let mat_element = self.get_sin_tayl(&states[r_ind1], &states[r_ind2], q,  mu, true, &lag_coeffs);

            //                     (0..n_com).map(move |com_ind| {
            //                         let i1 = n_com * r_ind1 + com_ind;
            //                         let i2 = n_com * r_ind2 + com_ind;
            //                         let a1 = vec1[i1];
            //                         let a2 = vec2[i2];

            //                         a1.conj() * a2 * mat_element
            //                     })
            //                 })
            //             })
            //             .sum();

            //         matrix[[idx1, idx2]] = element;
            //     }
            // }
            
            // // Return tuple of q_inds and computed matrices
            // matrix
        }).collect()

    }

    /// Evaluates overlap integral of radial wavefunctions for two states (n1,m1) and (n2,m2)
    /// multiplied by r^l 
    /// O(n2,m2,n1,m1,l) = ∫ r^(l) R_{n1,m1}(r) R_{n2,m2}(r)
    pub fn get_overlap(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
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
        i1: &(i32,i32),
        i2: &(i32,i32),
        l: i32,
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {
        // Calculate radial overlap integral
        let n2 = i2.0;   
        let m2 = i2.1;
        let beta_2 = mu * 2.0 / (n2 as f64 + 0.5);

        let mut radial = (m2.abs() as f64)/beta_2 * self.get_overlap(i1, i2, l - 1, mu, lag_coeffs);
        
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

    /// Calculate the sine term using analytic integrals of leading Taylor series terms
    ///  n1,m1 | sin(q.r/2) | n2,m2 > ~   n1,m1 | (q.r/2) | n2,m2 >
    pub fn get_sin_tayl(
        &self,
        i1: &(i32,i32),
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        from_real: bool,                     // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {

        // Choose angular basis according to the flag
        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    1  => c64::from_polar(1.0,  theta_q) * weight1 * weight2,
                    -1 => c64::from_polar(1.0, -theta_q) * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.norm() < 1e-10 {
            return c64::zero();
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);
        let rad_integral = self.get_overlap(i1, i2, 2, mu, lag_coeffs);

        c64::new(norms * rad_integral * PI * q.norm() / 2.0, 0.0) * angular
    }

    /// Calculate the cosine term using analytic integrals of leading Taylor series terms
    ///  n1,m1 | cos(q.r/2) | n2,m2 > ~   n1,m1 | 1 - (q.r/2)^2/2 | n2,m2 >
    pub fn get_cos_tayl(
        &self,
        i1: &(i32,i32),
        i2: &(i32,i32),
        q: &Array1<f64>,
        mu: f64,
        from_real: bool,                     // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {

        let overlap = if i1.0 == i2.0 && i1.1 == i2.1 { 1.0 } else { 0.0 };

        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let theta_q = f64::atan2(q[1], q[0]);

        let angular = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m1 - m2;
                
                match diff {
                    0  => c64::new(1.0, 0.0) * weight1 * weight2,
                    2  => c64::from_polar(1.0, -2.0 * theta_q) * weight1 * weight2 / 2.0,
                    -2 => c64::from_polar(1.0,  2.0 * theta_q) * weight1 * weight2 / 2.0,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if angular.norm() < 1e-10 {
            return c64::new(overlap, 0.0);
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);
        let rad_integral = self.get_overlap(i1, i2, 3, mu, lag_coeffs);

        c64::new(overlap, 0.0) - angular * c64::new(norms * rad_integral, 0.0)
    }

    /// Matrix element: i  n1,m1 | (epsilon·∇) cos(q·r/2) | n2,m2 >
    /// ≈  n1,m1 | (epsilon·∇) (1 - (q·r/2)^2/2) | n2,m2 > (Second order Taylor expansion)
    pub fn get_p_cos_tayl(
        &self,
        i1: &(i32,i32),
        i2: &(i32,i32),
        q: &Array1<f64>,
        eps: &Array1<f64>,
        mu: f64,
        from_real: bool,                     // <-- new flag
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> c64 {

        let m1_vec = if from_real {
            self.real_l_to_complex_m(i1.1)
        } else {
            vec![(i1.1, c64::new(1.0, 0.0))]
        };
        let m2_vec = if from_real {
            self.real_l_to_complex_m(i2.1)
        } else {
            vec![(i2.1, c64::new(1.0, 0.0))]
        };

        let q2 = q.map(|val| val*val).sum();
        let theta_q = f64::atan2(q[1], q[0]);
        let theta_eps = f64::atan2(eps[1], eps[0]);

        let integral = m1_vec.iter().flat_map( |(m1, weight1)| {
            m2_vec.iter().map( move |(m2, weight2)| {
                let diff = m2 - m1;

                let j1 = & (i1.0, *m1);
                let j2 = & (i2.0, *m2);
                
                match diff {
                    3  => c64::from_polar(1.0,  theta_eps + 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * (-(*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    -3 => c64::from_polar(1.0, -theta_eps - 2.0 * theta_q) 
                            * q2 * PI / 32.0
                            * ( (*m2 as f64) * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                - self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs))
                            * weight1 * weight2,
                    1  => c64::from_polar(PI,  theta_eps)
                        * (
                            (*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                            + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                            - q2 / 32.0 * (
                                 (*m2 as f64) * (2.0 - c64::from_polar(1.0,  2.0 * (theta_q - theta_eps))) 
                                * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                + (2.0 + c64::from_polar(1.0,  2.0 * (theta_q - theta_eps)))
                                * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                            )
                        )
                        * weight1 * weight2,
                    -1 => c64::from_polar(PI, -theta_eps)
                        * (
                            -(*m2 as f64) * self.get_overlap(j1, j2, 0, mu, lag_coeffs)
                            + self.get_dr_overlap(j1, j2, 1, mu, lag_coeffs)
                            - q2 / 32.0 * (
                                -(*m2 as f64) * (2.0 - c64::from_polar(1.0, -2.0 * (theta_q - theta_eps))) 
                                * self.get_overlap(j1, j2, 2, mu, lag_coeffs)
                                + (2.0 + c64::from_polar(1.0, -2.0 * (theta_q - theta_eps)))
                                * self.get_dr_overlap(j1, j2, 3, mu, lag_coeffs)
                            )
                        )
                        * weight1 * weight2,
                    _  => c64::new(0.0, 0.0),
                }
            })
        }).sum::<c64>();

        if integral.norm() < 1e-10 {
            return c64::zero();
        }

        let norms = self.norm_constant(i1,mu) * self.norm_constant(i2, mu);
        c64::new(norms, 0.0) * integral
    }

    fn real_l_to_complex_m(&self, l: i32) -> Vec<(i32, c64)>{
        match l {
            0 => vec![(0, c64::new(1.0, 0.0))],
            l if l > 0 => vec![
                ( l, c64::new( 1.0 / 2_f64.sqrt(), 0.0)), 
                (-l, c64::new( 1.0 / 2_f64.sqrt(), 0.0))],
            l if l < 0 => vec![
                ( l, c64::new( 1.0 / 2_f64.sqrt(), 0.0)), 
                (-l, c64::new(-1.0 / 2_f64.sqrt(), 0.0))],
            _ => panic!("Invalid l value in real_l_to_complex_m")
        }
    }

    fn norm_constant(&self, (n,m): &(i32,i32), mu: f64) -> f64 {

        let beta = mu * 2.0 / (*n as f64 + 0.5);

        beta
        * (factorial(*n as usize - m.abs() as usize) as f64
            / (2. * PI
                * factorial(*n as usize + m.abs() as usize) as f64
                * (2. * *n as f64 + 1.)))
            .sqrt()
    }

    fn _test_psi_norm(&self, states: &Vec<(i32, i32)>, mu: f64) {
        states.iter().for_each(|&(n,m)| {
            let n = n;
            let m = m.abs();

            let beta = mu / 8. / PI / PI / (n as f64 + 0.5);
            let norm = beta
                * (factorial(n as usize - m as usize) as f64
                    / (2.
                        * PI
                        * factorial((n + m) as usize) as f64
                        * (2. * n as f64 + 1.)))
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

                    acc2 + c1 * c2 * factorial_term / denom_term
                        * beta.powi(2 * m + i + j)
                })
            });

            let psi_norm = 2. * PI * norm.powi(2) * integral;

            timed_println!("State (n={},m={}): Norm = {}", n, m, psi_norm);
        });
    }

    pub fn get_x(
        &self,
        (n1,m1): &(i32,i32),
        (n2,m2): &(i32,i32),
        mu: f64,
        lag_coeffs: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    ) -> f64 {

        let n1 = *n1;
        let m1 = *m1;
        let n2 = *n2;   
        let m2 = *m2;

        // Calculate angular component
        let diff = (m1 -m2).abs();
        let sum = (m1 + m2).abs();

        // Only jumps in m by ±1 allowed
        if diff & sum != 1 {
            return 0.0
        }
        
        let m_sign = (m1 * m2).signum();

        if m_sign == 0 && (m1 + m2) < 0 {
            return 0.0;
        }

        let angular = match m_sign {
            0  => PI * 2.0_f64.sqrt(),
            1  => {
                let mut result = 0.0;
                if diff == 1 { result += PI};
                if sum  == 1 { result += PI * m1.signum() as f64};
                result
            },
            _ => 0.0
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

    pub fn get_laguerre_coeffs(&self, n_rel: usize) -> (HashMap<(i32, i32), Vec<(i32, f64)>>, Vec<(i32, i32)>) {
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
                        warn!("Skipping state (n={},m={}) because Laguerre polynomials of order >7 are not implemented", n, m);
                    }
                }
            }
            n += 1;
        }

        // Keep only the first n_rel states
        states.truncate(n_rel);

        let mut lag_coeffs: HashMap<(i32, i32), Vec<(i32, f64)>> = HashMap::new();

        for &(n, m) in &states {
            lag_coeffs.insert((n, m), self.laguerre((n - m.abs()) as usize, 2. * m.abs() as f64));
        }

        // Ensure all necessary keys are present
        for &(n1, m1) in &states {
            if !lag_coeffs.contains_key(&(n1, m1.abs())) {
                panic!("Missing key ({},{})", n1, m1.abs());
            }
        }

        (lag_coeffs, states)
    }

    fn laguerre(&self, order: usize, alpha: f64) -> Vec<(i32, f64)> {
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
                    (1,      -(alpha + 2.0)),
                    (2, 0.5),
                ];
            }
            3 => {
                coeffs = vec![
                    (0,(1.0 / 6.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -0.5       * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  0.5       * (alpha + 3.0)),
                    (3,-(1.0 / 6.0)),
                ];
            }
            4 => {
                coeffs = vec![
                    (0, (1.0 / 24.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 6.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,         0.25 * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 6.0) * (alpha + 4.0)),
                    (4, 1.0 / 24.0),
                ];
            }
            5 => {
                coeffs = vec![
                    (0, (1.0 / 120.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 24.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 12.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 24.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0)),
                ];
            }
            6 => {
                coeffs = vec![
                    (0,  (1.0 / 720.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 120.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3,  -(1.0 / 36.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,   (1.0 / 48.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 120.0) * (alpha + 6.0)),
                    (6, 1.0 / 720.0),
                ];
            }
            7 => {
                coeffs = vec![
                    (0, (1.0 / 5040.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0) * (alpha + 1.0)),
                    (1, -(1.0 / 720.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0) * (alpha + 2.0)),
                    (2,  (1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0) * (alpha + 3.0)),
                    (3, -(1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0) * (alpha + 4.0)),
                    (4,  (1.0 / 144.0) * (alpha + 7.0) * (alpha + 6.0) * (alpha + 5.0)),
                    (5, -(1.0 / 240.0) * (alpha + 7.0) * (alpha + 6.0)),
                    (6,  (1.0 / 720.0) * (alpha + 7.0)),
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