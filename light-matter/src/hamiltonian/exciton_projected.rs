use exciton_2d::FullHamiltonian;
use log::warn;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2, arr1, s};
use ndarray_linalg::c64;
use serde::{Deserialize, Serialize};
use std::iter::Sum;
use std::path::{Path, PathBuf};

const EXCITON_BASIS_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct ExcitonBasisCache {
    version: u32,
    n_rel: usize,
    n_ex_eig_requested: usize,
    n_com: usize,
    k_quantized: Vec<i64>,
    nrows: usize,
    ncols: usize,
    e_vals: Vec<f64>,
    e_vecs_re: Vec<f64>, // logical row-major
    e_vecs_im: Vec<f64>,
}

pub struct ExcitonicHamiltonian<M: MatrixOps> {
    pub exciton_ham: FullHamiltonian<M>,
    pub n_ex_eig: usize,
    pub n_rel: usize,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Array2<num_complex::Complex64>>,
    pub r_vecs: Option<Array2<f64>>,
    /// If set, the exciton eigenbasis is loaded from / saved to this file so that
    /// resumed checkpointed solves use an identical basis. Collective when set.
    pub basis_cache: Option<PathBuf>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> ExcitonicHamiltonian<M>
where
    M: Diagonalize,
    M::Scalar: ConvertFrom<c64>,
{
    pub fn new(exciton_ham: FullHamiltonian<M>, n_ex_eig: usize, n_rel: usize) -> Self {
        Self {
            exciton_ham,
            n_ex_eig,
            n_rel,
            e_vals: None,
            e_vecs: None,
            r_vecs: None,
            basis_cache: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn compute_eigenstates(&mut self) -> Result<(), MatrixOpsError> {
        let Some(path) = self.basis_cache.clone() else {
            return self.compute_eigenstates_fresh();
        };

        // Collective decision: either ALL ranks load the file, or ALL recompute.
        let local = self.read_basis_cache(&path);
        if matrix_ops::mpi_all_ranks_true(local.is_some())? {
            let (vals, vecs) = local.expect("checked on this rank");
            let (_, r_vecs, _) = self.exciton_ham.rel_ham.solve(self.n_rel);
            self.exciton_ham.matrix_elem_calc.set_cached_eigvecs(r_vecs.clone());
            self.n_ex_eig = vals.len();
            self.e_vals = Some(vals);
            self.e_vecs = Some(vecs);
            self.r_vecs = Some(r_vecs);
            crate::timed_println_rank0!("Loaded exciton eigenbasis from {}", path.display());
            return Ok(());
        }

        let requested = self.n_ex_eig;
        self.compute_eigenstates_fresh()?;

        if sim_common::mpi::is_mpi_rank0_env() {
            let e_vecs = self.e_vecs.as_ref().unwrap();
            let cache = ExcitonBasisCache {
                version: EXCITON_BASIS_VERSION,
                n_rel: self.n_rel,
                n_ex_eig_requested: requested,
                n_com: self.exciton_ham.kappa_vectors.len(),
                k_quantized: self.k_quantized(),
                nrows: e_vecs.nrows(),
                ncols: e_vecs.ncols(),
                e_vals: self.e_vals.as_ref().unwrap().to_vec(),
                e_vecs_re: e_vecs.iter().map(|z| z.re).collect(),
                e_vecs_im: e_vecs.iter().map(|z| z.im).collect(),
            };
            if let Err(e) = sim_common::cache::save_bincode_atomic(&path, &cache) {
                warn!("Failed to save exciton basis {}: {}", path.display(), e);
            }
        }
        matrix_ops::mpi_barrier()?;
        Ok(())
    }

    fn compute_eigenstates_fresh(&mut self) -> Result<(), MatrixOpsError> {
        let (exciton_energies, e_vecs, _, r_vecs) = self
            .exciton_ham
            .diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig)?;

        let n_returned = exciton_energies.len();
        let n_keep = n_returned.min(self.n_ex_eig);

        if n_returned != self.n_ex_eig {
            warn!(
                "Diagonalization returned {} eigenvalues, expected {}; keeping {}",
                n_returned, self.n_ex_eig, n_keep
            );
        }

        self.e_vals = Some(exciton_energies.slice_move(s![0..n_keep]));
        self.e_vecs = Some(e_vecs.slice_move(s![.., 0..n_keep]));
        self.r_vecs = Some(r_vecs);
        self.n_ex_eig = n_keep;

        Ok(())
    }

    fn k_quantized(&self) -> Vec<i64> {
        self.exciton_ham
            .k
            .iter()
            .map(|&k| crate::utils::quantize_f64_for_hash(k))
            .collect()
    }

    fn read_basis_cache(&self, path: &Path) -> Option<(Array1<f64>, Array2<c64>)> {
        if !path.exists() {
            return None;
        }
        let c: ExcitonBasisCache = sim_common::cache::load_bincode(path).ok()?;
        let n_com = self.exciton_ham.kappa_vectors.len();
        if c.version != EXCITON_BASIS_VERSION
            || c.n_rel != self.n_rel
            || c.n_ex_eig_requested != self.n_ex_eig
            || c.n_com != n_com
            || c.k_quantized != self.k_quantized()
            || c.nrows != n_com * self.n_rel
            || c.ncols != c.e_vals.len()
            || c.e_vecs_re.len() != c.nrows * c.ncols
            || c.e_vecs_im.len() != c.nrows * c.ncols
        {
            return None;
        }
        let data: Vec<c64> = c
            .e_vecs_re
            .iter()
            .zip(&c.e_vecs_im)
            .map(|(&re, &im)| c64::new(re, im))
            .collect();
        let vecs = Array2::from_shape_vec((c.nrows, c.ncols), data).ok()?;
        Some((Array1::from_vec(c.e_vals), vecs))
    }

    pub fn is_empty(&self) -> bool {
        self.e_vals.is_none() || self.e_vecs.is_none() || self.r_vecs.is_none()
    }

    pub fn get_p_com(&self, field_vec: &Array1<f64>) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let e_vecs = self
            .e_vecs
            .as_ref()
            .expect("Eigenvectors not computed. Call compute_eigenstates() first.");

        let kappas = &self.exciton_ham.kappa_vectors;
        let k = &self.exciton_ham.k;

        let n_com = kappas.len();
        let n_eig = e_vecs.ncols();

        let k_arr = arr1(k);

        let p_dot_unit_cache: Vec<f64> = kappas
            .iter()
            .map(|kappa| {
                (k_arr[0] + kappa[0]) * field_vec[0] + (k_arr[1] + kappa[1]) * field_vec[1]
            })
            .collect();

        M::from_dense(&Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            let vec1 = e_vecs.column(idx1);
            let vec2 = e_vecs.column(idx2);

            let n_rel = vec1.len() / n_com;

            (0..n_com)
                .flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];

                    (0..n_rel).map(move |rel_ind| {
                        let i = com_ind * n_rel + rel_ind;
                        vec1[i].conj() * vec2[i] * p
                    })
                })
                .sum::<c64>()
        }))
    }

    pub fn get_h(&mut self) -> Result<M, MatrixOpsError>
    where
        M::Scalar: ConvertFrom<f64>,
    {
        if self.is_empty() {
            self.compute_eigenstates()?;
        }

        let diags = self
            .e_vals
            .as_ref()
            .ok_or(MatrixOpsError::Other {
                message: "Eigenvalues not computed. Call compute_eigenstates() first.".to_string(),
            })?
            .as_slice()
            .ok_or(MatrixOpsError::Other {
                message: "Failed to convert eigenvalues to slice.".to_string(),
            })?;

        M::from_diag(diags)
    }

    pub fn dimension(&self) -> usize {
        self.e_vals.as_ref().map_or(0, |vals| vals.len())
    }
}
