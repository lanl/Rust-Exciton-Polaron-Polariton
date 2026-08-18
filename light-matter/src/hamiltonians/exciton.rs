use exciton_2d::FullHamiltonian;
use log::warn;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2, arr1, s};
use ndarray_linalg::c64;
use std::iter::Sum;

pub struct ExcitonicHamiltonian<M: MatrixOps> {
    pub exciton_ham: FullHamiltonian<M>,
    pub n_ex_eig: usize,
    pub n_rel: usize,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Array2<num_complex::Complex64>>,
    pub r_vecs: Option<Array2<f64>>,
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
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.e_vals.is_none() || self.e_vecs.is_none() || self.r_vecs.is_none()
    }

    pub fn compute_eigenstates(&mut self) -> Result<(), MatrixOpsError> {
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
