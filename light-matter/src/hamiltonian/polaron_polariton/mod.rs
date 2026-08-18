mod fock;

use super::{Boson, BosonicHamiltonian, ExcitonicHamiltonian, FullBosons, LMHamiltonian};
use crate::config::PPConfig;
use crate::matrix_elements::{LMMatrixElementArray, ModeKey};
use crate::modes::types::{BosonMode, Polarization};
use crate::timed_println;
use crate::timed_println_rank0;
use crate::types::Vector3D;
use exciton_2d::Exciton2DGridSystem;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError, Sqrt};
use ndarray::arr1;
use ndarray_linalg::c64;
use sprs::TriMat;
use std::iter::Sum;

pub struct PPHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub polariton: LMHamiltonian<M, B>,
    pub phonon: BosonicHamiltonian<M, FullBosons<M>>,
    pub pn_coupling: f64,
    sin_pn: Option<Vec<M>>,
    is_initialized: bool,
    cached_lm_ham: Option<M>,
    cached_k: Option<Vec<f64>>,
}

impl<M: MatrixOps, B: Boson<M>> PPHamiltonian<M, B>
where
    M: Diagonalize,
    M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
{
    pub fn new(
        polariton: LMHamiltonian<M, B>,
        phonon: BosonicHamiltonian<M, FullBosons<M>>,
        pn_coupling: f64,
    ) -> Self {
        Self {
            polariton,
            phonon,
            pn_coupling,
            sin_pn: None,
            is_initialized: false,
            cached_lm_ham: None,
            cached_k: None,
        }
    }

    pub fn from_config(pp_config: &PPConfig) -> Self {
        // Create exciton
        let exciton_grid = Exciton2DGridSystem::from_config(pp_config.e_config.clone());
        let exciton: ExcitonicHamiltonian<M> =
            ExcitonicHamiltonian::new(exciton_grid.full_ham, pp_config.n_ex_eig, pp_config.n_rel);

        // Create photon
        let photon_modes: Vec<BosonMode> = pp_config
            .q_values
            .iter()
            .map(|q| {
                BosonMode::new(
                    Vector3D {
                        x: q.x,
                        y: q.y,
                        z: q.z,
                    },
                    Polarization::TE,
                    2,
                )
                .clone()
            })
            .collect();
        let photon: BosonicHamiltonian<M, B> = BosonicHamiltonian::new(photon_modes);

        // Create Polariton with cache support
        let config_hash = pp_config.get_config_hash();
        let polariton = LMHamiltonian::new_with_cache(
            exciton,
            photon,
            pp_config.lm_couplings.clone(),
            pp_config.cache_dir.clone(),
            config_hash,
        );

        // Create Phonon
        let phonon_mode = BosonMode {
            eps: arr1(&[0.0, 0.0]),
            q: pp_config.pn_k.clone(),
            omega: pp_config.pn_omega,
            n_fock: pp_config.pn_n_fock,
        };
        let phonon: BosonicHamiltonian<M, FullBosons<M>> =
            BosonicHamiltonian::new(vec![phonon_mode]);

        // Create Polaron Polariton
        let pn_coupling = pp_config.pn_coupling;
        PPHamiltonian::new(polariton, phonon, pn_coupling)
    }

    // pub fn update_k(&mut self, k: Vec<f64>)
    // where
    //     M::Scalar:  Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    // {
    //     self.polariton.exciton.exciton_ham.k = k.clone();
    //     self.polariton.initialize(Some(k));
    // }

    pub fn initialize(&mut self, k: Option<Vec<f64>>) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        use std::time::Instant;

        if self.is_initialized {
            return Ok(());
        }

        let time_pol = Instant::now();
        self.polariton.initialize(k)?;
        timed_println_rank0!("  Polariton initialization: {:?}", time_pol.elapsed());

        let time_pn = Instant::now();
        let pn_mat_calc = LMMatrixElementArray::new();
        let sin_pn = pn_mat_calc.phonon_mat_elements(
            self.polariton.exciton.e_vecs.as_ref().unwrap(),
            self.phonon.boson.get_modes(),
            self.polariton.exciton.exciton_ham.rel_ham.mu,
            self.polariton.exciton.exciton_ham.kappa_vectors.len(),
        );
        timed_println_rank0!("  Phonon matrix elements: {:?}", time_pn.elapsed());

        let time_convert = Instant::now();

        let sin_pn: Result<Vec<M>, MatrixOpsError> = sin_pn
            .iter()
            .map(|mat| {
                let converted = M::from_dense(&mat)?;
                Ok(converted)
            })
            .collect();

        self.sin_pn = Some(sin_pn?);
        timed_println_rank0!("  Convert to sparse: {:?}", time_convert.elapsed());

        self.is_initialized = true;

        Ok(())
    }

    #[inline]
    fn add_term(acc: &mut Option<M>, term: M) -> Result<(), MatrixOpsError> {
        *acc = Some(match acc.take() {
            Some(existing) => existing.add(&term)?,
            None => term,
        });

        Ok(())
    }

    #[inline]
    fn single_transition(
        dim: usize,
        from: usize,
        to: usize,
        value: f64,
    ) -> Result<M, MatrixOpsError>
    where
        M::Scalar: ConvertFrom<f64>,
    {
        let mut tri = TriMat::new((dim, dim));

        if value.abs() > 1e-14 {
            tri.add_triplet(to, from, M::Scalar::convert_from(value));
        }

        M::from_trimat(&tri)
    }

    #[inline]
    fn push_transition_pair(
        pairs: &mut Vec<(M, M)>,
        left: M,
        right_dim: usize,
        from: usize,
        to: usize,
        value: f64,
    ) -> Result<(), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<f64>,
    {
        if value.abs() > 1e-14 {
            pairs.push((left, Self::single_transition(right_dim, from, to, value)?));
        }

        Ok(())
    }

    #[inline]
    fn use_sparse_pp_kronsum_backend() -> bool {
        let name = std::any::type_name::<M>();
        name.contains("CsMat") || name.contains("sprs")
    }

    #[inline]
    fn phonon_mode_dims(&self) -> Vec<usize> {
        self.phonon
            .boson
            .get_modes()
            .iter()
            .map(|mode| mode.n_fock)
            .collect()
    }

    /// Converts a flat product-Fock index into occupations.
    ///
    /// This matches the tensor-product order used by `FullBosons::a`, where
    /// mode 0 is the slowest-varying index and the last mode is fastest-varying:
    ///

    /// Converts product-Fock occupations into a flat index.
    ///

    /// Lazy-safe PP interaction construction.
    ///
    /// This returns only the terms that are not already contained in
    ///
    ///     H_LM ⊗ I_pn + I_LM ⊗ H_pn.
    ///
    /// It includes:
    ///
    /// 1. phonon-exciton Fröhlich coupling,
    /// 2. exciton-phonon momentum recoil,
    /// 3. full phonon-phonon recoil `sum_{k,k'} k·k' n_k n_k' / 2M`,
    /// 4. photon-phonon number recoil `sum_{q,k} q·k n_q n_k / M`,
    /// 5. photon-phonon correction to the CoM light-matter coupling.
    fn get_interaction_hamiltonian_as_operator(&self, ph_dim: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let sin_pn = self.sin_pn.as_ref().unwrap();
        let ex_dim = sin_pn[0].dimension();
        let lm_dim = ph_dim * ex_dim;
        let pn_dim = self.phonon.dimension();

        let m_com = self.polariton.exciton.exciton_ham.m;

        let ph_eye = M::eye(ph_dim);
        let lm_eye = M::eye(lm_dim);
        let ex_eye = M::eye(ex_dim);

        let mut interaction: Option<M> = None;

        self.phonon
            .boson
            .get_modes()
            .iter()
            .enumerate()
            .try_for_each(|(idx1, mode_1)| -> Result<(), MatrixOpsError> {
                let b = self.phonon.get_a(idx1)?;
                let b_dag = b.transpose();
                let k1 = &mode_1.q;

                // ------------------------------------------------------------
                // Phonon-exciton coupling:
                //
                //     -2 gamma_k sin(k·x/2) (b_k† + b_k)
                //
                // after the phase rotation i(b† - b) -> -(b† + b).
                // ------------------------------------------------------------
                let lm_op = sin_pn[idx1].kron(&ph_eye);
                let pn_op = (b.add(&b_dag))?.scale(-2.0 * self.pn_coupling)?;

                Self::add_term(&mut interaction, lm_op.kron(&pn_op))?;

                // ------------------------------------------------------------
                // Exciton-phonon momentum recoil:
                //
                //     -(P·k / M) n_k
                // ------------------------------------------------------------
                let p_k = self.polariton.exciton.get_p_com(k1)?;
                let lm_op = p_k.kron(&ph_eye);
                let n_k1 = self.phonon.get_adag_a(idx1)?;

                Self::add_term(&mut interaction, lm_op.kron(&n_k1.scale(-1.0 / m_com)?))?;

                // ------------------------------------------------------------
                // Full phonon-phonon recoil:
                //
                //     sum_{k,k'} (k·k') n_k n_k' / (2M)
                //
                // This includes k != k' terms. These terms are allowed because
                // phonons are represented in the full product Fock space.
                // ------------------------------------------------------------
                self.phonon
                    .boson
                    .get_modes()
                    .iter()
                    .enumerate()
                    .try_for_each(|(idx2, mode_2)| -> Result<(), MatrixOpsError> {
                        let k2 = &mode_2.q;
                        let n_k2 = self.phonon.get_adag_a(idx2)?;

                        let pn_op = n_k1.dot(&n_k2)?.scale(k1.dot(k2) / (2.0 * m_com))?;

                        Self::add_term(&mut interaction, lm_eye.clone().kron(&pn_op))?;

                        Ok(())
                    })?;

                // ------------------------------------------------------------
                // Photon-phonon number recoil from
                //
                //   (P - sum_q q n_q - sum_k k n_k)^2 / 2M.
                //
                // Since H_LM already contains (P - sum_q q n_q)^2 / 2M,
                // adding phonons contributes:
                //
                //   +(1/M) sum_{q,k} (q·k) n_q n_k.
                //
                // If this term is intentionally omitted in your model, remove
                // this block from both direct and sparse-KronSum paths.
                // ------------------------------------------------------------
                let cached_photon_numbers = self.polariton.cached_adag_a_ops.as_ref().unwrap();

                self.polariton
                    .photon
                    .boson
                    .get_modes()
                    .iter()
                    .enumerate()
                    .try_for_each(|(ph_idx, ph_mode)| -> Result<(), MatrixOpsError> {
                        let q_dot_k = ph_mode.q.dot(k1);

                        if q_dot_k.abs() <= 1e-14 {
                            return Ok(());
                        }

                        let lm_op = ex_eye
                            .kron(&cached_photon_numbers[ph_idx])
                            .scale(q_dot_k / m_com)?;

                        Self::add_term(&mut interaction, lm_op.kron(&n_k1))?;

                        Ok(())
                    })?;

                // ------------------------------------------------------------
                // A·k n_k sin(q·x/2) (a + a†) recoil correction:
                //
                //     -(2/M) (A_q·k) sin(q·x/2) (a_q + a_q†) n_k.
                // ------------------------------------------------------------
                self.polariton
                    .photon
                    .boson
                    .get_modes()
                    .iter()
                    .enumerate()
                    .try_for_each(|(ph_idx, ph_mode)| -> Result<(), MatrixOpsError> {
                        let eps = &ph_mode.eps;
                        let coupling = eps.dot(k1) * self.polariton.couplings[ph_idx];

                        if coupling.abs() <= 1e-14 {
                            return Ok(());
                        }

                        let key = ModeKey::single(ph_idx);

                        let ph_op = self.polariton.cached_a_ops.as_ref().unwrap()[ph_idx].add(
                            &self.polariton.cached_a_ops.as_ref().unwrap()[ph_idx].transpose(),
                        )?;

                        let ex_op = self
                            .polariton
                            .matrix_elements
                            .sin_mat
                            .get(&key)
                            .unwrap_or_else(|| panic!("Missing photon sine matrix for {:?}", key));

                        let lm_op = M::from_dense(ex_op)?.kron(&ph_op);
                        let pn_op = n_k1.scale(-2.0 * coupling / m_com)?;

                        Self::add_term(&mut interaction, lm_op.kron(&pn_op))?;

                        Ok(())
                    })?;

                Ok(())
            })?;

        let total_dim = lm_dim * pn_dim;

        Ok(interaction.unwrap_or_else(|| M::zeros((total_dim, total_dim))))
    }

    /// Sparse-backend optimized PP construction as a KronSum over phonon
    /// product-Fock transitions.
    ///
    /// This supports an arbitrary number of phonon modes in `FullBosons`.
    ///
    /// It builds
    ///
    ///     H_PP = sum_i L_i ⊗ T_i,
    ///
    /// where `L_i` acts on the LM space and `T_i` is a one-entry phonon-space
    /// transition matrix.
    ///
    /// The phonon basis is the product-Fock basis used by `FullBosons`.
    fn build_pp_hamiltonian_sparse_kronsum(
        &self,
        lm_ham: &M,
        ph_dim: usize,
        ex_dim: usize,
        pn_dim: usize,
    ) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let sin_pn = self.sin_pn.as_ref().unwrap();
        let m_com = self.polariton.exciton.exciton_ham.m;

        let lm_dim = ph_dim * ex_dim;

        let ph_eye = M::eye(ph_dim);
        let lm_eye = M::eye(lm_dim);
        let ex_eye = M::eye(ex_dim);

        let dims = self.phonon_mode_dims();

        if dims.iter().product::<usize>() != pn_dim {
            return Err(MatrixOpsError::Other {
                message: format!(
                    "Internal phonon dimension mismatch: product dims = {}, pn_dim = {}",
                    dims.iter().product::<usize>(),
                    pn_dim
                ),
            });
        }

        let occupations: Vec<Vec<usize>> = (0..pn_dim)
            .map(|idx| fock::occupations_from_index(idx, &dims))
            .collect();

        let mut pairs: Vec<(M, M)> = Vec::new();

        // ------------------------------------------------------------
        // H_LM ⊗ I_pn
        // ------------------------------------------------------------
        for pn_idx in 0..pn_dim {
            Self::push_transition_pair(&mut pairs, lm_ham.clone(), pn_dim, pn_idx, pn_idx, 1.0)?;
        }

        // ------------------------------------------------------------
        // I_LM ⊗ H_pn
        //
        // Current BosonicHamiltonian::get_h omits zero-point energy:
        //
        //     H_pn = sum_k omega_k n_k.
        // ------------------------------------------------------------
        for (pn_idx, occ) in occupations.iter().enumerate() {
            let energy = self
                .phonon
                .boson
                .get_modes()
                .iter()
                .zip(occ.iter())
                .map(|(mode, &n)| mode.omega * n as f64)
                .sum::<f64>();

            Self::push_transition_pair(&mut pairs, lm_eye.clone(), pn_dim, pn_idx, pn_idx, energy)?;
        }

        // ------------------------------------------------------------
        // Phonon-exciton coupling:
        //
        //     -2 gamma_k sin(k·x/2) (b_k† + b_k)
        // ------------------------------------------------------------
        for (mode_idx, _mode) in self.phonon.boson.get_modes().iter().enumerate() {
            let lm_sin = sin_pn[mode_idx].kron(&ph_eye);

            for (pn_from, occ) in occupations.iter().enumerate() {
                let n = occ[mode_idx];

                // b_k: |..., n_k, ...> -> sqrt(n_k) |..., n_k-1, ...>
                if n > 0 {
                    let mut occ_to = occ.clone();
                    occ_to[mode_idx] -= 1;
                    let pn_to = fock::index_from_occupations(&occ_to, &dims);

                    let amp = -2.0 * self.pn_coupling * (n as f64).sqrt();

                    Self::push_transition_pair(
                        &mut pairs,
                        lm_sin.clone(),
                        pn_dim,
                        pn_from,
                        pn_to,
                        amp,
                    )?;
                }

                // b_k†: |..., n_k, ...> -> sqrt(n_k+1) |..., n_k+1, ...>
                if n + 1 < dims[mode_idx] {
                    let mut occ_to = occ.clone();
                    occ_to[mode_idx] += 1;
                    let pn_to = fock::index_from_occupations(&occ_to, &dims);

                    let amp = -2.0 * self.pn_coupling * ((n + 1) as f64).sqrt();

                    Self::push_transition_pair(
                        &mut pairs,
                        lm_sin.clone(),
                        pn_dim,
                        pn_from,
                        pn_to,
                        amp,
                    )?;
                }
            }
        }

        // ------------------------------------------------------------
        // Exciton-phonon momentum recoil:
        //
        //     -(P·k / M) n_k.
        // ------------------------------------------------------------
        for (mode_idx, mode) in self.phonon.boson.get_modes().iter().enumerate() {
            let p_k_lm = self
                .polariton
                .exciton
                .get_p_com(&mode.q)?
                .kron(&ph_eye)
                .scale(-1.0 / m_com)?;

            for (pn_idx, occ) in occupations.iter().enumerate() {
                let n = occ[mode_idx] as f64;

                Self::push_transition_pair(&mut pairs, p_k_lm.clone(), pn_dim, pn_idx, pn_idx, n)?;
            }
        }

        // ------------------------------------------------------------
        // Full phonon-phonon recoil:
        //
        //     sum_{k,k'} (k·k') n_k n_k' / (2M).
        //
        // This includes all k != k' terms.
        // ------------------------------------------------------------
        for (pn_idx, occ) in occupations.iter().enumerate() {
            let mut recoil = 0.0;

            for (idx1, mode_1) in self.phonon.boson.get_modes().iter().enumerate() {
                let n1 = occ[idx1] as f64;

                if n1 == 0.0 {
                    continue;
                }

                for (idx2, mode_2) in self.phonon.boson.get_modes().iter().enumerate() {
                    let n2 = occ[idx2] as f64;

                    if n2 == 0.0 {
                        continue;
                    }

                    recoil += mode_1.q.dot(&mode_2.q) * n1 * n2 / (2.0 * m_com);
                }
            }

            Self::push_transition_pair(&mut pairs, lm_eye.clone(), pn_dim, pn_idx, pn_idx, recoil)?;
        }

        // ------------------------------------------------------------
        // Photon-phonon number recoil:
        //
        //     +(1/M) sum_{q,k} q·k n_q n_k.
        //
        // This is implied by
        //
        //     (P - sum_q q n_q - sum_k k n_k)^2 / 2M
        //
        // after H_LM has already included the photon-only part.
        // ------------------------------------------------------------
        let cached_photon_numbers = self.polariton.cached_adag_a_ops.as_ref().unwrap();

        for (ph_idx, ph_mode) in self.polariton.photon.boson.get_modes().iter().enumerate() {
            let photon_number_lm = ex_eye.kron(&cached_photon_numbers[ph_idx]);

            for (pn_mode_idx, pn_mode) in self.phonon.boson.get_modes().iter().enumerate() {
                let coeff = ph_mode.q.dot(&pn_mode.q) / m_com;

                if coeff.abs() <= 1e-14 {
                    continue;
                }

                let lm_op = photon_number_lm.scale(coeff)?;

                for (pn_idx, occ) in occupations.iter().enumerate() {
                    let n_k = occ[pn_mode_idx] as f64;

                    Self::push_transition_pair(
                        &mut pairs,
                        lm_op.clone(),
                        pn_dim,
                        pn_idx,
                        pn_idx,
                        n_k,
                    )?;
                }
            }
        }

        // ------------------------------------------------------------
        // A·k n_k sin(q·x/2) (a + a†) recoil correction:
        //
        //     -(2/M) (A_q·k) sin(q·x/2) (a_q + a_q†) n_k.
        // ------------------------------------------------------------
        for (pn_mode_idx, pn_mode) in self.phonon.boson.get_modes().iter().enumerate() {
            let k = &pn_mode.q;

            for (ph_idx, ph_mode) in self.polariton.photon.boson.get_modes().iter().enumerate() {
                let coupling = ph_mode.eps.dot(k) * self.polariton.couplings[ph_idx];

                if coupling.abs() <= 1e-14 {
                    continue;
                }

                let key = ModeKey::single(ph_idx);

                let ph_op = self.polariton.cached_a_ops.as_ref().unwrap()[ph_idx]
                    .add(&self.polariton.cached_a_ops.as_ref().unwrap()[ph_idx].transpose())?;

                let ex_op = self
                    .polariton
                    .matrix_elements
                    .sin_mat
                    .get(&key)
                    .unwrap_or_else(|| panic!("Missing photon sine matrix for {:?}", key));

                let lm_op = M::from_dense(ex_op)?.kron(&ph_op);

                for (pn_idx, occ) in occupations.iter().enumerate() {
                    let n_k = occ[pn_mode_idx] as f64;

                    Self::push_transition_pair(
                        &mut pairs,
                        lm_op.clone(),
                        pn_dim,
                        pn_idx,
                        pn_idx,
                        -2.0 * coupling * n_k / m_com,
                    )?;
                }
            }
        }

        timed_println_rank0!(
            "  Building sparse PP KronSum from {} LM⊗phonon transition pairs",
            pairs.len()
        );

        M::build_kron_sum(pairs, lm_dim, pn_dim)
    }

    pub fn build_pp_hamiltonian(&mut self, k: Option<Vec<f64>>) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        use std::time::Instant;

        let k_changed = match &k {
            Some(k_val) => Some(k_val) != self.cached_k.as_ref(),
            None => false,
        };

        if k_changed {
            self.is_initialized = false;
            self.cached_lm_ham = None;
            self.cached_k = k.clone();
        }

        if !self.is_initialized {
            self.initialize(k.clone())?;
        }

        let time_lm = Instant::now();
        let lm_ham = if let Some(cached) = &self.cached_lm_ham {
            timed_println_rank0!("  Using cached LM Hamiltonian");
            cached.clone()
        } else {
            let ham = self.polariton.build_lm_hamiltonian(k.clone())?;
            self.cached_lm_ham = Some(ham.clone());
            ham
        };
        timed_println_rank0!("  Build/retrieve LM Hamiltonian: {:?}", time_lm.elapsed());

        let ph_dim = self.polariton.photon.dimension();
        let pn_dim = self.phonon.dimension();
        let ex_dim = self.polariton.exciton.dimension();
        let lm_dim = ph_dim * ex_dim;

        if pn_dim == 0 || pn_dim == 1 || self.pn_coupling == 0.0 {
            timed_println_rank0!(
                "  No phonons (pn_dim={}, coupling={}), returning LM Hamiltonian only",
                pn_dim,
                self.pn_coupling
            );
            return Ok(lm_ham);
        }

        timed_println_rank0!(
            "  Dimensions: ex={}, ph={}, pn={}, total={}",
            ex_dim,
            ph_dim,
            pn_dim,
            lm_dim * pn_dim
        );

        // Explicit sparse matrices benefit from building PP as a KronSum over
        // phonon product-Fock transitions.
        //
        // Do not use this path for ComposedOperator: PP-level KronSum would store
        // dense LM-space left blocks and can destroy the lazy/matrix-free advantage.
        if Self::use_sparse_pp_kronsum_backend() {
            let time_pp_ks = Instant::now();

            let ham = self.build_pp_hamiltonian_sparse_kronsum(&lm_ham, ph_dim, ex_dim, pn_dim)?;

            timed_println_rank0!(
                "  Sparse PP KronSum construction: {:?}",
                time_pp_ks.elapsed()
            );

            return Ok(ham);
        }

        // Lazy-safe fallback for ComposedOperator and dense backends.
        //
        // Avoid constructing a giant explicit zero matrix first.
        let time_pn = Instant::now();
        let pn_ham = self.phonon.get_h()?;
        timed_println_rank0!("  Build phonon Hamiltonian: {:?}", time_pn.elapsed());

        let time_eye = Instant::now();
        let lm_eye = M::eye(lm_dim);
        let pn_eye = M::eye(pn_dim);
        timed_println_rank0!("  Create identity matrices: {:?}", time_eye.elapsed());

        let time_kron1 = Instant::now();
        let mut ham = lm_ham.kron(&pn_eye);
        timed_println_rank0!(
            "  First kron product (LM ⊗ I_pn): {:?}",
            time_kron1.elapsed()
        );

        let time_kron2 = Instant::now();
        let h_pn = lm_eye.kron(&pn_ham);
        ham = ham.add(&h_pn)?;
        timed_println_rank0!(
            "  Second kron product (I_lm ⊗ H_pn): {:?}",
            time_kron2.elapsed()
        );

        let time_int = Instant::now();
        let h_int = self.get_interaction_hamiltonian_as_operator(ph_dim)?;
        ham = ham.add(&h_int)?;
        timed_println_rank0!("  Interaction Hamiltonian + add: {:?}", time_int.elapsed());

        timed_println!("Total PP Hamiltonian build time: {:?}", time_lm.elapsed());

        Ok(ham)
    }
}
