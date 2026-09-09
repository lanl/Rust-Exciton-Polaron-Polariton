mod fock;

use super::{
    Boson, BosonicHamiltonian, ExcitonicHamiltonian, FullBosons, LMHamiltonian, PPBasisLayout,
};
use crate::config::PPConfig;
use crate::matrix_elements::{LMMatrixElementArray, ModeKey};
use crate::modes::types::BosonMode;
use crate::timed_println_rank0;
use exciton_2d::Exciton2DGridSystem;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError, Sqrt};
use ndarray::arr1;
use ndarray_linalg::c64;
use sprs::TriMat;
use std::iter::Sum;
use std::sync::Arc;

pub struct PPHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub polariton: LMHamiltonian<M, B>,
    pub phonon: BosonicHamiltonian<M, FullBosons<M>>,
    pub pn_coupling: f64,
    pub layout: PPBasisLayout,
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
            layout: PPBasisLayout::DEFAULT,
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
            .zip(pp_config.mode_polarizations.iter())
            .map(|(q, &pol)| BosonMode::new(*q, pol, 2))
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

    #[inline]
    fn kron_ex_ph(&self, ex_op: &M, ph_op: &M) -> M {
        self.layout.lm.kron_ex_ph(ex_op, ph_op)
    }

    #[inline]
    fn kron_lm_pn(&self, lm_op: &M, pn_op: &M) -> M {
        self.layout.kron_lm_pn(lm_op, pn_op)
    }

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
    fn _get_interaction_hamiltonian_as_operator(&self, ph_dim: usize) -> Result<M, MatrixOpsError>
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
                let lm_op = self.kron_ex_ph(&sin_pn[idx1], &ph_eye);
                let pn_op = (b.add(&b_dag))?.scale(-2.0 * self.pn_coupling)?;

                Self::add_term(&mut interaction, self.kron_lm_pn(&lm_op, &pn_op))?;

                // ------------------------------------------------------------
                // Exciton-phonon momentum recoil:
                //
                //     -(P·k / M) n_k
                // ------------------------------------------------------------
                let p_k = self.polariton.exciton.get_p_com(k1)?;
                let lm_op = self.kron_ex_ph(&p_k, &ph_eye);
                let n_k1 = self.phonon.get_adag_a(idx1)?;

                Self::add_term(
                    &mut interaction,
                    self.kron_lm_pn(&lm_op, &n_k1.scale(-1.0 / m_com)?),
                )?;

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

                        Self::add_term(&mut interaction, self.kron_lm_pn(&lm_eye, &pn_op))?;

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

                        let photon_number_lm =
                            self.kron_ex_ph(&ex_eye, &cached_photon_numbers[ph_idx]);
                        let lm_op = photon_number_lm.scale(q_dot_k / m_com)?;

                        Self::add_term(&mut interaction, self.kron_lm_pn(&lm_op, &n_k1))?;

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

                        let ex_op_m = M::from_dense_arc(Arc::clone(ex_op))?;
                        let lm_op = self.kron_ex_ph(&ex_op_m, &ph_op);
                        let pn_op = n_k1.scale(-2.0 * coupling / m_com)?;

                        Self::add_term(&mut interaction, self.kron_lm_pn(&lm_op, &pn_op))?;

                        Ok(())
                    })?;

                Ok(())
            })?;

        let total_dim = lm_dim * pn_dim;

        Ok(interaction.unwrap_or_else(|| M::zeros((total_dim, total_dim))))
    }

    fn build_expn_extra_hamiltonian(
        &self,
        ex_dim: usize,
        pn_dim: usize,
    ) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let sin_pn = self.sin_pn.as_ref().ok_or_else(|| MatrixOpsError::Other {
            message: "phonon sine matrix elements are not initialized".to_string(),
        })?;

        let m_com = self.polariton.exciton.exciton_ham.m;

        let ex_eye = M::eye(ex_dim);

        let mut terms: Vec<M> = Vec::new();

        // ---------------------------------------------------------------------
        // I_ex ⊗ H_pn
        // ---------------------------------------------------------------------
        let h_pn = self.phonon.get_h()?;
        terms.push(self.layout.kron_ex_pn(&ex_eye, &h_pn));

        // Cache phonon number operators once.
        let n_pn_modes = self.phonon.boson.get_modes().len();

        let mut phonon_number_ops = Vec::with_capacity(n_pn_modes);
        for mode_idx in 0..n_pn_modes {
            phonon_number_ops.push(self.phonon.get_adag_a(mode_idx)?);
        }

        // ---------------------------------------------------------------------
        // Phonon-exciton Fröhlich coupling:
        //
        //     -2 gamma_k sin(k·x/2) (b_k† + b_k)
        // ---------------------------------------------------------------------
        for mode_idx in 0..n_pn_modes {
            let b = self.phonon.get_a(mode_idx)?;
            let b_dag = b.transpose();

            let pn_op = b
                .add(&b_dag)?
                .scale(-2.0 * self.pn_coupling)?;

            terms.push(self.layout.kron_ex_pn(&sin_pn[mode_idx], &pn_op));
        }

        // ---------------------------------------------------------------------
        // Exciton-phonon momentum recoil:
        //
        //     -(P·k / M) n_k
        // ---------------------------------------------------------------------
        for (mode_idx, mode) in self.phonon.boson.get_modes().iter().enumerate() {
            let p_k = self.polariton.exciton.get_p_com(&mode.q)?;
            let p_k = p_k.scale(-1.0 / m_com)?;

            terms.push(self.layout.kron_ex_pn(&p_k, &phonon_number_ops[mode_idx]));
        }

        // ---------------------------------------------------------------------
        // Full phonon-phonon recoil:
        //
        //     sum_{k,k'} (k·k') n_k n_k' / (2M)
        // ---------------------------------------------------------------------
        for (idx1, mode_1) in self.phonon.boson.get_modes().iter().enumerate() {
            for (idx2, mode_2) in self.phonon.boson.get_modes().iter().enumerate() {
                let coeff = mode_1.q.dot(&mode_2.q) / (2.0 * m_com);

                if coeff.abs() <= 1e-14 {
                    continue;
                }

                let pn_op = phonon_number_ops[idx1]
                    .dot(&phonon_number_ops[idx2])?
                    .scale(coeff)?;

                terms.push(self.layout.kron_ex_pn(&ex_eye, &pn_op));
            }
        }

        if terms.is_empty() {
            return Ok(M::zeros((ex_dim * pn_dim, ex_dim * pn_dim)));
        }

        M::sum(terms)
    }

    #[inline]
    fn is_single_excitation_photon_basis(&self) -> bool {
        self.polariton.photon.dimension()
            == self.polariton.photon.boson.get_modes().len() + 1
    }

    #[inline]
    fn push_photon_transition_pair(
        pairs: &mut Vec<(M, M)>,
        block: M,
        ph_dim: usize,
        from: usize,
        to: usize,
        value: f64,
    ) -> Result<(), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<f64>,
    {
        if value.abs() > 1e-14 {
            pairs.push((block, Self::single_transition(ph_dim, from, to, value)?));
        }

        Ok(())
    }

    fn build_photon_transition_pp_corrections(
        &self,
        ph_dim: usize,
        ex_dim: usize,
        pn_dim: usize,
    ) -> Result<Option<M>, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        if !self.is_single_excitation_photon_basis() {
            return Err(MatrixOpsError::Other {
                message: "Photon-transition PP KronSum requires SingExBosons photon basis".to_string(),
            });
        }

        let expn_dim = self.layout.expn_dim(ex_dim, pn_dim)?;
        let ph_expn_layout = self.layout.ph_expn_layout()?;

        let m_com = self.polariton.exciton.exciton_ham.m;

        let ex_eye = M::eye(ex_dim);

        let mut pairs: Vec<(M, M)> = Vec::new();

        // Cache phonon number operators once.
        let n_pn_modes = self.phonon.boson.get_modes().len();

        let mut phonon_number_ops = Vec::with_capacity(n_pn_modes);
        for mode_idx in 0..n_pn_modes {
            phonon_number_ops.push(self.phonon.get_adag_a(mode_idx)?);
        }

        // ---------------------------------------------------------------------
        // Photon-phonon number recoil:
        //
        //     +(1/M) sum_{q,k} q·k n_q n_k
        //
        // In SingExBosons:
        //
        //     n_q = |q+1><q+1|
        // ---------------------------------------------------------------------
        for (ph_idx, ph_mode) in self.polariton.photon.boson.get_modes().iter().enumerate() {
            let ph_state = ph_idx + 1;

            for (pn_idx, pn_mode) in self.phonon.boson.get_modes().iter().enumerate() {
                let coeff = ph_mode.q.dot(&pn_mode.q) / m_com;

                if coeff.abs() <= 1e-14 {
                    continue;
                }

                let pn_block = phonon_number_ops[pn_idx].scale(coeff)?;
                let block = self.layout.kron_ex_pn(&ex_eye, &pn_block);

                Self::push_photon_transition_pair(
                    &mut pairs,
                    block,
                    ph_dim,
                    ph_state,
                    ph_state,
                    1.0,
                )?;
            }
        }

        // ---------------------------------------------------------------------
        // Photon-phonon correction to CoM light-matter coupling:
        //
        //     -(2/M) (A_q·k) sin(q·x/2) (a_q + a_q†) n_k
        //
        // In SingExBosons:
        //
        //     a_q  = |0><q+1|
        //     a_q† = |q+1><0|
        // ---------------------------------------------------------------------
        for (pn_idx, pn_mode) in self.phonon.boson.get_modes().iter().enumerate() {
            let k = &pn_mode.q;
            let n_k = &phonon_number_ops[pn_idx];

            for (ph_idx, ph_mode) in self.polariton.photon.boson.get_modes().iter().enumerate() {
                let coupling = ph_mode.eps.dot(k) * self.polariton.couplings[ph_idx];

                if coupling.abs() <= 1e-14 {
                    continue;
                }

                let key = ModeKey::single(ph_idx);

                let ex_op = self
                    .polariton
                    .matrix_elements
                    .sin_mat
                    .get(&key)
                    .unwrap_or_else(|| panic!("Missing photon sine matrix for {:?}", key));

                let ex_op_m = M::from_dense_arc(Arc::clone(ex_op))?;

                let pn_block = n_k.scale(-2.0 * coupling / m_com)?;
                let block = self.layout.kron_ex_pn(&ex_op_m, &pn_block);

                // a_q = |0><q+1|
                Self::push_photon_transition_pair(
                    &mut pairs,
                    block.clone(),
                    ph_dim,
                    ph_idx + 1,
                    0,
                    1.0,
                )?;

                // a_q† = |q+1><0|
                Self::push_photon_transition_pair(
                    &mut pairs,
                    block,
                    ph_dim,
                    0,
                    ph_idx + 1,
                    1.0,
                )?;
            }
        }

        if pairs.is_empty() {
            return Ok(None);
        }

        timed_println_rank0!(
            "  Building PP photon-transition correction KronSum: {} terms, block_dim={}, ph_dim={}",
            pairs.len(),
            expn_dim,
            ph_dim,
        );

        let correction = M::build_kron_sum_with_layout(
            pairs,
            expn_dim,
            ph_dim,
            ph_expn_layout,
        )?;

        Ok(Some(correction))
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

        let pn_dim = self.phonon.dimension();

        // Fast no-phonon path. Do not call PP initialize(), because that computes
        // phonon matrix elements that are unused when pn_dim <= 1 or pn_coupling == 0.
        if pn_dim == 0 || pn_dim == 1 || self.pn_coupling == 0.0 {
            let time_lm = Instant::now();
            let lm_ham = self.polariton.build_lm_hamiltonian(k.clone())?;

            timed_println_rank0!("  Build LM Hamiltonian: {:?}", time_lm.elapsed());
            timed_println_rank0!(
                "  No phonons (pn_dim={}, coupling={}), returning LM Hamiltonian only",
                pn_dim,
                self.pn_coupling
            );

            return Ok(lm_ham);
        }

        if !self.is_initialized {
            self.initialize(k.clone())?;
        }

        let ph_dim = self.polariton.photon.dimension();
        let ex_dim = self.polariton.exciton.dimension();

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

        if self.is_single_excitation_photon_basis() {
            let time_pp = Instant::now();
            let pn_eye = M::eye(pn_dim);
            let ph_eye = M::eye(ph_dim);

            // ------------------------------------------------------------
            // 1. H_LM ⊗ I_pn
            //
            // This uses the cached LM Hamiltonian, so all light-matter physics stays
            // centralized in LMHamiltonian.
            // ------------------------------------------------------------
            let h_lm_lifted = self.layout.kron_lm_pn(&lm_ham, &pn_eye);

            // ------------------------------------------------------------
            // 2. I_ph ⊗ H_expn_extra
            //
            // Photon-independent phonon/exciton terms.
            // ------------------------------------------------------------
            let h_expn_extra = self.build_expn_extra_hamiltonian(ex_dim, pn_dim)?;

            let h_expn_lifted = self
                .layout
                .kron_ph_expn(&ph_eye, &h_expn_extra)?;

            // ------------------------------------------------------------
            // 3. Photon-transition PP corrections.
            // ------------------------------------------------------------
            let mut terms = vec![h_lm_lifted, h_expn_lifted];

            if let Some(h_corr) =
                self.build_photon_transition_pp_corrections(ph_dim, ex_dim, pn_dim)?
            {
                terms.push(h_corr);
            }

            let ham = M::sum(terms)?;

            timed_println_rank0!(
                "  PP construction from cached H_LM + EXPN block + photon-transition KronSum: {:?}",
                time_pp.elapsed()
            );

            return Ok(ham);
        }
        else {
            return Result::Err(MatrixOpsError::Other {
                message: "PP Hamiltonian construction is only implemented for SingExBosons photon basis".to_string(),
            });
        }
    }
}
