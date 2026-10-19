use super::{Boson, BosonicHamiltonian, ExcitonicHamiltonian, LMBasisLayout};
use crate::matrix_elements::{LMMatrixElementArray, ModeKey};
use crate::timed_println_rank0;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError, Sqrt};
use ndarray_linalg::c64;
use num_traits::{One, Zero};
use sprs::TriMat;
use std::iter::Sum;
use std::sync::Arc;

pub struct LMHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub exciton: ExcitonicHamiltonian<M>,
    pub photon: BosonicHamiltonian<M, B>,
    pub matrix_elements: LMMatrixElementArray,
    pub couplings: Vec<f64>,
    pub layout: LMBasisLayout,
    pub p_c_sin: Option<Vec<M>>,
    pub p_q: Option<Vec<M>>,
    pub include_photon_recoil: bool,
    is_initialized: bool,
    pub(crate) cached_a_ops: Option<Vec<M>>, // Cache annihilation operators
    pub(crate) cached_adag_a_ops: Option<Vec<M>>, // Cache number operators
    pub cache_dir: Option<String>,           // Cache directory for matrix elements
    pub config_hash: u64,                    // Hash of configuration parameters for caching
}

impl<M: MatrixOps, B: Boson<M>> LMHamiltonian<M, B>
where
    M: Diagonalize,
{
    pub fn new(
        exciton: ExcitonicHamiltonian<M>,
        photon: BosonicHamiltonian<M, B>,
        couplings: Vec<f64>,
    ) -> Self {
        let matrix_elements = LMMatrixElementArray::new();

        Self {
            exciton,
            photon,
            matrix_elements,
            couplings,
            layout: LMBasisLayout::PHOTON_MAJOR,
            p_c_sin: None,
            p_q: None,
            include_photon_recoil: true,
            is_initialized: false,
            cached_a_ops: None,
            cached_adag_a_ops: None,
            cache_dir: None,
            config_hash: 0,
        }
    }

    pub fn new_with_cache(
        exciton: ExcitonicHamiltonian<M>,
        photon: BosonicHamiltonian<M, B>,
        couplings: Vec<f64>,
        cache_dir: Option<String>,
        config_hash: u64,
    ) -> Self {
        let matrix_elements = LMMatrixElementArray::new();

        Self {
            exciton,
            photon,
            matrix_elements,
            couplings,
            layout: LMBasisLayout::PHOTON_MAJOR,
            p_c_sin: None,
            p_q: None,
            include_photon_recoil: true,
            is_initialized: false,
            cached_a_ops: None,
            cached_adag_a_ops: None,
            cache_dir,
            config_hash,
        }
    }

    pub fn initialize(&mut self, k: Option<Vec<f64>>) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        use std::time::Instant;

        let k_changed = match &k {
            Some(k_val) => k_val != &self.exciton.exciton_ham.k,
            None => false,
        };

        if k_changed {
            if let Some(k_val) = k {
                self.update_k(k_val);
            }

            self.is_initialized = false;
            self.cached_a_ops = None;
            self.cached_adag_a_ops = None;
            self.p_c_sin = None;
            self.p_q = None;

            timed_println_rank0!("Updating k to {:?}", &self.exciton.exciton_ham.k);
        }

        if !self.is_initialized {
            let time_eig = Instant::now();
            self.exciton.compute_eigenstates()?;
            timed_println_rank0!("    Exciton diagonalization: {:?}", time_eig.elapsed());

            let all_couplings_zero = self.all_light_matter_couplings_zero();

            let time_pq = Instant::now();
            self.calc_p_q_matrix()?;
            timed_println_rank0!("    P·q matrix calculation: {:?}", time_pq.elapsed());

            if all_couplings_zero {
                timed_println_rank0!(
                    "    All light-matter couplings are zero; skipping Taylor sin/cos/p_cos matrix elements"
                );

                self.p_c_sin = Some(Vec::new());
            } else {
                let time_mat = Instant::now();
                let all_q_ind = self.photon.get_all_q_ind();
                let (relative_hash, exciton_hash) = self.matrix_element_cache_hashes();

                timed_println_rank0!(
                    "Matrix-element cache hashes: relative={:016x}, exciton_basis={:016x}, k={:?}, qk={:?}, k_quantum={}",
                    relative_hash,
                    exciton_hash,
                    self.exciton.exciton_ham.k,
                    self.exciton
                        .exciton_ham
                        .k
                        .iter()
                        .map(|&x| crate::utils::quantize_f64_for_hash(x))
                        .collect::<Vec<_>>(),
                    crate::utils::K_HASH_QUANTUM,
                );

                self.matrix_elements
                    .populate_taylor_matrix_elements_with_cache(
                        &all_q_ind,
                        self.exciton.e_vecs.as_ref().unwrap(),
                        self.photon.boson.get_modes(),
                        self.exciton.exciton_ham.kappa_vectors.len(),
                        self.exciton.exciton_ham.rel_ham.mu,
                        self.cache_dir.as_deref(),
                        false,
                        relative_hash,
                        exciton_hash,
                    );

                timed_println_rank0!("    Matrix elements calculation: {:?}", time_mat.elapsed());

                let time_p = Instant::now();
                self.calc_p_c_sin_matrix()?;
                timed_println_rank0!("    P·c·sin matrix calculation: {:?}", time_p.elapsed());
            }

            let time_cache = Instant::now();
            self.cache_photon_operators()?;
            timed_println_rank0!("    Cache photon operators: {:?}", time_cache.elapsed());

            self.is_initialized = true;
        }

        Ok(())
    }

    fn matrix_element_cache_hashes(&self) -> (u64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Relative-basis Taylor matrix elements depend on photon modes, n_rel,
        // and mu, but not on the exciton eigenvectors or current K.
        //
        // Keep this K-independent so the expensive relative-basis cache can be
        // reused across all bandstructure K points.
        let relative_hash = self.config_hash;

        // Exciton-basis matrix elements depend on the exciton eigenvectors.
        // The exciton eigenvectors depend on the current center-of-mass K point,
        // so the final exciton-basis cache must include K.
        //
        // Important: hash quantized K values rather than raw f64 bits. This prevents
        // harmless interpolation/formatting differences from causing cache misses.
        let mut hasher = DefaultHasher::new();

        self.config_hash.hash(&mut hasher);
        "exciton_basis_quantized_k_v1".hash(&mut hasher);
        crate::utils::K_HASH_QUANTUM.to_bits().hash(&mut hasher);

        for &ki in self.exciton.exciton_ham.k.iter() {
            crate::utils::hash_quantized_f64(ki, &mut hasher);
        }

        let exciton_hash = hasher.finish();

        (relative_hash, exciton_hash)
    }

    pub fn compute_lm_matrix_elements_only(
        &mut self,
        k: Option<Vec<f64>>,
        force_recompute: bool,
    ) -> Result<(), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<c64>,
    {
        use std::time::Instant;

        if let Some(k_val) = k {
            self.update_k(k_val);
        }

        let time_eig = Instant::now();
        self.exciton.compute_eigenstates()?;
        timed_println_rank0!(
            "Exciton diagonalization for matrix-element precompute: {:?}",
            time_eig.elapsed()
        );

        let n_modes = self.photon.boson.get_modes().len();
        let all_q_ind = self.photon.get_all_q_ind();

        timed_println_rank0!(
            "Precomputing LM matrix elements for {} photon modes: {} tasks",
            n_modes,
            all_q_ind.len()
        );
        timed_println_rank0!(
            "  singles = {}, pairs = {}",
            n_modes,
            n_modes * (n_modes.saturating_sub(1)) / 2
        );

        let time_mat = Instant::now();

        let (relative_hash, exciton_hash) = self.matrix_element_cache_hashes();

        timed_println_rank0!(
            "Matrix-element cache hashes: relative={:016x}, exciton_basis={:016x}, k={:?}, qk={:?}, k_quantum={}",
            relative_hash,
            exciton_hash,
            self.exciton.exciton_ham.k,
            self.exciton
                .exciton_ham
                .k
                .iter()
                .map(|&x| crate::utils::quantize_f64_for_hash(x))
                .collect::<Vec<_>>(),
            crate::utils::K_HASH_QUANTUM,
        );

        self.matrix_elements
            .populate_taylor_matrix_elements_with_cache(
                &all_q_ind,
                self.exciton.e_vecs.as_ref().unwrap(),
                self.photon.boson.get_modes(),
                self.exciton.exciton_ham.kappa_vectors.len(),
                self.exciton.exciton_ham.rel_ham.mu,
                self.cache_dir.as_deref(),
                force_recompute,
                relative_hash,
                exciton_hash,
            );

        timed_println_rank0!(
            "LM matrix-element precompute completed in {:?}",
            time_mat.elapsed()
        );

        Ok(())
    }

    fn cache_photon_operators(&mut self) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + One + Zero,
    {
        let n_modes = self.photon.boson.get_modes().len();
        let n_ph = self.photon.boson.dimension();
        let mut a_ops = Vec::with_capacity(n_modes);
        let mut adag_a_ops = Vec::with_capacity(n_modes);

        for i in 0..n_modes {
            a_ops.push(self.photon.get_a(i)?);

            // Materialize a†a as a diagonal operator for better KronSum optimization
            // For single-excitation: dimension is n_modes + 1 (vacuum + n_modes single-photon states)
            // The number operator for mode i is: diag(0, 0, ..., 1, ..., 0) with 1 at position i+1

            if n_ph == n_modes + 1 {
                // Single-excitation case: simple diagonal
                let mut tri_mat = TriMat::new((n_ph, n_ph));
                tri_mat.add_triplet(i + 1, i + 1, M::Scalar::one());

                adag_a_ops.push(M::from_trimat(&tri_mat)?);
            } else {
                // Full Fock space: use the symbolic operator
                // This is more complex, so keep it as-is for now
                adag_a_ops.push(self.photon.get_adag_a(i)?);
            }
        }

        self.cached_a_ops = Some(a_ops);
        self.cached_adag_a_ops = Some(adag_a_ops);

        Ok(())
    }

    fn calc_p_q_matrix(&mut self) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let n_modes = self.photon.boson.get_modes().len();

        let mut p_q: Vec<M> = Vec::with_capacity(n_modes);

        for i in 0..n_modes {
            let q = &self.photon.boson.get_modes()[i].q;
            p_q.push(self.exciton.get_p_com(q)?);
        }

        self.p_q = Some(p_q);
        Ok(())
    }

    fn calc_p_c_sin_matrix(&mut self) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let n_modes = self.photon.boson.get_modes().len();
        let sin_mat = &self.matrix_elements.sin_mat;

        let mut p_c_sin: Vec<M> = Vec::with_capacity(n_modes);

        for i in 0..n_modes {
            let eps = &self.photon.boson.get_modes()[i].eps;
            let key = ModeKey::single(i);

            p_c_sin.push(
                self.exciton
                    .get_p_com(eps)?
                    .dot(&M::from_dense_arc(Arc::clone(
                        sin_mat
                            .get(&key)
                            .unwrap_or_else(|| panic!("Missing sin matrix for {:?}", key)),
                    ))?)?,
            );
        }

        self.p_c_sin = Some(p_c_sin);
        Ok(())
    }

    fn get_interaction_hamiltonian_as_operator(&self) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        use std::time::Instant;

        let cached_a = self
            .cached_a_ops
            .as_ref()
            .expect("Photon operators not cached");
        let cached_adag_a = self
            .cached_adag_a_ops
            .as_ref()
            .expect("Number operators not cached");

        let n_modes = self.photon.boson.get_modes().len();
        let total_pairs = n_modes * n_modes;
        timed_println_rank0!(
            "        Building interaction for {} modes ({} pairs)",
            n_modes,
            total_pairs
        );

        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let n_ph: usize = self.photon.boson.dimension();
        let ex_eye: M = M::eye(n_ex);

        // Collect (exciton_op, photon_op) pairs for KronSum optimization
        let mut kron_pairs: Vec<(M, M)> = Vec::new();

        // Iterate over photon mode to build interaction terms
        self.photon
            .boson
            .get_modes()
            .iter()
            .enumerate()
            .try_for_each(|(i_ph, mode_i)| -> Result<(), MatrixOpsError> {
                let a_i = &cached_a[i_ph];
                let coupling_i = self.couplings[i_ph];

                // Coupling-dependent p·A terms.
                if coupling_i.abs() > 1e-14 {
                    let p_c_sin = &self.p_c_sin.as_ref().unwrap()[i_ph]
                        .scale(2.0 / self.exciton.exciton_ham.m * coupling_i)?;

                    let single_key = ModeKey::single(i_ph);

                    let p_cos = &M::from_dense_arc(Arc::clone(
                        self.matrix_elements
                            .p_cos_mat
                            .get(&single_key)
                            .unwrap_or_else(|| panic!("Missing p_cos matrix for {:?}", single_key)),
                    ))?
                    .scale(coupling_i / self.exciton.exciton_ham.rel_ham.mu)?;

                    // P•A sin(x)
                    kron_pairs.push((p_c_sin.clone(), a_i.clone()));
                    kron_pairs.push((p_c_sin.clone(), a_i.transpose()));

                    // The exact transformed CoM light-matter coupling contains
                    //
                    //   (2/M) [P - sum_{q' != q} q' n_{q'}] · A_q
                    //       sin(q·x/2) (a_q† + a_q).
                    //
                    // Here we include the P · A_q part. The recoil-subtraction part
                    //
                    //   -(2/M) sum_{q' != q} (q'·A_q) n_{q'}
                    //       sin(q·x/2) (a_q† + a_q)
                    //
                    // is projected out in the single-excitation photon basis used by
                    // SingExBosons: n_{q'}(a_q† + a_q) either annihilates the wrong mode
                    // or creates a two-photon state outside the truncated subspace.
                    // If FullBosons is used, this recoil-subtraction term should be added.

                    // p•A cos(x)
                    kron_pairs.push((p_cos.clone(), a_i.scale(-1.0)?));
                    kron_pairs.push((p_cos.clone(), a_i.transpose()));
                }

                // P•q a†a term is not proportional to LM coupling in the current Hamiltonian.
                if self.include_photon_recoil {
                    let p_q = &self.p_q.as_ref().unwrap()[i_ph]
                        .scale(-1.0 / self.exciton.exciton_ham.m)?;
                    kron_pairs.push((p_q.clone(), cached_adag_a[i_ph].clone()));
                }

                // A² terms
                self.photon
                    .boson
                    .get_modes()
                    .iter()
                    .enumerate()
                    .try_for_each(|(j_ph, mode_j)| -> Result<(), MatrixOpsError> {
                        let a_j = &cached_a[j_ph];

                        let coupling_i = self.couplings[i_ph];
                        let coupling_j = self.couplings[j_ph];
                        let coupling_2 = coupling_i * coupling_j * mode_i.eps.dot(&mode_j.eps)
                            / 2.0
                            / self.exciton.exciton_ham.rel_ham.mu;

                        if coupling_2.abs() > 1e-14 {
                            let cos_diff: M = if i_ph == j_ph {
                                ex_eye.clone()
                            } else {
                                let q_diff_key = ModeKey::diff(i_ph, j_ph);

                                M::from_dense_arc(Arc::clone(
                                    self.matrix_elements
                                        .cos_mat
                                        .get(&q_diff_key)
                                        .unwrap_or_else(|| panic!("Missing cos-diff matrix for {:?}", q_diff_key)),
                                ))?
                            };

                            kron_pairs.push((cos_diff.scale(coupling_2)?, a_i.dot(&a_j.transpose())?));
                            kron_pairs.push((cos_diff.scale(coupling_2)?, a_i.transpose().dot(a_j)?));
                        }

                        // NOTE on the A^2 term:
                        //
                        // The full transformed diamagnetic term contains both number-conserving pieces
                        //     cos[(q-q')·x/2] (a_q† a_q' + a_q a_q'†)
                        // and pair creation/annihilation pieces
                        //     cos[(q+q')·x/2] (a_q a_q' + a_q† a_q'†).
                        //
                        // In the single-excitation photon basis used by SingExBosons, the pair
                        // creation/annihilation pieces leave the truncated Hilbert space and are
                        // therefore projected out. Thus we retain only the number-conserving A^2 terms.
                        //
                        // If this code is used with FullBosons, the pair terms should be added explicitly.

                        // (q a†a)^2 term
                        if self.include_photon_recoil {
                            let q_q =
                                mode_i.q.dot(&mode_j.q) / 2.0 / self.exciton.exciton_ham.m;
                            kron_pairs.push((
                                ex_eye.scale(q_q)?,
                                cached_adag_a[i_ph].dot(&cached_adag_a[j_ph])?,
                            ));
                        }

                        Ok(())
                    })?;

                Ok(())
            })?;

        timed_println_rank0!(
            "        include_photon_recoil = {}, kron_pairs = {}",
            self.include_photon_recoil,
            kron_pairs.len()
        );

        // Use the optimized builder - will automatically detect and optimize KronSum patterns
        timed_println_rank0!(
            "        Building KronSum from {} term pairs...",
            kron_pairs.len()
        );
        let sum_time = Instant::now();
        let interaction =
            M::build_kron_sum_with_layout(kron_pairs, n_ex, n_ph, self.layout.ex_ph_layout);
        timed_println_rank0!("        Built interaction in {:?}", sum_time.elapsed());

        interaction
    }

    pub fn build_lm_hamiltonian(&mut self, k: Option<Vec<f64>>) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        use std::time::Instant;

        self.initialize(k)?;

        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let n_ph = self.photon.boson.dimension();

        let time_eye = Instant::now();
        let ex_eye: M = M::eye(n_ex);
        let ph_eye: M = M::eye(n_ph);
        timed_println_rank0!("      LM identity matrices: {:?}", time_eye.elapsed());

        let time_ex = Instant::now();
        let h_ex = self.exciton.get_h()?;
        let mut ham = self.layout.kron_ex_ph(&h_ex, &ph_eye);
        timed_println_rank0!("      LM exciton kron: {:?}", time_ex.elapsed());

        let time_ph = Instant::now();
        let h_photon = self.photon.get_h()?;
        let h_ph = self.layout.kron_ex_ph(&ex_eye, &h_photon);
        ham = ham.add(&h_ph)?;
        timed_println_rank0!("      LM photon kron: {:?}", time_ph.elapsed());

        let time_int = Instant::now();
        let h_int = self.get_interaction_hamiltonian_as_operator()?;
        ham = ham.add(&h_int)?;
        timed_println_rank0!("      LM interaction + sum: {:?}", time_int.elapsed());

        Ok(ham)
    }

    pub fn update_k(&mut self, k: Vec<f64>) {
        self.exciton.exciton_ham.k = k;
    }

    #[inline]
    fn all_light_matter_couplings_zero(&self) -> bool {
        self.couplings.iter().all(|g| g.abs() <= 1e-14)
    }
}
