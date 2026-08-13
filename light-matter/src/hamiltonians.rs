use crate::matrix_elements::{LMMatrixElementArray, ModeKey, ModeTask};
use crate::timed_println_rank0;
use crate::timed_println;
use crate::utils::{BosonMode, PPConfig, Polarization, Vector3D};
use exciton_2d::Exciton2DGridSystem;
use exciton_2d::FullHamiltonian;
use log::warn;
use matrix_ops::{ConvertFrom, Diagonalize, MatrixOps, MatrixOpsError, Sqrt};
use ndarray::{Array1, Array2, arr1, s};
use ndarray_linalg::c64;
use num_traits::{One, Zero};
use sprs::TriMat;
use std::iter::Sum;

pub trait Boson<M: MatrixOps> {
    fn new(modes: Vec<BosonMode>) -> Self;

    fn a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt;

    fn adag_a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt,
    {
        let a = self.a(mode_i)?;
        a.transpose().dot(&a)
    }

    fn get_modes(&self) -> &Vec<BosonMode>;

    fn dimension(&self) -> usize;

    // fn total_momentum(&self) -> Result<M, MatrixOpsError>;
}

pub struct FullBosons<M: MatrixOps> {
    pub bosonic_modes: Vec<BosonMode>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> Boson<M> for FullBosons<M> {
    fn new(modes: Vec<BosonMode>) -> Self {
        Self {
            bosonic_modes: modes,
            _phantom: std::marker::PhantomData,
        }
    }

    fn a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt,
    {
        self.bosonic_modes
            .iter()
            .enumerate()
            .try_fold(None, |acc: Option<M>, (idx, mode)| {
                let sing_mat = if idx == mode_i {
                    // Create annihilation operator for this mode
                    let dim = mode.n_fock;
                    let mut trimat = TriMat::new((dim, dim));

                    for n in 1..dim {
                        trimat.add_triplet(n - 1, n, (M::Scalar::convert_from(n)).sqrt());
                    }

                    M::from_trimat(&trimat)?
                } else {
                    // Identity for other modes
                    M::eye(mode.n_fock)
                };

                Ok(Some(match acc {
                    Some(acc) => M::kron(&acc, &sing_mat),
                    None => sing_mat,
                }))
            })
            .map(|opt| opt.expect("bosonic_modes should not be empty"))
    }

    fn get_modes(&self) -> &Vec<BosonMode> {
        &self.bosonic_modes
    }

    fn dimension(&self) -> usize {
        self.get_modes().iter().map(|mode| mode.n_fock).product()
    }

    // fn total_momentum(&self) -> Result<M, MatrixOpsError> {
    //     let dim = self.dimension();

    //     self
    //         .bosonic_modes
    //         .iter()
    //         .enumerate()
    //         .try_fold(None, |acc: Option<Array1<M::Scalar>>, (idx, mode)| {

    //             let sing_diag = Array1::from_iter((0..mode.n_fock).map(|n| M::Scalar::convert_from(n as f64 * mode.q.norm())));

    //             Ok(Some(match acc {
    //                 Some(acc) => acc.kron(&sing_diag),
    //                 None => sing_diag,
    //             }))
    //         })
    // }
}

/// Single-excitation boson basis:
///   |0>, |1_0>, |1_1>, ..., |1_{N-1}>.
/// The annihilation operator a_i maps |1_i> -> |0>.
/// The creation operator a_i† is represented as a_i^T and maps |0> -> |1_i>.
/// Operators that would create two photons are projected out of this truncated
/// Hilbert space. Therefore pair creation terms in A^2 and certain recoil terms
/// vanish by projection in this basis.
pub struct SingExBosons<M: MatrixOps> {
    pub bosonic_modes: Vec<BosonMode>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> Boson<M> for SingExBosons<M> {
    fn new(modes: Vec<BosonMode>) -> Self {
        Self {
            bosonic_modes: modes,
            _phantom: std::marker::PhantomData,
        }
    }

    fn a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt + One,
    {
        let dim = self.dimension();
        let mut trimat = TriMat::new((dim, dim));
        trimat.add_triplet(0, mode_i + 1, <M::Scalar>::one());

        M::from_trimat(&trimat)
    }

    fn get_modes(&self) -> &Vec<BosonMode> {
        &self.bosonic_modes
    }

    fn dimension(&self) -> usize {
        self.get_modes().len() + 1
    }
}

pub struct BosonicHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub boson: B,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps, B: Boson<M>> BosonicHamiltonian<M, B> {
    pub fn new(bosonic_modes: Vec<BosonMode>) -> Self {
        Self {
            boson: B::new(bosonic_modes),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt,
    {
        self.boson.a(mode_i)
    }

    pub fn get_all_q_ind(&self) -> Vec<ModeTask> {
        let mut tasks = Vec::new();
        let n_ph = self.boson.get_modes().len();

        for mode in 0..n_ph {
            tasks.push(ModeTask::PhotonSingle { mode });
        }

        for q1 in 0..n_ph {
            for q2 in (q1 + 1)..n_ph {
                tasks.push(ModeTask::PhotonPair { q1, q2 });
            }
        }

        tasks
    }

    pub fn get_adag_a(&self, mode_i: usize) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt,
    {
        self.boson.adag_a(mode_i)
    }

    /// Gets the complete photonic Hamiltonian
    pub fn get_h(&self) -> Result<M, MatrixOpsError>
    where
        M::Scalar: Sqrt,
    {
        let bosonic_modes = self.boson.get_modes();

        // Handle empty vector case
        if bosonic_modes.is_empty() {
            return Ok(M::eye(1));
        }

        // Sum Hamiltonians for each mode

        let dim = self.dimension();

        bosonic_modes
            .iter()
            .enumerate()
            .try_fold(M::zeros((dim, dim)), |acc, (mode_i, mode)| {
                let h_ph_sing = self.get_adag_a(mode_i)?.scale(mode.omega)?;

                acc.add(&h_ph_sing)
            })
    }

    pub fn dimension(&self) -> usize {
        self.boson.dimension()
    }
}

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

pub struct LMHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub exciton: ExcitonicHamiltonian<M>,
    pub photon: BosonicHamiltonian<M, B>,
    pub matrix_elements: LMMatrixElementArray,
    pub couplings: Vec<f64>,
    pub p_c_sin: Option<Vec<M>>,
    pub p_q: Option<Vec<M>>,
    is_initialized: bool,
    cached_a_ops: Option<Vec<M>>,      // Cache annihilation operators
    cached_adag_a_ops: Option<Vec<M>>, // Cache number operators
    pub cache_dir: Option<String>,     // Cache directory for matrix elements
    pub config_hash: u64,              // Hash of configuration parameters for caching
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
            p_c_sin: None,
            p_q: None,
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
            p_c_sin: None,
            p_q: None,
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

            let time_mat = Instant::now();
            let all_q_ind = self.photon.get_all_q_ind();
            self.matrix_elements.populate_tay_mat_elements_with_cache(
                &all_q_ind,
                self.exciton.e_vecs.as_ref().unwrap(),
                self.photon.boson.get_modes(),
                self.exciton.exciton_ham.kappa_vectors.len(),
                self.exciton.exciton_ham.rel_ham.mu,
                self.cache_dir.as_deref(),
                false,
                self.config_hash,
            );
            timed_println_rank0!("    Matrix elements calculation: {:?}", time_mat.elapsed());

            let time_p = Instant::now();
            self.calc_p_c_sin_matrix()?;
            timed_println_rank0!("    P·c·sin matrix calculation: {:?}", time_p.elapsed());

            let time_cache = Instant::now();
            self.cache_photon_operators()?;
            timed_println_rank0!("    Cache photon operators: {:?}", time_cache.elapsed());

            self.is_initialized = true;
        }

        Ok(())
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

        self.matrix_elements.populate_tay_mat_elements_with_cache(
            &all_q_ind,
            self.exciton.e_vecs.as_ref().unwrap(),
            self.photon.boson.get_modes(),
            self.exciton.exciton_ham.kappa_vectors.len(),
            self.exciton.exciton_ham.rel_ham.mu,
            self.cache_dir.as_deref(),
            force_recompute,
            self.config_hash,
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

    fn calc_p_c_sin_matrix(&mut self) -> Result<(), MatrixOpsError>
    where
        M::Scalar: Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let n_ph = self.photon.boson.get_modes().len() + 1; // +1 for ground state

        let sin_mat = &self.matrix_elements.sin_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // timed_println_rank0!("{:?}",test);

        let mut p_c_sin: Vec<M> = Vec::new();
        let mut p_q: Vec<M> = Vec::new();
        (0..n_ph - 1).try_for_each(|i| -> Result<(), MatrixOpsError> {
            let eps = &self.photon.boson.get_modes()[i].eps;
            let q = &self.photon.boson.get_modes()[i].q;
            let key = ModeKey::single(i);

            p_c_sin.push(
                self.exciton
                    .get_p_com(eps)? // P·eps
                    .dot(&M::from_dense(sin_mat.get(&key).unwrap_or_else(|| {
                        panic!("Missing sin matrix for {:?}", key)
                    }))?)?,
            );

            p_q.push(
                self.exciton.get_p_com(q)?, // P·q
            );

            Ok(())
        })?;

        self.p_c_sin = Some(p_c_sin);
        self.p_q = Some(p_q);
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
                // p.A terms
                let a_i = &cached_a[i_ph];
                let coupling_i = self.couplings[i_ph];
                let p_c_sin = &self.p_c_sin.as_ref().unwrap()[i_ph]
                    .scale(2.0 / self.exciton.exciton_ham.m * coupling_i)?;
                let p_q =
                    &self.p_q.as_ref().unwrap()[i_ph].scale(-1.0 / self.exciton.exciton_ham.m)?;

                let single_key = ModeKey::single(i_ph);

                let p_cos = &M::from_dense(
                    &(self
                        .matrix_elements
                        .p_cos_mat
                        .get(&single_key)
                        .unwrap_or_else(|| panic!("Missing p_cos matrix for {:?}", single_key))
                        * (coupling_i / self.exciton.exciton_ham.rel_ham.mu)),
                )?;

                // P•A sin(x) terms - these are perfect KronSum candidates
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

                // p•A cos(x) terms
                kron_pairs.push((p_cos.clone(), a_i.scale(-1.0)?));
                kron_pairs.push((p_cos.clone(), a_i.transpose()));

                // P•q a†a term
                kron_pairs.push((p_q.clone(), cached_adag_a[i_ph].clone()));

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

                        let cos_diff: M = if i_ph == j_ph {
                            // cos[(q-q)·x/2] = 1
                            ex_eye.clone()
                        } else {
                            let q_diff_key = ModeKey::diff(i_ph, j_ph);

                            M::from_dense(
                                self.matrix_elements
                                    .cos_mat
                                    .get(&q_diff_key)
                                    .unwrap_or_else(|| {
                                        panic!("Missing cos-diff matrix for {:?}", q_diff_key)
                                    }),
                            )?
                        };

                        // let a_diff = &(a_i.dot(&a_j.transpose()).add(&a_i.transpose().dot(a_j)));

                        // cos(q - q') term
                        kron_pairs.push((cos_diff.scale(coupling_2)?, a_i.dot(&a_j.transpose())?));
                        kron_pairs.push((cos_diff.scale(coupling_2)?, a_i.transpose().dot(a_j)?));

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
                        let q_q = mode_i.q.dot(&mode_j.q) / 2.0 / self.exciton.exciton_ham.m;
                        kron_pairs.push((
                            ex_eye.scale(q_q)?,
                            cached_adag_a[i_ph].dot(&cached_adag_a[j_ph])?,
                        ));

                        Ok(())
                    })?;

                Ok(())
            })?;

        // Use the optimized builder - will automatically detect and optimize KronSum patterns
        timed_println_rank0!(
            "        Building KronSum from {} term pairs...",
            kron_pairs.len()
        );
        let sum_time = Instant::now();
        let interaction = M::build_kron_sum(kron_pairs, n_ex, n_ph);
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
        let mut ham = M::kron(&self.exciton.get_h()?, &ph_eye);
        timed_println_rank0!("      LM exciton kron: {:?}", time_ex.elapsed());

        let time_ph = Instant::now();
        let h_ph = M::kron(&ex_eye, &self.photon.get_h()?);
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
}

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
    ///     index = (((n0 * d1 + n1) * d2 + n2) ...)
    #[inline]
    fn phonon_occupations_from_index(mut idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut occ = vec![0; dims.len()];

        for mode_idx in (0..dims.len()).rev() {
            let dim = dims[mode_idx];
            occ[mode_idx] = idx % dim;
            idx /= dim;
        }

        occ
    }

    /// Converts product-Fock occupations into a flat index.
    ///
    /// This is the inverse of `phonon_occupations_from_index`.
    #[inline]
    fn phonon_index_from_occupations(occ: &[usize], dims: &[usize]) -> usize {
        occ.iter()
            .zip(dims.iter())
            .fold(0usize, |acc, (&n, &dim)| acc * dim + n)
    }

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
            .map(|idx| Self::phonon_occupations_from_index(idx, &dims))
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
                    let pn_to = Self::phonon_index_from_occupations(&occ_to, &dims);

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
                    let pn_to = Self::phonon_index_from_occupations(&occ_to, &dims);

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
