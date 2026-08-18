use super::PreparedExcitonHamiltonian;
use crate::lattice::KappaKey;
use crate::matrix_elements::MatrixElementCalculator;
use crate::relative::RelativeHamiltonian;
use crate::utils::exciton_flat_index;

use log::{info, warn};
use matrix_ops::{ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use sprs::TriMat;
use std::collections::HashMap;
use std::time::Instant;

/// Full Hamiltonian for the exciton including center-of-mass motion
///
/// This struct combines the relative coordinate Hamiltonian with the center-of-mass
/// kinetic energy and the periodic potential. It provides methods for constructing
/// and diagonalizing the full Hamiltonian matrix, which describes both the internal
/// motion of the electron-hole pair and its motion through the periodic potential.
#[derive(Clone)]
pub struct FullHamiltonian<M: MatrixOps> {
    pub rel_ham: RelativeHamiltonian,
    pub matrix_elem_calc: MatrixElementCalculator,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub vector_sum_to_index: HashMap<(usize, KappaKey), usize>,
    pub w_kappa: HashMap<KappaKey, f64>,
    pub a0: f64,
    pub m: f64,
    pub k: Vec<f64>,
    pub verbose: bool,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> FullHamiltonian<M>
where
    M::Scalar: ConvertFrom<Complex64> + ConvertFrom<f64>,
{
    /// Creates a new full Hamiltonian
    ///
    /// # Arguments
    /// * `rel_ham` - Relative coordinate Hamiltonian
    /// * `matrix_elem_calc` - Matrix element calculator
    /// * `kappa_vectors` - Reciprocal lattice vectors
    /// * `vector_sum_to_index` - Lookup table for vector sums
    /// * `w_kappa` - Fourier coefficients of the potential
    /// * `m` - Total mass of the electron-hole pair
    /// * `k` - Wave vector
    /// * `verbose` - Whether to print detailed information
    pub fn new(
        rel_ham: RelativeHamiltonian,
        matrix_elem_calc: MatrixElementCalculator,
        kappa_vectors: Vec<Vec<f64>>,
        vector_sum_to_index: HashMap<(usize, KappaKey), usize>,
        w_kappa: HashMap<KappaKey, f64>,
        a0: f64,
        m: f64,
        k: Vec<f64>,
        verbose: bool,
    ) -> Self {
        Self {
            rel_ham,
            matrix_elem_calc,
            kappa_vectors,
            vector_sum_to_index,
            w_kappa,
            a0,
            m,
            k,
            verbose,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Prepares the K-independent part of the exciton Hamiltonian.
    ///
    /// This builds the Hamiltonian without the CoM kinetic term:
    ///
    ///     H_0 = I_com ⊗ H_rel + H_lattice,
    ///
    /// and stores the relative-coordinate eigenbasis and CoM reciprocal vectors.
    /// For each K point, call `PreparedExcitonHamiltonian::hamiltonian_at_k`.
    pub fn prepare(
        &self,
        n_rel_states: usize,
    ) -> Result<PreparedExcitonHamiltonian<M>, MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
    {
        let (h_no_k, rel_eigvals, rel_eigvecs) = self.build_hamiltonian_no_k(n_rel_states)?;

        let n_rel = rel_eigvals.len();
        let n_com = self.kappa_vectors.len();

        Ok(PreparedExcitonHamiltonian {
            h_no_k,
            rel_eigvals,
            rel_eigvecs,
            kappa_vectors: self.kappa_vectors.clone(),
            n_rel,
            n_com,
            mass: self.m,
        })
    }

    /// Builds the Hamiltonian without center-of-mass kinetic energy
    ///
    /// This method constructs the Hamiltonian matrix that includes the relative coordinate
    /// energy and the periodic potential, but not the center-of-mass kinetic energy.
    /// With -2i*sin(κ·r) matrix elements, the Hamiltonian is complex but Hermitian.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// * Tuple containing the complex Hamiltonian matrix, relative eigenvalues, and relative eigenvectors
    pub fn build_hamiltonian_no_k(
        &self,
        n_rel_states: usize,
    ) -> Result<(M, Array1<f64>, Array2<f64>), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
    {
        // Solve for relative coordinate eigenstates
        let (rel_eigvals, rel_eigvecs, _parities) = self.rel_ham.solve(n_rel_states);

        // With sine matrix elements, we cannot filter by parity as even and odd states couple
        // Use all states
        let n_rel_states_used = rel_eigvals.len();

        if self.verbose {
            info!(
                "Using all {} relative coordinate states (sine coupling mixes parities)",
                n_rel_states_used
            );
        }

        // Set the cached eigenvectors for matrix element calculations
        self.matrix_elem_calc
            .set_cached_eigvecs(rel_eigvecs.clone());

        // Get number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();

        // Total dimension of the Hamiltonian
        let dim = n_rel_states_used * n_com_states;

        if self.verbose {
            info!("Building full Hamiltonian with dimension {}x{}", dim, dim);
            info!(
                "Using {} relative states and {} COM states",
                n_rel_states_used, n_com_states
            );
        }

        // Convert w_kappa to use tuple keys and filter out zero values
        let mut w_kappa_tuples = HashMap::new();
        for (k, &v) in &self.w_kappa {
            if v.abs() > 1e-10 {
                // Only keep non-zero values
                w_kappa_tuples.insert(*k, v);
            }
        }

        // Debug: Print number of non-zero potential terms
        if self.verbose {
            info!(
                "Number of non-zero potential terms: {}",
                w_kappa_tuples.len()
            );
            if w_kappa_tuples.is_empty() {
                warn!("WARNING: No non-zero potential terms found! Check w_kappa initialization.");
                // Print some of the kappa vectors to debug
                info!("Sample kappa vectors:");
                for i in 0..std::cmp::min(10, self.kappa_vectors.len()) {
                    info!("  {}: {:?}", i, self.kappa_vectors[i]);
                }
            }
        }

        // Calculate or load matrix elements
        let v_rel = self.matrix_elem_calc.calculate_matrix_elements(
            &w_kappa_tuples,
            n_rel_states_used,
            self.a0,
        );

        // Build the full Hamiltonian matrix
        if self.verbose {
            info!("Building the full Hamiltonian matrix...");
        }

        let start_time = Instant::now();

        // Create a dense complex matrix for the Hamiltonian
        let mut h_full: M;
        let eye_com = M::eye(n_com_states);

        h_full = eye_com.kron(&M::from_diag(rel_eigvals.as_slice().unwrap())?);

        let mut h_int = TriMat::new((dim, dim));

        // Group sparse relative-coordinate sine matrix elements by transferred kappa.
        // This avoids scanning all relative-state pairs for every CoM state.
        let mut v_rel_by_kappa: HashMap<KappaKey, Vec<(usize, usize, f64)>> = HashMap::new();

        for (&(bra_rel, ket_rel, kappa_pot_tuple), &matrix_element) in &v_rel {
            v_rel_by_kappa.entry(kappa_pot_tuple).or_default().push((
                bra_rel,
                ket_rel,
                matrix_element,
            ));
        }

        for i_com in 0..n_com_states {
            for (&kappa_pot_tuple, rel_entries) in &v_rel_by_kappa {
                let Some(&j_com) = self.vector_sum_to_index.get(&(i_com, kappa_pot_tuple)) else {
                    continue;
                };

                let Some(&v0) = w_kappa_tuples.get(&kappa_pot_tuple) else {
                    continue;
                };

                for &(j_rel, i_rel, matrix_element) in rel_entries {
                    if matrix_element.abs() <= 1e-12 {
                        continue;
                    }

                    let col = exciton_flat_index(i_com, i_rel, n_rel_states_used);
                    let row = exciton_flat_index(j_com, j_rel, n_rel_states_used);

                    h_int.add_triplet(
                        row,
                        col,
                        M::Scalar::convert_from(Complex64::new(0.0, -2.0 * v0 * matrix_element)),
                    );
                }
            }
        }

        h_full = h_full.add(&M::from_trimat(&h_int)?)?;

        if self.verbose {
            info!(
                "Full Hamiltonian built in {:.2} seconds",
                start_time.elapsed().as_secs_f64()
            );
        }

        Ok((h_full, rel_eigvals, rel_eigvecs))
    }

    /// Builds the full Hamiltonian including center-of-mass kinetic energy
    ///
    /// This method constructs the complete Hamiltonian matrix that includes the relative
    /// coordinate energy, the periodic potential, and the center-of-mass kinetic energy.
    /// It adds the center-of-mass kinetic energy to the diagonal elements of the
    /// Hamiltonian matrix built by `build_hamiltonian_no_k`.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    ///
    /// # Returns
    /// * Tuple containing the complex Hamiltonian matrix, relative eigenvalues, and relative eigenvectors
    pub fn build_full_hamiltonian(
        &self,
        n_rel_states: usize,
    ) -> Result<(M, Array1<f64>, Array2<f64>), MatrixOpsError> {
        // Build the Hamiltonian without center-of-mass kinetic energy
        let (mut h_full, rel_eigvals, rel_eigvecs) = self.build_hamiltonian_no_k(n_rel_states)?;

        // Number of center-of-mass states
        let n_com_states = self.kappa_vectors.len();
        let n_rel_states_used = rel_eigvals.len();

        let eye_rel = M::eye(n_rel_states_used);

        let com_kinetic = (0..n_com_states)
            .map(|i_com| {
                let k_vec_i = &self.kappa_vectors[i_com];
                let mut k_plus_k = vec![0.0; 2];
                for j in 0..2 {
                    k_plus_k[j] = self.k[j] + k_vec_i[j];
                }
                (k_plus_k[0].powi(2) + k_plus_k[1].powi(2)) / (2.0 * self.m)
            })
            .collect::<Vec<f64>>();

        h_full = h_full.add(&M::from_diag(&com_kinetic)?.kron(&eye_rel))?;

        Ok((h_full, rel_eigvals, rel_eigvecs))
    }

    /// Diagonalizes the full Hamiltonian
    ///
    /// This method diagonalizes the full Hamiltonian matrix to find the eigenvalues
    /// and eigenvectors of the exciton in the periodic potential.
    ///
    /// The Hamiltonian is complex (with -i*sin matrix elements) but Hermitian, so
    /// eigenvalues are real and eigenvectors are complex.
    ///
    /// # Arguments
    /// * `n_rel_states` - Number of relative coordinate states to include
    /// * `k_eigvals` - Number of eigenvalues to compute
    ///
    /// # Returns
    /// * Tuple containing eigenvalues (real), eigenvectors (complex), relative eigenvalues,
    ///   relative eigenvectors, and parity values (of unperturbed states)
    pub fn diagonalize_full_hamiltonian(
        &self,
        n_rel_states: usize,
        k_eigvals: usize,
    ) -> Result<(Array1<f64>, Array2<Complex64>, Array1<f64>, Array2<f64>), MatrixOpsError>
    where
        M::Scalar: ConvertFrom<Complex64>,
        M: Diagonalize,
    {
        if self.verbose {
            info!(
                "Building complex Hermitian Hamiltonian with -2i*w_kappa*sin(kappa·x/2) matrix elements"
            );
        }

        // Build the full Hamiltonian with all states
        let (h_full, rel_eigvals, rel_eigvecs) = self.build_full_hamiltonian(n_rel_states)?;

        if self.verbose {
            info!("Computing {} lowest eigenvalues...", k_eigvals);
        }

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, right_eigenvectors) = h_full
            .diagonalize(DiagonalizationOptions::smallest_real(k_eigvals))?
            .into_arrays_hermitian();
        Ok((eigenvalues, right_eigenvectors, rel_eigvals, rel_eigvecs))
    }
}
