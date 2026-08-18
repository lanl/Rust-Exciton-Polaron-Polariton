use matrix_ops::{ConvertFrom, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2};

/// Prepared exciton Hamiltonian with all K-independent terms precomputed.
///
/// The full exciton Hamiltonian has the form
///
///     H(K) = H_0 + H_CoM(K),
///
/// where H_0 contains the relative-coordinate energies and the lattice coupling,
/// while H_CoM(K) is diagonal in the CoM plane-wave basis:
///
///     H_CoM(K) = |K + kappa|^2 / (2M).
///
/// This struct allows band-structure calculations to build H_0 once and only
/// update the diagonal CoM kinetic term for each K point.
#[derive(Clone)]
pub struct PreparedExcitonHamiltonian<M: MatrixOps> {
    pub h_no_k: M,
    pub rel_eigvals: Array1<f64>,
    pub rel_eigvecs: Array2<f64>,
    pub kappa_vectors: Vec<Vec<f64>>,
    pub n_rel: usize,
    pub n_com: usize,
    pub mass: f64,
}

impl<M: MatrixOps> PreparedExcitonHamiltonian<M>
where
    M::Scalar: ConvertFrom<f64>,
{
    /// Builds the full exciton Hamiltonian at a particular K point.
    ///
    /// The stored basis convention is
    ///
    ///     |state> = |CoM> ⊗ |rel>,
    ///
    /// so the CoM kinetic term is
    ///
    ///     diag(|K+kappa|^2 / 2M) ⊗ I_rel.
    pub fn hamiltonian_at_k(&self, k: &[f64]) -> Result<M, MatrixOpsError> {
        let eye_rel = M::eye(self.n_rel);

        let com_kinetic: Vec<f64> = self
            .kappa_vectors
            .iter()
            .map(|kappa| {
                let kx = k[0] + kappa[0];
                let ky = k[1] + kappa[1];

                (kx * kx + ky * ky) / (2.0 * self.mass)
            })
            .collect();

        self.h_no_k.add(&M::from_diag(&com_kinetic)?.kron(&eye_rel))
    }
}
