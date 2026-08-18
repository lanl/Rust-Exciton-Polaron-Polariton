use crate::matrix_elements::ModeTask;
use crate::modes::types::BosonMode;
use matrix_ops::{ConvertFrom, MatrixOps, MatrixOpsError, Sqrt};
use num_traits::One;
use sprs::TriMat;

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
