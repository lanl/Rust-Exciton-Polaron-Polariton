use super::builder::LMMatrixElementArray;

use matrix_ops::HermitianConjugate;
use ndarray::Array2;
use ndarray_linalg::c64;
use num_traits::{One, Zero};

impl LMMatrixElementArray {
    pub fn to_exciton_basis(
        &self,
        rel_matrix: &Array2<c64>,
        e_vecs: &Array2<c64>,
        n_com: usize,
    ) -> Array2<c64> {
        use ndarray::linalg::kron;

        // exciton_2d basis convention:
        //
        //     |state> = |CoM> ⊗ |rel>
        //
        // with flat index:
        //
        //     idx = com_index * n_rel + rel_index.
        //
        // Therefore a relative-coordinate operator is:
        //
        //     I_com ⊗ O_rel.
        let com_identity = Array2::eye(n_com);
        let matrix = kron(&com_identity, rel_matrix);

        let transform = e_vecs;
        let transform_dag = transform.dag();

        transform_dag.dot(&matrix).dot(transform)
    }

    // Transform complex angular-momentum states |n,m> into the real basis:
    //   l = 0: |n,0>
    //   l > 0: (|n,l> + |n,-l>)/sqrt(2)          ~ cos(lθ)
    //   l < 0: (|n,-l> - |n,l>)/(i sqrt(2))      ~ sin(|l|θ)
    //
    // The input relative-basis matrices are first computed in the complex |n,m>
    // basis ordered as n=0: m=0; n=1: m=-1,0,1; etc. This routine applies the
    // unitary basis rotation to obtain matrices in the real basis used by the
    // exciton solver.
    pub fn to_real_rel_basis(&self, rel_matrix: &Array2<c64>) -> Array2<c64> {
        let transform =
            Array2::from_shape_fn([rel_matrix.shape()[0], rel_matrix.shape()[1]], |(i, j)| {
                let n1 = (i as f64).sqrt().floor() as i32;
                let n2 = (j as f64).sqrt().floor() as i32;

                let m1 = i as i32 - n1 * n1 - n1;
                let m2 = j as i32 - n2 * n2 - n2;

                if i == j && m1 == 0 {
                    c64::one()
                } else if m1.abs() == m2.abs() && n1 == n2 {
                    if m1 < 0 {
                        -c64::i() / 2_f64.sqrt() * m2.signum() as f64
                    } else {
                        c64::one() / 2_f64.sqrt()
                    }
                } else {
                    c64::zero()
                }
            });

        // println!("Basis transform: \n{}", PrettyC64Array2(&transform, 2));
        // println!("Unitary test: \n{}", PrettyC64Array2(&transform.dot(&transform.dag()), 2));

        transform.dot(rel_matrix).dot(&transform.dag())
    }
}
