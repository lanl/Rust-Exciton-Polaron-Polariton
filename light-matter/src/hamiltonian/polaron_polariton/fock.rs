/// Converts a flat product-Fock index into occupations.
///
/// This matches the tensor-product order used by `FullBosons::a`, where
/// mode 0 is the slowest-varying index and the last mode is fastest-varying:
///
///     index = (((n0 * d1 + n1) * d2 + n2) ...)
#[inline]
pub fn occupations_from_index(mut idx: usize, dims: &[usize]) -> Vec<usize> {
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
/// This is the inverse of `occupations_from_index`.
#[inline]
pub fn index_from_occupations(occ: &[usize], dims: &[usize]) -> usize {
    occ.iter()
        .zip(dims.iter())
        .fold(0usize, |acc, (&n, &dim)| acc * dim + n)
}

#[cfg(test)]
mod tests {
    use super::{index_from_occupations, occupations_from_index};

    #[test]
    fn fock_index_round_trip() {
        let dims = vec![2, 3, 4];
        let dim: usize = dims.iter().product();

        for idx in 0..dim {
            let occ = occupations_from_index(idx, &dims);
            assert_eq!(index_from_occupations(&occ, &dims), idx);
        }
    }

    #[test]
    fn fock_order_matches_expected_mixed_radix() {
        let dims = vec![2, 3, 4];

        assert_eq!(occupations_from_index(0, &dims), vec![0, 0, 0]);
        assert_eq!(occupations_from_index(1, &dims), vec![0, 0, 1]);
        assert_eq!(occupations_from_index(4, &dims), vec![0, 1, 0]);
        assert_eq!(occupations_from_index(12, &dims), vec![1, 0, 0]);

        assert_eq!(index_from_occupations(&[1, 0, 0], &dims), 12);
        assert_eq!(index_from_occupations(&[0, 1, 0], &dims), 4);
        assert_eq!(index_from_occupations(&[0, 0, 1], &dims), 1);
    }
}
