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
