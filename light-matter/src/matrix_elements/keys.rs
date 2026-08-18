use serde::{Deserialize, Serialize};

/// Typed keys for matrix-element maps.
///
/// This replaces the old `Vec<i32>` convention:
///
/// - `vec![i]`       -> `ModeKey::Single(i)`
/// - `vec![hi,-lo]`  -> `ModeKey::Diff { hi, lo }`
/// - `vec![hi, lo]`  -> `ModeKey::Sum { hi, lo }`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModeKey {
    Single(usize),
    Diff { hi: usize, lo: usize },
    Sum { hi: usize, lo: usize },
}

impl ModeKey {
    #[inline]
    pub fn single(mode: usize) -> Self {
        Self::Single(mode)
    }

    #[inline]
    pub fn diff(q1: usize, q2: usize) -> Self {
        let hi = q1.max(q2);
        let lo = q1.min(q2);
        Self::Diff { hi, lo }
    }

    #[inline]
    pub fn sum(q1: usize, q2: usize) -> Self {
        let hi = q1.max(q2);
        let lo = q1.min(q2);
        Self::Sum { hi, lo }
    }
}

/// Typed work items for matrix-element generation.
///
/// Currently `populate_taylor_matrix_elements_with_cache` is used for photon-sector
/// matrix elements. Phonon single-mode matrix elements are generated separately
/// by `phonon_mat_elements`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModeTask {
    PhotonSingle { mode: usize },
    PhotonPair { q1: usize, q2: usize },
}
