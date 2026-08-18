//! Reciprocal-lattice utilities.
//!
//! Integer reciprocal-lattice keys replace float-hash keys.  A reciprocal vector is
//! identified by `(n1, n2)` and evaluated as
//!
//! ```text
//! kappa = (2π/a0) * (n1, n2)
//! ```

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KappaKey {
    pub n1: i32,
    pub n2: i32,
}

impl KappaKey {
    #[inline]
    pub fn new(n1: i32, n2: i32) -> Self {
        Self { n1, n2 }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            n1: self.n1 + other.n1,
            n2: self.n2 + other.n2,
        }
    }

    #[inline]
    pub fn vector(self, a0: f64) -> [f64; 2] {
        let b = 2.0 * std::f64::consts::PI / a0;
        [self.n1 as f64 * b, self.n2 as f64 * b]
    }
}
