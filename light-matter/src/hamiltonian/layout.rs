use matrix_ops::{KronLayout, MatrixOps, MatrixOpsError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LMBasisLayout {
    pub ex_ph_layout: KronLayout,
}

impl LMBasisLayout {
    /// Default desired LM basis:
    ///
    /// ```text
    /// |LM> = |photon> ⊗ |exciton>
    /// ```
    ///
    /// Code passes `(block=exciton, transition=photon)`.
    pub const PHOTON_MAJOR: Self = Self {
        ex_ph_layout: KronLayout::TransitionSlowBlockFast,
    };

    /// Legacy LM basis:
    ///
    /// ```text
    /// |LM> = |exciton> ⊗ |photon>
    /// ```
    pub const EXCITON_MAJOR: Self = Self {
        ex_ph_layout: KronLayout::BlockSlowTransitionFast,
    };

    #[inline]
    pub fn lm_index(
        self,
        ex: usize,
        ph: usize,
        n_ex: usize,
        n_ph: usize,
    ) -> matrix_ops::Result<usize> {
        self.ex_ph_layout
            .try_flat_index(ex, ph, n_ex, n_ph, "light-matter LM basis index")
    }

    #[inline]
    pub fn lm_unflatten(
        self,
        idx: usize,
        n_ex: usize,
        n_ph: usize,
    ) -> matrix_ops::Result<(usize, usize)> {
        self.ex_ph_layout
            .try_unflatten(idx, n_ex, n_ph, "light-matter LM basis unflatten")
    }

    #[inline]
    pub fn lm_dim(self, n_ex: usize, n_ph: usize) -> matrix_ops::Result<usize> {
        self.ex_ph_layout
            .total_dim(n_ex, n_ph)
            .ok_or_else(|| MatrixOpsError::DimensionOverflow {
                details: format!("LM dimension overflow: {n_ex} * {n_ph}"),
            })
    }

    /// Builds an LM-space Kronecker product from an exciton operator and a
    /// photon operator according to this LM layout.
    ///
    /// Production code should use this helper instead of manually writing
    /// `ex_op.kron(ph_op)` or `ph_op.kron(ex_op)`.
    #[inline]
    pub fn kron_ex_ph<M: MatrixOps>(self, ex_op: &M, ph_op: &M) -> M {
        match self.ex_ph_layout {
            KronLayout::BlockSlowTransitionFast => ex_op.kron(ph_op),
            KronLayout::TransitionSlowBlockFast => ph_op.kron(ex_op),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PPBasisLayout {
    pub lm: LMBasisLayout,

    /// Layout of LM block with phonon factor.
    ///
    /// Default PP basis:
    ///
    /// ```text
    /// |PP> = |LM> ⊗ |phonon>
    /// ```
    pub lm_pn_layout: KronLayout,
}

impl PPBasisLayout {
    pub const DEFAULT: Self = Self {
        lm: LMBasisLayout::PHOTON_MAJOR,
        lm_pn_layout: KronLayout::BlockSlowTransitionFast,
    };

    #[inline]
    pub fn pp_index(
        self,
        ex: usize,
        ph: usize,
        pn: usize,
        n_ex: usize,
        n_ph: usize,
        n_pn: usize,
    ) -> matrix_ops::Result<usize> {
        let lm_idx = self.lm.lm_index(ex, ph, n_ex, n_ph)?;
        let lm_dim = self.lm.lm_dim(n_ex, n_ph)?;

        self.lm_pn_layout
            .try_flat_index(lm_idx, pn, lm_dim, n_pn, "polaron-polariton basis index")
    }

    #[inline]
    pub fn pp_unflatten(
        self,
        idx: usize,
        n_ex: usize,
        n_ph: usize,
        n_pn: usize,
    ) -> matrix_ops::Result<(usize, usize, usize)> {
        let lm_dim = self.lm.lm_dim(n_ex, n_ph)?;

        let (lm_idx, pn) = self.lm_pn_layout.try_unflatten(
            idx,
            lm_dim,
            n_pn,
            "polaron-polariton basis unflatten",
        )?;

        let (ex, ph) = self.lm.lm_unflatten(lm_idx, n_ex, n_ph)?;

        Ok((ex, ph, pn))
    }

    #[inline]
    pub fn pp_dim(self, n_ex: usize, n_ph: usize, n_pn: usize) -> matrix_ops::Result<usize> {
        let lm_dim = self.lm.lm_dim(n_ex, n_ph)?;

        self.lm_pn_layout
            .total_dim(lm_dim, n_pn)
            .ok_or_else(|| MatrixOpsError::DimensionOverflow {
                details: format!("PP dimension overflow: {lm_dim} * {n_pn}"),
            })
    }

    /// Builds a PP-space Kronecker product from an LM operator and a phonon
    /// operator according to the PP outer layout.
    #[inline]
    pub fn kron_lm_pn<M: MatrixOps>(self, lm_op: &M, pn_op: &M) -> M {
        match self.lm_pn_layout {
            KronLayout::BlockSlowTransitionFast => lm_op.kron(pn_op),
            KronLayout::TransitionSlowBlockFast => pn_op.kron(lm_op),
        }
    }
}

impl PPBasisLayout {
    pub const EX_PN_LAYOUT: KronLayout = KronLayout::BlockSlowTransitionFast;

    /// For the default PP layout,
    ///
    /// ```text
    /// |PP> = |LM> ⊗ |phonon>
    ///      = (|photon> ⊗ |exciton>) ⊗ |phonon>
    ///      = |photon> ⊗ (|exciton> ⊗ |phonon>)
    /// ```
    ///
    /// so regrouping as photon transitions with exciton-phonon blocks uses:
    ///
    /// ```text
    /// idx(block=EXPN, transition=photon) = photon * expn_dim + expn
    /// ```
    #[inline]
    pub fn ph_expn_layout(self) -> matrix_ops::Result<KronLayout> {
        if self.lm.ex_ph_layout == KronLayout::TransitionSlowBlockFast
            && self.lm_pn_layout == KronLayout::BlockSlowTransitionFast
        {
            Ok(KronLayout::TransitionSlowBlockFast)
        } else {
            Err(MatrixOpsError::Other {
                message: format!(
                    "Current PPBasisLayout {:?} cannot be regrouped as |photon> ⊗ |exciton, phonon>",
                    self
                ),
            })
        }
    }

    #[inline]
    pub fn expn_index(
        self,
        ex: usize,
        pn: usize,
        n_ex: usize,
        n_pn: usize,
    ) -> matrix_ops::Result<usize> {
        Self::EX_PN_LAYOUT.try_flat_index(
            ex,
            pn,
            n_ex,
            n_pn,
            "exciton-phonon block index",
        )
    }

    #[inline]
    pub fn expn_dim(self, n_ex: usize, n_pn: usize) -> matrix_ops::Result<usize> {
        Self::EX_PN_LAYOUT
            .total_dim(n_ex, n_pn)
            .ok_or_else(|| MatrixOpsError::DimensionOverflow {
                details: format!("EXPN dimension overflow: {n_ex} * {n_pn}"),
            })
    }

    /// Builds an exciton-phonon block operator:
    ///
    /// ```text
    /// |EXPN> = |exciton> ⊗ |phonon>
    /// ```
    #[inline]
    pub fn kron_ex_pn<M: MatrixOps>(self, ex_op: &M, pn_op: &M) -> M {
        match Self::EX_PN_LAYOUT {
            KronLayout::BlockSlowTransitionFast => ex_op.kron(pn_op),
            KronLayout::TransitionSlowBlockFast => pn_op.kron(ex_op),
        }
    }

    /// Builds a PP-space operator from a photon operator and an EXPN block
    /// operator according to the regrouped photon-transition layout.
    ///
    /// For the default layout this is:
    ///
    /// ```text
    /// ph_op ⊗ expn_op
    /// ```
    #[inline]
    pub fn kron_ph_expn<M: MatrixOps>(
        self,
        ph_op: &M,
        expn_op: &M,
    ) -> matrix_ops::Result<M> {
        match self.ph_expn_layout()? {
            KronLayout::BlockSlowTransitionFast => Ok(expn_op.kron(ph_op)),
            KronLayout::TransitionSlowBlockFast => Ok(ph_op.kron(expn_op)),
        }
    }
}