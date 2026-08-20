use crate::types::Vector3D;
use ndarray::{Array1, array};

const C: f64 = 137.036;

#[derive(Debug, Clone)]
pub struct BosonMode {
    pub eps: Array1<f64>,
    pub q: Array1<f64>,
    pub omega: f64,
    pub n_fock: usize,
}

impl BosonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: usize) -> Self {
        let omega = q.magnitude() * C;
        let normal = Vector3D::new(0.0, 0.0, 1.0);
        let eps = pol.get_pol(q, normal);
        let q = array![q.x, q.y];
        let eps = array![eps.0, eps.1];

        Self {
            eps,
            q,
            omega,
            n_fock,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarization {
    TE,
    TM,
}

impl Polarization {
    pub fn get_pol(&self, q: Vector3D, normal: Vector3D) -> (f64, f64) {
        let te: Vector3D;
        let tm: Vector3D;

        if q.cross(&normal).normalize().magnitude() < 0.5 {
            te = Vector3D::new(0.0, 1.0, 0.0);
            tm = Vector3D::new(1.0, 0.0, 0.0);
        } else {
            te = q.cross(&normal).normalize();
            tm = te.cross(&q).normalize();
        }

        match self {
            Polarization::TE => (te.x, te.y),
            Polarization::TM => (tm.x, tm.y),
        }
    }
}
