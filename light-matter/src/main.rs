use exciton_2d::{Exciton2DGridSystem, ExcitonConfig};
use core::str;
use std::path::PathBuf;
use ndarray::{array, Array1, Array2};

const C: f64 = 137.036;

#[derive(Debug, Clone, Copy)]
pub struct Vector3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3D {
    /// Create a new 3D vector
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3D { x, y, z }
    }
    
    /// Calculate the magnitude (length) of the vector
    fn magnitude(&self) -> f64 {
        f64::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }
    
    /// Normalize the vector (make it unit length)
    fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            // Avoid division by zero
            return *self;
        }
        Vector3D {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }
    
    /// Calculate the cross product with another vector
    fn cross(&self, other: &Vector3D) -> Self {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    
    /// Calculate the dot product with another vector
    fn _dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

pub struct PhotonMode {
    pub eps: Array1<f64>,
    pub q: Array1<f64>,
    pub omega: f64,
    pub n_fock: f64
}

impl PhotonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: f64) -> Self{
        let omega = q.magnitude() * C;
        let normal = Vector3D::new(0.0, 0.0, 1.0);
        let eps = pol.get_pol(q, normal);

        let q = array![q.x, q.y];
        let eps = array![eps.0, eps.1];

        Self { eps: eps, q: q, omega: omega, n_fock: n_fock }
    }
}

pub enum Polarization {
    TE,
    TM
}

impl Polarization {
    pub fn get_pol(&self, q: Vector3D, normal: Vector3D) -> (f64,f64) {

        let te = q.cross(&normal).normalize();
        let tm = te.cross(&q).normalize();

        match self {
            Polarization::TE => (te.x, te.y),
            Polarization::TM => (tm.x, tm.y),
        }
    }
}

pub struct PhotonicHamiltonian {
    pub photon_modes: Vec<PhotonMode>,
}

pub struct LightMatter {
    pub exciton: Exciton2DGridSystem,
    pub photon_modes: Vec<PhotonMode>,
    pub coupling: f64,
    pub h_ph: PhotonicHamiltonian,
}


fn main() {

    // Create exciton config
    let e_config = ExcitonConfig {
        n_kappa: 11,
        a0: 3.0,
        v0: 0.2,
        k: vec![0.0, 0.0],
        m_e: 1.0,
        m_h: 1.0,
        verbose: false,
        grid_size: 128,
        box_size: 3.0,
        save_dir: PathBuf::from("saved_data_lm"),
        recalculate: false,
        n_jobs: 2,
        use_gpu: false,
        batch_size: 32,
    };

    // Create the 2D exciton system
    let system = Exciton2DGridSystem::from_config(e_config);
    
    // Solve for the relative coordinate eigenstates
    let n_rel = 16;



}
