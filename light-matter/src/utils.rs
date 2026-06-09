use serde::{Serialize, Deserialize};
use std::path::Path;
use bincode::{self};
use exciton_2d::ExcitonConfig;
use ndarray::{Array1, array};

// Speed of light in atomic units
const C: f64 = 137.036;

// Define a serializable structure for band structure data storage
#[derive(Serialize, Deserialize)]
struct SerializableData {
    matrix: Vec<Vec<f64>>,  // Stores k-points as a 2D matrix
    arrays: Vec<Vec<f64>>,  // Stores eigenvalues as vectors
}

/// Saves band structure data to a binary file using bincode serialization
pub fn save_band_data<P: AsRef<Path>>(
    path: P,
    data: (Vec<Vec<f64>>, Vec<Array1<f64>>)
) -> Result<(), Box<dyn std::error::Error>> {
    let (matrix, arrays) = data;
    let converted_arrays: Vec<Vec<f64>> = arrays
        .into_iter()
        .map(|arr| arr.to_vec())
        .collect();

    let serializable_data = SerializableData {
        matrix,
        arrays: converted_arrays,
    };

    let encoded: Vec<u8> = bincode::serialize(&serializable_data)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

/// Loads band structure data from a binary file
pub fn load_band_data<P: AsRef<Path>>(
    path: P
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let serializable_data: SerializableData = bincode::deserialize(&bytes)?;

    let arrays: Vec<Array1<f64>> = serializable_data.arrays
        .into_iter()
        .map(|vec| Array1::from_vec(vec))
        .collect();

    Ok((serializable_data.matrix, arrays))
}

/// Represents a 3D vector with x, y, z components
#[derive(Debug, Clone, Copy)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vector3D { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        f64::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return *self;
        }
        Vector3D {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    pub fn cross(&self, other: &Vector3D) -> Self {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn _dot(&self, other: &Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Represents a photon mode with polarization, wavevector, frequency, and Fock space dimension
#[derive(Debug, Clone)]
pub struct BosonMode {
    pub eps: Array1<f64>, // Polarization vector (x,y components)
    pub q: Array1<f64>,   // Wavevector (x,y components)
    pub omega: f64,       // Frequency
    pub n_fock: usize,    // Dimension of the Fock space
}

impl BosonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: usize) -> Self {
        let omega = q.magnitude() * C;
        let normal = Vector3D::new(0.0, 0.0, 1.0);
        let eps = pol.get_pol(q, normal);
        let q = array![q.x, q.y];
        let eps = array![eps.0, eps.1];

        Self {
            eps: eps,
            q: q,
            omega: omega,
            n_fock: n_fock,
        }
    }
}

/// Enum representing different polarization types for photons
pub enum Polarization {
    TE, // Transverse Electric
    TM, // Transverse Magnetic
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



#[derive(Clone)]
pub struct PPConfig {
    // Exciton Parameters
    pub e_config: ExcitonConfig,
    pub n_rel: usize,
    pub n_ex_eig: usize,

    // Polariton Parameters
    pub lm_couplings: Vec<f64>,
    pub omega_0: f64,
    pub q_values: Vec<Vector3D>,

    // Polaron Parameters
    pub pn_coupling: f64,
    pub pn_omega: f64,
    pub pn_k: Array1<f64>,
    pub pn_n_fock: usize,

    // Band Parameters
    pub n_k_points: usize,
    pub n_bands: usize,
    
    // Cache directory for matrix elements
    pub cache_dir: Option<String>,
    
    // Effective mode parameters (for cache filename generation)
    // These are the hand-entered parameters that define the mode structure
    pub d_omega: Option<f64>,
    pub omega_max: Option<f64>,
    pub reflectivity: Option<f64>,
}

impl PPConfig {
    pub fn new(
        // Exciton Parameters
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,

        // Polariton Parameters
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,

        // Polaron Parameters
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,

        // Band Parameters
        n_k_points: usize,
        n_bands: usize,
    ) -> Self {
        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("saved_data_lm/matrix_element_cache".to_string()),
            d_omega: None,
            omega_max: None,
            reflectivity: None,
        }
    }
    
    /// Create a new PPConfig with effective mode parameters for better cache identification
    pub fn new_with_effective_modes(
        // Exciton Parameters
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,

        // Polariton Parameters
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,

        // Polaron Parameters
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,

        // Band Parameters
        n_k_points: usize,
        n_bands: usize,
        
        // Effective mode parameters
        d_omega: f64,
        omega_max: f64,
        reflectivity: f64,
    ) -> Self {
        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("saved_data_lm/matrix_element_cache".to_string()),
            d_omega: Some(d_omega),
            omega_max: Some(omega_max),
            reflectivity: Some(reflectivity),
        }
    }
    
    /// Generate a deterministic hash from configuration parameters
    /// This hash uniquely identifies the physical system and can be used for caching
    pub fn get_config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash exciton parameters
        self.e_config.a0.to_bits().hash(&mut hasher);
        self.e_config.v0.to_bits().hash(&mut hasher);
        self.e_config.m_e.to_bits().hash(&mut hasher);
        self.e_config.m_h.to_bits().hash(&mut hasher);
        self.n_rel.hash(&mut hasher);
        self.n_ex_eig.hash(&mut hasher);
        
        // Hash polariton parameters
        for &coupling in &self.lm_couplings {
            coupling.to_bits().hash(&mut hasher);
        }
        self.omega_0.to_bits().hash(&mut hasher);
        for q in &self.q_values {
            q.x.to_bits().hash(&mut hasher);
            q.y.to_bits().hash(&mut hasher);
            q.z.to_bits().hash(&mut hasher);
        }
        
        // Hash polaron parameters
        self.pn_coupling.to_bits().hash(&mut hasher);
        self.pn_omega.to_bits().hash(&mut hasher);
        for &val in self.pn_k.iter() {
            val.to_bits().hash(&mut hasher);
        }
        self.pn_n_fock.hash(&mut hasher);
        
        hasher.finish()
    }
}

