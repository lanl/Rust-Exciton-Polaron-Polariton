use exciton_2d::{Exciton2DGridSystem, ExcitonConfig, utils::fast_matrix_element_calc};
use ndarray_linalg::generate;
use std::{path::PathBuf, usize};
use ndarray::{arr1, array, linalg::kron, Array1, Array2};

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
    pub eps: Array1<f64>, // Must be [x, y]
    pub q: Array1<f64>, // Must be [x, y]
    pub omega: f64,
    pub n_fock: usize
}

impl PhotonMode {
    pub fn new(q: Vector3D, pol: Polarization, n_fock: usize) -> Self{
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

impl PhotonicHamiltonian {
    pub fn new(photon_modes: Vec<PhotonMode>) -> Self {
        Self { photon_modes: photon_modes }
    }

    fn get_h_ph_sing(&self, omega:f64, n_fock: usize) -> Array2<f64> {
        let n_max = (n_fock - 1) as f64;
        let diags = Array1::linspace(0.0, omega * n_max, n_fock);

        return Array2::from_diag(&diags);
    }

    pub fn get_h_ph(&self, photon_modes: Vec<PhotonMode>) -> Array2<f64> {

        // Handle empty vector case
        if photon_modes.is_empty() {
            // Return an appropriate default or error handling
            // For example, a 1x1 zero matrix:
            return Array2::ones((1, 1));
        }

        photon_modes.iter().map(|mode| self.get_h_ph_sing(mode.omega, mode.n_fock))
            .reduce(|acc, h_ph_sing| kron(&acc, &h_ph_sing))
            .unwrap()
    }
}

pub struct IntHamiltonian {
    pub exciton: Exciton2DGridSystem,
    pub photon_modes: Vec<PhotonMode>,
    pub coupling: f64,
    pub exciton_vecs: Vec<Array1<f64>>,
    pub rel_vecs: Array2<f64>
}

impl IntHamiltonian {
    pub fn new (exciton: Exciton2DGridSystem, 
                modes:  Vec<PhotonMode>, 
                coupling: f64, 
                e_vecs: Vec<Array1<f64>>,
                r_vecs: Array2<f64>) -> Self {
        Self {  exciton: exciton, 
                photon_modes: modes, 
                coupling: coupling,
                exciton_vecs: e_vecs,
                rel_vecs: r_vecs}
    }

    pub fn get_int_h(&self, kappas: &Vec<Vec<f64>>, k: &Vec<f64>) {
        let n_ph = self.photon_modes[0].n_fock.pow(self.photon_modes.len() as u32);
        let n_ex = self.exciton_vecs.len();
        let mut h_int = Array2::<f64>::zeros((n_ex * n_ph, n_ex * n_ph));
        self.photon_modes.iter().enumerate().for_each(|(mode_i,mode)|{

            let m_com = self.exciton.m;

            // P a^+ a
            let p_dot_q = self.get_p_com(kappas, k, &mode.q);
            let aa_dag = self.get_h_ph_sing_full(mode_i);
            h_int = &h_int + kron(&(p_dot_q / m_com),&aa_dag);

            // P cos() (a^dag + a)
            let p_dot_eps = self.get_p_com(kappas, k, &mode.eps);
            let p_cos =self.coupling * &p_dot_eps;
            let a = self.get_a_sing_full(mode_i);
            h_int = &h_int + kron(&p_cos, &(&a + &a.t()))
        });
        
    }

    fn get_a_sing_full(&self, mode_i: usize)  -> Array2<f64> {
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                let dim = mode.n_fock;
                let mut a = Array2::<f64>::zeros((dim, dim));
                for n in 1..dim {
                    a[[n-1,n]] = (n as f64).sqrt();
                }

                return a
            }
            else {
                let diags = Array1::<f64>::ones(mode.n_fock);
                return Array2::from_diag(&diags)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    fn get_h_ph_sing_full(&self, mode_i: usize) -> Array2<f64>{
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                let n_max = (mode.n_fock - 1) as f64;
                let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                return Array2::from_diag(&diags)
            }
            else {
                let diags = Array1::<f64>::ones(mode.n_fock);
                return Array2::from_diag(&diags)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    pub fn get_cos_x(&self, mode_i: usize) {
        let n_com = self.exciton.n_kappa;
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();

        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];
            
            // Compute p_ij using iterator operations
            let cos_ij: f64 = vec1.iter().enumerate()
                .flat_map(|(i1, &a1)| {
                    let rel_ind1 = i1 % n_com;
                    vec2.iter().enumerate()
                        .map(move |(i2, &a2)| {
                            let rel_ind2 = i2 % n_com;
                            a1 * a2 // PLACEHOLDER!!!
                        })
                })
                .sum();

            cos_ij
        });
    }

    fn get_cos_elements(&self, mode_i: usize, i1: usize, i2: usize) -> f64{
        let x_grid = &self.exciton.grid.x;
        let y_grid = &self.exciton.grid.y;
        let eigvecs = &self.rel_vecs;
        let grid_size = self.exciton.grid_size;

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut cos_x = Array2::zeros((grid_size, 1));
        let mut cos_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            cos_x[[i, 0]] = (x_grid[i]).cos();
            cos_y[[0, i]] = (y_grid[i]).cos();
        }
        
        let cos_term = cos_x + cos_y;
        
        fast_matrix_element_calc(
            &psi_i, 
            &psi_j, 
            &cos_term, self.exciton.grid.dx
        )
    }

    pub fn get_p_com(&self, kappas: &Vec<Vec<f64>>, k: &Vec<f64>, field_vec: &Array1<f64>) -> Array2<f64> {
        let e_vecs = &self.exciton_vecs;
        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        
        // Convert k to Array1 once
        let k_arr = arr1(&k);
        
        // Precompute p_dot_unit for each component
        let p_dot_unit_cache: Vec<f64> = kappas.iter()
            .map(|kappa| (k_arr.clone() + arr1(kappa)).dot(field_vec))
            .collect();
        
        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];
            
            // Compute p_ij using iterator operations
            vec1.iter().enumerate()
                .flat_map(|(i1, &a1)| {
                    let com_ind1 = i1 / n_com;
                    let p = p_dot_unit_cache[com_ind1];
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 / n_com;
                            if com_ind1 == com_ind2 {
                                Some(a1 * a2 * p)
                            } else {
                                None
                            }
                        })
                })
                .sum()
        })
    }
}

pub struct LightMatter {
    pub exciton: Exciton2DGridSystem,
    pub photon_modes: Vec<PhotonMode>,
    pub coupling: f64,
    pub h_ph: PhotonicHamiltonian,
    pub h_int: IntHamiltonian
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
