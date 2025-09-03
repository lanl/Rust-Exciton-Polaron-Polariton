use exciton_2d::{utils::fast_matrix_element_calc, Exciton2DGridSystem, ExcitonConfig};
use ndarray_linalg::c64;
use std::{collections::HashMap, f64::consts::PI, path::PathBuf, usize};
use ndarray::{arr1, array, linalg::kron, Array1, Array2};
use std::time::Instant;

const C: f64 = 137.036;

pub fn to_c_2(arr:&Array2<f64>) -> Array2<c64>{

    let new_arr = arr.map(|x| c64::from(x));

    new_arr    
}

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

#[derive(Debug, Clone)]
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

pub struct LMMatrixElement {
    pub   cos_mat: HashMap<(usize,usize, Vec<i32>), f64>,
    pub p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64>,
}

impl LMMatrixElement {

    pub fn populate_mat_elements(&self, rel_vecs: &Array2<f64>, modes: &Vec<PhotonMode>, exciton: &Exciton2DGridSystem) -> Self{
        let n_rel = rel_vecs.shape()[1];
        let sing_q_ind: Vec<Vec<i32>> = (0 .. modes.len()).map(|ind| vec![ind as i32]).collect();

        let doub_q_ind: Vec<Vec<i32>> = (0..modes.len()).flat_map(|ind1| {
            (0..modes.len()).map(|ind2| vec![ind1.clone() as i32, ind2 as i32]).collect::<Vec<Vec<i32>>>()
        }).collect();

        let all_q_ind = [sing_q_ind,doub_q_ind].concat();

        let mut  cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let mut p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();


        let get_cos = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 { 
            self.get_cos_elements(exciton, rel_vecs, i1, i2, q)
        };

        let get_p_sin = |i1: usize, i2: usize, q: &Array1<f64>| -> f64 {
            self.get_psin_elements(exciton, rel_vecs, i1, i2, q)
        };

        all_q_ind.iter().for_each(|q_inds| 
            for i1 in 0 .. n_rel {
                for i2 in 0 .. n_rel {
                    if q_inds.len() == 1 {
                        let q = &modes[q_inds[0] as usize].q;

                        cos_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_cos(i1, i2, q));

                        p_sin_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_p_sin(i1, i2, q));
                        
                    }

                    else if q_inds.len() == 2 {
                        let q_ind1 = q_inds[0].max(q_inds[1]);
                        let q_ind2 = q_inds[0].min(q_inds[1]);
                        let q_1 = &modes[q_ind1 as usize].q;
                        let q_2 = &modes[q_ind2 as usize].q;
                        let q_diff = q_1 - q_2;
                        let q_sum = q_1 + q_2;

                        cos_mat.entry((i1,i2,vec![ q_ind1, -q_ind2]))
                            .or_insert_with(|| get_cos(i1, i2, &q_diff));
                        // cos_mat.entry((i1,i2,vec![-q_ind1,  q_ind2]))
                        //     .or_insert_with(|| get_cos(i1, i2, &q_diff));

                        cos_mat.entry((i1,i2,q_inds.clone()))
                            .or_insert_with(|| get_cos(i1, i2, &q_sum));
                    }
                }
            }
        );

        Self{cos_mat: cos_mat, p_sin_mat: p_sin_mat}
    }

    pub fn new() -> Self{
        let   cos_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();
        let p_sin_mat: HashMap<(usize,usize, Vec<i32>), f64> = HashMap::new();

        Self { cos_mat: cos_mat, p_sin_mat: p_sin_mat }
    }

    fn get_cos_elements(&self, exciton: &Exciton2DGridSystem,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        
        let x_grid = &exciton.grid.x;
        let y_grid = &exciton.grid.y;
        let eigvecs = &rel_vecs;
        let _test = eigvecs.shape();
        let grid_size = exciton.grid_size;

        // Get the eigenvectors for states i and j
        let psi_i: Array2<f64> = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let cos_term = (q_x + q_y).cos();
        
        fast_matrix_element_calc(
            &psi_i, 
            &psi_j, 
            &cos_term, exciton.grid.dx
        )
    }

    fn get_psin_elements(&self, exciton: &Exciton2DGridSystem,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        let x_grid =  &exciton.grid.x;
        let y_grid =  &exciton.grid.y;
        let eigvecs = rel_vecs;
        let grid_size = exciton.grid_size;
        let dx = exciton.grid.dx;

        // Get the eigenvectors for states i and j
        let psi_i = eigvecs.column(i1)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        let psi_j = eigvecs.column(i2)
            .to_shape((grid_size, grid_size)).unwrap().to_owned();
        
        // Calculate the cosine term
        let mut q_x = Array2::zeros((grid_size, 1));
        let mut q_y = Array2::zeros((1, grid_size));
        
        for i in 0..grid_size {
            q_x[[i, 0]] = x_grid[i] * q[0] / 2.0;
            q_y[[0, i]] = y_grid[i] * q[1] / 2.0;
        }
        
        let sin_term = (q_x + q_y).sin();
        let p = self.compute_sinc_dvr_rel(exciton);
        let p_sin = &p.dot(&sin_term) + &sin_term.dot(&p);
        
        (psi_i * p_sin * psi_j).sum() * dx * dx

    }

    fn compute_sinc_dvr_rel(&self, exciton: &Exciton2DGridSystem) -> Array2<f64> {
        let dx = exciton.grid.dx;
        let nx = exciton.grid_size;

        Array2::from_shape_fn((nx,nx), |(i, j)| {
            if i == j {
                PI.powi(2) / 3.0 / dx*dx
            }
            else {
                let diff: f64 = i as f64 - j as f64;
                (-1.0 as f64).powf(diff) / (diff + f64::EPSILON) / dx 
            }
        })
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

    pub fn get_h_ph(&self) -> Array2<f64> {

        let photon_modes = &self.photon_modes;

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

    pub fn get_int_h(&self) -> Array2<f64> {
        let matrix_elements = LMMatrixElement::new()
            .populate_mat_elements(&self.rel_vecs, &self.photon_modes, &self.exciton);

        let kappas = &self.exciton.kappa_vectors;
        let k = &self.exciton.k;

        let n_ph = self.photon_modes[0].n_fock.pow(self.photon_modes.len() as u32);
        let n_ex = self.exciton_vecs.len();
        let mut h_int = Array2::<f64>::zeros((n_ex * n_ph, n_ex * n_ph));
        self.photon_modes.iter().enumerate().for_each(|(mode_i,mode1)|{

            let m_com = self.exciton.m;
            let q = &mode1.q;

            // P • q a^+ a
            let p_dot_q = self.get_p_com(kappas, k, q);
            let aa_dag1 = self.get_h_ph_sing_full(mode_i);
            h_int = &h_int + &kron(&(p_dot_q / m_com),&aa_dag1);

            // P • A cos() (a^dag + a)
            let p_dot_eps = self.get_p_com(kappas, k, &mode1.eps);
            let cos_x = &matrix_elements.cos_mat;

            let q_ind = vec![mode_i as i32];

            let p_cos = 2.0 * self.coupling / m_com * self.get_cos_x(&q_ind, &cos_x) * &p_dot_eps;
            let a1 = self.get_a_sing_full(mode_i);
            h_int = &h_int + &kron(&p_cos, &(&a1 + &a1.t()));

            // -d/dx • A sin() (a^dag - a)
            let p_sin_x = &matrix_elements.p_sin_mat;
            let p_sin = self.get_p_sin_x(&q_ind, &p_sin_x) * self.coupling * 2.0 / self.exciton.mu;
            h_int = &h_int + &kron(&p_sin, &(&a1.t() - &a1));

            self.photon_modes.iter().enumerate().for_each(|(mode_j,mode2)| {
                let q_2 = &mode2.q;
                let qq = q.dot(q_2);

                // q_i a_i^+ a_i q_j a_j^+ a_j
                let ex_eye: Array2<f64> = Array2::eye(n_ex);
                let aa_2 = &aa_dag1.dot(&self.get_h_ph_sing_full(mode_j)) * qq / 2.0 / m_com;
                h_int = &h_int + &kron(&ex_eye, &aa_2);

                // Diamagnetic term
                let a2 = self.get_a_sing_full(mode_j);
                let q_ind1 = mode_i.max(mode_j) as i32;
                let q_ind2 = mode_i.min(mode_j) as i32;
                
                // Sum
                let cos_sum = self.get_cos_x(&vec![q_ind1, q_ind2], &cos_x) * self.coupling.powi(2);
                h_int = &h_int + kron(&cos_sum, &(&a1.dot(&a2) + &a1.t().dot(&a2.t())));

                // Diff
                let cos_diff = self.get_cos_x(&vec![q_ind1, -q_ind2], &cos_x) * self.coupling.powi(2);
                h_int = &h_int + kron(&cos_diff, &(&a1.t().dot(&a2) + &a1.dot(&a2.t())));

            });
        });
        
        h_int
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


    pub fn get_cos_x(&self, q_inds: &Vec<i32>, cos_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
        let n_com = self.exciton.kappa_vectors.len();
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
                    let com_ind1 = i1 % n_com;
                    let rel_ind1 = i1 / n_com;
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 % n_com;
                            let rel_ind2 = i2 / n_com;
                            if com_ind1 == com_ind2 {
                                Some(a1 * a2 * cos_mat.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                    .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                    i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds)))
                            } else {
                                None
                            }
                        })
                })
                .sum();

            cos_ij
        })
    }

    fn get_p_sin_x(&self, q_inds: &Vec<i32>, p_sin_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();

        // Build matrix using functional operations
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];
            
            // Compute p_ij using iterator operations
            let sin_ij: f64 = vec1.iter().enumerate()
                .flat_map(|(i1, &a1)| {
                    let com_ind1 = i1 % n_com;
                    let rel_ind1 = i1 / n_com;
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 % n_com;
                            let rel_ind2 = i2 / n_com;
                            if com_ind1 == com_ind2 {
                                Some(a1 * a2 * p_sin_mat.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                    .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                    i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds)))
                            } else {
                                None
                            }
                        })
                })
                .sum();

            sin_ij
        })
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
                    let com_ind1 = i1 % n_com;
                    let p = p_dot_unit_cache[com_ind1];
                    vec2.iter().enumerate()
                        .filter_map(move |(i2, &a2)| {
                            let com_ind2 = i2 % n_com;
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

    fn _get_aa_2(&self, mode_i: usize, mode_j: usize, mode1: &PhotonMode, mode2: &PhotonMode) -> Array2<f64>{
        let q2 = mode1.q.dot(&mode2.q);

        if mode_i == mode_j {
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i {
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                    return Array2::from_diag(&(&diags * &diags))
                }
                else {
                    let diags = Array1::<f64>::ones(mode.n_fock);
                    return Array2::from_diag(&diags)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
        else {
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i || idx == mode_j{
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                    return Array2::from_diag(&diags)
                }
                else {
                    let diags = Array1::<f64>::ones(mode.n_fock);
                    return Array2::from_diag(&diags)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
    }
}

pub struct LightMatter {
    pub h_ph: PhotonicHamiltonian,
    pub h_int: IntHamiltonian
}

impl LightMatter {
    pub fn new( exciton: Exciton2DGridSystem, 
                photon_modes: Vec<PhotonMode>, 
                coupling: f64,
                e_vecs: Vec<Array1<f64>>,
                r_vecs: Array2<f64>) -> Self {
        let h_ph = PhotonicHamiltonian::new(photon_modes.clone());
        let h_int = IntHamiltonian::new(exciton, photon_modes.clone(), coupling, e_vecs, r_vecs);
        
        Self { h_ph: h_ph, h_int: h_int }
    }
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
    let exciton = Exciton2DGridSystem::from_config(e_config);
    
    // Solve for the relative coordinate eigenstates
    let n_rel = 16;
    
    let (sorted_eigvals, sorted_eigvecs, _, rel_eigvecs, _) = exciton.diagonalize_full_hamiltonian(n_rel, 10, 1);

    let start_time = Instant::now();
    let h_ex = Array2::from_diag(&sorted_eigvals);
    println!("Exciton Hamiltonian of size: {:?}", h_ex.shape());
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Built Exciton Hamiltonian in {:?}", duration);
    

    let start_time = Instant::now();
    let coupling = 0.01;
    let omega_0 = 0.1;
    let qx_values = vec![-2e-4, 0.0, 2e-4];
    let qz = omega_0 / C;
    let n_fock = 2;
    let photon_modes: Vec<PhotonMode> = qx_values.iter().map(|qx| {
            PhotonMode::new(Vector3D { x: *qx, y: 0.0, z: qz }, Polarization::TE, n_fock)
        }).collect();
    
    let light_matter = LightMatter::new(exciton, photon_modes, coupling, sorted_eigvecs, rel_eigvecs);
    
    let h_ph = light_matter.h_ph.get_h_ph();
    println!("Photonic Hamiltonian of size: {:?}", h_ph.shape());
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Built Photonic Hamiltonian in {:?}", duration);

    let start_time = Instant::now();
    let h_int = light_matter.h_int.get_int_h();
    println!("Interaction Hamiltonian of size: {:?}", h_int.shape());
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Built Interaction Hamiltonian in {:?}", duration);

}
