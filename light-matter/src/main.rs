use exciton_2d::{utils::fast_matrix_element_calc, Exciton2DGridSystem, ExcitonConfig, FullHamiltonian};
use ndarray_linalg::{c64, Eigh};
use std::{collections::HashMap, f64::consts::PI, path::PathBuf, usize};
use ndarray::{arr1, array, linalg::kron, Array1, Array2};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::path::Path;
use bincode;
use indicatif::{ProgressBar, ProgressStyle};

use log::info;
use plotters::prelude::*;

// Define a serializable structure for our data
#[derive(Serialize, Deserialize)]
struct SerializableData {
    matrix: Vec<Vec<f64>>,
    arrays: Vec<Vec<f64>>,
}

pub fn save_band_data<P: AsRef<Path>>(
    path: P,
    data: (Vec<Vec<f64>>, Vec<Array1<f64>>)
) -> Result<(), Box<dyn std::error::Error>> {
    // Unpack the tuple
    let (matrix, arrays) = data;
    
    // Convert Vec<Array1<f64>> to Vec<Vec<f64>>
    let converted_arrays: Vec<Vec<f64>> = arrays
        .into_iter()
        .map(|arr| arr.to_vec())
        .collect();
    
    // Create the serializable structure
    let serializable_data = SerializableData {
        matrix,
        arrays: converted_arrays,
    };
    
    // Serialize to bytes
    let encoded: Vec<u8> = bincode::serialize(&serializable_data)?;
    
    // Write bytes to file
    std::fs::write(path, encoded)?;
    
    Ok(())
}

pub fn load_band_data<P: AsRef<Path>>(
    path: P
) -> Result<(Vec<Vec<f64>>, Vec<Array1<f64>>), Box<dyn std::error::Error>> {
    // Read file to bytes
    let bytes = std::fs::read(path)?;
    
    // Deserialize from bytes
    let serializable_data: SerializableData = bincode::deserialize(&bytes)?;
    
    // Convert Vec<Vec<f64>> back to Vec<Array1<f64>>
    let arrays: Vec<Array1<f64>> = serializable_data.arrays
        .into_iter()
        .map(|vec| Array1::from_vec(vec))
        .collect();
    
    Ok((serializable_data.matrix, arrays))
}


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

    pub fn populate_mat_elements(&self, rel_vecs: &Array2<f64>, modes: &Vec<PhotonMode>, exciton: &FullHamiltonian) -> Self{
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

    fn get_cos_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        
        let x_grid = &exciton.rel_ham.grid.x;
        let y_grid = &exciton.rel_ham.grid.y;
        let eigvecs = &rel_vecs;
        let _test = eigvecs.shape();
        let grid_size = exciton.rel_ham.grid.grid_size;

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
            &cos_term, exciton.rel_ham.grid.dx
        )
    }

    fn get_psin_elements(&self, exciton: &FullHamiltonian,
        rel_vecs: &Array2<f64>, i1: usize, i2: usize, q: &Array1<f64>) -> f64{
        let x_grid =  &exciton.rel_ham.grid.x;
        let y_grid =  &exciton.rel_ham.grid.y;
        let eigvecs = rel_vecs;
        let grid_size = exciton.rel_ham.grid.grid_size;
        let dx = exciton.rel_ham.grid.dx;

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

    fn compute_sinc_dvr_rel(&self, exciton: &FullHamiltonian) -> Array2<f64> {
        let dx = exciton.rel_ham.grid.dx;
        let nx = exciton.rel_ham.grid.grid_size;

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
    pub exciton: FullHamiltonian,
    pub photon_modes: Vec<PhotonMode>,
    pub coupling: f64,
    pub exciton_vecs: Vec<Array1<f64>>,
    pub rel_vecs: Array2<f64>
}

impl IntHamiltonian {
    pub fn new (exciton: FullHamiltonian, 
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

            // // P • A cos() (a^dag + a)
            let p_dot_eps = self.get_p_com(kappas, k, &mode1.eps);
            let cos_x = &matrix_elements.cos_mat;

            let q_ind = vec![mode_i as i32];

            let p_cos = 2.0 * self.coupling / m_com * self.get_matrix_from_hash(&q_ind, &cos_x).dot(&p_dot_eps);
            let a1 = self.get_a_sing_full(mode_i);
            h_int = &h_int + &kron(&p_cos, &(&a1 + &a1.t()));

            // -d/dx • A sin() (a^dag - a)
            let p_sin_x = &matrix_elements.p_sin_mat;
            let p_sin = self.get_matrix_from_hash(&q_ind, &p_sin_x) * self.coupling * 2.0 / self.exciton.rel_ham.mu;
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
                let cos_sum = self.get_matrix_from_hash(&vec![q_ind1, q_ind2], &cos_x) * self.coupling.powi(2);
                h_int = &h_int + kron(&cos_sum, &(&a1.dot(&a2) + &a1.t().dot(&a2.t())));

                // Diff
                let cos_diff = self.get_matrix_from_hash(&vec![q_ind1, -q_ind2], &cos_x) * self.coupling.powi(2);
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

                a
            }
            else {
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }

    fn get_h_ph_sing_full(&self, mode_i: usize) -> Array2<f64>{
        self.photon_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                let n_max = (mode.n_fock - 1) as f64;
                let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                Array2::from_diag(&diags)
            }
            else {
                Array2::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
        .unwrap()
    }


    pub fn get_matrix_from_hash(&self, q_inds: &Vec<i32>, mat_hash: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64> {
        let n_com = self.exciton.kappa_vectors.len();
        let e_vecs = &self.exciton_vecs;
        let n_eig = e_vecs.len();

        let n_rel = e_vecs[0].len() / n_com;

        // Build matrix using functional operations for relative coordinate operators
        // Note that these are all diagonal in the CoM picture
        Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
            // For each pair of eigenvectors
            let vec1 = &e_vecs[idx1];
            let vec2 = &e_vecs[idx2];

            let mat_ij: f64 = (0..n_rel).flat_map(|rel_ind1| {
                    (0 .. n_rel).flat_map(move |rel_ind2| {
                        let mat_element = mat_hash.get(&(rel_ind1,rel_ind2,q_inds.clone()))
                                    .unwrap_or_else(|| panic!("Matrix element missing for i1 = {}, 
                                    i2 = {}, q_ind = {:?}",rel_ind1,rel_ind2, q_inds));
                        (0..n_com).map(move |com_ind| {
                            let i1 = n_com * rel_ind1 + com_ind;
                            let i2 = n_com * rel_ind2 + com_ind;

                            let a1 = vec1[i1];
                            let a2 = vec2[i2];

                            a1 * a2 * mat_element
                        })
                    })
                }).sum();

            mat_ij
        })
    }

    fn _get_cos_x(&self, q_inds: &Vec<i32>, cos_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
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

    fn _get_p_sin_x(&self, q_inds: &Vec<i32>, p_sin_mat: &HashMap<(usize,usize, Vec<i32>), f64>) -> Array2<f64>{
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

        let n_rel = e_vecs[0].len() / n_com;
        
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

            // Compute P_ij using iterator operations
            // This is a purely diagonal operator
            (0 .. n_com).flat_map(|com_ind| {

                let p = p_dot_unit_cache[com_ind];
                
                (0 .. n_rel).map(move |rel_ind| {

                    let i = n_com * rel_ind + com_ind;

                    let a1 = vec1[i];
                    let a2 = vec2[i];

                    a1 * a2 * p
                })
            }).sum()
        })
    }

    fn _get_aa_2(&self, mode_i: usize, mode_j: usize, mode1: &PhotonMode, mode2: &PhotonMode) -> Array2<f64>{
        let q2 = mode1.q.dot(&mode2.q);

        if mode_i == mode_j {
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i {
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                    Array2::from_diag(&(&diags * &diags))
                }
                else {
                    Array2::eye(mode.n_fock)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
        else {
            self.photon_modes.iter().enumerate().map(|(idx,mode)| {
                if idx == mode_i || idx == mode_j{
                    let n_max = (mode.n_fock - 1) as f64;
                    let diags = Array1::linspace(0.0, mode.omega * n_max, mode.n_fock);

                    Array2::from_diag(&diags)
                }
                else {
                     Array2::eye(mode.n_fock)
                }
            }).reduce(|acc, sing_mat| kron(&acc, &sing_mat))
            .unwrap() * q2
        }
    }
}

pub struct LightMatter {
    pub photon: PhotonicHamiltonian,
    pub interaction: IntHamiltonian, // exciton info stored in interaction
    pub exciton_energies: Array1<f64>,
    n_rel: usize, 
    n_ex_eig: usize, 
    parity: i32
}

impl LightMatter {
    pub fn new( exciton_ham: FullHamiltonian, 
                photon_modes: Vec<PhotonMode>, 
                coupling: f64,
                n_rel:usize, 
                n_ex_eig:usize, 
                parity:i32) -> Self {

        let (exciton_energies, e_vecs, _, r_vecs, _) 
            = exciton_ham.diagonalize_full_hamiltonian(n_rel, n_ex_eig, parity);

        let photon = PhotonicHamiltonian::new(photon_modes.clone());
        let interaction = IntHamiltonian::new(exciton_ham, photon_modes.clone(), coupling, e_vecs, r_vecs);
        
        Self { photon: photon, interaction: interaction, exciton_energies: exciton_energies, n_rel: n_rel,
            n_ex_eig: n_ex_eig, parity: parity}
    }

    pub fn build_lm_hamiltonian(&self) -> Array2<f64>{

        let h_ex = Array2::from_diag(&self.exciton_energies);
        let h_ph = self.photon.get_h_ph();
        let h_int = self.interaction.get_int_h();

        kron(&h_ex, &Array2::eye(h_ph.shape()[0]))
        + kron(&Array2::eye(h_ex.shape()[0]), &h_ph)
        + h_int
    }

    pub fn diagonalize_lm_ham(&self) -> (Array1<f64> , Array2<f64>){
        
        let h_lm = self.build_lm_hamiltonian();

        h_lm.eigh(ndarray_linalg::UPLO::Upper).unwrap()

    }

    pub fn update_k(&self, k: Vec<f64>) -> Self {
        let mut exciton = self.interaction.exciton.clone();
        exciton.k = k;

        LightMatter::new( exciton, self.photon.photon_modes.clone(), self.interaction.coupling,self.n_rel,
            self.n_ex_eig, self.parity)
    }

}

pub struct BandStructureCalculatorLM {
    pub light_matter: LightMatter,
    pub a0: f64,
    pub v0: f64,
    pub save_dir: String,
    pub sym_points: Vec<Vec<f64>>
}

impl BandStructureCalculatorLM {
    pub fn new(light_matter: LightMatter, a0: f64, v0: f64, save_dir: String, sym_points: Vec<Vec<f64>>) -> Self {
        Self { light_matter: light_matter, a0: a0, v0: v0, save_dir: save_dir, sym_points: sym_points}
    }

    pub fn calculate_at_k_point_lm(&self, k: Vec<f64>) -> (Array1<f64> , Array2<f64>) {
        let light_matter = self.light_matter.update_k(k);

        light_matter.diagonalize_lm_ham()
    }

    pub fn get_lm_band_structure_filename(
        &self,
        n_points: usize,
    ) -> String {
        format!(
            "{}/band_a0{}_V0{}_grid{}_box{}_nrel{}_nbands{}_npoints{}_par{}_coup{}.json",
            self.save_dir,
            self.a0,
            self.v0,
            self.light_matter.interaction.exciton.rel_ham.grid.grid_size,
            self.light_matter.interaction.exciton.rel_ham.grid.box_size,
            self.light_matter.n_rel,
            self.light_matter.interaction.exciton.kappa_vectors.len(),
            n_points,
            self.light_matter.parity,
            self.light_matter.interaction.coupling
        )
    }
    
    pub fn calculate_lm_band_structure(&self, n_points: usize) -> (Vec<Vec<f64>>, Vec<Array1<f64>>) {
        
        let _filename = self.get_lm_band_structure_filename(n_points);

        let points_per_seg = n_points / self.sym_points.len();

        let k_points: Vec<Vec<f64>> = self.sym_points.iter().enumerate().flat_map( | (sym_ind,sym_point) | {
            let sym_ind2 = (sym_ind + 1)%&self.sym_points.len();
            (0 .. points_per_seg).map(move | t | {
                let t_frac = t as f64  / points_per_seg as f64;
                let x = sym_point[0] + t_frac * (&self.sym_points[sym_ind2][0] - sym_point[0]);
                let y = sym_point[1] + t_frac * (&self.sym_points[sym_ind2][1] - sym_point[1]);

                vec![x,y]
            })
        }).collect();

        let total = k_points.len() as u64;
        let pb = ProgressBar::new(total);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} k-points ({eta})")
            .unwrap()
            .progress_chars("#>-"));

        let eigvals_list: Vec<Array1<f64>> = k_points.iter().map(|k| {
            let (eigvals, _) =self.calculate_at_k_point_lm(k.clone());
            pb.inc(1);
            eigvals
        }).collect();
        pb.finish_with_message("Bandstructure calculation complete");

        // save_band_data(filename, (k_points.clone(), eigvals_list.clone())).unwrap();

        (k_points, eigvals_list)
    }

}


pub fn plot_band_structure(
    k_points: &Vec<Vec<f64>>,
    sym_points: Vec<Vec<f64>>,
    sym_labels: Vec<String>,
    eigvals_list: &Vec<Array1<f64>>,
    a0: f64,
    save_path: Option<&str>,
    _show: bool,
) {
    
    if let Some(path) = save_path {
        info!("Saving band structure plot to {}", path);
        
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        
        // Create x-axis for the plot (path length)
        let mut x = vec![0.0; k_points.len()];
        for i in 1..k_points.len() {
            let dx = k_points[i][0] - k_points[i-1][0];
            let dy = k_points[i][1] - k_points[i-1][1];
            x[i] = x[i-1] + (dx*dx + dy*dy).sqrt();
        }
        
        let dx = k_points[1][0] - k_points[0][0];

        // Define the high-symmetry points
        let _b1 = 2.0 * std::f64::consts::PI / a0;
        let points_per_segment = k_points.len() / 3;
        let n_sym = sym_points.len();
        
        let mut sym_positions:Vec<f64> = sym_points.iter().enumerate().map(|(ind,_)| {
                x[((ind * points_per_segment) as i32 - 1).max(0) as usize]
            }).collect();
        sym_positions.push(x[x.len() - 1]);

        println!("Sym_positions = {:?}", sym_positions);

        // Find min/max energy values
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;
        
        for eigvals in eigvals_list {
            min_energy = min_energy.min(eigvals.fold(f64::INFINITY, |a, &b| a.min(b)));
            max_energy = max_energy.max(eigvals.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
        
        // Set up chart
        let mut chart = ChartBuilder::on(&root)
            .caption("Exciton Band Structure", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                x[0]..x[x.len() - 1],
                min_energy..max_energy
            )
            .unwrap();
        
        // Configure mesh with custom x-ticks for high-symmetry points
        chart.configure_mesh()
            .x_labels(4)
            .x_label_formatter(&|v| {
                sym_positions.iter().enumerate().map(|(ind,x)| {
                    if (v - x).abs() < dx {
                        sym_labels[ind%n_sym].clone()
                    }
                    else {
                        "".to_string()
                    }
                }).collect()
            })
            .y_desc("Energy")
            .draw()
            .unwrap();
        
        // Draw vertical lines at high-symmetry points
        let line_vec: Vec<PathElement<_>> = sym_positions.iter().map(|x| {
            PathElement::new(vec![(*x, min_energy), (*x, max_energy)], BLACK.mix(0.5))
        }).collect();

        chart.draw_series(line_vec).unwrap();
        
        // Draw each band with colors based on parity values
        for band_idx in 0..eigvals_list[0].len() {
            for k_idx in 0..x.len() - 1 {
                let color = BLUE;
                
                // Plot the segment with the appropriate color
                chart.draw_series(LineSeries::new(
                    vec![
                        (x[k_idx], eigvals_list[k_idx][band_idx]),
                        (x[k_idx+1], eigvals_list[k_idx+1][band_idx])
                    ],
                    color.mix(0.8)
                )).unwrap();
            }
        }
        
        root.present().unwrap();
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
    let a0 = (&e_config).a0;
    let v0 = (&e_config).v0;
    let exciton = Exciton2DGridSystem::from_config(e_config);
    
    // Solve for the relative coordinate eigenstates
    let n_rel = 16;
    let n_ex_eig = 30;
    let parity = 1;

    // let start_time = Instant::now();

    // let (sorted_eigvals, _, _, _, _) 
    //     = exciton.diagonalize_full_hamiltonian(n_rel, n_ex_eig, parity);

    // let h_ex = Array2::from_diag(&sorted_eigvals);
    // println!("Exciton Hamiltonian of size: {:?}", h_ex.shape());
    // let end_time = Instant::now();
    // let duration = end_time.duration_since(start_time);
    // println!("Built Exciton Hamiltonian in {:?}", duration);
    

    
    let coupling = 0.01;
    let omega_0 = 0.1;
    let qx_values = vec![-2e-4, 0.0, 2e-4];
    let qz = omega_0 / C;
    let n_fock = 2;
    let photon_modes: Vec<PhotonMode> = qx_values.iter().map(|qx| {
            PhotonMode::new(Vector3D { x: *qx, y: 0.0, z: qz }, Polarization::TE, n_fock)
        }).collect();
    
    let light_matter = LightMatter::new(exciton.full_ham, photon_modes, coupling,
                                    n_rel,n_ex_eig,parity);

    let sym_points = vec![ vec![0.0,0.0], vec![PI / a0, 0.0], vec![PI/a0, PI/a0]];

    let start_time = Instant::now();
    let band_structure = BandStructureCalculatorLM::new(light_matter, a0,v0, "lm_data/".to_string(), sym_points.clone());
    let (k_points, eigvals_list) = band_structure.calculate_lm_band_structure(60);
    let end_time = Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Calc dispersion in {:?}", duration);

    let sym_labels = vec!["Γ".to_string(), "X".to_string(), "M".to_string()];
    plot_band_structure(&k_points, sym_points, sym_labels, &eigvals_list, a0, Some("disp_plot.png"), false);

    
    // let start_time = Instant::now();
    // let h_ph = light_matter.photon.get_h_ph();
    // println!("Photonic Hamiltonian of size: {:?}", h_ph.shape());
    // let end_time = Instant::now();
    // let duration = end_time.duration_since(start_time);
    // println!("Built Photonic Hamiltonian in {:?}", duration);

    // let start_time = Instant::now();
    // let h_int = light_matter.interaction.get_int_h();
    // println!("Interaction Hamiltonian of size: {:?}", h_int.shape());
    // let end_time = Instant::now();
    // let duration = end_time.duration_since(start_time);
    // println!("Built Interaction Hamiltonian in {:?}", duration);

    // let start_time = Instant::now();
    // // let light_matter = LightMatter::new(exciton.full_ham, photon_modes, coupling,
    // //                                 n_rel,n_ex_eig,parity);
    // let h = light_matter.build_lm_hamiltonian();
    // println!("Full Hamiltonian of size: {:?}", h.shape());
    // let end_time = Instant::now();
    // let duration = end_time.duration_since(start_time);
    // println!("Built Full Hamiltonian in {:?}", duration);



}
