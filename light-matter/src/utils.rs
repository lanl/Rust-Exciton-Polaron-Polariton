use ndarray::{Array1, Array2, array};
use serde::{Serialize, Deserialize};
use sprs::smmp::mul_csr_csr;
use std::path::Path;
use bincode::{self};
use exciton_2d::ExcitonConfig;
use num_complex::Complex;


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
    pub lm_coupling: f64,
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
}

impl PPConfig {
    pub fn new(
        // Exciton Parameters
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,

        // Polariton Parameters
        lm_coupling: f64,
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
            lm_coupling,
            omega_0,
            q_values,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
        }
    }
}

use std::fmt;
#[derive(Debug)]
pub struct PrettyArray2<'a>(&'a Array2<f64>, usize); // Second parameter is precision

impl<'a> fmt::Display for PrettyArray2<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = self.1;
        
        // First pass: determine the maximum width for each column
        let mut max_widths = vec![0; self.0.ncols()];
        
        for row in self.0.rows() {
            for (i, val) in row.iter().enumerate() {
                // Format the number in scientific notation with the specified precision
                let formatted = format!("{:.*e}", precision, val);
                max_widths[i] = max_widths[i].max(formatted.len());
            }
        }
        
        // Calculate the total width of each column including padding and divider
        let col_widths: Vec<usize> = max_widths.iter().map(|w| w + 2).collect(); // +2 for padding
        
        // Function to create a horizontal divider line
        let create_divider = |widths: &[usize]| {
            let mut divider = String::from("+");
            for width in widths {
                divider.push_str(&"-".repeat(*width));
                divider.push('+');
            }
            divider
        };
        
        let divider = create_divider(&col_widths);
        
        // Add a newline at the beginning
        writeln!(f)?;
        
        // Print the top divider
        writeln!(f, "{}", divider)?;
        
        // Second pass: print with proper alignment and dividers
        for row in self.0.rows() {
            write!(f, "|")?; // Start each row with a divider
            
            for (i, val) in row.iter().enumerate() {
                // Right-align each number within its column width
                write!(f, " {:>width$.prec$e} |", val, width = max_widths[i], prec = precision)?;
            }
            writeln!(f)?;
            
            // Print a horizontal divider after each row
            writeln!(f, "{}", divider)?;
        }
        
        // Add a newline at the end
        writeln!(f)?;
        
        Ok(())
    }
}

/// Helper trait for sqrt operation
pub trait Sqrt {
    fn sqrt(self) -> Self;
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Sqrt for Complex<f64> {
    fn sqrt(self) -> Self {
        Complex::sqrt(self)
    }
}

impl Sqrt for Complex<f32> {
    fn sqrt(self) -> Self {
        Complex::sqrt(self)
    }
}

use ndarray::linalg::kron;
use num_traits::{Zero, NumCast};
// use ndarray::ScalarOperand;
/// Trait for matrix operations that work with both dense and sparse matrices
pub trait MatrixOps: Clone {
    type Scalar: Clone + Copy
        + Zero 
        // + One 
        // + ScalarOperand
        // + std::ops::Add<Output = Self::Scalar>
        // + std::ops::Mul<Output = Self::Scalar>
        ;

    /// Create an identity matrix
    fn eye(n: usize) -> Self;
    
    /// Create a zero matrix
    fn zeros(shape: (usize, usize)) -> Self;
    
    /// Kronecker product
    fn kron(&self, other: &Self) -> Self;
    
    /// Matrix addition
    fn add(&self, other: &Self) -> Self;
    
    /// Get shape
    fn shape(&self) -> (usize, usize);

    /// Get from diagonal values
    fn from_diag(diags: Vec<Self::Scalar>) -> Self;

    /// Matrix Multiplication
    fn dot(&self, other:&Self) -> Self;

    fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>);

    fn from_trimat(trimat: &TriMat<Self::Scalar>)-> Self;

    fn transpose(&self) -> Self;

    fn scale<T>(&self, scalar: T) -> Self 
        where
            T: Clone + NumCast + Zero,
            Self::Scalar: NumCast + Zero;

    fn from_dense<T>(dense: &Array2<T>) -> Self
        where T: Clone + NumCast + Zero,;
}

// Implement for dense matrices
macro_rules! impl_matrix_ops_dense {
    ($scalar:ty) => {
        impl MatrixOps for Array2<$scalar>
        {
            type Scalar = $scalar;
            
            fn eye(n: usize) -> Self {
                Array2::eye(n)
            }
            
            fn zeros(shape: (usize, usize)) -> Self {
                Array2::zeros(shape)
            }
            
            fn kron(&self, other: &Self) -> Self {
                kron(self, other)
            }
            
            fn add(&self, other: &Self) -> Self {
                self + other
            }
            
            fn shape(&self) -> (usize, usize) {
                (self.nrows(), self.ncols())
            }

            fn dot(&self, other:&Self) -> Self {
                self.dot(other)
            }

            fn from_diag(diags: Vec<Self::Scalar>) -> Self {
                Array2::from_diag(&Array1::from_vec(diags))
            }

            fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>) {
                trimat.triplet_iter().for_each(|(&v, (i,j))| {
                    self[[i,j]] += v;
                });
            }
            
            fn from_trimat(trimat: &TriMat<Self::Scalar>) -> Self {
                let (rows, cols) = trimat.shape();
                let mut dense = Array2::zeros((rows, cols));
                
                trimat.triplet_iter().for_each(|(&v, (i,j))| {
                    dense[[i,j]] = v;
                });

                dense
            }

            fn transpose(&self) -> Self { self.t().to_owned() }

            fn scale<T>(&self, scalar: T) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: NumCast + Zero 
            {
                self * <Self::Scalar as NumCast>::from(scalar).unwrap()
            }

            fn from_dense<T>(dense: &Array2<T>) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: NumCast + Zero,
            { 
                dense.mapv(|x| NumCast::from(x).unwrap_or_else(|| Self::Scalar::zero()))
            }
        }
    };
}

use sprs::{CsMat};
use sprs::TriMat;
// Implement for sparse matrices
macro_rules! impl_matrix_ops_sparse {
    ($scalar:ty) => {
        impl MatrixOps for CsMat<$scalar>{
            type Scalar = $scalar;
            
            fn eye(n: usize) -> Self {
                CsMat::eye(n)
            }
            
            fn zeros(shape: (usize, usize)) -> Self {
                CsMat::zero(shape)
            }
            
            fn kron(&self, other: &Self) -> Self {
                sprs::kronecker_product(self.view(), other.view())
            }
            
            fn add(&self, other: &Self) -> Self {
                self + other
            }
            
            fn shape(&self) -> (usize, usize) {
                self.shape()
            }

            fn dot(&self, other:&Self) -> Self {
                mul_csr_csr(self.view(), other.view())
            }

            fn from_diag(diags: Vec<Self::Scalar>) -> Self {
                let n = diags.len();
                let mut tri_mat = TriMat::new((n, n));
                for i in 0..n {
                    let val = diags[i];
                    if val.abs() > 1e-10 {
                        tri_mat.add_triplet(i, i, val);
                    }
                }
                tri_mat.to_csr()
            }

            fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>) {
                self.add(&trimat.to_csr());
            }

            fn from_trimat(trimat: &TriMat<Self::Scalar>)-> Self {
                trimat.to_csr()
            }

            fn transpose(&self) -> Self {self.clone().transpose_into().to_csr()}

            fn scale<T>(&self, scalar: T) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: NumCast + Zero,
            {
                self * <Self::Scalar as NumCast>::from(scalar).unwrap()
            }
            
            fn from_dense<T>(dense: &Array2<T>) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: NumCast + Zero,
            {
                let (rows, cols) = (dense.nrows(), dense.ncols());
                let mut trimat = TriMat::new((rows, cols));
                
                for i in 0..rows {
                    for j in 0..cols {
                        let val = dense[[i, j]].clone();
                        if !val.is_zero() {
                            // Convert T to Self::Scalar
                            if let Some(converted) = <Self::Scalar as NumCast>::from(val) {
                                trimat.add_triplet(i, j, converted);
                            }
                        }
                    }
                }
                
                trimat.to_csr()
            }

        }
    };
}

impl_matrix_ops_dense!(f64);
impl_matrix_ops_sparse!(f64);

