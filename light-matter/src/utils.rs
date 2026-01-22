use log::warn;
use ndarray::{Array1, Array2, array};
use rust_slepc_bindings::EPSType;
use serde::{Serialize, Deserialize};
use sprs::smmp::mul_csr_csr;
use std::path::Path;
use bincode::{self};
use exciton_2d::ExcitonConfig;
use num_complex::Complex;
use ndarray::linalg::kron;
use num_traits::{NumCast, One, Zero};
use ndarray_linalg::{Eigh};
use crate::timed_println;
use crate::slepc::diagonalize_slepc;


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
pub struct PrettyArray2<'a>(pub &'a Array2<f64>, pub usize); // Second parameter is precision

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

/// Trait for matrix operations that work with both dense and sparse matrices
pub trait MatrixOps: Clone {
    type Scalar: 
        Clone 
        + Copy
        + Zero 
        + One 
        + NumCast
        + std::ops::Add<Output = Self::Scalar>
        + std::ops::Mul<Output = Self::Scalar>;

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
    fn from_diag<T>(diags: &[T]) -> Self 
        where
            T: Clone + NumCast + Zero + Copy;

    /// Matrix Multiplication
    fn dot(&self, other:&Self) -> Self;

    fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>);

    fn from_trimat(trimat: &TriMat<Self::Scalar>)-> Self;

    fn transpose(&self) -> Self;

    fn scale<T>(&self, scalar: T) -> Self 
        where
            T: Clone + NumCast + Zero;

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

            fn from_diag<T>(diags: &[T]) -> Self 
            where
                T: Clone + NumCast + Zero + Copy,
                Self::Scalar: NumCast + Zero 
            {
                Array2::from_diag(
                    &diags.into_iter().map(|val| {
                        <Self::Scalar as NumCast>::from(*val).unwrap()
                    }).collect::<Array1<Self::Scalar>>()
                )
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

            fn from_diag<T>(diags: &[T]) -> Self 
            where
                T: Clone + NumCast + Zero + Copy
            {
                let n = diags.len();
                let mut tri_mat = TriMat::new((n, n));
                
                diags.iter().enumerate().for_each(|(i, &val)|{
                    if !val.is_zero() {
                        tri_mat.add_triplet(i, i, <Self::Scalar as NumCast>::from(val).unwrap());
                    }
                });

                tri_mat.to_csr()
            }

            fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>) {
                *self = self.add(&trimat.to_csr());
            }

            fn from_trimat(trimat: &TriMat<Self::Scalar>)-> Self {
                trimat.to_csr()
            }

            fn transpose(&self) -> Self {self.clone().transpose_into().to_csr()}

            fn scale<T>(&self, scalar: T) -> Self 
            where
                T: Clone + NumCast + Zero
            {
                self * <Self::Scalar as NumCast>::from(scalar).unwrap()
            }
            
            fn from_dense<T>(dense: &Array2<T>) -> Self 
            where
                T: Clone + NumCast + Zero
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


pub trait ArrayOperator<T> {
    fn apply(&self, x: &Array1<T>) -> Vec<T>;
}

pub trait DiagOperator<T> {
    fn form_diagonal(&self) -> Array1<T>;
}

pub trait Operator<T> {
    /// Apply the operator to a vector
    fn apply(&self, x: &[T], y: &mut [T]);
}

/// Trait for operators that can be used with solvers
pub trait SolverOperator<T: Clone>: Send + Sync  {
    /// Apply the operator to a vector
    fn apply(&self, x: &[T], y: &mut [T]);

    /// Apply the transpose of the operator to a vector
    fn apply_transpose(&self, x: &[T], y: &mut [T]);
    
    /// Get the diagonal of the operator (used for preconditioning)
    fn get_diagonal(&self) -> Vec<T>;
    
    /// Get the dimension of the operator
    fn dimension(&self) -> usize;

    /// Diagonalize operator
    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<f64>>) -> (Array1<f64>, Array2<f64>);
}

#[derive(Clone)]
pub struct CsMatOperator {
    pub matrix: CsMat<f64>,
}

impl CsMatOperator {
    pub fn new(matrix: CsMat<f64>) -> Self{
        Self { matrix }
    }
}

impl SolverOperator<f64> for CsMatOperator {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        // Apply matrix-vector product
        sprs::prod::mul_acc_mat_vec_csr(self.matrix.view(), x, y);
    }

    fn apply_transpose(&self, x: &[f64], y: &mut [f64]) {
        // Apply matrix-vector product
        sprs::prod::mul_acc_mat_vec_csr(self.matrix.transpose_view(), x, y);
    }

    fn get_diagonal(&self) -> Vec<f64> {
        let mut diags = vec![0.0; self.matrix.shape().0];

        self.matrix.diag().iter().for_each(|(i, val)| {
            diags[i] = *val;
        });

        diags
    }

    fn dimension(&self) -> usize {
        self.matrix.shape().0
    }

    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<f64>>) -> (Array1<f64>, Array2<f64>) {
        if let Some(n) = n_eig {                   
            diagonalize_slepc(self, n, 128, solve_type, initial_guess)
        }
        else {
            timed_println!("Warning: No eigenvalue count specified, using default of 128");
            diagonalize_slepc(self, 128, 128, solve_type, initial_guess)
        }
    }
}

#[derive(Clone)]
pub struct DenseOperator {
    pub matrix: Array2<f64>,
}

impl DenseOperator {
    pub fn new(matrix: Array2<f64>) -> Self{
        Self { matrix }
    }
}

impl SolverOperator<f64> for DenseOperator {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        // Apply matrix-vector product
        let x_arr = Array1::from_iter(x.iter().map(|a| *a));
        let result = self.matrix.dot(&x_arr);
        y.iter_mut().zip(result.iter()).for_each(|(y_i, r_i)| {
            *y_i = *r_i
        });
    }
    
    fn apply_transpose(&self, x: &[f64], y: &mut [f64]) {
        // Apply matrix-vector product
        let x_arr = Array1::from_iter(x.iter().map(|a| *a));
        let result = self.matrix.t().dot(&x_arr);
        y.iter_mut().zip(result.iter()).for_each(|(y_i, r_i)| {
            *y_i = *r_i
        });
    }

    fn get_diagonal(&self) -> Vec<f64> {
        self.matrix.diag().to_vec()
    }

    fn dimension(&self) -> usize {
        self.matrix.shape()[0]
    }

    fn diagonalize(&self, _: Option<usize>, _: Option<EPSType>, _: Option<&Array2<f64>>) 
        -> (Array1<f64>, Array2<f64>) {
        
        self.matrix.eigh(ndarray_linalg::UPLO::Upper).unwrap()
    }
}

use std::sync::Arc;

/// Enum representing different types of linear operator compositions
#[derive(Clone)]
pub enum ComposedOperator<T: Clone> {
    /// Base operator (wraps a SolverOperator)
    Base(Arc<dyn SolverOperator<T>>),
    
    /// Sum of two operators: (A + B)x = Ax + Bx
    Add(Box<ComposedOperator<T>>, Box<ComposedOperator<T>>),

    /// Matrix multiplication of two operators: ABx = A(Bx)
    Dot(Box<ComposedOperator<T>>, Box<ComposedOperator<T>>),
    
    /// Scalar multiplication: (αA)x = α(Ax)
    ScalarMul(T, Box<ComposedOperator<T>>),
    
    /// Kronecker product: (A ⊗ B)x
    Kron(Box<ComposedOperator<T>>, Box<ComposedOperator<T>>),
    
    /// Diagonal operator
    Diagonal(Vec<T>),
    
    /// Identity operator
    Identity(usize),
    
    /// Zero operator
    Zero(usize, usize),

    /// Transpose operator
    Transpose(Box<ComposedOperator<T>>),
}

impl<T: Clone + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>> ComposedOperator<T> {
    /// Create from a base operator
    pub fn from_operator<O: SolverOperator<T> + 'static>(op: O) -> Self {
        ComposedOperator::Base(Arc::new(op))
    }
    
    /// Get the dimension of the operator
    pub fn dimension(&self) -> usize {
        match self {
            ComposedOperator::Base(op) => op.dimension(),
            ComposedOperator::Add(a, _) => a.dimension(),
            ComposedOperator::Dot(a, _) => a.dimension(),
            ComposedOperator::ScalarMul(_, a) => a.dimension(),
            ComposedOperator::Kron(a, b) => a.dimension() * b.dimension(),
            ComposedOperator::Diagonal(diag) => diag.len(),
            ComposedOperator::Identity(n) => *n,
            ComposedOperator::Zero(n, _) => *n,
            ComposedOperator::Transpose(a) => a.dimension()
        }
    }

    pub fn apply_transpose(&self, x: &[T], y: &mut [T]) 
    where
        T: Copy + One,
    {
        let dim = self.dimension();
        assert_eq!(dim, x.len(), "Input vector x has wrong dimension: expected {}, got {}", dim, x.len());
        assert_eq!(dim, y.len(), "Output vector y has wrong dimension: expected {}, got {}", dim, y.len());

        match self {
            ComposedOperator::Base(op) => {
                op.apply_transpose(x, y);
            }
            
            ComposedOperator::Add(a, b) => {
                // y = Ax + Bx
                let mut temp = vec![T::zero(); y.len()];
                a.apply_transpose(x, y);
                b.apply_transpose(x, &mut temp);
                for i in 0..y.len() {
                    y[i] = y[i] + temp[i];
                }
            }
            
            ComposedOperator::Dot(a, b) => {
                // y = ABx
                let mut temp = vec![T::zero(); y.len()];
                b.apply_transpose(x, &mut temp);
                a.apply_transpose(&temp, y);
            }

            ComposedOperator::ScalarMul(alpha, a) => {
                // y = α(Ax)
                a.apply_transpose(x, y);
                for i in 0..y.len() {
                    y[i] = *alpha * y[i];
                }
            }
            
            ComposedOperator::Kron(a, b) => {
                let na = a.dimension();
                let nb = b.dimension();

                match (a.as_ref(), b.as_ref()) {
                    // I ⊗ B = B repeated na times
                    (ComposedOperator::Identity(_), _) => {
                        for j in 0..na {
                            b.apply_transpose(&x[j * nb..(j + 1) * nb], &mut y[j * nb..(j + 1) * nb]);
                        }
                    }
                    
                    // A ⊗ I = A applied to strided elements
                    (_, ComposedOperator::Identity(_)) => {
                        for i in 0..nb {
                            // Gather strided input
                            let x_col: Vec<T> = (0..na).map(|j| x[j * nb + i]).collect();
                            let mut y_col = vec![T::zero(); na];
                            
                            // Apply A
                            a.apply_transpose(&x_col, &mut y_col);
                            
                            // Scatter strided output
                            for (j, &val) in y_col.iter().enumerate() {
                                y[j * nb + i] = val;
                            }
                        }
                    }
                    
                    // General case: (A ⊗ B) = (I ⊗ B)(A ⊗ I)
                    _ => {
                        let mut z = vec![T::zero(); na * nb];
                        
                        // Step 1: z = (A ⊗ I)x
                        for i in 0..nb {
                            // Gather strided input
                            let x_col: Vec<T> = (0..na).map(|j| x[j * nb + i]).collect();
                            let mut z_col = vec![T::zero(); na];
                            
                            // Apply A
                            a.apply_transpose(&x_col, &mut z_col);
                            
                            // Scatter strided output
                            for (j, &val) in z_col.iter().enumerate() {
                                z[j * nb + i] = val;
                            }
                        }
                        
                        // Step 2: y = (I ⊗ B)z
                        for j in 0..na {
                            b.apply_transpose(&z[j * nb..(j + 1) * nb], &mut y[j * nb..(j + 1) * nb]);
                        }
                    }
                }
            }
            
            ComposedOperator::Diagonal(diag) => {
                for i in 0..diag.len() {
                    y[i] = diag[i] * x[i];
                }
            }
            
            ComposedOperator::Identity(n) => {
                y[..*n].copy_from_slice(&x[..*n]);
            }
            
            ComposedOperator::Zero(_, _) => {
                y.fill(T::zero());
            }

            ComposedOperator::Transpose(a) => {
                a.apply(x, y);
            }
        }
    }
    
    
    /// Apply the composed operator to a vector
    pub fn apply(&self, x: &[T], y: &mut [T]) 
    where
        T: Copy + One,
    {
        let dim = self.dimension();
        assert_eq!(dim, x.len(), "Input vector x has wrong dimension: expected {}, got {}", dim, x.len());
        assert_eq!(dim, y.len(), "Output vector y has wrong dimension: expected {}, got {}", dim, y.len());

        match self {
            ComposedOperator::Base(op) => {
                op.apply(x, y);
            }
            
            ComposedOperator::Add(a, b) => {
                // y = Ax + Bx
                let mut temp = vec![T::zero(); y.len()];
                a.apply(x, y);
                b.apply(x, &mut temp);
                for i in 0..y.len() {
                    y[i] = y[i] + temp[i];
                }
            }
            
            ComposedOperator::Dot(a, b) => {
                // y = ABx
                let mut temp = vec![T::zero(); y.len()];
                b.apply(x, &mut temp);
                a.apply(&temp, y);
            }

            ComposedOperator::ScalarMul(alpha, a) => {
                // y = α(Ax)
                a.apply(x, y);
                for i in 0..y.len() {
                    y[i] = *alpha * y[i];
                }
            }
            
            ComposedOperator::Kron(a, b) => {
                let na = a.dimension();
                let nb = b.dimension();

                match (a.as_ref(), b.as_ref()) {
                    // I ⊗ B = B repeated na times
                    (ComposedOperator::Identity(_), _) => {
                        for j in 0..na {
                            b.apply(&x[j * nb..(j + 1) * nb], &mut y[j * nb..(j + 1) * nb]);
                        }
                    }
                    
                    // A ⊗ I = A applied to strided elements
                    (_, ComposedOperator::Identity(_)) => {
                        for i in 0..nb {
                            // Gather strided input
                            let x_col: Vec<T> = (0..na).map(|j| x[j * nb + i]).collect();
                            let mut y_col = vec![T::zero(); na];
                            
                            // Apply A
                            a.apply(&x_col, &mut y_col);
                            
                            // Scatter strided output
                            for (j, &val) in y_col.iter().enumerate() {
                                y[j * nb + i] = val;
                            }
                        }
                    }
                    
                    // General case: (A ⊗ B) = (I ⊗ B)(A ⊗ I)
                    _ => {
                        let mut z = vec![T::zero(); na * nb];
                        
                        // Step 1: z = (A ⊗ I)x
                        for i in 0..nb {
                            // Gather strided input
                            let x_col: Vec<T> = (0..na).map(|j| x[j * nb + i]).collect();
                            let mut z_col = vec![T::zero(); na];
                            
                            // Apply A
                            a.apply(&x_col, &mut z_col);
                            
                            // Scatter strided output
                            for (j, &val) in z_col.iter().enumerate() {
                                z[j * nb + i] = val;
                            }
                        }
                        
                        // Step 2: y = (I ⊗ B)z
                        for j in 0..na {
                            b.apply(&z[j * nb..(j + 1) * nb], &mut y[j * nb..(j + 1) * nb]);
                        }
                    }
                }
            }
            
            ComposedOperator::Diagonal(diag) => {
                for i in 0..diag.len() {
                    y[i] = diag[i] * x[i];
                }
            }
            
            ComposedOperator::Identity(n) => {
                y[..*n].copy_from_slice(&x[..*n]);
            }
            
            ComposedOperator::Zero(_, _) => {
                y.fill(T::zero());
            }

            ComposedOperator::Transpose(a) => {
                a.apply_transpose(x, y);
            }
        }
    }
    
    /// Get the diagonal of the operator
    pub fn get_diagonal(&self) -> Vec<T> 
    where
        T: Copy + One,
    {
        match self {
            ComposedOperator::Base(op) => op.get_diagonal(),
            
            ComposedOperator::Add(a, b) => {
                let diag_a = a.get_diagonal();
                let diag_b = b.get_diagonal();
                diag_a.iter().zip(diag_b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }

            ComposedOperator::Dot(a, b) => {
                // Fast paths for special cases
                match (a.as_ref(), b.as_ref()) {
                    (ComposedOperator::Diagonal(d_a), ComposedOperator::Diagonal(d_b)) => {
                        d_a.iter().zip(d_b.iter()).map(|(&a, &b)| a * b).collect()
                    }
                    (ComposedOperator::Diagonal(d_a), _) => {
                        let d_b = b.get_diagonal();
                        d_a.iter().zip(d_b.iter()).map(|(&a, &b)| a * b).collect()
                    }
                    (_, ComposedOperator::Diagonal(d_b)) => {
                        let d_a = a.get_diagonal();
                        d_a.iter().zip(d_b.iter()).map(|(&a, &b)| a * b).collect()
                    }
                    (ComposedOperator::Identity(_), _) => b.get_diagonal(),
                    (_, ComposedOperator::Identity(_)) => a.get_diagonal(),
                    
                    // General case: apply to basis vectors
                    _ => {
                        let n = self.dimension();
                        (0..n).map(|i| {
                            let mut x = vec![T::zero(); n];
                            let mut y_b = vec![T::zero(); n];
                            let mut y_ab = vec![T::zero(); n];
                            
                            x[i] = T::one();
                            b.apply(&x, &mut y_b);
                            a.apply(&y_b, &mut y_ab);
                            
                            y_ab[i]
                        }).collect()
                    }
                }
            }
            
            ComposedOperator::ScalarMul(alpha, a) => {
                a.get_diagonal().iter()
                    .map(|&x| *alpha * x)
                    .collect()
            }
            
            ComposedOperator::Kron(a, b) => {
                let diag_a = a.get_diagonal();
                let diag_b = b.get_diagonal();
                diag_a.iter().flat_map(|&a_val| {
                    diag_b.iter().map(move |&b_val| a_val * b_val)
                }).collect()
            }
            
            ComposedOperator::Diagonal(diag) => diag.clone(),
            
            ComposedOperator::Identity(n) => vec![T::one(); *n],
            
            ComposedOperator::Zero(n, _) => vec![T::zero(); *n],

            ComposedOperator::Transpose(a) => a.get_diagonal()
        }
    }
}

// Implement SolverOperator for ComposedOperator
impl SolverOperator<f64> for ComposedOperator<f64> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        ComposedOperator::apply(self, x, y)
    }

    fn apply_transpose(&self, x: &[f64], y: &mut [f64]) {
        ComposedOperator::apply_transpose(self, x, y)
    }
    
    fn get_diagonal(&self) -> Vec<f64> {
        ComposedOperator::get_diagonal(self)
    }
    
    fn dimension(&self) -> usize {
        ComposedOperator::dimension(self)
    }

    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<f64>>) -> (Array1<f64>, Array2<f64>) {
        if let Some(n) = n_eig {                   
            diagonalize_slepc(self, n, 128, solve_type, initial_guess)
        }
        else {
            warn!("Warning: No eigenvalue count specified, using default of 128");
            diagonalize_slepc(self, 128, 128, solve_type, initial_guess)
        }
    }
}


// Implement for linear operators
macro_rules! impl_matrix_ops_lin_op {
    ($scalar:ty) => {
        impl MatrixOps for ComposedOperator<$scalar>{
            type Scalar = $scalar;
    
            fn eye(n: usize) -> Self {
                ComposedOperator::Identity(n)
            }
            
            fn zeros(shape: (usize, usize)) -> Self {
                ComposedOperator::Zero(shape.0, shape.1)
            }
            
            fn kron(&self, other: &Self) -> Self {
                ComposedOperator::Kron(Box::new(self.clone()), Box::new(other.clone()))
            }
            
            fn add(&self, other: &Self) -> Self {
                ComposedOperator::Add(Box::new(self.clone()), Box::new(other.clone()))
            }
            
            fn scale<T>(&self, scalar: T) -> Self 
                    where
                        T: Clone + NumCast + Zero
            {
                ComposedOperator::ScalarMul(<Self::Scalar as NumCast>::from(scalar).unwrap(), Box::new(self.clone()))
            }
            
            fn shape(&self) -> (usize, usize) {
                let n = self.dimension();
                (n, n)
            }
            
            fn from_diag<T>(diags: &[T]) -> Self 
            where
                T: Clone + NumCast + Zero + Copy
            {
                ComposedOperator::Diagonal(diags.iter().map(|val| {
                    <Self::Scalar as NumCast>::from(*val).unwrap()
                }).collect::<Vec<Self::Scalar>>())
            }
            
            fn from_trimat(trimat: &TriMat<Self::Scalar>) -> Self {
                // Convert TriMat to sparse and wrap as an operator
                let csmat = trimat.to_csr();
                ComposedOperator::from_operator(CsMatOperator::new(csmat))
            }
            
            fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>) {
                *self = self.add(&Self::from_trimat(trimat))
            }

            fn from_dense<T>(dense: &Array2<T>) -> Self
                    where T: Clone + NumCast + Zero, {
                
                let dense = dense.mapv(|x| NumCast::from(x).unwrap_or_else(|| Self::Scalar::zero()));
                
                ComposedOperator::from_operator(DenseOperator::new(dense))
            }

            fn transpose(&self) -> Self {
                match self {
                    ComposedOperator::Transpose(a) => {
                        // (A^T)^T = A
                        (**a).clone()
                    }
                    _ => ComposedOperator::Transpose(Box::new(self.clone()))
                }
            }

            fn dot(&self, other: &Self) -> Self {
                // Matrix-matrix product as composition
                // (AB)x = A(Bx)
                ComposedOperator::Dot(Box::new(self.clone()), Box::new(other.clone()))
            }
        }
    };
}

impl_matrix_ops_lin_op!(f64);