use log::warn;
use ndarray::{Array1, Array2, LinalgScalar, array};
use rust_slepc_bindings::EPSType;
use serde::{Serialize, Deserialize};
use sprs::smmp::mul_csr_csr;
use std::path::Path;
use bincode::{self};
use exciton_2d::ExcitonConfig;
use num_complex::Complex;
use ndarray::linalg::kron;
use num_traits::{NumCast, One, Zero};
use ndarray_linalg::{Eigh, c64};
use crate::timed_println;
use crate::slepc::diagonalize_slepc;

/// Helper trait for converting values, with special handling for complex numbers
pub trait ConvertFrom<T> {
    fn convert_from(val: T) -> Option<Self> where Self: Sized;
}

// Implement for f64 using NumCast
impl<T: NumCast> ConvertFrom<T> for f64 {
    fn convert_from(val: T) -> Option<Self> {
        NumCast::from(val)
    }
}

// Implement for c64 - identity conversion
impl ConvertFrom<c64> for c64 {
    fn convert_from(val: c64) -> Option<Self> {
        Some(val) // Identity conversion
    }
}

// Implement for c64 from f64
impl ConvertFrom<f64> for c64 {
    fn convert_from(val: f64) -> Option<Self> {
        Some(c64::new(val, 0.0))
    }
}

// Implement for c64 from other numeric types via f64
macro_rules! impl_convert_from_for_c64 {
    ($($t:ty),*) => {
        $(
            impl ConvertFrom<$t> for c64 {
                fn convert_from(val: $t) -> Option<Self> {
                    Some(c64::new(val as f64, 0.0))
                }
            }
        )*
    };
}

impl ConvertFrom<Array2<f64>> for Array2<c64>{
    fn convert_from(val: Array2<f64>) -> Option<Self> where Self: Sized {

        let shape = val.shape();

        Some(Array2::from_shape_fn([shape[0], shape[1]], |(i,j)| {
            c64::new(val[[i,j]], 0.0)
        }))
    }
}


/// Helper macro to try downcasting and converting from a specific scalar type
macro_rules! try_downcast_and_convert {
    ($any_ref:expr, $dim:expr, $scalar_type:ty) => {{
        // Try Array2<S>
        if let Some(dense_op) = $any_ref.downcast_ref::<Array2<$scalar_type>>() {
            // Convert Array2<S> to Array2<T> using ConvertFrom
            return dense_op.mapv(|val| {
                T::convert_from(val).expect("Failed to convert between scalar types")
            });
        }
        
        // Try CsMat<S> - convert to dense Array2<T>
        if let Some(sparse_op) = $any_ref.downcast_ref::<CsMat<$scalar_type>>() {
            let mut matrix = Array2::zeros(($dim, $dim));
            for (val, (row, col)) in sparse_op.iter() {
                matrix[[row, col]] = T::convert_from(*val)
                    .expect("Failed to convert between scalar types");
            }

            return matrix;
        }
        
        None::<Array2<T>>
    }};
}

pub trait HermitianConjugate {
    fn dag(&self) -> Self;
}

impl HermitianConjugate for Array2<f64> {
    fn dag(&self) -> Self {
        self.t().to_owned()
    }
}

impl HermitianConjugate for Array2<c64> {
    fn dag(&self) -> Self {
        self.t().to_owned().mapv_into(|val| val.conj())
    }
}

impl HermitianConjugate for c64 {
    fn dag(&self) -> Self {
        self.conj()
    }
}

impl HermitianConjugate for f64 {
    fn dag(&self) -> Self {
        *self
    }
}

pub trait Norm {
    fn norm(val: Self) -> f64;
}

impl Norm for c64 {
    fn norm(val: Self) -> f64 {
        c64::norm(val)
    }
}
impl Norm for f64 {
    fn norm(val: Self) -> f64 {
        val.abs()
    }
}

// Implement for norm from other numeric types via f64
macro_rules! impl_norm {
    ($($t:ty),*) => {
        $(
            impl Norm for $t {
                fn norm(val: Self) -> f64 {
                    (val as f64).abs()
                }
            }
        )*
    };
}

impl_convert_from_for_c64!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32);
impl_norm!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32);


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

use std::fmt;
// helper shared by both pretty‑printers
fn fmt_scientific(val: f64, prec: usize) -> String {
    let mut s = format!("{:.*e}", prec, val); // e.g. "1.00e0" or "1.35e-17"
    if let Some(e_pos) = s.find('e') {
        let exp_part = &s[e_pos + 1..];
        let (sign, digits) = match exp_part.chars().next() {
            Some('-') => ("-", &exp_part[1..]),
            Some('+') => ("+", &exp_part[1..]),
            _ => ("+", exp_part),
        };
        let digits_padded = if digits.len() < 2 {
            format!("0{}", digits)
        } else {
            digits.to_string()
        };
        s = format!("{}e{}{}", &s[..e_pos], sign, digits_padded);
    }
    s
}

#[derive(Debug)]
pub struct PrettyArray2<'a>(pub &'a Array2<f64>, pub usize); // second param = precision

impl<'a> fmt::Display for PrettyArray2<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prec = self.1;
        const THRESH: f64 = 1e-15;
        const GREEN: &str = "\x1b[32m";
        const RED:   &str = "\x1b[31m";
        const RESET: &str = "\x1b[0m";

        fn colourize(val: f64, s: String) -> String {
            let mag = val.abs();
            if mag <= THRESH {
                s
            } else if val > 0.0 {
                format!("{}{}{}", GREEN, s, RESET)
            } else {
                format!("{}{}{}", RED, s, RESET)
            }
        }

        // compute maximum plain width for each column
        let mut maxw = vec![0; self.0.ncols()];
        for row in self.0.rows() {
            for (j, &v) in row.iter().enumerate() {
                let plain = fmt_scientific(v, prec);
                maxw[j] = maxw[j].max(plain.len());
            }
        }
        let colw: Vec<usize> = maxw.iter().map(|w| w + 2).collect(); // +2 for the fixed
                                                                 // one‑space margins
        let make_div = |widths: &[usize]| {
            let mut d = String::from("+");
            for &w in widths {
                d.push_str(&"-".repeat(w));
                d.push('+');
            }
            d
        };
        let divider = make_div(&colw);

        writeln!(f)?;
        writeln!(f, "{}", divider)?;
        for row in self.0.rows() {
            write!(f, "|")?;
            for (j, &v) in row.iter().enumerate() {
                let plain = fmt_scientific(v, prec);
                let coloured = colourize(v, plain.clone());
                let pad = maxw[j] - plain.len();       // everything goes on the right

                write!(
                    f,
                    " {}{}{} |",
                    coloured,
                    " ".repeat(pad),
                    ""          // trailing space is already in the format string
                )?;
            }
            writeln!(f)?;
            writeln!(f, "{}", divider)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

pub struct PrettyC64Array2<'a>(pub &'a Array2<c64>, pub usize); // second parameter is precision

impl<'a> fmt::Display for PrettyC64Array2<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prec = self.1;
        const THRESH: f64 = 1e-15;
        const GREEN: &str = "\x1b[32m";
        const RED:   &str = "\x1b[31m";
        const RESET: &str = "\x1b[0m";

        fn colourize(val: c64, s: String) -> String {
            let real = val.re;
            let imag = val.im;
            let mag_real = real.abs();
            let mag_imag = imag.abs();
            if mag_real <= THRESH && mag_imag <= THRESH {
                s
            } else if real > THRESH || (mag_real <= THRESH && imag > THRESH) {
                format!("{}{}{}", GREEN, s, RESET)
            } else {
                format!("{}{}{}", RED, s, RESET)
            }
        }

        let mut maxw = vec![0; self.0.ncols()];
        for row in self.0.rows() {
            for (j, v) in row.iter().enumerate() {
                let real_s = fmt_scientific(v.re, prec);
                let imag_s = fmt_scientific(v.im.abs(), prec);
                let sign = if v.im >= 0.0 { "+" } else { "-" };
                let plain = format!("({} {} {} i)", real_s, sign, imag_s);
                maxw[j] = maxw[j].max(plain.len());
            }
        }
        let colw: Vec<usize> = maxw.iter().map(|w| w + 2).collect();
        let make_div = |widths: &[usize]| {
            let mut d = String::from("+");
            for &w in widths {
                d.push_str(&"-".repeat(w));
                d.push('+');
            }
            d
        };
        let divider = make_div(&colw);

        writeln!(f)?;
        writeln!(f, "{}", divider)?;
        for row in self.0.rows() {
            write!(f, "|")?;
            for (j, v) in row.iter().enumerate() {
                let real_s = fmt_scientific(v.re, prec);
                let imag_s = fmt_scientific(v.im.abs(), prec);
                let sign = if v.im >= 0.0 { "+" } else { "-" };
                let plain = format!("({} {} {} i)", real_s, sign, imag_s);
                let coloured = colourize(*v, plain.clone());
                let pad = maxw[j] - plain.len();

                write!(
                    f,
                    " {}{}{} |",
                    coloured,
                    " ".repeat(pad),
                    ""
                )?;
            }
            writeln!(f)?;
            writeln!(f, "{}", divider)?;
        }
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
        + std::ops::Mul<Output = Self::Scalar>
        + ConvertFrom<Self::Scalar>
        + ConvertFrom<f64>
        + LinalgScalar;

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
            T: Clone + NumCast + Zero + Copy,
            Self::Scalar: ConvertFrom<T>;

    /// Matrix Multiplication
    fn dot(&self, other:&Self) -> Self;

    fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>);

    fn from_trimat(trimat: &TriMat<Self::Scalar>)-> Self;

    fn transpose(&self) -> Self;

    fn scale<T>(&self, scalar: T) -> Self 
        where
            T: Clone + NumCast + Zero,
            Self::Scalar: ConvertFrom<T>;

    fn from_dense<T>(dense: &Array2<T>) -> Self
        where 
            T: Clone + NumCast + Zero + 'static  + Copy,
            Self::Scalar: ConvertFrom<T>;
    
    /// Sum multiple matrices efficiently
    /// Default implementation uses pairwise addition, but can be overridden
    fn sum(matrices: Vec<Self>) -> Self 
    where 
        Self: Sized,
    {
        if matrices.is_empty() {
            panic!("Cannot sum empty list of matrices");
        }
        matrices.into_iter().reduce(|acc, m| acc.add(&m)).unwrap()
    }

    /// Build a sum of Kronecker products efficiently
    /// Input: vector of (left_op, right_op) tuples
    /// Output: Σᵢ (left_op_i ⊗ right_op_i)
    /// 
    /// Default implementation creates Kron operators then sums them.
    /// ComposedOperator overrides this to directly build KronSum when possible.
    fn build_kron_sum(terms: Vec<(Self, Self)>, _: usize, _: usize) -> Self 
    where 
        Self: Sized,
    {
        // Default: just do normal kron + sum
        let kron_terms: Vec<Self> = terms
            .into_iter()
            .map(|(left, right)| left.kron(&right))
            .collect();
        Self::sum(kron_terms)
    }
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
                ndarray::linalg::Dot::dot(self,other)
            }

            fn from_diag<T>(diags: &[T]) -> Self 
            where
                T: Clone + NumCast + Zero + Copy,
                Self::Scalar: ConvertFrom<T>
            {
                Array2::from_diag(
                    &diags.into_iter().map(|val| {
                        <Self::Scalar as ConvertFrom<T>>::convert_from(*val).unwrap()
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
                Self::Scalar: ConvertFrom<T>
            {
                self * <Self::Scalar as ConvertFrom<T>>::convert_from(scalar).unwrap()
            }

            fn from_dense<T>(dense: &Array2<T>) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: ConvertFrom<T> + Zero,
            { 
                dense.mapv(|x| <Self::Scalar as ConvertFrom<T>>::convert_from(x).unwrap_or_else(|| Self::Scalar::zero()))
            }
        }
    };
}

use sprs::{CsMat};
use sprs::{TriMat};
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
                T: Clone + NumCast + Zero + Copy,
                Self::Scalar: ConvertFrom<T>
            {
                let n = diags.len();
                let mut tri_mat = TriMat::new((n, n));
                
                diags.iter().enumerate().for_each(|(i, &val)|{
                    if !val.is_zero() {
                        tri_mat.add_triplet(i, i, <Self::Scalar as ConvertFrom<T>>::convert_from(val).unwrap());
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

            fn transpose(&self) -> Self {
                self.transpose_view().to_csr()
            }

            fn scale<T>(&self, scalar: T) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: ConvertFrom<T>
            {
                // For sparse matrices, we need to scale element-wise
                let scalar_converted = <Self::Scalar as ConvertFrom<T>>::convert_from(scalar).unwrap();
                // Use map to create a new matrix with scaled values
                self.map(|&x| x * scalar_converted)
            }
            
            fn from_dense<T>(dense: &Array2<T>) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: ConvertFrom<T>
            {
                let (rows, cols) = (dense.nrows(), dense.ncols());
                let mut trimat = TriMat::new((rows, cols));
                
                for i in 0..rows {
                    for j in 0..cols {
                        let val = dense[[i, j]].clone();
                        if !val.is_zero() {
                            // Convert T to Self::Scalar
                            if let Some(converted) = <Self::Scalar as ConvertFrom<T>>::convert_from(val) {
                                trimat.add_triplet(i, j, converted);
                            }
                        }
                    }
                }
                
                trimat.to_csr()
            }
            
            /// Optimized sum for sparse matrices using TriMat accumulator
            /// This avoids O(n) intermediate CSR matrix allocations
            fn sum(matrices: Vec<Self>) -> Self 
            where 
                Self: Sized,
            {
                if matrices.is_empty() {
                    panic!("Cannot sum empty list of matrices");
                }
                
                if matrices.len() == 1 {
                    return matrices.into_iter().next().unwrap();
                }
                
                // Get dimensions from first matrix
                let shape = matrices[0].shape();
                
                // Use TriMat as accumulator - much more efficient for summing many sparse matrices
                let mut trimat = TriMat::new(shape);
                
                // Add all non-zero entries from all matrices
                for matrix in matrices {
                    for (val, (row, col)) in matrix.iter() {
                        trimat.add_triplet(row, col, *val);
                    }
                }
                
                // Convert to CSR once at the end
                trimat.to_csr()
            }
        }
    };
}

impl_matrix_ops_dense!(f64);
impl_matrix_ops_sparse!(f64);
impl_matrix_ops_dense!(c64);
impl_matrix_ops_sparse!(c64);

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
pub trait SolverOperator<T: Clone>: Send + Sync + std::any::Any {
    /// Apply the operator to a vector
    fn apply(&self, x: &[T], y: &mut [T]);

    /// Apply the transpose of the operator to a vector
    fn apply_transpose(&self, x: &[T], y: &mut [T]);
    
    /// Get the diagonal of the operator (used for preconditioning)
    fn get_diagonal(&self) -> Vec<T>;
    
    /// Get the dimension of the operator
    fn dimension(&self) -> usize;

    /// Diagonalize operator
    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<c64>>) -> (Array1<f64>, Array2<c64>);
}

impl SolverOperator<c64> for CsMat<c64> {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Apply matrix-vector product
        // Note: sprs only provides accumulating matrix-vector multiplication (mul_acc_*)
        // so we must zero the output vector first
        y.fill(c64::zero());
        sprs::prod::mul_acc_mat_vec_csr(self.view(), x, y);
    }

    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        // Apply transpose (not conjugate transpose)
        // Note: sprs only provides accumulating matrix-vector multiplication (mul_acc_*)
        // so we must zero the output vector first
        y.fill(c64::zero());
        sprs::prod::mul_acc_mat_vec_csc(self.transpose_view(), x, y);
    }

    fn get_diagonal(&self) -> Vec<c64> {
        let mut diags: Vec<c64> = vec![c64::zero(); self.shape().0];

        self.diag().iter().for_each(|(i, val)| {
            diags[i] = *val;
        });

        diags
    }

    fn dimension(&self) -> usize {
        self.shape().0
    }

    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<c64>>) -> (Array1<f64>, Array2<c64>) {
        if let Some(n) = n_eig {                   
            diagonalize_slepc(self, n, 128, solve_type, initial_guess)
        }
        else {
            timed_println!("Warning: No eigenvalue count specified, using default of 128");
            diagonalize_slepc(self, 128, 128, solve_type, initial_guess)
        }
    }
}

impl SolverOperator<c64> for Array2<c64> {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        // Apply matrix-vector product
        let x_arr = Array1::from_iter(x.iter().copied());
        let result = ndarray::linalg::Dot::dot(self, &x_arr);
        y.iter_mut().zip(result.iter()).for_each(|(y_i, r_i)| {
            *y_i = *r_i
        });
    }
    
    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        // Apply matrix-vector product
        let x_arr = Array1::from_iter(x.iter().copied());
        let result = ndarray::linalg::Dot::dot(&self.t(), &x_arr);
        y.iter_mut().zip(result.iter()).for_each(|(y_i, r_i)| {
            *y_i = *r_i
        });
    }

    fn get_diagonal(&self) -> Vec<c64> {
        self.diag().to_vec()
    }

    fn dimension(&self) -> usize {
        self.shape()[0]
    }

    fn diagonalize(&self, _: Option<usize>, _: Option<EPSType>, _: Option<&Array2<c64>>) 
        -> (Array1<f64>, Array2<c64>) {
        
        self.eigh(ndarray_linalg::UPLO::Upper).unwrap()
    }
}

use std::sync::Arc;

use std::collections::HashMap;

/// Enum representing different types of linear operator compositions
#[derive(Clone)]
pub enum ComposedOperator<T: Clone> {
    /// Base operator (wraps a SolverOperator)
    Base(Arc<dyn SolverOperator<T>>),
    
    /// Sum of two operators: (A + B)x = Ax + Bx
    Add(Box<ComposedOperator<T>>, Box<ComposedOperator<T>>),
    
    /// Sum of multiple operators: (A + B + C + ...)x = Ax + Bx + Cx + ...
    /// This avoids creating a deep tree of binary Add nodes
    Sum(Vec<Box<ComposedOperator<T>>>),

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
    Zero(usize),

    /// Transpose operator
    Transpose(Box<ComposedOperator<T>>),

    /// Single‑photon transition: value * |to⟩⟨from| acting on a `dim`‑dimensional
    /// photon space. Represents a single matrix element at position [to, from].
    SingTrans {
        from: usize,
        to: usize,
        value: T,
        dim: usize,
    },
    
    /// Optimized sum of Kronecker products: Σᵢ Mᵢ ⊗ |to_i⟩⟨from_i|
    /// This efficiently handles the common pattern of light-matter coupling
    /// where exciton matrices are tensored with photon transition operators.
    /// 
    /// State vector layout (standard Kronecker product convention):
    /// x = [ex₀_ph₀, ex₀_ph₁, ..., ex₀_ph_{n-1}, ex₁_ph₀, ex₁_ph₁, ..., ex_{m-1}_ph_{n-1}]
    /// where exciton index varies slowest (outer loop) and photon index varies fastest (inner loop).
    KronSum {
        /// Maps photon transition (from_state, to_state) -> summed exciton matrix
        transition_matrices: HashMap<(usize, usize), Array2<T>>,
        /// Dimension of the exciton space
        exciton_dim: usize,
        /// Dimension of the photon space
        photon_dim: usize,
    },
}

/// Result of photon transition extraction
enum PhotonTransitionResult<T> {
    /// Single transition: (from_state, to_state, value)
    SingleTransition(usize, usize, T),
    /// Zero operator (no transitions)
    Zero,
    /// Complex operator (multiple transitions or not parseable)
    Complex,
}

impl<T> PhotonTransitionResult<T> {
    fn into_option(self) -> Option<(usize, usize, T)> {
        match self {
            PhotonTransitionResult::SingleTransition(from, to, value) => Some((from, to, value)),
            _ => None,
        }
    }
}

impl<T: Clone + num_traits::Zero + LinalgScalar + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + PartialEq> ComposedOperator<T> {
    /// Create from a base operator
    pub fn from_operator<O: SolverOperator<T> + 'static>(op: O) -> Self {
        ComposedOperator::Base(Arc::new(op))
    }

    /// Convenience constructor for the new variant.
    pub fn sing_trans(from: usize, to: usize, value: T, dim: usize) -> Self {
        ComposedOperator::SingTrans { from, to, value, dim }
    }
    
    /// Create a sum of multiple operators
    /// This is more efficient than chaining binary Add operations
    /// Additionally attempts to detect and optimize KronSum patterns
    pub fn sum(ops: Vec<ComposedOperator<T>>) -> Self 
    where
        T: Copy + One + std::fmt::Debug + Norm + ConvertFrom<c64> + ConvertFrom<f64>,
        Array2<T>: SolverOperator<T>,
        ComposedOperator<T>: MatrixOps,
    {
        if ops.is_empty() {
            panic!("Cannot create sum of empty operator list");
        } else if ops.len() == 1 {
            ops.into_iter().next().unwrap()
        } else {
            if let Some(kron_sum) = Self::try_build_kron_sum(&ops) {
                match &kron_sum {
                    ComposedOperator::KronSum { transition_matrices, .. } => {
                        timed_println!(
                            "Optimized {} terms into KronSum with {} unique transitions",
                            ops.len(),
                            transition_matrices.len()
                        );
                    }
                    _ => {}
                }
                return kron_sum;
            }

            timed_println!("Debug: Not using KronSum");
            ComposedOperator::Sum(ops.into_iter().map(Box::new).collect())
        }
    }
    
    /// Extract photon transition from a sparse operator that represents |to⟩⟨from|
    /// Returns Some((from, to, value)) if single transition, None if zero or complex
    /// Optimized for single-excitation operators where only vacuum->mode transitions exist
    fn extract_photon_transition(op: &ComposedOperator<T>) -> Option<(usize, usize, T)>
    where
        T: Copy + Zero + One,
    {
        Self::extract_photon_transition_detailed(op).into_option()
    }
    
    /// Detailed extraction that distinguishes between zero and complex operators
    fn extract_photon_transition_detailed(op: &ComposedOperator<T>) -> PhotonTransitionResult<T>
    where
        T: Copy + Zero + One,
    {
        use PhotonTransitionResult::*;
        
        match op {
            ComposedOperator::SingTrans { from, to, value, .. } => {
                if value.is_zero() {
                    Zero
                } else {
                    SingleTransition(*from, *to, *value)
                }
            }

            ComposedOperator::Transpose(inner) => {
                // For transpose, swap from and to
                match Self::extract_photon_transition_detailed(inner) {
                    SingleTransition(from, to, value) => SingleTransition(to, from, value),
                    Zero => Zero,
                    Complex => Complex,
                }
            }
            
            ComposedOperator::ScalarMul(scalar, inner) => {
                // Scalar multiplication multiplies the value
                if scalar.is_zero() {
                    return Zero;
                }
                match Self::extract_photon_transition_detailed(inner) {
                    SingleTransition(from, to, value) => SingleTransition(from, to, *scalar * value),
                    Zero => Zero,
                    Complex => Complex,
                }
            }
            
            ComposedOperator::Dot(left, right) => {
                // Matrix multiplication is right-to-left: Dot(A, B)x = A(Bx)
                let left_result = Self::extract_photon_transition_detailed(left);
                let right_result = Self::extract_photon_transition_detailed(right);
                
                match (left_result, right_result) {
                    (Zero, _) | (_, Zero) => Zero,
                    (SingleTransition(from_l, to_l, val_l), SingleTransition(from_r, to_r, val_r)) => {
                        if to_r == from_l {
                            // Valid composition
                            SingleTransition(from_r, to_l, val_l * val_r)
                        } else {
                            // Incompatible transitions = zero operator
                            Zero
                        }
                    }
                    _ => Complex,
                }
            }
            
            ComposedOperator::Diagonal(diag) => {
                // Diagonal operators can only represent transitions where from == to
                // Check if exactly one diagonal element is non-zero
                let mut non_zero_idx = None;
                let mut non_zero_val = T::zero();
                let mut count = 0;
                
                for (i, &val) in diag.iter().enumerate() {
                    if !val.is_zero() {
                        non_zero_idx = Some(i);
                        non_zero_val = val;
                        count += 1;
                        if count > 1 {
                            return Complex; // Multiple non-zero elements
                        }
                    }
                }
                
                if count == 1 {
                    let idx = non_zero_idx.unwrap();
                    SingleTransition(idx, idx, non_zero_val)
                } else {
                    Zero // All zeros
                }
            }
            
            ComposedOperator::Identity(_) => {
                // Identity cannot be represented as a single transition
                Complex
            }
            
            ComposedOperator::Zero(_) => {
                // Zero operator has no transitions
                Zero
            }
            
            ComposedOperator::Add(_,_) | ComposedOperator::Sum(_) | 
            ComposedOperator::Kron(_, _) | ComposedOperator::KronSum { .. } => {
                // These are complex operators
                Complex
            }
            
            ComposedOperator::Base(base_op) => {
                // For single-excitation operators, test vacuum state (index 0)
                // and look for non-zero transitions
                let dim = base_op.dimension();
                
                // First, try the vacuum state (most common case for single excitation)
                let mut x_vacuum = vec![T::zero(); dim];
                let mut y_vacuum = vec![T::zero(); dim];
                x_vacuum[0] = T::one();
                base_op.apply(&x_vacuum, &mut y_vacuum);
                
                // Check if vacuum produces exactly one non-zero mode
                let mut element = T::zero();
                let mut to_idx = None;
                let mut count = 0;
                for (j, &val) in y_vacuum.iter().enumerate() {
                    if !val.is_zero() {
                        to_idx = Some(j);
                        count += 1;
                        element = val;
                        if count > 1 {
                            break; // More than one transition from vacuum
                        }
                    }
                }
                
                if count == 1 {
                    // This is a vacuum -> mode transition
                    return SingleTransition(0, to_idx.unwrap(), element);
                }
                
                // If not from vacuum, test each photon mode state
                // For single-excitation, we expect mode -> vacuum transitions
                for i in 1..dim {
                    let mut x = vec![T::zero(); dim];
                    let mut y = vec![T::zero(); dim];
                    x[i] = T::one();
                    base_op.apply(&x, &mut y);
                    
                    // Check if this mode produces exactly one transition
                    let mut transition_found = false;
                    let mut target = None;
                    for (j, &val) in y.iter().enumerate() {
                        if !val.is_zero() {
                            if transition_found {
                                return Complex; // Multiple transitions
                            }
                            transition_found = true;
                            target = Some(j);
                            element = val;
                        }
                    }
                    
                    if transition_found {
                        // Found exactly one transition from state i
                        // Check if other states also have transitions (should not for single-element op)
                        let mut other_transitions = false;
                        for k in (1..dim).filter(|&k| k != i) {
                            let mut x_test = vec![T::zero(); dim];
                            let mut y_test = vec![T::zero(); dim];
                            x_test[k] = T::one();
                            base_op.apply(&x_test, &mut y_test);
                            
                            if y_test.iter().any(|&v| !v.is_zero()) {
                                other_transitions = true;
                                break;
                            }
                        }
                        
                        if !other_transitions {
                            return SingleTransition(i, target.unwrap(), element);
                        } else {
                            return Complex; // Multiple independent transitions
                        }
                    }
                }
                
                // No transitions found = zero operator
                Zero
            }
        }
    }
    
    /// Flatten nested Add operators into a list of terms
    /// Also distributes Kron over Add: Kron(M, Add(a,b)) -> [Kron(M,a), Kron(M,b)]
    /// And handles ScalarMul: ScalarMul(s, Add(a,b)) -> [ScalarMul(s,a), ScalarMul(s,b)]
    fn flatten_adds(op: &ComposedOperator<T>) -> Vec<ComposedOperator<T>>
    where
        T: Copy,
    {
        match op {
            ComposedOperator::Add(left, right) => {
                let mut terms = Self::flatten_adds(left);
                terms.extend(Self::flatten_adds(right));
                terms
            }
            ComposedOperator::Sum(ops) => {
                ops.iter().flat_map(|op| Self::flatten_adds(op)).collect()
            }
            ComposedOperator::ScalarMul(scalar, inner) => {
                // Flatten the inner operator first
                let inner_terms = Self::flatten_adds(inner);
                if inner_terms.len() > 1 {
                    // Distribute scalar: ScalarMul(s, Add(a,b)) -> [ScalarMul(s,a), ScalarMul(s,b)]
                    inner_terms.into_iter()
                        .map(|term| ComposedOperator::ScalarMul(*scalar, Box::new(term)))
                        .collect()
                } else {
                    // No distribution needed
                    vec![op.clone()]
                }
            }
            ComposedOperator::Kron(left, right) => {
                // Check if right side is an Add - if so, distribute
                let right_terms = Self::flatten_adds(right);
                if right_terms.len() > 1 {
                    // Distribute: Kron(M, Add(a,b)) -> [Kron(M,a), Kron(M,b)]
                    right_terms.into_iter()
                        .map(|r_term| ComposedOperator::Kron(left.clone(), Box::new(r_term)))
                        .collect()
                } else {
                    // No distribution needed
                    vec![op.clone()]
                }
            }
            _ => vec![op.clone()],
        }
    }
    
    /// Check if an operator is effectively zero by testing it
    fn is_zero_operator(op: &ComposedOperator<T>) -> bool
    where
        T: Copy + Zero + One + Norm,
    {
        match op {
            ComposedOperator::Zero(_) => true,
            ComposedOperator::ScalarMul(scalar, inner) => {
                scalar.is_zero() || Self::is_zero_operator(inner)
            }
            ComposedOperator::Dot(left, right ) => {
                if Self::is_zero_operator(left) || Self::is_zero_operator(right) {
                    return true
                }
                else {
                    let dim = op.dimension();

                    let mut y = vec![T::zero(); dim];
                    for i in 0..dim {
                        let mut x = vec![T::zero(); dim];
                        x[i] = T::one();
                        op.apply(&x, &mut y);
                        
                        for j in 0..dim {
                            if Norm::norm(y[j]) > f64::EPSILON {return false};
                        }
                    }

                    return true
                }
            }
            ComposedOperator::Kron(left, right) => {
                Self::is_zero_operator(left) || Self::is_zero_operator(right) 
            }
            ComposedOperator::Add(left, right) => {
                Self::is_zero_operator(left) && Self::is_zero_operator(right) 
            }
            ComposedOperator::Diagonal(diags) => {
                for i in 0 .. diags.len() {
                    if Norm::norm(diags[i]) > f64::EPSILON { return false; }
                }

                return true;
            }
            _ => false,
        }
    }
    
   
    /// Fast extraction of matrix from an operator
    /// Uses direct access for Base operators with automatic type conversion, falls back to basis vector application
    fn extract_matrix_fast(op: &ComposedOperator<T>, dim: usize) -> Array2<T>
    where
        T: Copy + Zero + One + LinalgScalar + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        match op {
            // Fast path: Direct extraction from Base operators
            ComposedOperator::Base(base_op) => {
                // Try to downcast Arc<dyn SolverOperator> to concrete types
                let any_ref = base_op.as_ref() as &dyn std::any::Any;
                
                // Try all known scalar types that implement MatrixOps
                // The macro will attempt to downcast to Array2<S> and CsMat<S> for each type S
                if let Some(matrix) = try_downcast_and_convert!(any_ref, dim, c64) {
                    return matrix;
                }
                
                if let Some(matrix) = try_downcast_and_convert!(any_ref, dim, f64) {
                    return matrix;
                }
                
                // Fall through to basis vector extraction for unknown types
                Self::extract_matrix_by_apply(op, dim)
            }
            
            // Fast path: Identity matrix
            ComposedOperator::Identity(_) => Array2::eye(dim),
            
            // Fast path: Zero matrix
            ComposedOperator::Zero(_) => Array2::zeros((dim, dim)),
            
            // Fast path: Diagonal matrix
            ComposedOperator::Diagonal(diag) => Array2::from_diag(&Array1::from_vec(diag.clone())),
            
            // Fast path: Scalar multiplication
            ComposedOperator::ScalarMul(scalar, inner) => {
                let inner_matrix = Self::extract_matrix_fast(inner, dim);
                inner_matrix.mapv(|v| *scalar * v)
            }
            
            // Fast path: Transpose
            ComposedOperator::Transpose(inner) => {
                let inner_matrix = Self::extract_matrix_fast(inner, dim);
                inner_matrix.t().to_owned()
            }
            
            // Fast path: Matrix multiplication (Dot)
            ComposedOperator::Dot(left, right) => {
                // Try to extract both matrices and multiply them
                // This is faster than applying to basis vectors if extraction is cheap
                let left_matrix = Self::extract_matrix_fast(left, dim);
                let right_matrix = Self::extract_matrix_fast(right, dim);
                ndarray::linalg::Dot::dot(&left_matrix, &right_matrix)
            }
            
            // Fast path: Addition
            ComposedOperator::Add(left, right) => {
                let left_matrix = Self::extract_matrix_fast(left, dim);
                let right_matrix = Self::extract_matrix_fast(right, dim);
                &left_matrix + &right_matrix
            }

            ComposedOperator::SingTrans { from, to, value, dim: d } => {
                let mut m = Array2::zeros((*d, *d));
                m[[*to, *from]] = *value;
                m
            }
            
            // For other cases, fall back to basis vector application
            _ => Self::extract_matrix_by_apply(op, dim),
        }
    }
    
    /// Extract matrix by applying operator to basis vectors (slow fallback)
    /// This is only called when we can't directly access the matrix representation
    fn extract_matrix_by_apply(op: &ComposedOperator<T>, dim: usize) -> Array2<T>
    where
        T: Copy + Zero + One,
    {
        let mut matrix = Array2::zeros((dim, dim));
        
        // Reuse these vectors across all columns to avoid repeated allocations
        let mut x = vec![T::zero(); dim];
        let mut y = vec![T::zero(); dim];
        
        for col in 0..dim {
            // Set up basis vector (x is already zeroed from previous iteration)
            x[col] = T::one();
            
            // Apply operator
            op.apply(&x, &mut y);
            
            // Copy column into matrix
            for row in 0..dim {
                matrix[[row, col]] = y[row];
            }
            
            // Reset x for next iteration (y will be overwritten by apply)
            x[col] = T::zero();
        }
        
        matrix
    }
    
    /// Try to detect and build a KronSum from a list of operators (parallelized version)
    /// Returns a potentially optimized operator that may be:
    /// - A single KronSum if all terms match the pattern
    /// - A Sum containing a KronSum and other terms if some match
    /// - The original Sum if no terms match
    fn try_build_kron_sum(ops: &[ComposedOperator<T>]) -> Option<ComposedOperator<T>>
    where
        T: Copy + One + std::fmt::Debug + LinalgScalar + Norm + ConvertFrom<f64> + ConvertFrom<c64> + Send + Sync,
    {
        use std::time::Instant;
        use rayon::prelude::*;
        
        let start = Instant::now();
        
        // First, flatten any Add operators in the list
        let flattened_ops: Vec<ComposedOperator<T>> = ops.iter()
            .flat_map(|op| Self::flatten_adds(op))
            .collect();
        
        timed_println!("  Flattened {} ops into {} terms", ops.len(), flattened_ops.len());
        
        // Structure to hold thread-local results
        struct ThreadLocalResults<T: Clone> {
            transition_matrices: HashMap<(usize, usize), Array2<T>>,
            matrix_cache: HashMap<usize, Array2<T>>,
            non_kronsum_terms: Vec<ComposedOperator<T>>,
            exciton_dim: Option<usize>,
            photon_dim: Option<usize>,
            n_valid: usize,
            n_zero: usize,
            n_other: usize,
            n_not_kron: usize,
            n_no_transition: usize,
            n_dim_mismatch: usize,
        }
        
        impl<T: Clone> ThreadLocalResults<T> {
            fn new() -> Self {
                ThreadLocalResults {
                    transition_matrices: HashMap::new(),
                    matrix_cache: HashMap::new(),
                    non_kronsum_terms: Vec::new(),
                    exciton_dim: None,
                    photon_dim: None,
                    n_valid: 0,
                    n_zero: 0,
                    n_other: 0,
                    n_not_kron: 0,
                    n_no_transition: 0,
                    n_dim_mismatch: 0,
                }
            }
        }
        
        // Process operators in parallel using fold-reduce pattern
        let results = flattened_ops.par_iter()
            .fold(
                || ThreadLocalResults::new(),
                |mut local, op| {
                    // Skip zero operators
                    if Self::is_zero_operator(op) {
                        local.n_zero += 1;
                        return local;
                    }
                    
                    // Unwrap ScalarMul to get to the underlying operator
                    let (scalar, inner_op) = match op {
                        ComposedOperator::ScalarMul(s, inner) => (Some(*s), inner.as_ref()),
                        _ => (None, op),
                    };
                    
                    // Check if this is a Kron(exciton_op, photon_op) pattern
                    if let ComposedOperator::Kron(left, right) = inner_op {
                        // Check if right operator is a photon transition
                        let transition = Self::extract_photon_transition(right);
                        
                        match transition {
                            Some((from, to, ph_value)) => {
                                // Extract the exciton matrix from the left operator
                                let ex_dim = left.dimension();
                                let ph_dim = right.dimension();
                                
                                // Verify dimensions are consistent with local state
                                let dims_ok = if let Some(stored_ex_dim) = local.exciton_dim {
                                    stored_ex_dim == ex_dim
                                } else {
                                    local.exciton_dim = Some(ex_dim);
                                    true
                                } && if let Some(stored_ph_dim) = local.photon_dim {
                                    stored_ph_dim == ph_dim
                                } else {
                                    local.photon_dim = Some(ph_dim);
                                    true
                                };
                                
                                if !dims_ok {
                                    // Inconsistent dimensions - treat as non-KronSum term
                                    local.non_kronsum_terms.push(op.clone());
                                    local.n_other += 1;
                                    local.n_dim_mismatch += 1;
                                    return local;
                                }
                                
                                // Extract the exciton matrix - use local cache
                                let cache_key = left.as_ref() as *const ComposedOperator<T> as usize;
                                let mut ex_matrix = if let Some(cached) = local.matrix_cache.get(&cache_key) {
                                    cached.clone()
                                } else {
                                    let matrix = Self::extract_matrix_fast(left.as_ref(), ex_dim);
                                    local.matrix_cache.insert(cache_key, matrix.clone());
                                    matrix
                                };
                                
                                // Apply both the outer scalar (if present) and the photon value
                                let combined_scalar = if let Some(s) = scalar {
                                    s * ph_value
                                } else {
                                    ph_value
                                };

                                // Only scale if the combined scalar is not 1
                                if combined_scalar != T::one() {
                                    ex_matrix = ex_matrix.mapv(|v| combined_scalar * v);
                                }
                                // Add or accumulate to the transition matrix
                                local.transition_matrices.entry((from, to))
                                    .and_modify(|existing| {
                                        *existing = &*existing + &ex_matrix;
                                    })
                                    .or_insert(ex_matrix);
                                
                                local.n_valid += 1;
                            }
                            None => {
                                local.non_kronsum_terms.push(op.clone());
                                local.n_other += 1;
                                local.n_no_transition += 1;
                            }
                        }
                    } else {
                        // Not a Kron pattern - keep as separate term
                        local.non_kronsum_terms.push(op.clone());
                        local.n_other += 1;
                        local.n_not_kron += 1;
                    }
                    
                    local
                }
            )
            .reduce(
                || ThreadLocalResults::new(),
                |mut acc, local| {
                    // Merge transition matrices
                    for ((from, to), matrix) in local.transition_matrices {
                        acc.transition_matrices.entry((from, to))
                            .and_modify(|existing| {
                                *existing = &*existing + &matrix;
                            })
                            .or_insert(matrix);
                    }
                    
                    // Merge matrix cache
                    acc.matrix_cache.extend(local.matrix_cache);
                    
                    // Merge non-kronsum terms
                    acc.non_kronsum_terms.extend(local.non_kronsum_terms);
                    
                    // Merge dimensions (take first non-None)
                    if acc.exciton_dim.is_none() {
                        acc.exciton_dim = local.exciton_dim;
                    }
                    if acc.photon_dim.is_none() {
                        acc.photon_dim = local.photon_dim;
                    }
                    
                    // Accumulate counters
                    acc.n_valid += local.n_valid;
                    acc.n_zero += local.n_zero;
                    acc.n_other += local.n_other;
                    acc.n_not_kron += local.n_not_kron;
                    acc.n_no_transition += local.n_no_transition;
                    acc.n_dim_mismatch += local.n_dim_mismatch;
                    
                    acc
                }
            );
        
        let ThreadLocalResults {
            transition_matrices,
            matrix_cache,
            non_kronsum_terms,
            exciton_dim,
            photon_dim,
            n_valid,
            n_zero,
            n_other,
            n_not_kron,
            n_no_transition,
            n_dim_mismatch,
        } = results;
        
        let elapsed = start.elapsed();
        timed_println!("  KronSum pattern detection took {:?}", elapsed);
        timed_println!("  Pattern matching: {} KronSum terms, {} other terms, {} zero terms", 
            n_valid, n_other, n_zero);
        let cache_hits = if n_valid > matrix_cache.len() { n_valid - matrix_cache.len() } else { 0 };
        timed_println!("  Matrix cache: {} unique matrices extracted, {} cache hits",
            matrix_cache.len(), cache_hits);
        if n_other > 0 {
            timed_println!("    Other terms breakdown: {} not Kron, {} no transition, {} dim mismatch",
                n_not_kron, n_no_transition, n_dim_mismatch);
        }
        
        // Decide what to return based on what we found
        if !transition_matrices.is_empty() {
            let kronsum = ComposedOperator::KronSum {
                transition_matrices,
                exciton_dim: exciton_dim.unwrap(),
                photon_dim: photon_dim.unwrap(),
            };
            
            if non_kronsum_terms.is_empty() {
                // All terms matched the KronSum pattern!
                timed_println!("  ✓ Pure KronSum with {} unique transitions", 
                    if let ComposedOperator::KronSum { ref transition_matrices, .. } = kronsum {
                        transition_matrices.len()
                    } else { 0 });
                Some(kronsum)
            } else {
                // Hybrid: some terms matched, some didn't
                timed_println!("  ✓ Hybrid: KronSum with {} transitions + {} other terms",
                    if let ComposedOperator::KronSum { ref transition_matrices, .. } = kronsum {
                        transition_matrices.len()
                    } else { 0 },
                    non_kronsum_terms.len());
                
                // Create a Sum with the KronSum and the other terms
                let mut all_terms = vec![Box::new(kronsum)];
                all_terms.extend(non_kronsum_terms.into_iter().map(Box::new));
                Some(ComposedOperator::Sum(all_terms))
            }
        } else {
            // No terms matched the KronSum pattern
            timed_println!("  ✗ KronSum not applied: no valid KronSum terms found");
            None
        }
    }
        
    /// Check if this operator is a KronSum variant
    pub fn is_kron_sum(&self) -> bool {
        matches!(self, ComposedOperator::KronSum { .. })
    }
    
    /// Get information about KronSum if this is a KronSum operator
    pub fn get_kron_sum_info(&self) -> Option<(usize, usize, usize)> {
        if let ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } = self {
            Some((transition_matrices.len(), *exciton_dim, *photon_dim))
        } else {
            None
        }
    }
    
    /// Get the dimension of the operator
    pub fn dimension(&self) -> usize {
        match self {
            ComposedOperator::Base(op) => op.dimension(),
            ComposedOperator::Add(a, _) => a.dimension(),
            ComposedOperator::Sum(ops) => ops.first().map(|op| op.dimension()).unwrap_or(0),
            ComposedOperator::Dot(a, _) => a.dimension(),
            ComposedOperator::ScalarMul(_, a) => a.dimension(),
            ComposedOperator::Kron(a, b) => a.dimension() * b.dimension(),
            ComposedOperator::Diagonal(diag) => diag.len(),
            ComposedOperator::Identity(n) => *n,
            ComposedOperator::Zero(n) => *n,
            ComposedOperator::Transpose(a) => a.dimension(),
            ComposedOperator::KronSum { exciton_dim, photon_dim, .. } => exciton_dim * photon_dim,
            ComposedOperator::SingTrans { from: _, to: _, value: _, dim } => *dim,
        }
    }

    pub fn apply_transpose(&self, x: &[T], y: &mut [T]) 
    where
        T: Copy + One + LinalgScalar,
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
            
            ComposedOperator::Sum(ops) => {
                // y = A₁ᵀx + A₂ᵀx + ... + Aₙᵀx
                if ops.is_empty() {
                    y.fill(T::zero());
                    return;
                }
                
                ops[0].apply_transpose(x, y);
                
                if ops.len() > 1 {
                    let mut temp = vec![T::zero(); y.len()];
                    for op in &ops[1..] {
                        temp.fill(T::zero());
                        op.apply_transpose(x, &mut temp);
                        for i in 0..y.len() {
                            y[i] = y[i] + temp[i];
                        }
                    }
                }
            }
            
            ComposedOperator::Dot(a, b) => {
                // (AB)^T = B^T A^T, so (AB)^T x = B^T(A^T x)
                let mut temp = vec![T::zero(); y.len()];
                a.apply_transpose(x, &mut temp);  // temp = A^T x
                b.apply_transpose(&temp, y);      // y = B^T temp = B^T A^T x
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

                // Zero output first since we write to slices
                y.fill(T::zero());

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
            
            ComposedOperator::Zero(_) => {
                y.fill(T::zero());
            }

            ComposedOperator::Transpose(a) => {
                a.apply(x, y);
            }

            ComposedOperator::SingTrans { from, to, value, dim: _ } => {
                y.fill(T::zero());
                y[*from] = x[*to] * (*value);
            }
            
            ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } => {
                // Apply transpose: y = Σᵢ (Mᵢᵀ ⊗ |from_i⟩⟨to_i|) x
                // For transpose, swap from and to in the photon transition
                
                y.fill(T::zero());
                
                for (&(from, to), matrix) in transition_matrices {
                    // For transpose: read from photon state 'to', write to photon state 'from'
                    // and apply transposed exciton matrix
                    
                    // Gather input: x_block[ex] = x[ex * photon_dim + to]
                    let mut x_block = vec![T::zero(); *exciton_dim];
                    for ex in 0..*exciton_dim {
                        x_block[ex] = x[ex * photon_dim + to];
                    }
                    
                    // Apply the transposed exciton matrix: result = Mᵀ * x_block
                    let x_arr = Array1::from_iter(x_block.iter().copied());
                    let result = ndarray::linalg::Dot::dot(&matrix.t(), &x_arr);
                    
                    // Scatter output: y[ex * photon_dim + from] += result[ex]
                    for (ex, &val) in result.iter().enumerate() {
                        y[ex * photon_dim + from] = y[ex * photon_dim + from] + val;
                    }
                }
            }
        }
    }
    
    
    /// Apply the composed operator to a vector
    pub fn apply(&self, x: &[T], y: &mut [T]) 
    where
        T: Copy + One + LinalgScalar,
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
            
            ComposedOperator::Sum(ops) => {
                // y = A₁x + A₂x + ... + Aₙx
                // Apply first operator directly to y, then accumulate the rest
                if ops.is_empty() {
                    y.fill(T::zero());
                    return;
                }
                
                ops[0].apply(x, y);
                
                if ops.len() > 1 {
                    let mut temp = vec![T::zero(); y.len()];
                    for op in &ops[1..] {
                        temp.fill(T::zero());
                        op.apply(x, &mut temp);
                        for i in 0..y.len() {
                            y[i] = y[i] + temp[i];
                        }
                    }
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

                // Zero output first since we write to slices
                y.fill(T::zero());

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
            
            ComposedOperator::Zero(_) => {
                y.fill(T::zero());
            }

            ComposedOperator::Transpose(a) => {
                a.apply_transpose(x, y);
            }

            ComposedOperator::SingTrans { from, to, value, dim: _ } => {
                y.fill(T::zero());
                y[*to] = x[*from] * (*value);
            }
            
            ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } => {
                // Optimized application: y = Σᵢ (Mᵢ ⊗ |to_i⟩⟨from_i|) x
                // Standard Kronecker product layout: x = [ex₀_ph₀, ex₀_ph₁, ..., ex₁_ph₀, ex₁_ph₁, ...]
                // Exciton index varies slowest (outer loop), photon index varies fastest (inner loop)
                
                y.fill(T::zero());
                
                for (&(from, to), matrix) in transition_matrices {
                    // For each exciton state, extract the component in photon state 'from'
                    // and write to photon state 'to' after applying the exciton matrix
                    
                    // Gather input: x_block[ex] = x[ex * photon_dim + from]
                    let mut x_block = vec![T::zero(); *exciton_dim];
                    for ex in 0..*exciton_dim {
                        x_block[ex] = x[ex * photon_dim + from];
                    }
                    
                    // Apply the exciton matrix: result = M * x_block
                    let x_arr = Array1::from_iter(x_block.iter().copied());
                    let result = ndarray::linalg::Dot::dot(matrix, &x_arr);
                    
                    // Scatter output: y[ex * photon_dim + to] += result[ex]
                    for (ex, &val) in result.iter().enumerate() {
                        y[ex * photon_dim + to] = y[ex * photon_dim + to] + val;
                    }
                }
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
            
            ComposedOperator::Sum(ops) => {
                if ops.is_empty() {
                    return vec![];
                }
                
                let mut diag = ops[0].get_diagonal();
                for op in &ops[1..] {
                    let op_diag = op.get_diagonal();
                    for (i, &val) in op_diag.iter().enumerate() {
                        diag[i] = diag[i] + val;
                    }
                }
                diag
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
            
            ComposedOperator::Zero(n) => vec![T::zero(); *n],

            ComposedOperator::Transpose(a) => a.get_diagonal(),

            ComposedOperator::SingTrans { from, to, value, dim } => {
                let mut diag = vec![T::zero(); *dim];
                if from == to {
                    diag[*from] = *value;
                }
                diag
            }
            
            ComposedOperator::KronSum { transition_matrices, exciton_dim, photon_dim } => {
                // Diagonal elements come from transitions where from == to
                // Standard layout: diag[ex * photon_dim + ph] for exciton state ex, photon state ph
                let mut diag = vec![T::zero(); exciton_dim * photon_dim];
                
                for (&(from, to), matrix) in transition_matrices {
                    if from == to {
                        // This is a diagonal block contribution
                        // For each exciton state, add the diagonal element of the matrix
                        for ex in 0..*exciton_dim {
                            let idx = ex * photon_dim + from;
                            diag[idx] = diag[idx] + matrix[[ex, ex]];
                        }
                    }
                }
                
                diag
            }
        }
    }
}

// Implement SolverOperator for ComposedOperator
impl SolverOperator<c64> for ComposedOperator<c64> {
    fn apply(&self, x: &[c64], y: &mut [c64]) {
        ComposedOperator::apply(self, x, y)
    }

    fn apply_transpose(&self, x: &[c64], y: &mut [c64]) {
        ComposedOperator::apply_transpose(self, x, y)
    }
    
    fn get_diagonal(&self) -> Vec<c64> {
        ComposedOperator::get_diagonal(self)
    }
    
    fn dimension(&self) -> usize {
        ComposedOperator::dimension(self)
    }

    fn diagonalize(&self, n_eig: Option<usize>, solve_type: Option<EPSType>, initial_guess: Option<&Array2<c64>>) -> (Array1<f64>, Array2<c64>) {
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
                ComposedOperator::Zero(shape.0)
            }
            
            fn kron(&self, other: &Self) -> Self {
                ComposedOperator::Kron(Box::new(self.clone()), Box::new(other.clone()))
            }
            
            fn add(&self, other: &Self) -> Self {
                ComposedOperator::Add(Box::new(self.clone()), Box::new(other.clone()))
            }
            
            fn scale<T>(&self, scalar: T) -> Self 
            where
                T: Clone + NumCast + Zero,
                Self::Scalar: ConvertFrom<T>
            {
                ComposedOperator::ScalarMul(<Self::Scalar as ConvertFrom<T>>::convert_from(scalar).unwrap(), Box::new(self.clone()))
            }
            
            fn shape(&self) -> (usize, usize) {
                let n = self.dimension();
                (n, n)
            }
            
            fn from_diag<T>(diags: &[T]) -> Self 
            where
                T: Clone + NumCast + Zero + Copy,
                Self::Scalar: ConvertFrom<T>
            {
                ComposedOperator::Diagonal(diags.iter().map(|val| {
                    <Self::Scalar as ConvertFrom<T>>::convert_from(*val)
                        .expect("Failed to convert diagonal value")
                }).collect::<Vec<Self::Scalar>>())
            }
            
            fn from_trimat(trimat: &TriMat<Self::Scalar>) -> Self {

                let nnz = trimat.nnz();
                let threshold = 100;
                let dim = trimat.shape().0;

                if nnz == 0 {
                    return ComposedOperator::Zero(dim)
                }

                if nnz < threshold {
                    // Collect a Vec of elementary operators
                    let mut terms: Vec<Box<ComposedOperator<Self::Scalar>>> = Vec::with_capacity(nnz);

                    // `triplet_iter` yields ((row, col), value) for each non‑zero entry
                    for (val, (row, col)) in trimat.triplet_iter() {
                        // In the photon‑space convention used elsewhere,
                        // a matrix element |to⟩⟨from| corresponds to (row = to,
                        // col = from).  The `SingTrans` variant stores the value directly.
                        let sing = ComposedOperator::sing_trans(col, row, val.clone(), dim);
                        terms.push(Box::new(sing));
                    }

                    if terms.len() == 1 {
                        return *terms[0].clone()
                    }

                    return ComposedOperator::Sum(terms);
                }



                // Convert TriMat to sparse and wrap as an operator
                let csmat = trimat.to_csr();
                ComposedOperator::from_operator(csmat)
            }
            
            fn add_trimat(&mut self, trimat: &TriMat<Self::Scalar>) {
                *self = self.add(&Self::from_trimat(trimat))
            }

            fn from_dense<T>(dense: &Array2<T>) -> Self
                    where T: Clone + NumCast + Zero + 'static + Copy,Self::Scalar: ConvertFrom<T> {
                
                // For other types, convert element-wise to Self::Scalar
                let (rows, cols) = (dense.nrows(), dense.ncols());
                let mut new_dense = Array2::zeros((rows,cols));
                for i in 0..rows {
                    for j in 0..cols {
                        let val = dense[[i, j]];
                        if let Some(converted) = <Self::Scalar as ConvertFrom<T>>::convert_from(val) {
                            new_dense[[i, j]] = converted;
                        }
                    }
                }
                ComposedOperator::from_operator(new_dense)
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
            
            fn sum(matrices: Vec<Self>) -> Self {
                ComposedOperator::sum(matrices)
            }
            
            fn build_kron_sum(terms: Vec<(Self, Self)>, exciton_dim: usize, photon_dim: usize) -> Self 
            where 
                Self: Sized,
            {
                // Use the optimized implementation defined below
                <ComposedOperator<$scalar>>::build_kron_sum(terms, exciton_dim, photon_dim)
            }
        }
    };
}

impl_matrix_ops_lin_op!(c64);

// Implement std::ops::Add for ComposedOperator to enable + syntax
impl<T> std::ops::Add for ComposedOperator<T> 
    where T: Clone, ComposedOperator<T>: MatrixOps
{
    type Output = Self;
    
    fn add(self, other: Self) -> Self::Output {
        MatrixOps::add(&self, &other)
    }
}

impl<T> std::ops::Add<&ComposedOperator<T>> for ComposedOperator<T> 
    where T: Clone, ComposedOperator<T>: MatrixOps
{
    type Output = Self;
    
    fn add(self, other: &ComposedOperator<T>) -> Self::Output {
        MatrixOps::add(&self, other)
    }
}

impl<T> std::ops::Add<ComposedOperator<T>> for &ComposedOperator<T> 
    where T: Clone, ComposedOperator<T>: MatrixOps
{
    type Output = ComposedOperator<T>;
    
    fn add(self, other: ComposedOperator<T>) -> Self::Output {
        MatrixOps::add(self, &other)
    }
}

impl<T> std::ops::Add for &ComposedOperator<T> 
    where T: Clone, ComposedOperator<T>: MatrixOps
{
    type Output = ComposedOperator<T>;
    
    fn add(self, other: &ComposedOperator<T>) -> Self::Output {
        MatrixOps::add(self, other)
    }
}

impl<T> ComposedOperator<T>
where
    T: Clone
        + num_traits::Zero
        + LinalgScalar
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::fmt::Debug
        + ConvertFrom<c64>
        + ConvertFrom<f64>
        + Send
        + Sync
        + PartialEq,
        
    ComposedOperator<T>: MatrixOps
{
    pub fn build_kron_sum(terms: Vec<(Self, Self)>, exciton_dim: usize, photon_dim: usize) -> Self 
    where 
        Self: Sized,
        T: Norm,
        Array2<T>: SolverOperator<T>,

    {
        // Try to extract photon transitions from all terms
        let mut kron_sum_terms = Vec::new();
        let mut fallback_terms = Vec::new();
        let mut n_zero = 0;
        
        for (ex_op, ph_op) in terms {
            let result = Self::extract_photon_transition_detailed(&ph_op);
            match result {
                PhotonTransitionResult::SingleTransition(from, to, value) => {
                    // This term can be optimized into KronSum
                    kron_sum_terms.push((ex_op, from, to, value));
                }
                PhotonTransitionResult::Zero => {
                    // Skip zero operators - they don't contribute
                    n_zero += 1;
                }
                PhotonTransitionResult::Complex => {
                    // Fall back to normal kron for this term
                    fallback_terms.push(ex_op.kron(&ph_op));
                }
            }
        }
        
        if n_zero > 0 {
            timed_println!("Skipped {} zero operators", n_zero);
        }
        
        // If we can optimize at least some terms, use KronSum
        if !kron_sum_terms.is_empty() {
            timed_println!(
                "Optimizing {}/{} terms into KronSum (fallback: {})",
                kron_sum_terms.len(),
                kron_sum_terms.len() + fallback_terms.len(),
                fallback_terms.len()
            );
            
            let kron_sum = Self::build_kron_sum_from_terms(
                kron_sum_terms, 
                exciton_dim, 
                photon_dim
            );
            
            if fallback_terms.is_empty() {
                // All terms were optimized
                kron_sum
            } else {
                // Mix of optimized and fallback terms
                fallback_terms.push(kron_sum);
                Self::sum(fallback_terms)
            }
        } else {
            // No terms could be optimized, use normal path
            timed_println!(
                "No KronSum optimization possible for {} terms",
                fallback_terms.len()
            );
            Self::sum(fallback_terms)
        }
    }
    
    
    /// Build a `KronSum` *without* ever constructing the intermediate
    /// `Kron` products.  The caller supplies an iterator of
    /// `(exciton_operator, from_photon, to_photon)`.  All exciton
    /// operators are assumed to act on the **same** exciton dimension.
    ///
    /// This is the fast‑path that replaces the old `ComposedOperator::sum`
    /// when every term is of the form `Kron(ex, SingTrans{from,to,dim})`.
    pub fn build_kron_sum_from_terms<I>(terms: I, exciton_dim: usize, photon_dim: usize) -> Self
    where
        I: IntoIterator<Item = (Self, usize, usize, T)>, 
    {
        let mut map: HashMap<(usize, usize), Self> = HashMap::new();

        // Step 1: Build the HashMap (sequential - fast enough)
        for (ex_op, from, to, ph_value) in terms {
            let scaled_ex_op = if ph_value == T::one() {
                ex_op
            } else {
                ComposedOperator::ScalarMul(ph_value, Box::new(ex_op))
            };
            
            map.entry((from, to))
                .and_modify(|existing| {
                    *existing = existing.clone() + scaled_ex_op.clone();
                })
                .or_insert(scaled_ex_op);
        }

        // Step 2: Convert to matrices IN PARALLEL
        use rayon::prelude::*;
        
        let transition_matrices: HashMap<(usize, usize), Array2<T>> = map
            .into_par_iter()  // ← Parallel iterator!
            .map(|((from, to), op)| {
                // Each thread extracts its own matrix independently
                let mat = Self::extract_matrix_fast(&op, exciton_dim);
                ((from, to), mat)
            })
            .collect();

        ComposedOperator::KronSum {
            transition_matrices,
            exciton_dim,
            photon_dim,
        }
    }
}