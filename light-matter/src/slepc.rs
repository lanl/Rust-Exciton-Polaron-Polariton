use crate::utils::{SolverOperator};
use ndarray::{Array1, Array2};
use std::ffi::{c_int, c_double, c_void};
use rust_slepc_bindings::{
    Eigensolver, EPS_WHICH_SMALLEST_REAL,
    Mat, PETSC_COMM_WORLD,
    create_complex_vector as create_vector, destroy_vector, get_vector_array, EPSType, PetscError,
    EPSGDSetBlockSize, EPSJDSetBlockSize
};
use crate::timed_println;
use num_complex::Complex64 as c64;
use num_traits::Zero;

// PETSc/SLEPc function declarations needed for shell matrix
unsafe extern "C" {
    // Matrix creation and manipulation
    fn MatCreateShell(comm: rust_slepc_bindings::MPIComm, m: c_int, n: c_int, M: c_int, N: c_int, ctx: *mut c_void, A: *mut Mat) -> c_int;
    fn MatShellSetOperation(mat: Mat, op: c_int, f: *mut c_void) -> c_int;
    fn MatShellGetContext(mat: Mat, ctx: *mut *mut c_void) -> c_int;
    fn MatDestroy(A: *mut Mat) -> c_int;
    
    // Vector operations
    fn VecGetArray(v: rust_slepc_bindings::Vec, a: *mut *mut c_double) -> c_int;
    fn VecRestoreArray(v: rust_slepc_bindings::Vec, a: *mut *mut c_double) -> c_int;
}

// Constants needed for this implementation
const MATOP_MULT: c_int = 3; // Matrix-vector product operation

// Callback function for matrix-vector product (complex version)
extern "C" fn mv_product_complex<T: SolverOperator<c64>>(
    mat: Mat, 
    x: rust_slepc_bindings::Vec, 
    y: rust_slepc_bindings::Vec
) -> c_int {
    unsafe {
        // Extract the operator from the context
        let mut ctx_ptr: *mut c_void = std::ptr::null_mut();
        MatShellGetContext(mat, &mut ctx_ptr);
        let op = &*(ctx_ptr as *const T);
        
        // Get arrays from vectors (PETSc stores complex as interleaved real/imag)
        let mut x_array: *mut c_double = std::ptr::null_mut();
        let mut y_array: *mut c_double = std::ptr::null_mut();
        VecGetArray(x, &mut x_array);
        VecGetArray(y, &mut y_array);
        
        // Convert to Rust slices
        let n = op.dimension();
        // PETSc complex vectors have 2*n doubles (real and imaginary parts interleaved)
        let x_raw = std::slice::from_raw_parts(x_array, 2 * n);
        let y_raw = std::slice::from_raw_parts_mut(y_array, 2 * n);
        
        // Convert from interleaved format to Complex64
        let mut x_complex = vec![c64::new(0.0, 0.0); n];
        for i in 0..n {
            x_complex[i] = c64::new(x_raw[2*i], x_raw[2*i + 1]);
        }
        
        // Apply the operator
        let mut y_complex = vec![c64::new(0.0, 0.0); n];
        op.apply(&x_complex, &mut y_complex);
        
        // Convert back to interleaved format
        for i in 0..n {
            y_raw[2*i] = y_complex[i].re;
            y_raw[2*i + 1] = y_complex[i].im;
        }
        
        // Restore arrays
        VecRestoreArray(x, &mut x_array);
        VecRestoreArray(y, &mut y_array);
        
        0 // PETSC_SUCCESS
    }
}

/// Diagonalize using SLEPc with complex Hermitian matrices
/// 
/// # Requirements
/// - PETSc/SLEPc must be compiled with --with-scalar-type=complex
/// - The operator must implement SolverOperator<Complex64>
/// - For Hermitian matrices, eigenvalues will be real
/// 
/// # Arguments
/// * `op` - The operator to diagonalize (must be Hermitian)
/// * `n_eig` - Number of eigenvalues to compute
/// * `block_size` - Block size for block methods
/// * `solve_type` - Solver algorithm to use
/// * `initial_guess` - Optional initial guess vectors (real values, will be converted to complex)
/// 
/// # Returns
/// * Tuple of (eigenvalues, eigenvectors) where eigenvectors are the real parts
///   For Hermitian matrices, eigenvectors can be chosen real (up to a global phase)
pub fn diagonalize_slepc<T: SolverOperator<c64>>(
    op: &T, 
    n_eig: usize,
    block_size: usize, 
    solve_type: Option<EPSType>,
    initial_guess: Option<&Array2<c64>>
) -> (Array1<f64>, Array2<c64>) {
    // Create a shell matrix for the operator
    let mut a = std::ptr::null_mut();
    let n = op.dimension() as c_int;
    
    timed_println!("Creating shell matrix of dimension {} (complex Hermitian)", n);
    
    unsafe {
        // Create shell matrix
        let op_ptr = op as *const T as *const c_void as *mut c_void;
        MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, op_ptr, &mut a);
        
        // Set the matrix-vector product operation
        MatShellSetOperation(a, MATOP_MULT, mv_product_complex::<T> as *mut c_void);
    }
    
    // Create and configure the eigensolver
    let mut solver = Eigensolver::new().expect("Failed to create eigensolver");
    solver.set_operators(a, None).expect("Failed to set operators");
    
    // Set dimensions with proper values
    solver.set_dimensions(n_eig, None, None).expect("Failed to set dimensions");

    // For Davidson, set block size
    match solve_type {
        Some(EPSType::JacobiDavidson) => unsafe {
            EPSJDSetBlockSize(solver.eps, block_size as c_int);
        },
        Some(EPSType::GeneralizedDavidson) => unsafe {
            EPSGDSetBlockSize(solver.eps, block_size as c_int);
        },
        Some(EPSType::KrylovSchur) => {
            let mpd = Some(block_size * 4);
            let ncv = Some(mpd.unwrap() + n_eig);
            solver.set_dimensions(n_eig, ncv, Some(block_size * 4)).expect("Failed to set dimensions");
        }
        _ => {}
    }
    
    solver.set_which_eigenpairs(EPS_WHICH_SMALLEST_REAL).expect("Failed to set which eigenpairs");
    
    // Set as Hermitian problem type for complex Hermitian matrices
    solver.set_problem_type(rust_slepc_bindings::EPSProblemType::Hermitian).expect("Failed to set problem type");
    
    // Set more lenient tolerances and more iterations
    solver.set_tolerances(1e-10, 10000).expect("Failed to set tolerances");
    
    // Set the solver type
    let solver_type = solve_type.unwrap_or(EPSType::KrylovSchur);
    timed_println!("Using solver type: {:?}", solver_type);
    solver.set_type(solver_type).expect("Failed to set eigensolver");
    
    // Apply options to ensure DS is properly initialized
    unsafe {
        unsafe extern "C" {
            fn EPSSetFromOptions(eps: rust_slepc_bindings::EPS) -> c_int;
            fn EPSSetUp(eps: rust_slepc_bindings::EPS) -> c_int;
            fn MatAssemblyBegin(mat: Mat, type_: c_int) -> c_int;
            fn MatAssemblyEnd(mat: Mat, type_: c_int) -> c_int;
        }
        
        // Assemble the shell matrix before using it
        const MAT_FINAL_ASSEMBLY: c_int = 0;
        let assembly_result = MatAssemblyBegin(a, MAT_FINAL_ASSEMBLY);
        if assembly_result != 0 {
            timed_println!("Warning: MatAssemblyBegin returned error code {}", assembly_result);
        }
        let assembly_result = MatAssemblyEnd(a, MAT_FINAL_ASSEMBLY);
        if assembly_result != 0 {
            timed_println!("Warning: MatAssemblyEnd returned error code {}", assembly_result);
        }
        
        // Apply options and set up the solver
        // Note: This may print error messages about missing optional features (BV types, etc.)
        // but these are non-fatal and SLEPc will fall back to available alternatives
        let opts_result = EPSSetFromOptions(solver.eps);
        if opts_result != 0 {
            timed_println!("Warning: EPSSetFromOptions returned error code {}", opts_result);
        }
        let setup_result = EPSSetUp(solver.eps);
        if setup_result != 0 {
            timed_println!("Warning: EPSSetUp returned error code {}", setup_result);
        }
    }
    
    // Set initial guess if provided
    if let Some(initial_vecs) = initial_guess {
        timed_println!("Using provided initial guess with {} vectors", initial_vecs.ncols());
        unsafe {
            // Convert Array2 to Vec<Vec>
            let petsc_vecs = match create_petsc_vecs_from_array2(initial_vecs) {
                Ok(vecs) => vecs,
                Err(e) => {
                    timed_println!("Error creating PETSc vectors from initial guess: {:?}", e);
                    Vec::new()
                }
            };
            
            if !petsc_vecs.is_empty() {
                // Set the initial space
                match solver.set_initial_space(&petsc_vecs) {
                    Ok(_) => timed_println!("Successfully set initial space with {} vectors", petsc_vecs.len()),
                    Err(e) => {
                        timed_println!("ERROR: Failed to set initial space: {:?}", e);
                        timed_println!("Initial guess will NOT be used - solver will start from scratch");
                    }
                }
                
                // Clean up the vectors
                for v in petsc_vecs {
                    let _ = destroy_vector(v);
                }
            } else {
                timed_println!("WARNING: No PETSc vectors created from initial guess");
            }
        }
    } else {
        timed_println!("No initial guess provided - starting from scratch");
    }
    
    timed_println!("Starting eigenvalue computation...");
    
    // Solve the eigenvalue problem
    let solve_result = unsafe { rust_slepc_bindings::EPSSolve(solver.eps) };
    if solve_result != 0 {
        timed_println!("Warning: EPSSolve returned error code {}", solve_result);
    }
    
    // Get the eigenvalues
    let nconv = solver.get_converged().expect("Failed to get converged eigenvalues");
    timed_println!("SLEPc converged {} out of {} requested eigenvalues", nconv, n_eig);
    
    // Get iteration count
    let iterations = solver.get_iteration_number().expect("Failed to get iteration count");
    timed_println!("Solver took {} iterations", iterations);
    
    // Always create arrays of the requested size
    let n_eig_actual = std::cmp::min(n_eig, nconv);
    let dimension = op.dimension();
    let mut eigenvalues = Array1::zeros(n_eig);
    let mut eigenvectors = Array2::zeros((dimension, n_eig_actual));
    
    // Extract eigenvalues and eigenvectors for the converged ones
    for i in 0..n_eig_actual {       

        // PETSc complex vectors store data as interleaved real/imag
        let mut vec_temp = vec![0.0; 2 * dimension];

        // timed_println!("Creating dummy vector");
        // Create vectors for eigenvectors (complex vectors need 2x space)
        let vr = unsafe {create_vector(dimension).expect("Failed to create vector")};
        // timed_println!("Made dummy vector");

        // Get eigenvector
        unsafe {
            // Get eigenvalue (should be real for Hermitian matrices)
            let re = solver.get_eigenvalue(i).expect("Failed to get eigenvalue");
            eigenvalues[i] = re;
            // timed_println!("Got eigenvalue {}", i);

            solver.get_eigenpair(i, vr).expect("Failed to get eigenvector");

            // CRITICAL: get_vector_array expects the slice to be the right size
            // For complex vectors, PETSc uses 2*dimension doubles
            // timed_println!("Getting {}th eigenvector", {i});
            let result = get_vector_array(vr, &mut vec_temp);
            if let Err(e) = result {
                timed_println!("Error getting eigenvector {}: {:?}", i, e);
                continue;
            }
        }
        
        // timed_println!("Got {}th eigenvector", i);
        // Extract complex eigenvector components
        for j in 0..dimension {
            let re_part = vec_temp[2*j];
            let im_part = vec_temp[2*j + 1];
            eigenvectors[[j, i]] = c64::new(re_part, im_part);
        }

        // timed_println!("Cleaning up PETSc objects");
        // Clean up vectors
        unsafe {
            destroy_vector(vr).expect("Failed to destroy vector");
        }
    
    }
    
    // Clean up matrix
    unsafe {
        MatDestroy(&mut a);
    }
    
    (eigenvalues, eigenvectors)
}

/// Convert an Array2 of initial vectors to a Vec of PETSc Vec objects
/// for use with solver.set_initial_space()
/// 
/// For Hermitian problems, we align the phase of each eigenvector to maximize
/// the real component, which provides a consistent gauge across k-points.
fn create_petsc_vecs_from_array2(initial_vecs: &Array2<c64>) -> Result<Vec<rust_slepc_bindings::Vec>, PetscError> {
    let n_vecs = initial_vecs.ncols();
    let vec_size = initial_vecs.nrows();
    let mut petsc_vecs = Vec::with_capacity(n_vecs);
    
    for i in 0..n_vecs {
        // Create a new PETSc complex vector
        let v = unsafe {create_vector(vec_size)?};
        
        // Get a mutable slice to the vector's data
        // For complex vectors, PETSc stores as interleaved real/imag
        let mut vec_data = vec![0.0; 2 * vec_size];
        
        // Phase-align the eigenvector to maximize real component
        // Find the element with largest magnitude
        let mut max_mag = 0.0;
        let mut max_idx = 0;
        for j in 0..vec_size {
            let mag = initial_vecs[[j, i]].norm();
            if mag > max_mag {
                max_mag = mag;
                max_idx = j;
            }
        }
        
        // Calculate phase factor to rotate largest element to be real and positive
        let phase_factor = if max_mag > 1e-14 {
            let max_elem = initial_vecs[[max_idx, i]];
            // Phase factor is e^(-i*arg(max_elem)) to make it real positive
            c64::new(max_elem.re / max_mag, -max_elem.im / max_mag)
        } else {
            c64::new(1.0, 0.0)
        };
        
        // Copy data from the Array2 column to our temporary buffer with phase alignment
        for j in 0..vec_size {
            let rotated = initial_vecs[[j, i]] * phase_factor;
            vec_data[2*j] = rotated.re;     // Real part
            vec_data[2*j + 1] = rotated.im; // Imaginary part
        }
        
        // Copy our phase-aligned buffer to the PETSc vector
        let mut vec_ptr: *mut c_double = std::ptr::null_mut();
        unsafe extern "C" {
            fn VecGetArray(v: rust_slepc_bindings::Vec, a: *mut *mut c_double) -> c_int;
            fn VecRestoreArray(v: rust_slepc_bindings::Vec, a: *mut *mut c_double) -> c_int;
        }
        
        let result = unsafe {VecGetArray(v, &mut vec_ptr)};
        if result != 0 {
            // Clean up already created vectors
            for v in petsc_vecs {
                let _ = unsafe {destroy_vector(v)};
            }
            return Err(PetscError(result));
        }
        
        // Copy data to the PETSc vector (2*vec_size for complex)
        unsafe {std::ptr::copy_nonoverlapping(vec_data.as_ptr(), vec_ptr, 2 * vec_size)};
        
        // Restore the array
        let result = unsafe {VecRestoreArray(v, &mut vec_ptr)};
        if result != 0 {
            // Clean up already created vectors
            for v in petsc_vecs {
                let _ = unsafe {destroy_vector(v)};
            }
            return Err(PetscError(result));
        }
        
        petsc_vecs.push(v);
    }
    
    Ok(petsc_vecs)
}

/// Check if an operator is Hermitian by testing H*v vs (H*v)†
/// This is a diagnostic function to verify your operator is correctly implemented
#[allow(dead_code)]
pub fn verify_hermitian<T: SolverOperator<c64>>(op: &T, tolerance: f64) -> bool {
    use rand::Rng;
    let mut rng = rand::rng();
    
    let n = op.dimension();
    let n_tests = 5.min(n); // Test with 5 random vectors
    
    for _ in 0..n_tests {
        // Create two random complex vectors
        let x: Vec<c64> = (0..n).map(|_| {
            c64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        }).collect();
        
        let y: Vec<c64> = (0..n).map(|_| {
            c64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        }).collect();
        
        // Compute H*x and H*y
        let mut hx = vec![c64::zero(); n];
        let mut hy = vec![c64::zero(); n];
        op.apply(&x, &mut hx);
        op.apply(&y, &mut hy);
        
        // Compute <y, H*x> and <H*y, x>
        let y_hx: c64 = y.iter().zip(hx.iter()).map(|(yi, hxi)| yi.conj() * hxi).sum();
        let hy_x: c64 = hy.iter().zip(x.iter()).map(|(hyi, xi)| hyi.conj() * xi).sum();
        
        // For Hermitian matrix: <y, H*x> = <H*y, x>
        // (NOT <H*y, x>*, they should be equal!)
        let diff = (y_hx - hy_x).norm();
        let scale = y_hx.norm().max(hy_x.norm());
        
        if scale > 1e-10 && diff / scale > tolerance {
            timed_println!("WARNING: Operator may not be Hermitian!");
            timed_println!("  <y, H*x> = {}", y_hx);
            timed_println!("  <H*y, x> = {}", hy_x);
            timed_println!("  Difference: {}", y_hx - hy_x);
            timed_println!("  Relative difference: {:.2e}", diff / scale);
            return false;
        }
    }
    
    timed_println!("Hermiticity check passed (tolerance: {:.2e})", tolerance);
    true
}