use crate::sing_ex::{PolaronPolariton};
use crate::utils::{SolverOperator};
use ndarray::{Array1, Array2};
use std::ffi::{c_int, c_double, c_void};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::time::Instant;
use rust_slepc_bindings::{
    Eigensolver, EPS_WHICH_SMALLEST_REAL,
    Mat, PETSC_COMM_WORLD,
    create_complex_vector as create_vector, destroy_vector, get_vector_array, EPSType, PetscError,
    EPSGDSetBlockSize, EPSJDSetBlockSize
};
use num_complex::Complex64 as c64;
use crate::timed_println;

// Global counters for tracking callback overhead
pub static MATVEC_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static MATVEC_TOTAL_TIME_NS: AtomicU64 = AtomicU64::new(0);

// PETSc/SLEPc function declarations needed for shell matrix
unsafe extern "C" {
    // Matrix creation and manipulation
    fn MatCreateShell(comm: rust_slepc_bindings::MPIComm, m: c_int, n: c_int, M: c_int, N: c_int, ctx: *mut c_void, A: *mut Mat) -> c_int;
    fn MatShellSetOperation(mat: Mat, op: c_int, f: *mut c_void) -> c_int;
    fn MatShellGetContext(mat: Mat, ctx: *mut *mut c_void) -> c_int;
    fn MatDestroy(A: *mut Mat) -> c_int;
    
    // Vector operations
    fn VecGetArray(v: rust_slepc_bindings::PetscVec, a: *mut *mut c_double) -> c_int;
    fn VecRestoreArray(v: rust_slepc_bindings::PetscVec, a: *mut *mut c_double) -> c_int;
}

// Constants needed for this implementation
const MATOP_MULT: c_int = 3; // Matrix-vector product operation

// Instrumented callback function for matrix-vector product (complex version)
extern "C" fn mv_product_instrumented(mat: Mat, x: rust_slepc_bindings::PetscVec, y: rust_slepc_bindings::PetscVec) -> c_int {
    let callback_start = Instant::now();
    
    unsafe {
        // Extract the operator from the context
        let mut ctx_ptr: *mut c_void = std::ptr::null_mut();
        MatShellGetContext(mat, &mut ctx_ptr);
        let op = &*(ctx_ptr as *const PolaronPolariton);
        
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
        
        // Time the actual apply operation
        let _apply_start = Instant::now();
        let mut y_complex = vec![c64::new(0.0, 0.0); n];
        op.apply(&x_complex, &mut y_complex);
        let _apply_duration = _apply_start.elapsed();
        
        // Convert back to interleaved format
        for i in 0..n {
            y_raw[2*i] = y_complex[i].re;
            y_raw[2*i + 1] = y_complex[i].im;
        }
        
        // Restore arrays
        VecRestoreArray(x, &mut x_array);
        VecRestoreArray(y, &mut y_array);
        
        // Update counters
        MATVEC_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        let callback_duration = callback_start.elapsed();
        MATVEC_TOTAL_TIME_NS.fetch_add(callback_duration.as_nanos() as u64, Ordering::SeqCst);
        
        0 // PETSC_SUCCESS
    }
}

/// Reset the instrumentation counters
pub fn reset_instrumentation() {
    MATVEC_CALL_COUNT.store(0, Ordering::SeqCst);
    MATVEC_TOTAL_TIME_NS.store(0, Ordering::SeqCst);
}

/// Get the current instrumentation data
pub fn get_instrumentation_data() -> (usize, f64) {
    let count = MATVEC_CALL_COUNT.load(Ordering::SeqCst);
    let total_time_ns = MATVEC_TOTAL_TIME_NS.load(Ordering::SeqCst);
    let total_time_s = total_time_ns as f64 / 1e9;
    (count, total_time_s)
}

/// Diagonalize the PolaronPolariton using SLEPc with instrumentation
pub fn diagonalize_slepc_instrumented(
    pp: &PolaronPolariton, 
    n_eig: usize,
    block_size: usize, 
    solve_type: Option<EPSType>,
    initial_guess: Option<&Array2<f64>>
) -> (Array1<f64>, Array2<f64>) {
    // Reset instrumentation
    reset_instrumentation();
    
    // Create a shell matrix for the operator
    let mut a = std::ptr::null_mut();
    let n = pp.dimension() as c_int;
    
    timed_println!("Creating shell matrix of dimension {}", n);
    
    unsafe {
        // Create shell matrix
        let op_ptr = pp as *const _ as *mut c_void;
        MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, op_ptr, &mut a);
        
        // Set the instrumented matrix-vector product operation
        MatShellSetOperation(a, MATOP_MULT, mv_product_instrumented as *mut c_void);
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
    solver.set_problem_type(rust_slepc_bindings::EPSProblemType::NonHermitian).expect("Failed to set problem type");
    solver.set_tolerances(1e-8, 10000).expect("Failed to set tolerances");
    
    let solver_type = solve_type.unwrap_or(EPSType::ARPACK);
    timed_println!("Using solver type: {:?}", solver_type);
    solver.set_type(solver_type).expect("Failed to set eigensolver");
    
    // Apply options to ensure DS is properly initialized
    unsafe {
        unsafe extern "C" {
            fn EPSSetFromOptions(eps: rust_slepc_bindings::EPS) -> c_int;
            fn EPSSetUp(eps: rust_slepc_bindings::EPS) -> c_int;
        }
        
        EPSSetFromOptions(solver.eps);
        EPSSetUp(solver.eps);
    }
    
    // Set initial guess if provided
    if let Some(initial_vecs) = initial_guess {
        timed_println!("Using provided initial guess with {} vectors", initial_vecs.ncols());
        unsafe {
            let petsc_vecs = match create_petsc_vecs_from_array2(initial_vecs) {
                Ok(vecs) => vecs,
                Err(e) => {
                    timed_println!("Error creating PETSc vectors from initial guess: {:?}", e);
                    Vec::new()
                }
            };
            
            if !petsc_vecs.is_empty() {
                if let Err(e) = solver.set_initial_space(&petsc_vecs) {
                    timed_println!("Error setting initial space: {:?}", e);
                }
                
                for v in petsc_vecs {
                    let _ = destroy_vector(v);
                }
            }
        }
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
    
    // Get instrumentation data
    let (matvec_count, matvec_time) = get_instrumentation_data();
    timed_println!("Matrix-vector products called {} times, total time: {:.3} s", matvec_count, matvec_time);
    
    // Always create arrays of the requested size
    let mut eigenvalues = Array1::zeros(n_eig);
    let mut eigenvectors = Array2::zeros((pp.dimension(), n_eig));
    
    // Create vectors for eigenvectors (complex vectors need 2x space)
    let vr = unsafe { create_vector(pp.dimension()).expect("Failed to create vector") };
    
    // Extract eigenvalues and eigenvectors for the converged ones
    let n_eig_actual = std::cmp::min(n_eig, nconv);
    for i in 0..n_eig_actual {
        let re = solver.get_eigenvalue(i).expect("Failed to get eigenvalue");
        eigenvalues[i] = re;
        
        unsafe {
            solver.get_eigenpair(i, vr).expect("Failed to get eigenvector");
            
            // PETSc complex vectors store data as interleaved real/imag
            let mut vec_temp = vec![0.0; 2 * pp.dimension()];
            get_vector_array(vr, &mut vec_temp).expect("Failed to get vector array");
            
            // Extract real parts (for Hermitian matrices, eigenvectors can be chosen real)
            for j in 0..pp.dimension() {
                let re_part = vec_temp[2*j];
                // Store real part (imaginary part should be ~0 for Hermitian)
                eigenvectors[[j, i]] = re_part;
            }
        }
    }
    
    // Clean up vectors
    unsafe {
        destroy_vector(vr).expect("Failed to destroy vector");
    }
    
    // Clean up matrix
    unsafe {
        MatDestroy(&mut a);
    }
    
    (eigenvalues, eigenvectors)
}

/// Convert an Array2 of initial vectors to a Vec of PETSc Vec objects
/// For complex vectors, the real values are used as the real part with zero imaginary part
fn create_petsc_vecs_from_array2(initial_vecs: &Array2<f64>) -> Result<Vec<rust_slepc_bindings::PetscVec>, PetscError> {
    let n_vecs = initial_vecs.ncols();
    let vec_size = initial_vecs.nrows();
    let mut petsc_vecs = Vec::with_capacity(n_vecs);
    
    for i in 0..n_vecs {
        let v = unsafe { create_vector(vec_size)? };
        
        // For complex vectors, PETSc stores as interleaved real/imag
        let mut vec_data = vec![0.0; 2 * vec_size];
        for j in 0..vec_size {
            vec_data[2*j] = initial_vecs[[j, i]];     // Real part
            vec_data[2*j + 1] = 0.0;                   // Imaginary part
        }
        
        let mut vec_ptr: *mut c_double = std::ptr::null_mut();
        unsafe extern "C" {
            fn VecGetArray(v: rust_slepc_bindings::PetscVec, a: *mut *mut c_double) -> c_int;
            fn VecRestoreArray(v: rust_slepc_bindings::PetscVec, a: *mut *mut c_double) -> c_int;
        }
        
        let result = unsafe { VecGetArray(v, &mut vec_ptr) };
        if result != 0 {
            for v in petsc_vecs {
                let _ = unsafe { destroy_vector(v) };
            }
            return Err(PetscError(result));
        }
        
        // Copy data to the PETSc vector (2*vec_size for complex)
        unsafe { std::ptr::copy_nonoverlapping(vec_data.as_ptr(), vec_ptr, 2 * vec_size) };
        
        let result = unsafe { VecRestoreArray(v, &mut vec_ptr) };
        if result != 0 {
            for v in petsc_vecs {
                let _ = unsafe { destroy_vector(v) };
            }
            return Err(PetscError(result));
        }
        
        petsc_vecs.push(v);
    }
    
    Ok(petsc_vecs)
}
