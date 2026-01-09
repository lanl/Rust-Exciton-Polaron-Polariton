use crate::utils::{SolverOperator};
use ndarray::{Array1, Array2};
use std::ffi::{c_int, c_double, c_void};
use rust_slepc_bindings::{
    Eigensolver, EPS_WHICH_SMALLEST_REAL,
    Mat, PETSC_COMM_WORLD,
    create_vector, destroy_vector, get_vector_array, EPSType, PetscError,
    EPSGDSetBlockSize, EPSJDSetBlockSize
};
use crate::timed_println;

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

// Callback function for matrix-vector product
extern "C" fn mv_product<T: SolverOperator<f64>>(
    mat: Mat, 
    x: rust_slepc_bindings::Vec, 
    y: rust_slepc_bindings::Vec
) -> c_int {
    unsafe {
        // Extract the operator from the context
        let mut ctx_ptr: *mut c_void = std::ptr::null_mut();
        MatShellGetContext(mat, &mut ctx_ptr);
        let op = &*(ctx_ptr as *const T);
        
        // Get arrays from vectors
        let mut x_array: *mut c_double = std::ptr::null_mut();
        let mut y_array: *mut c_double = std::ptr::null_mut();
        VecGetArray(x, &mut x_array);
        VecGetArray(y, &mut y_array);
        
        // Convert to Rust slices
        let n = op.dimension();
        let x_slice = std::slice::from_raw_parts(x_array, n);
        let y_slice = std::slice::from_raw_parts_mut(y_array, n);
        
        // Apply the operator
        op.apply(x_slice, y_slice);
        
        // Restore arrays
        VecRestoreArray(x, &mut x_array);
        VecRestoreArray(y, &mut y_array);
        
        0 // PETSC_SUCCESS
    }
}

/// Diagonalize the PolaronPolariton using SLEPc
pub fn diagonalize_slepc<T: SolverOperator<f64>>(
    op: &T, 
    n_eig: usize,
    block_size: usize, 
    solve_type: Option<EPSType>,
    initial_guess: Option<&Array2<f64>>
) -> (Array1<f64>, Array2<f64>) {
    // Create a shell matrix for the operator
    let mut a = std::ptr::null_mut();
    let n = op.dimension() as c_int;
    
    timed_println!("Creating shell matrix of dimension {}", n);
    
    unsafe {
        // Create shell matrix
        let op_ptr = op as *const T as *const c_void as *mut c_void;
        MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, op_ptr, &mut a);
        
        // Set the matrix-vector product operation
        MatShellSetOperation(a, MATOP_MULT, mv_product::<T> as *mut c_void);
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
    
    // Try with non-Hermitian problem type since we're not sure about the matrix properties
    solver.set_problem_type(rust_slepc_bindings::EPSProblemType::NonHermitian).expect("Failed to set problem type");
    
    // Set more lenient tolerances and more iterations
    solver.set_tolerances(1e-8, 10000).expect("Failed to set tolerances");
    
    // Set the solver type
    let solver_type = solve_type.unwrap_or(EPSType::ARPACK);
    timed_println!("Using solver type: {:?}", solver_type);
    solver.set_type(solver_type).expect("Failed to set eigensolver");
    
    // Apply options to ensure DS is properly initialized
    unsafe {
        unsafe extern "C" {
            fn EPSSetFromOptions(eps: rust_slepc_bindings::EPS) -> c_int;
            fn EPSSetUp(eps: rust_slepc_bindings::EPS) -> c_int;
        }
        
        // Apply options and set up the solver
        EPSSetFromOptions(solver.eps);
        EPSSetUp(solver.eps);
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
                if let Err(e) = solver.set_initial_space(&petsc_vecs) {
                    timed_println!("Error setting initial space: {:?}", e);
                }
                
                // Clean up the vectors
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
    
    // Always create arrays of the requested size
    let mut eigenvalues = Array1::zeros(n_eig);
    let mut eigenvectors = Array2::zeros((op.dimension(), n_eig));
    
    // Create vectors for eigenvectors
    let vr = unsafe { create_vector(op.dimension()).expect("Failed to create vector") };
    
    // Extract eigenvalues and eigenvectors for the converged ones
    let n_eig_actual = std::cmp::min(n_eig, nconv);
    for i in 0..n_eig_actual {
        // Get eigenvalue
        let re = solver.get_eigenvalue(i).expect("Failed to get eigenvalue");
        eigenvalues[i] = re;
        
        // Get eigenvector
        unsafe {
            solver.get_eigenpair(i, vr).expect("Failed to get eigenvector");
            
            // Copy to eigenvectors array
            let mut vec_temp = vec![0.0; op.dimension()];
            get_vector_array(vr, &mut vec_temp).expect("Failed to get vector array");
            
            for j in 0..op.dimension() {
                eigenvectors[[j, i]] = vec_temp[j];
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
/// for use with solver.set_initial_space()
fn create_petsc_vecs_from_array2(initial_vecs: &Array2<f64>) -> Result<Vec<rust_slepc_bindings::Vec>, PetscError> {
    let n_vecs = initial_vecs.ncols();
    let vec_size = initial_vecs.nrows();
    let mut petsc_vecs = Vec::with_capacity(n_vecs);
    
    for i in 0..n_vecs {
        // Create a new PETSc vector
        let v = unsafe {create_vector(vec_size)?};
        
        // Get a mutable slice to the vector's data
        let mut vec_data = vec![0.0; vec_size];
        
        // Copy data from the Array2 column to our temporary buffer
        for j in 0..vec_size {
            vec_data[j] = initial_vecs[[j, i]];
        }
        
        // Copy our buffer to the PETSc vector
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
        
        // Copy data to the PETSc vector
        unsafe {std::ptr::copy_nonoverlapping(vec_data.as_ptr(), vec_ptr, vec_size)};
        
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