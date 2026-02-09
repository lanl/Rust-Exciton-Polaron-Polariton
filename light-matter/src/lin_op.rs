// Import statements for the various libraries used in this code
use ndarray::{Array1, Array2};
use crate::hamiltonians::{LMHamiltonian, PPHamiltonian};
use crate::utils::{ComposedOperator, SolverOperator};
use rust_slepc_bindings::EPSType;
use ndarray_linalg::c64;

pub fn diagonalize_lm_ham_linop(
    light_matter: &mut LMHamiltonian<ComposedOperator<c64>>, 
    n_eig: Option<usize>, 
    solve_type: Option<EPSType>, 
    initial_guess: Option<&Array2<c64>>) 
    -> (Array1<f64> , Array2<c64>){
    let h_lm = light_matter.build_lm_hamiltonian(None);

    h_lm.diagonalize(n_eig, solve_type, initial_guess)
}

pub fn diagonalize_pp_ham_linop(
    polaron_polariton: &mut PPHamiltonian<ComposedOperator<c64>>, 
    n_eig: Option<usize>, 
    solve_type: Option<EPSType>, 
    initial_guess: Option<&Array2<c64>>)
    -> (Array1<f64> , Array2<c64>){
    let h_pp = polaron_polariton.build_pp_hamiltonian(None);

    h_pp.diagonalize(n_eig, solve_type, initial_guess)
}
