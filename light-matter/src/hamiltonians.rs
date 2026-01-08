use std::iter::Sum;

use crate::{utils::{BosonMode, MatrixOps, Sqrt}};
use num_traits::NumCast;
use exciton_2d::FullHamiltonian;
use ndarray::{arr1, Array1, Array2, s};
use sprs::TriMat;
use crate::matrix_elements::LMMatrixElementArray;

pub struct BosonicHamiltonian<M: MatrixOps> {
    pub bosonic_modes: Vec<BosonMode>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> BosonicHamiltonian<M> {
    pub fn new(bosonic_modes: Vec<BosonMode>) -> Self {
        Self { 
            bosonic_modes,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_a(&self, mode_i: usize) -> M
    where 
        M::Scalar: Sqrt + NumCast,
    {
        self.bosonic_modes.iter().enumerate().map(|(idx,mode)| {
            if idx == mode_i {
                // Create annihilation operator for this mode
                let dim = mode.n_fock;
                let mut trimat = TriMat::new((dim, dim));
                for n in 1..dim {
                    trimat.add_triplet(n-1, n, (<M::Scalar as NumCast>::from(n).unwrap()).sqrt());
                }
                M::from_trimat(&trimat)
            }
            else {
                // Identity for other modes
                M::eye(mode.n_fock)
            }
        }).reduce(|acc, sing_mat| M::kron(&acc, &sing_mat))
        .unwrap()
    }

    pub fn get_all_q_ind(&self) -> Vec<Vec<i32>> {
        let mut all_q_ind: Vec<Vec<i32>> = Vec::new();
        let n_ph = self.bosonic_modes.len();

        // Single photon modes
        for i in 0..n_ph {
            all_q_ind.push(vec![i as i32]);
        }

        // Pairs of bosonic_modes modes for A^2 terms
        for i in 0..n_ph {
            for j in i..n_ph {
                if i != j {
                    all_q_ind.push(vec![i as i32, j as i32]);
                }
            }
        }

        all_q_ind
    }

    pub fn get_adag_a(&self, mode_i: usize) -> M 
    where 
        M::Scalar: Sqrt + NumCast,
    {
        let a = self.get_a(mode_i);
        a.transpose().dot(&a)
    }

    /// Gets the complete photonic Hamiltonian
    pub fn get_h(&self) -> M 
    where 
        M::Scalar: Sqrt + NumCast,
    {
        let bosonic_modes = &self.bosonic_modes;

        // Handle empty vector case
        if bosonic_modes.is_empty() {
            return M::eye(1);
        }

        // Sum Hamiltonians for each mode
        bosonic_modes.iter().enumerate().map(|(mode_i, mode)| {
            self.get_adag_a(mode_i).scale(<M::Scalar as NumCast>::from(mode.omega).unwrap())
            })
            .reduce(|acc, h_ph_sing| acc.add(&h_ph_sing))
            .unwrap()
    }
}

pub struct ExcitonicHamiltonian <M: MatrixOps> {
    pub exciton_ham: FullHamiltonian,
    pub n_ex_eig: usize,
    pub n_rel: usize,
    pub parity: i32,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Vec<Array1<f64>>>,
    pub r_vecs: Option<Array2<f64>>,
    pub parities: Option<Array1<i32>>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> ExcitonicHamiltonian<M> {
    pub fn new(exciton_ham: FullHamiltonian, n_ex_eig: usize, n_rel: usize, parity: i32) -> Self {
        Self { exciton_ham, 
            n_ex_eig, 
            n_rel, 
            parity, 
            e_vals: None, 
            e_vecs: None, 
            r_vecs: None, 
            parities: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool{
        self.e_vals.is_none() || self.e_vals.is_none() || self.r_vecs.is_none() || self.parities.is_none()
    }

    pub fn compute_eigenstates(&mut self) {
        let (exciton_energies, e_vecs, _, r_vecs, parities) 
            = self.exciton_ham.diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig, self.parity);
        
        self.e_vals = Some(exciton_energies.slice(s![..self.n_ex_eig]).to_owned());
        // timed_println!("Exciton energies: {:?}", self.e_vals);

        self.e_vecs = Some(e_vecs[0..self.n_ex_eig].to_vec());
        self.r_vecs = Some(r_vecs);
        self.parities = Some(parities.slice(s![..self.n_ex_eig]).to_owned());
    }

    pub fn get_p_com(&self, field_vec: &Array1<f64>, diag: bool) -> M
    where 
        M::Scalar: NumCast + Sum,
    {
        let e_vecs = self.e_vecs.as_ref().expect("Eigenvectors not computed. Call compute_eigenstates() first.");
        let parities = self.parities.as_ref().expect("Parities not computed. Call compute_eigenstates() first.");
        let kappas= &self.exciton_ham.kappa_vectors;
        let k = &self.exciton_ham.k;

        let n_com = kappas.len();
        let n_eig = e_vecs.len();
        
        // Convert k to Array1 once for efficiency
        let k_arr = arr1(&k);
        
        // Precompute p_dot_unit for each component - avoid unnecessary cloning
        let p_dot_unit_cache: Vec<f64> = kappas.iter()
            .map(|kappa| {
                // Calculate (k + kappa)·field_vec for each kappa point more efficiently
                (k_arr[0] + kappa[0]) * field_vec[0] + (k_arr[1] + kappa[1]) * field_vec[1]
            })
            .collect();
            
        if diag {
            M::from_dense(&Array2::from_shape_fn((n_eig,1), |(idx1,_)| {
                // For each eigenvector
                let vec1 = &e_vecs[idx1];
                let n_rel = vec1.len() / n_com;

                // Compute P_ii using iterator operations
                // This is a purely diagonal operator
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        a1 * a1 * p
                    })
                }).sum::<f64>()
            }))
        }
        else {
            // Build matrix using functional operations
            M::from_dense(&Array2::from_shape_fn((n_eig, n_eig), |(idx1, idx2)| {
                // For each pair of eigenvectors
                let vec1 = &e_vecs[idx1];
                let vec2 = &e_vecs[idx2];

                if parities[idx1] != parities[idx2]  {
                    return 0.0;
                }
                
                let n_rel = vec1.len() / n_com;

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
                }).sum::<f64>()
            }))
        }
    }

    fn get_h(&mut self) -> M  
    where 
        M::Scalar: NumCast,
    {
        if self.is_empty() {
            self.compute_eigenstates();
        }
        let diag: Vec<M::Scalar> = self.e_vals.as_ref().unwrap().iter().map(|val| {
            <M::Scalar as NumCast>::from(*val).unwrap()
        }).collect();

        M::from_diag(diag)
    }
}

pub struct LMHamiltonian <M: MatrixOps> {
    pub exciton: ExcitonicHamiltonian<M>,
    pub photon: BosonicHamiltonian<M>,
    matrix_elements: LMMatrixElementArray,
    coupling: f64,
    p_c_sin: Option<Vec<M>>,
    p_q: Option<Vec<M>>,
}

impl<M: MatrixOps>  LMHamiltonian<M> {
        pub fn new( 
            exciton: ExcitonicHamiltonian<M>, 
            photon: BosonicHamiltonian<M>, 
            coupling: f64) -> Self {

        let matrix_elements = LMMatrixElementArray::new();

        Self { exciton, photon, matrix_elements, coupling, p_c_sin: None, p_q: None }
    }

    fn calc_p_c_sin_matrix(&mut self) 
    where 
        M: MatrixOps<Scalar = f64>,
    {
        
        let n_ph = self.photon.bosonic_modes.len() + 1; // +1 for ground state

        let sin_mat = &self.matrix_elements.sin_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // timed_println!("{:?}",test);
        
        let mut p_c_sin: Vec<M> = Vec::new();
        let mut p_q: Vec<M> = Vec::new();
        (0 .. n_ph-1).for_each(|i| {
            let eps = &self.photon.bosonic_modes[i].eps;
            let q = &self.photon.bosonic_modes[i].q;
            let q_ind_key = i as i32;

            p_c_sin.push( 
                    self.exciton.get_p_com(eps, false)// P·eps 
                    .dot(&M::from_dense(sin_mat.get(&vec![q_ind_key]).unwrap())) // sin(qx/2) matrix element
                );
            
            p_q.push( 
                    self.exciton.get_p_com(q, false)// P·q 
                );

        });

        self.p_c_sin = Some(p_c_sin);
        self.p_q = Some(p_q);
    }

    fn get_interaction_hamiltonian(&self, ham: &mut M) 
    where 
        M::Scalar: NumCast + Sqrt,  
    {
        // Iterate over photon modes to build interaction terms
        self.photon.bosonic_modes.iter().enumerate().for_each(|(i_ph, mode_i)| {

            let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
            let ex_eye:M = M::eye(n_ex);

            // p.A terms
            let a_i = &self.photon.get_a(i_ph);
            let p_c_sin = &self.p_c_sin.as_ref().unwrap()[i_ph].scale( 2.0 / self.exciton.exciton_ham.m * self.coupling);
            let p_q = &self.p_q.as_ref().unwrap()[i_ph].scale(1.0 / 2.0 / self.exciton.exciton_ham.m);

            let p_cos = &M::from_dense(&(self.matrix_elements.p_cos_mat.get(&vec![i_ph as i32]).unwrap() * self.coupling));
            
            // P•A sin(x) term
            *ham = ham.add(&M::kron(p_c_sin, &(a_i.add(&a_i.transpose()))));

            // p•A cos(x) term
            *ham = ham.add(&M::kron(p_cos, &(a_i.scale(-1.0).add(&a_i.transpose()))));

            // P•q a†a term
            *ham = ham.add(&M::kron(p_q, &(&a_i.transpose().dot(a_i))));

            // A² terms
            self.photon.bosonic_modes.iter().enumerate().for_each(|(j_ph, mode_j)| {
                let q_ind1 = (i_ph as i32).max(j_ph as i32);
                let q_ind2 = (i_ph as i32).min(j_ph as i32);
                let q_diff_key = vec![q_ind1, -q_ind2];
                let q_sum_key = vec![q_ind1, q_ind2];

                let a_j = &self.photon.get_a(j_ph);

                let coupling_2 = self.coupling * self.coupling * mode_i.eps.dot(&mode_j.eps) / 2. / self.exciton.exciton_ham.rel_ham.mu;

                let cos_diff = &M::from_dense(self.matrix_elements.cos_mat.get(&q_diff_key).unwrap());
                let cos_sum  = &M::from_dense(self.matrix_elements.cos_mat.get(&q_sum_key ).unwrap());

                let a_diff = &( a_i.dot(&a_j.transpose()).add(&a_i.transpose().dot(a_j)) );
                let a_sum  = &( a_i.transpose().dot(&a_j.transpose()).add( &a_i.dot(a_j)) );

                // cos(q + q') term
                *ham = ham.add(&M::kron(cos_diff, a_diff).scale(coupling_2));

                // cos(q - q') term
                *ham = ham.add(&M::kron(cos_sum, a_sum).scale(-coupling_2));

                // (q a†a)^2 term
                let q_q = mode_i.q.dot(&mode_j.q);
                *ham = ham.add(&M::kron(&ex_eye, &(a_i.transpose().dot(a_i).dot(&a_j.transpose()).dot(a_j)) ).scale(q_q));
            });
        });
    }
}