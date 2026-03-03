use std::iter::Sum;
use exciton_2d::{Exciton2DGridSystem};
use log::warn;
use crate::timed_println;
use crate::utils::{BosonMode, ConvertFrom, MatrixOps, PPConfig, Polarization, Sqrt, Vector3D};
use num_traits::{NumCast, One};
use exciton_2d::FullHamiltonian;
use ndarray::{Array1, Array2, arr1, s};
use sprs::{TriMat};
use crate::matrix_elements::LMMatrixElementArray;
use ndarray_linalg::c64;

pub trait Boson<M:MatrixOps> {
    fn new(modes: Vec<BosonMode>) -> Self;

    fn a(&self, mode_i: usize) -> M
    where 
        M::Scalar: Sqrt + NumCast;

    fn adag_a(&self, mode_i: usize) -> M
    where 
        M::Scalar: Sqrt + NumCast,
    {
        let a = self.a(mode_i);
        a.transpose().dot(&a)
    }

    fn get_modes(&self) -> &Vec<BosonMode>;

    fn dimension (&self) -> usize;
}

pub struct FullBosons<M: MatrixOps> {
    pub bosonic_modes: Vec<BosonMode>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> Boson<M> for FullBosons<M> {

    fn new(modes: Vec<BosonMode>) -> Self {
        Self {bosonic_modes: modes, _phantom: std::marker::PhantomData}
    }

    fn a(&self, mode_i: usize) -> M
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

    fn get_modes(&self) -> &Vec<BosonMode> {
        &self.bosonic_modes
    }

    fn dimension (&self) -> usize {
        self.get_modes().iter().map(|mode| mode.n_fock).product()
    }
}

pub struct SingExBosons<M: MatrixOps> {
    pub bosonic_modes: Vec<BosonMode>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> Boson<M> for SingExBosons<M> {

    fn new(modes: Vec<BosonMode>) -> Self {
        Self {bosonic_modes: modes, _phantom: std::marker::PhantomData}
    }

    fn a(&self, mode_i: usize) -> M
    where 
        M::Scalar: Sqrt + NumCast + One,
    {
        let dim = self.dimension();
        let mut trimat = TriMat::new((dim, dim));
        trimat.add_triplet(0, mode_i+1, <M::Scalar>::one());
        
        M::from_trimat(&trimat)
    }

    fn get_modes(&self) -> &Vec<BosonMode> {
        &self.bosonic_modes
    }

    fn dimension (&self) -> usize {
        self.get_modes().len() + 1
    }
}

pub struct BosonicHamiltonian<M: MatrixOps, B: Boson<M>> {
    pub boson: B,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps, B: Boson<M>> BosonicHamiltonian<M,B> {
    pub fn new(bosonic_modes: Vec<BosonMode>) -> Self {
        Self { 
            boson: B::new(bosonic_modes),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn get_a(&self, mode_i: usize) -> M
    where 
        M::Scalar: Sqrt + NumCast,
    {
        self.boson.a(mode_i)
    }

    pub fn get_all_q_ind(&self) -> Vec<Vec<i32>> {
        let mut all_q_ind: Vec<Vec<i32>> = Vec::new();
        let n_ph = self.boson.get_modes().len();

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
        self.boson.adag_a(mode_i)
    }

    /// Gets the complete photonic Hamiltonian
    pub fn get_h(&self) -> M 
    where 
        M::Scalar: Sqrt + NumCast,
    {
        let bosonic_modes = self.boson.get_modes();

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

    pub fn dimension (&self) -> usize {
        self.boson.dimension()
    }
}


pub struct ExcitonicHamiltonian <M: MatrixOps> {
    pub exciton_ham: FullHamiltonian,
    pub n_ex_eig: usize,
    pub n_rel: usize,
    pub e_vals: Option<Array1<f64>>,
    pub e_vecs: Option<Vec<Array1<num_complex::Complex64>>>,
    pub r_vecs: Option<Array2<f64>>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: MatrixOps> ExcitonicHamiltonian<M> {
    pub fn new(exciton_ham: FullHamiltonian, n_ex_eig: usize, n_rel: usize) -> Self {
        Self { exciton_ham, 
            n_ex_eig, 
            n_rel, 
            e_vals: None, 
            e_vecs: None, 
            r_vecs: None, 
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool{
        self.e_vals.is_none() || self.e_vals.is_none() || self.r_vecs.is_none()
    }

    pub fn compute_eigenstates(&mut self) {
        let (exciton_energies, e_vecs, _, r_vecs) 
            = self.exciton_ham.diagonalize_full_hamiltonian(self.n_rel, self.n_ex_eig);
        
        self.e_vals = Some(exciton_energies.slice(s![..self.n_ex_eig]).to_owned());
        // timed_println!("Exciton energies: {:?}", self.e_vals);

        self.e_vecs = Some(e_vecs[0..self.n_ex_eig].to_vec());
        self.r_vecs = Some(r_vecs);
    }

    pub fn get_p_com(&self, field_vec: &Array1<f64>, diag: bool) -> M
    where 
        M::Scalar: NumCast + Sum + ConvertFrom<c64> + ConvertFrom<f64>,
    {
        let e_vecs = self.e_vecs.as_ref().expect("Eigenvectors not computed. Call compute_eigenstates() first.");
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
                // For complex eigenvectors: <ψ|P|ψ> = Σ |ψ|² P
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        a1.norm_sqr() * p  // |ψ|² for complex numbers
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
                
                let n_rel = vec1.len() / n_com;

                // Compute P_ij using iterator operations
                // For complex eigenvectors: <ψ_i|P|ψ_j> = Σ ψ_i* P ψ_j
                (0 .. n_com).flat_map(|com_ind| {
                    let p = p_dot_unit_cache[com_ind];
                    
                    (0 .. n_rel).map(move |rel_ind| {
                        let i = n_com * rel_ind + com_ind;
                        let a1 = vec1[i];
                        let a2 = vec2[i];
                        (a1.conj() * a2) * p  // p * ψ_i* ψ_j
                    })
                }).sum::<c64>()
            }))
        }
    }

    pub fn get_h(&mut self) -> M  
    where 
        M::Scalar: NumCast,
    {
        if self.is_empty() {
            self.compute_eigenstates();
        }
        let diag: Vec<M::Scalar> = self.e_vals.as_ref().unwrap().iter().map(|val| {
            <M::Scalar as NumCast>::from(*val).unwrap()
        }).collect();

        M::from_diag(&diag)
    }

    pub fn dimension(&self) -> usize {
        self.e_vals.as_ref().map_or(0, |vals| vals.len())
    }
}

pub struct LMHamiltonian <M: MatrixOps, B: Boson<M>> {
    pub exciton: ExcitonicHamiltonian<M>,
    pub photon: BosonicHamiltonian<M, B>,
    pub matrix_elements: LMMatrixElementArray,
    pub coupling: f64,
    pub p_c_sin: Option<Vec<M>>,
    pub p_q: Option<Vec<M>>,
    is_initialized: bool,
    cached_a_ops: Option<Vec<M>>,  // Cache annihilation operators
    cached_adag_a_ops: Option<Vec<M>>,  // Cache number operators
}

impl<M: MatrixOps, B: Boson<M>>  LMHamiltonian<M,B> {
    pub fn new( 
            exciton: ExcitonicHamiltonian<M>, 
            photon: BosonicHamiltonian<M,B>, 
            coupling: f64) -> Self {

        let matrix_elements = LMMatrixElementArray::new();

        Self { exciton, photon, matrix_elements, coupling, p_c_sin: None, p_q: None, is_initialized: false, cached_a_ops: None, cached_adag_a_ops: None}
    }

    pub fn initialize(&mut self, k: Option<Vec<f64>>)  
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>, 
    {
        use std::time::Instant;
        
        // Check if we need to update k
        let k_changed = if let Some(k_val) = &k {
            k_val != &self.exciton.exciton_ham.k
        } else {
            false
        };

        // Only proceed if not initialized or k changed
        if !self.is_initialized || k_changed {
            if k_changed {
                self.update_k(k.unwrap());
            }
            
            let time_eig = Instant::now();
            self.exciton.compute_eigenstates();
            timed_println!("    Exciton diagonalization: {:?}", time_eig.elapsed());

            let time_mat = Instant::now();
            let all_q_ind = self.photon.get_all_q_ind();
            self.matrix_elements.populate_tay_mat_elements(
                &all_q_ind, 
                self.exciton.e_vecs.as_ref().unwrap(), 
                self.photon.boson.get_modes(), 
                self.exciton.exciton_ham.kappa_vectors.len(),
                self.exciton.exciton_ham.rel_ham.mu
            );
            timed_println!("    Matrix elements calculation: {:?}", time_mat.elapsed());

            let time_p = Instant::now();
            self.calc_p_c_sin_matrix();
            timed_println!("    P·c·sin matrix calculation: {:?}", time_p.elapsed());
            
            // Pre-compute and cache photon operators
            let time_cache = Instant::now();
            self.cache_photon_operators();
            timed_println!("    Cache photon operators: {:?}", time_cache.elapsed());
            
            self.is_initialized = true;
        }
    }
    
    fn cache_photon_operators(&mut self)
    where 
        M::Scalar: NumCast + Sqrt,
    {
        let n_modes = self.photon.boson.get_modes().len();
        let mut a_ops = Vec::with_capacity(n_modes);
        let mut adag_a_ops = Vec::with_capacity(n_modes);
        
        for i in 0..n_modes {
            a_ops.push(self.photon.get_a(i));
            adag_a_ops.push(self.photon.get_adag_a(i));
        }
        
        self.cached_a_ops = Some(a_ops);
        self.cached_adag_a_ops = Some(adag_a_ops);
    }

    fn calc_p_c_sin_matrix(&mut self) 
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>, 
    {
        
        let n_ph = self.photon.boson.get_modes().len() + 1; // +1 for ground state

        let sin_mat = &self.matrix_elements.sin_mat;

        // let test = cos_mat.clone().into_keys().collect::<Vec<Vec<i32>>>();
        // timed_println!("{:?}",test);
        
        let mut p_c_sin: Vec<M> = Vec::new();
        let mut p_q: Vec<M> = Vec::new();
        (0 .. n_ph-1).for_each(|i| {
            let eps = &self.photon.boson.get_modes()[i].eps;
            let q = &self.photon.boson.get_modes()[i].q;
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

    fn get_interaction_hamiltonian_as_operator(&self) -> M 
    where 
        M::Scalar: NumCast + Sqrt + ConvertFrom<c64> + ConvertFrom<f64>,  
    {
        use std::time::Instant;
        
        let cached_a = self.cached_a_ops.as_ref().expect("Photon operators not cached");
        let cached_adag_a = self.cached_adag_a_ops.as_ref().expect("Number operators not cached");
        
        let n_modes = self.photon.boson.get_modes().len();
        let total_pairs = n_modes * n_modes;
        timed_println!("        Building interaction for {} modes ({} pairs)", n_modes, total_pairs);
        
        let mut terms: Vec<M> = Vec::new();
        
        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let ex_eye:M = M::eye(n_ex);
        
        // Iterate over photon mode to build interaction terms
        self.photon.boson.get_modes().iter().enumerate().for_each(|(i_ph, mode_i)| {

            // p.A terms
            let a_i = &cached_a[i_ph];
            let p_c_sin = &self.p_c_sin.as_ref().unwrap()[i_ph].scale( 2.0 / self.exciton.exciton_ham.m * self.coupling);
            let p_q = &self.p_q.as_ref().unwrap()[i_ph].scale(1.0 / 2.0 / self.exciton.exciton_ham.m);

            let p_cos = &M::from_dense(&(self.matrix_elements.p_cos_mat.get(&vec![i_ph as i32]).unwrap() * (self.coupling / self.exciton.exciton_ham.rel_ham.mu)));
            
            // P•A sin(x) term
            terms.push(M::kron(p_c_sin, a_i));
            terms.push(M::kron(p_c_sin, &a_i.transpose()));

            // p•A cos(x) term
            terms.push(M::kron(p_cos, &(a_i.scale(-1.0))));
            terms.push(M::kron(p_cos, &a_i.transpose()));

            // P•q a†a term
            terms.push(M::kron(p_q, &cached_adag_a[i_ph]));
            
            // A² terms
            self.photon.boson.get_modes().iter().enumerate().for_each(|(j_ph, mode_j)| {
                
                let q_ind1 = (i_ph as i32).max(j_ph as i32);
                let q_ind2 = (i_ph as i32).min(j_ph as i32);
                let q_diff_key = vec![q_ind1, -q_ind2];

                let a_j = &cached_a[j_ph];

                let coupling_2 = self.coupling * self.coupling * mode_i.eps.dot(&mode_j.eps) / 2. / self.exciton.exciton_ham.rel_ham.mu;

                let eye = Array2::eye(n_ex);
                let cos_diff = &M::from_dense(self.matrix_elements.cos_mat.get(&q_diff_key).unwrap_or_else(|| {
                    warn!("No cos mat found for q_diff_key = {:?}, returning identity", q_diff_key);
                    &eye
                }));

                let a_diff = &( a_i.dot(&a_j.transpose()).add(&a_i.transpose().dot(a_j)) );

                // cos(q - q') term
                terms.push(M::kron(cos_diff, a_diff).scale(coupling_2));

                // (q a†a)^2 term
                let q_q = mode_i.q.dot(&mode_j.q) / 2.0 / self.exciton.exciton_ham.m;
                terms.push(M::kron(&ex_eye, &(cached_adag_a[i_ph].dot(&cached_adag_a[j_ph])) ).scale(q_q));
            });
        });
        
        // Sum all terms at once using the efficient Sum operator
        timed_println!("        Summing {} interaction terms...", terms.len());
        let sum_time = Instant::now();
        let interaction = M::sum(terms);
        timed_println!("        Summed terms in {:?}", sum_time.elapsed());
        
        interaction
    }
    
    fn _get_interaction_hamiltonian(&self, ham: &mut M) 
    where 
        M::Scalar: NumCast + Sqrt + ConvertFrom<c64> + ConvertFrom<f64>,  
    {
        use std::time::Instant;
        
        let cached_a = self.cached_a_ops.as_ref().expect("Photon operators not cached");
        let cached_adag_a = self.cached_adag_a_ops.as_ref().expect("Number operators not cached");
        
        let n_modes = self.photon.boson.get_modes().len();
        let total_pairs = n_modes * n_modes;
        timed_println!("        Building interaction for {} modes ({} pairs)", n_modes, total_pairs);
        
        // let start_time = Instant::now();
        // let mut last_print = Instant::now();
        // let mut pairs_processed = 0;
        
        // IMPORTANT: Collect all terms first, then sum at the end
        // This avoids the O(n²) tree growth problem where each ham.add() clones
        // the entire accumulated tree. By collecting terms in a Vec and summing
        // once, we create a balanced tree with O(log n) depth instead of O(n).
        let mut terms: Vec<M> = Vec::new();
        
        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let ex_eye:M = M::eye(n_ex);
        
        // Iterate over photon modes to build interaction terms
        self.photon.boson.get_modes().iter().enumerate().for_each(|(i_ph, mode_i)| {

            // p.A terms
            let a_i = &cached_a[i_ph];
            let p_c_sin = &self.p_c_sin.as_ref().unwrap()[i_ph].scale( 2.0 / self.exciton.exciton_ham.m * self.coupling);
            let p_q = &self.p_q.as_ref().unwrap()[i_ph].scale(1.0 / 2.0 / self.exciton.exciton_ham.m);

            let p_cos = &M::from_dense(&(self.matrix_elements.p_cos_mat.get(&vec![i_ph as i32]).unwrap() * (self.coupling / self.exciton.exciton_ham.rel_ham.mu)));
            
            // P•A sin(x) term
            terms.push(M::kron(p_c_sin, a_i));
            terms.push(M::kron(p_c_sin, &a_i.transpose()));

            // p•A cos(x) term
            terms.push(M::kron(p_cos, &(a_i.scale(-1.0))));
            terms.push(M::kron(p_cos, &a_i.transpose()));

            // P•q a†a term
            terms.push(M::kron(p_q, &cached_adag_a[i_ph]));
            
            // A² terms
            self.photon.boson.get_modes().iter().enumerate().for_each(|(j_ph, mode_j)| {
                
                // pairs_processed += 1;
                
                // // Print progress every 5 seconds
                // if last_print.elapsed().as_secs() >= 5 {
                //     let elapsed = start_time.elapsed().as_secs_f64();
                //     let rate = pairs_processed as f64 / elapsed;
                //     let remaining = (total_pairs - pairs_processed) as f64 / rate;
                //     timed_println!("        Progress: {}/{} pairs ({:.1}%), ETA: {:.0}s", 
                //         pairs_processed, total_pairs, 
                //         100.0 * pairs_processed as f64 / total_pairs as f64,
                //         remaining);
                //     last_print = Instant::now();
                // }
                
                let q_ind1 = (i_ph as i32).max(j_ph as i32);
                let q_ind2 = (i_ph as i32).min(j_ph as i32);
                let q_diff_key = vec![q_ind1, -q_ind2];
                let q_sum_key = vec![q_ind1, q_ind2];

                let a_j = &cached_a[j_ph];

                let coupling_2 = self.coupling * self.coupling * mode_i.eps.dot(&mode_j.eps) / 2. / self.exciton.exciton_ham.rel_ham.mu;

                let eye = Array2::eye(n_ex);
                let cos_diff = &M::from_dense(self.matrix_elements.cos_mat.get(&q_diff_key).unwrap_or_else(|| {
                    warn!("No cos mat found for q_diff_key = {:?}, returning identity", q_diff_key);
                    &eye
                }));
                let cos_sum  = &M::from_dense(self.matrix_elements.cos_mat.get(&q_sum_key ).unwrap_or_else(|| {
                    warn!("No cos mat found for q_sum_key =  {:?}, returning identity", q_sum_key);
                    &eye
                }));

                let a_diff = &( a_i.dot(&a_j.transpose()).add(&a_i.transpose().dot(a_j)) );
                let a_sum  = &( a_i.transpose().dot(&a_j.transpose()).add( &a_i.dot(a_j)) );

                // cos(q - q') term
                terms.push(M::kron(cos_diff, a_diff).scale(coupling_2));

                // cos(q + q') term
                terms.push(M::kron(cos_sum, a_sum).scale(-coupling_2));

                // (q a†a)^2 term
                let q_q = mode_i.q.dot(&mode_j.q) / 2.0 / self.exciton.exciton_ham.m;
                terms.push(M::kron(&ex_eye, &(cached_adag_a[i_ph].dot(&cached_adag_a[j_ph])) ).scale(q_q));
            });
        });
        
        // Sum all terms at once using the efficient Sum operator
        // This avoids creating a deep tree of binary Add nodes
        timed_println!("        Summing {} interaction terms...", terms.len());
        let sum_time = Instant::now();
        if !terms.is_empty() {
            let interaction = M::sum(terms);
            *ham = ham.add(&interaction);
        }
        timed_println!("        Summed terms in {:?}", sum_time.elapsed());
    }

    pub fn build_lm_hamiltonian(&mut self, k: Option<Vec<f64>>) -> M 
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>,  
    {
        use std::time::Instant;
        
        // Initialize
        self.initialize(k);
        
        // Get matrix dimensions
        let n_ex = self.exciton.e_vals.as_ref().unwrap().len();
        let n_ph: usize = self.photon.boson.dimension();
        
        let time_eye = Instant::now();
        let ex_eye: M = M::eye(n_ex);
        let ph_eye: M = M::eye(n_ph);
        timed_println!("      LM identity matrices: {:?}", time_eye.elapsed());

        // Collect all Hamiltonian terms
        let mut ham_terms: Vec<M> = Vec::new();

        // Exciton Hamiltonian
        let time_ex = Instant::now();
        ham_terms.push(M::kron(&self.exciton.get_h(), &ph_eye));
        timed_println!("      LM exciton kron: {:?}", time_ex.elapsed());

        // Photonic Hamiltonian
        let time_ph = Instant::now();
        ham_terms.push(M::kron(&ex_eye, &self.photon.get_h()));
        timed_println!("      LM photon kron: {:?}", time_ph.elapsed());

        // Interaction Hamiltonian
        let time_int = Instant::now();
        ham_terms.push(self.get_interaction_hamiltonian_as_operator());
        timed_println!("      LM interaction: {:?}", time_int.elapsed());

        // Build the final Hamiltonian by adding terms
        // Don't use M::sum() here because it would try to optimize the diagonal terms
        let time_sum = Instant::now();
        let ham = ham_terms.into_iter().reduce(|acc, term| acc.add(&term)).unwrap();
        // let ham = M::sum(ham_terms);
        timed_println!("      LM sum all terms: {:?}", time_sum.elapsed());

        ham
    }

    pub fn update_k(&mut self, k: Vec<f64>) {
        self.exciton.exciton_ham.k = k;
    }
}

pub struct PPHamiltonian <M: MatrixOps, B: Boson<M>> {
    pub polariton: LMHamiltonian<M,B>,
    pub phonon: BosonicHamiltonian<M, FullBosons<M>>,
    pub pn_coupling: f64,
    sin_pn: Option<Vec<M>>,
    is_initialized: bool,
    cached_lm_ham: Option<M>,
    cached_k: Option<Vec<f64>>,
}

impl<M: MatrixOps, B: Boson<M>>  PPHamiltonian<M,B> {

    pub fn new(polariton: LMHamiltonian<M,B>, phonon: BosonicHamiltonian<M,FullBosons<M>>, pn_coupling: f64) -> Self {
        Self { polariton, phonon, pn_coupling, sin_pn: None, is_initialized: false, cached_lm_ham: None, cached_k: None}
    }

    pub fn from_config(pp_config: &PPConfig) -> Self{
        
        // Create exciton
        let exciton_grid = Exciton2DGridSystem::from_config(pp_config.e_config.clone());
        let exciton: ExcitonicHamiltonian<M> = ExcitonicHamiltonian::new(exciton_grid.full_ham, pp_config.n_ex_eig, pp_config.n_rel);

        // Create photon
        let photon_modes: Vec<BosonMode> = pp_config.q_values.iter().map(|q| {
            BosonMode::new(Vector3D { x: q.x, y: q.y, z: q.z }, Polarization::TE, 2).clone()
        }).collect();
        let photon: BosonicHamiltonian<M,B> = BosonicHamiltonian::new(photon_modes);

        // Create Polariton
        let polariton = LMHamiltonian::new(exciton, photon, pp_config.lm_coupling);

        // Create Phonon
        let phonon_mode = BosonMode { eps: arr1(&[0.0,0.0]), q: pp_config.pn_k.clone(), 
                omega: pp_config.pn_omega, n_fock: pp_config.pn_n_fock };
        let phonon: BosonicHamiltonian<M,FullBosons<M>> = BosonicHamiltonian::new(vec![phonon_mode]);

        // Create Polaron Polariton
        let pn_coupling = pp_config.pn_coupling;
        PPHamiltonian::new(polariton, phonon, pn_coupling)
    }

    pub fn update_k(&mut self, k: Vec<f64>) 
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>, 
    {
        self.polariton.exciton.exciton_ham.k = k.clone();
        self.polariton.initialize(Some(k));
    } 

    pub fn initialize(&mut self)  
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>, 
    {
        use std::time::Instant;
        
        if self.is_initialized {
            return;
        }
        
        let time_pol = Instant::now();
        self.polariton.initialize(None);
        timed_println!("  Polariton initialization: {:?}", time_pol.elapsed());
        
        let time_pn = Instant::now();
        let pn_mat_calc = LMMatrixElementArray::new();
        let sin_pn = pn_mat_calc.phonon_mat_elements(  
            self.polariton.exciton.e_vecs.as_ref().unwrap(), 
            self.phonon.boson.get_modes(), 
            self.polariton.exciton.exciton_ham.rel_ham.mu,
            self.polariton.exciton.exciton_ham.kappa_vectors.len());
        timed_println!("  Phonon matrix elements: {:?}", time_pn.elapsed());

        let time_convert = Instant::now();
        self.sin_pn = Some(sin_pn.iter().map(|mat| M::from_dense(mat)).collect::<Vec<M>>());
        timed_println!("  Convert to sparse: {:?}", time_convert.elapsed());

        self.is_initialized = true;
    }

    fn get_interaction_hamiltonian(&self, ham: &mut M, ph_dim: usize) 
    where 
        M::Scalar: NumCast + Sqrt + Sum, 
    {
        let sin_pn = self.sin_pn.as_ref().unwrap();

        let ph_eye = M::eye(ph_dim);

        self.phonon.boson.get_modes().iter().enumerate().for_each(|(idx, _)|{
            let b = self.phonon.get_a(idx);
            let b_dag = b.transpose().scale(-1.0);
            let neg_b = b.scale(-1.0);

            // 2 gamma cos() (b† + b)
            let lm_op = sin_pn[idx].kron(&ph_eye);
            let pn_op = (neg_b.add(&b_dag)).scale(2.0 * self.pn_coupling);

            *ham = ham.add(&lm_op.kron(&pn_op));
        });
    }

    pub fn build_pp_hamiltonian(&mut self, k: Option<Vec<f64>>) -> M
    where 
        M::Scalar: NumCast + Sqrt + Sum + ConvertFrom<c64> + ConvertFrom<f64>, 
    {
        use std::time::Instant;
        
        // Check if k changed
        let k_changed = if let Some(k_val) = &k {
            Some(k_val) != self.cached_k.as_ref()
        } else {
            false
        };
        
        // Initialize if needed (this will also handle k updates)
        if k_changed {
            self.is_initialized = false;
            self.cached_lm_ham = None;
            self.cached_k = k.clone();
        }
        
        if !self.is_initialized{
            self.initialize();
        }

        // Use cached LM Hamiltonian if available and k hasn't changed
        let time_lm = Instant::now();
        let lm_ham = if let Some(cached) = &self.cached_lm_ham {
            timed_println!("  Using cached LM Hamiltonian");
            cached.clone()
        } else {
            let ham = self.polariton.build_lm_hamiltonian(k.clone());
            self.cached_lm_ham = Some(ham.clone());
            ham
        };
        timed_println!("  Build/retrieve LM Hamiltonian: {:?}", time_lm.elapsed());
        
        let ph_dim = self.polariton.photon.dimension();
        let pn_dim = self.phonon.dimension();
        let ex_dim = self.polariton.exciton.dimension();

        // If no phonons, just return the LM Hamiltonian
        // pn_dim == 0: no phonon modes, pn_dim == 1: single Fock state (no dynamics)
        if pn_dim == 0 || pn_dim == 1 || self.pn_coupling == 0.0 {
            timed_println!("  No phonons (pn_dim={}, coupling={}), returning LM Hamiltonian only", pn_dim, self.pn_coupling);
            return lm_ham;
        }
        
        let time_pn = Instant::now();
        let pn_ham = self.phonon.get_h();
        timed_println!("  Build phonon Hamiltonian: {:?}", time_pn.elapsed());

        timed_println!("  Dimensions: ex={}, ph={}, pn={}, total={}", ex_dim, ph_dim, pn_dim, ph_dim * ex_dim * pn_dim);

        let time_eye = Instant::now();
        let lm_eye = M::eye(ph_dim * ex_dim);
        let pn_eye = M::eye(pn_dim);
        timed_println!("  Create identity matrices: {:?}", time_eye.elapsed());

        let time_zeros = Instant::now();
        let mut ham = M::zeros((ph_dim * ex_dim * pn_dim, ph_dim * ex_dim * pn_dim));
        timed_println!("  Create zero matrix: {:?}", time_zeros.elapsed());

        let time_kron1 = Instant::now();
        ham = ham.add(&lm_ham.kron(&pn_eye));
        timed_println!("  First kron product (LM ⊗ I_pn): {:?}", time_kron1.elapsed());

        let time_kron2 = Instant::now();
        ham = ham.add(&lm_eye.kron(&pn_ham));
        timed_println!("  Second kron product (I_lm ⊗ pn): {:?}", time_kron2.elapsed());

        let time_int = Instant::now();
        self.get_interaction_hamiltonian(&mut ham, ph_dim);
        timed_println!("  Interaction Hamiltonian: {:?}", time_int.elapsed());

        ham
    }
}