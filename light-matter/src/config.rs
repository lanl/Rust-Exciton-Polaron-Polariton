use crate::modes::types::Polarization;
use crate::types::Vector3D;
use exciton_2d::ExcitonConfig;
use ndarray::Array1;

#[derive(Clone)]
pub struct PPConfig {
    pub e_config: ExcitonConfig,
    pub n_rel: usize,
    pub n_ex_eig: usize,

    pub lm_couplings: Vec<f64>,
    pub mode_polarizations: Vec<Polarization>,
    pub omega_0: f64,
    pub q_values: Vec<Vector3D>,

    pub pn_coupling: f64,
    pub pn_omega: f64,
    pub pn_k: Array1<f64>,
    pub pn_n_fock: usize,

    pub n_k_points: usize,
    pub n_bands: usize,

    pub cache_dir: Option<String>,

    pub d_omega: Option<f64>,
    pub omega_max: Option<f64>,
    pub reflectivity: Option<f64>,
}

impl PPConfig {
    pub fn new(
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,
        n_k_points: usize,
        n_bands: usize,
    ) -> Self {
        let mode_polarizations = vec![Polarization::TE; q_values.len()];

        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            mode_polarizations,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache".to_string()),
            d_omega: None,
            omega_max: None,
            reflectivity: None,
        }
    }

    pub fn new_with_polarizations(
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,
        mode_polarizations: Vec<Polarization>,
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,
        n_k_points: usize,
        n_bands: usize,
    ) -> Self {
        assert_eq!(
            q_values.len(),
            mode_polarizations.len(),
            "q_values and mode_polarizations must have the same length"
        );
        assert_eq!(
            q_values.len(),
            lm_couplings.len(),
            "q_values and lm_couplings must have the same length"
        );

        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            mode_polarizations,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache".to_string()),
            d_omega: None,
            omega_max: None,
            reflectivity: None,
        }
    }

    pub fn new_with_effective_modes(
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,
        n_k_points: usize,
        n_bands: usize,
        d_omega: f64,
        omega_max: f64,
        reflectivity: f64,
    ) -> Self {
        let mode_polarizations = vec![Polarization::TE; q_values.len()];

        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            mode_polarizations,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache".to_string()),
            d_omega: Some(d_omega),
            omega_max: Some(omega_max),
            reflectivity: Some(reflectivity),
        }
    }

    pub fn new_with_effective_modes_and_polarizations(
        e_config: ExcitonConfig,
        n_rel: usize,
        n_ex_eig: usize,
        lm_couplings: Vec<f64>,
        omega_0: f64,
        q_values: Vec<Vector3D>,
        mode_polarizations: Vec<Polarization>,
        pn_coupling: f64,
        pn_omega: f64,
        pn_k: Array1<f64>,
        pn_n_fock: usize,
        n_k_points: usize,
        n_bands: usize,
        d_omega: f64,
        omega_max: f64,
        reflectivity: f64,
    ) -> Self {
        assert_eq!(
            q_values.len(),
            mode_polarizations.len(),
            "q_values and mode_polarizations must have the same length"
        );
        assert_eq!(
            q_values.len(),
            lm_couplings.len(),
            "q_values and lm_couplings must have the same length"
        );

        Self {
            e_config,
            n_rel,
            n_ex_eig,
            lm_couplings,
            omega_0,
            q_values,
            mode_polarizations,
            pn_coupling,
            pn_omega,
            pn_k,
            pn_n_fock,
            n_k_points,
            n_bands,
            cache_dir: Some("/lustre/scratch5/mike_tay/saved_data_lm/matrix_element_cache".to_string()),
            d_omega: Some(d_omega),
            omega_max: Some(omega_max),
            reflectivity: Some(reflectivity),
        }
    }

    pub fn get_config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        self.e_config.a0.to_bits().hash(&mut hasher);
        self.e_config.v0.to_bits().hash(&mut hasher);
        self.e_config.m_e.to_bits().hash(&mut hasher);
        self.e_config.m_h.to_bits().hash(&mut hasher);
        self.n_rel.hash(&mut hasher);
        self.n_ex_eig.hash(&mut hasher);

        for &coupling in &self.lm_couplings {
            coupling.to_bits().hash(&mut hasher);
        }

        self.omega_0.to_bits().hash(&mut hasher);

        for q in &self.q_values {
            q.x.to_bits().hash(&mut hasher);
            q.y.to_bits().hash(&mut hasher);
            q.z.to_bits().hash(&mut hasher);
        }

        for pol in &self.mode_polarizations {
            let pol_code: u8 = match pol {
                Polarization::TE => 0,
                Polarization::TM => 1,
            };
            pol_code.hash(&mut hasher);
        }

        self.pn_coupling.to_bits().hash(&mut hasher);
        self.pn_omega.to_bits().hash(&mut hasher);

        for &val in self.pn_k.iter() {
            val.to_bits().hash(&mut hasher);
        }

        self.pn_n_fock.hash(&mut hasher);

        hasher.finish()
    }
}
