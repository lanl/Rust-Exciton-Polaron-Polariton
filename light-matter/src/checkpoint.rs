use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};
use crate::modes::types::Polarization;
use crate::utils::quantize_f64_for_hash;
use matrix_ops::{CheckpointConfig, MatrixOps};
use std::path::PathBuf;
use std::sync::Mutex;

const KEY_VERSION: &str = "lm-diag-ckpt-v1";

/// Stable across Rust versions (unlike DefaultHasher).
struct Fnv64(u64);
impl Fnv64 {
    fn new() -> Self { Self(0xcbf2_9ce4_8422_2325) }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) { self.bytes(&v.to_le_bytes()) }
    fn i64(&mut self, v: i64) { self.bytes(&v.to_le_bytes()) }
    fn f64(&mut self, v: f64) { self.u64(v.to_bits()) }
    fn str(&mut self, s: &str) { self.u64(s.len() as u64); self.bytes(s.as_bytes()) }
}

fn operator_key(pp: &PPConfig, tag: &str, energy_cutoff: f64) -> u64 {
    let mut h = Fnv64::new();
    h.str(KEY_VERSION);
    h.str(tag);
    let e = &pp.e_config;
    for v in [e.a0, e.v0, e.m_e, e.m_h, e.box_size] { h.f64(v); }
    h.u64(e.n_kappa as u64);
    h.u64(e.grid_size as u64);
    h.u64(e.k.len() as u64);
    for &k in &e.k { h.i64(quantize_f64_for_hash(k)); }
    h.u64(pp.n_rel as u64);
    h.u64(pp.n_ex_eig as u64);
    h.f64(pp.omega_0);
    h.u64(pp.lm_couplings.len() as u64);
    for &c in &pp.lm_couplings { h.f64(c); }
    h.u64(pp.q_values.len() as u64);
    for q in &pp.q_values { h.f64(q.x); h.f64(q.y); h.f64(q.z); }
    for p in &pp.mode_polarizations {
        h.u64(match p { Polarization::TE => 0, Polarization::TM => 1 });
    }
    h.f64(pp.pn_coupling);
    h.f64(pp.pn_omega);
    for &v in pp.pn_k.iter() { h.f64(v); }
    h.u64(pp.pn_n_fock as u64);
    h.u64(pp.include_photon_recoil as u64);
    h.f64(energy_cutoff);
    h.0
}

pub struct DiagCheckpoint {
    pub dir: PathBuf,
    pub resume: bool,
    pub keep: bool,
    used_roots: Mutex<Vec<PathBuf>>,
}

impl DiagCheckpoint {
    pub fn new(dir: PathBuf, resume: bool, keep: bool) -> Self {
        Self { dir, resume, keep, used_roots: Mutex::new(Vec::new()) }
    }

    pub fn config_for(&self, pp_config: &PPConfig, tag: &str, energy_cutoff: f64) -> CheckpointConfig {
        let cfg = CheckpointConfig {
            dir: self.dir.clone(),
            key: operator_key(pp_config, tag, energy_cutoff),
            resume: self.resume,
        };
        self.used_roots.lock().unwrap().push(cfg.root());
        crate::timed_println_rank0!(
            "Diagonalization checkpoint [{}]: {} (resume = {})",
            tag, cfg.root().display(), cfg.resume
        );
        cfg
    }

    /// Builds the config AND points the exciton basis cache at the checkpoint dir.
    /// Must be called BEFORE `pp.build_pp_hamiltonian(..)`.
    pub fn prepare<M: MatrixOps, B: Boson<M>>(
        &self,
        pp: &mut PPHamiltonian<M, B>,
        pp_config: &PPConfig,
        tag: &str,
        energy_cutoff: f64,
    ) -> CheckpointConfig {
        let cfg = self.config_for(pp_config, tag, energy_cutoff);
        pp.polariton.exciton.basis_cache = Some(Self::exciton_basis_path(&cfg));
        cfg
    }

    pub fn exciton_basis_path(cfg: &CheckpointConfig) -> PathBuf {
        cfg.root().join("exciton_basis.bin")
    }

    /// Call only once the reduced result is safely on disk. Rank 0 deletes the
    /// checkpoint directories used by this run (unless --keep-checkpoint).
    pub fn finish(&self) {
        if self.keep || !sim_common::mpi::is_mpi_rank0_env() {
            return;
        }
        for root in self.used_roots.lock().unwrap().drain(..) {
            match std::fs::remove_dir_all(&root) {
                Ok(_) => crate::timed_println!("Removed checkpoint {}", root.display()),
                Err(e) => crate::timed_println!("Could not remove {}: {}", root.display(), e),
            }
        }
    }
}