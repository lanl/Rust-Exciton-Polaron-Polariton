use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};
use matrix_ops::MatrixOps;
use ndarray::Array1;

pub fn get_dielectric_cache_filename<M, B>(
    pp: &PPHamiltonian<M, B>,
    pp_config: &PPConfig,
    save_dir: &str,
    n_eig: usize,
) -> String
where
    M: MatrixOps,
    B: Boson<M>,
{
    // Use effective mode parameters if available (hand-entered, no floating-point issues)
    if let (Some(d_omega), Some(omega_max), Some(reflectivity)) = (
        pp_config.d_omega,
        pp_config.omega_max,
        pp_config.reflectivity,
    ) {
        format!(
            "{}/dielectric_cache_a0{}_V0{}_nrel{}_omega0{:.6}_domega{:.1e}_omegamax{:.6}_refl{:.7}_nmodes{}_pncoup{:.6}_npn{}_neig{}.bin",
            save_dir,
            pp_config.e_config.a0,
            pp_config.e_config.v0,
            pp_config.n_rel,
            pp_config.omega_0,
            d_omega,
            omega_max,
            reflectivity,
            pp.polariton.photon.boson.get_modes().len(),
            pp_config.pn_coupling,
            pp_config.pn_n_fock,
            n_eig
        )
    } else {
        // Fallback to coupling sum (for backward compatibility or non-effective-mode cases)
        let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();
        format!(
            "{}/dielectric_cache_a0{}_V0{}_nrel{}_lmcoup{:.6}_nmodes{}_pncoup{:.6}_npn{}_neig{}.bin",
            save_dir,
            pp_config.e_config.a0,
            pp_config.e_config.v0,
            pp_config.n_rel,
            total_coupling,
            pp.polariton.photon.boson.get_modes().len(),
            pp_config.pn_coupling,
            pp_config.pn_n_fock,
            n_eig
        )
    }
}

pub fn get_dielectric_function_filename<M, B>(
    pp: &PPHamiltonian<M, B>,
    pp_config: &PPConfig,
    save_dir: &str,
    n_eig: usize,
    gamma: f64,
) -> String
where
    M: MatrixOps,
    B: Boson<M>,
{
    // Use effective mode parameters if available (hand-entered, no floating-point issues)
    if let (Some(d_omega), Some(omega_max), Some(reflectivity)) = (
        pp_config.d_omega,
        pp_config.omega_max,
        pp_config.reflectivity,
    ) {
        let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();

        format!(
            "{}/dielectric_a0{}_V0{}_nrel{}_omega0{:.6}_domega{:.1e}_omegamax{:.6}_refl{:.7}_lmcoup{:.6}_nmodes{}_pncoup{:.6}_npn{}_neig{}_gamma{:.1e}.dat",
            save_dir,
            pp_config.e_config.a0,
            pp_config.e_config.v0,
            pp_config.n_rel,
            pp_config.omega_0,
            d_omega,
            omega_max,
            reflectivity,
            total_coupling,
            pp.polariton.photon.boson.get_modes().len(),
            pp_config.pn_coupling,
            pp_config.pn_n_fock,
            n_eig,
            gamma
        )
    } else {
        // Fallback to coupling sum (for backward compatibility or non-effective-mode cases)
        let total_coupling = pp_config.lm_couplings.iter().sum::<f64>();
        format!(
            "{}/dielectric_a0{}_V0{}_nrel{}_lmcoup{:.6}_nmodes{}_pncoup{:.6}_npn{}_neig{}_gamma{:.1e}.dat",
            save_dir,
            pp_config.e_config.a0,
            pp_config.e_config.v0,
            pp_config.n_rel,
            total_coupling,
            pp.polariton.photon.boson.get_modes().len(),
            pp_config.pn_coupling,
            pp_config.pn_n_fock,
            n_eig,
            gamma
        )
    }
}

pub fn save_dielectric_cache(
    filename: &str,
    pp_vals: &Array1<f64>,
    x_pp: &Array1<f64>,
    photonic_chars: &Vec<f64>,
    phonon_numbers: &Vec<f64>,
    ground_state_index: usize,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;

    // Write dimensions (number of eigenstates)
    let n_eig = pp_vals.len() as u64;
    file.write_all(&n_eig.to_le_bytes())?;

    // Write pp_vals
    for &val in pp_vals.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    // Write x_pp
    for &val in x_pp.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    // Write photonic_chars
    for &val in photonic_chars.iter() {
        file.write_all(&val.to_le_bytes())?;
    }
    // Write phonon_numbers
    for &val in phonon_numbers.iter() {
        file.write_all(&val.to_le_bytes())?;
    }

    // Write ground state index (usize stored as u64 for consistency)
    let gs_idx = ground_state_index as u64;
    file.write_all(&gs_idx.to_le_bytes())?;

    Ok(())
}

pub fn load_dielectric_cache(
    filename: &str,
) -> std::io::Result<(Array1<f64>, Array1<f64>, Vec<f64>, Vec<f64>, usize)> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(filename)?;

    // Read dimensions
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    let n_eig = u64::from_le_bytes(buf) as usize;

    // Read pp_vals
    let mut pp_vals = Array1::<f64>::zeros(n_eig);
    for i in 0..n_eig {
        file.read_exact(&mut buf)?;
        pp_vals[i] = f64::from_le_bytes(buf);
    }
    // Read x_pp
    let mut x_pp = Array1::<f64>::zeros(n_eig);
    for i in 0..n_eig {
        file.read_exact(&mut buf)?;
        x_pp[i] = f64::from_le_bytes(buf);
    }
    // ---------------------------------------------------------------------
    // Read photonic_chars – if the file ends early we fall back to zeros.
    // This makes the loader compatible with caches created before the
    // composition vectors were added (older two‑vector format).
    // ---------------------------------------------------------------------
    let mut photonic_chars = Vec::with_capacity(n_eig);
    for _ in 0..n_eig {
        match file.read_exact(&mut buf) {
            Ok(_) => photonic_chars.push(f64::from_le_bytes(buf)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Old cache – fill the rest with zeros and break.
                photonic_chars.resize(n_eig, 0.0);
                break;
            }
            Err(e) => return Err(e),
        }
    }

    // ---------------------------------------------------------------------
    // Read phonon_numbers – same fallback logic as above.
    // ---------------------------------------------------------------------
    let mut phonon_numbers = Vec::with_capacity(n_eig);
    for _ in 0..n_eig {
        match file.read_exact(&mut buf) {
            Ok(_) => phonon_numbers.push(f64::from_le_bytes(buf)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                phonon_numbers.resize(n_eig, 0.0);
                break;
            }
            Err(e) => return Err(e),
        }
    }

    // Finally, read the ground state index (stored as u64)
    let mut gs_buf = [0u8; 8];
    let ground_state_index = match file.read_exact(&mut gs_buf) {
        Ok(_) => u64::from_le_bytes(gs_buf) as usize,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            // Older cache without index – default to 0
            0
        }
        Err(e) => return Err(e),
    };

    Ok((
        pp_vals,
        x_pp,
        photonic_chars,
        phonon_numbers,
        ground_state_index,
    ))
}
