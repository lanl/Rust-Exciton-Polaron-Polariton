use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone)]
pub struct DielectricParams {
    pub a0: f64,
    pub v0: f64,
    pub n_rel: usize,
    pub lm_coupling: f64,
    pub n_modes: usize,
    pub pn_coupling: f64,
    pub pn_n_fock: usize,
    pub n_eig: usize,
    pub energy_cutoff: Option<f64>,
    pub gamma: f64,
    pub method: String,
    pub n_x: Option<f64>,
    pub d_eff: Option<f64>,
    pub q_probe: Option<f64>,
    pub antiresonant: bool,
    pub lwa: bool,
}

pub fn extract_params_from_filename(path: &Path) -> Option<DielectricParams> {
    let filename = path.file_name()?.to_str()?;

    if !filename.starts_with("dielectric_") {
        return None;
    }

    let filename = filename.trim_end_matches(".dat");
    let parts: Vec<&str> = filename.split('_').collect();
    let mut params = DielectricParams {
        a0: 0.0,
        v0: 0.0,
        n_rel: 0,
        lm_coupling: 0.0,
        n_modes: 0,
        pn_coupling: 0.0,
        pn_n_fock: 0,
        n_eig: 0,
        energy_cutoff: None,
        gamma: 0.0,
        method: "legacy".to_string(),
        n_x: None,
        d_eff: None,
        q_probe: None,
        antiresonant: false,
        lwa: false,
    };

    for part in parts.iter() {
        if part.starts_with("a0") {
            params.a0 = part[2..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("V0") {
            params.v0 = part[2..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("nrel") {
            params.n_rel = part[4..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("lmcoup") {
            params.lm_coupling = part[6..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("nmodes") {
            params.n_modes = part[6..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("pncoup") {
            params.pn_coupling = part[6..].parse::<f64>().unwrap_or(0.0);
        } else if part.starts_with("npn") {
            params.pn_n_fock = part[3..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("neig") {
            params.n_eig = part[4..].parse::<usize>().unwrap_or(0);
        } else if part.starts_with("ecut") {
            params.energy_cutoff = part[4..].parse::<f64>().ok();
        } else if part.starts_with("gamma") {
            params.gamma = part[5..].parse::<f64>().unwrap_or(0.0);
        } else if let Some(method) = part.strip_prefix("meth") {
            params.method = match method {
                "sheet" => "sheet2d".to_string(),
                other => other.to_string(),
            };
        } else if let Some(n_x) = part.strip_prefix("nx") {
            params.n_x = n_x.parse::<f64>().ok();
        } else if let Some(d_eff) = part.strip_prefix("deff") {
            params.d_eff = d_eff.parse::<f64>().ok();
        } else if let Some(q_probe) = part.strip_prefix("qpr") {
            params.q_probe = q_probe.parse::<f64>().ok();
        } else if *part == "ar" {
            params.antiresonant = true;
        } else if *part == "lwa" {
            params.lwa = true;
        }
    }

    Some(params)
}

pub fn load_dielectric_data(
    path: &Path,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut omegas = Vec::new();
    let mut re_eps = Vec::new();
    let mut im_eps = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let values: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if values.len() >= 3 {
            omegas.push(values[0]);
            re_eps.push(values[1]);
            im_eps.push(values[2]);
        }
    }

    Ok((omegas, re_eps, im_eps))
}

pub fn load_dielectric_alpha(
    path: &Path,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut omegas = Vec::new();
    let mut re_alpha = Vec::new();
    let mut im_alpha = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let values: Vec<f64> = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;

        if values.len() < 5 {
            return Err(format!(
                "dielectric alpha data line has {} columns, expected at least 5",
                values.len()
            )
            .into());
        }

        omegas.push(values[0]);
        re_alpha.push(values[3]);
        im_alpha.push(values[4]);
    }

    Ok((omegas, re_alpha, im_alpha))
}
