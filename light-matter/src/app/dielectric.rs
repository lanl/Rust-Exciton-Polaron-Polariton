use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, MatrixOpsError, Result};
use ndarray::{Array1, arr1};
use ndarray_linalg::c64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use super::config::build_dielectric_pp_config_for_reflectivity;
use crate::cli;
use crate::dielectric::DielectricNormalization;
use crate::hamiltonian::{PPHamiltonian, SingExBosons};
use crate::visualization::generate_stacked_plots;
use crate::visualization::{DielectricPlotType, plot_dielectric_function};
use crate::{dielectric, timed_println, timed_println_rank0, visualization};

fn run_dielectric_scan(e_config: ExcitonConfig, args: cli::DielectricArgs) -> Result<()> {
    timed_println_rank0!("Starting dielectric function scan...");

    validate_dielectric_args(&args)?;

    configure_petsc_slepc_diagnostics(&args)?;

    let calculate = args.calculate;
    let use_cached = args.use_cached;
    let zoomed = args.zoomed;

    let n_rel = args.n_rel;
    let n_ex_eig = args.n_ex_eig;

    let omega_0 = args.omega_0;
    let n_modes = args.n_modes;

    let omega_max = args.omega_max;
    let reflectivities = args.reflectivities.clone();
    let d_omega = args.d_omega;

    let pn_coupling = args.pn_coupling;
    let pn_omega = args.pn_omega;
    let pn_k = arr1(&[args.pn_k_x, 0.0]);
    let pn_n_fock = args.pn_n_fock;

    let n_eig = args.n_eig;
    let energy_cutoff = args.energy_cutoff;
    let omegas = Array1::linspace(
        args.omega_min_scan,
        args.omega_max_scan,
        args.n_omega_points,
    );

    let gamma = args.gamma.unwrap_or(d_omega);

    let omega_range = match (args.plot_omega_min, args.plot_omega_max) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => {
            if !zoomed {
                Some((0.152, 0.21))
            } else {
                Some((0.16, 0.17))
            }
        }
    };

    let save_dir = args.output_dir.as_str();
    std::fs::create_dir_all(save_dir).unwrap_or_else(|_| {
        timed_println_rank0!(
            "Directory {} already exists or couldn't be created",
            save_dir
        );
    });

    timed_println_rank0!("Dielectric CLI configuration:");
    timed_println_rank0!("  calculate       = {}", calculate);
    timed_println_rank0!("  use_cached      = {}", use_cached);
    timed_println_rank0!("  zoomed          = {}", zoomed);
    timed_println_rank0!("  n_rel           = {}", n_rel);
    timed_println_rank0!("  n_ex_eig        = {}", n_ex_eig);
    timed_println_rank0!("  n_eig           = {}", n_eig);
    timed_println_rank0!(
        "  energy_cutoff   = {}",
        match energy_cutoff {
            Some(c) => format!("{:.6e} (overrides n_eig)", c),
            None => "none".to_string(),
        }
    );
    timed_println_rank0!("  omega_0         = {}", omega_0);
    timed_println_rank0!("  omega_max       = {}", omega_max);
    timed_println_rank0!("  d_omega         = {}", d_omega);
    timed_println_rank0!("  gamma           = {}", gamma);
    timed_println_rank0!("  n_modes target  = {}", n_modes);
    timed_println_rank0!("  reflectivities  = {:?}", reflectivities);
    timed_println_rank0!("  pn_coupling     = {}", pn_coupling);
    timed_println_rank0!("  pn_omega        = {}", pn_omega);
    timed_println_rank0!("  pn_k            = {:?}", pn_k);
    timed_println_rank0!("  pn_n_fock       = {}", pn_n_fock);
    timed_println_rank0!(
        "  omega scan      = [{}, {}], n={}",
        args.omega_min_scan,
        args.omega_max_scan,
        args.n_omega_points
    );
    timed_println_rank0!("  output_dir      = {}", save_dir);

    // Generate photon modes and couplings for each reflectivity/mode-construction choice.
    let mut all_q_values = Vec::new();
    let mut all_mode_couplings = Vec::new();
    let mut all_mode_polarizations = Vec::new();
    let mut total_coupling_values = Vec::new();
    let mut d_q = None;

    for &reflectivity in reflectivities.iter() {
        let mode_set = crate::modes::photon::from_dielectric_args(&args, reflectivity);
        let total_coupling: f64 = mode_set.couplings.iter().sum();
        d_q.get_or_insert(mode_set.d_q);

        timed_println_rank0!(
            "Reflectivity {}: generated {} modes with total coupling {:.6} using {:?}",
            reflectivity,
            mode_set.q_values.len(),
            total_coupling,
            args.photon_modes
        );

        all_q_values.push(mode_set.q_values);
        all_mode_couplings.push(mode_set.couplings);
        all_mode_polarizations.push(mode_set.polarizations);
        total_coupling_values.push(total_coupling);
    }

    let total_coupling_values = Array1::from_vec(total_coupling_values);
    let d_q = d_q.unwrap_or(0.0);
    let q_probe = args.probe_q.unwrap_or(args.probe_q_in_dq * d_q);
    if matches!(args.dielectric_method, cli::DielectricMethodArg::Sheet2d)
        && (e_config.m_e - e_config.m_h).abs() > 1e-12
    {
        return Err(MatrixOpsError::Other {
            message: "sheet2d K-boost assumes m_e == m_h (X = (x_e+x_h)/2)".to_string(),
        });
    }
    if matches!(args.dielectric_method, cli::DielectricMethodArg::Sheet2d) {
        let closest = all_q_values
            .first()
            .and_then(|q_values| {
                q_values.iter().min_by(|left, right| {
                    let left_distance =
                        ((left.x - q_probe).powi(2) + left.y.powi(2)).sqrt();
                    let right_distance =
                        ((right.x - q_probe).powi(2) + right.y.powi(2)).sqrt();
                    left_distance
                        .partial_cmp(&right_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
        if let Some(mode) = closest {
            let distance = ((mode.x - q_probe).powi(2) + mode.y.powi(2)).sqrt();
            timed_println_rank0!(
                "sheet2d: closest photon mode to ({}, 0) is ({}, {}) with in-plane distance {}",
                q_probe,
                mode.x,
                mode.y,
                distance
            );
        }
        timed_println_rank0!(
            "sheet2d: anti-resonant term uses E_n(+q) for E_n(-q)"
        );
    }
    let norm = match args.dielectric_method {
        cli::DielectricMethodArg::Legacy => DielectricNormalization::Legacy,
        cli::DielectricMethodArg::Slab => DielectricNormalization::Slab {
            n_x: args.exciton_density,
            d: args.slab_thickness_bohr,
        },
        cli::DielectricMethodArg::Sheet2d => DielectricNormalization::Sheet2D {
            n_x: args.exciton_density,
            q: q_probe,
        },
    };
    let method_name = match args.dielectric_method {
        cli::DielectricMethodArg::Legacy => "legacy",
        cli::DielectricMethodArg::Slab => "slab",
        cli::DielectricMethodArg::Sheet2d => "sheet2d",
    };
    let response_tag = dielectric::method_tag(norm, args.antiresonant, args.long_wavelength);
    timed_println_rank0!("  dielectric_method = {}", method_name);
    timed_println_rank0!("  exciton_density = {}", args.exciton_density);
    timed_println_rank0!("  slab_thickness = {}", args.slab_thickness_bohr);
    timed_println_rank0!("  antiresonant   = {}", args.antiresonant);
    timed_println_rank0!("  long_wavelength = {}", args.long_wavelength);
    timed_println_rank0!("  d_q = {}, probe q = {}", d_q, q_probe);

    // Validate mode counts.
    if let Some(first) = all_q_values.first() {
        let n_modes_first = first.len();

        for (idx, q_vals) in all_q_values.iter().enumerate() {
            if q_vals.len() != n_modes_first {
                timed_println_rank0!(
                    "Warning: Reflectivity {} generated {} modes, but first generated {}.",
                    reflectivities[idx],
                    q_vals.len(),
                    n_modes_first
                );
            }
        }
    }

    // Output plot filenames.
    let suffix = if zoomed { "_zoomed.svg" } else { ".svg" };
    let lwa_tag = if args.long_wavelength { "_lwa" } else { "" };

    let real_filename = format!(
        "{}/dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, lwa_tag, suffix,
    );

    let imag_filename = format!(
        "{}/dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, lwa_tag, suffix,
    );

    let both_filename = format!(
        "{}/dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, lwa_tag, suffix,
    );

    if !calculate {
        if !sim_common::mpi::is_mpi_rank0_env() {
            return Ok(());
        }

        timed_println!("Plotting dielectric function from existing data...");

        let q_values = &all_q_values[0];

        let ref_params = visualization::DielectricParams {
            a0: e_config.a0,
            v0: e_config.v0,
            n_rel,
            lm_coupling: 0.0,
            n_modes: q_values.len(),
            pn_coupling,
            pn_n_fock,
            // Filenames with a cutoff carry no "neig" field, so the parser yields 0.
            n_eig: if energy_cutoff.is_some() { 0 } else { n_eig },
            energy_cutoff,
            gamma,
            method: method_name.to_string(),
            n_x: match norm {
                DielectricNormalization::Slab { n_x, .. }
                | DielectricNormalization::Sheet2D { n_x, .. } => Some(n_x),
                DielectricNormalization::Legacy => None,
            },
            d_eff: match norm {
                DielectricNormalization::Slab { d, .. } => Some(d),
                _ => None,
            },
            q_probe: match norm {
                DielectricNormalization::Sheet2D { q, .. } => Some(q),
                _ => None,
            },
            antiresonant: args.antiresonant,
            lwa: args.long_wavelength,
        };

        match plot_dielectric_function(
            save_dir,
            Some(&real_filename),
            DielectricPlotType::Real,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            Some(q_values),
        ) {
            Ok(_) => timed_println!("Successfully plotted real part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        match plot_dielectric_function(
            save_dir,
            Some(&imag_filename),
            DielectricPlotType::Imaginary,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            None,
        ) {
            Ok(_) => timed_println!("Successfully plotted imaginary part of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        match plot_dielectric_function(
            save_dir,
            Some(&both_filename),
            DielectricPlotType::Both,
            omega_range,
            &total_coupling_values,
            Some(&ref_params),
            zoomed,
            Some(q_values),
        ) {
            Ok(_) => timed_println!("Successfully plotted both parts of dielectric function"),
            Err(e) => timed_println!("Error plotting dielectric function: {}", e),
        }

        if matches!(args.dielectric_method, cli::DielectricMethodArg::Sheet2d) {
            timed_println!("Skipping stacked plots for sheet2d");
        } else {
            generate_stacked_plots(
                &total_coupling_values,
                &all_mode_couplings,
                &all_q_values,
                &all_mode_polarizations,
                &reflectivities,
                &e_config,
                n_rel,
                n_ex_eig,
                omega_0,
                d_omega,
                omega_max,
                pn_coupling,
                pn_omega,
                &pn_k,
                pn_n_fock,
                n_eig,
                energy_cutoff,
                &omegas,
                gamma,
                save_dir,
                zoomed,
                omega_range,
                norm,
                args.antiresonant,
                args.long_wavelength,
            );
        }

        if !matches!(args.dielectric_method, cli::DielectricMethodArg::Legacy) {
            let alpha_filename = format!(
                "{}/dielectric_alpha_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
                save_dir,
                e_config.a0,
                e_config.v0,
                n_rel,
                n_modes,
                pn_coupling,
                pn_n_fock,
                suffix,
            );
            match plot_dielectric_function(
                save_dir,
                Some(&alpha_filename),
                DielectricPlotType::Alpha2D,
                omega_range,
                &total_coupling_values,
                Some(&ref_params),
                zoomed,
                Some(q_values),
            ) {
                Ok(_) => timed_println!("Successfully plotted alpha2D"),
                Err(e) => timed_println!("Error plotting alpha2D: {}", e),
            }
        }

        return Ok(());
    }

    timed_println_rank0!(
        "Starting dielectric function scan for {} reflectivity values",
        reflectivities.len()
    );

    // Select exactly one reflectivity for this process group.
    // PETSc/SLEPc solves are collective over COMM_WORLD, so use job arrays for scan parallelism.
    let refl_idx: usize = args
        .reflectivity_index
        .or_else(|| sim_common::mpi::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0);

    if refl_idx >= reflectivities.len() {
        panic!(
            "Reflectivity index {} out of range 0..{}",
            refl_idx,
            reflectivities.len()
        );
    }

    let total_coupling = total_coupling_values[refl_idx];
    let q_values = &all_q_values[refl_idx];
    let reflectivity = reflectivities[refl_idx];

    timed_println_rank0!(
        "Selected reflectivity index {}: R={}, modes={}, total coupling={:.6}",
        refl_idx,
        reflectivity,
        q_values.len(),
        total_coupling
    );

    let start_time = Instant::now();

    let pp_config = build_dielectric_pp_config_for_reflectivity(e_config.clone(), &args, refl_idx);

    timed_println_rank0!("Using ComposedOperator + distributed SLEPc shell backend");

    let filename = dielectric::get_dielectric_function_filename_from_config(
        &pp_config,
        save_dir,
        n_eig,
        energy_cutoff,
        gamma,
        &response_tag,
    );
    let result = if matches!(args.dielectric_method, cli::DielectricMethodArg::Sheet2d) {
        let cache_filename = dielectric::get_sheet_dielectric_cache_filename(
            &pp_config,
            save_dir,
            q_probe,
            n_eig,
            energy_cutoff,
        );
        let cache_exists = std::path::Path::new(&cache_filename).exists();
        let (cache, save_cache) = if use_cached && cache_exists {
            match sim_common::cache::load_bincode::<dielectric::SheetDielectricCache>(
                &cache_filename,
            ) {
                Ok(cache)
                    if cache.version == dielectric::SHEET_CACHE_VERSION
                        && cache.ground_state_index_q < cache.energies_q.len()
                        && (cache.q_probe - q_probe).abs() < 1e-12
                        && cache.include_photon_recoil == pp_config.include_photon_recoil =>
                {
                    timed_println_rank0!(
                        "Successfully loaded cached sheet2d data ({} states)",
                        cache.energies_q.len()
                    );
                    (cache, false)
                }
                Ok(_) => {
                    timed_println_rank0!(
                        "Sheet2D cache metadata is inconsistent, recalculating..."
                    );
                    (
                        dielectric::compute_sheet_dielectric_cache::<
                            ComposedOperator<c64>,
                            SingExBosons<ComposedOperator<c64>>,
                        >(&pp_config, q_probe, n_eig, energy_cutoff)?,
                        true,
                    )
                }
                Err(error) => {
                    timed_println_rank0!(
                        "Failed to load sheet2d cache ({}), recalculating...",
                        error
                    );
                    (
                        dielectric::compute_sheet_dielectric_cache::<
                            ComposedOperator<c64>,
                            SingExBosons<ComposedOperator<c64>>,
                        >(&pp_config, q_probe, n_eig, energy_cutoff)?,
                        true,
                    )
                }
            }
        } else {
            (
                dielectric::compute_sheet_dielectric_cache::<
                    ComposedOperator<c64>,
                    SingExBosons<ComposedOperator<c64>>,
                >(&pp_config, q_probe, n_eig, energy_cutoff)?,
                true,
            )
        };

        if sim_common::mpi::is_mpi_rank0_env() && save_cache {
            if let Err(error) = sim_common::cache::save_bincode_atomic(&cache_filename, &cache) {
                timed_println_rank0!("Warning: Failed to save sheet2d cache: {}", error);
            } else {
                timed_println_rank0!("Successfully saved sheet2d cache");
            }
        }

        dielectric::sheet_dielectric_result(
            &cache,
            &omegas,
            gamma,
            args.exciton_density,
            q_probe,
            args.antiresonant,
        )
    } else {
        let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
            PPHamiltonian::from_config(&pp_config);
        dielectric::get_dielectric_function(
            &pp_config,
            &mut pp,
            n_eig,
            energy_cutoff,
            omegas.clone(),
            gamma,
            save_dir,
            use_cached,
            norm,
            args.antiresonant,
        )?
    };

    if sim_common::mpi::is_mpi_rank0_env() {
        timed_println!("Saving dielectric function data to {}", filename);

        let mut file = File::create(&filename).unwrap();

        if let (Some(alpha_re), Some(alpha_im)) = (&result.alpha_re, &result.alpha_im) {
            match norm {
                DielectricNormalization::Slab { n_x, d } => writeln!(
                    file,
                    "# method=slab n_x={:.3e} d={:.3} antiresonant={} lwa={}",
                    n_x, d, args.antiresonant, args.long_wavelength
                )
                .unwrap(),
                DielectricNormalization::Sheet2D { n_x, q } => writeln!(
                    file,
                    "# method=sheet2d n_x={:.3e} q={:.4e} antiresonant={} lwa={}",
                    n_x, q, args.antiresonant, args.long_wavelength
                )
                .unwrap(),
                DielectricNormalization::Legacy => unreachable!(),
            }
            writeln!(file, "# omega Re[eps] Im[eps] Re[alpha2D] Im[alpha2D]").unwrap();
            for i in 0..omegas.len() {
                writeln!(
                    file,
                    "{:.10e} {:.10e} {:.10e} {:.10e} {:.10e}",
                    omegas[i], result.real[i], result.imag[i], alpha_re[i], alpha_im[i]
                )
                .unwrap();
            }
        } else {
            writeln!(file, "# omega Re[eps] Im[eps]").unwrap();
            for i in 0..omegas.len() {
                writeln!(
                    file,
                    "{:.6} {:.6} {:.6}",
                    omegas[i], result.real[i], result.imag[i]
                )
                .unwrap();
            }
        }
    }

    timed_println_rank0!(
        "Completed reflectivity index {} in {:?}",
        refl_idx,
        start_time.elapsed()
    );

    Ok(())
}

pub(super) fn run_dielectric_cli(args: cli::DielectricArgs) -> Result<()> {
    let e_config = args.exciton.to_config();
    run_dielectric_scan(e_config, args)
}

fn configure_petsc_slepc_diagnostics(args: &cli::DielectricArgs) -> matrix_ops::Result<()> {
    let opts = slepc::petsc::Options::global();

    if args.petsc_log_view {
        opts.set_flag("-log_view")
            .map_err(|e| matrix_ops::MatrixOpsError::slepc(e, "setting -log_view"))?;
    }

    if args.eps_monitor {
        opts.set_flag("-eps_monitor")
            .map_err(|e| matrix_ops::MatrixOpsError::slepc(e, "setting -eps_monitor"))?;

        opts.set_flag("-eps_converged_reason")
            .map_err(|e| {
                matrix_ops::MatrixOpsError::slepc(e, "setting -eps_converged_reason")
            })?;
    }

    if args.eps_view {
        opts.set_flag("-eps_view")
            .map_err(|e| matrix_ops::MatrixOpsError::slepc(e, "setting -eps_view"))?;
    }

    if args.mat_view_info {
        opts.set_flag("-mat_view_info")
            .map_err(|e| matrix_ops::MatrixOpsError::slepc(e, "setting -mat_view_info"))?;
    }

    Ok(())
}

fn validate_dielectric_args(args: &cli::DielectricArgs) -> Result<()> {
    let invalid_positive = |name: &str, value: f64| MatrixOpsError::Other {
        message: format!("{} must be finite and positive, got {}", name, value),
    };

    if !args.exciton_density.is_finite() || args.exciton_density <= 0.0 {
        return Err(invalid_positive("--exciton-density", args.exciton_density));
    }
    if !args.slab_thickness_bohr.is_finite() || args.slab_thickness_bohr <= 0.0 {
        return Err(invalid_positive(
            "--slab-thickness-bohr",
            args.slab_thickness_bohr,
        ));
    }
    if !args.probe_q_in_dq.is_finite() || args.probe_q_in_dq <= 0.0 {
        return Err(invalid_positive("--probe-q-in-dq", args.probe_q_in_dq));
    }
    if let Some(probe_q) = args.probe_q {
        if !probe_q.is_finite() || probe_q < 0.0 {
            return Err(MatrixOpsError::Other {
                message: format!("--probe-q must be finite and non-negative, got {}", probe_q),
            });
        }
    }

    Ok(())
}
