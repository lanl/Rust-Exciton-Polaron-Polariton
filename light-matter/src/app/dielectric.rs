use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, Result};
use ndarray::{Array1, arr1};
use ndarray_linalg::c64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use super::config::build_dielectric_pp_config_for_reflectivity;
use crate::cli;
use crate::hamiltonian::{PPHamiltonian, SingExBosons};
use crate::visualization::generate_stacked_plots;
use crate::visualization::{DielectricPlotType, plot_dielectric_function};
use crate::{dielectric, timed_println, timed_println_rank0, visualization};

fn run_dielectric_scan(e_config: ExcitonConfig, args: cli::DielectricArgs) -> Result<()> {
    timed_println_rank0!("Starting dielectric function scan...");

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
                Some((0.182, 0.202))
            } else {
                Some((0.183, 0.188))
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

    for &reflectivity in reflectivities.iter() {
        let mode_set = crate::modes::photon::from_dielectric_args(&args, reflectivity);
        let total_coupling: f64 = mode_set.couplings.iter().sum();

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

    let real_filename = format!(
        "{}/dielectric_real_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
    );

    let imag_filename = format!(
        "{}/dielectric_imag_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
    );

    let both_filename = format!(
        "{}/dielectric_both_a0{}_V0{}_nrel{}_nmodes{}_pncoup{}_npn{}{}",
        save_dir, e_config.a0, e_config.v0, n_rel, n_modes, pn_coupling, pn_n_fock, suffix,
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
            n_eig,
            gamma,
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
            &omegas,
            gamma,
            save_dir,
            zoomed,
            omega_range,
        );

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

    let mut pp: PPHamiltonian<ComposedOperator<c64>, SingExBosons<ComposedOperator<c64>>> =
        PPHamiltonian::from_config(&pp_config);

    let result = dielectric::get_dielectric_function(
        &pp_config,
        &mut pp,
        n_eig,
        omegas.clone(),
        gamma,
        save_dir,
        use_cached,
    )?;

    let filename =
        dielectric::get_dielectric_function_filename(&pp, &pp_config, save_dir, n_eig, gamma);

    if sim_common::mpi::is_mpi_rank0_env() {
        timed_println!("Saving dielectric function data to {}", filename);

        let mut file = File::create(&filename).unwrap();

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
