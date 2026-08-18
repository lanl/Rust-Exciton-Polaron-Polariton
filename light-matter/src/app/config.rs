use exciton_2d::ExcitonConfig;
use ndarray::arr1;

use crate::cli;
use crate::cli::MethodArg;
use crate::config::PPConfig;

pub(super) enum Method {
    Sparse,
    Dense,
    LinOp,
}

pub(super) fn method_from_arg(method: MethodArg) -> Method {
    match method {
        MethodArg::Sparse => Method::Sparse,
        MethodArg::Dense => Method::Dense,
        MethodArg::Linop => Method::LinOp,
    }
}

pub(super) fn build_bandstructure_pp_config(
    e_config: ExcitonConfig,
    args: &cli::BandstructureArgs,
) -> (PPConfig, Vec<Vec<f64>>, Vec<String>) {
    let a0 = e_config.a0;

    let mode_set = crate::modes::photon::from_bandstructure_args(args);

    let pn_k = arr1(&[0.0, args.pn_k_y]);

    let pp_config = match mode_set.kind {
        crate::modes::photon::PhotonModeKind::Effective {
            d_omega,
            omega_max,
            reflectivity,
            ..
        } => PPConfig::new_with_effective_modes(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            args.n_k_points,
            args.n_bands,
            d_omega,
            omega_max,
            reflectivity,
        ),
        crate::modes::photon::PhotonModeKind::Cartesian { .. } => PPConfig::new(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            args.n_k_points,
            args.n_bands,
        ),
    };

    let (sym_points, sym_labels) = crate::modes::photon::square_lattice_symmetry_points(a0);

    (pp_config, sym_points, sym_labels)
}

pub(super) fn selected_dielectric_reflectivity_index(args: &cli::DielectricArgs) -> usize {
    args.reflectivity_index
        .or_else(|| sim_common::mpi::env_usize(&["REFLECTIVITY_INDEX", "SLURM_ARRAY_TASK_ID"]))
        .unwrap_or(0)
}

pub(super) fn build_dielectric_pp_config_for_reflectivity(
    e_config: ExcitonConfig,
    args: &cli::DielectricArgs,
    refl_idx: usize,
) -> PPConfig {
    if refl_idx >= args.reflectivities.len() {
        panic!(
            "Reflectivity index {} out of range 0..{}",
            refl_idx,
            args.reflectivities.len()
        );
    }

    let reflectivity = args.reflectivities[refl_idx];
    let mode_set = crate::modes::photon::from_dielectric_args(args, reflectivity);

    let pn_k = arr1(&[args.pn_k_x, 0.0]);

    match mode_set.kind {
        crate::modes::photon::PhotonModeKind::Effective {
            d_omega,
            omega_max,
            reflectivity,
            ..
        } => PPConfig::new_with_effective_modes(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            1,
            1,
            d_omega,
            omega_max,
            reflectivity,
        ),
        crate::modes::photon::PhotonModeKind::Cartesian { .. } => PPConfig::new(
            e_config,
            args.n_rel,
            args.n_ex_eig,
            mode_set.couplings,
            args.omega_0,
            mode_set.q_values,
            args.pn_coupling,
            args.pn_omega,
            pn_k,
            args.pn_n_fock,
            1,
            1,
        ),
    }
}
