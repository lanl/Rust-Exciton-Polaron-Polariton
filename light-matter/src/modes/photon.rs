use ndarray::Array1;

use crate::cli::{BandstructureArgs, DielectricArgs, PhotonModeArg};
use crate::modes::effective;
use crate::modes::types::Polarization;
use crate::types::Vector3D;

const C: f64 = 137.036;

#[derive(Debug, Clone)]
pub enum PhotonModeKind {
    Cartesian {
        n_modes_side: usize,
        q_max: f64,
        total_coupling: f64,
        coupling_scale: f64,
    },
    Effective {
        n_modes_target: usize,
        d_omega: f64,
        omega_max: f64,
        reflectivity: f64,
        coupling_scale: f64,
    },
}

#[derive(Debug, Clone)]
pub struct PhotonModeSet {
    pub q_values: Vec<Vector3D>,
    pub polarizations: Vec<Polarization>,
    pub couplings: Vec<f64>,
    pub d_q: f64,
    pub omega_0: f64,
    pub kind: PhotonModeKind,
}

pub fn construct_cartesian_modes(
    n_modes_side: usize,
    q_max: f64,
    omega_0: f64,
    total_coupling: f64,
    coupling_scale: f64,
) -> PhotonModeSet {
    let qx_values = Array1::linspace(-q_max, q_max, n_modes_side);
    let qy_values = Array1::linspace(-q_max, q_max, n_modes_side);
    let qz = omega_0 / C;

    let q_values: Vec<Vector3D> = qx_values
        .iter()
        .flat_map(|qx| {
            qy_values.iter().map(move |qy| Vector3D {
                x: *qx,
                y: *qy,
                z: qz,
            })
        })
        .collect();

    let per_mode = if q_values.is_empty() {
        0.0
    } else {
        coupling_scale * total_coupling / n_modes_side as f64
    };

    let polarizations = vec![Polarization::TE; q_values.len()];
    let d_q = if n_modes_side > 1 {
        2.0 * q_max / (n_modes_side as f64 - 1.0)
    } else {
        0.0
    };

    PhotonModeSet {
        q_values: q_values.clone(),
        polarizations,
        couplings: vec![per_mode; q_values.len()],
        d_q,
        omega_0,
        kind: PhotonModeKind::Cartesian {
            n_modes_side,
            q_max,
            total_coupling,
            coupling_scale,
        },
    }
}

pub fn construct_effective_modes(
    n_modes_target: usize,
    d_omega: f64,
    omega_0: f64,
    omega_max: f64,
    reflectivity: f64,
    coupling_scale: f64,
) -> PhotonModeSet {
    let (q_values, polarizations, couplings, d_q) = effective::generate_modes(
        d_omega,
        n_modes_target,
        omega_0,
        omega_max,
        reflectivity,
        coupling_scale,
    );

    PhotonModeSet {
        q_values,
        polarizations,
        couplings,
        d_q,
        omega_0,
        kind: PhotonModeKind::Effective {
            n_modes_target,
            d_omega,
            omega_max,
            reflectivity,
            coupling_scale,
        },
    }
}

pub fn from_bandstructure_args(args: &BandstructureArgs) -> PhotonModeSet {
    match args.photon_modes {
        PhotonModeArg::Cartesian => construct_cartesian_modes(
            args.n_modes,
            args.q_max,
            args.omega_0,
            args.total_coupling,
            args.photon_coupling_scale,
        ),
        PhotonModeArg::Effective => construct_effective_modes(
            args.n_modes,
            args.d_omega,
            args.omega_0,
            args.omega_max,
            args.reflectivity,
            args.photon_coupling_scale,
        ),
    }
}

pub fn from_dielectric_args(args: &DielectricArgs, reflectivity: f64) -> PhotonModeSet {
    match args.photon_modes {
        PhotonModeArg::Cartesian => construct_cartesian_modes(
            args.n_modes,
            args.q_max,
            args.omega_0,
            args.total_coupling,
            args.photon_coupling_scale,
        ),
        PhotonModeArg::Effective => construct_effective_modes(
            args.n_modes,
            args.d_omega,
            args.omega_0,
            args.omega_max,
            reflectivity,
            args.photon_coupling_scale,
        ),
    }
}

pub fn square_lattice_symmetry_points(a0: f64) -> (Vec<Vec<f64>>, Vec<String>) {
    sim_common::kpath::square_lattice_symmetry_points(a0)
}
