use ndarray::Array1;

use crate::cli::{BandstructureArgs, DielectricArgs, PhotonModeArg};
use crate::modes::effective;
use crate::types::Vector3D;

const C: f64 = 137.036;

#[derive(Debug, Clone)]
pub enum PhotonModeKind {
    Cartesian {
        n_modes_side: usize,
        q_max: f64,
        total_coupling: f64,
    },
    Effective {
        n_modes_target: usize,
        d_omega: f64,
        omega_max: f64,
        reflectivity: f64,
    },
}

#[derive(Debug, Clone)]
pub struct PhotonModeSet {
    pub q_values: Vec<Vector3D>,
    pub couplings: Vec<f64>,
    pub omega_0: f64,
    pub kind: PhotonModeKind,
}

pub fn construct_cartesian_modes(
    n_modes_side: usize,
    q_max: f64,
    omega_0: f64,
    total_coupling: f64,
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
        total_coupling / n_modes_side as f64
    };

    PhotonModeSet {
        q_values: q_values.clone(),
        couplings: vec![per_mode; q_values.len()],
        omega_0,
        kind: PhotonModeKind::Cartesian {
            n_modes_side,
            q_max,
            total_coupling,
        },
    }
}

pub fn construct_effective_modes(
    n_modes_target: usize,
    d_omega: f64,
    omega_0: f64,
    omega_max: f64,
    reflectivity: f64,
) -> PhotonModeSet {
    let (q_values, couplings) =
        effective::generate_modes(d_omega, n_modes_target, omega_0, omega_max, reflectivity);

    PhotonModeSet {
        q_values,
        couplings,
        omega_0,
        kind: PhotonModeKind::Effective {
            n_modes_target,
            d_omega,
            omega_max,
            reflectivity,
        },
    }
}

pub fn from_bandstructure_args(args: &BandstructureArgs) -> PhotonModeSet {
    match args.photon_modes {
        PhotonModeArg::Cartesian => {
            construct_cartesian_modes(args.n_modes, args.q_max, args.omega_0, args.total_coupling)
        }
        PhotonModeArg::Effective => construct_effective_modes(
            args.n_modes,
            args.d_omega,
            args.omega_0,
            args.omega_max,
            args.reflectivity,
        ),
    }
}

pub fn from_dielectric_args(args: &DielectricArgs, reflectivity: f64) -> PhotonModeSet {
    match args.photon_modes {
        PhotonModeArg::Cartesian => {
            construct_cartesian_modes(args.n_modes, args.q_max, args.omega_0, args.total_coupling)
        }
        PhotonModeArg::Effective => construct_effective_modes(
            args.n_modes,
            args.d_omega,
            args.omega_0,
            args.omega_max,
            reflectivity,
        ),
    }
}

pub fn square_lattice_symmetry_points(a0: f64) -> (Vec<Vec<f64>>, Vec<String>) {
    sim_common::kpath::square_lattice_symmetry_points(a0)
}
