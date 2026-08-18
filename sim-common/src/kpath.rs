use crate::types::Vec2;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct KPath {
    pub points: Vec<Vec<f64>>,
    pub labels: Vec<String>,
    pub label_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SquareSymmetry {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    MirrorX,
    MirrorY,
    MirrorDiag,
    MirrorAntiDiag,
}

impl SquareSymmetry {
    pub const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rot90,
        Self::Rot180,
        Self::Rot270,
        Self::MirrorX,
        Self::MirrorY,
        Self::MirrorDiag,
        Self::MirrorAntiDiag,
    ];

    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Identity,
            1 => Self::Rot90,
            2 => Self::Rot180,
            3 => Self::Rot270,
            4 => Self::MirrorX,
            5 => Self::MirrorY,
            6 => Self::MirrorDiag,
            7 => Self::MirrorAntiDiag,
            _ => panic!("Invalid square symmetry operation: {idx}"),
        }
    }
}

pub fn apply_square_symmetry(k: &[f64], op: SquareSymmetry) -> Vec<f64> {
    let kx = k[0];
    let ky = k[1];

    match op {
        SquareSymmetry::Identity => vec![kx, ky],
        SquareSymmetry::Rot90 => vec![-ky, kx],
        SquareSymmetry::Rot180 => vec![-kx, -ky],
        SquareSymmetry::Rot270 => vec![ky, -kx],
        SquareSymmetry::MirrorX => vec![kx, -ky],
        SquareSymmetry::MirrorY => vec![-kx, ky],
        SquareSymmetry::MirrorDiag => vec![ky, kx],
        SquareSymmetry::MirrorAntiDiag => vec![-ky, -kx],
    }
}

pub fn interpolate_segment(start: Vec2, end: Vec2, n: usize) -> Vec<Vec<f64>> {
    match n {
        0 => Vec::new(),
        1 => vec![vec![start[0], start[1]]],
        _ => (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                vec![
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                ]
            })
            .collect(),
    }
}

pub fn square_lattice_symmetry_points(a0: f64) -> (Vec<Vec<f64>>, Vec<String>) {
    (
        vec![
            vec![0.0, 0.0],
            vec![PI / a0, 0.0],
            vec![PI / a0, PI / a0],
        ],
        vec!["Γ".to_string(), "X".to_string(), "M".to_string()],
    )
}

pub fn square_lattice_path(a0: f64, n_points: usize) -> Vec<Vec<f64>> {
    if n_points == 0 {
        return Vec::new();
    }

    if n_points == 1 {
        return vec![vec![0.0, 0.0]];
    }

    let b = 2.0 * PI / a0;

    let gamma = [0.0, 0.0];
    let x = [b / 2.0, 0.0];
    let m = [b / 2.0, b / 2.0];

    let points_per_segment = (n_points / 3).max(2);

    let mut points = Vec::new();
    points.extend(interpolate_segment(gamma, x, points_per_segment));
    points.extend(interpolate_segment(x, m, points_per_segment));
    points.extend(interpolate_segment(m, gamma, points_per_segment));
    points.truncate(n_points);
    points
}

pub fn generic_closed_path(
    n_k_points: usize,
    sym_points: &[Vec<f64>],
    a0: f64,
    zoomed: bool,
) -> Vec<Vec<f64>> {
    if n_k_points == 0 {
        return Vec::new();
    }

    if n_k_points == 1 {
        return vec![vec![0.0, 0.0]];
    }

    if zoomed {
        let start = [0.0, 0.0];
        let end = [PI / a0 / 6.0, 0.0];

        return interpolate_segment(start, end, n_k_points);
    }

    assert!(!sym_points.is_empty(), "sym_points must not be empty");

    let points_per_seg = (n_k_points / sym_points.len()).max(2);

    let mut points = Vec::new();

    for i in 0..sym_points.len() {
        let j = (i + 1) % sym_points.len();

        points.extend(interpolate_segment(
            [sym_points[i][0], sym_points[i][1]],
            [sym_points[j][0], sym_points[j][1]],
            points_per_seg,
        ));
    }

    points.truncate(n_k_points);
    points
}

pub fn path_distances(k_points: &[Vec<f64>]) -> Vec<f64> {
    let mut x = vec![0.0; k_points.len()];

    for i in 1..k_points.len() {
        let dx = k_points[i][0] - k_points[i - 1][0];
        let dy = k_points[i][1] - k_points[i - 1][1];

        x[i] = x[i - 1] + (dx * dx + dy * dy).sqrt();
    }

    x
}
