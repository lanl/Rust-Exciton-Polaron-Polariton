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

fn closed_polyline_path(vertices: &[Vec2], n_points: usize) -> Vec<Vec<f64>> {
    if n_points == 0 {
        return Vec::new();
    }

    if vertices.is_empty() {
        return vec![vec![0.0, 0.0]; n_points];
    }

    if n_points == 1 {
        return vec![vec![vertices[0][0], vertices[0][1]]];
    }

    let n_segments = vertices.len();

    (0..n_points)
        .map(|idx| {
            let u = idx as f64 * n_segments as f64 / (n_points - 1) as f64;
            let seg = (u.floor() as usize).min(n_segments - 1);
            let t = u - seg as f64;

            let a = vertices[seg];
            let b = vertices[(seg + 1) % n_segments];

            vec![a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
        })
        .collect()
}

pub fn square_lattice_symmetry_points(a0: f64) -> (Vec<Vec<f64>>, Vec<String>) {
    (
        vec![vec![0.0, 0.0], vec![PI / a0, 0.0], vec![PI / a0, PI / a0]],
        vec!["Γ".to_string(), "X".to_string(), "M".to_string()],
    )
}

pub fn square_lattice_path(a0: f64, n_points: usize) -> Vec<Vec<f64>> {
    let b = 2.0 * PI / a0;

    let gamma = [0.0, 0.0];
    let x = [b / 2.0, 0.0];
    let m = [b / 2.0, b / 2.0];

    closed_polyline_path(&[gamma, x, m], n_points)
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

    let vertices: Vec<Vec2> = sym_points.iter().map(|p| [p[0], p[1]]).collect();

    closed_polyline_path(&vertices, n_k_points)
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
