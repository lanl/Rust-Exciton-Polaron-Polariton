use ndarray::Array1;
use sim_common::{data::ColoredBandData, kpath, mpi};

#[test]
fn square_lattice_path_has_requested_len() {
    let points = kpath::square_lattice_path(16.0, 17);
    assert_eq!(points.len(), 17);
    assert_eq!(points[0], vec![0.0, 0.0]);
}

#[test]
fn path_distances_are_monotonic() {
    let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];

    let x = kpath::path_distances(&points);

    assert_eq!(x.len(), 3);
    assert!(x[0] <= x[1]);
    assert!(x[1] <= x[2]);
    assert!((x[2] - 2.0).abs() < 1e-12);
}

#[test]
fn chunk_bounds_cover_expected_ranges() {
    assert_eq!(mpi::chunk_bounds(10, 0, 3), (0, 4));
    assert_eq!(mpi::chunk_bounds(10, 1, 3), (4, 8));
    assert_eq!(mpi::chunk_bounds(10, 2, 3), (8, 10));
}

#[test]
fn colored_band_data_round_trip_shape() {
    let data = ColoredBandData::from_results(
        vec![vec![0.0, 0.0], vec![1.0, 0.0]],
        vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![1.5, 2.5]),
        ],
        vec![vec![0.1, 0.2], vec![0.3, 0.4]],
    );

    let (k, e, c) = data.to_results();

    assert_eq!(k.len(), 2);
    assert_eq!(e.len(), 2);
    assert_eq!(c.len(), 2);
    assert_eq!(e[0].to_vec(), vec![1.0, 2.0]);
}
