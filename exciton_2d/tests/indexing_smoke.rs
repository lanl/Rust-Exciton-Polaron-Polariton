use exciton_2d::lattice::KappaKey;
use exciton_2d::{exciton_flat_index, exciton_unflatten_index};

#[test]
fn exciton_flat_unflat_round_trip() {
    let n_rel = 7;

    for com in 0..5 {
        for rel in 0..n_rel {
            let idx = exciton_flat_index(com, rel, n_rel);
            assert_eq!(exciton_unflatten_index(idx, n_rel), (com, rel));
        }
    }
}

#[test]
fn kappa_key_adds_integer_indices() {
    let a = KappaKey::new(1, -2);
    let b = KappaKey::new(-3, 5);

    assert_eq!(a.add(b), KappaKey::new(-2, 3));
}
