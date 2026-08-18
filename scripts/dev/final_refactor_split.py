from pathlib import Path
import re

ROOT = Path(".")

def read(path):
    return Path(path).read_text()

def write(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

def find_matching_brace(s, open_idx):
    depth = 0
    in_str = False
    in_char = False
    escape = False
    line_comment = False
    block_comment = False

    i = open_idx
    while i < len(s):
        c = s[i]
        n = s[i + 1] if i + 1 < len(s) else ""

        if line_comment:
            if c == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if c == "*" and n == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue

        if in_char:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == "'":
                in_char = False
            i += 1
            continue

        if c == "/" and n == "/":
            line_comment = True
            i += 2
            continue

        if c == "/" and n == "*":
            block_comment = True
            i += 2
            continue

        if c == '"':
            in_str = True
            i += 1
            continue

        if c == "'":
            in_char = True
            i += 1
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    raise RuntimeError("matching brace not found")

def extract_item(s, marker):
    start = s.find(marker)
    if start == -1:
        raise RuntimeError(f"marker not found: {marker}")

    open_idx = s.find("{", start)
    if open_idx == -1:
        raise RuntimeError(f"open brace not found for {marker}")

    close_idx = find_matching_brace(s, open_idx)
    return s[start:close_idx + 1]

def extract_fn(s, name):
    # Match either `fn name` or `pub fn name` etc.
    m = re.search(
        rf"(?m)^(?:pub(?:\([^)]+\))?\s+)?fn\s+{name}(?:\s*<[^{{}};=]*>)?\s*\(",
        s,
    )
    if not m:
        raise RuntimeError(f"function not found: {name}")

    start = m.start()
    open_idx = s.find("{", start)
    close_idx = find_matching_brace(s, open_idx)
    return s[start:close_idx + 1]

def remove_item(s, item):
    return s.replace(item + "\n\n", "").replace(item + "\n", "").replace(item, "")

def make_pub_super(fn_src):
    if fn_src.startswith("fn "):
        return "pub(super) " + fn_src
    return re.sub(r"^pub\s+", "pub(super) ", fn_src)

def replace_fn_visibility(fn_src, vis="pub(super)"):
    return re.sub(r"^(?:pub(?:\([^)]+\))?\s+)?fn\s+", f"{vis} fn ", fn_src)

def split_app():
    p = Path("light-matter/src/app/mod.rs")
    if not p.exists():
        return

    s = p.read_text()

    # If already split, skip.
    if "mod bandstructure;" in s and "Command::Bandstructure(args) => bandstructure::run(args)?" in s:
        return

    enum_method = extract_item(s, "pub enum Method")

    names = [
        "run_bandstructure_with_args",
        "run_bandstructure_for_backend",
        "run_bandstructure_parallel_mode",
        "run_dielectric_scan",
        "build_bandstructure_pp_config",
        "selected_dielectric_reflectivity_index",
        "build_dielectric_pp_config_for_reflectivity",
        "matrix_elem_band_k_indices",
        "run_matrix_elem_calc_bandstructure_cli",
        "run_matrix_elem_calc_dielectric_cli",
        "run_matrix_elem_calc_cli",
        "method_from_arg",
        "run_bandstructure_cli",
        "run_dielectric_cli",
        "run",
    ]

    fns = {name: extract_fn(s, name) for name in names}

    config_rs = """use exciton_2d::ExcitonConfig;
use ndarray::arr1;

use crate::cli;
use crate::cli::MethodArg;
use crate::config::PPConfig;

""" + enum_method.replace("pub enum Method", "pub(super) enum Method") + "\n\n" + \
        replace_fn_visibility(fns["method_from_arg"]) + "\n\n" + \
        replace_fn_visibility(fns["build_bandstructure_pp_config"]) + "\n\n" + \
        replace_fn_visibility(fns["selected_dielectric_reflectivity_index"]) + "\n\n" + \
        replace_fn_visibility(fns["build_dielectric_pp_config_for_reflectivity"]) + "\n"

    band_rs = """use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, MatrixOps, Result};
use ndarray::{Array1, Array2};
use ndarray_linalg::c64;
use sprs::CsMat;
use std::time::Instant;

use super::config::{build_bandstructure_pp_config, method_from_arg, Method};
use crate::cli;
use crate::cli::BandParallelModeArg;
use crate::config::PPConfig;
use crate::hamiltonian::{self, PPHamiltonian, SingExBosons};
use crate::{bands, timed_println, timed_println_rank0};

""" + \
        fns["run_bandstructure_with_args"] + "\n\n" + \
        fns["run_bandstructure_for_backend"] + "\n\n" + \
        fns["run_bandstructure_parallel_mode"] + "\n\n" + \
        replace_fn_visibility(fns["run_bandstructure_cli"]) + "\n"

    dielectric_rs = """use exciton_2d::ExcitonConfig;
use matrix_ops::{ComposedOperator, Result};
use ndarray::{Array1, arr1};
use ndarray_linalg::c64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use super::config::build_dielectric_pp_config_for_reflectivity;
use crate::cli;
use crate::hamiltonian::{PPHamiltonian, SingExBosons};
use crate::visualization::{plot_dielectric_function, DielectricPlotType};
use crate::visualization_dielectric::generate_stacked_plots;
use crate::{dielectric, timed_println, timed_println_rank0, visualization};

""" + \
        fns["run_dielectric_scan"] + "\n\n" + \
        replace_fn_visibility(fns["run_dielectric_cli"]) + "\n"

    matrix_rs = """use matrix_ops::Result;
use ndarray_linalg::c64;
use sprs::CsMat;

use super::config::{
    build_bandstructure_pp_config,
    build_dielectric_pp_config_for_reflectivity,
    selected_dielectric_reflectivity_index,
};
use crate::cli;
use crate::hamiltonian::{PPHamiltonian, SingExBosons};
use crate::{bands, timed_println};

""" + \
        fns["matrix_elem_band_k_indices"] + "\n\n" + \
        fns["run_matrix_elem_calc_bandstructure_cli"] + "\n\n" + \
        fns["run_matrix_elem_calc_dielectric_cli"] + "\n\n" + \
        replace_fn_visibility(fns["run_matrix_elem_calc_cli"]) + "\n"

    mod_rs = """mod bandstructure;
mod config;
mod dielectric;
mod matrix_elem;

use matrix_ops::Result;

use crate::cli::{Cli, Command};

pub fn run(cli: Cli) -> Result<()> {
    let num_threads = cli
        .rayon_threads
        .or_else(|| {
            std::env::var("OMP_NUM_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    match cli.command {
        Command::Bandstructure(args) => bandstructure::run_bandstructure_cli(args)?,
        Command::Dielectric(args) => dielectric::run_dielectric_cli(args)?,
        Command::MatrixElemCalc(args) => matrix_elem::run_matrix_elem_calc_cli(args)?,
    }

    Ok(())
}
"""

    write("light-matter/src/app/config.rs", config_rs)
    write("light-matter/src/app/bandstructure.rs", band_rs)
    write("light-matter/src/app/dielectric.rs", dielectric_rs)
    write("light-matter/src/app/matrix_elem.rs", matrix_rs)
    write("light-matter/src/app/mod.rs", mod_rs)

def split_exciton_hamiltonian():
    p = Path("exciton_2d/src/hamiltonian/mod.rs")
    if not p.exists():
        return

    s = p.read_text()

    if "pub mod full;" in s and "pub use full::FullHamiltonian;" in s:
        return

    rel_start = s.find("/// Hamiltonian for the relative coordinate motion of an exciton")
    full_start = s.find("/// Full Hamiltonian for the exciton including center-of-mass motion")
    prep_start = s.find("/// Prepared exciton Hamiltonian with all K-independent terms precomputed.")
    full_impl_start = s.find("impl<M: MatrixOps> FullHamiltonian<M>")
    prep_impl_start = s.find("impl<M: MatrixOps> PreparedExcitonHamiltonian<M>")

    if min(rel_start, full_start, prep_start, full_impl_start, prep_impl_start) < 0:
        raise RuntimeError("failed to find hamiltonian split markers")

    rel_code = s[rel_start:full_start].strip()
    full_struct = s[full_start:prep_start].strip()
    prep_struct = s[prep_start:full_impl_start].strip()
    full_impl = s[full_impl_start:prep_impl_start].strip()
    prep_impl = s[prep_impl_start:].strip()

    relative_rs = """use crate::grid::Grid;

use log::info;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::time::Instant;

""" + rel_code + "\n"

    full_rs = """use super::PreparedExcitonHamiltonian;
use crate::lattice::KappaKey;
use crate::matrix_elements::MatrixElementCalculator;
use crate::relative::RelativeHamiltonian;
use crate::utils::exciton_flat_index;

use log::{info, warn};
use matrix_ops::{ConvertFrom, DiagonalizationOptions, Diagonalize, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use sprs::TriMat;
use std::collections::HashMap;
use std::time::Instant;

""" + full_struct + "\n\n" + full_impl + "\n"

    prepared_rs = """use matrix_ops::{ConvertFrom, MatrixOps, MatrixOpsError};
use ndarray::{Array1, Array2};

""" + prep_struct + "\n\n" + prep_impl + "\n"

    ham_mod = """pub mod full;
pub mod prepared;

pub use crate::relative::RelativeHamiltonian;
pub use full::FullHamiltonian;
pub use prepared::PreparedExcitonHamiltonian;
"""

    write("exciton_2d/src/relative/hamiltonian.rs", relative_rs)
    write("exciton_2d/src/hamiltonian/full.rs", full_rs)
    write("exciton_2d/src/hamiltonian/prepared.rs", prepared_rs)
    write("exciton_2d/src/hamiltonian/mod.rs", ham_mod)

    rel_mod = Path("exciton_2d/src/relative/mod.rs").read_text() if Path("exciton_2d/src/relative/mod.rs").exists() else ""
    if "pub mod hamiltonian;" not in rel_mod:
        rel_mod = "pub mod hamiltonian;\npub mod laguerre;\n\npub use hamiltonian::RelativeHamiltonian;\n"
    else:
        if "pub use hamiltonian::RelativeHamiltonian;" not in rel_mod:
            rel_mod += "\npub use hamiltonian::RelativeHamiltonian;\n"
    write("exciton_2d/src/relative/mod.rs", rel_mod)

def split_pp_fock_helpers():
    src_file = Path("light-matter/src/hamiltonian/polaron_polariton.rs")
    dir_mod = Path("light-matter/src/hamiltonian/polaron_polariton/mod.rs")

    if src_file.exists():
        dir_mod.parent.mkdir(parents=True, exist_ok=True)
        src_file.rename(dir_mod)

    p = dir_mod
    if not p.exists():
        return

    s = p.read_text()

    if "mod fock;" not in s:
        s = "mod fock;\n\n" + s

    s = s.replace("Self::phonon_occupations_from_index", "fock::occupations_from_index")
    s = s.replace("Self::phonon_index_from_occupations", "fock::index_from_occupations")

    for name in ["phonon_occupations_from_index", "phonon_index_from_occupations"]:
        while True:
            m = re.search(rf"(?m)^    (?:#\[inline\]\n    )?fn\s+{name}\s*\(", s)
            if not m:
                break
            start = m.start()
            # include preceding doc comments if immediately above
            doc_start = s.rfind("\n    ///", 0, start)
            if doc_start != -1:
                # only include if there is no non-doc function boundary between doc_start and start
                chunk = s[doc_start:start]
                if all(line.startswith("    ///") or line.strip() == "" for line in chunk.splitlines()[1:]):
                    start = doc_start + 1
            open_idx = s.find("{", m.end())
            close_idx = find_matching_brace(s, open_idx)
            s = s[:start] + s[close_idx + 1:]

    write(p, s)

    write("light-matter/src/hamiltonian/polaron_polariton/fock.rs", """/// Converts a flat product-Fock index into occupations.
///
/// This matches the tensor-product order used by `FullBosons::a`, where
/// mode 0 is the slowest-varying index and the last mode is fastest-varying:
///
///     index = (((n0 * d1 + n1) * d2 + n2) ...)
#[inline]
pub fn occupations_from_index(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut occ = vec![0; dims.len()];

    for mode_idx in (0..dims.len()).rev() {
        let dim = dims[mode_idx];
        occ[mode_idx] = idx % dim;
        idx /= dim;
    }

    occ
}

/// Converts product-Fock occupations into a flat index.
///
/// This is the inverse of `occupations_from_index`.
#[inline]
pub fn index_from_occupations(occ: &[usize], dims: &[usize]) -> usize {
    occ.iter()
        .zip(dims.iter())
        .fold(0usize, |acc, (&n, &dim)| acc * dim + n)
}
""")

def split_band_plot():
    p = Path("light-matter/src/bands/calculate.rs")
    if not p.exists():
        return

    s = p.read_text()

    if Path("light-matter/src/bands/plot.rs").exists():
        return

    plot_fn = extract_fn(s, "plot_color_band_structure")
    s2 = remove_item(s, plot_fn)

    # Clean now-unused imports from calculate.rs.
    s2 = s2.replace("use ndarray::{Array1, Array2, s};", "use ndarray::{Array1, Array2};")
    for line in [
        "use std::fs::File;\n",
        "use std::io::Write;\n",
        "use std::process::Command;\n",
    ]:
        s2 = s2.replace(line, "")

    plot_rs = """use ndarray::{Array1, s};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::timed_println;

""" + plot_fn + "\n"

    write("light-matter/src/bands/calculate.rs", s2)
    write("light-matter/src/bands/plot.rs", plot_rs)

    mod_p = Path("light-matter/src/bands/mod.rs")
    mod_s = mod_p.read_text()
    if "pub mod plot;" not in mod_s:
        mod_s = mod_s.replace("pub mod calculate;\n", "pub mod calculate;\npub mod plot;\n")
    mod_s = mod_s.replace(", plot_color_band_structure", "")
    if "pub use plot::plot_color_band_structure;" not in mod_s:
        mod_s += "\npub use plot::plot_color_band_structure;\n"
    write(mod_p, mod_s)

def split_dielectric_cache():
    p = Path("light-matter/src/dielectric/compute.rs")
    if not p.exists():
        return

    if Path("light-matter/src/dielectric/cache.rs").exists():
        return

    s = p.read_text()

    fn_names = [
        "get_dielectric_cache_filename",
        "get_dielectric_function_filename",
        "save_dielectric_cache",
        "load_dielectric_cache",
    ]
    extracted = [extract_fn(s, name) for name in fn_names]
    s2 = s
    for item in extracted:
        s2 = remove_item(s2, item)

    if "use super::cache::" not in s2:
        s2 = s2.replace(
            "use crate::{bands, timed_println};",
            "use crate::{bands, timed_println};\nuse super::cache::{get_dielectric_cache_filename, load_dielectric_cache, save_dielectric_cache};"
        )

    cache_rs = """use crate::config::PPConfig;
use crate::hamiltonian::{Boson, PPHamiltonian};
use matrix_ops::MatrixOps;
use ndarray::Array1;

""" + "\n\n".join(extracted) + "\n"

    write("light-matter/src/dielectric/compute.rs", s2)
    write("light-matter/src/dielectric/cache.rs", cache_rs)

    mod_p = Path("light-matter/src/dielectric/mod.rs")
    mod_s = mod_p.read_text()
    if "pub mod cache;" not in mod_s:
        mod_s = mod_s.replace("pub mod compute;\n", "pub mod cache;\npub mod compute;\n")
    mod_s = """pub mod cache;
pub mod compute;

pub use cache::{
    get_dielectric_cache_filename,
    get_dielectric_function_filename,
    load_dielectric_cache,
    save_dielectric_cache,
};

pub use compute::{
    DielectricResult,
    get_dielectric_function,
};
"""
    write(mod_p, mod_s)

def cleanup_utils_and_empty_files():
    # Ensure light-matter utils is only hash helpers.
    write("light-matter/src/utils.rs", """/// Quantization scale for K-point cache hashes.
///
/// This should be much smaller than the spacing between physically distinct
/// K points in your scans. It prevents tiny floating-point differences from
/// changing cache filenames.
pub const K_HASH_QUANTUM: f64 = 1e-10;

pub fn quantize_f64_for_hash(x: f64) -> i64 {
    if !x.is_finite() {
        panic!("Cannot hash non-finite floating-point value: {}", x);
    }

    let q = (x / K_HASH_QUANTUM).round();

    if q.abs() > i64::MAX as f64 {
        panic!(
            "Floating-point value {} is too large to quantize with quantum {}",
            x, K_HASH_QUANTUM
        );
    }

    if q == 0.0 { 0 } else { q as i64 }
}

pub fn hash_quantized_f64<H: std::hash::Hasher>(x: f64, hasher: &mut H) {
    use std::hash::Hash;

    quantize_f64_for_hash(x).hash(hasher);
}
""")

    # Remove empty placeholder files if present.
    for path in [
        "exciton_2d/src/relative/analytic.rs",
        "exciton_2d/src/relative/hamiltonian.rs.bak",
    ]:
        p = Path(path)
        if p.exists() and not p.read_text().strip():
            p.unlink()

def main():
    split_app()
    split_exciton_hamiltonian()
    split_pp_fock_helpers()
    split_band_plot()
    split_dielectric_cache()
    cleanup_utils_and_empty_files()

if __name__ == "__main__":
    main()
