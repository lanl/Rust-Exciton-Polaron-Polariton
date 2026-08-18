from pathlib import Path
import re

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

def extract_fn(s, name):
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

def extract_item(s, marker):
    start = s.find(marker)
    if start == -1:
        raise RuntimeError(f"item marker not found: {marker}")

    open_idx = s.find("{", start)
    close_idx = find_matching_brace(s, open_idx)
    return s[start:close_idx + 1]

def extract_macro(s, macro_name):
    m = re.search(rf"(?m)^#\[macro_export\]\s*\nmacro_rules!\s+{macro_name}\s*{{", s)
    if not m:
        raise RuntimeError(f"macro not found: {macro_name}")

    start = m.start()
    open_idx = s.find("{", m.start())
    close_idx = find_matching_brace(s, open_idx)
    return s[start:close_idx + 1]

def refactor_exciton_visualization():
    src = Path("exciton_2d/src/visualization.rs")
    dst_mod = Path("exciton_2d/src/visualization/mod.rs")

    if not src.exists():
        return

    s = src.read_text()

    rel = extract_fn(s, "plot_rel_eigenstate")
    spectrum = extract_fn(s, "plot_spectrum")
    bands = extract_fn(s, "plot_band_structure")
    colored = extract_fn(s, "plot_n_colored_band_structure")

    write("exciton_2d/src/visualization/eigenstate.rs", """use crate::grid::Grid;
use log::info;
use ndarray::{Array1, Array2};
use plotters::prelude::*;

""" + rel + "\n")

    write("exciton_2d/src/visualization/spectrum.rs", """use log::info;
use ndarray::Array1;
use plotters::prelude::*;

""" + spectrum + "\n")

    write("exciton_2d/src/visualization/bands.rs", """use log::info;
use ndarray::Array1;
use plotters::prelude::*;

""" + bands + "\n")

    write("exciton_2d/src/visualization/colored_bands.rs", """use log::info;
use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

""" + colored + "\n")

    write(dst_mod, """pub mod bands;
pub mod colored_bands;
pub mod eigenstate;
pub mod spectrum;

pub use bands::plot_band_structure;
pub use colored_bands::plot_n_colored_band_structure;
pub use eigenstate::plot_rel_eigenstate;
pub use spectrum::plot_spectrum;
""")

    src.unlink()

def refactor_light_matter_visualization():
    src = Path("light-matter/src/visualization.rs")
    if not src.exists():
        return

    s = src.read_text()

    params = extract_item(s, "pub struct DielectricParams")
    extract_params = extract_fn(s, "extract_params_from_filename")
    macro = extract_macro(s, "plot_photon_freqs")
    dielectric_plot = extract_fn(s, "plot_dielectric_function")
    load_data = extract_fn(s, "load_dielectric_data")
    plot_type = extract_item(s, "pub enum DielectricPlotType")
    band_plot = extract_fn(s, "plot_band_structure")
    colormap = extract_fn(s, "interpolate_colormap")

    write("light-matter/src/visualization/data.rs", """use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

""" + params + "\n\n" + extract_params + "\n\n" + load_data + "\n")

    write("light-matter/src/visualization/dielectric.rs", """use log::info;
use ndarray::Array1;
use plotters::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::data::{extract_params_from_filename, load_dielectric_data, DielectricParams};
use crate::{plot_photon_freqs, timed_println, Vector3D};

""" + plot_type + "\n\n" + dielectric_plot + "\n\n" + colormap + "\n")

    write("light-matter/src/visualization/bands.rs", """use log::info;
use ndarray::{s, Array1};
use plotters::prelude::*;

use crate::timed_println;

""" + band_plot + "\n")

    stacked_src = Path("light-matter/src/visualization_dielectric.rs")
    if stacked_src.exists():
        stacked = stacked_src.read_text()
        stacked = stacked.replace("use crate::visualization_dielectric;", "")
        stacked = stacked.replace("use crate::dielectric_function;", "use crate::dielectric;")
        stacked = stacked.replace("dielectric_function::", "dielectric::")
        write("light-matter/src/visualization/stacked.rs", stacked)
        stacked_src.unlink()
    else:
        # If already moved, leave current stacked.rs untouched.
        if not Path("light-matter/src/visualization/stacked.rs").exists():
            write("light-matter/src/visualization/stacked.rs", "")

    write("light-matter/src/visualization/mod.rs", """pub mod bands;
pub mod data;
pub mod dielectric;
pub mod stacked;

pub use bands::plot_band_structure;
pub use data::{extract_params_from_filename, load_dielectric_data, DielectricParams};
pub use dielectric::{plot_dielectric_function, DielectricPlotType};
pub use stacked::{
    compute_weight_fields,
    generate_stacked_plots,
    plot_dielectric_stacked_phonon,
    plot_dielectric_stacked_photonic,
};

""" + macro + "\n")

    src.unlink()

def patch_references():
    replacements = {
        "crate::visualization_dielectric::": "crate::visualization::",
        "use crate::visualization_dielectric::": "use crate::visualization::",
        "pub mod visualization_dielectric;": "",
    }

    for root in ["light-matter/src", "exciton_2d/src"]:
        for p in Path(root).rglob("*.rs"):
            s = p.read_text()
            old = s

            for a, b in replacements.items():
                s = s.replace(a, b)

            if s != old:
                p.write_text(s)

def main():
    refactor_exciton_visualization()
    refactor_light_matter_visualization()
    patch_references()

if __name__ == "__main__":
    main()
