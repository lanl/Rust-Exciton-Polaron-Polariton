extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.c");

    // Path to the ARPACK installation from Spack
    let arpack_path = "/vast/home/mike_tay/spack/opt/spack/linux-rhel8-haswell/gcc-9.4.0/arpack-ng-3.9.0-v3yoapcihbvwth4yo35fbbcmpuljes7b";
    
    // Add the library path (using lib64 instead of lib)
    println!("cargo:rustc-link-search=native={}/lib64", arpack_path);
    
    // Link against ARPACK and OpenBLAS
    println!("cargo:rustc-link-lib=arpack");
    println!("cargo:rustc-link-lib=openblas");
    
    // Compile our wrapper code
    cc::Build::new()
        .file("wrapper.c")
        .include(format!("{}/include/arpack-ng", arpack_path))
        .compile("arpack_wrapper");
    
    // Generate bindings to ARPACK
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include/arpack-ng", arpack_path))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
