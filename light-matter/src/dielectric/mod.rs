pub mod cache;
pub mod compute;

pub use cache::{
    get_dielectric_cache_filename, get_dielectric_function_filename,
    load_dielectric_cache, save_dielectric_cache,
};

pub use compute::{DielectricResult, get_dielectric_function};