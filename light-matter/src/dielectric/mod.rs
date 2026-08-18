pub mod compute;

pub use compute::{
    DielectricResult,
    get_dielectric_cache_filename,
    get_dielectric_function_filename,
    get_dielectric_function,
    save_dielectric_cache,
    load_dielectric_cache,
};
