#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Try to include the generated bindings
#[cfg(not(feature = "stub"))]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Provide minimal stubs in case bindings generation fails
#[cfg(feature = "stub")]
mod stub {
    // Minimal ARPACK interface
    #[no_mangle]
    pub unsafe extern "C" fn dsaupd_(
        _ido: *mut i32, _bmat: *const i8, _n: *const i32, _which: *const i8,
        _nev: *const i32, _tol: *const f64, _resid: *mut f64,
        _ncv: *const i32, _v: *mut f64, _ldv: *const i32,
        _iparam: *mut i32, _ipntr: *mut i32,
        _workd: *mut f64, _workl: *mut f64, _lworkl: *const i32,
        info: *mut i32,
    ) -> i32 {
        // Stub implementation that just returns an error
        if !info.is_null() {
            *info = -1; // Error code
        }
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn dseupd_(
        _rvec: *const i32, _howmny: *const i8, _select: *mut i32,
        _d: *mut f64, _z: *mut f64, _ldz: *const i32,
        _sigma: *const f64, _bmat: *const i8, _n: *const i32,
        _which: *const i8, _nev: *const i32, _tol: *const f64,
        _resid: *mut f64, _ncv: *const i32, _v: *mut f64,
        _ldv: *const i32, _iparam: *mut i32, _ipntr: *mut i32,
        _workd: *mut f64, _workl: *mut f64, _lworkl: *const i32,
        info: *mut i32,
    ) -> i32 {
        // Stub implementation that just returns an error
        if !info.is_null() {
            *info = -1; // Error code
        }
        0
    }
    
    // Add the _c variants with the signatures expected by the arpack crate
    pub unsafe extern "C" fn dsaupd_c(
        ido: *mut i32, bmat: *const i8, n: i32, which: *const i8,
        nev: i32, tol: f64, resid: *mut f64,
        ncv: i32, v: *mut f64, ldv: i32,
        iparam: *mut i32, ipntr: *mut i32,
        workd: *mut f64, workl: *mut f64, lworkl: i32,
        info: *mut i32,
    ) -> i32 {
        // Stub implementation that just returns an error
        if !info.is_null() {
            *info = -1; // Error code
        }
        0
    }

    pub unsafe extern "C" fn dseupd_c(
        rvec: i32, howmny: *const i8, select: *mut i32,
        d: *mut f64, z: *mut f64, ldz: i32,
        sigma: f64, bmat: *const i8, n: i32,
        which: *const i8, nev: i32, tol: f64,
        resid: *mut f64, ncv: i32, v: *mut f64,
        ldv: i32, iparam: *mut i32, ipntr: *mut i32,
        workd: *mut f64, workl: *mut f64, lworkl: i32,
        info: *mut i32,
    ) -> i32 {
        // Stub implementation that just returns an error
        if !info.is_null() {
            *info = -1; // Error code
        }
        0
    }
}

#[cfg(feature = "stub")]
pub use stub::*;
