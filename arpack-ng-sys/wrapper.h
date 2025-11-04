/* Include ARPACK definitions */
#include <arpackdef.h>

/* Declare ARPACK Fortran interface functions */
#ifdef __cplusplus
extern "C" {
#endif

/* Double precision symmetric routines (Fortran interface) */
void dsaupd_(a_int* ido, char* bmat, a_int* n, char* which, a_int* nev, 
             double* tol, double* resid, a_int* ncv, double* v, a_int* ldv, 
             a_int* iparam, a_int* ipntr, double* workd, double* workl, 
             a_int* lworkl, a_int* info);

void dseupd_(a_int* rvec, char* howmny, a_int* select, double* d, double* z, 
             a_int* ldz, double* sigma, char* bmat, a_int* n, char* which, 
             a_int* nev, double* tol, double* resid, a_int* ncv, double* v, 
             a_int* ldv, a_int* iparam, a_int* ipntr, double* workd, 
             double* workl, a_int* lworkl, a_int* info);

/* C interface wrappers - match the types expected by the arpack crate */
void dsaupd_c(a_int* ido, char* bmat, a_int n, char* which, a_int nev, 
              double tol, double* resid, a_int ncv, double* v, a_int ldv, 
              a_int* iparam, a_int* ipntr, double* workd, double* workl, 
              a_int lworkl, a_int* info);

void dseupd_c(a_int rvec, char* howmny, a_int* select, double* d, double* z, 
              a_int ldz, double sigma, char* bmat, a_int n, char* which, 
              a_int nev, double tol, double* resid, a_int ncv, double* v, 
              a_int ldv, a_int* iparam, a_int* ipntr, double* workd, 
              double* workl, a_int lworkl, a_int* info);

#ifdef __cplusplus
}
#endif
