#include "wrapper.h"

/* Wrapper functions that call the actual ARPACK functions */
void dsaupd_c(a_int* ido, char* bmat, a_int n, char* which, a_int nev, 
              double tol, double* resid, a_int ncv, double* v, a_int ldv, 
              a_int* iparam, a_int* ipntr, double* workd, double* workl, 
              a_int lworkl, a_int* info) {
    /* Call the actual ARPACK function */
    dsaupd_(ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, 
            iparam, ipntr, workd, workl, &lworkl, info);
}

void dseupd_c(a_int rvec, char* howmny, a_int* select, double* d, double* z, 
              a_int ldz, double sigma, char* bmat, a_int n, char* which, 
              a_int nev, double tol, double* resid, a_int ncv, double* v, 
              a_int ldv, a_int* iparam, a_int* ipntr, double* workd, 
              double* workl, a_int lworkl, a_int* info) {
    /* Call the actual ARPACK function */
    dseupd_(&rvec, howmny, select, d, z, &ldz, &sigma, bmat, &n, which, 
            &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, 
            workl, &lworkl, info);
}
