/*
    -- MAGMA (version 2.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2019

       @generated from magmablas/blas_zbatched.cpp, normal z -> c, Wed Jan  2 14:18:52 2019

       @author Ahmad Abdelfattah
       
       Implementation of batch BLAS on the host ( CPU ) using OpenMP
*/
#include "magma_internal.h"
#include "commonblas_c.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

/*******************************************************************************/
extern "C" void
blas_cgemm_batched( 
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaFloatComplex alpha,
        magmaFloatComplex const * const * hA_array, magma_int_t lda,
        magmaFloatComplex const * const * hB_array, magma_int_t ldb,
        magmaFloatComplex beta,
        magmaFloatComplex **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_cgemm( lapack_trans_const(transA),
                       lapack_trans_const(transB),
                       &m, &n, &k,
                       &alpha, hA_array[i], &lda,
                               hB_array[i], &ldb, 
                       &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_ctrsm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **hA_array, magma_int_t lda,
        magmaFloatComplex **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_ctrsm(
            lapack_side_const(side), lapack_uplo_const(uplo),
            lapack_trans_const(transA), lapack_diag_const(diag),
            &m, &n, &alpha,
            hA_array[s], &lda,
            hB_array[s], &ldb );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_ctrmm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **hA_array, magma_int_t lda,
        magmaFloatComplex **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_ctrmm(
            lapack_side_const(side), lapack_uplo_const(uplo),
            lapack_trans_const(transA), lapack_diag_const(diag),
            &m, &n, &alpha,
            hA_array[s], &lda,
            hB_array[s], &ldb );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_chemm_batched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **hA_array, magma_int_t lda,
        magmaFloatComplex **hB_array, magma_int_t ldb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_chemm( lapack_side_const(side),
                       lapack_uplo_const(uplo),
                       &m, &n,
                       &alpha, hA_array[i], &lda,
                               hB_array[i], &ldb,
                       &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_cherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    float alpha, magmaFloatComplex const * const * hA_array, magma_int_t lda,
    float beta,  magmaFloatComplex               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_cherk( lapack_uplo_const(uplo),
                       lapack_trans_const(trans),
                       &n, &k,
                       &alpha, hA_array[s], &lda,
                       &beta,  hC_array[s], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

/*******************************************************************************/
extern "C" void
blas_cher2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha, magmaFloatComplex const * const * hA_array, magma_int_t lda,
                              magmaFloatComplex const * const * hB_array, magma_int_t ldb, 
    float beta,              magmaFloatComplex               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_cher2k( lapack_uplo_const(uplo),
                        lapack_trans_const(trans),
                        &n, &k,
                        &alpha, hA_array[i], &lda,
                                hB_array[i], &ldb,
                        &beta,  hC_array[i], &ldc );
    }
    #if defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);            
    #endif
}

