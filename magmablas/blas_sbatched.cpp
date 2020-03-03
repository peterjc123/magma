/*
    -- MAGMA (version 2.5.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2019

       @generated from magmablas/blas_zbatched.cpp, normal z -> s, Sun Nov 24 14:37:34 2019

       @author Ahmad Abdelfattah
       
       Implementation of batch BLAS on the host ( CPU ) using OpenMP
*/
#include "magma_internal.h"
#include "commonblas_s.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

/*******************************************************************************/
extern "C" void
blas_sgemm_batched( 
        magma_trans_t transA, magma_trans_t transB, 
        magma_int_t m, magma_int_t n, magma_int_t k,
        float alpha,
        float const * const * hA_array, magma_int_t lda,
        float const * const * hB_array, magma_int_t ldb,
        float beta,
        float **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_sgemm( lapack_trans_const(transA),
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
blas_strsm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        float alpha, 
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_strsm(
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
blas_strmm_batched( 
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        float alpha, 
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_strmm(
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
blas_ssymm_batched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        float alpha, 
        float **hA_array, magma_int_t lda,
        float **hB_array, magma_int_t ldb, 
        float beta, 
        float **hC_array, magma_int_t ldc, 
        magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_ssymm( lapack_side_const(side),
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
blas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    float alpha, float const * const * hA_array, magma_int_t lda,
    float beta,  float               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s=0; s < batchCount; s++) {
        blasf77_ssyrk( lapack_uplo_const(uplo),
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
blas_ssyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    float alpha, float const * const * hA_array, magma_int_t lda,
                              float const * const * hB_array, magma_int_t ldb, 
    float beta,              float               **hC_array, magma_int_t ldc, 
    magma_int_t batchCount )
{
    #if defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i=0; i < batchCount; i++) {
        blasf77_ssyr2k( lapack_uplo_const(uplo),
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

