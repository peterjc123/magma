/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotf2_batched.cpp, normal z -> d, Thu Oct  8 23:05:31 2020
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define REAL

/******************************************************************************/
// This is a recursive routine
extern "C" magma_int_t
magma_dpotf2_batched(
    magma_uplo_t uplo, magma_int_t n,
    double **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo=0;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    double c_neg_one = MAGMA_D_NEG_ONE;
    double c_one     = MAGMA_D_ONE;

    magma_int_t crossover = magma_get_dpotrf_batched_crossover();

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
    }
    else {
        if( n <= crossover ){
            arginfo = magma_dpotrf_lpout_batched(uplo, n, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);
        }
        else{
            magma_int_t n1 = n / 2;
            magma_int_t n2 = n - n1;
            // panel
            magma_dpotrf_lpout_batched(uplo, n1, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);

            // trsm
            magmablas_dtrsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n2, n1, MAGMA_D_ONE, 
                    dAarray(ai   , aj), ldda, 
                    dAarray(ai+n1, aj), ldda, batchCount, queue );

            // herk
            magmablas_dsyrk_batched_core( 
                    MagmaLower, MagmaNoTrans, 
                    n2, n1, 
                    c_neg_one, dAarray(ai+n1, aj   ), ldda,
                               dAarray(ai+n1, aj   ), ldda,
                    c_one,     dAarray(ai+n1, aj+n1), ldda, batchCount, queue );

            // panel
            arginfo = magma_dpotrf_lpout_batched(uplo, n2, dAarray(ai+n1, aj+n1), ldda, gbstep + n1, info_array, batchCount, queue);
        }
    }
    return arginfo;

#undef dAarray
}
