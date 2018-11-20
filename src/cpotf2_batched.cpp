/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotf2_batched.cpp, normal z -> c, Mon Jun 25 18:24:10 2018
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define COMPLEX

/******************************************************************************/
// This is a recursive routine
extern "C" magma_int_t
magma_cpotf2_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo=0;

    // Quick return if possible
    if (n == 0) {
        return 1;
    }

    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex c_one     = MAGMA_C_ONE;

    magma_int_t crossover = magma_get_cpotrf_batched_crossover();

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
    }
    else {
        if( n <= crossover ){
            arginfo = magma_cpotrf_lpout_batched(uplo, n, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);
        }
        else{
            magma_int_t n1 = n / 2;
            magma_int_t n2 = n - n1;
            // panel
            magma_cpotrf_lpout_batched(uplo, n1, dAarray(ai, aj), ldda, gbstep, info_array, batchCount, queue);

            // trsm
            magmablas_ctrsm_recursive_batched( 
                    MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                    n2, n1, MAGMA_C_ONE, 
                    dAarray(ai   , aj), ldda, 
                    dAarray(ai+n1, aj), ldda, batchCount, queue );

            // herk
            magmablas_cherk_batched_core( 
                    MagmaLower, MagmaNoTrans, 
                    n2, n1, 
                    c_neg_one, dAarray(ai+n1, aj   ), ldda,
                               dAarray(ai+n1, aj   ), ldda,
                    c_one,     dAarray(ai+n1, aj+n1), ldda, batchCount, queue );

            // panel
            arginfo = magma_cpotrf_lpout_batched(uplo, n2, dAarray(ai+n1, aj+n1), ldda, gbstep + n1, info_array, batchCount, queue);
        }
    }
    return arginfo;

#undef dAarray
}
