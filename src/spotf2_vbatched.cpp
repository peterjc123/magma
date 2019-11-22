/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotf2_vbatched.cpp, normal z -> s, Fri Aug  2 17:10:09 2019
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_s
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_spotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    float **dA_array, magma_int_t* lda,
    float **dA_displ, 
    float **dW_displ,
    float **dB_displ, 
    float **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo=0;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else{
        arginfo = magma_spotrf_lpout_vbatched(uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue);
    }

    return arginfo;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
