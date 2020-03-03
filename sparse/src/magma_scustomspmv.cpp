/*
    -- MAGMA (version 2.5.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2019

       @generated from sparse/src/magma_zcustomspmv.cpp, normal z -> s, Sun Nov 24 14:37:48 2019
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    This is an interface to any custom sparse matrix vector product.
    It should compute y = alpha*FUNCTION(x) + beta*y
    The vectors are located on the device, the scalars on the CPU.


    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows
                
    @param[in]
    n           magma_int_t
                number of columns
                
    @param[in]
    alpha       float
                scalar alpha
                
    @param[in]
    x           float *
                input vector x
                
    @param[in]
    beta        float
                scalar beta
    @param[out]
    y           float *
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sblas
    ********************************************************************/

extern "C" magma_int_t
magma_scustomspmv(
    magma_int_t m,
    magma_int_t n,
    float alpha,
    float beta,
    float *x,
    float *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y.dval
    // sizes are x.num_rows, x.num_cols
    
    magma_sge3pt( m, n, alpha, beta, x, y, queue );
    
    return info;
}
