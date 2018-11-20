/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from sparse/src/magma_zcustomspmv.cpp, normal z -> d, Mon Jun 25 18:24:31 2018
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
    alpha       double
                scalar alpha
                
    @param[in]
    x           double *
                input vector x
                
    @param[in]
    beta        double
                scalar beta
    @param[out]
    y           double *
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dcustomspmv(
    magma_int_t m,
    magma_int_t n,
    double alpha,
    double beta,
    double *x,
    double *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y.dval
    // sizes are x.num_rows, x.num_cols
    
    magma_dge3pt( m, n, alpha, beta, x, y, queue );
    
    return info;
}
