/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/src/magma_zcustomspmv.cpp, normal z -> c, Thu Oct  8 23:05:55 2020
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
    alpha       magmaFloatComplex
                scalar alpha
                
    @param[in]
    x           magmaFloatComplex *
                input vector x
                
    @param[in]
    beta        magmaFloatComplex
                scalar beta
    @param[out]
    y           magmaFloatComplex *
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_ccustomspmv(
    magma_int_t m,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex beta,
    magmaFloatComplex *x,
    magmaFloatComplex *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y.dval
    // sizes are x.num_rows, x.num_cols
    
    magma_cge3pt( m, n, alpha, beta, x, y, queue );
    
    return info;
}
