/*
    -- MAGMA (version 2.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2017

       @generated from sparse/blas/zgeaxpy.cu, normal z -> c, Wed Nov 15 00:34:24 2017

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
cgeaxpy_kernel( 
    int num_rows, 
    int num_cols, 
    magmaFloatComplex alpha, 
    magmaFloatComplex * dx, 
    magmaFloatComplex beta, 
    magmaFloatComplex * dy)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            int idx = row + j*num_rows;
            dy[ idx ] = alpha * dx[ idx ] + beta * dy[ idx ];
        }
    }
}

/**
    Purpose
    -------
    
    This routine computes Y = alpha *  X + beta * Y on the GPU.
    The input format is magma_c_matrix. It can handle both,
    dense matrix (vector block) and CSR matrices. For the latter,
    it interfaces the cuSPARSE library.
    
    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex
                scalar multiplier.
                
    @param[in]
    X           magma_c_matrix
                input/output matrix Y.
                
    @param[in]
    beta        magmaFloatComplex
                scalar multiplier.
                
    @param[in,out]
    Y           magma_c_matrix*
                input matrix X.
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_cgeaxpy(
    magmaFloatComplex alpha,
    magma_c_matrix X,
    magmaFloatComplex beta,
    magma_c_matrix *Y,
    magma_queue_t queue )
{
    int m = X.num_rows;
    int n = X.num_cols;
    magma_c_matrix C={Magma_CSR};
    
    if( X.storage_type == Magma_DENSE && Y->storage_type == Magma_DENSE ){
        
        dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
        magma_int_t threads = BLOCK_SIZE;
        cgeaxpy_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( m, n, alpha, X.dval, beta, Y->dval );
                        
    } else if( X.storage_type == Magma_CSR && Y->storage_type == Magma_CSR ) {
        
        magma_ccuspaxpy( &alpha, X, &beta, *Y, &C, queue );
        magma_cmfree( Y, queue );
        magma_cmtransfer( C, Y, Magma_DEV, Magma_DEV, queue );
        magma_cmfree( &C, queue );
    } else {
        printf("%% error: matrix addition only supported for DENSE and CSR format.\n");   
    }
                    
    return MAGMA_SUCCESS;
}
