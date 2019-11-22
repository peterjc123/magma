/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from sparse/blas/zgeaxpy.cu, normal z -> d, Fri Aug  2 17:10:12 2019

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
dgeaxpy_kernel( 
    int num_rows, 
    int num_cols, 
    double alpha, 
    double * dx, 
    double beta, 
    double * dy)
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
    The input format is magma_d_matrix. It can handle both,
    dense matrix (vector block) and CSR matrices. For the latter,
    it interfaces the cuSPARSE library.
    
    Arguments
    ---------

    @param[in]
    alpha       double
                scalar multiplier.
                
    @param[in]
    X           magma_d_matrix
                input/output matrix Y.
                
    @param[in]
    beta        double
                scalar multiplier.
                
    @param[in,out]
    Y           magma_d_matrix*
                input matrix X.
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_dgeaxpy(
    double alpha,
    magma_d_matrix X,
    double beta,
    magma_d_matrix *Y,
    magma_queue_t queue )
{
    int m = X.num_rows;
    int n = X.num_cols;
    magma_d_matrix C={Magma_CSR};
    
    if( X.storage_type == Magma_DENSE && Y->storage_type == Magma_DENSE ){
        
        dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
        magma_int_t threads = BLOCK_SIZE;
        dgeaxpy_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                        ( m, n, alpha, X.dval, beta, Y->dval );
                        
    } else if( X.storage_type == Magma_CSR && Y->storage_type == Magma_CSR ) {
        
        magma_dcuspaxpy( &alpha, X, &beta, *Y, &C, queue );
        magma_dmfree( Y, queue );
        magma_dmtransfer( C, Y, Magma_DEV, Magma_DEV, queue );
        magma_dmfree( &C, queue );
    } else {
        printf("%% error: matrix addition only supported for DENSE and CSR format.\n");   
    }
                    
    return MAGMA_SUCCESS;
}
