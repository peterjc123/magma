/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from sparse/blas/zlobpcg_maxpy.cu, normal z -> s, Fri Aug  2 17:10:12 2019

*/
#include "magmasparse_internal.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512



__global__ void
magma_slobpcg_maxpy_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    float * X, 
    float * Y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // global row index

    if ( row < num_rows ) {
        for( int i=0; i < num_vecs; i++ ) {
            Y[ row + i*num_rows ] += X[ row + i*num_rows ];
        }
    }
}


/**
    Purpose
    -------
    
    This routine computes a axpy for a mxn matrix:
        
        Y = X + Y
        
    It replaces:
            magma_saxpy(m*n, c_one, Y, 1, X, 1);


        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    X = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows

    @param[in]
    num_vecs    magma_int_t
                number of vectors

    @param[in]
    X           magmaFloat_ptr 
                input vector X

    @param[in,out]
    Y           magmaFloat_ptr 
                input/output vector Y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_slobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaFloat_ptr X,
    magmaFloat_ptr Y,
    magma_queue_t queue )
{
    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;
     magma_int_t threads = BLOCK_SIZE;
    dim3 block( block_size );
    dim3 grid( magma_ceildiv( num_rows, block_size ) );

    magma_slobpcg_maxpy_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                                ( num_rows, num_vecs, X, Y );


    return MAGMA_SUCCESS;
}
