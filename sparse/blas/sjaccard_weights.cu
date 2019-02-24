/*
    -- MAGMA (version 2.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2019

       @author Hartwig Anzt

       @generated from sparse/blas/zjaccard_weights.cu, normal z -> s, Wed Jan  2 14:18:54 2019
*/
#include "magmasparse_internal.h"

#define PRECISION_s


__global__ void 
magma_sjaccardweights_kernel(   
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magma_int_t nnzJ,  
    magma_index_t *rowidxJ, 
    magma_index_t *colidxJ,
    float *valJ, 
    magma_index_t *rowptrA, 
    magma_index_t *colidxA, 
    float *valA ) {
    int i, j;
    int k = blockDim.x * gridDim.x * blockIdx.y
            + blockDim.x * blockIdx.x + threadIdx.x;


    float zero = MAGMA_S_MAKE(0.0, 0.0);
    float one = MAGMA_S_MAKE(1.0, 0.0);
    float sum_i, sum_j, intersect;
    int il, iu, jl, ju;
    

    if (k < nnzJ)
    {
        i = rowidxJ[k];
        j = colidxJ[k];
        if( i != j ){
            il = rowptrA[i];
            iu = rowptrA[j];
            
            sum_i = zero;
            sum_j = zero;
            intersect = zero;
            
            sum_i = MAGMA_S_MAKE((float)rowptrA[i+1] - rowptrA[i], 0.0);
            sum_j = MAGMA_S_MAKE((float)rowptrA[j+1] - rowptrA[j], 0.0);
    
            while (il < rowptrA[i+1] && iu < rowptrA[j+1])
            {
            
                jl = colidxJ[il];
                ju = rowidxJ[iu];
            
                // avoid branching
                // if there are actual values:
                // intersect = ( jl == ju ) ? valJ[il] * valJ[iu] : sp;
                // else
                intersect = ( jl == ju ) ? intersect + one : intersect;
                il = ( jl <= ju ) ? il+1 : il;
                iu = ( ju <= jl ) ? iu+1 : iu;
            }
            

            
            valJ[k] = MAGMA_S_MAKE(MAGMA_S_REAL(intersect) / MAGMA_S_REAL( sum_i + sum_j - intersect), 0.0 );
        } else {
            valJ[k] = MAGMA_S_ONE;
        }
            
    }
}// end kernel 

/**
    Purpose
    -------

    Computes Jaccard weights for a matrix

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix

    @param[out]
    J           magma_s_matrix*
                Jaccard weights
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sjaccard_weights(
    magma_s_matrix A,
    magma_s_matrix *J,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m = J->num_rows;
    magma_int_t n = J->num_rows;
    magma_int_t nnz = J->nnz;
    
    int blocksize1 = 32;
    int blocksize2 = 1;

    int dimgrid1 = sqrt( magma_ceildiv( nnz, blocksize1 ) );
    int dimgrid2 = magma_ceildiv(nnz, blocksize1*dimgrid1);
    int dimgrid3 = 1;
    // printf("thread block: ( %d x %d  ) x [%d x %d]\n", blocksize1, blocksize2, dimgrid1, dimgrid2);

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

    magma_sjaccardweights_kernel<<< grid, block, 0, queue->cuda_stream() >>>( 
        m, n, nnz, 
        J->drowidx,
        J->dcol,
        J->dval,
        A.drow,
        A.dcol,
        A.dval );

    return info;
}
