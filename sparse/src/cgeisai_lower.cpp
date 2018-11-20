/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Hartwig Anzt

       @generated from sparse/src/zgeisai_lower.cpp, normal z -> c, Mon Jun 25 18:24:31 2018
*/
#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#define PRECISION_c


/***************************************************************************//**
    Purpose
    -------

    Prepares Incomplete LU preconditioner using a sparse approximate inverse
    instead of sparse triangular solves.
    
    This routine only handles the lower triangular part. The return value is 0
    in case of success, and Magma_CUSOLVE if the pattern is too large to be 
    handled.

    Arguments
    ---------

    @param[in]
    L           magma_c_matrix
                lower triangular factor

    @param[in]
    S           magma_c_matrix
                pattern for the ISAI preconditioner for L
                
    @param[out]
    ISAIL       magma_c_matrix*
                ISAI preconditioner for L

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_ciluisaisetup_lower(
    magma_c_matrix L,
    magma_c_matrix S,
    magma_c_matrix *ISAIL,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_index_t *sizes_h = NULL;
    magma_int_t maxsize, nnzloc;
    magma_c_matrix MT={Magma_CSR};

    int warpsize=32;

    // we need this in any case as the ISAI matrix is generated in transpose fashion
    CHECK( magma_cmtranspose( S, &MT, queue ) );

    CHECK( magma_index_malloc_cpu( &sizes_h, L.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    magma_index_getvector( S.num_rows+1, S.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<L.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%% error for ISAI L: size of system %d is too large by %d\n", (int) i, (int) (maxsize-32));
            printf("%% fallback: use exact triangular solve (cuSOLVE)\n");
            info = Magma_CUSOLVE;
            goto cleanup;
        }
    }

    // printf("%% nnz in ISAI factor L (total max/row): %d %d\n", (int) S.nnz, (int) maxsize);
    // generation of the ISAI on the GPU - all operations in registers
     CHECK( magma_cisai_generator_regs( MagmaLower, MagmaNoTrans, MagmaNonUnit,
                    L, &MT, queue ) );

    CHECK( magma_cmtranspose( MT, ISAIL, queue ) );

cleanup:
    magma_free_cpu( sizes_h );
    magma_cmfree( &MT, queue );
    return info;
}

