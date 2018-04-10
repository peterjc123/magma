/*
    -- MAGMA (version 2.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2017

       @author Hartwig Anzt

       @generated from sparse/src/zgeisai_lower.cpp, normal z -> d, Wed Nov 15 00:34:25 2017
*/
#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#define PRECISION_d


/***************************************************************************//**
    Purpose
    -------

    Prepares Incomplete LU preconditioner using a sparse approximate inverse
    instead of sparse triangular solves.
    
    This routine only handles the lower triangular part.


    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in]
    S           magma_d_matrix
                pattern for the ISAI preconditioner

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_diluisaisetup_lower(
    magma_d_matrix A,
    magma_d_matrix S,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // real_Double_t start, end;

    magma_index_t *sizes_h = NULL;
    magma_int_t maxsize, nnzloc;
    magma_d_matrix MT={Magma_CSR};

    int warpsize=32;

   // we need this in any case as the ISAI matrix is generated in transpose fashion
    CHECK( magma_dmtranspose( S, &MT, queue ) );

    CHECK( magma_index_malloc_cpu( &sizes_h, A.num_rows+1 ) );
    // only needed in case the systems are generated in GPU main memory
    // CHECK( magma_index_malloc( &sizes_d, A.num_rows ) );
    // CHECK( magma_index_malloc( &locations_d, A.num_rows*warpsize ) );
    // CHECK( magma_dmalloc( &trisystems_d, min(320000,A.num_rows) *warpsize*warpsize ) ); // fixed size - go recursive
    // CHECK( magma_dmalloc( &rhs_d, A.num_rows*warpsize ) );
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    magma_index_getvector( S.num_rows+1, S.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI L: size of system %d is too large by %d\n", (int) i, (int) (maxsize-32));
            printf("%% fallback: use exact triangular solve (cuSOLVE)\n");
            precond->trisolver = Magma_CUSOLVE;
            goto cleanup;
        }
    }

    printf("%% nnz in L-ISAI (total max/row): %d %d\n", (int) S.nnz, (int) maxsize);
    // via registers
     CHECK( magma_disai_generator_regs( MagmaLower, MagmaNoTrans, MagmaNonUnit,
                    precond->L, &MT, queue ) );

    CHECK( magma_dmtranspose( MT, &precond->LD, queue ) );

cleanup:
    magma_free_cpu( sizes_h );
    magma_dmfree( &MT, queue );
    return info;
}

