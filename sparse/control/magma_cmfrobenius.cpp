/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @generated from sparse/control/magma_zmfrobenius.cpp, normal z -> c, Sun Mar 29 20:48:35 2020
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define PRECISION_c

/**
    Purpose
    -------

    Computes the Frobenius norm || A - B ||_S on the sparsity pattern of S.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input sparse matrix in CSR

    @param[in]
    B           magma_c_matrix
                input sparse matrix in CSR

    @param[in]
    S           magma_c_matrix
                input sparsity pattern in CSR

    @param[out]
    norm        float*
                Frobenius norm of difference on sparsity pattern S
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmfrobenius(
    magma_c_matrix A,
    magma_c_matrix B,
    magma_c_matrix S,
    float *norm,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    float tmp;
    magma_int_t i,j,k;
        
    magma_c_matrix hA={Magma_CSR}, hB={Magma_CSR}, hS={Magma_CSR};

    CHECK( magma_cmtransfer( A, &hA, A.memory_location, Magma_CPU, queue  ));
    CHECK( magma_cmtransfer( B, &hB, B.memory_location, Magma_CPU, queue  ));
    CHECK( magma_cmtransfer( S, &hS, S.memory_location, Magma_CPU, queue  ));
    
    if( hA.num_rows == hB.num_rows && hA.num_rows == hS.num_rows ) {
        for(i=0; i<hS.num_rows; i++){
            for(j=hS.row[i]; j<hS.row[i+1]; j++){
                magma_index_t lcol = hS.col[j];
                magmaFloatComplex Aval = MAGMA_C_MAKE(0.0, 0.0);
                magmaFloatComplex Bval = MAGMA_C_MAKE(0.0, 0.0);
                for(k=hA.row[i]; k<hA.row[i+1]; k++){
                    if( hA.col[k] == lcol ){
                        Aval = hA.val[k];
                    }
                }
                for(k=hB.row[i]; k<hB.row[i+1]; k++){
                    if( hB.col[k] == lcol ){
                        Bval = hB.val[k];
                    }
                }
                tmp = MAGMA_C_ABS(Aval - Bval) ;
                (*norm) = (*norm) + tmp * tmp;
            }
        }
        
        (*norm) =  sqrt((*norm));
    }
    
    
cleanup:
    magma_cmfree( &hA, queue );
    magma_cmfree( &hB, queue );
    magma_cmfree( &hS, queue );
    
    return info;
}
