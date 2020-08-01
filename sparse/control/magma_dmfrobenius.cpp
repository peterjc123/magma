/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @generated from sparse/control/magma_zmfrobenius.cpp, normal z -> d, Sun Mar 29 20:48:35 2020
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define PRECISION_d

/**
    Purpose
    -------

    Computes the Frobenius norm || A - B ||_S on the sparsity pattern of S.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input sparse matrix in CSR

    @param[in]
    B           magma_d_matrix
                input sparse matrix in CSR

    @param[in]
    S           magma_d_matrix
                input sparsity pattern in CSR

    @param[out]
    norm        double*
                Frobenius norm of difference on sparsity pattern S
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmfrobenius(
    magma_d_matrix A,
    magma_d_matrix B,
    magma_d_matrix S,
    double *norm,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    double tmp;
    magma_int_t i,j,k;
        
    magma_d_matrix hA={Magma_CSR}, hB={Magma_CSR}, hS={Magma_CSR};

    CHECK( magma_dmtransfer( A, &hA, A.memory_location, Magma_CPU, queue  ));
    CHECK( magma_dmtransfer( B, &hB, B.memory_location, Magma_CPU, queue  ));
    CHECK( magma_dmtransfer( S, &hS, S.memory_location, Magma_CPU, queue  ));
    
    if( hA.num_rows == hB.num_rows && hA.num_rows == hS.num_rows ) {
        for(i=0; i<hS.num_rows; i++){
            for(j=hS.row[i]; j<hS.row[i+1]; j++){
                magma_index_t lcol = hS.col[j];
                double Aval = MAGMA_D_MAKE(0.0, 0.0);
                double Bval = MAGMA_D_MAKE(0.0, 0.0);
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
                tmp = MAGMA_D_ABS(Aval - Bval) ;
                (*norm) = (*norm) + tmp * tmp;
            }
        }
        
        (*norm) =  sqrt((*norm));
    }
    
    
cleanup:
    magma_dmfree( &hA, queue );
    magma_dmfree( &hB, queue );
    magma_dmfree( &hS, queue );
    
    return info;
}
