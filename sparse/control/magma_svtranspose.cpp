/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from sparse/control/magma_zvtranspose.cpp, normal z -> s, Thu Oct  8 23:05:51 2020
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Transposes a vector from col to row major and vice versa.


    Arguments
    ---------

    @param[in]
    x           magma_s_matrix
                input vector

    @param[out]
    y           magma_s_matrix*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_svtranspose(
    magma_s_matrix x,
    magma_s_matrix *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t    m = x.num_rows;
    magma_int_t    n = x.num_cols;
    
    // make sure the target structure is empty
    magma_smfree( y, queue );
    y->ownership = MagmaTrue;
    
    magma_s_matrix dx={Magma_CSR}, dy={Magma_CSR};
            
    if ( x.memory_location == Magma_DEV ) {
        CHECK( magma_svinit( y, Magma_DEV, x.num_rows,x.num_cols, MAGMA_S_ZERO, queue ));
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->storage_type = x.storage_type;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_stranspose( m, n, x.val, m, y->val, n, queue );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_stranspose( n, m, x.val, n, y->val, m, queue );
        }
    } else {
        CHECK( magma_smtransfer( x, &dx, Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_svtranspose( dx, &dy, queue ));
        CHECK( magma_smtransfer( dy, y, Magma_DEV, Magma_CPU, queue ));
    }
    
cleanup:
    if( info != 0 ){
        magma_smfree( y, queue );
    }
    magma_smfree( &dx, queue );
    magma_smfree( &dy, queue );
    return info;
}
