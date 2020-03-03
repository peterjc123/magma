/*
    -- MAGMA (version 2.5.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2019

       @generated from sparse/control/magma_zvpass_gpu.cpp, normal z -> s, Sun Nov 24 14:37:45 2019
       @author Hartwig Anzt
*/

#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Passes a vector to MAGMA (located on DEV).

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    val         float*
                array containing vector entries

    @param[out]
    v           magma_s_matrix*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_svset_dev(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr val,
    magma_s_matrix *v,
    magma_queue_t queue )
{
    
    // make sure the target structure is empty
    magma_smfree( v, queue );
    
    v->num_rows = m;
    v->num_cols = n;
    v->nnz = m*n;
    v->memory_location = Magma_DEV;
    v->storage_type = Magma_DENSE;
    v->dval = val;
    v->major = MagmaColMajor;
    v->ownership = MagmaFalse;
    
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Passes a MAGMA vector back.

    Arguments
    ---------

    @param[in]
    v           magma_s_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         float*
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_svget(
    magma_s_matrix v,
    magma_int_t *m, magma_int_t *n,
    float **val,
    magma_queue_t queue )
{
    magma_s_matrix v_CPU={Magma_CSR};
    magma_int_t info =0;
    
    if ( v.memory_location == Magma_CPU ) {
        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.val;
    } else {
        CHECK( magma_smtransfer( v, &v_CPU, v.memory_location, Magma_CPU, queue ));
        CHECK( magma_svget( v_CPU, m, n, val, queue ));
    }
    
cleanup:
    return info;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back (located on DEV).

    Arguments
    ---------

    @param[in]
    v           magma_s_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaFloat_ptr
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_svget_dev(
    magma_s_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloat_ptr *val,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    magma_s_matrix v_DEV={Magma_CSR};
    
    if ( v.memory_location == Magma_DEV ) {
        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.dval;
    } else {
        CHECK( magma_smtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ));
        CHECK( magma_svget_dev( v_DEV, m, n, val, queue ));
    }
    
cleanup:
    magma_smfree( &v_DEV, queue );
    return info;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back (located on DEV).
    This function requires the array val to be 
    already allocated (of size m x n).

    Arguments
    ---------

    @param[in]
    v           magma_s_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         float*
                array of size m x n on the device the vector entries 
                are copied into
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_svcopy_dev(
    magma_s_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloat_ptr val,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    magma_s_matrix v_DEV={Magma_CSR};
    
    if ( v.memory_location == Magma_DEV ) {
        *m = v.num_rows;
        *n = v.num_cols;
        magma_scopyvector( v.num_rows * v.num_cols, v.dval, 1, val, 1, queue );
    } else {
        CHECK( magma_smtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ));
        CHECK( magma_svcopy_dev( v_DEV, m, n, val, queue ));
    }
    
cleanup:
    magma_smfree( &v_DEV, queue );
    return info;
}
