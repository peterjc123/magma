/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from sparse/control/magma_zvpass_gpu.cpp, normal z -> c, Mon Jun 25 18:24:27 2018
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
    val         magmaFloatComplex*
                array containing vector entries

    @param[out]
    v           magma_c_matrix*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvset_dev(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr val,
    magma_c_matrix *v,
    magma_queue_t queue )
{
    
    // make sure the target structure is empty
    magma_cmfree( v, queue );
    
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
    v           magma_c_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaFloatComplex*
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvget(
    magma_c_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex **val,
    magma_queue_t queue )
{
    magma_c_matrix v_CPU={Magma_CSR};
    magma_int_t info =0;
    
    if ( v.memory_location == Magma_CPU ) {
        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.val;
    } else {
        CHECK( magma_cmtransfer( v, &v_CPU, v.memory_location, Magma_CPU, queue ));
        CHECK( magma_cvget( v_CPU, m, n, val, queue ));
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
    v           magma_c_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaFloatComplex_ptr
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvget_dev(
    magma_c_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex_ptr *val,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    magma_c_matrix v_DEV={Magma_CSR};
    
    if ( v.memory_location == Magma_DEV ) {
        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.dval;
    } else {
        CHECK( magma_cmtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ));
        CHECK( magma_cvget_dev( v_DEV, m, n, val, queue ));
    }
    
cleanup:
    magma_cmfree( &v_DEV, queue );
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
    v           magma_c_matrix
                magma vector

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    val         magmaFloatComplex*
                array of size m x n on the device the vector entries 
                are copied into
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cvcopy_dev(
    magma_c_matrix v,
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex_ptr val,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    magma_c_matrix v_DEV={Magma_CSR};
    
    if ( v.memory_location == Magma_DEV ) {
        *m = v.num_rows;
        *n = v.num_cols;
        magma_ccopyvector( v.num_rows * v.num_cols, v.dval, 1, val, 1, queue );
    } else {
        CHECK( magma_cmtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ));
        CHECK( magma_cvcopy_dev( v_DEV, m, n, val, queue ));
    }
    
cleanup:
    magma_cmfree( &v_DEV, queue );
    return info;
}
