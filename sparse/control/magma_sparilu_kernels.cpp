/*
    -- MAGMA (version 2.5.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2019

       @generated from sparse/control/magma_zparilu_kernels.cpp, normal z -> s, Sun Nov 24 14:37:46 2019
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define SWAP(a, b)  { val_swap = a; a = b; b = val_swap; }


/***************************************************************************//**
    Purpose
    -------
    This function does one asynchronous ParILU sweep. 
    Input and output array are identical.
    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                System matrix in COO.

    @param[in]
    L           magma_s_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_s_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC (U^T in CSR).
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
*******************************************************************************/


extern "C" magma_int_t
magma_sparilu_sweep(
    magma_s_matrix A,
    magma_s_matrix *L,
    magma_s_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    int i, j;


    float zero = MAGMA_S_MAKE(0.0, 0.0);
    int il, iu, jl, ju;

    #pragma omp parallel for
    for (int k=0; k < A.nnz; k++) {
        i = A.rowidx[k];
        j = A.col[k];

        float s, sp;
        s =  A.val[k];
        sp = zero;

        il = L->row[i];
        iu = U->row[j];

        while (il < L->row[i+1] && iu < U->row[j+1])
        {
            sp = zero;
            jl = L->col[il];
            ju = U->col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L->val[il] * U->val[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        s += sp;
        
        if ( i > j )      // modify l entry
            L->val[il-1] =  s / U->val[U->row[j+1]-1];
        else {            // modify u entry
            U->val[iu-1] = s;
        }
    }
    
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function does one synchronized ParILU sweep. Input and output are 
    different arrays.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                System matrix in COO.

    @param[in]
    L           magma_s_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_s_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC (U^T in CSR).
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
*******************************************************************************/


extern "C" magma_int_t
magma_sparilu_sweep_sync(
    magma_s_matrix A,
    magma_s_matrix *L,
    magma_s_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    int i, j;


    float zero = MAGMA_S_MAKE(0.0, 0.0);
    
    int il, iu, jl, ju;
    
    float *L_new_val = NULL, *U_new_val = NULL, *val_swap = NULL;
    
    CHECK( magma_smalloc_cpu( &L_new_val, L->nnz ));
    CHECK( magma_smalloc_cpu( &U_new_val, U->nnz ));
    
    // we need 1 on the main diagonal of L
    #pragma omp parallel for
    for (int k=0; k < L->num_rows; k++) {
        L_new_val[L->row[k+1]-1] = MAGMA_S_ONE;
    }
    
    #pragma omp parallel for
    for (int k=0; k < A.nnz; k++) {
        i = A.rowidx[k];
        j = A.col[k];
        
        float s, sp;
        s =  A.val[k];
        sp = zero;

        il = L->row[i];
        iu = U->row[j];

        while (il < L->row[i+1] && iu < U->row[j+1])
        {
            sp = zero;
            jl = L->col[il];
            ju = U->col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L->val[il] * U->val[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        }
        // undo the last operation (it must be the last)
        s += sp;
        
        if ( i > j )      // modify l entry
            L_new_val[il-1] =  s / U->val[U->row[j+1]-1];
        else {            // modify u entry
            U_new_val[iu-1] = s;
        }
    }
    
    // swap old and new values
    SWAP( L_new_val, L->val );
    SWAP( U_new_val, U->val );
    
cleanup:
    magma_free_cpu( L_new_val );
    magma_free_cpu( U_new_val );
    
    return info;
}
