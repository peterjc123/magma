/*
    -- MAGMA (version 2.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2017

       @generated from sparse/control/magma_zparilut_tools_sync.cpp, normal z -> c, Wed Nov 15 00:34:25 2017
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }

/***************************************************************************//**
    Purpose
    -------
    This function does one synchronized ParILU sweep. Input and output are 
    different arrays.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix*
                System matrix.

    @param[in]
    L           magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_c_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC.
                
    @param[out]
    L_new       magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR.

    @param[out]
    U_new       magma_c_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/


extern "C" magma_int_t
magma_cparilut_sweep_sync(
    magma_c_matrix *A,
    magma_c_matrix *L,
    magma_c_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    
    // temporary vectors to swap the col/rowidx later
    // magma_index_t *tmpi;
    
    magma_c_matrix L_new={Magma_CSR}, U_new={Magma_CSR};
    // magma_index_t *index_swap;
    magmaFloatComplex *L_new_val = NULL, *U_new_val = NULL, *val_swap = NULL;
    
    // CHECK( magma_cmtransfer( *L, &L_new, Magma_CPU, Magma_CPU, queue ) );
    // CHECK( magma_cmtransfer( *U, &U_new, Magma_CPU, Magma_CPU, queue ) );
    CHECK( magma_cmalloc_cpu( &L_new_val, L->nnz ));
    CHECK( magma_cmalloc_cpu( &U_new_val, U->nnz ));
    
    #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        magma_int_t i,j,icol,jcol,iold;

        magma_index_t row = U->col[ e ];
        magma_index_t col = U->rowidx[ e ];
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( row == col ){ 
            U_new_val[ e ] = MAGMA_C_ONE; // upper triangular has diagonal equal 1
        } else {
            //printf("(%d,%d) ", row, col); fflush(stdout);
            magmaFloatComplex A_e = MAGMA_C_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    break;
                }
            }

            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaFloatComplex sum = MAGMA_C_ZERO;
            magmaFloatComplex lsum = MAGMA_C_ZERO;
            do{
                lsum = MAGMA_C_ZERO;
                iold = i;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j++;
                }
            }while( i<endi && j<endj );
            sum = sum - lsum;

            // write back to location e
            U_new_val[ e ] =  ( A_e - sum ) / L->val[iold];
        }
    }// end omp parallel section
    
    
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){

        magma_int_t i,j,icol,jcol;

        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
        magmaFloatComplex A_e = MAGMA_C_ZERO;
        // check whether A contains element in this location
        for( i = A->row[row]; i<A->row[row+1]; i++){
            if( A->col[i] == col ){
                A_e = A->val[i];
                break;
            }
        }
        //now do the actual iteration
        i = L->row[ row ];
        j = U->row[ col ];
        magma_int_t endi = L->row[ row+1 ];
        magma_int_t endj = U->row[ col+1 ]; 
        magmaFloatComplex sum = MAGMA_C_ZERO;
        magmaFloatComplex lsum = MAGMA_C_ZERO;
        do{
            lsum = MAGMA_C_ZERO;
            icol = L->col[i];
            jcol = U->col[j];
            
            if( icol == jcol ){
                lsum = L->val[i] * U_new_val[j];
                sum = sum + lsum;
                i++;
                j++;
            }
            else if( icol<jcol ){
                i++;
            }
            else {
                j++;
            }
        }while( i<endi && j<endj );
        sum = sum - lsum;
        // write back to location e
        L_new_val[ e ] =  ( A_e - sum );

    }// end omp parallel section


    



    
    val_swap = L_new_val;
    L_new_val = L->val;
    L->val = val_swap;
    
    val_swap = U_new_val;
    U_new_val = U->val;
    U->val = val_swap;
    
    magma_cmfree( &L_new, queue );
    magma_cmfree( &U_new, queue );
    magma_free_cpu( L_new_val );
    magma_free_cpu( U_new_val );
    
cleanup:
    return info;
}





/***************************************************************************//**
    Purpose
    -------
    This function scales the residuals of a lower triangular factor L with the 
    diagonal of U. The intention is to generate a good initial guess for 
    inserting the elements.

    Arguments
    ---------

    @param[in]
    L           magma_c_matrix
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_c_matrix
                Current approximation for the upper triangular factor
                The format is sorted CSC.

    @param[in]
    hL          magma_c_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    hU          magma_c_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/


extern "C" magma_int_t
magma_cparilut_align_residuals(
    magma_c_matrix L,
    magma_c_matrix U,
    magma_c_matrix *Lnew,
    magma_c_matrix *Unew,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<L.num_rows; row++){
        magmaFloatComplex Lscal = L.val[L.row[row+1]-1]; // last element in row
        for( magma_int_t el=Unew->row[row]; el<Unew->row[row+1]; el++){
            Unew->val[el] = Unew->val[el] / Lscal;           
        }
    }
    
// cleanup:
    return info;
}
