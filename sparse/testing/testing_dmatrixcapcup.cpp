/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from sparse/testing/testing_zmatrixcapcup.cpp, normal z -> d, Fri Aug  2 17:10:14 2019
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_dopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_d_matrix Z={Magma_CSR};
    magma_d_matrix Z1={Magma_CSR};
    magma_d_matrix Z2={Magma_CSR};
    magma_d_matrix Z3={Magma_CSR};
    magma_d_matrix Z4={Magma_CSR};
    magma_d_matrix Z5={Magma_CSR};
    
    int i=1;
    TESTING_CHECK( magma_dparse_opts( argc, argv, &zopts, &i, queue ));
    printf("matrixinfo = [\n");
    printf("%%   size (n)   ||   nonzeros (nnz)   ||   nnz/n\n");
    printf("%%=============================================================%%\n");
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_dm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_d_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("   %10lld          %10lld          %10lld\n",
               (long long) Z.num_rows, (long long) Z.nnz, (long long) (Z.nnz/Z.num_rows) );
        
        // multiply by itself
        TESTING_CHECK( magma_dmconvert( Z, &Z1,  Magma_CSR, Magma_CSRL, queue ));
        TESTING_CHECK( magma_dmtranspose( Z1, &Z2, queue ) );
        magma_dprint_matrix( Z, queue );
        magma_dprint_matrix( Z1, queue );
        printf("B = tril(A)^T:\n");
        magma_dprint_matrix( Z2, queue );
        
        // now the negcap:
        printf("C = B cup B^T :\n");
        TESTING_CHECK( magma_dmatrix_cup( Z1, Z2, &Z3, queue ));
        magma_dprint_matrix( Z3, queue );
        
        // now the negcap:
        printf("C = B cap B^T :\n");
        TESTING_CHECK( magma_dmatrix_cap( Z1, Z2, &Z4, queue ));
        magma_dprint_matrix( Z4, queue );
        
        
        // now the negcap:
        printf("C = B negcap B^T :\n");
        TESTING_CHECK( magma_dmatrix_negcap( Z1, Z2, &Z5, queue ));
        magma_dprint_matrix( Z5, queue );
        
        magma_dmfree(&Z, queue );
        magma_dmfree(&Z1, queue );
        magma_dmfree(&Z2, queue );
        magma_dmfree(&Z3, queue );
        magma_dmfree(&Z4, queue );
        magma_dmfree(&Z5, queue );

        i++;
    }
    printf("%%=============================================================%%\n");
    printf("];\n");
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
