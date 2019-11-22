/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from sparse/testing/testing_zsolver_rhs_scaling.cpp, normal z -> d, Fri Aug  2 17:10:14 2019
       @author Hartwig Anzt
       @author Stephen Wood
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_dopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    //Chronometry
    real_Double_t tempo1, tempo2, t_transfer = 0.0;
    
    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    double negone = MAGMA_D_MAKE(-1.0, 0.0);
    magma_d_matrix A={Magma_CSR}, B={Magma_CSR}, B_d={Magma_CSR};
    magma_d_matrix x={Magma_CSR}, x_h={Magma_CSR}, b_h={Magma_DENSE}, b={Magma_DENSE};
    magma_d_matrix A_org={Magma_CSR};
    magma_d_matrix b_org={Magma_DENSE};
    magma_d_matrix scaling_factors={Magma_DENSE};
    magma_d_matrix y_check={Magma_DENSE};
    double residual = 0.0;
    
    int i=1;
    TESTING_CHECK( magma_dparse_opts( argc, argv, &zopts, &i, queue ));
    // TODO: integrate option to specify which sides diagonal matrices 
    // of scaling factors are applied to the orginal matrix into the 
    // command line options
    //magma_side_t side = MagmaLeft;
    //magma_side_t side = MagmaRight;
    magma_side_t side = MagmaBothSides;
    
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    // make sure preconditioner is NONE for unpreconditioned systems
    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PCGMERGE &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF  &&
         zopts.solver_par.solver != Magma_PIDR  &&
         zopts.solver_par.solver != Magma_PCGS  &&
         zopts.solver_par.solver != Magma_PCGSMERGE &&
         zopts.solver_par.solver != Magma_PTFQMR &&
         zopts.solver_par.solver != Magma_PTFQMRMERGE &&
         zopts.solver_par.solver != Magma_LOBPCG ) 
    {
         zopts.precond_par.solver = Magma_NONE;
    }
    TESTING_CHECK( magma_dsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));
    // more iterations

    
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_dm_5stencil(  laplace_size, &A, queue ));
            TESTING_CHECK( magma_dvinit( &b_h, Magma_CPU, A.num_cols, 1, one, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_d_csr_mtx( &A,  argv[i], queue ));
            if ( strcmp("ONES", argv[i+1]) == 0 ) {
                TESTING_CHECK( magma_dvinit( &b_h, Magma_CPU, A.num_cols, 1, one, queue ));
            }
            else {
                TESTING_CHECK( magma_dvread( &b_h, A.num_cols, argv[i+1], queue ));
            }
            i++;
        }

        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        
        printf("matrixinfo = [\n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m\n");
        printf("%%=============================================================%%\n");
        printf("  %8lld  %8lld      %10lld        %10lld\n",
               (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz,
               (long long) (A.nnz/A.num_rows) );
        printf("%%=============================================================%%\n");
        printf("];\n");
        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        TESTING_CHECK( magma_deigensolverinfo_init( &zopts.solver_par, queue ));
        fflush(stdout);
        

        t_transfer = 0.0;
        zopts.precond_par.setuptime = 0.0;
        zopts.precond_par.runtime = 0.0;

        // copy the original system before scaling
        TESTING_CHECK( magma_dmtransfer( A, &A_org, Magma_CPU, Magma_DEV, queue ));
        TESTING_CHECK( magma_dmtransfer( b_h, &b_org, Magma_CPU, Magma_DEV, queue ));
        
        // scale matrix
        if ( zopts.scaling != Magma_NOSCALE ) {
            TESTING_CHECK( magma_dvinit( &scaling_factors, Magma_CPU, A.num_rows, 1, zero, queue ));
            
            // magma_dmscale_matrix_rhs to be deprecated
            //TESTING_CHECK( magma_dmscale_matrix_rhs( &A, &b_h, &scaling_factors, zopts.scaling, queue ));
            
            TESTING_CHECK( magma_dmscale_generate( 1, &zopts.scaling, &side,
              &A, &scaling_factors, queue ) );
            TESTING_CHECK( magma_dmscale_apply( 1, &side,
              &scaling_factors, &A, queue ) );
            TESTING_CHECK( magma_ddimv( &scaling_factors, &b_h, queue ) );
        }
        
        i++;
        tempo1 = magma_sync_wtime( queue );
        magma_d_vtransfer(b_h, &b, Magma_CPU, Magma_DEV, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            TESTING_CHECK( magma_d_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }
        // make sure alignment is 1 for SELLP
        B.alignment = 1;
        B.blocksize = 256;
        TESTING_CHECK( magma_dmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_dmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        TESTING_CHECK( magma_dvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));
        
        info = magma_d_solver( B_d, b, &x, &zopts, queue );
        if( info != 0 ) {
            printf("%% error: solver returned: %s (%lld).\n",
                magma_strerror( info ), (long long) info );
        }
        
        TESTING_CHECK( magma_dvinit( &y_check, Magma_DEV, A.num_rows, 1, zero, queue ));
        TESTING_CHECK( magma_d_spmv( one, B_d, x, zero, y_check, queue ) );
        magma_daxpy( A.num_rows, negone, b.val, 1, y_check.val, 1, queue );
        residual = magma_dnrm2( A.num_rows, y_check.val, 1, queue ); 
        printf("%% scaled system residual check = %e\n", residual);
        
        TESTING_CHECK( magma_dvinit( &y_check, Magma_DEV, A.num_rows, 1, zero, queue ));
        TESTING_CHECK( magma_d_spmv( one, A_org, x, zero, y_check, queue ) );
        magma_daxpy( A_org.num_rows, negone, b_org.val, 1, y_check.val, 1, queue );
        residual = magma_dnrm2( A_org.num_rows, y_check.val, 1, queue ); 
        printf("%% original system residual check = %e\n", residual);
        
        if ( ( zopts.scaling != Magma_NOSCALE ) && 
            ( ( side == MagmaRight ) // Magma_UNITROWCOL and Magma_UNITDIAGCOL to be deprecated 
            || ( side == MagmaBothSides )
            || ( zopts.scaling == Magma_UNITROWCOL ) 
            || ( zopts.scaling == Magma_UNITDIAGCOL ) ) ) {
            printf("%% rescaling computed solution for scaling %d\n", zopts.scaling);
            
            //magma_dmfree(&x_h, queue );
            //magma_d_vtransfer(x, &x_h, Magma_DEV, Magma_CPU, queue);
            
            //TESTING_CHECK( magma_ddimv( &scaling_factors, &x_h, queue ) );
            TESTING_CHECK( magma_ddimv( &scaling_factors, &x, queue ) );
            
            //magma_dmfree(&x, queue );
            //magma_d_vtransfer(x_h, &x, Magma_CPU, Magma_DEV, queue);
            
            TESTING_CHECK( magma_dvinit( &y_check, Magma_DEV, A.num_rows, 1, zero, queue ));
            TESTING_CHECK( magma_d_spmv( one, A_org, x, zero, y_check, queue ) );
            magma_daxpy( A_org.num_rows, negone, b_org.val, 1, y_check.val, 1, queue );
            residual = magma_dnrm2( A_org.num_rows, y_check.val, 1, queue ); 
            printf("%% original system residual check = %e\n", residual);
        }
        
        magma_dmfree(&x_h, queue );
        tempo1 = magma_sync_wtime( queue );
        magma_d_vtransfer(x, &x_h, Magma_DEV, Magma_CPU, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;  
        
        printf("data = [\n");
        magma_dsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
        
        printf("precond_info = [\n");
        printf("%%   setup  runtime\n");        
        printf("  %.6f  %.6f\n",
           zopts.precond_par.setuptime, zopts.precond_par.runtime );
        printf("];\n\n");
        
        printf("transfer_time = %.6f;\n\n", t_transfer);
        magma_dmfree(&x, queue );
        magma_dmfree(&b, queue );
        magma_dmfree(&B_d, queue );
        magma_dmfree(&B, queue );
        magma_dsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
        fflush(stdout);
        
        magma_dmfree(&B_d, queue );
        magma_dmfree(&B, queue );
        magma_dmfree(&A, queue );
        magma_dmfree(&x, queue );
        magma_dmfree(&x_h, queue );
        magma_dmfree(&A_org, queue );
        magma_dmfree(&b_org, queue );
        magma_dmfree(&scaling_factors, queue );
        magma_dmfree(&y_check, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
