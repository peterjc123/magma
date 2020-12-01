/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from testing/testing_zlacpy_batched.cpp, normal z -> c, Thu Oct  8 23:05:44 2020
       @author Mark Gates

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing clacpy_batched
   Code is very similar to testing_cgeadd_batched.cpp
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           error, work[1];
    magmaFloatComplex  c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *h_B, *h_R;
    magmaFloatComplex_ptr d_A, d_B;
    magmaFloatComplex **hA_array, **hB_array, **dA_array, **dB_array;
    magma_int_t M, N, sizeA, sizeB, lda, ldda, ldb, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    magma_int_t batchCount = opts.batchcount;

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    printf("%% BatchCount   uplo      M     N   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("%%===============================================================================\n");
    for( int iuplo = 0; iuplo < 3; ++iuplo ) {
        for( int itest = 0; itest < opts.ntest; ++itest ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda   = M;
            ldb   = lda;
            ldda  = magma_roundup( M, opts.align );  // multiple of 32 by default
            lddb  = ldda;
            sizeA = batchCount*lda*N;
            sizeB = batchCount*ldb*N;

            if ( uplo[iuplo] == MagmaLower ) {
                // load & save lower trapezoid (with diagonal)
                if ( M > N ) {
                    gbytes = batchCount * 2. * sizeof(magmaFloatComplex) * (1.*M*N - 0.5*N*(N-1)) / 1e9;
                } else {
                    gbytes = batchCount * 2. * sizeof(magmaFloatComplex) * 0.5*M*(M+1) / 1e9;
                }
            }
            else if ( uplo[iuplo] == MagmaUpper ) {
                // load & save upper trapezoid (with diagonal)
                if ( N > M ) {
                    gbytes = batchCount * 2. * sizeof(magmaFloatComplex) * (1.*M*N - 0.5*M*(M-1)) / 1e9;
                } else {
                    gbytes = batchCount * 2. * sizeof(magmaFloatComplex) * 0.5*N*(N+1) / 1e9;
                }
            }
            else {
                // load & save entire matrix
                gbytes = batchCount * 2. * sizeof(magmaFloatComplex) * 1.*M*N / 1e9;
            }

            TESTING_CHECK( magma_cmalloc_cpu( &h_A, batchCount * lda * N ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B, batchCount * ldb * N ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_R, batchCount * ldb * N ));
            TESTING_CHECK( magma_cmalloc( &d_A, batchCount * ldda*N ));
            TESTING_CHECK( magma_cmalloc( &d_B, batchCount * ldda*N ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array, batchCount*sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array, batchCount*sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc_cpu( (void**) &hA_array, batchCount * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc_cpu( (void**) &hB_array, batchCount * sizeof(magmaFloatComplex*) ));


            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix( M, N*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( M, N*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            // setup pointers
            magma_cset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_cset_pointer( dB_array, d_B, lddb, 0, 0, lddb*N, batchCount, opts.queue );

            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_clacpy_batched( uplo[iuplo], M, N, dA_array, ldda, dB_array, lddb, batchCount, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;

            magma_cgetmatrix( M, N*batchCount, d_B, lddb, h_R, ldb, opts.queue );
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            for(int s = 0; s < batchCount; s++) {
                hA_array[s] = h_A + s * lda * N;
                hB_array[s] = h_B + s * ldb * N;
            }

            cpu_time = magma_wtime();
            blas_clacpy_batched(uplo[iuplo], M, N, hA_array, lda, hB_array, ldb, batchCount );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;

            /* =====================================================================
               Check the result
               =================================================================== */
            magma_int_t NN = batchCount*N;
            blasf77_caxpy(&sizeB, &c_neg_one, h_B, &ione, h_R, &ione);
            error = lapackf77_clange("f", &M, &NN, h_R, &ldb, work);
            bool okay = (error == 0);
            status += ! okay;

            printf("%10lld   %7s %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (long long) batchCount, lapack_uplo_const(uplo[iuplo]),
                   (long long) M, (long long) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (okay ? "ok" : "failed") );

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_R );
            magma_free( d_A );
            magma_free( d_B );

            magma_free_cpu( hA_array );
            magma_free_cpu( hB_array );
            magma_free( dA_array );
            magma_free( dB_array );
            fflush( stdout );
        }
            if ( opts.niter > 1 ) {
                printf( "\n" );
            }
        }
        printf("\n");
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
