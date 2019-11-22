/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from testing/testing_zlacpy_batched.cpp, normal z -> c, Fri Aug  2 17:10:12 2019
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
    magmaFloatComplex *h_A, *h_B;
    magmaFloatComplex_ptr d_A, d_B;
    magmaFloatComplex **hAarray, **hBarray, **dAarray, **dBarray;
    magma_int_t M, N, mb, nb, size, lda, ldda, mstride, nstride, ntile;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    mb = (opts.nb == 0 ? 32 : opts.nb);
    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*mb;
    nstride = 3*nb;
    
    printf("%% mb=%lld, nb=%lld, mstride=%lld, nstride=%lld\n", (long long) mb, (long long) nb, (long long) mstride, (long long) nstride );
    printf("%%   M     N ntile    CPU Gflop/s (ms)    GPU Gflop/s (ms)   check\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            size   = lda*N;
            
            if ( N < nb || M < nb ) {
                ntile = 0;
            } else {
                ntile = min( (M - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            gbytes = 2.*mb*nb*ntile / 1e9;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A, lda *N ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B, lda *N ));
            TESTING_CHECK( magma_cmalloc( &d_A, ldda*N ));
            TESTING_CHECK( magma_cmalloc( &d_B, ldda*N ));
            
            TESTING_CHECK( magma_malloc_cpu( (void**) &hAarray, ntile * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc_cpu( (void**) &hBarray, ntile * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dAarray, ntile * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dBarray, ntile * sizeof(magmaFloatComplex*) ));
            
            lapackf77_clarnv( &ione, ISEED, &size, h_A );
            lapackf77_clarnv( &ione, ISEED, &size, h_B );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( M, N, h_B, lda, d_B, ldda, opts.queue );
            
            // setup pointers
            for( magma_int_t tile = 0; tile < ntile; ++tile ) {
                magma_int_t offset = tile*mstride + tile*nstride*ldda;
                hAarray[tile] = &d_A[offset];
                hBarray[tile] = &d_B[offset];
            }
            magma_setvector( ntile, sizeof(magmaFloatComplex*), hAarray, 1, dAarray, 1, opts.queue );
            magma_setvector( ntile, sizeof(magmaFloatComplex*), hBarray, 1, dBarray, 1, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_clacpy_batched( MagmaFull, mb, nb, dAarray, ldda, dBarray, ldda, ntile, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( magma_int_t tile = 0; tile < ntile; ++tile ) {
                magma_int_t offset = tile*mstride + tile*nstride*lda;
                lapackf77_clacpy( MagmaFullStr, &mb, &nb,
                                  &h_A[offset], &lda,
                                  &h_B[offset], &lda );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_cgetmatrix( M, N, d_B, ldda, h_A, lda, opts.queue );
            
            blasf77_caxpy(&size, &c_neg_one, h_A, &ione, h_B, &ione);
            error = lapackf77_clange("f", &M, &N, h_B, &lda, work);
            bool okay = (error == 0);
            status += ! okay;

            printf("%5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (long long) M, (long long) N, (long long) ntile,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (okay ? "ok" : "failed") );
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free( d_A );
            magma_free( d_B );
            
            magma_free_cpu( hAarray );
            magma_free_cpu( hBarray );
            magma_free( dAarray );
            magma_free( dBarray );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
