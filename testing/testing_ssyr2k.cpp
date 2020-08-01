/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @generated from testing/testing_zher2k.cpp, normal z -> s, Sun Mar 29 20:48:32 2020
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define REAL


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssyr2k
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    float          magma_error, dev_error, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An, Bk, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    float *hA, *hB, *hC, *hCmagma, *hCdev;
    magmaFloat_ptr dA, dB, dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // See testing_sgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans; 
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #endif

    #ifdef HAVE_CUBLAS
    // for CUDA, we can check MAGMA vs. CUBLAS, without running LAPACK
    printf("%% If running lapack (option --lapack), MAGMA and %s errors are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
            g_platform_str, g_platform_str );
        
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   CUBLAS error\n");
   #else
    // for others, we need LAPACK for check
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error\n");
    #endif
    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SSYR2K(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
                ldb = Bn = N;
                Bk = K;
            } else {
                lda = An = K;
                Ak = N;
                ldb = Bn = K;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeB = ldb*Ak;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_smalloc_cpu( &hA,      lda*Ak ));
            TESTING_CHECK( magma_smalloc_cpu( &hB,      ldb*Bk ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,      ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCmagma, ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCdev,   ldc*N  ));
            
            TESTING_CHECK( magma_smalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_smalloc( &dB, lddb*Bk ));
            TESTING_CHECK( magma_smalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );
            
            // for error checks
            float Anorm = lapackf77_slange( "F", &An, &Ak, hA, &lda, work );
            float Bnorm = lapackf77_slange( "F", &Bn, &Bk, hB, &ldb, work );
            float Cnorm = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hC, &ldc, work );
            
            
            magma_ssetmatrix( An, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_ssetmatrix( Bn, Bk, hB, ldb, dB(0,0), lddb, opts.queue );
            /* =====================================================================
               Performs operation using MAGMABLAS ( for CUDA and HIP)
               =================================================================== */
            magma_ssetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ssyr2k( opts.uplo, opts.transA, N, K,
                              alpha, dA(0,0), ldda,
                                     dB(0,0), lddb,
                              beta,  dC(0,0), lddc, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_perf = gflops / magma_time;

            magma_sgetmatrix( N, N, dC(0,0), lddc, hCmagma, ldc, opts.queue );

           /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_ssetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            #if HAVE_CUBLAS
            dev_time = magma_sync_wtime( opts.queue );
            magma_ssyr2k( opts.uplo, opts.transA, N, K,
                          alpha, dA(0,0), ldda,
                                 dB(0,0), lddb,
                          beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            #endif
            
            magma_sgetmatrix( N, N, dC(0,0), lddc, hCdev, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ssyr2k( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_sgemm for formula.
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCmagma, &ione );
                magma_error = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hCmagma, &ldc, work )
                            / (2*sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                #ifdef HAVE_CUBLAS
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hCdev, &ldc, work )
                            / (2*sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                bool okay = (magma_error < tol && dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %7.2f (%7.2f)   %8.2e      %8.2e   %s\n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, dev_error, (okay ? "ok" : "failed"));
                #else
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,   1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
                #endif
                
            }
            else {
                #ifdef HAVE_CUBLAS
                blasf77_saxpy( &sizeC, &c_neg_one, hCdev, &ione, hCmagma, &ione );
                magma_error = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hCmagma, &ldc, work )
                            / (2*sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )    %8.2e         ---      %s\n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time,
                       dev_perf,   1000.*dev_time,
                       magma_error, (okay ? "ok" : "failed"));
                #else
                printf("%5lld %5lld   %7.2f (%7.2f)   ---   (  ---  )     ---  \n",
                       (long long) N, (long long) K,
                       magma_perf, 1000.*magma_time);
                #endif
                
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCdev );
            magma_free_cpu( hCmagma );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
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
