/*
    -- MAGMA (version 2.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2019

       @generated from testing/testing_zgenerate.cpp, normal z -> c, Wed Jan  2 14:18:52 2019

       @author Mark Gates
*/
#include "testings.h"

/******************************************************************************/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );

    // constants
    const float eps = lapackf77_slamch( "precision" ); // 1.2e-7 or 2.2e-16

    // locals
    real_Double_t time, time2;
    magma_int_t m, n, minmn, lda;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    printf( "%% * cond and condD are not applicable to all matrix types.\n" );
    printf( "%%     M     N       cond*      condD*   CPU time (sec)   time2 (sec)   Matrix\n" );
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            float cond = opts.cond;
            if (cond == 0) {
                cond = 1/eps;  // default value
            }
            m = opts.msize[itest];
            n = opts.nsize[itest];
            lda = m;
            minmn = min( m, n );
            Matrix<magmaFloatComplex> A( m, n, lda );
            Vector<float> sigma( minmn );

            time2 = magma_wtime();
            magma_generate_matrix( opts, A, sigma );
            time2 = magma_wtime() - time2;

            time = magma_wtime();
            magma_generate_matrix( opts, A.m, A.n, A(0,0), A.ld, sigma(0) );
            time = magma_wtime() - time;

            printf( "%% %5lld %5lld   %9.2e   %9.2e   %9.4f     %9.4f        %s\n",
                    (long long) m, (long long) n,
                    cond, opts.condD, time, time2, opts.matrix.c_str() );

            if (opts.verbose) {
                printf( "sigma = " ); magma_sprint( 1, minmn, sigma(0), 1 );
                printf( "A = "     ); magma_cprint( m, n, A(0,0), lda );
            }
        }
    }

    TESTING_CHECK( magma_finalize() );
}
