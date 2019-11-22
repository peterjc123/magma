/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @generated from testing/testing_zhetrf.cpp, normal z -> c, Fri Aug  2 17:10:11 2019
       @author Ichitaro Yamazaki
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
#include "magma_operators.h"  // for MAGMA_C_DIV
#include "testings.h"

/******************************************************************************/
// Initialize matrix to random.
// This ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix(
    magma_opts &opts,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda )
{
    magma_int_t iseed_save[4];
    for (magma_int_t i = 0; i < 4; ++i) {
        iseed_save[i] = opts.iseed[i];
    }

    magma_generate_matrix( opts, m, n, A, lda );

    // restore iseed
    for (magma_int_t i = 0; i < 4; ++i) {
        opts.iseed[i] = iseed_save[i];
    }
}

/******************************************************************************/
// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
float get_residual(
    magma_opts &opts,
    bool nopiv, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t ione = 1;
    
    magma_int_t upper = (uplo == MagmaUpper);
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info = 0;
    magma_int_t i;
    magmaFloatComplex *x, *b;
    
    // initialize RHS
    TESTING_CHECK( magma_cmalloc_cpu( &x, n ));
    TESTING_CHECK( magma_cmalloc_cpu( &b, n ));
    lapackf77_clarnv( &ione, ISEED, &n, b );
    blasf77_ccopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    if (nopiv) {
        if (upper) {
            blasf77_ctrsm( MagmaLeftStr, MagmaUpperStr,
                           MagmaConjTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (i=0; i < n; i++) {
                x[i] = MAGMA_C_DIV( x[i], A[i+i*lda] );
            }
            blasf77_ctrsm( MagmaLeftStr, MagmaUpperStr,
                           MagmaNoTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
        }
        else {
            blasf77_ctrsm( MagmaLeftStr, MagmaLowerStr,
                           MagmaNoTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (i=0; i < n; i++) {
                x[i] = MAGMA_C_DIV( x[i], A[i+i*lda] );
            }
            blasf77_ctrsm( MagmaLeftStr, MagmaLowerStr,
                           MagmaConjTransStr, MagmaUnitStr,
                           &n, &ione, &c_one,
                           A, &lda, x, &n );
        }
    }
    else {
        lapackf77_chetrs( lapack_uplo_const(uplo), &n, &ione, A, &lda, ipiv, x, &n, &info );
    }
    if (info != 0) {
        printf("lapackf77_chetrs returned error %lld: %s.\n",
               (long long) info, magma_strerror( info ));
    }
    // reset to original A
    init_matrix( opts, n, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_chemv( lapack_uplo_const(uplo), &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_clanhe( "Fro", lapack_uplo_const(uplo), &n, A, &lda, work );
    norm_r = lapackf77_clange( "Fro", &n, &ione, b, &n, work );
    norm_x = lapackf77_clange( "Fro", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_cprint( 1, n, b, 1 );
    
    magma_free_cpu( x );
    magma_free_cpu( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}

/******************************************************************************/
float get_residual_aasen(
    magma_opts &opts,
    bool nopiv, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    const magma_int_t ione = 1;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magmaFloatComplex *L, *T;
    #define  A(i,j) ( A[(i) + (j)*lda])
    #define  L(i,j) ( L[(i) + (j)*n])
    TESTING_CHECK( magma_cmalloc_cpu( &L, n*n ));
    memset( L, 0, n*n*sizeof(magmaFloatComplex) );

    magma_int_t i, j, piv;
    magma_int_t nb = magma_get_chetrf_aasen_nb(n);
    // extract L
    for (i=0; i < min(n,nb); i++) {
        L(i,i) = c_one;
    }
    for (i=nb; i < n; i++) {
        for (j=0; j < i-nb; j++) {
            L(i,nb+j) = A(i,j);
        }
        L(i,i) = c_one;
    }

    // solve
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info = 0;
    magmaFloatComplex *x, *b;
    
    // initialize RHS
    TESTING_CHECK( magma_cmalloc_cpu( &x, n ));
    TESTING_CHECK( magma_cmalloc_cpu( &b, n ));
    lapackf77_clarnv( &ione, ISEED, &n, b );
    blasf77_ccopy( &n, b, &ione, x, &ione );
    // pivot..
    for (i=0; i < n; i++) {
        piv = ipiv[i]-1;
        magmaFloatComplex val = x[i];
        x[i] = x[piv];
        x[piv] = val;
    }
    // forward solve
    blasf77_ctrsv( MagmaLowerStr, MagmaNoTransStr, MagmaUnitStr, &n, &L(0,0), &n, x, &ione );
    // banded solver
    magma_int_t nrhs = 1, *p = NULL;
    TESTING_CHECK( magma_imalloc_cpu( &p, n ));
    //#define CHESV_USE_CGESV
    #ifdef CHESV_USE_CGESV
        // using CGESV on banded matrix
        #define  T(i,j) ( T[(i) + (j)*n])
        // extract T
        TESTING_CHECK( magma_cmalloc_cpu( &T, n*n ));
        memset( T, 0, n*n*sizeof(magmaFloatComplex) );
        for (i=0; i < n; i++) {
            magma_int_t istart = max(0, i-nb);
            for (j=istart; j <= i; j++) {
                T(i,j) = A(i,j);
            }
            for (j=istart; j < i; j++) {
                T(j,i) = MAGMA_C_CONJ(A(i,j));
            }
        }
        // solve with T
        lapackf77_cgesv( &n, &nrhs, &T(0, 0), &n, p, x, &n, &info );
    #else
        // using CGBSV on banded matrix
        magma_int_t ldtb = 3*nb+1;
        // extract T
        TESTING_CHECK( magma_cmalloc_cpu( &T, ldtb * n ));
        memset( T, 0, ldtb*n*sizeof(magmaFloatComplex) );
        for (j=0; j<n; j++) {
            magma_int_t i0 = max(0, j-nb);
            magma_int_t i1 = min(n-1, j+nb);
            for (i=i0; i<j; i++) {
                T[nb + i-(j-nb) + j*ldtb] = MAGMA_C_CONJ(A(j,i));
            }
            for (i=j; i<=i1; i++) {
                T[nb + i-(j-nb) + j*ldtb] = A(i,j);
            }
        }
        // solve with T
        lapackf77_cgbsv(&n,&nb,&nb, &nrhs, T,&ldtb, p,x,&n, &info);
    #endif
    magma_free_cpu( p );

    // backward solve
    blasf77_ctrsv( MagmaLowerStr, MagmaConjTransStr, MagmaUnitStr, &n, &L(0,0), &n, x, &ione );
    // pivot..
    for (i=n-1; i >= 0; i--) {
        piv = ipiv[i]-1;
        magmaFloatComplex val = x[i];
        x[i] = x[piv];
        x[piv] = val;
    }

    // reset to original A
    init_matrix( opts, n, n, A, lda );

    // compute r = Ax - b, saved in b
    blasf77_chemv( lapack_uplo_const(uplo), &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_clanhe( "Fro", lapack_uplo_const(uplo), &n, A, &lda, work );
    norm_r = lapackf77_clange( "Fro", &n, &ione, b, &n, work );
    norm_x = lapackf77_clange( "Fro", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_cprint( 1, n, b, 1 );
    magma_free_cpu( L );
    magma_free_cpu( T );
    
    magma_free_cpu( x );
    magma_free_cpu( b );
    
    #undef T
    #undef L
    #undef A
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}

/******************************************************************************/
// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
float get_LDLt_error(
    magma_opts &opts,
    bool nopiv, magma_uplo_t uplo, magma_int_t N,
    magmaFloatComplex *LD, magma_int_t lda,
    magma_int_t *ipiv)
{
    const magmaFloatComplex c_one  = MAGMA_C_ONE;
    const magmaFloatComplex c_zero = MAGMA_C_ZERO;
    
    magma_int_t i, j, piv;
    magmaFloatComplex *A, *L, *D;
    float work[1], matnorm, residual;
    
    #define LD(i,j) (LD[(i) + (j)*lda])
    #define  A(i,j) ( A[(i) + (j)*N])
    #define  L(i,j) ( L[(i) + (j)*N])
    #define  D(i,j) ( D[(i) + (j)*N])

    TESTING_CHECK( magma_cmalloc_cpu( &A, N*N ));
    TESTING_CHECK( magma_cmalloc_cpu( &L, N*N ));
    TESTING_CHECK( magma_cmalloc_cpu( &D, N*N ));
    memset( L, 0, N*N*sizeof(magmaFloatComplex) );
    memset( D, 0, N*N*sizeof(magmaFloatComplex) );

    // set to original A, and apply pivoting
    init_matrix( opts, N, N, A, N );

    // symmetrize; the pivoting code below assumes a full matrix
    if (opts.uplo == MagmaLower) {
        // copy L to U
        for (j = 0; j < N; ++j) {
            for (i = 0; i < j; ++i) {
                A(i,j) = A(j,i);
            }
        }
    }
    else {
        // copy U to L
        for (j = 0; j < N; ++j) {
            for (i = 0; i < j; ++i) {
                A(j,i) = A(i,j);
            }
        }
    }

    if (uplo == MagmaUpper) {
        for (j=N-1; j >= 0; j--) {
            piv = (nopiv ? j+1 : ipiv[j]);
            if (piv < 0) {
                piv = -(piv+1);
                // extract 2-by-2 pivot
                D(j,j)     = LD(j,j);
                D(j,j-1)   = MAGMA_C_CONJ(LD(j-1,j));
                D(j-1,j)   = LD(j-1,j);
                D(j-1,j-1) = LD(j-1,j-1);
                // exract L
                L(j,j) = c_one;
                for (i=0; i < j-1; i++) {
                    L(i,j) = LD(i,j);
                }
                j--;
                L(j,j) = c_one;
                for (i=0; i < j; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=j+2; i < N; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
            else {
                piv = piv-1;
                // extract 1-by-1 pivot
                D(j,j) = LD(j,j);
                // exract L
                L(j,j) = c_one;
                for (i=0; i < j; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=j+1; i < N; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
        }
        if (nopiv) {
            // compute W = D*U
            blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, D, &N, L, &N, &c_zero, LD, &lda);
            // compute D = U^H*W
            blasf77_cgemm(MagmaConjTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, L, &N, LD, &lda, &c_zero, D, &N);
        }
        else {
            // compute W = U*D
            blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                          &c_one, L, &N, D, &N, &c_zero, LD, &lda);
            // compute D = W*U^H
            blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                          &c_one, LD, &lda, L, &N, &c_zero, D, &N);
        }
    }
    else {
        for (j=0; j < N; j++) {
            piv = (nopiv ? j+1 : ipiv[j]);
            if (piv < 0) {
                piv = -(piv+1);
                // extract 2-by-2 pivot
                D(j,j)     = LD(j,j);
                D(j,j+1)   = MAGMA_C_CONJ(LD(j+1,j));
                D(j+1,j)   = LD(j+1,j);
                D(j+1,j+1) = LD(j+1,j+1);
                // exract L
                L(j,j) = c_one;
                for (i=j+2; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                j++;
                L(j,j) = c_one;
                for (i=j+1; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=0; i < j-1; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
            else {
                piv = piv-1;
                // extract 1-by-1 pivot
                D(j,j) = LD(j,j);
                // exract L
                L(j,j) = c_one;
                for (i=j+1; i < N; i++) {
                    L(i,j) = LD(i,j);
                }
                if (piv != j) {
                    // apply row-pivoting to previous L
                    for (i=0; i < j; i++) {
                        magmaFloatComplex val = L(j,i);
                        L(j,i) = L(piv,i);
                        L(piv,i) = val;
                    }
                    // apply row-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(j,i);
                        A(j,i) = A(piv,i);
                        A(piv,i) = val;
                    }
                    // apply col-pivoting to A
                    for (i=0; i < N; i++) {
                        magmaFloatComplex val = A(i,j);
                        A(i,j) = A(i,piv);
                        A(i,piv) = val;
                    }
                }
            }
        }
        // compute W = L*D
        blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                      &c_one, L, &N, D, &N, &c_zero, LD, &lda);
        // compute D = W*L^H
        blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                      &c_one, LD, &lda, L, &N, &c_zero, D, &N);
    }
    // compute norm of A
    matnorm = lapackf77_clanhe( "Fro", lapack_uplo_const(uplo), &N, A, &lda, work);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < N; i++ ) {
            D(i,j) = MAGMA_C_SUB( D(i,j), A(i,j) );
        }
    }
    residual = lapackf77_clange( "Fro", &N, &N, D, &N, work);

    magma_free_cpu( A );
    magma_free_cpu( L );
    magma_free_cpu( D );

    return residual / (matnorm * N);
}

/******************************************************************************/
float get_LTLt_error(
    magma_opts &opts,
    bool nopiv, magma_uplo_t uplo, magma_int_t N,
    magmaFloatComplex *LT, magma_int_t lda,
    magma_int_t *ipiv)
{
    float work[1], matnorm, residual;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex *A, *L, *T;
    
    #define LT(i,j) (LT[(i) + (j)*lda])
    #define  T(i,j) ( T[(i) + (j)*N])
    
    TESTING_CHECK( magma_cmalloc_cpu( &A, N*N ));
    TESTING_CHECK( magma_cmalloc_cpu( &L, N*N ));
    TESTING_CHECK( magma_cmalloc_cpu( &T, N*N ));
    memset( L, 0, N*N*sizeof(magmaFloatComplex) );
    memset( T, 0, N*N*sizeof(magmaFloatComplex) );

    magma_int_t i, j, istart, piv;
    magma_int_t nb = magma_get_chetrf_aasen_nb(N);
    
    // for debuging
    /*
    magma_int_t *p;
    TESTING_CHECK( magma_imalloc_cpu( &p, n ));
    for (i=0; i < N; i++) {
        p[i] = i;
    }
    for (i=0; i < N; i++) {
        piv = ipiv[i]-1;
        i2 = p[piv];
        p[piv] = p[i];
        p[i] = i2;
    }
    printf( " p=[" );
    for (i=0; i < N; i++) {
        printf("%lld ", (long long) p[i] );
    }
    printf( "];\n" );
    magma_free_cpu( p );
    */
    
    // extract T
    for (i=0; i < N; i++) {
        istart = max(0, i-nb);
        for (j=istart; j <= i; j++) {
            T(i,j) = LT(i,j);
        }
        for (j=istart; j < i; j++) {
            T(j,i) = MAGMA_C_CONJ( LT(i,j) );
        }
    }
    //printf( "T=" );
    //magma_cprint(N,N, &T(0,0),N);
    // extract L
    for (i=0; i < min(N,nb); i++) 
    {
        L(i,i) = c_one;
    }
    for (i=nb; i < N; i++)
    {
        for (j=0; j < i-nb; j++) {
            L(i,nb+j) = LT(i,j);
        }
        L(i,i) = c_one;
    }
    //printf( "L=" );
    //magma_cprint(N,N, &L(0,0),N);

    // compute LD = L*T
    blasf77_cgemm(MagmaNoTransStr, MagmaNoTransStr, &N, &N, &N,
                  &c_one, L, &N, T, &N, &c_zero, LT, &lda);
    // compute T = LD*L^H
    blasf77_cgemm(MagmaNoTransStr, MagmaConjTransStr, &N, &N, &N,
                  &c_one, LT, &lda, L, &N, &c_zero, T, &N);

    // compute norm of A
    init_matrix( opts, N, N, A, N );
    matnorm = lapackf77_clanhe( "Fro", lapack_uplo_const(uplo), &N, A, &lda, work);
    //printf( "A0=" );
    //magma_cprint(N,N, &A(0,0),N);

    // symmetrize; the pivoting code below assumes a full matrix
    if (opts.uplo == MagmaLower) {
        // copy L to U
        for (j = 0; j < N; ++j) {
            for (i = 0; i < j; ++i) {
                A(i,j) = A(j,i);
            }
        }
    }
    else {
        // copy U to L
        for (j = 0; j < N; ++j) {
            for (i = 0; i < j; ++i) {
                A(j,i) = A(i,j);
            }
        }
    }

    // apply symmetric pivoting
    for (j=0; j < N; j++) {
        piv = ipiv[j]-1;
        if (piv != j) {
            // apply row-pivoting to A
            for (i=0; i < N; i++) {
                magmaFloatComplex val = A(j,i);
                A(j,i) = A(piv,i);
                A(piv,i) = val;
            }
            // apply col-pivoting to A
            for (i=0; i < N; i++) {
                magmaFloatComplex val = A(i,j);
                A(i,j) = A(i,piv);
                A(i,piv) = val;
            }
        }
    }

    // compute factorization error
    for(j = 0; j < N; j++ ) {
        for(i = 0; i < N; i++ ) {
            T(i,j) = MAGMA_C_SUB( T(i,j), A(i,j) );
        }
    }
    residual = lapackf77_clange( "Fro", &N, &N, T, &N, work);

    magma_free_cpu( A );
    magma_free_cpu( L );
    magma_free_cpu( T );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chetrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magmaFloatComplex *h_A, *work, temp;
    real_Double_t   gflops, gpu_perf, gpu_time = 0.0, cpu_perf=0, cpu_time=0;
    float          error, error_lapack = 0.0;
    magma_int_t     *ipiv;
    magma_int_t     cpu_panel = 1, N, n2, lda, lwork, info;
    magma_int_t     cpu = 0, nopiv = 0, nopiv_gpu = 0, row = 0, aasen = 0;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    if (opts.version == 3 || opts.version == 4) {
        // default in these cases; re-parse args
        opts.matrix = "rand_dominant";
        opts.parse_opts( argc, argv );
        //printf( "matrix %s\n", opts.matrix.c_str() );
    }

    // TODO: this doesn't work. Options need to be added to parse_opts()
    //for (int i = 1; i < argc; ++i) {
    //    if ( strcmp("--cpu-panel", argv[i]) == 0) cpu_panel = 1;
    //    if ( strcmp("--gpu-panel", argv[i]) == 0) cpu_panel = 0;
    //}

    printf( "%% --version 1 = Bunch-Kauffman (CPU)\n"
            "%%           2 = Bunch-Kauffman (GPU) -- not yet available\n"
            "%%           3 = No-piv (CPU) -- uses random, diagonally dominant matrix by default\n"
            "%%           4 = No-piv (GPU) -- uses random, diagonally dominant matrix by default\n"
            "%%           6 = Aasen's\n"
            "\n" );
    printf( "%% version %lld: ", (long long) opts.version );
    switch (opts.version) {
        case 1:
            cpu = 1;
            printf( "CPU-interface to Bunch-Kauffman on GPU" );
            break;
        case 2:
            //gpu = 1;
            printf( "GPU-interface to Bunch-Kauffman on GPU" );
            printf( "\n%% not yet available.\n" );
            return 0;
            break;
        case 3:
            nopiv = 1;
            printf( "CPU-interface to hybrid non-pivoted LDLt (A is SPD)" );
            break;
        case 4:
            nopiv_gpu = 1;
            printf( "GPU-interface to hybrid non-pivoted LDLt (A is SPD)" );
            break;
        //case 5:
        //    row = 1;
        //    printf( "%% Bunch-Kauffman: GPU-only version (row-major)" );
        //    break;
        case 6:
            aasen = 1;
            printf( "CPU-Interface to Aasen's, %s", (cpu_panel ? "CPU panel" : "GPU panel") );
            break;
        default:
            printf( "unknown version\n" );
            return 0;
    }
    printf( ", %s\n", lapack_uplo_const(opts.uplo) );

    float tol = opts.tolerance * lapackf77_slamch("E");

    if ( opts.check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PAP^H - LDL^H|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            gflops = FLOPS_CPOTRF( N ) / 1e9;
            
            TESTING_CHECK( magma_imalloc_pinned( &ipiv, N ));
            TESTING_CHECK( magma_cmalloc_pinned( &h_A,  n2 ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lwork = -1;
                lapackf77_chetrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, ipiv, &temp, &lwork, &info );
                lwork = (magma_int_t)MAGMA_C_REAL( temp );
                TESTING_CHECK( magma_cmalloc_cpu( &work, lwork ));

                init_matrix( opts, N, N, h_A, lda );
                cpu_time = magma_wtime();
                lapackf77_chetrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, ipiv, work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_chetrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                error_lapack = get_residual( opts, nopiv, opts.uplo, N, h_A, lda, ipiv );

                magma_free_cpu( work );
            }
           
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( opts, N, N, h_A, lda );

            //printf( "A0=" );
            //magma_cprlong( N, N, h_A, lda );
            if (nopiv) {
                // CPU-interface to non-piv LDLt
                gpu_time = magma_wtime();
                magma_chetrf_nopiv( opts.uplo, N, h_A, lda, &info);
                gpu_time = magma_wtime() - gpu_time;
            }
            else if (cpu) {
                // CPU-interface to Bunch-Kauffman LDLt
                gpu_time = magma_wtime();
                magma_chetrf( opts.uplo, N, h_A, lda, ipiv, &info);
                gpu_time = magma_wtime() - gpu_time;
            }
            else if (nopiv_gpu) {
                // GPU-interface to non-piv LDLt
                magma_int_t ldda = magma_roundup( N, opts.align );
                magmaFloatComplex_ptr d_A;
                TESTING_CHECK( magma_cmalloc( &d_A, N*ldda ));
                magma_csetmatrix(N, N, h_A, lda, d_A, ldda, opts.queue );
                gpu_time = magma_wtime();
                magma_chetrf_nopiv_gpu( opts.uplo, N, d_A, ldda, &info);
                gpu_time = magma_wtime() - gpu_time;
                magma_cgetmatrix(N, N, d_A, ldda, h_A, lda, opts.queue );
                magma_free( d_A );
            }
            else if (aasen) {
                // CPU-interface to Aasen's LTLt
                gpu_time = magma_wtime();
                magma_chetrf_aasen( opts.uplo, cpu_panel, N, h_A, lda, ipiv, &info);
                gpu_time = magma_wtime() - gpu_time;
            }
            else if (row) {
                //magma_chetrf_gpu_row( opts.uplo, N, h_A, lda, ipiv, work, lwork, &info);
            }
            else {
                //magma_chetrf_hybrid( opts.uplo, N, h_A, lda, ipiv, work, lwork, &info);
            }
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_chetrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) N, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) N, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 && info == 0) {
                if (aasen) {
                    error = get_residual_aasen( opts, (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                else {
                    error = get_residual( opts, (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                printf("   %8.2e   %s", error, (error < tol ? "ok" : "failed"));
                if (opts.lapack)
                    printf(" (lapack rel.res. = %8.2e)", error_lapack);
                printf("\n");
                status += ! (error < tol);
            }
            else if ( opts.check && info == 0 ) {
                if (aasen) {
                    error = get_LTLt_error( opts, (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                else {
                    error = get_LDLt_error( opts, (nopiv | nopiv_gpu), opts.uplo, N, h_A, lda, ipiv );
                }
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---   \n");
            }
 
            magma_free_pinned( ipiv );
            magma_free_pinned( h_A  );
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
