/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Stan Tomov

       @generated from src/zggrqf.cpp, normal z -> s, Mon Jun 25 18:24:05 2018 
*/
#include "magma_internal.h"

/*
 * Version 1 - LAPACK
 * Version 2 - MAGMA
 */
#define VERSION 2

/***************************************************************************//**

    Purpose   
    -------
    SGGRQF computes a generalized RQ factorization of an M-by-N matrix A   
    and a P-by-N matrix B:   

                A = R*Q,        B = Z*T*Q,   

    where Q is an N-by-N orthogonal matrix, Z is a P-by-P orthogonal   
    matrix, and R and T assume one of the forms:   

    if M <= N,  R = ( 0  R12 ) M,   or if M > N,  R = ( R11 ) M-N,   
                     N-M  M                           ( R21 ) N   
                                                         N   

    where R12 or R21 is upper triangular, and   

    if P >= N,  T = ( T11 ) N  ,   or if P < N,  T = ( T11  T12 ) P,   
                    (  0  ) P-N                         P   N-P   
                       N   

    where T11 is upper triangular.   

    In particular, if B is square and nonsingular, the GRQ factorization   
    of A and B implicitly gives the RQ factorization of A*inv(B):   

                 A*inv(B) = (R*inv(T))*Z'   

    where inv(B) denotes the inverse of the matrix B, and Z' denotes the   
    conjugate transpose of the matrix Z.   

    Arguments   
    ---------
    @param[in]
    m       INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    @param[in]
    p       INTEGER   
            The number of rows of the matrix B.  P >= 0.   

    @param[in]
    n       INTEGER   
            The number of columns of the matrices A and B. N >= 0.   

    @param[in,out]
    A       REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, if M <= N, the upper triangle of the subarray   
            A(1:M,N-M+1:N) contains the M-by-M upper triangular matrix R;   
            if M > N, the elements on and above the (M-N)-th subdiagonal   
            contain the M-by-N upper trapezoidal matrix R; the remaining   
            elements, with the array TAUA, represent the orthogonal   
            matrix Q as a product of elementary reflectors (see Further   
            Details).   

    @param[in]
    lda     INTEGER   
            The leading dimension of the array A. LDA >= max(1,M).   

    @param[out]
    taua    REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors which   
            represent the orthogonal matrix Q (see Further Details).   

    @param[in,out]
    B       REAL array, dimension (LDB,N)   
            On entry, the P-by-N matrix B.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(P,N)-by-N upper trapezoidal matrix T (T is   
            upper triangular if P >= N); the elements below the diagonal,   
            with the array TAUB, represent the orthogonal matrix Z as a   
            product of elementary reflectors (see Further Details).   

    @param[in]
    ldb     INTEGER   
            The leading dimension of the array B. LDB >= max(1,P).   

    @param[out]
    taub    REAL array, dimension (min(P,N))   
            The scalar factors of the elementary reflectors which   
            represent the orthogonal matrix Z (see Further Details).   

    @param[out]
    work    (workspace) REAL array, dimension (LWORK)   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    @param[in]
    lwork   INTEGER   
            The dimension of the array WORK. LWORK >= max(1,N,M,P).   
            For optimum performance LWORK >= max(N,M,P)*NB,   
            where NB is the optimal blocksize for the QR factorization 
            of a P-by-N matrix.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    @param[out]
    info    INTEGER   
      -     = 0:  successful exit   
      -     < 0:  if INFO=-i, the i-th argument had an illegal value.   

    Further Details   
    --------------- 
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - taua * v * v'   

    where taua is a real scalar, and v is a real vector with   
    v(n-k+i+1:n) = 0 and v(n-k+i) = 1; v(1:n-k+i-1) is stored on exit in   
    A(m-k+i,1:n-k+i-1), and taua in TAUA(i).   
    To form Q explicitly, use LAPACK subroutine ZUNGRQ.   
    To use Q to update another matrix, use LAPACK subroutine SORMRQ.   

    The matrix Z is represented as a product of elementary reflectors   

       Z = H(1) H(2) . . . H(k), where k = min(p,n).   

    Each H(i) has the form   

       H(i) = I - taub * v * v'   

    where taub is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:p) is stored on exit in B(i+1:p,i),   
    and taub in TAUB(i).   
    To form Z explicitly, use LAPACK subroutine SORGQR.   
    To use Z to update another matrix, use LAPACK subroutine SORMQR.   

    @ingroup magma_ggrqf
*******************************************************************************/
extern "C" magma_int_t
magma_sggrqf(magma_int_t m, magma_int_t p, magma_int_t n, 
             float *A, magma_int_t lda, 
             float *taua, 
             float *B, magma_int_t ldb, 
             float *taub, 
             float *work, magma_int_t lwork, 
             magma_int_t *info)
{
    #define  A(i_,j_)  (A + (i_) + (j_)*lda)
    
    magma_int_t i__1, lopt, lwkopt;
    
    *info = 0;
    magma_int_t nb = magma_get_sgeqrf_nb( p, n );
    lwkopt = max(max(n, m), p) * nb;
    work[0] = magma_smake_lwork( lwkopt );
    bool lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (p < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    } else if (ldb < max(1,p)) {
        *info = -8;
    } else /* if(complicated condition) */ {
        i__1 = max(1,m), i__1 = max(i__1,p);
        if (lwork < max(i__1,n) && ! lquery) {
            *info = -11;
        }
    }
    if (*info != 0) {
        magma_xerbla(__func__, -(*info) );
        return *info;
    } else if (lquery) {
        return *info;
    }
    
    /*     RQ factorization of M-by-N matrix A: A = R*Q */
    lapackf77_sgerqf(&m, &n, A, &lda, taua, work, &lwork, info);
    lopt = (magma_int_t) MAGMA_S_REAL(work[0]);
    
    /*     Update B := B*Q'    */
    i__1 = min(m,n);
    #if VERSION == 1
        lapackf77_sormrq(MagmaRightStr, MagmaTransStr, &p, &n, &i__1, 
                         A(max(0,m-n),0), &lda, taua, B, &ldb, work, &lwork, info);
    #else
        magma_sormrq(MagmaRight, MagmaTrans, p, n, i__1,
                     A(max(0,m-n),0), lda, taua, B, ldb, work, lwork, info);
    #endif

    lopt = max(lopt, (magma_int_t)MAGMA_S_REAL(work[0]));
    
    /* QR factorization of P-by-N matrix B: B = Z*T */
    magma_sgeqrf(p, n, B, ldb, taub, work, lwork, info);

    i__1 = max(lopt, (magma_int_t)MAGMA_S_REAL(work[0]));
    work[0] = magma_smake_lwork( i__1 );
    
    return *info;    
} /* magma_sggrqf */




