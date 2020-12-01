/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from magmablas/zher2k.cpp, normal z -> d, Thu Oct  8 23:05:32 2020
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"

#define PRECISION_d
/******************************************************************************/
// symmetric case (real precisions only)
#if defined(PRECISION_c) || defined(PRECISION_z)
extern "C"
void magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    double cbeta = MAGMA_D_MAKE(beta, 0.);
    double c_one = MAGMA_D_MAKE(1., 0.);
    
    if ( uplo != MagmaLower && uplo != MagmaUpper) {
        info = -1; 
    } else if ( trans != MagmaNoTrans && trans != MagmaConjTrans) {
        info = -2;
    } else if ( n < 0 ) {
        info = -3;
    } else if ( k < 0 ) {
        info = -4;
    } else if ( ((trans == MagmaNoTrans) && ldda < max(1,n)) ||
                ((trans != MagmaNoTrans) && ldda < max(1,k)) ) {
        info = -7;
    } else if ( ((trans == MagmaNoTrans) && lddb < max(1,n)) ||
                ((trans != MagmaNoTrans) && lddb < max(1,k)) ) {
        info = -9;
    } else if ( lddc < max(1,n) ) {
        info = -12;
    }

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // Quick return if possible
    if( (n == 0) || ((alpha == 0 || k == 0) && (beta == 1)) ) return;

    // TODO: tune nb?
    magma_int_t nb = 512; 
    if( trans == MagmaNoTrans){
        magmablas_dsyrk_internal(
                uplo, MagmaNoTrans, 
                n, k, nb,
                alpha, dA, ldda, 
                       dB, lddb, 
                cbeta, dC, lddc, 
                1, queue );
        magmablas_dsyrk_internal(
                uplo, MagmaNoTrans, 
                n, k, nb,
                MAGMA_D_CONJ(alpha), dB, lddb, 
                                     dA, ldda, 
                c_one,               dC, lddc, 
                1, queue );    
    }else{
        magmablas_dsyrk_internal(
                uplo, MagmaTrans, 
                n, k, nb,
                alpha, dA, ldda, 
                       dB, lddb, 
                cbeta, dC, lddc, 
                1, queue );
        magmablas_dsyrk_internal(
                uplo, MagmaTrans, 
                n, k, nb,
                MAGMA_D_CONJ(alpha), dB, lddb, 
                                     dA, ldda, 
                c_one,               dC, lddc, 
                1, queue );
    }
}
#endif

/******************************************************************************/
// Symmetric case (all precisions)
extern "C"
void magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    double c_one = MAGMA_D_MAKE(1., 0.);
    
    if ( uplo != MagmaLower && uplo != MagmaUpper) {
        info = -1; 
    #if defined(PRECISION_c) || defined(PRECISION_z)
    } else if ( trans != MagmaNoTrans && trans != MagmaConjTrans) {
    #else
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans) {
    #endif
        info = -2;
    } else if ( n < 0 ) {
        info = -3;
    } else if ( k < 0 ) {
        info = -4;
    } else if ( ((trans == MagmaNoTrans) && ldda < max(1,n)) ||
                ((trans != MagmaNoTrans) && ldda < max(1,k)) ) {
        info = -7;
    } else if ( ((trans == MagmaNoTrans) && lddb < max(1,n)) ||
                ((trans != MagmaNoTrans) && lddb < max(1,k)) ) {
        info = -9;
    } else if ( lddc < max(1,n) ) {
        info = -12;
    }

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    // Quick return if possible
    if( (n == 0) || ((alpha == 0 || k == 0) && (beta == 1)) ) return;

    // TODO: tune nb?
    magma_int_t nb = 512; 
    if( trans == MagmaNoTrans){
        magmablas_dsyrk_internal(
                uplo, MagmaNoTrans, 
                n, k, nb,
                alpha, dA, ldda, 
                       dB, lddb, 
                beta,  dC, lddc, 
                0, queue );
        magmablas_dsyrk_internal(
                uplo, MagmaNoTrans, 
                n, k, nb,
                MAGMA_D_CONJ(alpha), dB, lddb, 
                                     dA, ldda, 
                c_one,               dC, lddc, 
                0, queue );    
    }else{
        magmablas_dsyrk_internal(
                uplo, MagmaTrans, 
                n, k, nb,
                alpha, dA, ldda, 
                       dB, lddb, 
                beta, dC, lddc, 0, queue );
        magmablas_dsyrk_internal(
                uplo, MagmaTrans, 
                n, k, nb,
                MAGMA_D_CONJ(alpha), dB, lddb, 
                                     dA, ldda, 
                c_one,               dC, lddc, 
                0, queue );
    }
}
