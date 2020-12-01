/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @generated from magmablas/zherk.cpp, normal z -> c, Thu Oct  8 23:05:32 2020
       @author Azzam Haidar 
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#define PRECISION_c

/******************************************************************************/
extern "C"
void magmablas_cherk_internal(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloatComplex_ptr dB, magma_int_t lddb, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc, 
    magma_int_t conjugate, magma_queue_t queue)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    
    magma_trans_t transA;
    magma_trans_t transB;
    magmaFloatComplex_ptr dAi, dBi;
    
    if (trans == MagmaNoTrans) {
        transA = MagmaNoTrans;
        transB = ( conjugate == 0 ) ? MagmaTrans : Magma_ConjTrans;
    } else {
        transA = ( conjugate == 0 ) ? MagmaTrans : Magma_ConjTrans;
        transB = MagmaNoTrans;
    }

    magma_int_t ib;
    for( magma_int_t i = 0; i < n; i += ib ) {
        ib     = min( nb, n-i );
        if(uplo == MagmaLower){
            dAi = (trans == MagmaNoTrans) ? dA(i,0) : dA(0,i);
            dBi = (trans == MagmaNoTrans) ? dB(i,0) : dB(0,i);
        }
        else{
            dAi = (trans == MagmaNoTrans) ? dA(0,0) : dA(0,0);
            dBi = (trans == MagmaNoTrans) ? dB(i,0) : dB(0,i);
        }

        magmaFloatComplex_ptr dCi = (uplo  == MagmaLower  ) ? dC(i,i) : dC(0,i);
        magma_int_t im = (uplo == MagmaLower) ? n-i : min(i+nb, n);
        magma_int_t in = ib;
        magma_cgemm( transA, transB, im, in, k,
                     alpha, dAi, ldda,
                            dBi, lddb,
                     beta,  dCi, lddc, queue);
    }
}

/******************************************************************************/
#if defined(PRECISION_c) || defined(PRECISION_z)
extern "C"
void magmablas_cherk(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k, 
    float alpha,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    float beta,
    magmaFloatComplex_ptr dC, magma_int_t lddc, 
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( n <= 0 || k <= 0 )
        return;

    // TODO: tune nb?
    magma_int_t nb = 512; 
    magmablas_cherk_internal(
        uplo, trans, 
        n, k, nb,
        MAGMA_C_MAKE(alpha, 0.), dA, ldda, dA, ldda, 
        MAGMA_C_MAKE(beta, 0.),  dC, lddc, 1, queue);
}
#endif


/******************************************************************************/
extern "C"
void magmablas_csyrk(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dC, magma_int_t lddc, 
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    #if defined(PRECISION_c) || defined(PRECISION_z)
    else if ( trans != MagmaNoTrans && trans != MagmaTrans )
    #else
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
    #endif
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( n <= 0 || k <= 0 )
        return;

    // TODO: tune nb?
    magma_int_t nb = 512; 
    magmablas_cherk_internal(
        uplo, trans, 
        n, k, nb,
        alpha, dA, ldda, dA, ldda, 
        beta,  dC, lddc, 0, queue);
}
