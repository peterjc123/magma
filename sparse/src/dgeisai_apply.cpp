/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Hartwig Anzt

       @generated from sparse/src/zgeisai_apply.cpp, normal z -> d, Mon Jun 25 18:24:31 2018
*/
#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#define PRECISION_d


/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_l(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->LD, b, MAGMA_D_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->LD, b, MAGMA_D_ZERO, precond->d, queue ); // d=L_d^(-1)b
        magma_d_spmv( MAGMA_D_ONE, precond->LD, b, MAGMA_D_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_d_spmv( MAGMA_D_ONE, precond->L, *x, MAGMA_D_ZERO, precond->work1, queue ); // work1 = L * x
            magma_d_spmv( MAGMA_D_ONE, precond->LD, precond->work1, MAGMA_D_ZERO, precond->work2, queue ); // work2 = L_d^(-1)work1
            magma_daxpy( b.num_rows*b.num_cols, -MAGMA_D_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // x = - work2
            magma_daxpy( b.num_rows*b.num_cols, MAGMA_D_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // x = d + x = L_d^(-1)b - work2 = L_d^(-1)b - L_d^(-1) * L * x
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_r(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->UD, b, MAGMA_D_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->UD, b, MAGMA_D_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_d_spmv( MAGMA_D_ONE, precond->UD, b, MAGMA_D_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_d_spmv( MAGMA_D_ONE, precond->U, *x, MAGMA_D_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_d_spmv( MAGMA_D_ONE, precond->UD, precond->work1, MAGMA_D_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_daxpy( b.num_rows*b.num_cols, -MAGMA_D_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_daxpy( b.num_rows*b.num_cols, MAGMA_D_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_l_t(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->LDT, b, MAGMA_D_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->LDT, b, MAGMA_D_ZERO, precond->d, queue ); // d=M_L*b
        magma_d_spmv( MAGMA_D_ONE, precond->LDT, b, MAGMA_D_ZERO, *x, queue ); // x = M_L*b
        for( int z=0; z<precond->maxiter; z++ ){
            magma_d_spmv( MAGMA_D_ONE, precond->LT, *x, MAGMA_D_ZERO, precond->work1, queue ); // work1=L*M_L*b
            magma_d_spmv( MAGMA_D_ONE, precond->LDT, precond->work1, MAGMA_D_ZERO, precond->work2, queue ); // work2 = M_L*L*M_L*b
            magma_daxpy( b.num_rows*b.num_cols, -MAGMA_D_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // x = M_L*x -M_L*L*M_L*b
            magma_daxpy( b.num_rows*b.num_cols, MAGMA_D_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // x = M_L*x + M_L*b - M_L *L*M_L *b 
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_r_t(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->UDT, b, MAGMA_D_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_d_spmv( MAGMA_D_ONE, precond->UDT, b, MAGMA_D_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_d_spmv( MAGMA_D_ONE, precond->UDT, b, MAGMA_D_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_d_spmv( MAGMA_D_ONE, precond->UT, *x, MAGMA_D_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_d_spmv( MAGMA_D_ONE, precond->UDT, precond->work1, MAGMA_D_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_daxpy( b.num_rows*b.num_cols, -MAGMA_D_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_daxpy( b.num_rows*b.num_cols, MAGMA_D_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}
