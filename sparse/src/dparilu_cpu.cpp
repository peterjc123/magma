/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @author Hartwig Anzt

       @generated from sparse/src/zparilu_cpu.cpp, normal z -> d, Fri Aug  2 17:10:14 2019
*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_d


/***************************************************************************//**
    Purpose
    -------

    Generates an ILU(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the CPU implementation of the ParILU

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
*******************************************************************************/

extern "C"
magma_int_t
magma_dparilu_cpu(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = MAGMA_ERR_NOT_SUPPORTED;
    
#ifdef _OPENMP
    info = 0;

    magma_d_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR};

    // copy original matrix as COO to device
    if (A.memory_location != Magma_CPU || A.storage_type != Magma_CSR) {
        CHECK(magma_dmtransfer(A, &hAT, A.memory_location, Magma_CPU, queue));
        CHECK(magma_dmconvert(hAT, &hA, hAT.storage_type, Magma_CSR, queue));
        magma_dmfree(&hAT, queue);
    } else {
        CHECK(magma_dmtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    }

    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_dsymbilu(&hA, precond->levels, &hAL, &hAUT,  queue));
        magma_dmfree(&hAL, queue);
        magma_dmfree(&hAUT, queue);
    }
    CHECK(magma_dmconvert(hA, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    
    //get L
    magma_dmatrix_tril(hA, &hAL, queue);
    // we need 1 on the main diagonal of L
    #pragma omp parallel for
    for (int k=0; k < hAL.num_rows; k++) {
        hAL.val[hAL.row[k+1]-1] = MAGMA_D_ONE;
    }
    
    // get U
    magma_dmtranspose(hA, &hAT, queue);
    magma_dmatrix_tril(hAT, &hAU, queue);
    magma_dmfree(&hAT, queue);
    
    // This is the actual ParILU kernel. 
    // It can be called directly if
    // - the system matrix hACOO is available in COO format on the CPU 
    // - hAL is the lower triangular in CSR on the CPU
    // - hAU is the upper triangular in CSC on the CPU (U transpose in CSR)
    // The kernel is located in sparse/control/magma_dparilu_kernels.cpp
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_dparilu_sweep(hACOO, &hAL, &hAU, queue));
    }
    CHECK(magma_d_cucsrtranspose(hAU, &hAUT, queue));

    CHECK(magma_dmtransfer(hAL, &precond->L, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_dmtransfer(hAUT, &precond->U, Magma_CPU, Magma_DEV, queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_dcumilugeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_djacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_dvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_D_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_djacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_dvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_D_ZERO, queue));
    }

cleanup:
    magma_dmfree(&hAT, queue);
    magma_dmfree(&hA, queue);
    magma_dmfree(&hAL, queue);
    magma_dmfree(&hAU, queue);
    magma_dmfree(&hAUT, queue);
    magma_dmfree(&hAtmp, queue);
    magma_dmfree(&hACOO, queue);

#endif
    return info;
}
