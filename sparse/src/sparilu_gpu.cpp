/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @author Hartwig Anzt

       @generated from sparse/src/zparilu_gpu.cpp, normal z -> s, Sun Mar 29 20:48:36 2020
*/

#include "magmasparse_internal.h"

#define PRECISION_s


/***************************************************************************//**
    Purpose
    -------

    Generates an ILU(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the GPU implementation of the ParILU

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                input RHS b

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
*******************************************************************************/
extern "C"
magma_int_t
magma_sparilu_gpu(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = 0;

    magma_s_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR},
    dAL={Magma_CSR}, dAU={Magma_CSR}, dAUT={Magma_CSR}, dACOO={Magma_CSR};

    // copy original matrix as COO to device
    if (A.memory_location != Magma_CPU || A.storage_type != Magma_CSR) {
        CHECK(magma_smtransfer(A, &hAT, A.memory_location, Magma_CPU, queue));
        CHECK(magma_smconvert(hAT, &hA, hAT.storage_type, Magma_CSR, queue));
        magma_smfree(&hAT, queue);
    } else {
        CHECK(magma_smtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    }

    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_ssymbilu(&hA, precond->levels, &hAL, &hAUT,  queue));
        magma_smfree(&hAL, queue);
        magma_smfree(&hAUT, queue);
    }
    CHECK(magma_smconvert(hA, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    
    //get L
    magma_smatrix_tril(hA, &hAL, queue);
    // we need 1 on the main diagonal of L
    #pragma omp parallel for
    for (int k=0; k < hAL.num_rows; k++) {
        hAL.val[hAL.row[k+1]-1] = MAGMA_S_ONE;
    }
    
    // get U
    magma_smtranspose(hA, &hAT, queue);
    magma_smatrix_tril(hAT, &hAU, queue);
    magma_smfree(&hAT, queue);
    
    CHECK(magma_smtransfer(hAL, &dAL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_smtransfer(hAU, &dAU, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_smtransfer(hACOO, &dACOO, Magma_CPU, Magma_DEV, queue));
    
    // This is the actual ParILU kernel. 
    // It can be called directly if
    // - the system matrix hACOO is available in COO format on the CPU 
    // - hAL is the lower triangular in CSR on the CPU
    // - hAU is the upper triangular in CSC on the CPU (U transpose in CSR)
    // The kernel is located in sparse/blas/sparilu_kernels.cu
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_sparilu_csr(dACOO, dAL, dAU, queue));
    }
    CHECK(magma_s_cucsrtranspose(dAU, &dAUT, queue));

    CHECK(magma_smtransfer(dAL, &precond->L, Magma_DEV, Magma_DEV, queue));
    CHECK(magma_smtransfer(dAUT, &precond->U, Magma_DEV, Magma_DEV, queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_scumilugeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_sjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_svinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_S_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_sjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_svinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_S_ZERO, queue));
    }
    
cleanup:
    magma_smfree(&dAL, queue);
    magma_smfree(&dAU, queue);
    magma_smfree(&dAUT, queue);
    magma_smfree(&dACOO, queue);
    magma_smfree(&hAT, queue);
    magma_smfree(&hA, queue);
    magma_smfree(&hAL, queue);
    magma_smfree(&hAU, queue);
    magma_smfree(&hAUT, queue);
    magma_smfree(&hAtmp, queue);
    magma_smfree(&hACOO, queue);

    
    return info;
}

