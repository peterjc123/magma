/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Hartwig Anzt

       @generated from sparse/src/zparic_gpu.cpp, normal z -> s, Thu Oct  8 23:05:55 2020
*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_s


/***************************************************************************//**
    Purpose
    -------

    Generates an IC(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the GPU implementation of the ParIC

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
magma_sparic_gpu(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = MAGMA_ERR_NOT_SUPPORTED;
    
#ifdef _OPENMP
    info = 0;

    magma_s_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR},
    dAL={Magma_CSR}, dACOO={Magma_CSR};

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
    
    CHECK(magma_smconvert(hAL, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    CHECK(magma_smtransfer(hAL, &dAL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_smtransfer(hACOO, &dACOO, Magma_CPU, Magma_DEV, queue));
    
    // This is the actual ParIC kernel. 
    // It can be called directly if
    // - the system matrix hALCOO is available in COO format on the GPU 
    // - hAL is the lower triangular in CSR on the GPU
    // The kernel is located in sparse/blas/sparic_kernels.cu
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_sparic_csr(dACOO, dAL, queue));
    }
    

    CHECK(magma_smtransfer(dAL, &precond->L, Magma_DEV, Magma_DEV, queue));
    CHECK(magma_s_cucsrtranspose(precond->L, &precond->U, queue));
    CHECK(magma_smtransfer(precond->L, &precond->M, Magma_DEV, Magma_DEV, queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_scumicgeneratesolverinfo(precond, queue));
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
    magma_smfree(&hAT, queue);
    magma_smfree(&hA, queue);
    magma_smfree(&hAL, queue);
    magma_smfree(&dAL, queue);
    magma_smfree(&hAU, queue);
    magma_smfree(&hAUT, queue);
    magma_smfree(&hAtmp, queue);
    magma_smfree(&hACOO, queue);
    magma_smfree(&dACOO, queue);

#endif
    return info;
}
