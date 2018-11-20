/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @author Hartwig Anzt

       @generated from sparse/src/zparilut_cpu.cpp, normal z -> d, Mon Jun 25 18:24:31 2018
*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_d


/***************************************************************************//**
    Purpose
    -------

    Generates an incomplete threshold LU preconditioner via the ParILUT 
    algorithm. The strategy is to interleave a parallel fixed-point 
    iteration that approximates an incomplete factorization for a given nonzero 
    pattern with a procedure that adaptively changes the pattern. 
    Much of this algorithm has fine-grained parallelism, and can efficiently 
    exploit the compute power of shared memory architectures.

    This is the routine used in the publication by Anzt, Chow, Dongarra:
    ''ParILUT - A new parallel threshold ILU factorization''
    submitted to SIAM SISC in 2017.
    
    This version uses the default setting which adds all candidates to the
    sparsity pattern.

    This function requires OpenMP, and is only available if OpenMP is activated.
    
    The parameter list is:
    
    precond.sweeps : number of ParILUT steps
    precond.atol   : absolute fill ratio (1.0 keeps nnz count constant)


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
magma_dparilut_cpu(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
#ifdef _OPENMP

    real_Double_t start, end;
    real_Double_t t_rm=0.0, t_add=0.0, t_res=0.0, t_sweep1=0.0, t_sweep2=0.0, 
        t_cand=0.0, t_sort=0.0, t_transpose1=0.0, t_transpose2=0.0, t_selectrm=0.0,
        t_nrm=0.0, t_total = 0.0, accum=0.0;
                    
    double sum, sumL, sumU;

    magma_d_matrix hA={Magma_CSR}, hAT={Magma_CSR}, hL={Magma_CSR}, 
        hU={Magma_CSR}, oneL={Magma_CSR}, oneU={Magma_CSR},
        L={Magma_CSR}, U={Magma_CSR}, L_new={Magma_CSR}, U_new={Magma_CSR}, 
        UT={Magma_CSR}, L0={Magma_CSR}, U0={Magma_CSR};
    magma_int_t num_rmL, num_rmU;
    double thrsL = 0.0;
    double thrsU = 0.0;

    magma_int_t num_threads = 1, timing = 1; // print timing
    magma_int_t L0nnz, U0nnz;

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    
    CHECK(magma_dmtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    
    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_dsymbilu(&hA, precond->levels, &hL, &hU , queue));
        magma_dmfree(&hU, queue);
        magma_dmfree(&hL, queue);
    }
    CHECK(magma_dmatrix_tril(hA, &L0, queue));
    CHECK(magma_dmatrix_triu(hA, &U0, queue));
    magma_dmfree(&hU, queue);
    magma_dmfree(&hL, queue);
    CHECK(magma_dmatrix_tril(hA, &L, queue));
    CHECK(magma_dmtranspose(hA, &hAT, queue));
    CHECK(magma_dmatrix_tril(hAT, &U, queue));
    CHECK(magma_dmatrix_addrowindex(&L, queue)); 
    CHECK(magma_dmatrix_addrowindex(&U, queue)); 
    L0nnz=L.nnz;
    U0nnz=U.nnz;
    oneL.memory_location = Magma_CPU;
    oneU.memory_location = Magma_CPU;
        
    if (timing == 1) {
        printf("ilut_fill_ratio = %.6f;\n\n", precond->atol);  
        printf("performance_%d = [\n%%iter      L.nnz      U.nnz    ILU-Norm    transp    candidat  resid     sort    transcand    add      sweep1   selectrm    remove    sweep2     total       accum\n", 
            (int) num_threads);
    }

    //##########################################################################

    for (magma_int_t iters =0; iters<precond->sweeps; iters++) {
        t_rm=0.0; t_add=0.0; t_res=0.0; t_sweep1=0.0; t_sweep2=0.0; t_cand=0.0;
        t_transpose1=0.0; t_transpose2=0.0;  t_selectrm=0.0; t_sort = 0;
        t_sort=0.0; t_nrm=0.0; t_total = 0.0;
     
        // step 1: transpose U
        start = magma_sync_wtime(queue);
        magma_dmfree(&UT, queue);
        CHECK(magma_dcsrcoo_transpose(U, &UT, queue));
        end = magma_sync_wtime(queue); t_transpose1+=end-start;
        
        
        // step 2: find candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_dparilut_candidates(L0, U0, L, UT, &hL, &hU, queue));
        end = magma_sync_wtime(queue); t_cand=+end-start;
        
        
        // step 3: compute residuals (optional when adding all candidates)
        start = magma_sync_wtime(queue);
        CHECK(magma_dparilut_residuals(hA, L, U, &hL, queue));
        CHECK(magma_dparilut_residuals(hA, L, U, &hU, queue));
        end = magma_sync_wtime(queue); t_res=+end-start;
        start = magma_sync_wtime(queue);
        CHECK(magma_dmatrix_abssum(hL, &sumL, queue));
        CHECK(magma_dmatrix_abssum(hU, &sumU, queue));
        sum = sumL + sumU;
        end = magma_sync_wtime(queue); t_nrm+=end-start;
        CHECK(magma_dmatrix_swap(&hL, &oneL, queue));
        magma_dmfree(&hL, queue);
        
        
        // step 4: sort candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_dcsr_sort(&hL, queue));
        CHECK(magma_dcsr_sort(&hU, queue));
        end = magma_sync_wtime(queue); t_sort+=end-start;
        
        
        // step 5: transpose candidates
        start = magma_sync_wtime(queue);
        magma_dcsrcoo_transpose(hU, &oneU, queue);
        end = magma_sync_wtime(queue); t_transpose2+=end-start;
        
        
        // step 6: add candidates
        start = magma_sync_wtime(queue);
        CHECK(magma_dmatrix_cup(L, oneL, &L_new, queue));   
        CHECK(magma_dmatrix_cup(U, oneU, &U_new, queue));
        end = magma_sync_wtime(queue); t_add=+end-start;
        magma_dmfree(&oneL, queue);
        magma_dmfree(&oneU, queue);
       
        
        // step 7: sweep
        start = magma_sync_wtime(queue);
        CHECK(magma_dparilut_sweep_sync(&hA, &L_new, &U_new, queue));
        end = magma_sync_wtime(queue); t_sweep1+=end-start;
        
        
        // step 8: select threshold to remove elements
        start = magma_sync_wtime(queue);
        num_rmL = max((L_new.nnz-L0nnz*(1+(precond->atol-1.)
            *(iters+1)/precond->sweeps)), 0);
        num_rmU = max((U_new.nnz-U0nnz*(1+(precond->atol-1.)
            *(iters+1)/precond->sweeps)), 0);
        // pre-select: ignore the diagonal entries
        CHECK(magma_dparilut_preselect(0, &L_new, &oneL, queue));
        CHECK(magma_dparilut_preselect(0, &U_new, &oneU, queue));
        if (num_rmL>0) {
            CHECK(magma_dparilut_set_thrs_randomselect_approx(num_rmL, 
                &oneL, 0, &thrsL, queue));
        } else {
            thrsL = 0.0;
        }
        if (num_rmU>0) {
            CHECK(magma_dparilut_set_thrs_randomselect_approx(num_rmU, 
                &oneU, 0, &thrsU, queue));
        } else {
            thrsU = 0.0;
        }
        magma_dmfree(&oneL, queue);
        magma_dmfree(&oneU, queue);
        end = magma_sync_wtime(queue); t_selectrm=end-start;

        
        // step 9: remove elements
        start = magma_sync_wtime(queue);
        CHECK(magma_dparilut_thrsrm(1, &L_new, &thrsL, queue));
        CHECK(magma_dparilut_thrsrm(1, &U_new, &thrsU, queue));
        CHECK(magma_dmatrix_swap(&L_new, &L, queue));
        CHECK(magma_dmatrix_swap(&U_new, &U, queue));
        magma_dmfree(&L_new, queue);
        magma_dmfree(&U_new, queue);
        end = magma_sync_wtime(queue); t_rm=end-start;
        
        
        // step 10: sweep
        start = magma_sync_wtime(queue);
        CHECK(magma_dparilut_sweep_sync(&hA, &L, &U, queue));
        end = magma_sync_wtime(queue); t_sweep2+=end-start;
        
        if (timing == 1) {
            t_total = t_transpose1+ t_cand+ t_res+ t_sort+ t_transpose2+ t_add+ t_sweep1+ t_selectrm+ t_rm+ t_sweep2;
            accum = accum + t_total;
            printf("%5lld %10lld %10lld  %.4e   %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e      %.2e\n",
                (long long) iters, (long long) L.nnz, (long long) U.nnz, 
                (double) sum, 
                t_transpose1, t_cand, t_res, t_sort, t_transpose2, t_add, t_sweep1, t_selectrm, t_rm, t_sweep2, t_total, accum);
            fflush(stdout);
        }
    }

    if (timing == 1) {
        printf("]; \n");
        fflush(stdout);
    }
    //##########################################################################

    // for CUSPARSE
    CHECK(magma_dmtransfer(L, &precond->L, Magma_CPU, Magma_DEV , queue));
    CHECK(magma_dcsrcoo_transpose(U, &UT, queue));
    //magma_dmtranspose(U, &UT, queue);
    CHECK(magma_dmtransfer(UT, &precond->U, Magma_CPU, Magma_DEV , queue));
    
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
    magma_dmfree(&hA, queue);
    magma_dmfree(&hAT, queue);
    magma_dmfree(&L, queue);
    magma_dmfree(&U, queue);
    magma_dmfree(&UT, queue);
    magma_dmfree(&L0, queue);
    magma_dmfree(&U0, queue);
    magma_dmfree(&L_new, queue);
    magma_dmfree(&U_new, queue);
    magma_dmfree(&hL, queue);
    magma_dmfree(&hU, queue);
#endif
    return info;
}
