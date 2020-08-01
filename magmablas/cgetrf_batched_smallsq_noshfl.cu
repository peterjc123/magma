/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas/zgetrf_batched_smallsq_noshfl.cu, normal z -> c, Sun Mar 29 20:48:31 2020
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.cuh"
#include "shuffle.cuh"
#include "batched_kernel_param.h"

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
extern __shared__ magmaFloatComplex zdata[];
template<int N, int NPOW2>
__global__ void
cgetrf_batched_smallsq_noshfl_kernel( magmaFloatComplex** dA_array, int ldda, 
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;
    
    magmaFloatComplex* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];
    
    magmaFloatComplex rA[N]  = {MAGMA_C_ZERO};
    magmaFloatComplex reg    = MAGMA_C_ZERO; 
    magmaFloatComplex update = MAGMA_C_ZERO;
    
    int max_id, rowid = tx;
    int linfo = 0;
    float rx_abs_max = MAGMA_D_ZERO;
    
    magmaFloatComplex *sx = (magmaFloatComplex*)(zdata);
    float* dsx = (float*)(sx + blockDim.y * NPOW2);
    int* sipiv = (int*)(dsx + blockDim.y * NPOW2);
    sx    += ty * NPOW2;
    dsx   += ty * NPOW2;
    sipiv += ty * NPOW2;

    // read 
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }
        
    #pragma unroll
    for(int i = 0; i < N; i++){
        // icamax and find pivot
        dsx[ rowid ] = fabs(MAGMA_C_REAL( rA[i] )) + fabs(MAGMA_C_IMAG( rA[i] ));
        magmablas_syncwarp();
        rx_abs_max = dsx[i];
        max_id = i; 
        #pragma unroll
        for(int j = i+1; j < N; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_C_ZERO : MAGMA_C_ONE;
        
        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = i; j < N; j++){
                sx[j] = update * rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id; 
        }
        magmablas_syncwarp();
        
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_C_ONE : MAGMA_C_DIV(MAGMA_C_ONE, sx[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        magmablas_syncwarp();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }
    // write
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);    // fortran indexing
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    cgetrf_batched_smallsq_noshfl computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges. 
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_cgetrf_batched_smallsq_noshfl( 
    magma_int_t n, 
    magmaFloatComplex** dA_array, magma_int_t ldda, 
    magma_int_t** ipiv_array, magma_int_t* info_array, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m = n;
    
    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }
    
    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    
    if( m == 0) return 0;
    
    const magma_int_t ntcol = magma_get_cgetrf_batched_ntcol(m, n);
    magma_int_t shmem  = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(float);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(magmaFloatComplex);
    dim3 threads(magma_ceilpow2(m), ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    switch(m){
        case  1: cgetrf_batched_smallsq_noshfl_kernel< 1, magma_ceilpow2( 1)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  2: cgetrf_batched_smallsq_noshfl_kernel< 2, magma_ceilpow2( 2)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  3: cgetrf_batched_smallsq_noshfl_kernel< 3, magma_ceilpow2( 3)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  4: cgetrf_batched_smallsq_noshfl_kernel< 4, magma_ceilpow2( 4)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  5: cgetrf_batched_smallsq_noshfl_kernel< 5, magma_ceilpow2( 5)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  6: cgetrf_batched_smallsq_noshfl_kernel< 6, magma_ceilpow2( 6)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  7: cgetrf_batched_smallsq_noshfl_kernel< 7, magma_ceilpow2( 7)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  8: cgetrf_batched_smallsq_noshfl_kernel< 8, magma_ceilpow2( 8)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  9: cgetrf_batched_smallsq_noshfl_kernel< 9, magma_ceilpow2( 9)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 10: cgetrf_batched_smallsq_noshfl_kernel<10, magma_ceilpow2(10)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 11: cgetrf_batched_smallsq_noshfl_kernel<11, magma_ceilpow2(11)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 12: cgetrf_batched_smallsq_noshfl_kernel<12, magma_ceilpow2(12)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 13: cgetrf_batched_smallsq_noshfl_kernel<13, magma_ceilpow2(13)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 14: cgetrf_batched_smallsq_noshfl_kernel<14, magma_ceilpow2(14)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 15: cgetrf_batched_smallsq_noshfl_kernel<15, magma_ceilpow2(15)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 16: cgetrf_batched_smallsq_noshfl_kernel<16, magma_ceilpow2(16)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 17: cgetrf_batched_smallsq_noshfl_kernel<17, magma_ceilpow2(17)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 18: cgetrf_batched_smallsq_noshfl_kernel<18, magma_ceilpow2(18)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 19: cgetrf_batched_smallsq_noshfl_kernel<19, magma_ceilpow2(19)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 20: cgetrf_batched_smallsq_noshfl_kernel<20, magma_ceilpow2(20)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 21: cgetrf_batched_smallsq_noshfl_kernel<21, magma_ceilpow2(21)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 22: cgetrf_batched_smallsq_noshfl_kernel<22, magma_ceilpow2(22)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 23: cgetrf_batched_smallsq_noshfl_kernel<23, magma_ceilpow2(23)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 24: cgetrf_batched_smallsq_noshfl_kernel<24, magma_ceilpow2(24)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 25: cgetrf_batched_smallsq_noshfl_kernel<25, magma_ceilpow2(25)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 26: cgetrf_batched_smallsq_noshfl_kernel<26, magma_ceilpow2(26)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 27: cgetrf_batched_smallsq_noshfl_kernel<27, magma_ceilpow2(27)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 28: cgetrf_batched_smallsq_noshfl_kernel<28, magma_ceilpow2(28)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 29: cgetrf_batched_smallsq_noshfl_kernel<29, magma_ceilpow2(29)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 30: cgetrf_batched_smallsq_noshfl_kernel<30, magma_ceilpow2(30)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 31: cgetrf_batched_smallsq_noshfl_kernel<31, magma_ceilpow2(31)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 32: cgetrf_batched_smallsq_noshfl_kernel<32, magma_ceilpow2(32)><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}
