/*
    -- MAGMA (version 2.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2019

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas/zgemm_batched_smallsq.cu, normal z -> c, Wed Jan  2 14:18:51 2019
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

extern __shared__ magmaFloatComplex zdata[];
template<int N>
__global__ void
cgemm_batched_smallsq_kernel(
        const magma_trans_t transA, magma_trans_t transB, 
        const magmaFloatComplex alpha, magmaFloatComplex const * const * dA_array, int ai, int aj, int ldda, 
                                        magmaFloatComplex const * const * dB_array, int bi, int bj, int lddb, 
        const magmaFloatComplex beta,  magmaFloatComplex**               dC_array, int ci, int cj, int lddc, 
        const int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x;
    
    const int batchid = bx * blockDim.z + tz;
    if(batchid >= batchCount) return;
    
    const magmaFloatComplex* __restrict__ dA = dA_array[batchid] + aj * ldda + ai;
    const magmaFloatComplex* __restrict__ dB = dB_array[batchid] + bj * lddb + bi;
          magmaFloatComplex* __restrict__ dC = dC_array[batchid] + cj * lddc + ci;
    
    magmaFloatComplex rC = MAGMA_C_ZERO; 
    magmaFloatComplex rTmp = MAGMA_C_ZERO; 
    
    const int slda = SLDA(N);
    const int sldb = SLDA(N);
    magmaFloatComplex* sA = (magmaFloatComplex*)(zdata);
    magmaFloatComplex* sB = (magmaFloatComplex*)(zdata + blockDim.z * slda * N);
    
    sA += tz * slda * N;
    sB += tz * sldb * N;
    
    // read A & B 
    if(transA == MagmaNoTrans){
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else{
        sA[tx * slda + ty] = (transA == MagmaTrans) ? dA[ty * ldda + tx] : MAGMA_C_CONJ( dA[ty * ldda + tx] );
    }

    if(transB == MagmaNoTrans){
        sB[ty * sldb + tx] = dB[ty * lddb + tx];
    }
    else{
        sB[tx * sldb + ty] = (transB == MagmaTrans) ? dB[ty * lddb + tx] : MAGMA_C_CONJ( dB[ty * lddb + tx] );
    }
    __syncthreads(); 

    if(beta != MAGMA_C_ZERO){
        rC = beta * dC[ty * lddc + tx];
    }

    // multiply
    rTmp = MAGMA_C_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++){
        rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
    }
    rC += alpha * rTmp;

    // write from rC
    dC[ty * lddc + tx] = rC;
}


extern "C" void 
magmablas_cgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    magmaFloatComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if( !(m == n  && n == k) ){
        printf("Only square sizes are supported\n");
        info = -1;
    }

    if( m > 32){
        printf("Only square sizes of up to 32 are supported\n");
        info = -1;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
    }

    if ( m <= 0 || n <= 0 || k <= 0 ) return;
    
    magma_int_t ntcol  = magma_get_cgemm_batched_ntcol( m );
    magma_int_t shmem  = ( SLDA(m)*m + SLDA(n)*n ) * sizeof(magmaFloatComplex);
                shmem *= ntcol;

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(m, m, ntcol);

    switch(m){
        case  1: cgemm_batched_smallsq_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  2: cgemm_batched_smallsq_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  3: cgemm_batched_smallsq_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  4: cgemm_batched_smallsq_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  5: cgemm_batched_smallsq_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  6: cgemm_batched_smallsq_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  7: cgemm_batched_smallsq_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  8: cgemm_batched_smallsq_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  9: cgemm_batched_smallsq_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 10: cgemm_batched_smallsq_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 11: cgemm_batched_smallsq_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 12: cgemm_batched_smallsq_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 13: cgemm_batched_smallsq_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 14: cgemm_batched_smallsq_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 15: cgemm_batched_smallsq_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 16: cgemm_batched_smallsq_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 17: cgemm_batched_smallsq_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 18: cgemm_batched_smallsq_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 19: cgemm_batched_smallsq_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 20: cgemm_batched_smallsq_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 21: cgemm_batched_smallsq_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 22: cgemm_batched_smallsq_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 23: cgemm_batched_smallsq_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 24: cgemm_batched_smallsq_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 25: cgemm_batched_smallsq_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 26: cgemm_batched_smallsq_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 27: cgemm_batched_smallsq_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 28: cgemm_batched_smallsq_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 29: cgemm_batched_smallsq_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 30: cgemm_batched_smallsq_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 31: cgemm_batched_smallsq_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 32: cgemm_batched_smallsq_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        default:;
    }
}
