/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas/zgemm_batched_smallsq.cu, normal z -> d, Thu Oct  8 23:05:36 2020
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

extern __shared__ double zdata[];
template<int N>
__global__ void
dgemm_batched_smallsq_kernel(
        const magma_trans_t transA, magma_trans_t transB, 
        const double alpha, double const * const * dA_array, int ai, int aj, int ldda, 
                                        double const * const * dB_array, int bi, int bj, int lddb, 
        const double beta,  double**               dC_array, int ci, int cj, int lddc, 
        const int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x;
    
    const int batchid = bx * blockDim.z + tz;
    if(batchid >= batchCount) return;
    
    const double* __restrict__ dA = dA_array[batchid] + aj * ldda + ai;
    const double* __restrict__ dB = dB_array[batchid] + bj * lddb + bi;
          double* __restrict__ dC = dC_array[batchid] + cj * lddc + ci;
    
    double rC = MAGMA_D_ZERO; 
    double rTmp = MAGMA_D_ZERO; 
    
    const int slda = SLDA(N);
    const int sldb = SLDA(N);
    double* sA = (double*)(zdata);
    double* sB = (double*)(zdata + blockDim.z * slda * N);
    
    sA += tz * slda * N;
    sB += tz * sldb * N;
    
    // read A & B 
    if(transA == MagmaNoTrans){
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else{
        sA[tx * slda + ty] = (transA == MagmaTrans) ? dA[ty * ldda + tx] : MAGMA_D_CONJ( dA[ty * ldda + tx] );
    }

    if(transB == MagmaNoTrans){
        sB[ty * sldb + tx] = dB[ty * lddb + tx];
    }
    else{
        sB[tx * sldb + ty] = (transB == MagmaTrans) ? dB[ty * lddb + tx] : MAGMA_D_CONJ( dB[ty * lddb + tx] );
    }
    __syncthreads(); 

    if(beta != MAGMA_D_ZERO){
        rC = beta * dC[ty * lddc + tx];
    }

    // multiply
    rTmp = MAGMA_D_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++){
        rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
    }
    rC += alpha * rTmp;

    // write from rC
    dC[ty * lddc + tx] = rC;
}


extern "C" void 
magmablas_dgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    double alpha,
    double const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    double const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    double beta,
    double **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
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
    
    magma_int_t ntcol  = magma_get_dgemm_batched_ntcol( m );
    magma_int_t shmem  = ( SLDA(m)*m + SLDA(n)*n ) * sizeof(double);
                shmem *= ntcol;

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(m, m, ntcol);

    switch(m){
        case  1: dgemm_batched_smallsq_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  2: dgemm_batched_smallsq_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  3: dgemm_batched_smallsq_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  4: dgemm_batched_smallsq_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  5: dgemm_batched_smallsq_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  6: dgemm_batched_smallsq_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  7: dgemm_batched_smallsq_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  8: dgemm_batched_smallsq_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  9: dgemm_batched_smallsq_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 10: dgemm_batched_smallsq_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 11: dgemm_batched_smallsq_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 12: dgemm_batched_smallsq_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 13: dgemm_batched_smallsq_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 14: dgemm_batched_smallsq_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 15: dgemm_batched_smallsq_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 16: dgemm_batched_smallsq_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 17: dgemm_batched_smallsq_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 18: dgemm_batched_smallsq_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 19: dgemm_batched_smallsq_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 20: dgemm_batched_smallsq_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 21: dgemm_batched_smallsq_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 22: dgemm_batched_smallsq_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 23: dgemm_batched_smallsq_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 24: dgemm_batched_smallsq_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 25: dgemm_batched_smallsq_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 26: dgemm_batched_smallsq_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 27: dgemm_batched_smallsq_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 28: dgemm_batched_smallsq_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 29: dgemm_batched_smallsq_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 30: dgemm_batched_smallsq_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 31: dgemm_batched_smallsq_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 32: dgemm_batched_smallsq_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        default:;
    }
}
