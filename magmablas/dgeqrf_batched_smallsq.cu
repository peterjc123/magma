/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas/zgeqrf_batched_smallsq.cu, normal z -> d, Sun Mar 29 20:48:31 2020
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.cuh"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? (N+2) : (N+1) )
extern __shared__ double zdata[];
template<int N>
__global__ void
dgeqrf_batched_sq1d_reg_kernel( 
    double **dA_array, magma_int_t ldda,
    double **dtau_array, magma_int_t *info_array, 
    magma_int_t batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;
    if(tx >= N) return;
    
    const int slda  = SLDA(N);
    double* dA   = dA_array[batchid];
    double* dtau = dtau_array[batchid];
    magma_int_t* info = &info_array[batchid];
    // shared memory pointers
    double* sA = (double*)(zdata + ty * slda * N);
    double* sdw = (double*)(zdata + blockDim.y * slda * N);
    sdw += ty * N;

    double rA[N] = {MAGMA_D_ZERO};
    double alpha, tau, tmp, zsum, scale = MAGMA_D_ZERO;
    double sum = MAGMA_D_ZERO, norm = MAGMA_D_ZERO, beta; 

    if( tx == 0 ){
        (*info) = 0;
    }

    // init tau
    dtau[tx] = MAGMA_D_ZERO;
    // read 
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }
    
    #pragma unroll
    for(int i = 0; i < N-1; i++){
        sA[ i * slda + tx] = rA[i];
        sdw[tx] = ( MAGMA_D_REAL(rA[i]) * MAGMA_D_REAL(rA[i]) + MAGMA_D_IMAG(rA[i]) * MAGMA_D_IMAG(rA[i]) );
        magmablas_syncwarp();
        alpha = sA[i * slda + i];
        sum = MAGMA_D_ZERO; 
        #pragma unroll
        for(int j = i; j < N; j++){
            sum += sdw[j];
        }
        norm = sqrt(sum);
        beta = -copysign(norm, real(alpha));
        scale = MAGMA_D_DIV( MAGMA_D_ONE,  alpha - MAGMA_D_MAKE(beta, 0));
        tau = MAGMA_D_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );

        if(tx == i){
            dtau[i] = tau;
        }
        
        tmp = (tx == i)? MAGMA_D_MAKE(beta, MAGMA_D_ZERO) : rA[i] * scale;
        
        if(tx >= i){
            rA[i] = tmp;
        }
        
        dA[ i * ldda + tx ] = rA[i];
        rA[i] = (tx == i) ? MAGMA_D_ONE  : rA[i]; 
        rA[i] = (tx < i ) ? MAGMA_D_ZERO : rA[i];
        tmp = MAGMA_D_CONJ( rA[i] ) * MAGMA_D_CONJ( tau );
        
        magmablas_syncwarp();
        #pragma unroll
        for(int j = i+1; j < N; j++){
            sA[j * slda + tx] = rA[j] * tmp;
        }
        magmablas_syncwarp();

        zsum = MAGMA_D_ZERO;
        #pragma unroll
        for(int j = i; j < N; j++){
            zsum += sA[tx * slda + j];
        }
        sA[tx * slda + N] = zsum;
        magmablas_syncwarp();
        
        #pragma unroll
        for(int j = i+1; j < N; j++){
            rA[j] -= rA[i] * sA[j * slda + N];
        }
        magmablas_syncwarp();
    }
    // write the last column
    dA[ (N-1) * ldda + tx ] = rA[N-1];
}

/***************************************************************************//**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.
    
    This is a batched version of the routine, and works only for small 
    square matrices of size up to 32.
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit, the elements on and above the diagonal of the array
             contain the min(M,N)-by-N upper trapezoidal matrix R (R is
             upper triangular if m >= n); the elements below the diagonal,
             with the array TAU, represent the orthogonal matrix Q as a
             product of min(m,n) elementary reflectors (see Further
             Details).

    @param[in]
    ldda     INTEGER
             The leading dimension of the array dA.  LDDA >= max(1,M).
             To benefit from coalescent memory accesses LDDA must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_dgeqrf_batched_smallsq( 
    magma_int_t n,  
    double** dA_array, magma_int_t ldda, 
    double **dtau_array, magma_int_t* info_array, 
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
    
    if( m == 0 || n == 0) return 0;
    
    const magma_int_t ntcol = magma_get_dgeqrf_batched_ntcol(m, n);
    
    magma_int_t shmem = ( SLDA(m) * m * sizeof(double) );
    shmem            += ( m * sizeof(double) );
    shmem            *= ntcol;
    magma_int_t nth = magma_ceilpow2(m);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads(nth, ntcol, 1);
    
    switch(m){
        
        case  1: dgeqrf_batched_sq1d_reg_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  2: dgeqrf_batched_sq1d_reg_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  3: dgeqrf_batched_sq1d_reg_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  4: dgeqrf_batched_sq1d_reg_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  5: dgeqrf_batched_sq1d_reg_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  6: dgeqrf_batched_sq1d_reg_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  7: dgeqrf_batched_sq1d_reg_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  8: dgeqrf_batched_sq1d_reg_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case  9: dgeqrf_batched_sq1d_reg_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 10: dgeqrf_batched_sq1d_reg_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 11: dgeqrf_batched_sq1d_reg_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 12: dgeqrf_batched_sq1d_reg_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 13: dgeqrf_batched_sq1d_reg_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 14: dgeqrf_batched_sq1d_reg_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 15: dgeqrf_batched_sq1d_reg_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 16: dgeqrf_batched_sq1d_reg_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 17: dgeqrf_batched_sq1d_reg_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 18: dgeqrf_batched_sq1d_reg_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 19: dgeqrf_batched_sq1d_reg_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 20: dgeqrf_batched_sq1d_reg_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 21: dgeqrf_batched_sq1d_reg_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 22: dgeqrf_batched_sq1d_reg_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 23: dgeqrf_batched_sq1d_reg_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 24: dgeqrf_batched_sq1d_reg_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 25: dgeqrf_batched_sq1d_reg_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 26: dgeqrf_batched_sq1d_reg_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 27: dgeqrf_batched_sq1d_reg_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 28: dgeqrf_batched_sq1d_reg_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 29: dgeqrf_batched_sq1d_reg_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 30: dgeqrf_batched_sq1d_reg_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 31: dgeqrf_batched_sq1d_reg_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        case 32: dgeqrf_batched_sq1d_reg_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(dA_array, ldda, dtau_array, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}
