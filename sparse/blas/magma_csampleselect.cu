/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date October 2020

       @author Tobias Ribizel

       @generated from sparse/blas/magma_zsampleselect.cu, normal z -> c, Thu Oct  8 23:05:49 2020
*/

#include "magma_sampleselect.h"
#include <cstdint>

#define PRECISION_c


namespace magma_sampleselect {

__global__ void compute_abs(const magmaFloatComplex* __restrict__ in, float* __restrict__ out, int32_t size) 
{
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    auto v = in[idx];
    out[idx] = real(v) * real(v) + imag(v) * imag(v);
}

} // namespace magma_sampleselect

using namespace magma_sampleselect;

/**
    Purpose
    -------

    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaFloatComplex
                array containing the values

    @param[out]
    thrs        float*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_csampleselect(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaFloatComplex *val,
    float *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{    
    magma_int_t info = 0;
    magma_int_t arch = magma_getdevice_arch();

    if( arch >= 350 ) {
        magma_int_t num_blocks = magma_ceildiv(total_size, block_size);
        magma_int_t required_size = sizeof(float) * (total_size * 2 + searchtree_size)
                                    + sizeof(int32_t) * sampleselect_alloc_size(total_size);
        auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

        float* gputmp1 = (float*)*tmp_ptr;
        float* gputmp2 = gputmp1 + total_size;
        float* gputree = gputmp2 + total_size;
        float* gpuresult = gputree + searchtree_size;
        int32_t* gpuints = (int32_t*)(gpuresult + 1);

        CHECK(realloc_result);

        compute_abs<<<num_blocks, block_size, 0, queue->cuda_stream()>>>
            (val, gputmp1, total_size);
        sampleselect<<<1, 1, 0, queue->cuda_stream()>>>
            (gputmp1, gputmp2, gputree, gpuints, total_size, subset_size, gpuresult);
        magma_sgetvector(1, gpuresult, 1, thrs, 1, queue );
        *thrs = std::sqrt(*thrs);    
    }
    else {
        printf("error: this functionality needs CUDA architecture >= 3.5\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    return info;
}

/**
    Purpose
    -------

    This routine selects an approximate threshold separating the subset_size
    smallest magnitude elements from the rest.

    Arguments
    ---------

    @param[in]
    total_size  magma_int_t
                size of array val

    @param[in]
    subset_size magma_int_t
                number of smallest elements to separate

    @param[in]
    val         magmaFloatComplex
                array containing the values

    @param[out]
    thrs        float*
                computed threshold

    @param[inout]
    tmp_ptr     magma_ptr*
                pointer to pointer to temporary storage.
                May be reallocated during execution.

    @param[inout]
    tmp_size    magma_int_t*
                pointer to size of temporary storage.
                May be increased during execution.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_csampleselect_approx(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaFloatComplex *val,
    float *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    auto num_blocks = magma_ceildiv(total_size, block_size);
    auto local_work = (total_size + num_threads - 1) / num_threads;
    auto required_size = sizeof(float) * (total_size + searchtree_size)
             + sizeof(int32_t) * (searchtree_width * (num_grouped_blocks + 1) + 1);
    auto realloc_result = realloc_if_necessary(tmp_ptr, tmp_size, required_size);

    float* gputmp = (float*)*tmp_ptr;
    float* gputree = gputmp + total_size;
    uint32_t* gpubucketidx = (uint32_t*)(gputree + searchtree_size);
    int32_t* gpurankout = (int32_t*)(gpubucketidx + 1);
    int32_t* gpucounts = gpurankout + 1;
    int32_t* gpulocalcounts = gpucounts + searchtree_width;
    uint32_t bucketidx{};

    CHECK(realloc_result);

    compute_abs<<<num_blocks, block_size, 0, queue->cuda_stream()>>>
        (val, gputmp, total_size);
    build_searchtree<<<1, sample_size, 0, queue->cuda_stream()>>>
        (gputmp, gputree, total_size);
    count_buckets<<<num_grouped_blocks, block_size, 0, queue->cuda_stream()>>>
        (gputmp, gputree, gpulocalcounts, total_size, local_work);
    reduce_counts<<<searchtree_width, num_grouped_blocks, 0, queue->cuda_stream()>>>
        (gpulocalcounts, gpucounts, num_grouped_blocks);
    sampleselect_findbucket<<<1, searchtree_width / 2, 0, queue->cuda_stream()>>>
        (gpucounts, subset_size, gpubucketidx, gpurankout);
    magma_getvector(1, sizeof(uint32_t), gpubucketidx, 1, &bucketidx, 1, queue);
    magma_sgetvector(1, gputree + searchtree_width - 1 + bucketidx, 1, thrs, 1, queue);
    *thrs = std::sqrt(*thrs);

cleanup:
    return info;
}
