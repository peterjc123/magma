/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @generated from sparse/blas/magma_z_blaswrapper.cpp, normal z -> c, Sun Mar 29 20:48:34 2020
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and scalars alpha, beta
    the wrapper determines the suitable SpMV computing
              y = alpha * A * x + beta * y.
    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex
                scalar alpha

    @param[in]
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    x           magma_c_matrix
                input vector x
                
    @param[in]
    beta        magmaFloatComplex
                scalar beta
    @param[out]
    y           magma_c_matrix
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_c_spmv(
    magmaFloatComplex alpha,
    magma_c_matrix A,
    magma_c_matrix x,
    magmaFloatComplex beta,
    magma_c_matrix y,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_c_matrix x2={Magma_CSR};
    magma_c_matrix dA={Magma_CSR};
    magma_c_matrix dx={Magma_CSR};
    magma_c_matrix dy={Magma_CSR};

    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descr = 0;
    // make sure RHS is a dense matrix
    if ( x.storage_type != Magma_DENSE ) {
         printf("error: only dense vectors are supported for SpMV.\n");
         info = MAGMA_ERR_NOT_SUPPORTED;
         goto cleanup;
    }

    if ( A.memory_location != x.memory_location ||
         x.memory_location != y.memory_location ) {
        printf("error: linear algebra objects are not located in same memory!\n");
        printf("memory locations are: %d   %d   %d\n",
                        A.memory_location, x.memory_location, y.memory_location );
        info = MAGMA_ERR_INVALID_PTR;
        goto cleanup;
    }

    // DEV case
    if ( A.memory_location == Magma_DEV ) {
        if ( A.num_cols == x.num_rows && x.num_cols == 1 ) {
            if ( A.storage_type == Magma_CSR   ||
                 A.storage_type == Magma_CUCSR ||
                 A.storage_type == Magma_CSRL  ||
                 A.storage_type == Magma_CSRU )
            {
                CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
                CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
                CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
                
                CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
                CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
                
                CHECK_CUSPARSE( cusparseCcsrmv( cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                              A.num_rows, A.num_cols, A.nnz, &alpha, descr,
                              A.dval, A.drow, A.dcol, x.dval, &beta, y.dval ) );
            }
            else if ( A.storage_type == Magma_CSC )
            {
                CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
                CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
                CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
                
                CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
                CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
                
                cusparseCcsrmv( cusparseHandle,CUSPARSE_OPERATION_TRANSPOSE,
                              A.num_rows, A.num_cols, A.nnz, &alpha, descr,
                              A.dval, A.drow, A.dcol, x.dval, &beta, y.dval );
            }
            else if ( A.storage_type == Magma_ELL ) {
                //printf("using ELLPACKT kernel for SpMV: ");
                CHECK( magma_cgeelltmv( MagmaNoTrans, A.num_rows, A.num_cols,
                   A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, beta,
                   y.dval, queue ));
                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_ELLPACKT ) {
                //printf("using ELL kernel for SpMV: ");
                CHECK( magma_cgeellmv( MagmaNoTrans, A.num_rows, A.num_cols,
                   A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, beta,
                   y.dval, queue ));
                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_ELLRT ) {
                //printf("using ELLRT kernel for SpMV: ");
                CHECK( magma_cgeellrtmv( MagmaNoTrans, A.num_rows, A.num_cols,
                           A.max_nnz_row, alpha, A.dval, A.dcol, A.drow, x.dval,
                        beta, y.dval, A.alignment, A.blocksize, queue ));
                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_SELLP ) {
                //printf("using SELLP kernel for SpMV: ");
                CHECK( magma_cgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols,
                   A.blocksize, A.numblocks, A.alignment,
                   alpha, A.dval, A.dcol, A.drow, x.dval, beta, y.dval, queue ));

                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_CSR5 ) {
                //printf("using CSR5 kernel for SpMV: ");
                CHECK( magma_cgecsr5mv( MagmaNoTrans, A.num_rows, A.num_cols, 
                   A.csr5_p, alpha, A.csr5_sigma, A.csr5_bit_y_offset, 
                   A.csr5_bit_scansum_offset, A.csr5_num_packets, 
                   A.dtile_ptr, A.dtile_desc, A.dtile_desc_offset_ptr, A.dtile_desc_offset, 
                   A.dcalibrator, A.csr5_tail_tile_start, 
                   A.dval, A.drow, A.dcol, x.dval, beta, y.dval, queue ));

                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_DENSE ) {
                //printf("using DENSE kernel for SpMV: ");
                magmablas_cgemv( MagmaNoTrans, A.num_rows, A.num_cols, alpha,
                               A.dval, A.num_rows, x.dval, 1, beta,  y.dval,
                               1, queue );
                //printf("done.\n");
            }
            else if ( A.storage_type == Magma_SPMVFUNCTION ) {
                //printf("using DENSE kernel for SpMV: ");
                CHECK( magma_ccustomspmv( x.num_rows, x.num_cols, alpha, beta, x.dval, y.dval, queue ));
                // magma_cge3pt(  x.num_rows, x.num_cols, &alpha, &beta, x.dval, y.dval, queue );
                // printf("done.\n");
            }
            else if ( A.storage_type == Magma_BCSR ) {
                //printf("using CUSPARSE BCSR kernel for SpMV: ");
               // CUSPARSE context //
               cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
               int mb = magma_ceildiv( A.num_rows, A.blocksize );
               int nb = magma_ceildiv( A.num_cols, A.blocksize );
               CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
               CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
               CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
               cusparseCbsrmv( cusparseHandle, dirA,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, A.numblocks,
                   &alpha, descr, A.dval, A.drow, A.dcol, A.blocksize, x.dval,
                   &beta, y.dval );
            }
            else {
                printf("error: format not supported.\n");
                info = MAGMA_ERR_NOT_SUPPORTED; 
            }
        }
        else if ( A.num_cols < x.num_rows || x.num_cols > 1 ) {
            magma_int_t num_vecs = x.num_rows / A.num_cols * x.num_cols;
            if ( A.storage_type == Magma_CSR ) {
                CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
                CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
                CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
                CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
                CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));

                if ( x.major == MagmaColMajor) {
                    cusparseCcsrmm(cusparseHandle,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   A.num_rows,   num_vecs, A.num_cols, A.nnz,
                                   &alpha, descr, A.dval, A.drow, A.dcol,
                                   x.dval, A.num_cols, &beta, y.dval, A.num_cols);
                } else if ( x.major == MagmaRowMajor) {
                    /*cusparseCcsrmm2(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    A.num_rows,   num_vecs, A.num_cols, A.nnz,
                    &alpha, descr, A.dval, A.drow, A.dcol,
                    x.dval, A.num_cols, &beta, y.dval, A.num_cols);
                    */
                    printf("error: format not supported.\n");
                    info = MAGMA_ERR_NOT_SUPPORTED;
                } else if ( A.storage_type == Magma_CSC )
                    {
                        CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
                        CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
                        CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
                        
                        CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
                        CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
                        
                        cusparseCcsrmm( cusparseHandle,CUSPARSE_OPERATION_TRANSPOSE,
                                        A.num_rows,   num_vecs, A.num_cols, A.nnz,
                                        &alpha, descr, A.dval, A.drow, A.dcol,
                                        x.dval, A.num_cols, &beta, y.dval, A.num_cols);
                    }
            } else if ( A.storage_type == Magma_SELLP ) {
                if ( x.major == MagmaRowMajor) {
                    CHECK( magma_cmgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols,
                                              num_vecs, A.blocksize, A.numblocks, A.alignment,
                                              alpha, A.dval, A.dcol, A.drow, x.dval, beta, y.dval, queue ));
                } else if ( x.major == MagmaColMajor) {
                    // transpose first to row major
                    CHECK( magma_cvtranspose( x, &x2, queue ));
                    CHECK( magma_cmgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols,
                                              num_vecs, A.blocksize, A.numblocks, A.alignment,
                                              alpha, A.dval, A.dcol, A.drow, x2.dval, beta, y.dval, queue ));
                }
            }
            /*if ( A.storage_type == Magma_DENSE ) {
                //printf("using DENSE kernel for SpMV: ");
                magmablas_cmgemv( MagmaNoTrans, A.num_rows, A.num_cols,
                           num_vecs, alpha, A.dval, A.num_rows, x.dval, 1,
                           beta,  y.dval, 1 );
                //printf("done.\n");
            }*/
            else {
                printf("error: format not supported.\n");
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
    }
    // CPU case missing!
    else {
        CHECK( magma_cmtransfer( x, &dx, x.memory_location, Magma_DEV, queue ));
        CHECK( magma_cmtransfer( y, &dy, y.memory_location, Magma_DEV, queue ));
        CHECK( magma_cmtransfer( A, &dA, A.memory_location, Magma_DEV, queue ));
        CHECK( magma_c_spmv( alpha, dA, dx, beta, dy, queue ) );
        magma_cmfree(&x, queue );
        magma_cmfree(&y, queue );
        magma_cmfree(&A, queue );
        CHECK( magma_cmtransfer( dx, &x, dx.memory_location, Magma_CPU, queue ));
        CHECK( magma_cmtransfer( dy, &y, dy.memory_location, Magma_CPU, queue ));
        CHECK( magma_cmtransfer( dA, &A, dA.memory_location, Magma_CPU, queue ));

    }

cleanup:
    cusparseDestroyMatDescr( descr );
    cusparseDestroy( cusparseHandle );
    cusparseHandle = 0;
    descr = 0;
    magma_cmfree(&x2, queue );
    magma_cmfree(&dx, queue );
    magma_cmfree(&dy, queue );
    magma_cmfree(&dA, queue );
    
    return info;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and scalars alpha, beta
    the wrapper determines the suitable SpMV computing
              y = alpha * ( A - lambda I ) * x + beta * y.
    Arguments
    ---------

    @param
    alpha       magmaFloatComplex
                scalar alpha

    @param
    A           magma_c_matrix
                sparse matrix A

    @param
    lambda      magmaFloatComplex
                scalar lambda

    @param
    x           magma_c_matrix
                input vector x

    @param
    beta        magmaFloatComplex
                scalar beta
                
    @param
    offset      magma_int_t
                in case not the main diagonal is scaled
                
    @param
    blocksize   magma_int_t
                in case of processing multiple vectors
                
    @param
    add_rows    magma_int_t*
                in case the matrixpowerskernel is used
                
    @param
    y           magma_c_matrix
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_c_spmv_shift(
    magmaFloatComplex alpha,
    magma_c_matrix A,
    magmaFloatComplex lambda,
    magma_c_matrix x,
    magmaFloatComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magma_index_t *add_rows,
    magma_c_matrix y,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // make sure RHS is a dense matrix
    if ( x.storage_type != Magma_DENSE ) {
         printf("error: only dense vectors are supported.\n");
         info = MAGMA_ERR_NOT_SUPPORTED;
         goto cleanup;
    }


    if ( A.memory_location != x.memory_location ||
         x.memory_location != y.memory_location ) {
        printf("error: linear algebra objects are not located in same memory!\n");
        printf("memory locations are: %d   %d   %d\n",
                    A.memory_location, x.memory_location, y.memory_location );
        info = MAGMA_ERR_INVALID_PTR;
        goto cleanup;
    }

    // DEV case
    if ( A.memory_location == Magma_DEV ) {
        if ( A.storage_type == Magma_CSR ) {
            //printf("using CSR kernel for SpMV: ");
            CHECK( magma_cgecsrmv_shift( MagmaNoTrans, A.num_rows, A.num_cols,
               alpha, lambda, A.dval, A.drow, A.dcol, x.dval, beta, offset,
               blocksize, add_rows, y.dval, queue ));
            //printf("done.\n");
        }
        else if ( A.storage_type == Magma_ELLPACKT ) {
            //printf("using ELLPACKT kernel for SpMV: ");
            CHECK( magma_cgeellmv_shift( MagmaNoTrans, A.num_rows, A.num_cols,
               A.max_nnz_row, alpha, lambda, A.dval, A.dcol, x.dval, beta, offset,
               blocksize, add_rows, y.dval, queue ));
            //printf("done.\n");
        }
        else if ( A.storage_type == Magma_ELL ) {
            //printf("using ELL kernel for SpMV: ");
            CHECK( magma_cgeelltmv_shift( MagmaNoTrans, A.num_rows, A.num_cols,
               A.max_nnz_row, alpha, lambda, A.dval, A.dcol, x.dval, beta, offset,
               blocksize, add_rows, y.dval, queue ));
            //printf("done.\n");
        }
        else {
            printf("error: format not supported.\n");
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    // CPU case missing!
    else {
        printf("error: CPU not yet supported.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    For a given input matrix A and B and scalar alpha,
    the wrapper determines the suitable SpMV computing
              C = alpha * A * B.
    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex
                scalar alpha

    @param[in]
    A           magma_c_matrix
                sparse matrix A
                
    @param[in]
    B           magma_c_matrix
                sparse matrix C
                
    @param[out]
    C           magma_c_matrix *
                outpur sparse matrix C

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_c_spmm(
    magmaFloatComplex alpha,
    magma_c_matrix A,
    magma_c_matrix B,
    magma_c_matrix *C,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_c_matrix dA = {Magma_CSR};
    magma_c_matrix dB = {Magma_CSR};
    magma_c_matrix dC = {Magma_CSR};
    
    if ( A.memory_location != B.memory_location ) {
        printf("error: linear algebra objects are not located in same memory!\n");
        printf("memory locations are: %d   %d\n",
                        A.memory_location, B.memory_location );
        info = MAGMA_ERR_INVALID_PTR;
        goto cleanup;
    }

    // DEV case
    if ( A.memory_location == Magma_DEV ) {
        if ( A.num_cols == B.num_rows ) {
            if ( A.storage_type == Magma_CSR  ||
                 A.storage_type == Magma_CSRL ||
                 A.storage_type == Magma_CSRU ||
                 A.storage_type == Magma_CSRCOO ) {
               CHECK( magma_ccuspmm( A, B, C, queue ));
            }
            else {
                printf("error: format not supported.\n");
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
    }
    // CPU case missing!
    else {
        A.storage_type = Magma_CSR;
        B.storage_type = Magma_CSR;
        CHECK( magma_cmtransfer( A, &dA, Magma_CPU, Magma_DEV, queue ) );
        CHECK( magma_cmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ) );
        CHECK(  magma_ccuspmm( dA, dB, &dC, queue ) );
        CHECK( magma_cmtransfer( dC, C, Magma_DEV, Magma_CPU, queue ) );
    }
    
cleanup:
    magma_cmfree( &dA, queue );
    magma_cmfree( &dB, queue );
    magma_cmfree( &dC, queue );
    return info;
}
