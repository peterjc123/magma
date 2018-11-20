/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from sparse/control/magma_zsort.cpp, normal z -> d, Mon Jun 25 18:24:28 2018
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


#define SWAP(a, b)  { tmp = val[a]; val[a] = val[b]; val[b] = tmp; }
#define SWAPM(a, b) { tmpv = val[a]; val[a] = val[b]; val[b] = tmpv;  \
                      tmpc = col[a]; col[a] = col[b]; col[b] = tmpc;  \
                      tmpr = row[a]; row[a] = row[b]; row[b] = tmpr; }
                      
#define UP 0
#define DOWN 1

/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           double*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dsort(
    double *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double temp;
    magma_index_t pivot,j,i;
    
    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_D_ABS(x[i]) <= MAGMA_D_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_D_ABS(x[j]) > MAGMA_D_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_dsort( x, first, j-1, queue ));
        CHECK( magma_dsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           double*
                array to sort
                
    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.
                
    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dmsort(
    double *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double temp;
    magma_index_t pivot,j,i, tmpcol, tmprow;
    
    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_D_ABS(x[i]) <= MAGMA_D_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_D_ABS(x[j]) > MAGMA_D_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tmpcol = col[i];
                col[i] = col[j];
                col[j] = tmpcol;
                tmprow = row[i];
                row[i] = row[j];
                row[j] = tmprow;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_dmsort( x, col, row, first, j-1, queue ));
        CHECK( magma_dmsort( x, col, row, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dindexsort(
    magma_index_t *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_dindexsort( x, first, j-1, queue ));
        CHECK( magma_dindexsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers, updates a respective array of values.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort
                
    @param[in,out]
    y           double*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dindexsortval(
    magma_index_t *x,
    double *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t pivot,j,temp,i;
    double tempval;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tempval = y[i];
                y[i] = y[j];
                y[j] = tempval;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        
        tempval=y[pivot];
        y[pivot]=y[j];
        y[j]=tempval;
        CHECK( magma_dindexsortval( x, y, first, j-1, queue ));
        CHECK( magma_dindexsortval( x, y, j+1, last, queue ));
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array and reorders 
    such that these elements come to the front. The related arrays col and row
    are also reordered.

    Arguments
    ---------
    
    @param[in,out]
    val         double*
                Target array, will be modified during operation.
                
    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.
                
    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     double*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dmorderstatistics(
    double *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    double *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t i, st;
    double tmpv;
    magma_index_t tmpc, tmpr;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%% error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) > MAGMA_D_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }
     
        SWAPM(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dmorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_dmorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%% error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) < MAGMA_D_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }
     
        SWAPM(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dmorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
            CHECK( magma_dmorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array.

    Arguments
    ---------
    
    @param[in,out]
    val         double*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     double*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dorderstatistics(
    double *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    double *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    double tmp;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) > MAGMA_D_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_dorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) < MAGMA_D_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }
     
        SWAP(length-1, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_dorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}



/**
    Purpose
    -------

    Approximates the k-th smallest element in an array by
    using order-statistics with step-size inc.

    Arguments
    ---------
    
    @param[in,out]
    val         double*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).
                
    @param[in]
    inc         magma_int_t
                Stepsize in the approximation.
                
    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest
                
    @param[out]
    element     double*
                location of the respective element
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dorderstatistics_inc(
    double *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t inc,
    magma_int_t r,
    double *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    double tmp;
    if( r == 0 ){
        for ( st = i = 0; i < length - inc; i=i+inc ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) > MAGMA_D_ABS(val[length-inc]) ){
                continue;
            }
            SWAP(i, st);
            st=st+inc;
        }
     
        SWAP(length-inc, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_dorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - inc; i=i+inc ) {
            if ( magma_d_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_D_REAL(val[i]), MAGMA_D_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_D_ABS(val[i]) < MAGMA_D_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st=st+inc;
        }
     
        SWAP(length-inc, st);
     
        if ( k == st ){
            *element = val[st];    
        }
        else if ( st > k ) {
            CHECK( magma_dorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_dorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }
    
cleanup:
    return info;
}




void swap(double *a, double *b)
{
    double t;
    t = *a;
    *a = *b;
    *b = t;
}


/**
    Purpose
    -------

    Approximates the k-th smallest element in an array by
    using order-statistics with step-size inc.

    Arguments
    ---------
    
    @param[in]                                                                                                                                                         
    start       magma_int_t
                Start position of the target array.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in,out]
    seq         double*
                Target array, will be modified during operation.

    @param[in]
    flag        magma_int_t
                ???
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dbitonic_sort(
    magma_int_t start, 
    magma_int_t length, 
    double *seq, 
    magma_int_t flag,
    magma_queue_t queue )
{
    
    magma_int_t info =0;
    
    magma_int_t m, i, num_threads=1;
    magma_int_t split_length;
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#endif

    m = length/num_threads;


    if (length == 1)
        return 0;

    if (length % 2 !=0 )
    {
        printf("The length of a (sub)sequence can not be divided by 2.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }

    split_length = length / 2;

    // bitonic split
    #pragma omp parallel for shared(seq, flag, start, split_length) private(i)
    for (i = start; i < start + split_length; i++)
    {
        if (flag == UP)
        {
            if (MAGMA_D_ABS(seq[i]) > MAGMA_D_ABS(seq[i + split_length]))
                swap(&seq[i], &seq[i + split_length]);
        }
        else
        {
            if (MAGMA_D_ABS(seq[i]) < MAGMA_D_ABS(seq[i + split_length]))
                swap(&seq[i], &seq[i + split_length]);
        }
    }

    if (split_length > m)
    {
        // m is the size of sub part-> n/numThreads
        magma_dbitonic_sort(start, split_length, seq, flag, queue);
        magma_dbitonic_sort(start + split_length, split_length, seq, flag, queue);
    }

cleanup:
    return info;
}
