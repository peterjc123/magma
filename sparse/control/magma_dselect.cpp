/*
    -- MAGMA (version 2.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date June 2018

       @generated from sparse/control/magma_zselect.cpp, normal z -> d, Mon Jun 25 18:24:27 2018
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"
#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }


magma_int_t
magma_dpartition( 
    double *a, 
    magma_int_t size, 
    magma_int_t pivot,
    magma_queue_t queue ) {
    
    
    // using std::swap;
    double tmp;
    
    double pivotValue = a[pivot];
    SWAP(a[pivot], a[size-1]);
    int storePos = 0;
    for(int loadPos=0; loadPos < size-1; loadPos++) {
        if( MAGMA_D_ABS(a[loadPos]) < MAGMA_D_ABS(pivotValue) ) {
            SWAP(a[loadPos], a[storePos]);
            storePos++;
        }
    }
    SWAP(a[storePos], a[size-1]);
    return storePos;
}

magma_int_t
magma_dmedian5( 
    double *a,
    magma_queue_t queue ) {
    
    
    
    // using std::swap;
    double tmp;

    double a0 = a[0];
    double a1 = a[1];
    double a2 = a[2];
    double a3 = a[3];
    double a4 = a[4];
    if ( MAGMA_D_ABS(a1) < MAGMA_D_ABS( a0))
        SWAP( a0, a1);
    if ( MAGMA_D_ABS(a2) < MAGMA_D_ABS( a0))
        SWAP( a0, a2);
    if ( MAGMA_D_ABS(a3) < MAGMA_D_ABS( a0))
        SWAP( a0, a3);
    if ( MAGMA_D_ABS(a4) < MAGMA_D_ABS( a0))
        SWAP( a0, a4);
    if ( MAGMA_D_ABS(a2) < MAGMA_D_ABS( a1))
        SWAP( a1, a2);
    if ( MAGMA_D_ABS(a3) < MAGMA_D_ABS( a1))
        SWAP( a1, a3);
    if ( MAGMA_D_ABS(a4) < MAGMA_D_ABS( a1))
        SWAP( a1, a4);
    if ( MAGMA_D_ABS(a3) < MAGMA_D_ABS( a2))
        SWAP( a2, a3);
    if ( MAGMA_D_ABS(a4) < MAGMA_D_ABS( a2))
        SWAP( a2, a4);
    if ( MAGMA_D_ABS(a2) == MAGMA_D_ABS(a[0]))
        return 0;
    if ( MAGMA_D_ABS(a2) == MAGMA_D_ABS(a[1]))
        return 1;
    if ( MAGMA_D_ABS(a2) == MAGMA_D_ABS(a[2]))
        return 2;
    if ( MAGMA_D_ABS(a2) == MAGMA_D_ABS(a[3]))
        return 3;
    // else if ( MAGMA_D_ABS(a2) == MAGMA_D_ABS(a[4]))
    return 4;
}



/**
    Purpose
    -------

    An efficient implementation of Blum, Floyd,
    Pratt, Rivest, and Tarjan's worst-case linear
    selection algorithm
    
    Derrick Coetzee, webmaster@moonflare.com
    January 22, 2004
    http://moonflare.com/code/select/select.pdf

    Arguments
    ---------

    @param[in,out]
    a           double*
                array to select from

    @param[in]
    size        magma_int_t
                size of array

    @param[in]
    k           magma_int_t
                k-th smallest element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dselect(
    double *a, 
    magma_int_t size, 
    magma_int_t k,
    magma_queue_t queue ) {
        
    magma_int_t info = 0;
    
    // using std::swap;
    double tmp;
    
    if (size < 5) {
        for (int i=0; i<size; i++)
            for (int j=i+1; j<size; j++)
                if (MAGMA_D_ABS(a[j]) < MAGMA_D_ABS(a[i]))
                    SWAP(a[i], a[j]);
        return info;
    }
    
    int groupNum = 0;
    double *group = a;
    for( ; groupNum*5 <= size-5; group += 5, groupNum++) {
        SWAP(group[magma_dmedian5(group, queue)], a[groupNum]);
    }
    int numMedians = size/5;
    // Index of median of medians
    int MOMIdx = numMedians/2;
    magma_dselect(a, numMedians, MOMIdx, queue);
    int newMOMIdx = magma_dpartition(a, size, MOMIdx, queue);
    if (k != newMOMIdx) {
        if (k < newMOMIdx) {
                magma_dselect(a, newMOMIdx, k, queue);
        } else /* if (k > newMOMIdx) */ {
            magma_dselect(a + newMOMIdx + 1, size - newMOMIdx - 1, k - newMOMIdx - 1, queue);
        }
    }
    
    return info;
}


/**
    Purpose
    -------

    An efficient implementation of Blum, Floyd,
    Pratt, Rivest, and Tarjan's worst-case linear
    selection algorithm
    
    Derrick Coetzee, webmaster@moonflare.com
    January 22, 2004
    http://moonflare.com/code/select/select.pdf

    Arguments
    ---------

    @param[in,out]
    a           double*
                array to select from

    @param[in]
    size        magma_int_t
                size of array

    @param[in]
    k           magma_int_t
                k-th smallest element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dselectrandom( 
    double *a, 
    magma_int_t size, 
    magma_int_t k,
    magma_queue_t queue ) {
    
    magma_int_t info = 0;
    //using std::swap;
    double tmp;
    
    if (size < 5) {
        for (int i=0; i<size; i++)
            for (int j=i+1; j<size; j++)
                if (MAGMA_D_ABS(a[j]) < MAGMA_D_ABS(a[i]))
                    SWAP(a[i], a[j]);
        return info;
    }
    int pivotIdx = magma_dpartition(a, size, rand() % size, queue);
    if (k != pivotIdx) {
        if (k < pivotIdx) {
            magma_dselectrandom(a, pivotIdx, k, queue);
        } else /* if (k > pivotIdx) */ {
            magma_dselectrandom(a + pivotIdx + 1, size - pivotIdx - 1, k - pivotIdx - 1, queue);
        }
    }
    return info;
}
