/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020
       
       @author Mark Gates
*/

#include <complex>

#include "magma_operators.h"

extern "C"
magmaDoubleComplex magma_zsqrt( magmaDoubleComplex x )
{
    std::complex<double> y = std::sqrt( std::complex<double>( real(x), imag(x) ));
    return MAGMA_Z_MAKE( real(y), imag(y) );
}


extern "C"
magmaFloatComplex magma_csqrt( magmaFloatComplex x )
{
    std::complex<float> y = std::sqrt( std::complex<float>( real(x), imag(x) ));
    return MAGMA_C_MAKE( real(y), imag(y) );
}
