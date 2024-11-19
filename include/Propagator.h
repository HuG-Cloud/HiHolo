#ifndef PROPAGATOR_H_
#define PROPAGATOR_H_

#include "WaveField.h"
#include "datatypes.h"
#include "cuda_utils.h"

class Propagator
{   
    private:
        IntArray imSize;
        F2DArray fresnelNumbers;
        int numImages;
        cuFloatComplex *propKernels;
        CUFFTUtils fftUtils;

    public:
        Propagator() = default;
        Propagator(const IntArray &imsize, const F2DArray &fresnelnumbers, CUDAPropKernel::Type type);
        void propagate(cuFloatComplex *complexWave, cuFloatComplex *propagatedWave);
        void backPropagate(cuFloatComplex *propagatedWave, cuFloatComplex *complexWave);
        ~Propagator();
};
                                                                                                                                               
#endif