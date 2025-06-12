#ifndef PROPAGATOR_H_
#define PROPAGATOR_H_

#include "WaveField.h"

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

typedef std::shared_ptr<Propagator> PropagatorPtr;
                                                                                                                                             
#endif