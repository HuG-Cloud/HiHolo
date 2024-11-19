#include "Propagator.h"

Propagator::Propagator(const IntArray &imsize, const F2DArray &fresnelnumbers, CUDAPropKernel::Type type): imSize(imsize),
                       fresnelNumbers(fresnelnumbers), fftUtils(imsize[0], imsize[1], fresnelnumbers.size())
{
    numImages = fresnelNumbers.size();
    cudaMalloc(&propKernels, numImages * imSize[0] * imSize[1] * sizeof(cuFloatComplex));

    for (size_t i = 0; i < numImages; i++) {
        FArray fresnelNumber(fresnelNumbers[i].begin(), fresnelNumbers[i].end());
        CUDAPropKernel::generateKernel(propKernels + i * imSize[0] * imSize[1], imSize, fresnelNumber, type);
    }
}

void Propagator::propagate(cuFloatComplex *complexWave, cuFloatComplex *propagatedWave)
{       
    // imProp = obj.iFT(obj.propKernel .* fftn(imProp));
    fftUtils.fft_fwd(complexWave);

    // Process of propagation
    int blockSize = 1024;
    int numBlocks = (numImages * imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    propProcess<<<numBlocks, blockSize>>>(propagatedWave, complexWave, propKernels, imSize[0] * imSize[1], numImages);

    fftUtils.fft_bwd_batch(propagatedWave);
}

void Propagator::backPropagate(cuFloatComplex *propagatedWave, cuFloatComplex *complexWave)
{       
    // Process of back propagation
    fftUtils.fft_fwd_batch(propagatedWave);

    int blockSize = 1024;
    int numBlocks = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    backPropProcess<<<numBlocks, blockSize>>>(complexWave, propagatedWave, propKernels, imSize[0] * imSize[1], numImages);

    fftUtils.fft_bwd(complexWave);
}

Propagator::~Propagator()
{
    cudaFree(propKernels);
}