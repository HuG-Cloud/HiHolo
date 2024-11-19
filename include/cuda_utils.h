#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "datatypes.h"
#include "math_utils.h"

class CUFFTUtils
{
    private:
        int numel;
        int rows;
        int cols;
        int batchSize;
        cufftHandle plan;
        cufftHandle plan_batch;

    public:
        CUFFTUtils(int in_numel);
        CUFFTUtils(int in_rows, int in_cols, int in_batchSize);
        void fft_fwd(cuFloatComplex *complexVec);
        void fft_bwd(cuFloatComplex *complexVec);

        // The size of complexWave is numel * batchSize
        void fft_fwd_batch(cuFloatComplex *complexWave);
        void fft_bwd_batch(cuFloatComplex *complexWave);
        ~CUFFTUtils();
};

namespace CUDAPropKernel
{
    enum Type{Fourier, Chirp, ChirpLimited};
    void generateKernel(cuFloatComplex* kernel, const IntArray &imSize, FArray &fresnelNumber, Type type);
    void genByFourier(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber);
    void genByChirp(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber);
    void genByChirpLimited(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber);
}

namespace CUDAUtils
{
    void genFFTFreq(float* rowRange, float* colRange, const IntArray &imSize, FArray &spacing);
    
    enum PaddingType {Constant, Replicate, Fadeout};

    void cropMatrix(cuFloatComplex* matrix, cuFloatComplex* matrix_new, int rows, int cols, int cropPreRows, int cropPreCols, int cropPostRows, int cropPostCols);
    void padByConstant(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, float padValue);
    void padByReplicate(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols);
    void padByFadeout(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols);
    // Pad matrix to given size in different ways
    void padMatrix(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, PaddingType type, float padValue = 0.0f);
}

// Normalize the inverse FFT result
__global__ void scaleComplexData(cuFloatComplex* data, int numel, float scale);
// Generate the FFT frequency and shift it
__global__ void genShiftedFFTFreq(float* output, int size, float spacing);

__global__ void genFourierComponent(cuFloatComplex *component, float *fftFreq, int size, float fresnelNumber);
__global__ void genChirpComponent(cuFloatComplex *component, float *fftFreq, int size, float fresnelNumber);

__global__ void genMaskComponent(float *component, int newSize, int padSize, float spacing = 1.0f);
__global__ void genMaskMatrix(float *mask, float *rowGrid, float *colGrid, int rows, int cols);
__global__ void multiplyMaskMatrix(float *matrix, float *mask, int numel, float value);

// Multiply obliquity factor to generate the chirplimited kernel
__global__ void multiplyObliFactor_1(float *fftFreq, int size, float fresnelNumber);
__global__ void multiplyObliFactor_2(float *matrix, float *rowRange, float *colRange, int rows, int cols);
__global__ void multiplyObliFactor_3(cuFloatComplex *kernel, float *matrix, int size);

__global__ void genFourierKernel(cuFloatComplex *kernel, cuFloatComplex *rowComponent, cuFloatComplex *colComponent, int rows, int cols);
__global__ void genChirpKernel(cuFloatComplex *kernel, cuFloatComplex *rowComponent, cuFloatComplex *colComponent, int rows, int cols, cuFloatComplex initCoeff);

__global__ void propProcess(cuFloatComplex *propagatedWave, cuFloatComplex *complexWave, cuFloatComplex *kernel, int numel, int batchSize);
__global__ void backPropProcess(cuFloatComplex *complexWave, cuFloatComplex *propagatedWave, cuFloatComplex *kernel, int numel, int batchSize);

__global__ void computeAmplitude(cuFloatComplex *complexWave, float *amplitude, int numel);
__global__ void computePhase(cuFloatComplex *complexWave, float *phase, int numel);
__global__ void setAmplitude(cuFloatComplex *complexWave, const float *targetAmplitude, int numel);
__global__ void setPhase(cuFloatComplex *complexWave, const float *targetPhase, int numel);

__global__ void addWaveField(cuFloatComplex *complexWave, const cuFloatComplex *waveField, int numel);
__global__ void subWaveField(cuFloatComplex *complexWave, const cuFloatComplex *waveField, int numel);
__global__ void reflectWaveField(cuFloatComplex *reflectedWave, const cuFloatComplex *waveField, int numel);

__global__ void adjustAmplitude(float *amplitude, float maxAmplitude, float minAmplitude, int numel);
__global__ void limitAmplitude(cuFloatComplex *complexWave, const float *amplitude, const float *targetAmplitude, int numel);

__global__ void sqrtAmplitude(float *amplitude, int numel);

// Initialize data with a given value
template <typename T>
__global__ void initializeData(T *data, T value, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = value;
    }
}

__global__ void initByPhase(cuFloatComplex *data, const float *phase, int numel);

#endif
