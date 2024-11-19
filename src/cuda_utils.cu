#include "cuda_utils.h"

CUFFTUtils::CUFFTUtils(int in_numel): numel(in_numel)
{
    cufftPlan1d(&plan, numel, CUFFT_C2C, 1);
}

CUFFTUtils::CUFFTUtils(int in_rows, int in_cols, int in_batchSize):
rows(in_rows), cols(in_cols), batchSize(in_batchSize), numel(in_rows * in_cols)
{
    cufftPlan2d(&plan, rows, cols, CUFFT_C2C);
    int size[2] = {rows, cols};
    cufftPlanMany(&plan_batch, 2, size, nullptr, 1, numel, nullptr, 1, numel, CUFFT_C2C, batchSize);
}

void CUFFTUtils::fft_fwd(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan, complexWave, complexWave, CUFFT_FORWARD);
}

void CUFFTUtils::fft_bwd(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan, complexWave, complexWave, CUFFT_INVERSE);
    float scale = 1.0f / numel;
    int blockSize = 1024;
    int numBlocks = (numel + blockSize - 1) / blockSize;
    scaleComplexData<<<numBlocks, blockSize>>>(complexWave, numel, scale);
}

void CUFFTUtils::fft_fwd_batch(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan_batch, complexWave, complexWave, CUFFT_FORWARD);
}

void CUFFTUtils::fft_bwd_batch(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan_batch, complexWave, complexWave, CUFFT_INVERSE);
    float scale = 1.0f / numel;
    int blockSize = 1024;
    int numBlocks = (numel * batchSize + blockSize - 1) / blockSize;
    scaleComplexData<<<numBlocks, blockSize>>>(complexWave, numel * batchSize, scale);
}

CUFFTUtils::~CUFFTUtils()
{   
    cufftDestroy(plan);
    if (rows != 0 && cols != 0) {
        cufftDestroy(plan_batch);
    }
}

__global__ void scaleComplexData(cuFloatComplex* data, int numel, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void computeAmplitude(cuFloatComplex *complexWave, float *amplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        amplitude[idx] = hypotf(complexWave[idx].x, complexWave[idx].y);
    }

}

__global__ void computePhase(cuFloatComplex *complexWave, float *phase, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        phase[idx] = atan2f(complexWave[idx].y, complexWave[idx].x);
    }
}

__global__ void setAmplitude(cuFloatComplex *complexWave, const float *targetAmplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float amplitude = hypotf(complexWave[idx].x, complexWave[idx].y);
        if (amplitude >= 1e-10) {
            float scale = targetAmplitude[idx] / amplitude;
            complexWave[idx].x *= scale;
            complexWave[idx].y *= scale;
        } else {
            complexWave[idx] = make_cuFloatComplex(targetAmplitude[idx], 0.0f);
        }
    }
}

__global__ void setPhase(cuFloatComplex *complexWave, const float *targetPhase, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float amplitude = hypotf(complexWave[idx].x, complexWave[idx].y);
        complexWave[idx] = make_cuFloatComplex(amplitude * cosf(targetPhase[idx]), amplitude * sinf(targetPhase[idx]));
    }

}

__global__ void initByPhase(cuFloatComplex *data, const float *phase, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = make_cuFloatComplex(cosf(phase[idx]), sinf(phase[idx]));
    }
}

__global__ void limitAmplitude(cuFloatComplex *complexWave, const float *amplitude, const float *targetAmplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if (amplitude[idx] >= 1e-10) {
            float scale = targetAmplitude[idx] / amplitude[idx];
            complexWave[idx].x *= scale;
            complexWave[idx].y *= scale;
        } else {
            complexWave[idx] = make_cuFloatComplex(targetAmplitude[idx], 0.0f);
        }
    }
}

__global__ void adjustAmplitude(float *amplitude, float maxAmplitude, float minAmplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if (maxAmplitude < INFINITY) {
            amplitude[idx] = min(amplitude[idx], maxAmplitude);
        }
        if (minAmplitude > 0) {
            amplitude[idx] = max(amplitude[idx], minAmplitude);
        }
    }
}

__global__ void sqrtAmplitude(float *amplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        amplitude[idx] = sqrtf(amplitude[idx]);
    }
}

__global__ void addWaveField(cuFloatComplex *complexWave, const cuFloatComplex *waveField, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        complexWave[idx] = cuCaddf(complexWave[idx], waveField[idx]);
    }
}

__global__ void subWaveField(cuFloatComplex *complexWave, const cuFloatComplex *waveField, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        complexWave[idx] = cuCsubf(complexWave[idx], waveField[idx]);
    }
}

__global__ void reflectWaveField(cuFloatComplex *reflectedWave, const cuFloatComplex *waveField, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        cuFloatComplex tmp = make_cuFloatComplex(2.0f * reflectedWave[idx].x, 2.0f * reflectedWave[idx].y);
        reflectedWave[idx] = cuCsubf(tmp, waveField[idx]);
    }
}

__global__ void genShiftedFFTFreq(float* output, int size, float spacing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 首先生成范围
        float start = - floorf(0.5f * size);
        float value = start + idx;
        
        // 计算频率
        value *= ((2.0f * M_PIf32) / (size * spacing));
            
        // 执行ifftshift
        int mid = size / 2;
        int new_idx;
        if (idx < mid) {
            new_idx = idx + (size - mid);
        } else {
            new_idx = idx - mid;
        }
        output[new_idx] = value;
    }
}

__global__ void genMaskComponent(float *component, int newSize, int padSize, float spacing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < newSize) {
        int i = -(newSize - 1) + 2 * idx;
        float tmpValue = max(0.0f, abs(i * 0.5f * spacing) - (newSize - padSize) / 2.0f);
        component[idx] = min(M_PIf32 / max(1, padSize) * tmpValue, M_PIf32);
    }
}

__global__ void genFourierComponent(cuFloatComplex *component, float *fftFreq, int size, float fresnelNumber)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float freq2 = fftFreq[idx] * fftFreq[idx];
        float angle = - freq2 / (4.0f * M_PIf32 * fresnelNumber);
        component[idx] = make_cuFloatComplex(cosf(angle), sinf(angle));
    }
}

__global__ void genChirpComponent(cuFloatComplex *component, float *fftFreq, int size, float fresnelNumber)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float freq2 = fftFreq[idx] * fftFreq[idx];
        float angle = freq2 * M_PIf32 * fresnelNumber;
        component[idx] = make_cuFloatComplex(cosf(angle), sinf(angle));
    }
    
}

__global__ void genMaskMatrix(float *mask, float *rowGrid, float *colGrid, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        mask[idx] = (1.0f + cosf(rowGrid[row])) * (0.25f * (1.0f + cosf(colGrid[col])));
    }
}

__global__ void multiplyObliFactor_1(float *fftFreq, int size, float fresnelNumber)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        fftFreq[idx] = powf(2.0f * fresnelNumber * fftFreq[idx], 2);
    }
}

__global__ void multiplyObliFactor_2(float *matrix, float *rowRange, float *colRange, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        matrix[row * cols + col] = rowRange[row] + colRange[col];
    }
}

__global__ void multiplyObliFactor_3(cuFloatComplex *kernel, float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tmp = cosf((M_PIf32 / 2.0f) * min(matrix[idx], 1.0f));
        kernel[idx].x *= tmp;
        kernel[idx].y *= tmp;
    }
}

__global__ void multiplyMaskMatrix(float *matrix, float *mask, int numel, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        matrix[idx] = mask[idx] * matrix[idx] + (1.0f - mask[idx]) * value;
    }
}

__global__ void genFourierKernel(cuFloatComplex *kernel, cuFloatComplex *rowComponent, cuFloatComplex *colComponent, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        kernel[idx] = cuCmulf(rowComponent[row], colComponent[col]);
    }
}

__global__ void genChirpKernel(cuFloatComplex *kernel, cuFloatComplex *rowComponent, cuFloatComplex *colComponent, int rows, int cols, cuFloatComplex initCoeff)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        kernel[idx] = cuCmulf(cuCmulf(rowComponent[row], colComponent[col]), initCoeff);
    }
}

__global__ void propProcess(cuFloatComplex *propagatedWave, cuFloatComplex *complexWave, cuFloatComplex *kernel, int numel, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int waveIdx = idx % numel;
    if (idx < numel * batchSize) {
        propagatedWave[idx] = cuCmulf(kernel[idx], complexWave[waveIdx]);
    }

}

__global__ void backPropProcess(cuFloatComplex *complexWave, cuFloatComplex *propagatedWave, cuFloatComplex *kernel, int numel, int batchSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numel) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int i = 0; i < batchSize; i++) {
            int idx = i * numel + col;
            cuFloatComplex conjKernel = cuConjf(kernel[idx]);
            sum = cuCaddf(sum, cuCmulf(propagatedWave[idx], conjKernel));
        }
        complexWave[col] = sum;
    }
}

void CUDAUtils::genFFTFreq(float* rowRange, float* colRange, const IntArray &imSize, FArray &spacing)
{
    if (spacing.size() != imSize.size() && spacing.size() != 1) {
        throw std::invalid_argument("Invalid sample spacings!");
    }

    if (spacing.size() == 1)
        spacing.push_back(spacing[0]);

    int rows = imSize[0];
    int cols = imSize[1];

    int blockSize = 512;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    genShiftedFFTFreq<<<numBlocks, blockSize>>>(rowRange, rows, spacing[0]);

    numBlocks = (cols + blockSize - 1) / blockSize;
    genShiftedFFTFreq<<<numBlocks, blockSize>>>(colRange, cols, spacing[1]);
}

void CUDAUtils::cropMatrix(cuFloatComplex* matrix, cuFloatComplex* matrix_new, int rows, int cols, int cropPreRows, int cropPreCols, int cropPostRows, int cropPostCols)
{
    int newRows = rows - cropPreRows - cropPostRows;
    int newCols = cols - cropPreCols - cropPostCols;

    if (newRows <= 0 || newCols <= 0) {
        throw std::runtime_error("The size of cropped matrix is non-positive!");
    }

    NppiSize srcSize = {newCols, newRows};
    NppiRect srcOffset = {cropPreCols, cropPreRows};
    int srcStep = cols * sizeof(Npp32fc);
    int dstStep = newCols * sizeof(Npp32fc);

    nppiCopy_32fc_C1R(reinterpret_cast<const Npp32fc*>(matrix) + srcOffset.y * cols + srcOffset.x, srcStep, reinterpret_cast<Npp32fc*>(matrix_new), dstStep, srcSize);
}

void CUDAUtils::padByConstant(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, float padValue)
{
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;

    NppiSize srcSize = {cols, rows};
    NppiSize dstSize = {newCols, newRows};
    int srcStep = cols * sizeof(float);
    int dstStep = newCols * sizeof(float);

    nppiCopyConstBorder_32f_C1R(reinterpret_cast<const Npp32f*>(matrix), srcStep, srcSize, reinterpret_cast<Npp32f*>(matrix_new), dstStep, dstSize, padRows, padCols, padValue);
}

void CUDAUtils::padByReplicate(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols)
{
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;

    NppiSize srcSize = {cols, rows};
    NppiSize dstSize = {newCols, newRows};
    int srcStep = cols * sizeof(float);
    int dstStep = newCols * sizeof(float);

    // Copy the original matrix to the center of the new matrix
    nppiCopyReplicateBorder_32f_C1R(reinterpret_cast<const Npp32f*>(matrix), srcStep, srcSize, reinterpret_cast<Npp32f*>(matrix_new), dstStep, dstSize, padRows, padCols);
}

void CUDAUtils::padByFadeout(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols)
{
    // Calculate the mean of the original matrix as the padding value
    thrust::device_ptr<float> ptr(matrix);
    float padValue = thrust::reduce(ptr, ptr + rows * cols) / (rows * cols);

    // Calculate the new size after padding and copy the original matrix to the center
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;
    padByReplicate(matrix, matrix_new, rows, cols, padRows, padCols);

    // Allocate memory for row and column grids and generate mask components
    float *rowGrid, *colGrid;
    cudaMalloc(&rowGrid, newRows * sizeof(float));
    cudaMalloc(&colGrid, newCols * sizeof(float));

    int blockSize = 1024;
    int numBlocks = (newRows + blockSize - 1) / blockSize;
    genMaskComponent<<<numBlocks, blockSize>>>(rowGrid, newRows, padRows);

    numBlocks = (newCols + blockSize - 1) / blockSize;
    genMaskComponent<<<numBlocks, blockSize>>>(colGrid, newCols, padCols);

    // Generate and apply gradient mask
    float *matrix_mask;
    cudaMalloc(&matrix_mask, newRows * newCols * sizeof(float));
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((newCols + blockSize2D.x - 1) / blockSize2D.x, (newRows + blockSize2D.y - 1) / blockSize2D.y);
    genMaskMatrix<<<numBlocks2D, blockSize2D>>>(matrix_mask, rowGrid, colGrid, newRows, newCols);

    numBlocks = (newRows * newCols + blockSize - 1) / blockSize;
    multiplyMaskMatrix<<<numBlocks, blockSize>>>(matrix_new, matrix_mask, newRows * newCols, padValue);

    cudaFree(matrix_mask);
    cudaFree(rowGrid);
    cudaFree(colGrid);
}

void CUDAUtils::padMatrix(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, PaddingType type, float padValue)
{
    if (padRows < 0 || padCols < 0) {
        throw std::invalid_argument("Padding size cannot be less than 0!");
    }

    switch (type) {
        case Constant: 
            padByConstant(matrix, matrix_new, rows, cols, padRows, padCols, padValue);
            break;
        case Replicate: 
            padByReplicate(matrix, matrix_new, rows, cols, padRows, padCols);
            break;
        case Fadeout: 
            padByFadeout(matrix, matrix_new, rows, cols, padRows, padCols);
            break;
        default:
            throw std::invalid_argument("Invalid padding type!");
    }
}

void CUDAPropKernel::genByFourier(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber)
{   
    float *rowRange;
    float *colRange;
    cudaMalloc(&rowRange, imSize[0] * sizeof(float));
    cudaMalloc(&colRange, imSize[1] * sizeof(float));

    FArray spacing(2, 1.0f);
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing);

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMalloc(&rowFreq, imSize[0] * sizeof(cuFloatComplex));
    cudaMalloc(&colFreq, imSize[1] * sizeof(cuFloatComplex));

    // Generate row and column components
    int blockSize1D = 512;
    int numBlocks1D = (imSize[0] + blockSize1D - 1) / blockSize1D;
    genFourierComponent<<<numBlocks1D, blockSize1D>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);

    numBlocks1D = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genFourierComponent<<<numBlocks1D, blockSize1D>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    genFourierKernel<<<numBlocks2D, blockSize2D>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1]);

    cudaFree(rowFreq);
    cudaFree(colFreq);
    cudaFree(rowRange);
    cudaFree(colRange);
}

void CUDAPropKernel::genByChirp(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber)
{
    float *rowRange;
    float *colRange;
    cudaMalloc(&rowRange, imSize[0] * sizeof(float));
    cudaMalloc(&colRange, imSize[1] * sizeof(float));

    FArray spacing {2.0f * M_PIf32 / imSize[0], 2.0f * M_PIf32 / imSize[1]};
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing);

    // Compute kernel initial coefficient
    std::complex<float> init = MathUtils::getInitCoeff(fresnelNumber);
    cuFloatComplex initCoeff = make_cuFloatComplex(init.real(), init.imag());

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMalloc(&rowFreq, imSize[0] * sizeof(cuFloatComplex));
    cudaMalloc(&colFreq, imSize[1] * sizeof(cuFloatComplex));
    
    // Generate row and column components
    int blockSize1D = 512;
    int numBlocks1D = (imSize[0] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<numBlocks1D, blockSize1D>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);

    numBlocks1D = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<numBlocks1D, blockSize1D>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    CUFFTUtils rowFFTUtils(imSize[0]);
    CUFFTUtils colFFTUtils(imSize[1]);
    rowFFTUtils.fft_fwd(rowFreq);
    colFFTUtils.fft_fwd(colFreq);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    genChirpKernel<<<numBlocks2D, blockSize2D>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1], initCoeff);

    cudaFree(rowFreq);
    cudaFree(colFreq);
    cudaFree(rowRange);
    cudaFree(colRange);
}

void CUDAPropKernel::genByChirpLimited(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber)
{
    float *rowRange;
    float *colRange;
    cudaMalloc(&rowRange, imSize[0] * sizeof(float));
    cudaMalloc(&colRange, imSize[1] * sizeof(float));

    FArray spacing {2.0f * M_PIf32 / imSize[0], 2.0f * M_PIf32 / imSize[1]};
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing);

    // Compute kernel initial coefficient
    std::complex<float> init = MathUtils::getInitCoeff(fresnelNumber);
    cuFloatComplex initCoeff = make_cuFloatComplex(init.real(), init.imag());

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMalloc(&rowFreq, imSize[0] * sizeof(cuFloatComplex));
    cudaMalloc(&colFreq, imSize[1] * sizeof(cuFloatComplex));

    int blockSize1D = 512;
    int rowBlocks = (imSize[0] + blockSize1D - 1) / blockSize1D;
    int colBlocks = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<rowBlocks, blockSize1D>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);
    genChirpComponent<<<colBlocks, blockSize1D>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    genChirpKernel<<<numBlocks2D, blockSize2D>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1], initCoeff);

    // Multiply by obliquity factor
    multiplyObliFactor_1<<<rowBlocks, blockSize1D>>>(rowRange, imSize[0], fresnelNumber[0]);
    multiplyObliFactor_1<<<colBlocks, blockSize1D>>>(colRange, imSize[1], fresnelNumber[1]);
    
    float *matrix;
    cudaMalloc(&matrix, imSize[0] * imSize[1] * sizeof(float));
    multiplyObliFactor_2<<<numBlocks2D, blockSize2D>>>(matrix, rowRange, colRange, imSize[0], imSize[1]);
    
    int blockSize = 1024;
    int numBlocks = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    multiplyObliFactor_3<<<numBlocks, blockSize>>>(kernel, matrix, imSize[0] * imSize[1]);

    CUFFTUtils fftUtils(imSize[0], imSize[1], 1);
    fftUtils.fft_fwd(kernel);

    cudaFree(matrix);
    cudaFree(rowFreq);
    cudaFree(colFreq);
    cudaFree(rowRange);
    cudaFree(colRange);
}

void CUDAPropKernel::generateKernel(cuFloatComplex* kernel, const IntArray &imSize, FArray &fresnelNumber, Type type)
{
    if (fresnelNumber.size() != 1 && fresnelNumber.size() != imSize.size()) {
        throw std::invalid_argument("Invalid Fresnel number!");
    }
    if (fresnelNumber.size() == 1) {
        fresnelNumber.push_back(fresnelNumber[0]);
    }

    switch (type)
    {
        case Fourier:
            genByFourier(kernel, imSize, fresnelNumber);
            break;
        case Chirp:
            genByChirp(kernel, imSize, fresnelNumber);
            break;
        case ChirpLimited:
            genByChirpLimited(kernel, imSize, fresnelNumber);
            break;
        default:
            throw std::invalid_argument("Invalid propagation kernel type!");
            break;
    }
}