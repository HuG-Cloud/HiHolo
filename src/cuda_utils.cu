#include "cuda_utils.h"

CUFFTUtils::CUFFTUtils(int in_numel): numel(in_numel)
{
    cufftPlan1d(&plan, numel, CUFFT_C2C, 1);
}

CUFFTUtils::CUFFTUtils(int in_rows, int in_cols, int in_batchSize):
rows(in_rows), cols(in_cols), batchSize(in_batchSize), numel(in_rows * in_cols)
{
    cufftPlan2d(&plan, cols, rows, CUFFT_C2C);
    int size[2] = {cols, rows};
    cufftPlanMany(&plan_batch, 2, size, nullptr, 1, numel, nullptr, 1, numel, CUFFT_C2C, batchSize);
}

void CUFFTUtils::fft_fwd(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan, complexWave, complexWave, CUFFT_FORWARD);
}

void CUFFTUtils::fft_bwd(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan, complexWave, complexWave, CUFFT_INVERSE);
    int blockSize = 1024;
    int numBlocks = (numel + blockSize - 1) / blockSize;
    scaleComplexData<<<numBlocks, blockSize>>>(complexWave, numel, 1.0f / numel);
}

void CUFFTUtils::fft_fwd_batch(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan_batch, complexWave, complexWave, CUFFT_FORWARD);
}

void CUFFTUtils::fft_bwd_batch(cuFloatComplex *complexWave)
{
    cufftExecC2C(plan_batch, complexWave, complexWave, CUFFT_INVERSE);
    int blockSize = 1024;
    int numBlocks = (numel * batchSize + blockSize - 1) / blockSize;
    scaleComplexData<<<numBlocks, blockSize>>>(complexWave, numel * batchSize, 1.0f / numel);
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

__global__ void scaleFloatData(float *data, int numel, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] *= scale;
    }
}

__global__ void computeComplexData(cuFloatComplex *complexData, const float *amplitude, const float *phase, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        complexData[idx] = make_cuFloatComplex(amplitude[idx] * cosf(phase[idx]), amplitude[idx] * sinf(phase[idx]));
    }
}

__global__ void computeAmplitude(const cuFloatComplex *complexWave, float *amplitude, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        amplitude[idx] = hypotf(complexWave[idx].x, complexWave[idx].y);
    }
}

__global__ void computePhase(const cuFloatComplex *complexWave, float *phase, int numel)
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

__global__ void adjustPhase(float *phase, float maxPhase, float minPhase, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if (maxPhase < INFINITY) {
            phase[idx] = min(phase[idx], maxPhase);
        }
        if (minPhase > -INFINITY) {
            phase[idx] = max(phase[idx], minPhase);
        }
    }
}

__global__ void adjustComplexWave(cuFloatComplex *complexWave, const float *support, float outsideValue, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if (support[idx] == 0) {
            complexWave[idx] = make_cuFloatComplex(outsideValue, 0.0f);
        }
    }
}

__global__ void sqrtIntensity(float *amplitude, int numel)
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

__global__ void subNormFloat(float *result, const float *data1, const float *data2, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        result[idx] = powf(data1[idx] - data2[idx], 2);
    }
}

__global__ void subNormComplex(float *result, const cuFloatComplex *cmpData1, const cuFloatComplex *cmpData2, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        cuFloatComplex tmp = cuCsubf(cmpData1[idx], cmpData2[idx]);
        result[idx] = tmp.x * tmp.x + tmp.y * tmp.y;
    }
}

__global__ void multiplyWaveField(cuFloatComplex *result, const cuFloatComplex *wf1, const cuFloatComplex *wf2, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        result[idx] = cuCmulf(wf1[idx], wf2[idx]);
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

__global__ void updateDM(cuFloatComplex *probe, const cuFloatComplex *probeWave, const cuFloatComplex *complexWave, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float intensity = complexWave[idx].x * complexWave[idx].x + complexWave[idx].y * complexWave[idx].y;
        probe[idx] = cuCdivf(cuCmulf(probeWave[idx], cuConjf(complexWave[idx])), make_cuFloatComplex(intensity, 0.0f));
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
    extern __shared__ float shared_data[];
    float *shared_row = shared_data;
    float *shared_col = &shared_data[blockDim.y];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0 && row < rows) {
        shared_row[threadIdx.y] = rowGrid[row];
    }
    if (threadIdx.y == 0 && col < cols) {
        shared_col[threadIdx.x] = colGrid[col];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        mask[idx] = (1.0f + cosf(shared_row[threadIdx.y])) * (0.25f * (1.0f + cosf(shared_col[threadIdx.x])));
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

    extern __shared__ float shared_data[];
    float* shared_row = shared_data;
    float* shared_col = &shared_data[blockDim.y];

    if (threadIdx.x == 0 && row < rows) {
        shared_row[threadIdx.y] = rowRange[row];
    }
    if (threadIdx.y == 0 && col < cols) {
        shared_col[threadIdx.x] = colRange[col];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] = shared_row[threadIdx.y] + shared_col[threadIdx.x];
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

    extern __shared__ cuFloatComplex shared_mem[];
    cuFloatComplex* shared_row = shared_mem;
    cuFloatComplex* shared_col = &shared_mem[blockDim.y];

    if (threadIdx.x == 0 && row < rows) {
        shared_row[threadIdx.y] = rowComponent[row];
    }
    if (threadIdx.y == 0 && col < cols) {
        shared_col[threadIdx.x] = colComponent[col];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        kernel[idx] = cuCmulf(shared_row[threadIdx.y], shared_col[threadIdx.x]);
    }
}

__global__ void genChirpKernel(cuFloatComplex *kernel, cuFloatComplex *rowComponent, cuFloatComplex *colComponent, int rows, int cols, cuFloatComplex initCoeff)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ cuFloatComplex shared_mem[];
    cuFloatComplex* shared_row = shared_mem;
    cuFloatComplex* shared_col = &shared_mem[blockDim.y];

    if (threadIdx.x == 0 && row < rows) {
        shared_row[threadIdx.y] = rowComponent[row];
    }
    if (threadIdx.y == 0 && col < cols) {
        shared_col[threadIdx.x] = colComponent[col];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        kernel[idx] = cuCmulf(cuCmulf(shared_row[threadIdx.y], shared_col[threadIdx.x]), initCoeff);
    }
}

__global__ void genCTFComponent(float *component, const float *range, float fresnelNumber, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        component[idx] = powf(range[idx], 2) / (4.0f * M_PIf32 * fresnelNumber);
    }
}

__global__ void computeCTF(float *data, float ratio, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float tmp = sinf(data[idx]);
        if (ratio > 0) {
            tmp += cosf(data[idx]) * ratio;
        }
        data[idx] = tmp;
    }
}

__global__ void CTFMultiplyHologram(cuFloatComplex *hologram, const float *data, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        hologram[idx] = cuCmulf(hologram[idx], make_cuFloatComplex(data[idx], 0.0f));
    }
}

__global__ void addSquareData(float *data, const float *newData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] += powf(newData[idx], 2);
    }
}

__global__ void subtractConstant(cuFloatComplex *data, const float constant, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = cuCsubf(data[idx], make_cuFloatComplex(constant, 0.0f));
    }
}

__global__ void genRegComponent(float *component, float fresnelNumber, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        component[idx] = powf(component[idx], 2) / (2.0f * powf(M_PIf32, 2) * fresnelNumber);
    }
}

__device__ float erfc_approx(float x) {
    // 使用 Abramowitz 和 Stegun 提供的近似公式
    float t = 1.0f / (1.0f + 0.5f * fabsf(x));
    float tau = t * expf(-x * x - 1.26551223f + t * (1.00002368f + t * (0.37409196f + 
                t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + 
                t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + 
                t * 0.17087277f)))))))));
    return x >= 0.0f ? tau : 2.0f - tau;
}

__global__ void computeErfcWeights(float* data, float param, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float tmp = (sqrtf(data[idx]) - 1.0f) / param;
        data[idx] = 0.5f * erfc_approx(tmp);
    }
}

__global__ void genRegWeights(float* data, float lim1, float lim2, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = lim1 * data[idx] + lim2 * (1.0f - data[idx]);
    }
}

__global__ void addFloatData(float* data, const float* newData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] += newData[idx];
    }
}

__global__ void divideFloatData(float* data, const float* newData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] /= newData[idx];
    }
}

__global__ void complexDivideFloat(cuFloatComplex* data, const float* newData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float denominator = newData[idx] + 1e-10f;
        data[idx] = cuCdivf(data[idx], make_cuFloatComplex(denominator, 0.0f));
    }
}

__global__ void extractRealData(const cuFloatComplex* data, float* realData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        realData[idx] = data[idx].x;
    }
}

__global__ void floatToComplex(const float* data, cuFloatComplex* complexData, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        complexData[idx] = make_cuFloatComplex(data[idx], 0.0f);
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

void CUDAUtils::genFFTFreq(float* rowRange, float* colRange, const IntArray &imSize, FArray &spacing, cudaStream_t stream)
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
    genShiftedFFTFreq<<<numBlocks, blockSize, 0, stream>>>(rowRange, rows, spacing[0]);

    numBlocks = (cols + blockSize - 1) / blockSize;
    genShiftedFFTFreq<<<numBlocks, blockSize, 0, stream>>>(colRange, cols, spacing[1]);
}

void CUDAUtils::cropMatrix(const cuFloatComplex* matrix, cuFloatComplex* matrix_new, int rows, int cols, int cropPreRows, int cropPreCols, int cropPostRows, int cropPostCols)
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

void CUDAUtils::cropMatrix(const float* matrix, float* matrix_new, int rows, int cols, int cropPreRows, int cropPreCols, int cropPostRows, int cropPostCols)
{
    int newRows = rows - cropPreRows - cropPostRows;
    int newCols = cols - cropPreCols - cropPostCols;

    if (newRows <= 0 || newCols <= 0) {
        throw std::runtime_error("The size of cropped matrix is non-positive!");
    }
    
    NppiSize srcSize = {newCols, newRows};
    NppiRect srcOffset = {cropPreCols, cropPreRows};
    int srcStep = cols * sizeof(Npp32f);
    int dstStep = newCols * sizeof(Npp32f);

    nppiCopy_32f_C1R(reinterpret_cast<const Npp32f*>(matrix) + srcOffset.y * cols + srcOffset.x, srcStep, reinterpret_cast<Npp32f*>(matrix_new), dstStep, srcSize);
}

void CUDAUtils::padByConstant(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, float padValue, cudaStream_t stream)
{
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;

    NppiSize srcSize = {cols, rows};
    NppiSize dstSize = {newCols, newRows};
    int srcStep = cols * sizeof(float);
    int dstStep = newCols * sizeof(float);

    NppStreamContext streamCtx;
    streamCtx.hStream = stream;

    nppiCopyConstBorder_32f_C1R_Ctx(reinterpret_cast<const Npp32f*>(matrix), srcStep, srcSize, reinterpret_cast<Npp32f*>(matrix_new),
                                    dstStep, dstSize, padRows, padCols, padValue, streamCtx);
}

void CUDAUtils::padByReplicate(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, cudaStream_t stream)
{
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;

    NppiSize srcSize = {cols, rows};
    NppiSize dstSize = {newCols, newRows};
    int srcStep = cols * sizeof(float);
    int dstStep = newCols * sizeof(float);

    NppStreamContext streamCtx;
    streamCtx.hStream = stream;

    // Copy the original matrix to the center of the new matrix
    nppiCopyReplicateBorder_32f_C1R_Ctx(reinterpret_cast<const Npp32f*>(matrix), srcStep, srcSize,
                                        reinterpret_cast<Npp32f*>(matrix_new), dstStep, dstSize, padRows, padCols, streamCtx);
}

void CUDAUtils::padByFadeout(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, cudaStream_t stream)
{
    // Calculate the mean of the original matrix as the padding value
    thrust::device_ptr<float> ptr(matrix);
    float padValue = thrust::reduce(thrust::cuda::par.on(stream), ptr, ptr + rows * cols) / (rows * cols);

    // Calculate the new size after padding and copy the original matrix to the center
    int newRows = rows + 2 * padRows;
    int newCols = cols + 2 * padCols;
    padByReplicate(matrix, matrix_new, rows, cols, padRows, padCols, stream);

    // Allocate memory for row and column grids and generate mask components
    float *rowGrid, *colGrid;
    cudaMallocAsync(&rowGrid, newRows * sizeof(float), stream);
    cudaMallocAsync(&colGrid, newCols * sizeof(float), stream);

    int blockSize = 1024;
    int numBlocks = (newRows + blockSize - 1) / blockSize;
    genMaskComponent<<<numBlocks, blockSize, 0, stream>>>(rowGrid, newRows, padRows);

    numBlocks = (newCols + blockSize - 1) / blockSize;
    genMaskComponent<<<numBlocks, blockSize, 0, stream>>>(colGrid, newCols, padCols);

    // Generate and apply gradient mask
    float *matrix_mask;
    cudaMallocAsync(&matrix_mask, newRows * newCols * sizeof(float), stream);
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((newCols + blockSize2D.x - 1) / blockSize2D.x, 
                     (newRows + blockSize2D.y - 1) / blockSize2D.y);
    size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(float);
    genMaskMatrix<<<numBlocks2D, blockSize2D, sharedMemSize, stream>>>(matrix_mask, rowGrid, colGrid, newRows, newCols);

    numBlocks = (newRows * newCols + blockSize - 1) / blockSize;
    multiplyMaskMatrix<<<numBlocks, blockSize, 0, stream>>>(matrix_new, matrix_mask, newRows * newCols, padValue);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float* matrix_mask = static_cast<float*>(userData);
        cudaFreeAsync(matrix_mask, stream);
    }, matrix_mask, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float* rowGrid = static_cast<float*>(userData);
        cudaFreeAsync(rowGrid, stream);
    }, rowGrid, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float* colGrid = static_cast<float*>(userData);
        cudaFreeAsync(colGrid, stream);
    }, colGrid, 0);
}

void CUDAUtils::padMatrix(float* matrix, float* matrix_new, int rows, int cols, int padRows, int padCols, PaddingType type, float padValue, cudaStream_t stream)
{
    if (padRows < 0 || padCols < 0) {
        throw std::invalid_argument("Padding size cannot be less than 0!");
    }

    switch (type) {
        case Constant: 
            padByConstant(matrix, matrix_new, rows, cols, padRows, padCols, padValue, stream);
            break;
        case Replicate: 
            padByReplicate(matrix, matrix_new, rows, cols, padRows, padCols, stream);
            break;
        case Fadeout: 
            padByFadeout(matrix, matrix_new, rows, cols, padRows, padCols, stream);
            break;
        default:
            throw std::invalid_argument("Invalid padding type!");
    }
}

void CUDAUtils::copyBatchMatrix(float* l_data, const float* s_data, int l_rows, int l_cols, int s_rows, int s_cols, int batchSize, int start_row, int start_col)
{
    for (int i = 0; i < batchSize; i++) {
        int l_offset = i * l_rows * l_cols + start_row * l_cols + start_col;
        int s_offset = i * s_rows * s_cols;

        cudaMemcpy2D(reinterpret_cast<void*>(l_data + l_offset), l_cols * sizeof(float), 
                     reinterpret_cast<void*>(const_cast<float*>(s_data) + s_offset), s_cols * sizeof(float),
                     s_cols * sizeof(float), s_rows, cudaMemcpyDeviceToDevice);
    }
}

float* CUDAUtils::padInputData(float* inputData, const IntArray& imSize, const IntArray& padSize, PaddingType padType, float padValue)
{
    if (padSize.empty()) {
        return inputData; // Un-padded case, return original data
    }
    
    float* paddedData;
    IntArray newSize {imSize[0] + 2 * padSize[0], imSize[1] + 2 * padSize[1]};
    cudaMalloc((void**)&paddedData, newSize[0] * newSize[1] * sizeof(float));
    CUDAUtils::padMatrix(inputData, paddedData, imSize[0], imSize[1], padSize[0], padSize[1], padType, padValue);
        
    return paddedData;
}

float CUDAUtils::computeL2Norm(const float* data1, const float* data2, int numel)
{
    float *temp;
    cudaMalloc((void**)&temp, sizeof(float) * numel);

    int blockSize = 1024;
    int numBlocks = (numel + blockSize - 1) / blockSize;
    subNormFloat<<<numBlocks, blockSize>>>(temp, data1, data2, numel);
    
    thrust::device_ptr<float> dev_ptr(temp);
    float sqSum = thrust::reduce(dev_ptr, dev_ptr + numel, 0.0f, thrust::plus<float>());
    cudaFree(temp);
    return std::sqrt(sqSum);
}

void CUDAUtils::ctf_recons_kernel(const float *holograms, float *result, const IntArray &imSize, int numImages,
                                  const F2DArray &fresnelNumbers, float betaDeltaRatio, float *regWeights)
{
        // 分配GPU内存并生成频率网格
        float *rowRange, *colRange;
        cudaMalloc((void**)&rowRange, imSize[0] * sizeof(float));
        cudaMalloc((void**)&colRange, imSize[1] * sizeof(float));
        FArray spacing(2, 1.0f);
        CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing);

        // 为CTF相关数据分配GPU内存
        cuFloatComplex *CTFHolograms;
        float *CTFSq;
        cudaMalloc((void**)&CTFHolograms, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
        cudaMalloc((void**)&CTFSq, imSize[0] * imSize[1] * sizeof(float));

        float *rowComponent, *colComponent;
        cudaMalloc((void**)&rowComponent, imSize[0] * sizeof(float));
        cudaMalloc((void**)&colComponent, imSize[1] * sizeof(float));
        float *tempCTF;
        cuFloatComplex *tempHologram;
        cudaMalloc((void**)&tempCTF, imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&tempHologram, imSize[0] * imSize[1] * sizeof(cuFloatComplex));

        CUFFTUtils cufftUtils(imSize[0], imSize[1], 1);

        // 定义不同内核的网格和块大小
        int blockSize1D = 512;
        int gridRowSize1D = (imSize[0] + blockSize1D - 1) / blockSize1D;
        int gridColSize1D = (imSize[1] + blockSize1D - 1) / blockSize1D;
        dim3 blockSize2D(32, 32);
        dim3 gridSize2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                        (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
        size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(float);
        int blockSize = 1024;
        int gridSize = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;

        // 初始化CTF数据
        initializeData<<<gridSize, blockSize>>>(CTFHolograms, make_cuFloatComplex(0.0f, 0.0f), imSize[0] * imSize[1]);
        initializeData<<<gridSize, blockSize>>>(CTFSq, 0.0f, imSize[0] * imSize[1]);
        
        // CTF重建主循环
        for (size_t i = 0; i < numImages; i++) {
            // 计算CTF传递函数
            genCTFComponent<<<gridRowSize1D, blockSize1D>>>(rowComponent, rowRange, fresnelNumbers[i][0], imSize[0]);
            genCTFComponent<<<gridColSize1D, blockSize1D>>>(colComponent, colRange, fresnelNumbers[i][1], imSize[1]);
            multiplyObliFactor_2<<<gridSize2D, blockSize2D, sharedMemSize>>>(tempCTF, rowComponent, colComponent, imSize[0], imSize[1]);
            computeCTF<<<gridSize, blockSize>>>(tempCTF, betaDeltaRatio, imSize[0] * imSize[1]);
            
            // 将全息图的FFT乘以CTF传递函数
            floatToComplex<<<gridSize, blockSize>>>(holograms + i * imSize[0] * imSize[1], tempHologram, imSize[0] * imSize[1]);
            cufftUtils.fft_fwd(tempHologram);
            CTFMultiplyHologram<<<gridSize, blockSize>>>(tempHologram, tempCTF, imSize[0] * imSize[1]);
            addWaveField<<<gridSize, blockSize>>>(CTFHolograms, tempHologram, imSize[0] * imSize[1]);
            addSquareData<<<gridSize, blockSize>>>(CTFSq, tempCTF, imSize[0] * imSize[1]);
        }
        
        scaleFloatData<<<gridSize, blockSize>>>(CTFSq, imSize[0] * imSize[1], 2.0f);
        // 傅里叶变换零频率的校正
        subtractConstant<<<1, 1>>>(CTFHolograms, imSize[0] * imSize[1] * numImages * betaDeltaRatio, 1);

        // 应用正则化权重
        addFloatData<<<gridSize, blockSize>>>(regWeights, CTFSq, imSize[0] * imSize[1]);

        // 最终计算和逆傅里叶变换
        complexDivideFloat<<<gridSize, blockSize>>>(CTFHolograms, regWeights, imSize[0] * imSize[1]);
        cufftUtils.fft_bwd(CTFHolograms);
        extractRealData<<<gridSize, blockSize>>>(CTFHolograms, result, imSize[0] * imSize[1]);
        
        // 释放临时内存
        cudaFree(rowComponent); cudaFree(colComponent); 
        cudaFree(rowRange); cudaFree(colRange);
        cudaFree(CTFHolograms); cudaFree(CTFSq);
        cudaFree(tempCTF); cudaFree(tempHologram);
}

float CUDAUtils::computeL2Norm(const cuFloatComplex* cmplxData1, const cuFloatComplex* cmplxData2, int numel)
{
    float *temp;
    cudaMalloc((void**)&temp, sizeof(float) * numel);

    int blockSize = 1024;
    int numBlocks = (numel + blockSize - 1) / blockSize;
    subNormComplex<<<numBlocks, blockSize>>>(temp, cmplxData1, cmplxData2, numel);

    thrust::device_ptr<float> dev_ptr(temp);
    float sqSum = thrust::reduce(dev_ptr, dev_ptr + numel, 0.0f, thrust::plus<float>());
    cudaFree(temp);
    return std::sqrt(sqSum);
}

void CUDAUtils::ctfRegWeights(float *regWeights, const IntArray &imSize, const FArray &fresnelNumber, float lowFreqLim, float highFreqLim)
{
    float *rowRange, *colRange;
    cudaMalloc((void**)&rowRange, imSize[0] * sizeof(float));
    cudaMalloc((void**)&colRange, imSize[1] * sizeof(float));
    FArray spacing(2, 1.0f);
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing);
    
    int blockSize = 512;
    int gridRowSize = (imSize[0] + blockSize - 1) / blockSize;
    int gridColSize = (imSize[1] + blockSize - 1) / blockSize;
    genRegComponent<<<gridRowSize, blockSize>>>(rowRange, fresnelNumber[0], imSize[0]);
    genRegComponent<<<gridColSize, blockSize>>>(colRange, fresnelNumber[1], imSize[1]);

    dim3 blockSize2D(32, 32);
    dim3 gridSize2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                     (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(float);
    multiplyObliFactor_2<<<gridSize2D, blockSize2D, sharedMemSize>>>(regWeights, rowRange, colRange, imSize[0], imSize[1]);

    float sigma = std::sqrt(2.0f) * (std::sqrt(2.0f) - 1.0f) / 2.0f;
    blockSize = 1024;
    int numBlocks = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    computeErfcWeights<<<numBlocks, blockSize>>>(regWeights, sigma, imSize[0] * imSize[1]);
    genRegWeights<<<numBlocks, blockSize>>>(regWeights, lowFreqLim, highFreqLim, imSize[0] * imSize[1]);

    cudaFree(rowRange);
    cudaFree(colRange);
}

void CUDAPropKernel::genByFourier(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber, cudaStream_t stream)
{
    float *rowRange;
    float *colRange;
    cudaMallocAsync(&rowRange, imSize[0] * sizeof(float), stream);
    cudaMallocAsync(&colRange, imSize[1] * sizeof(float), stream);

    FArray spacing(2, 1.0f);
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing, stream);

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMallocAsync(&rowFreq, imSize[0] * sizeof(cuFloatComplex), stream);
    cudaMallocAsync(&colFreq, imSize[1] * sizeof(cuFloatComplex), stream);

    // Generate row and column components
    int blockSize1D = 512;
    int numBlocks1D = (imSize[0] + blockSize1D - 1) / blockSize1D;
    genFourierComponent<<<numBlocks1D, blockSize1D, 0, stream>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);

    numBlocks1D = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genFourierComponent<<<numBlocks1D, blockSize1D, 0, stream>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                     (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(cuFloatComplex);
    genFourierKernel<<<numBlocks2D, blockSize2D, sharedMemSize, stream>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1]);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *rowRange = static_cast<float*>(userData);
        cudaFreeAsync(rowRange, stream);
    }, rowRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *colRange = static_cast<float*>(userData);
        cudaFreeAsync(colRange, stream);
    }, colRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *rowFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(rowFreq, stream);
    }, rowFreq, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *colFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(colFreq, stream);
    }, colFreq, 0);
}

void CUDAPropKernel::genByChirp(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber, cudaStream_t stream)
{
    float *rowRange;
    float *colRange;
    cudaMallocAsync(&rowRange, imSize[0] * sizeof(float), stream);
    cudaMallocAsync(&colRange, imSize[1] * sizeof(float), stream);

    FArray spacing {2.0f * M_PIf32 / imSize[0], 2.0f * M_PIf32 / imSize[1]};
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing, stream);

    // Compute kernel initial coefficient
    std::complex<float> init = MathUtils::getInitCoeff(fresnelNumber);
    cuFloatComplex initCoeff = make_cuFloatComplex(init.real(), init.imag());

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMallocAsync(&rowFreq, imSize[0] * sizeof(cuFloatComplex), stream);
    cudaMallocAsync(&colFreq, imSize[1] * sizeof(cuFloatComplex), stream);
    
    // Generate row and column components
    int blockSize1D = 512;
    int numBlocks1D = (imSize[0] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<numBlocks1D, blockSize1D, 0, stream>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);

    numBlocks1D = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<numBlocks1D, blockSize1D, 0, stream>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    // Execute FFT in parallel
    CUFFTUtils rowFFTUtils(imSize[0]);
    CUFFTUtils colFFTUtils(imSize[1]);
    cufftSetStream(rowFFTUtils.getPlan(), stream);
    cufftSetStream(colFFTUtils.getPlan(), stream);
    rowFFTUtils.fft_fwd(rowFreq);
    colFFTUtils.fft_fwd(colFreq);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                     (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(cuFloatComplex);
    genChirpKernel<<<numBlocks2D, blockSize2D, sharedMemSize, stream>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1], initCoeff);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *rowRange = static_cast<float*>(userData);
        cudaFreeAsync(rowRange, stream);
    }, rowRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *colRange = static_cast<float*>(userData);
        cudaFreeAsync(colRange, stream);
    }, colRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *rowFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(rowFreq, stream);
    }, rowFreq, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *colFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(colFreq, stream);
    }, colFreq, 0);
}

void CUDAPropKernel::genByChirpLimited(cuFloatComplex* kernel, const IntArray &imSize, const FArray &fresnelNumber, cudaStream_t stream)
{
    float *rowRange;
    float *colRange;
    cudaMallocAsync(&rowRange, imSize[0] * sizeof(float), stream);
    cudaMallocAsync(&colRange, imSize[1] * sizeof(float), stream);

    FArray spacing {2.0f * M_PIf32 / imSize[0], 2.0f * M_PIf32 / imSize[1]};
    CUDAUtils::genFFTFreq(rowRange, colRange, imSize, spacing, stream);

    // Compute kernel initial coefficient
    std::complex<float> init = MathUtils::getInitCoeff(fresnelNumber);
    cuFloatComplex initCoeff = make_cuFloatComplex(init.real(), init.imag());

    cuFloatComplex *rowFreq;
    cuFloatComplex *colFreq;
    cudaMallocAsync(&rowFreq, imSize[0] * sizeof(cuFloatComplex), stream);
    cudaMallocAsync(&colFreq, imSize[1] * sizeof(cuFloatComplex), stream);

    int blockSize1D = 512;
    int rowBlocks = (imSize[0] + blockSize1D - 1) / blockSize1D;
    int colBlocks = (imSize[1] + blockSize1D - 1) / blockSize1D;
    genChirpComponent<<<rowBlocks, blockSize1D, 0, stream>>>(rowFreq, rowRange, imSize[0], fresnelNumber[0]);
    genChirpComponent<<<colBlocks, blockSize1D, 0, stream>>>(colFreq, colRange, imSize[1], fresnelNumber[1]);

    // Generate kernel
    dim3 blockSize2D(32, 32);
    dim3 numBlocks2D((imSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                     (imSize[0] + blockSize2D.y - 1) / blockSize2D.y);
    size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(cuFloatComplex);
    genChirpKernel<<<numBlocks2D, blockSize2D, sharedMemSize, stream>>>(kernel, rowFreq, colFreq, imSize[0], imSize[1], initCoeff);

    // Multiply by obliquity factor
    multiplyObliFactor_1<<<rowBlocks, blockSize1D, 0, stream>>>(rowRange, imSize[0], fresnelNumber[0]);
    multiplyObliFactor_1<<<colBlocks, blockSize1D, 0, stream>>>(colRange, imSize[1], fresnelNumber[1]);
    
    float *matrix;
    cudaMallocAsync(&matrix, imSize[0] * imSize[1] * sizeof(float), stream);
    sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(float);
    multiplyObliFactor_2<<<numBlocks2D, blockSize2D, sharedMemSize, stream>>>(matrix, rowRange, colRange, imSize[0], imSize[1]);
    
    int blockSize = 1024;
    int numBlocks = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;
    multiplyObliFactor_3<<<numBlocks, blockSize, 0, stream>>>(kernel, matrix, imSize[0] * imSize[1]);

    CUFFTUtils fftUtils(imSize[0], imSize[1], 1);
    cufftSetStream(fftUtils.getPlan(), stream);
    fftUtils.fft_fwd(kernel);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *matrix = static_cast<float*>(userData);
        cudaFreeAsync(matrix, stream);
    }, matrix, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *rowRange = static_cast<float*>(userData);
        cudaFreeAsync(rowRange, stream);
    }, rowRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        float *colRange = static_cast<float*>(userData);
        cudaFreeAsync(colRange, stream);
    }, colRange, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *rowFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(rowFreq, stream);
    }, rowFreq, 0);

    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cuFloatComplex *colFreq = static_cast<cuFloatComplex*>(userData);
        cudaFreeAsync(colFreq, stream);
    }, colFreq, 0);
}

void CUDAPropKernel::generateKernel(cuFloatComplex* kernel, const IntArray &imSize, FArray &fresnelNumber, Type type, cudaStream_t stream)
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
            genByFourier(kernel, imSize, fresnelNumber, stream);
            break;
        case Chirp:
            genByChirp(kernel, imSize, fresnelNumber, stream);
            break;
        case ChirpLimited:
            genByChirpLimited(kernel, imSize, fresnelNumber, stream);
            break;
        default:
            throw std::invalid_argument("Invalid propagation kernel type!");
            break;
    }

}

__global__ void displayMatrix(const float* matrix, int rows, int cols) 
{
    // Print the matrix using only one thread
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("\nMatrix %dx%d:\n", rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%8.4f ", matrix[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void displayComplexMatrix(const cuFloatComplex* matrix, int rows, int cols) 
{
    // Print the matrix using only one thread
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("\nComplex Matrix %dx%d:\n", rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                if (matrix[idx].y >= 0) {
                    printf("%6.3f+%6.3fi ", matrix[idx].x, matrix[idx].y);
                } else {
                    printf("%6.3f%6.3fi ", matrix[idx].x, matrix[idx].y);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}