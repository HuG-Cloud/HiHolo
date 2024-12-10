#include "holo_recons.h"

namespace PhaseRetrieval
{
    // void preprocess_data(const std::string &inFileName, const std::string &inDatasetName, std::vector<hsize_t> &dims,
    //                          const std::string &outFileName, const std::string &outDatasetName, int kernelSize, float threshold,
    //                          bool applyFilter, int rangeRows, int rangeCols, int movmeanSize, const std::string &method)
    // {
    //     if (inFileName.empty() || inDatasetName.empty() || dims.size() != 3 || outFileName.empty() || outDatasetName.empty()) {
    //         throw std::invalid_argument("Empty file name, dataset name or invalid dimensions!");
    //     }
        
    //     // Read original data from HDF5 file
    //     std::vector<uint16_t> originalData;
    //     if (IOUtils::readRawData(inFileName, inDatasetName, originalData, dims))
    //         std::cout << "Read experimental data successfully!" << std::endl;

    //     // Convert vector(uint16) to cv::Mats(float) and remove outliers
    //     std::vector<cv::Mat> mats = ImageUtils::convertVecToMats(originalData, dims[0], dims[1], dims[2]);
    //     for (auto &mat: mats) {
    //         ImageUtils::removeOutliers(mat, kernelSize, threshold);
    //     }

    //     // Align multi-distance images and calculate holograms
    //     for (int i = 1; i < dims[0]; i++) {
    //         ImageUtils::alignImages(mats[i], mats[0], applyFilter, kernelSize);
    //     }
    //     /* (data - dark) ./ flat */

    //     for (auto &mat: mats) {
    //         ImageUtils::removeStripes(mat, rangeRows, rangeCols, movmeanSize, method);
    //     }

    //     // Convert cv::Mats(float) to D2DArray and save to HDF5 file
    //     D2DArray holograms = ImageUtils::convertMatsToVec(mats);
    //     if (IOUtils::saveProcessedGrams(outFileName, outDatasetName, holograms, dims[1], dims[2]))
    //         std::cout << "Process and save holograms successfully!" << std::endl;

    // }

    FArray reconstruct_ctf(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelnumbers, float lowFreqLim,
                           float highFreqLim, float betaDeltaRatio, const IntArray &padSize, CUDAUtils::PaddingType padType)
    {
        // Add GPU environment check
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA capable GPU device found!");
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Using GPU device: " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

        /* Check correctness and compatibility of measurements and fresnel numbers */
        if (holograms.empty() || imSize.empty())
            throw std::invalid_argument("Invalid measurement data or image size!");
  
        if (fresnelnumbers.size() != numImages)
            throw std::invalid_argument("The number of images and fresnel numbers does not match!");

        F2DArray fresnelNumbers = fresnelnumbers;
        for (auto &fresnelNumber: fresnelNumbers) {
            if (fresnelNumber.size() != 1 && fresnelNumber.size() != imSize.size()) {
                throw std::invalid_argument("Invalid Fresnel number!");
            }
            if (fresnelNumber.size() == 1) {
                fresnelNumber.push_back(fresnelNumber[0]);
            }
        }
        
        // Set low frequency limit to zero if the ratio is too large
        if (lowFreqLim < (2.0f * numImages * std::pow(betaDeltaRatio, 2.0f))) {
            lowFreqLim = 0.0f;
        }
        
        // transfer holograms to GPU
        float *holograms_gpu;
        cudaMalloc((void**)&holograms_gpu, holograms.size() * sizeof(float));
        cudaMemcpy(holograms_gpu, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);

        IntArray newSize(imSize);
        // Optional padding operations on holograms
        if (!padSize.empty()) {
            newSize[0] += 2 * padSize[0];
            newSize[1] += 2 * padSize[1];
            float *paddedHolograms_gpu;
            cudaMalloc((void**)&paddedHolograms_gpu, newSize[0] * newSize[1] * numImages * sizeof(float));

            cudaStream_t *streams = new cudaStream_t[numImages];
            for (int i = 0; i < numImages; i++) {
                cudaStreamCreate(&streams[i]);
                CUDAUtils::padMatrix(holograms_gpu + i * imSize[0] * imSize[1], paddedHolograms_gpu + i * newSize[0] * newSize[1],
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType, 0.0f, streams[i]);
            }

            for (int i = 0; i < numImages; i++) {
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
            
            float *temp = holograms_gpu;
            holograms_gpu = paddedHolograms_gpu;
            cudaFree(temp);
        }

        // Allocate GPU memory and generate frequency grid
        float *rowRange, *colRange;
        cudaMalloc((void**)&rowRange, newSize[0] * sizeof(float));
        cudaMalloc((void**)&colRange, newSize[1] * sizeof(float));
        FArray spacing(2, 1.0f);
        CUDAUtils::genFFTFreq(rowRange, colRange, newSize, spacing);

        std::cout << "Choosing Algorithm: CTF" << std::endl;

        // Allocate GPU memory for CTF related data
        cuFloatComplex *CTFHolograms;
        float *CTFSq;
        cudaMalloc((void**)&CTFHolograms, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
        cudaMalloc((void**)&CTFSq, newSize[0] * newSize[1] * sizeof(float));

        float *rowComponent, *colComponent;
        cudaMalloc((void**)&rowComponent, newSize[0] * sizeof(float));
        cudaMalloc((void**)&colComponent, newSize[1] * sizeof(float));
        float *tempCTF;
        cuFloatComplex *tempHologram;
        cudaMalloc((void**)&tempCTF, newSize[0] * newSize[1] * sizeof(float));
        cudaMalloc((void**)&tempHologram, newSize[0] * newSize[1] * sizeof(cuFloatComplex));

        cufftHandle plan;
        cufftPlan2d(&plan, newSize[1], newSize[0], CUFFT_C2C);

        // Define grid and block sizes for different kernels
        int blockSize1D = 512;
        int gridRowSize1D = (newSize[0] + blockSize1D - 1) / blockSize1D;
        int gridColSize1D = (newSize[1] + blockSize1D - 1) / blockSize1D;
        dim3 blockSize2D(32, 32);
        dim3 gridSize2D((newSize[1] + blockSize2D.x - 1) / blockSize2D.x, 
                        (newSize[0] + blockSize2D.y - 1) / blockSize2D.y);
        size_t sharedMemSize = (blockSize2D.x + blockSize2D.y) * sizeof(float);
        int blockSize = 1024;
        int gridSize = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;

        // Initialize CTF data
        initializeData<<<gridSize, blockSize>>>(CTFHolograms, make_cuFloatComplex(0.0f, 0.0f), newSize[0] * newSize[1]);
        initializeData<<<gridSize, blockSize>>>(CTFSq, 0.0f, newSize[0] * newSize[1]);
        
        // CTF reconstruction main loop
        for (size_t i = 0; i < numImages; i++) {
            // Compute CTF transfer function
            genCTFComponent<<<gridRowSize1D, blockSize1D>>>(rowComponent, rowRange, fresnelNumbers[i][0], newSize[0]);
            genCTFComponent<<<gridColSize1D, blockSize1D>>>(colComponent, colRange, fresnelNumbers[i][1], newSize[1]);
            multiplyObliFactor_2<<<gridSize2D, blockSize2D, sharedMemSize>>>(tempCTF, rowComponent, colComponent, newSize[0], newSize[1]);
            computeCTF<<<gridSize, blockSize>>>(tempCTF, betaDeltaRatio, newSize[0] * newSize[1]);
            
            // Multiply FFT of holograms by CTF transfer function
            floatToComplex<<<gridSize, blockSize>>>(holograms_gpu + i * newSize[0] * newSize[1], tempHologram, newSize[0] * newSize[1]);
            cufftExecC2C(plan, tempHologram, tempHologram, CUFFT_FORWARD);
            CTFMultiplyHologram<<<gridSize, blockSize>>>(tempHologram, tempCTF, newSize[0] * newSize[1]);
            addWaveField<<<gridSize, blockSize>>>(CTFHolograms, tempHologram, newSize[0] * newSize[1]);
            addSquareData<<<gridSize, blockSize>>>(CTFSq, tempCTF, newSize[0] * newSize[1]);
        }
        
        scaleFloatData<<<gridSize, blockSize>>>(CTFSq, newSize[0] * newSize[1], 2.0f);
        // Correction for zero-frequency of Fourier transform
        subtractConstant<<<1, 1>>>(CTFHolograms, newSize[0] * newSize[1] * numImages * betaDeltaRatio, 1);
        
        // Calculate the fresnel mean by dimension
        FArray fresnelMean(2);
        for (int i = 0; i < 2; i++) {
            float sum = 0.0f;
            for (int j = 0; j < numImages; j++) {
                sum += fresnelNumbers[j][i];
            }
            fresnelMean[i] = sum / numImages;
        }

        // Apply regularization weights
        float *regWeights;
        cudaMalloc((void**)&regWeights, newSize[0] * newSize[1] * sizeof(float));
        CUDAUtils::ctfRegWeights(regWeights, newSize, fresnelMean, lowFreqLim, highFreqLim);
        addFloatData<<<gridSize, blockSize>>>(regWeights, CTFSq, newSize[0] * newSize[1]);

        // Final calculation and inverse Fourier transform
        complexDivideFloat<<<gridSize, blockSize>>>(CTFHolograms, regWeights, newSize[0] * newSize[1]);
        cufftExecC2C(plan, CTFHolograms, CTFHolograms, CUFFT_INVERSE);
        scaleComplexData<<<gridSize, blockSize>>>(CTFHolograms, newSize[0] * newSize[1], 1.0f / (newSize[0] * newSize[1]));
        extractRealData<<<gridSize, blockSize>>>(CTFHolograms, tempCTF, newSize[0] * newSize[1]);
        
        // Crop the result if padding is applied
        if (!padSize.empty()) {
            float *croppedResult;
            cudaMalloc((void**)&croppedResult, imSize[0] * imSize[1] * sizeof(float));
            CUDAUtils::cropMatrix(tempCTF, croppedResult, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            
            float *temp = tempCTF;
            tempCTF = croppedResult;
            cudaFree(temp);
        }

        FArray result(imSize[0] * imSize[1]);
        cudaMemcpy(result.data(), tempCTF, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

        // Free memory and destroy plan
        cufftDestroy(plan);
        cudaFree(regWeights); cudaFree(tempCTF); cudaFree(tempHologram);
        cudaFree(rowComponent); cudaFree(colComponent); cudaFree(rowRange); cudaFree(colRange);
        cudaFree(holograms_gpu); cudaFree(CTFHolograms); cudaFree(CTFSq);
        std::cout << "Deconstruction finished!" << std::endl;
        
        return result;
    }

    F2DArray reconstruct_iter(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelNumbers, int iterations, 
                             const FArray &initialPhase, ProjectionSolver::Algorithm algorithm, const FArray &algoParameters, const IntArray &padSize, 
                             float minAmplitude, float maxAmplitude, PMagnitudeCons::Type projectionType, CUDAPropKernel::Type kernelType, 
                             CUDAUtils::PaddingType padType, bool calcError)
    {
        // Add GPU environment check
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA capable GPU device found!");
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Using GPU device: " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

        if (fresnelNumbers.size() != numImages)
            throw std::invalid_argument("The number of images and fresnel numbers does not match!");

        // Allocate memory for holograms on GPU
        float *holograms_gpu;
        cudaMalloc((void**)&holograms_gpu, holograms.size() * sizeof(float));
        cudaMemcpy(holograms_gpu, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);

        IntArray newSize(imSize);
        // Optional padding operations on holograms
        if (!padSize.empty()) {
            newSize[0] += 2 * padSize[0];
            newSize[1] += 2 * padSize[1];
            float *paddedHolograms_gpu;
            cudaMalloc((void**)&paddedHolograms_gpu, newSize[0] * newSize[1] * numImages * sizeof(float));

            cudaStream_t *streams = new cudaStream_t[numImages];
            for (int i = 0; i < numImages; i++) {
                cudaStreamCreate(&streams[i]);
                CUDAUtils::padMatrix(holograms_gpu + i * imSize[0] * imSize[1], paddedHolograms_gpu + i * newSize[0] * newSize[1],
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType, 0.0f, streams[i]);
            }

            for (int i = 0; i < numImages; i++) {
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
            
            float *temp = holograms_gpu;
            holograms_gpu = paddedHolograms_gpu;
            cudaFree(temp);
        }

        // print iterative algorithm
        std::cout << "Choosing algorithm: ";
        switch (algorithm) {
            case ProjectionSolver::AP: std::cout << "AP"; break;
            case ProjectionSolver::RAAR: std::cout << "RAAR"; break;
            case ProjectionSolver::HIO: std::cout << "HIO"; break;
            case ProjectionSolver::DRAP: std::cout << "DRAP"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << std::endl;

        // Construct projector on measured holograms
        int blockSize = 1024;
        int gridSize = (newSize[0] * newSize[1] * numImages + blockSize - 1) / blockSize;
        sqrtAmplitude<<<gridSize, blockSize>>>(holograms_gpu, newSize[0] * newSize[1] * numImages);

        std::vector<PropagatorPtr> propagators;
        if (projectionType == PMagnitudeCons::Averaged) {
            propagators.push_back(std::make_shared<Propagator>(newSize, fresnelNumbers, kernelType));
        } else {
            for (const auto &fNumber: fresnelNumbers) {
                F2DArray singleFresnel {fNumber};
                propagators.push_back(std::make_shared<Propagator>(newSize, singleFresnel, kernelType));
            }
        }
        Projector *PM = new PMagnitudeCons(holograms_gpu, numImages, newSize, propagators, projectionType, calcError);

        // Construct projector on amplitude constraint of object
        Projector *PS = new PAmplitudeCons(minAmplitude, maxAmplitude);

        // Initialize wave field from the guess phase
        cuFloatComplex *complexWave;
        cudaMalloc((void**)&complexWave, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
        gridSize = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;

        if (!initialPhase.empty()) {
            // Initialize wave field from the guess phase
            if (initialPhase.size() != imSize[0] * imSize[1]) {
                throw std::invalid_argument("The sizes of guess phase and wave field do not match!");
            }

            // The size of initial phase is the same as the original image size
            float *initPhase_gpu;
            cudaMalloc((void**)&initPhase_gpu, initialPhase.size() * sizeof(float));
            cudaMemcpy(initPhase_gpu, initialPhase.data(), initialPhase.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Optional padding operations on initial phase
            if (!padSize.empty()) {
                float *paddedInitPhase_gpu;
                cudaMalloc((void**)&paddedInitPhase_gpu, newSize[0] * newSize[1] * sizeof(float));
                CUDAUtils::padMatrix(initPhase_gpu, paddedInitPhase_gpu, imSize[0], imSize[1], padSize[0], padSize[1], padType);
                
                float *temp = initPhase_gpu;
                initPhase_gpu = paddedInitPhase_gpu;
                cudaFree(temp);
            }

            initByPhase<<<gridSize, blockSize>>>(complexWave, initPhase_gpu, newSize[0] * newSize[1]);
            cudaFree(initPhase_gpu);
        } else {
            // Initialize wave field from the zero phase
            initializeData<<<gridSize, blockSize>>>(complexWave, make_cuFloatComplex(1.0f, 0.0f), newSize[0] * newSize[1]);
        }
        WaveField waveField(newSize[0], newSize[1], complexWave);

        ProjectionSolver projectionSolver(PM, PS, waveField, algorithm, algoParameters, calcError);
        
        // Reconstruct wave field by iterative projection algorithm
        projectionSolver.execute(iterations).reconsPsi.getComplexWave(complexWave);
        
        if (!padSize.empty()) {
            cuFloatComplex *croppedComplexWave;
            cudaMalloc((void**)&croppedComplexWave, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
            CUDAUtils::cropMatrix(complexWave, croppedComplexWave, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            
            cuFloatComplex *temp = complexWave;
            complexWave = croppedComplexWave;
            cudaFree(temp);
        }

        // Calculate phase and amplitude from reconstructed wave field
        WaveField reconsPsi(imSize[0], imSize[1], complexWave);
        float *phase, *amplitude;
        cudaMalloc((void**)&phase, imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&amplitude, imSize[0] * imSize[1] * sizeof(float));
        reconsPsi.getPhase(phase);
        reconsPsi.getAmplitude(amplitude);

        F2DArray result(2, FArray(imSize[0] * imSize[1]));
        cudaMemcpy(result[0].data(), phase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result[1].data(), amplitude, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

        delete PS; delete PM;
        cudaFree(phase); cudaFree(amplitude); cudaFree(complexWave); cudaFree(holograms_gpu);
        std::cout << "Deconstruction finished!" << std::endl;

        return result;
    }

    Reconstructor::Reconstructor(int batchsize, int images, const IntArray &imsize, const F2DArray &fresnelNumbers, int iter, ProjectionSolver::Algorithm algo,
                                  const FArray &algoParams, const IntArray &padsize, float minAmplitude, float maxAmplitude, PMagnitudeCons::Type projType, 
                                  CUDAPropKernel::Type kernelType, CUDAUtils::PaddingType padtype): batchSize(batchsize), numImages(images), imSize(imsize),
                                  newSize(imsize), iteration(iter), algorithm(algo), algoParameters(algoParams), padSize(padsize), projectionType(projType), padType(padtype)
    {
        if (!padSize.empty()) {
            newSize[0] += 2 * padSize[0];
            newSize[1] += 2 * padSize[1];
            cudaMalloc((void**)&d_paddedHolograms, newSize[0] * newSize[1] * numImages * sizeof(float));
            streams = new cudaStream_t[numImages];
            for (int i = 0; i < numImages; i++) {
                cudaStreamCreate(&streams[i]);
            }
        }

        cudaMalloc((void**)&d_holograms, batchSize * numImages * imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&complexWave, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
        cudaMalloc((void**)&croppedComplexWave, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
        cudaMalloc((void**)&d_phase, imSize[0] * imSize[1] * sizeof(float));
        
        // print iterative algorithm
        std::cout << "Choosing algorithm: ";
        switch (algorithm) {
            case ProjectionSolver::AP: std::cout << "AP"; break;
            case ProjectionSolver::RAAR: std::cout << "RAAR"; break;
            case ProjectionSolver::HIO: std::cout << "HIO"; break;
            case ProjectionSolver::DRAP: std::cout << "DRAP"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << std::endl;

        if (projectionType == PMagnitudeCons::Averaged) {
            propagators.push_back(std::make_shared<Propagator>(newSize, fresnelNumbers, kernelType));
        } else {
            for (const auto &fNumber: fresnelNumbers) {
                F2DArray singleFresnel {fNumber};
                propagators.push_back(std::make_shared<Propagator>(newSize, singleFresnel, kernelType));
            }
        }
        
        PS = new PAmplitudeCons(minAmplitude, maxAmplitude);
    }

    FArray Reconstructor::reconsBatch(const FArray &holograms)
    {
        cudaMemcpy(d_holograms, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);
        FArray result(batchSize * imSize[0] * imSize[1]);

        for (int i = 0; i < batchSize; i++) {
            // Optional padding operations on holograms
            if (!padSize.empty()) {
                for (int j = 0; j < numImages; j++) {
                    CUDAUtils::padMatrix(d_holograms + i * numImages * imSize[0] * imSize[1] + j * imSize[0] * imSize[1], 
                                         d_paddedHolograms + j * newSize[0] * newSize[1], imSize[0], imSize[1], padSize[0],
                                         padSize[1], padType, 0.0f, streams[j]);
                }

                for (int j = 0; j < numImages; j++) {
                    cudaStreamSynchronize(streams[j]);
                }

                d_temp = d_paddedHolograms;
            } else {
                d_temp = d_holograms + i * numImages * imSize[0] * imSize[1];
            }

            int blockSize = 1024;
            int numBlocks = (newSize[0] * newSize[1] * numImages + blockSize - 1) / blockSize;
            sqrtAmplitude<<<numBlocks, blockSize>>>(d_temp, newSize[0] * newSize[1] * numImages);

            // Construct projector on measured holograms
            Projector *PM = new PMagnitudeCons(d_temp, numImages, newSize, propagators, projectionType);

            numBlocks = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;
            initializeData<<<numBlocks, blockSize>>>(complexWave, make_cuFloatComplex(1.0f, 0.0f), newSize[0] * newSize[1]);
            WaveField waveField(newSize[0], newSize[1], complexWave);

            ProjectionSolver projectionSolver(PM, PS, waveField, algorithm, algoParameters);
            projectionSolver.execute(iteration).reconsPsi.getComplexWave(complexWave);

            if (!padSize.empty()) {
                CUDAUtils::cropMatrix(complexWave, croppedComplexWave, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            } else {
                croppedComplexWave = complexWave;
            }

            WaveField reconsPsi(imSize[0], imSize[1], croppedComplexWave);
            reconsPsi.getPhase(d_phase);
            cudaMemcpy(result.data() + i * imSize[0] * imSize[1], d_phase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

            delete PM;
        }

        return result;
    }
    
    Reconstructor::~Reconstructor()
    {
        cudaFree(d_holograms); cudaFree(d_phase);
        cudaFree(complexWave); cudaFree(croppedComplexWave);

        if (!padSize.empty()) {
            cudaFree(d_paddedHolograms);
            for (int i = 0; i < numImages; i++) {
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
        }

        delete PS;
    }
}