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

    FArray reconstruct_ctf(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelnumbers, float lowFreqLim, float highFreqLim,
                             float betaDeltaRatio, const IntArray &padSize, CUDAUtils::PaddingType padType)
    {
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
        
        if (lowFreqLim < (2.0f * numImages * std::pow(betaDeltaRatio, 2))) {
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
            for (int i = 0; i < numImages; i++) {
                CUDAUtils::padMatrix(holograms_gpu + i * imSize[0] * imSize[1], paddedHolograms_gpu + i * newSize[0] * newSize[1],
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType);
            }
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
        cufftPlan2d(&plan, newSize[0], newSize[1], CUFFT_R2C);

        // Define grid and block sizes for different kernels
        int blockSize1D = 512;
        int gridRowSize1D = (newSize[0] + blockSize1D - 1) / blockSize1D;
        int gridColSize1D = (newSize[1] + blockSize1D - 1) / blockSize1D;
        dim3 blockSize2D(32, 32);
        dim3 gridSize2D((newSize[1] + blockSize2D.x - 1) / blockSize2D.x, (newSize[0] + blockSize2D.y - 1) / blockSize2D.y);
        int blockSize = 1024;
        int gridSize = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;

        // Initialize CTF data
        initializeData<<<gridSize, blockSize>>>(CTFHolograms, make_cuFloatComplex(0.0f, 0.0f), newSize[0] * newSize[1]);
        initializeData<<<gridSize, blockSize>>>(CTFSq, 0.0f, newSize[0] * newSize[1]);
        
        // CTF reconstruction main loop
        for (size_t i = 0; i < numImages; i++) {
            genCTFComponent<<<gridRowSize1D, blockSize1D>>>(rowComponent, rowRange, fresnelNumbers[i][0], newSize[0]);
            genCTFComponent<<<gridColSize1D, blockSize1D>>>(colComponent, colRange, fresnelNumbers[i][1], newSize[1]);
            multiplyObliFactor_2<<<gridSize2D, blockSize2D>>>(tempCTF, rowComponent, colComponent, newSize[0], newSize[1]);
            
            computeCTF<<<gridSize, blockSize>>>(tempCTF, betaDeltaRatio, newSize[0] * newSize[1]);
            cufftExecR2C(plan, holograms_gpu + i * newSize[0] * newSize[1], tempHologram);
            
            CTFMultiplyHologram<<<gridSize, blockSize>>>(tempHologram, tempCTF, newSize[0] * newSize[1]);
            addWaveField<<<gridSize, blockSize>>>(CTFHolograms, tempHologram, newSize[0] * newSize[1]);
            addSquareData<<<gridSize, blockSize>>>(CTFSq, tempCTF, newSize[0] * newSize[1]);
        }
        scaleFloatData<<<gridSize, blockSize>>>(CTFSq, newSize[0] * newSize[1], 2.0f);

        // Correction for zero-frequency of Fourier transform
        subtractConstant<<<gridSize, blockSize>>>(CTFHolograms, newSize[0] * newSize[1] * numImages * betaDeltaRatio, 1);
        
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
        cufftPlan2d(&plan, newSize[0], newSize[1], CUFFT_C2C);
        cufftExecC2C(plan, CTFHolograms, CTFHolograms, CUFFT_INVERSE);
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

    F2DArray reconstruct_iter(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelNumbers, int iterations, const FArray &initialPhase,
                          ProjectionSolver::Algorithm algorithm, const FArray &algoParameters, const IntArray &padSize, float minAmplitude, float maxAmplitude,
                          PMagnitudeCons::Type projectionType, CUDAPropKernel::Type kernelType, CUDAUtils::PaddingType padType, bool calcError)
    {
        /* Check correctness and compatibility of measurements and fresnel numbers */
        if (holograms.empty() || imSize.empty())
            throw std::invalid_argument("Invalid measurement data or image size!");
  
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
            for (int i = 0; i < numImages; i++) {
                CUDAUtils::padMatrix(holograms_gpu + i * imSize[0] * imSize[1], paddedHolograms_gpu + i * newSize[0] * newSize[1],
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType);
            }
            float *temp = holograms_gpu;
            holograms_gpu = paddedHolograms_gpu;
            cudaFree(temp);
        }

        // Construct projector on measured holograms
        int blockSize = 1024;
        int gridSize = (newSize[0] * newSize[1] * numImages + blockSize - 1) / blockSize;
        sqrtAmplitude<<<gridSize, blockSize>>>(holograms_gpu, newSize[0] * newSize[1] * numImages);
        Projector *PM = new PMagnitudeCons(holograms_gpu, numImages, newSize, fresnelNumbers, projectionType, kernelType, calcError);

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

}