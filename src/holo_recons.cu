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

    F2DArray reconstruct_ctf(const FArray &holograms, const IntArray &imSize, const F2DArray &fresnelNumbers)
    {

    }

    F2DArray reconstruct_iter(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelNumbers, int iterations, const FArray &initialPhase,
                          ProjectionSolver::Algorithm algorithm, const FArray &algoParameters, const IntArray &padSize, float minAmplitude, float maxAmplitude,
                          PMagnitudeCons::Type projectionType, CUDAPropKernel::Type kernelType, CUDAUtils::PaddingType padType, bool calcError)
    {
        /* Check correctness and compatibility of measurements and fresnel numbers */
        if (holograms.empty() || imSize.empty())
            throw std::invalid_argument("Invalid measurement data or image size!");
  
        if (fresnelNumbers.size() != numImages)
            throw "The number of images and fresnel numbers does not match!";

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