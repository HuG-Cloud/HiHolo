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
                           float highFreqLim, float betaDeltaRatio, const IntArray &padSize, CUDAUtils::PaddingType padType, float padValue)
    {
        // Add GPU environment check
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA capable GPU device found!");
        }
  
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

        std::cout << "Choosing Algorithm: CTF" << std::endl;
        
        // transfer holograms to GPU
        float *holograms_gpu, *phase_gpu;
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
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType, padValue, streams[i]);
            }

            for (int i = 0; i < numImages; i++) {
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
            
            cudaFree(holograms_gpu);
            holograms_gpu = paddedHolograms_gpu;
        }

        // 按维度计算菲涅尔平均值
        FArray fresnelMean(2);
        for (int i = 0; i < 2; i++) {
            float sum = 0.0f;
            for (int j = 0; j < numImages; j++) {
                sum += fresnelNumbers[j][i];
            }
            fresnelMean[i] = sum / numImages;
        }

        float *regWeights;
        cudaMalloc((void**)&regWeights, newSize[0] * newSize[1] * sizeof(float));
        CUDAUtils::ctfRegWeights(regWeights, newSize, fresnelMean, lowFreqLim, highFreqLim);

        cudaMalloc((void**)&phase_gpu, newSize[0] * newSize[1] * sizeof(float));
        CUDAUtils::ctf_recons_kernel(holograms_gpu, phase_gpu, newSize, numImages, fresnelNumbers, betaDeltaRatio, regWeights);

        // 如果应用了填充，则需要裁剪结果
        if (!padSize.empty()) {
            float *croppedPhase;
            cudaMalloc((void**)&croppedPhase, imSize[0] * imSize[1] * sizeof(float));
            CUDAUtils::cropMatrix(phase_gpu, croppedPhase, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            cudaFree(phase_gpu);
            phase_gpu = croppedPhase;
        }

        FArray result(imSize[0] * imSize[1]);
        cudaMemcpy(result.data(), phase_gpu, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(phase_gpu); cudaFree(holograms_gpu); cudaFree(regWeights);

        return result;
    }

    CTFReconstructor::CTFReconstructor(int batchsize, int images, const IntArray &imsize, const F2DArray &fresnelnumbers, float lowFreqLim, float highFreqLim, float ratio,
                                       const IntArray &padsize, CUDAUtils::PaddingType padtype, float padvalue): batchSize(batchsize), numImages(images), imSize(imsize),
                                       newSize(imsize), fresnelNumbers(fresnelnumbers), betaDeltaRatio(ratio), padSize(padsize), padType(padtype), padValue(padvalue)
    {
        for (auto &fresnelNumber: fresnelNumbers) {
            if (fresnelNumber.size() != 1 && fresnelNumber.size() != imSize.size()) {
                throw std::invalid_argument("Invalid Fresnel number!");
            }
            if (fresnelNumber.size() == 1) {
                fresnelNumber.push_back(fresnelNumber[0]);
            }
        }

        if (lowFreqLim < (2.0f * numImages * std::pow(betaDeltaRatio, 2.0f))) {
            lowFreqLim = 0.0f;
        }

        if (!padSize.empty()) {
            newSize[0] += 2 * padSize[0];
            newSize[1] += 2 * padSize[1];
            cudaMalloc((void**)&d_paddedHolograms, newSize[0] * newSize[1] * numImages * sizeof(float));
            cudaMalloc((void**)&d_croppedPhase, imSize[0] * imSize[1] * sizeof(float));
            streams = new cudaStream_t[numImages];
            for (int i = 0; i < numImages; i++) {
                cudaStreamCreate(&streams[i]);
            }
        }

        cudaMalloc((void**)&d_holograms, batchSize * numImages * imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&d_phase, newSize[0] * newSize[1] * sizeof(float));
        cudaMalloc((void**)&regWeights, newSize[0] * newSize[1] * sizeof(float));
        cudaMalloc((void**)&d_regTemp, newSize[0] * newSize[1] * sizeof(float));

        FArray fresnelMean(2);
        for (int i = 0; i < 2; i++) {
            float sum = 0.0f;
            for (int j = 0; j < numImages; j++) {
                sum += fresnelNumbers[j][i];
            }
            fresnelMean[i] = sum / numImages;
        }
        CUDAUtils::ctfRegWeights(regWeights, newSize, fresnelMean, lowFreqLim, highFreqLim);
    }

    FArray CTFReconstructor::reconsBatch(const FArray &holograms)
    {
        cudaMemcpy(d_holograms, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);
        FArray result(imSize[0] * imSize[1] * batchSize);

        for (int i = 0; i < batchSize; i++) {
            if (!padSize.empty()) {
                for (int j = 0; j < numImages; j++) {
                    CUDAUtils::padMatrix(d_holograms + i * numImages * imSize[0] * imSize[1] + j * imSize[0] * imSize[1],
                                         d_paddedHolograms + j * newSize[0] * newSize[1], imSize[0], imSize[1], padSize[0],
                                         padSize[1], padType, padValue, streams[j]);
                }
                for (int j = 0; j < numImages; j++) {
                    cudaStreamSynchronize(streams[j]);
                }

                d_temp = d_paddedHolograms;
            } else {
                d_temp = d_holograms + i * numImages * imSize[0] * imSize[1];
            }

            cudaMemcpy(d_regTemp, regWeights, newSize[0] * newSize[1] * sizeof(float), cudaMemcpyDeviceToDevice);
            CUDAUtils::ctf_recons_kernel(d_temp, d_phase, newSize, numImages, fresnelNumbers, betaDeltaRatio, d_regTemp);

            if (!padSize.empty()) {
                CUDAUtils::cropMatrix(d_phase, d_croppedPhase, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            } else {
                d_croppedPhase = d_phase;
            }

            cudaMemcpy(result.data() + i * imSize[0] * imSize[1], d_croppedPhase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        }

        return result;
    }

    CTFReconstructor::~CTFReconstructor()
    {
        cudaFree(d_holograms); cudaFree(d_phase);
        cudaFree(regWeights); cudaFree(d_regTemp);
        if (!padSize.empty()) {
            cudaFree(d_paddedHolograms);
            cudaFree(d_croppedPhase);
            for (int i = 0; i < numImages; i++) {
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
        }
    }

    F2DArray reconstruct_iter(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelNumbers, int iterations, const FArray &initialPhase,
                              ProjectionSolver::Algorithm algorithm, const FArray &algoParameters, float minPhase, float maxPhase, float minAmplitude, float maxAmplitude,
                              const IntArray &support, float outsideValue, const IntArray &padSize, CUDAUtils::PaddingType padType, float padValue, PMagnitudeCons::Type projectionType,
                              CUDAPropKernel::Type kernelType, const FArray &holoProbes, const FArray &initProbePhase, bool calcError)
    {
        // Add GPU environment check
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA capable GPU device found!");
        }

        if (fresnelNumbers.size() != numImages)
            throw std::invalid_argument("The number of images and fresnel numbers does not match!");

        // Allocate memory for holograms on GPU
        float *holograms_gpu, *holoprobes_gpu;
        cudaMalloc((void**)&holograms_gpu, holograms.size() * sizeof(float));
        cudaMemcpy(holograms_gpu, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);

        bool isAPWP = (algorithm == ProjectionSolver::APWP);

        if (isAPWP) {
            if (holoProbes.size() != holograms.size())
                throw std::invalid_argument("The number of probes and holograms does not match!");
            cudaMalloc((void**)&holoprobes_gpu, holoProbes.size() * sizeof(float));
            cudaMemcpy(holoprobes_gpu, holoProbes.data(), holoProbes.size() * sizeof(float), cudaMemcpyHostToDevice);
        }

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
                                     imSize[0], imSize[1], padSize[0], padSize[1], padType, padValue, streams[i]);
            }
            for (int i = 0; i < numImages; i++) {
                cudaStreamSynchronize(streams[i]);
            }

            cudaFree(holograms_gpu);
            holograms_gpu = paddedHolograms_gpu;

            if (isAPWP) {
                float *paddedProbes_gpu;
                cudaMalloc((void**)&paddedProbes_gpu, newSize[0] * newSize[1] * numImages * sizeof(float));

                for (int i = 0; i < numImages; i++) {
                    CUDAUtils::padMatrix(holoprobes_gpu + i * imSize[0] * imSize[1], paddedProbes_gpu + i * newSize[0] * newSize[1],
                                         imSize[0], imSize[1], padSize[0], padSize[1], padType, padValue, streams[i]);
                }
                for (int i = 0; i < numImages; i++) {
                    cudaStreamSynchronize(streams[i]);
                }

                cudaFree(holoprobes_gpu);
                holoprobes_gpu = paddedProbes_gpu;
            }

            for (int i = 0; i < numImages; i++) {
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
        }

        // Construct projector on measured holograms
        int blockSize = 1024;
        int gridSize = (newSize[0] * newSize[1] * numImages + blockSize - 1) / blockSize;
        // divideFloatData<<<gridSize, blockSize>>>(holograms_gpu, holoprobes_gpu, newSize[0] * newSize[1] * numImages);
        sqrtIntensity<<<gridSize, blockSize>>>(holograms_gpu, newSize[0] * newSize[1] * numImages);
        if (isAPWP) {
            sqrtIntensity<<<gridSize, blockSize>>>(holoprobes_gpu, newSize[0] * newSize[1] * numImages);
        }

        std::vector<PropagatorPtr> propagators;
        if (projectionType == PMagnitudeCons::Averaged) {
            propagators.push_back(std::make_shared<Propagator>(newSize, fresnelNumbers, kernelType));
        } else {
            for (const auto &fNumber: fresnelNumbers) {
                F2DArray singleFresnel {fNumber};
                propagators.push_back(std::make_shared<Propagator>(newSize, singleFresnel, kernelType));
            }
        }

        Projector *PM;
        if (isAPWP) {
            PM = new PMagnitudeCons(holograms_gpu, holoprobes_gpu, numImages, newSize, propagators, projectionType, calcError);
        } else {
            PM = new PMagnitudeCons(holograms_gpu, numImages, newSize, propagators, projectionType, calcError);
        }

        // Construct projector on constraints of object plane
        float *support_gpu = nullptr;
        if (!support.empty()) {
            if (support[0] > newSize[0] || support[1] > newSize[1]) {
                throw std::invalid_argument("The support size is larger than the image size!");
            }

            // Initialize support based on the support size
            if (!(support[0] == newSize[0] && support[1] == newSize[1])) {
                cudaMalloc((void**)&support_gpu, support[0] * support[1] * sizeof(float));
                gridSize = (support[0] * support[1] + blockSize - 1) / blockSize;
                initializeData<<<gridSize, blockSize>>>(support_gpu, 1.0f, support[0] * support[1]);

                IntArray suppPadSize {(newSize[0] - support[0]) / 2, (newSize[1] - support[1]) / 2};
                float *paddedSupport_gpu = CUDAUtils::padInputData(support_gpu, support, suppPadSize, CUDAUtils::Constant, 0.0f);
                cudaFree(support_gpu);
                support_gpu = paddedSupport_gpu;
            }
        }

        Projector *pAmplitude = new PAmplitudeCons(minAmplitude, maxAmplitude);
        Projector *pPhase, *pSupport, *PS;
        bool onlyAmpCons = (minPhase == -FloatInf && maxPhase == FloatInf && support_gpu == nullptr);
        if (onlyAmpCons) {
            PS = pAmplitude;
        } else {
            pPhase = new PPhaseCons(minPhase, maxPhase);
            pSupport = new PSupportCons(support_gpu, newSize[0] * newSize[1], outsideValue);
            PS = new MultiObjectCons(pPhase, pAmplitude, pSupport);
        }

        // Initialize wave field from the guess phase
        cuFloatComplex *complexWave, *probe;
        cudaMalloc((void**)&complexWave, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
        gridSize = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;

        // Initialize wave field from the guess phase
        if (!initialPhase.empty()) {
            // Initialize wave field from the guess phase
            if (initialPhase.size() != imSize[0] * imSize[1]) {
                throw std::invalid_argument("The sizes of guess phase and wave field do not match!");
            }

            // The size of initial phase is the same as the original image size
            float *initPhase_gpu;
            cudaMalloc((void**)&initPhase_gpu, initialPhase.size() * sizeof(float));
            cudaMemcpy(initPhase_gpu, initialPhase.data(), initialPhase.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Pad initial phase if needed
            float *paddedInitPhase_gpu = CUDAUtils::padInputData(initPhase_gpu, imSize, padSize, padType, padValue);
            if (paddedInitPhase_gpu != initPhase_gpu) {
                cudaFree(initPhase_gpu);
            }

            initByPhase<<<gridSize, blockSize>>>(complexWave, paddedInitPhase_gpu, newSize[0] * newSize[1]);
            cudaFree(paddedInitPhase_gpu);
        } else {
            // Initialize wave field from the zero phase
            initializeData<<<gridSize, blockSize>>>(complexWave, make_cuFloatComplex(1.0f, 0.0f), newSize[0] * newSize[1]);
        }
        WaveField waveField(newSize[0], newSize[1], complexWave);

        // Initialize probe field from the guess phase
        if (isAPWP) {
            cudaMalloc((void**)&probe, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
            if (!initProbePhase.empty()) {
                if (initProbePhase.size() != imSize[0] * imSize[1]) {
                    throw std::invalid_argument("The sizes of guess probe phase and wave field do not match!");
                }

                float *initProbePhase_gpu;
                cudaMalloc((void**)&initProbePhase_gpu, initProbePhase.size() * sizeof(float));
                cudaMemcpy(initProbePhase_gpu, initProbePhase.data(), initProbePhase.size() * sizeof(float), cudaMemcpyHostToDevice);

                // Pad probe phase if needed
                float *paddedProbePhase_gpu = CUDAUtils::padInputData(initProbePhase_gpu, imSize, padSize, padType, padValue);
                if (paddedProbePhase_gpu != initProbePhase_gpu) {
                    cudaFree(initProbePhase_gpu);
                }

                initByPhase<<<gridSize, blockSize>>>(probe, paddedProbePhase_gpu, newSize[0] * newSize[1]);
                cudaFree(paddedProbePhase_gpu);
            } else {
                initializeData<<<gridSize, blockSize>>>(probe, make_cuFloatComplex(1.0f, 0.0f), newSize[0] * newSize[1]);
            }
        }

        ProjectionSolver *projectionSolver;
        if (isAPWP) {
            WaveField probeField(newSize[0], newSize[1], probe);
            projectionSolver = new ProjectionSolver(PM, PS, waveField, probeField, calcError);
        } else {
            projectionSolver = new ProjectionSolver(PM, PS, waveField, algorithm, algoParameters, calcError);
        }
        
        // Reconstruct wave field by iterative projection algorithm
        auto iterResult = projectionSolver->execute(iterations);
        iterResult.reconsPsi.getComplexWave(complexWave);
        
        if (!padSize.empty()) {
            cuFloatComplex *croppedComplexWave;
            cudaMalloc((void**)&croppedComplexWave, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
            CUDAUtils::cropMatrix(complexWave, croppedComplexWave, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            
            cudaFree(complexWave);
            complexWave = croppedComplexWave;
        }

        if (isAPWP) {
            iterResult.reconsProbe.getComplexWave(probe);

            if (!padSize.empty()) {
                cuFloatComplex *croppedProbe;
                cudaMalloc((void**)&croppedProbe, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
                CUDAUtils::cropMatrix(probe, croppedProbe, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
                cudaFree(probe);
                probe = croppedProbe;
            }
        }

        // Calculate phase and amplitude from reconstructed wave field
        WaveField reconsPsi(imSize[0], imSize[1], complexWave);
        float *phase, *amplitude, *probePhase;
        cudaMalloc((void**)&phase, imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&amplitude, imSize[0] * imSize[1] * sizeof(float));
        reconsPsi.getPhase(phase);
        reconsPsi.getAmplitude(amplitude);
        
        if (isAPWP) {
            WaveField reconsProbe(imSize[0], imSize[1], probe);
            cudaMalloc((void**)&probePhase, imSize[0] * imSize[1] * sizeof(float));
            reconsProbe.getPhase(probePhase);
        }

        F2DArray result(3, FArray(imSize[0] * imSize[1]));
        cudaMemcpy(result[0].data(), phase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result[1].data(), amplitude, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        if (isAPWP) {
            cudaMemcpy(result[2].data(), probePhase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        }

        if (calcError) {
            result.push_back(iterResult.finalError[0]);
            result.push_back(iterResult.finalError[1]);
        }
        
        delete projectionSolver; delete PM; delete PS;
        if (!onlyAmpCons) {
            delete pPhase; delete pSupport; delete pAmplitude;
        }
        cudaFree(phase); cudaFree(amplitude); cudaFree(complexWave); cudaFree(holograms_gpu);
        if (isAPWP) {
             cudaFree(probePhase); cudaFree(probe); cudaFree(holoprobes_gpu);
        }
        if (support_gpu)
            cudaFree(support_gpu);

        return result;
    }

    F2DArray reconstruct_bipepi(const FArray &holograms, int numImages, const IntArray &measSize, const F2DArray &fresnelNumbers, int iterations,
                                const IntArray &imSize, const FArray &initialPhase, const FArray &initialAmplitude, float minPhase, float maxPhase,
                                float minAmplitude, float maxAmplitude, const IntArray &support, float outsideValue, PMagnitudeCons::Type projectionType,
                                CUDAPropKernel::Type kernelType, bool calcError)
    {
        // Add GPU environment check
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0)
            throw std::runtime_error("No CUDA capable GPU device found!");

        if (fresnelNumbers.size() != numImages)
            throw std::invalid_argument("The number of images and fresnel numbers does not match!");

        if (support.empty())
            throw std::invalid_argument("BIPEPI algorithm requires the support size!");

        // Allocate memory for holograms on GPU
        float *holograms_gpu;
        cudaMalloc((void**)&holograms_gpu, holograms.size() * sizeof(float));
        cudaMemcpy(holograms_gpu, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Construct projector on measured holograms
        int blockSize = 1024;
        int gridSize = (measSize[0] * measSize[1] * numImages + blockSize - 1) / blockSize;
        sqrtIntensity<<<gridSize, blockSize>>>(holograms_gpu, measSize[0] * measSize[1] * numImages);

        std::vector<PropagatorPtr> propagators;
        propagators.push_back(std::make_shared<Propagator>(imSize, fresnelNumbers, kernelType));
        Projector *PM = new PMagnitudeCons(holograms_gpu, numImages, measSize, propagators, imSize, projectionType, calcError);

        // Construct projector on constraints of object plane
        float *support_gpu = nullptr;
        if (support[0] > imSize[0] || support[1] > imSize[1]) {
            throw std::invalid_argument("The support size is larger than the image size!");
        }

        // Initialize support based on the support size
        if (!(support[0] == imSize[0] && support[1] == imSize[1])) {
            cudaMalloc((void**)&support_gpu, support[0] * support[1] * sizeof(float));
            gridSize = (support[0] * support[1] + blockSize - 1) / blockSize;
            initializeData<<<gridSize, blockSize>>>(support_gpu, 1.0f, support[0] * support[1]);

            IntArray suppPadSize {(imSize[0] - support[0]) / 2, (imSize[1] - support[1]) / 2};
            float *paddedSupport_gpu = CUDAUtils::padInputData(support_gpu, support, suppPadSize, CUDAUtils::Constant, 0.0f);
            cudaFree(support_gpu);
            support_gpu = paddedSupport_gpu;
        }

        Projector *pAmplitude = new PAmplitudeCons(minAmplitude, maxAmplitude);
        Projector *pPhase, *pSupport, *PS;
        bool onlyAmpCons = (minPhase == -FloatInf && maxPhase == FloatInf && support_gpu == nullptr);
        if (onlyAmpCons) {
            PS = pAmplitude;
        } else {
            pPhase = new PPhaseCons(minPhase, maxPhase);
            pSupport = new PSupportCons(support_gpu, imSize[0] * imSize[1], outsideValue);
            PS = new MultiObjectCons(pPhase, pAmplitude, pSupport);
        }

        // Initialize wave field from the guess phase
        cuFloatComplex *complexWave;
        cudaMalloc((void**)&complexWave, imSize[0] * imSize[1] * sizeof(cuFloatComplex));
        gridSize = (imSize[0] * imSize[1] + blockSize - 1) / blockSize;

        // Initialize wave field from the guess phase and amplitude
        if (!initialPhase.empty()) {
            float *initPhase_gpu, *initAmplitude_gpu;
            cudaMalloc((void**)&initPhase_gpu, initialPhase.size() * sizeof(float));
            cudaMalloc((void**)&initAmplitude_gpu, initialAmplitude.size() * sizeof(float));
            cudaMemcpy(initPhase_gpu, initialPhase.data(), initialPhase.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(initAmplitude_gpu, initialAmplitude.data(), initialAmplitude.size() * sizeof(float), cudaMemcpyHostToDevice);

            computeComplexData<<<gridSize, blockSize>>>(complexWave, initAmplitude_gpu, initPhase_gpu, imSize[0] * imSize[1]);
            cudaFree(initAmplitude_gpu); cudaFree(initPhase_gpu);
        } else {
            // Initialize wave field from the zero phase
            initializeData<<<gridSize, blockSize>>>(complexWave, make_cuFloatComplex(1.0f, 0.0f), imSize[0] * imSize[1]);
        }
        WaveField waveField(imSize[0], imSize[1], complexWave);

        ProjectionSolver *projectionSolver = new ProjectionSolver(PM, PS, waveField, ProjectionSolver::BIPEPI, FArray(), calcError);
        
        // Reconstruct wave field by iterative projection algorithm
        auto iterResult = projectionSolver->execute(iterations);
        iterResult.reconsPsi.getComplexWave(complexWave);

        // Calculate phase and amplitude from reconstructed wave field
        WaveField reconsPsi(imSize[0], imSize[1], complexWave);
        float *phase, *amplitude;
        cudaMalloc((void**)&phase, imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&amplitude, imSize[0] * imSize[1] * sizeof(float));
        reconsPsi.getPhase(phase);
        reconsPsi.getAmplitude(amplitude);

        F2DArray result(3, FArray(imSize[0] * imSize[1]));
        cudaMemcpy(result[0].data(), phase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result[1].data(), amplitude, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

        if (calcError) {
            result.push_back(iterResult.finalError[0]);
            result.push_back(iterResult.finalError[1]);
        }
        
        delete projectionSolver; delete PM; delete PS;
        if (!onlyAmpCons) {
            delete pPhase; delete pSupport; delete pAmplitude;
        }
        cudaFree(phase); cudaFree(amplitude); cudaFree(complexWave); cudaFree(holograms_gpu);
        if (support_gpu)
            cudaFree(support_gpu);

        return result;
    }

    Reconstructor::Reconstructor(int batchsize, int images, const IntArray &imsize, const F2DArray &fresnelNumbers, int iter, ProjectionSolver::Algorithm algo,
                                 const FArray &algoParams, float minPhase, float maxPhase, float minAmplitude, float maxAmplitude, const IntArray &support, float outsideValue,
                                 const IntArray &padsize, CUDAUtils::PaddingType padtype, float padvalue, PMagnitudeCons::Type projType, CUDAPropKernel::Type kernelType):
                                 batchSize(batchsize), numImages(images), imSize(imsize), newSize(imsize), iteration(iter), algorithm(algo), algoParameters(algoParams),
                                 padSize(padsize), projectionType(projType), padType(padtype), padValue(padvalue), d_support(nullptr)
    {
        if (!padSize.empty()) {
            newSize[0] += 2 * padSize[0];
            newSize[1] += 2 * padSize[1];
            cudaMalloc((void**)&d_paddedHolograms, newSize[0] * newSize[1] * numImages * sizeof(float));
            cudaMalloc((void**)&d_paddedInitPhase, newSize[0] * newSize[1] * sizeof(float));
            cudaMalloc((void**)&d_croppedPhase, imSize[0] * imSize[1] * sizeof(float));
            streams = new cudaStream_t[numImages];
            for (int i = 0; i < numImages; i++) {
                cudaStreamCreate(&streams[i]);
            }
        }

        cudaMalloc((void**)&d_holograms, batchSize * numImages * imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&d_initPhase, batchSize * imSize[0] * imSize[1] * sizeof(float));
        cudaMalloc((void**)&complexWave, newSize[0] * newSize[1] * sizeof(cuFloatComplex));
        cudaMalloc((void**)&d_phase, newSize[0] * newSize[1] * sizeof(float));

        // Construct propagators according to the projection type
        if (projectionType == PMagnitudeCons::Averaged) {
            propagators.push_back(std::make_shared<Propagator>(newSize, fresnelNumbers, kernelType));
        } else {
            for (const auto &fNumber: fresnelNumbers) {
                F2DArray singleFresnel {fNumber};
                propagators.push_back(std::make_shared<Propagator>(newSize, singleFresnel, kernelType));
            }
        }

        // Construct projector on constraints of object plane
        if (!support.empty()) {
            if (support[0] > newSize[0] || support[1] > newSize[1]) {
                throw std::invalid_argument("The support size is larger than the image size!");
            }

            if (!(support[0] == newSize[0] && support[1] == newSize[1])) {
                cudaMalloc((void**)&d_support, support[0] * support[1] * sizeof(float));
                int blockSize = 1024;
                int gridSize = (support[0] * support[1] + blockSize - 1) / blockSize;
                initializeData<<<gridSize, blockSize>>>(d_support, 1.0f, support[0] * support[1]);

                IntArray suppPadSize {(newSize[0] - support[0]) / 2, (newSize[1] - support[1]) / 2};
                float *paddedSupport_gpu = CUDAUtils::padInputData(d_support, support, suppPadSize, CUDAUtils::Constant, 0.0f);
                cudaFree(d_support);
                d_support = paddedSupport_gpu;
            }
        }

        pAmplitude = new PAmplitudeCons(minAmplitude, maxAmplitude);
        onlyAmpCons = (minPhase == -FloatInf && maxPhase == FloatInf && d_support == nullptr);
        if (onlyAmpCons) {
            PS = pAmplitude;
        } else {
            pPhase = new PPhaseCons(minPhase, maxPhase);
            pSupport = new PSupportCons(d_support, newSize[0] * newSize[1], outsideValue);
            PS = new MultiObjectCons(pPhase, pAmplitude, pSupport);
        }
    }

    FArray Reconstructor::reconsBatch(const FArray &holograms, const FArray &initialPhase)
    {
        cudaMemcpy(d_holograms, holograms.data(), holograms.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (!initialPhase.empty()) {
            if (initialPhase.size() != imSize[0] * imSize[1] * batchSize) {
                throw std::invalid_argument("The sizes of guess phase and wave field do not match!");
            }
            cudaMemcpy(d_initPhase, initialPhase.data(), initialPhase.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        FArray result(batchSize * imSize[0] * imSize[1]);

        for (int i = 0; i < batchSize; i++) {
            // Optional padding operations on holograms
            if (!padSize.empty()) {
                for (int j = 0; j < numImages; j++) {
                    CUDAUtils::padMatrix(d_holograms + i * numImages * imSize[0] * imSize[1] + j * imSize[0] * imSize[1], 
                                         d_paddedHolograms + j * newSize[0] * newSize[1], imSize[0], imSize[1], padSize[0],
                                         padSize[1], padType, padValue, streams[j]);
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
            sqrtIntensity<<<numBlocks, blockSize>>>(d_temp, newSize[0] * newSize[1] * numImages);

            // Construct projector on measured holograms
            Projector *PM = new PMagnitudeCons(d_temp, numImages, newSize, propagators, projectionType, false);
            
            numBlocks = (newSize[0] * newSize[1] + blockSize - 1) / blockSize;
            if (!initialPhase.empty()) {
                if (!padSize.empty()) {
                    CUDAUtils::padMatrix(d_initPhase + i * imSize[0] * imSize[1], d_paddedInitPhase,
                                         imSize[0], imSize[1], padSize[0], padSize[1], padType, padValue);
                    d_temp = d_paddedInitPhase;
                } else {
                    d_temp = d_initPhase + i * imSize[0] * imSize[1];
                }
                initByPhase<<<numBlocks, blockSize>>>(complexWave, d_temp, newSize[0] * newSize[1]);
            } else {
                initializeData<<<numBlocks, blockSize>>>(complexWave, make_cuFloatComplex(1.0f, 0.0f), newSize[0] * newSize[1]);
            }
            WaveField waveField(newSize[0], newSize[1], complexWave);

            ProjectionSolver projectionSolver(PM, PS, waveField, algorithm, algoParameters, false);
            projectionSolver.execute(iteration).reconsPsi.getPhase(d_phase);

            if (!padSize.empty()) {
                CUDAUtils::cropMatrix(d_phase, d_croppedPhase, newSize[0], newSize[1], padSize[0], padSize[1], padSize[0], padSize[1]);
            } else {
                d_croppedPhase = d_phase;
            }

            cudaMemcpy(result.data() + i * imSize[0] * imSize[1], d_croppedPhase, imSize[0] * imSize[1] * sizeof(float), cudaMemcpyDeviceToHost);

            delete PM;
        }

        return result;
    }
    
    Reconstructor::~Reconstructor()
    {
        cudaFree(d_holograms);
        cudaFree(d_phase);
        cudaFree(complexWave);
        cudaFree(d_initPhase);
        if (d_support)
            cudaFree(d_support);

        if (!padSize.empty()) {
            cudaFree(d_paddedHolograms);
            cudaFree(d_paddedInitPhase);
            cudaFree(d_croppedPhase);
            for (int i = 0; i < numImages; i++) {
                cudaStreamDestroy(streams[i]);
            }
            delete[] streams;
        }

        delete PS;
        if (!onlyAmpCons) {
            delete pPhase; delete pSupport; delete pAmplitude;
        }
    }
}