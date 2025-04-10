#ifndef HOLO_RECONS_H_
#define HOLO_RECONS_H_

#include "ProjectionSolver.h"
// #include "imageio_utils.h"

namespace PhaseRetrieval
{   
    // void preprocess_data(const std::string &inFileName, const std::string &inDatasetName, std::vector<hsize_t> &dims, const std::string &outFileName,
    //                          const std::string &outDatasetName, int kernelSize = 3, float threshold = 2.0f, bool applyFilter = false, int rangeRows = 0,
    //                          int rangeCols = 0, int movmeanSize = 5, const std::string &method = "multiplication");
    F2DArray reconstruct_iter(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelNumbers, int iterations, const FArray &initialPhase = FArray(),
                              ProjectionSolver::Algorithm algorithm = ProjectionSolver::AP, const FArray &algoParameters = FArray(), const IntArray &padSize = IntArray(),
                              float minPhase = -FloatInf, float maxPhase = FloatInf, float minAmplitude = 0.0f, float maxAmplitude = FloatInf, const IntArray &support = IntArray(),
                              float outsideValue = 1.0f, PMagnitudeCons::Type projectionType = PMagnitudeCons::Averaged, CUDAPropKernel::Type kernelType = CUDAPropKernel::Fourier,
                              CUDAUtils::PaddingType padType = CUDAUtils::PaddingType::Replicate, const FArray &holoProbes = FArray(), const FArray &initProbePhase = FArray(),
                              bool calcError = true);

    F2DArray reconstruct_bipepi(const FArray &holograms, int numImages, const IntArray &measSize, const F2DArray &fresnelNumbers, int iterations, const IntArray &imSize,
                                const FArray &initialPhase = FArray(), const FArray &initialAmplitude = FArray(), float minPhase = -FloatInf, float maxPhase = FloatInf,
                                float minAmplitude = 0.0f, float maxAmplitude = FloatInf, const IntArray &support = IntArray(), float outsideValue = 1.0f,
                                PMagnitudeCons::Type projectionType = PMagnitudeCons::Averaged, CUDAPropKernel::Type kernelType = CUDAPropKernel::Fourier,
                                bool calcError = true);
                              
    FArray reconstruct_ctf(const FArray &holograms, int numImages, const IntArray &imSize, const F2DArray &fresnelnumbers, float lowFreqLim = 1e-3f, float highFreqLim = 1e-1f,
                           float betaDeltaRatio = 0.0f, const IntArray &padSize = IntArray(), CUDAUtils::PaddingType padType = CUDAUtils::PaddingType::Replicate);

    class CTFReconstructor
    {
        private:
            int batchSize;
            int numImages;
            IntArray imSize;
            IntArray newSize;
            F2DArray fresnelNumbers;
            float betaDeltaRatio;
            IntArray padSize;
            CUDAUtils::PaddingType padType;

            float *d_holograms;
            float *d_paddedHolograms;
            float *d_temp;
            float *regWeights;
            float *d_regTemp;
            float *d_phase;
            float *d_croppedPhase;
            cudaStream_t *streams;
            
        public:
            CTFReconstructor(int batchsize, int images, const IntArray &imsize, const F2DArray &fresnelnumbers, float lowFreqLim, float highFreqLim,
                            float ratio, const IntArray &padsize, CUDAUtils::PaddingType padtype);
            FArray reconsBatch(const FArray &holograms);
            ~CTFReconstructor();
    };

    class Reconstructor
    {
        private:
            int batchSize;
            int numImages;
            IntArray imSize;
            IntArray newSize;
            int iteration;
            bool onlyAmpCons;
            std::vector<PropagatorPtr> propagators;
            ProjectionSolver::Algorithm algorithm;
            FArray algoParameters;
            
            IntArray padSize;
            CUDAUtils::PaddingType padType;
            Projector *pPhase;
            Projector *pAmplitude;
            Projector *pSupport;
            Projector *PS;
            PMagnitudeCons::Type projectionType;

            float *d_holograms;
            float *d_paddedHolograms;
            float *d_temp;
            float *d_phase;
            float *d_support;
            float *d_initPhase;
            float *d_paddedInitPhase;
            float *d_croppedPhase;
            cuFloatComplex *complexWave;
            cudaStream_t *streams;

        public:
            Reconstructor(int batchsize, int images, const IntArray &imsize, const F2DArray &fresnelNumbers, int iter, ProjectionSolver::Algorithm algo,
                          const FArray &algoParams, const IntArray &padsize, float minPhase, float maxPhase, float minAmplitude, float maxAmplitude, 
                          const IntArray &support, float outsideValue, PMagnitudeCons::Type projType, CUDAPropKernel::Type kernelType, CUDAUtils::PaddingType padtype);
            FArray reconsBatch(const FArray &holograms, const FArray &initialPhase);
            ~Reconstructor();
    };
}

#endif