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
                          float minAmplitude = 0.0f, float maxAmplitude = FloatInf, PMagnitudeCons::Type projectionType = PMagnitudeCons::Averaged, 
                          CUDAPropKernel::Type kernelType = CUDAPropKernel::Fourier, CUDAUtils::PaddingType padType = CUDAUtils::PaddingType::Replicate, bool calcError = false);
    F2DArray reconstruct_ctf(const FArray &holograms, const IntArray &imSize, const F2DArray &fresnelNumbers);
}

#endif
