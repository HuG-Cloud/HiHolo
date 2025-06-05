#ifndef IO_UTILS_H_
#define IO_UTILS_H_

#include <iostream>
#include <mpi.h>
#include <hdf5.h>

#include "datatypes.h"

namespace IOUtils
{
    bool readRawData(const std::string &filename, const std::vector<std::string> &datasetNames,
                     std::vector<hsize_t> &dims, U16Array &data, U16Array &dark, U16Array &flat);
    bool readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims, MPI_Comm comm);
    bool readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims);
    bool readPhaseGram(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims);
    bool readSingleGram(const std::string &filename, const std::string &datasetName, U16Array &data, std::vector<hsize_t> &dims, MPI_Comm comm);
    bool readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims);
    bool savePhaseGram(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols);
    bool save3DGrams(const std::string &filename, const std::string &datasetName, const FArray &registeredGrams, int numImages, int rows, int cols);
    bool read3DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count, MPI_Comm comm);
    bool read3DimData(const std::string &filename, const std::string &datasetName, U16Array &data, hsize_t offset, hsize_t count);
    bool read4DimData(const std::string &filename, const std::string &datasetName, U16Array &data, hsize_t offset, hsize_t count);
    bool read4DimData(const std::string &filename, const std::string &datasetName, FArray &data, hsize_t offset, hsize_t count, MPI_Comm comm);
    bool read4DimData(const std::string &filename, const std::string &datasetName, U16Array &data, hsize_t offset, hsize_t count, MPI_Comm comm);
    bool createFileDataset(const std::string &filename, const std::string &datasetName, const std::vector<hsize_t> &dims, MPI_Comm comm);
    
    bool write3DimData(const std::string &filename, const std::string &datasetName, const FArray &data,
                       const std::vector<hsize_t> &dims, hsize_t offset, MPI_Comm comm);
    bool write4DimData(const std::string &filename, const std::string &datasetName, const FArray &data,
                       const std::vector<hsize_t> &dims, hsize_t offset, MPI_Comm comm);
}

#endif