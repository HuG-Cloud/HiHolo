#include <iostream>
#include <cstring>
#include "H5Cpp.h"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input file> <input dataset> <output file> <output dataset>" << std::endl;
        return 1;
    }

    try {
        // Open input file and dataset
        H5::H5File inFile(argv[1], H5F_ACC_RDONLY);
        H5::DataSet inDataset = inFile.openDataSet(argv[2]);

        // Get input data dimensions
        H5::DataSpace inSpace = inDataset.getSpace();
        int ndims = inSpace.getSimpleExtentNdims();
        if (ndims != 3) {
            std::cout << "Input data must be 3-dimensional!" << std::endl;
            return 1;
        }

        hsize_t dims[3];
        inSpace.getSimpleExtentDims(dims, NULL);
        
        // Read input data
        float* inData = new float[dims[0] * dims[1] * dims[2]];
        inDataset.read(inData, H5::PredType::NATIVE_FLOAT);

        // Create output file
        H5::H5File outFile(argv[3], H5F_ACC_TRUNC);
        
        // Set output data dimensions (add angle dimension)
        const int numAngles = 200; // Set number of angles
        hsize_t outDims[4] = {numAngles, dims[0], dims[1], dims[2]};
        H5::DataSpace outSpace(4, outDims);
        
        // Create output dataset
        H5::DataSet outDataset = outFile.createDataSet(argv[4], 
                                                      H5::PredType::NATIVE_FLOAT,
                                                      outSpace);

        // Allocate output data memory
        float* outData = new float[numAngles * dims[0] * dims[1] * dims[2]];
        
        // Copy data multiple times
        for (int angle = 0; angle < numAngles; angle++) {
            size_t offset = angle * dims[0] * dims[1] * dims[2];
            std::memcpy(outData + offset, inData, 
                       dims[0] * dims[1] * dims[2] * sizeof(float));
        }

        // Write data
        outDataset.write(outData, H5::PredType::NATIVE_FLOAT);

        // Cleanup
        delete[] inData;
        delete[] outData;
        
        std::cout << "Successfully expanded 3D data to 4D data!" << std::endl;
        std::cout << "Output dimensions: " << numAngles << " x " 
                  << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << std::endl;
        return 1;
    }

    return 0;
}
