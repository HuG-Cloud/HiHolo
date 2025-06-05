#include "io_utils.h"

bool IOUtils::readRawData(const std::string &filename, const std::vector<std::string> &datasetNames,
                          std::vector<hsize_t> &dims, U16Array &data, U16Array &dark, U16Array &flat)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
    
    // 打开H5文件
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    
    // 首先读取主数据集的维度信息
    dataset_id = H5Dopen2(file_id, datasetNames[0].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetNames[0] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetNames[0] << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    if (ndims != 3) {
        std::cerr << "Error: data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    dims.resize(ndims);
    status = H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions of dataspace" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    // 计算数据总大小
    hsize_t total_size = dims[0] * dims[1] * dims[2];
    
    // 分配内存并读取主数据
    data.resize(total_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    if (status < 0) {
        std::cerr << "Error reading dataset: " << datasetNames[0] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 读取dark数据集
    dataset_id = H5Dopen2(file_id, datasetNames[1].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dark dataset: " << datasetNames[1] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (H5Sget_simple_extent_ndims(dataspace_id) != 3) {
        std::cerr << "Error: dark data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    std::vector<hsize_t> dark_dims(ndims);
    H5Sget_simple_extent_dims(dataspace_id, dark_dims.data(), nullptr);
    
    // 计算dark数据大小
    hsize_t dark_size = dark_dims[0] * dark_dims[1] * dark_dims[2];
    
    dark.resize(dark_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dark.data());
    if (status < 0) {
        std::cerr << "Error reading dark dataset: " << datasetNames[1] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 读取flat数据集
    dataset_id = H5Dopen2(file_id, datasetNames[2].c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening flat dataset: " << datasetNames[2] << std::endl;
        H5Fclose(file_id);
        return false;
    }
    
    dataspace_id = H5Dget_space(dataset_id);
    if (H5Sget_simple_extent_ndims(dataspace_id) != 3) {
        std::cerr << "Error: flat data must be 3-dimensional" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    std::vector<hsize_t> flat_dims(ndims);
    H5Sget_simple_extent_dims(dataspace_id, flat_dims.data(), nullptr);
    
    // 计算flat数据大小
    hsize_t flat_size = flat_dims[0] * flat_dims[1] * flat_dims[2];
    
    flat.resize(flat_size);
    status = H5Dread(dataset_id, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    if (status < 0) {
        std::cerr << "Error reading flat dataset: " << datasetNames[2] << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }
    
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    
    // 关闭文件
    H5Fclose(file_id);
    return true;
}

bool IOUtils::readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 获取数据空间和维度
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    // 调整dims向量大小并获取维度信息
    dims.resize(ndims);
    status = H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
    if (status < 0) {
        std::cerr << "Error getting dimensions of dataspace for dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::readDataDims(const std::string &filename, const std::string &datasetName, std::vector<hsize_t> &dims, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t space_id = H5I_INVALID_HID;
    bool success = true;

    try {
        if (rank == 0) {
            // Open HDF5 file and dataset
            file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0) throw std::runtime_error("Cannot open file");

            dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
            if (dset_id < 0) throw std::runtime_error("Cannot open dataset");

            // Get dataspace and dimensions
            space_id = H5Dget_space(dset_id);
            if (space_id < 0) throw std::runtime_error("Cannot get dataspace");

            int ndims = H5Sget_simple_extent_ndims(space_id);
            dims.resize(ndims);
            H5Sget_simple_extent_dims(space_id, dims.data(), NULL);
        }

        // First broadcast number of dimensions
        int ndims = dims.size();
        MPI_Bcast(&ndims, 1, MPI_INT, 0, comm);
        
        // Non-root processes resize dims
        if (rank != 0) {
            dims.resize(ndims);
        }

        // Broadcast dimension information
        MPI_Bcast(dims.data(), ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);

    } catch (const std::exception& e) {
        std::cerr << "Error reading dataset dimensions: " << e.what() << std::endl;
        success = false;
    }

    // Clean up resources
    if (rank == 0) {
        if (space_id >= 0) H5Sclose(space_id);
        if (dset_id >= 0) H5Dclose(dset_id);
        if (file_id >= 0) H5Fclose(file_id);
    }

    // Ensure all processes get the same result
    int local_success = success ? 1 : 0;
    int global_success;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);

    return global_success == 1;
}

bool IOUtils::readProcessedGrams(const std::string &filename, const std::string &datasetName, FArray &holograms, std::vector<hsize_t> &dims)
{
    // 使用HDF5的C接口实现
    hid_t file_id, dataset_id, dataspace_id;

    // 打开H5文件和数据集
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 确保数据集维度是3D
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 3) {
        std::cerr << "Error: DataSet is not 3-dimensional!" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取维度信息并读取数据到FArray
    dims.resize(rank);
    H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

    holograms.resize(dims[0] * dims[1] * dims[2]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, holograms.data());

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::readPhaseGram(const std::string &filename, const std::string &datasetName, FArray &phase, std::vector<hsize_t> &dims)
{
    // 打开H5文件和数据集
    hid_t file_id, dataset_id, dataspace_id;
    
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error opening dataset: " << datasetName << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 确保数据集维度是2D
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0) {
        std::cerr << "Error getting dataspace for dataset: " << datasetName << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    int rank = H5Sget_simple_extent_ndims(dataspace_id);
    if (rank != 2) {
        std::cerr << "Error: DataSet is not 2-dimensional!" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    // 获取维度信息并读取数据到FArray
    dims.resize(rank);
    H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

    phase.resize(dims[0] * dims[1]);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phase.data());

    // 关闭资源
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::readSingleGram(const std::string &filename, const std::string &datasetName, 
                             U16Array &data, std::vector<hsize_t> &dims, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t space_id = H5I_INVALID_HID;
    bool success = true;

    try {
        if (rank == 0) {
            // 只有rank 0进程读取数据
            file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file_id < 0) throw std::runtime_error("Cannot open file");

            dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
            if (dset_id < 0) throw std::runtime_error("Cannot open dataset");

            space_id = H5Dget_space(dset_id);
            if (space_id < 0) throw std::runtime_error("Cannot get dataspace");

            // 获取维度信息
            int ndims = H5Sget_simple_extent_ndims(space_id);
            dims.resize(ndims);
            H5Sget_simple_extent_dims(space_id, dims.data(), NULL);

            // 读取数据
            size_t total_size = 1;
            for (const auto &dim : dims) {
                total_size *= dim;
            }
            data.resize(total_size);
            
            if (H5Dread(dset_id, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
                throw std::runtime_error("Cannot read dataset");
            }
        }

        // 广播维度信息
        int ndims = dims.size();
        MPI_Bcast(&ndims, 1, MPI_INT, 0, comm);
        
        if (rank != 0) {
            dims.resize(ndims);
        }
        MPI_Bcast(dims.data(), ndims, MPI_UNSIGNED_LONG_LONG, 0, comm);

        // 计算数据大小并广播数据
        size_t total_size = 1;
        for (const auto &dim : dims) {
            total_size *= dim;
        }
        
        if (rank != 0) {
            data.resize(total_size);
        }
        MPI_Bcast(data.data(), total_size, MPI_UNSIGNED_SHORT, 0, comm);

    } catch (const std::exception& e) {
        std::cerr << "Process " << rank << " Error reading dataset: " 
                  << e.what() << std::endl;
        success = false;
    }

    // 清理资源
    if (rank == 0) {
        if (space_id >= 0) H5Sclose(space_id);
        if (dset_id >= 0) H5Dclose(dset_id);
        if (file_id >= 0) H5Fclose(file_id);
    }

    // 确保所有进程得到相同的结果
    int local_success = success ? 1 : 0;
    int global_success;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);

    return global_success == 1;
}

bool IOUtils::savePhaseGram(const std::string &filename, const std::string &datasetName, const FArray &reconsPhase, int rows, int cols)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 创建文件
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return false;
    }

    // 创建数据空间
    hsize_t dims[2] {rows, cols};
    dataspace_id = H5Screate_simple(2, dims, nullptr);
    if (dataspace_id < 0) {
        std::cerr << "Error creating dataspace" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 创建数据集
    dataset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error creating dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reconsPhase.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::save3DGrams(const std::string &filename, const std::string &datasetName, const FArray &registeredGrams, int numImages, int rows, int cols)
{
    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    // 创建文件
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return false;
    }

    // 创建数据空间
    hsize_t dims[3] {numImages, rows, cols};
    dataspace_id = H5Screate_simple(3, dims, nullptr);
    if (dataspace_id < 0) {
        std::cerr << "Error creating dataspace" << std::endl;
        H5Fclose(file_id);
        return false;
    }

    // 创建数据集
    dataset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error creating dataset: " << datasetName << std::endl;
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, registeredGrams.data());
    if (status < 0) {
        std::cerr << "Error writing dataset" << std::endl;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return false;
    }

    // 关闭资源
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return true;
}

bool IOUtils::read3DimData(const std::string &filename, const std::string &datasetName,
                           FArray &data, hsize_t offset, hsize_t count, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;
    hid_t xfer_plist = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
        
        // Open file and dataset
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        
        // Get dataspace
        filespace = H5Dget_space(dset_id);
        hsize_t dims[3];
        H5Sget_simple_extent_dims(filespace, dims, NULL);
        
        // Set read region
        hsize_t offset_[3] = {offset, 0, 0};
        hsize_t count_[3] = {count, dims[1], dims[2]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);
        
        // Create memory space
        memspace = H5Screate_simple(3, count_, NULL);
        
        // Set collective data transfer properties
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        // Adjust data array size and read data
        if (H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, xfer_plist, data.data()) < 0) {
            throw std::runtime_error("Cannot read dataset");
        }
        
    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error reading dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (xfer_plist >= 0) H5Pclose(xfer_plist);
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}

bool IOUtils::read3DimData(const std::string &filename, const std::string &datasetName,
                           U16Array &data, hsize_t offset, hsize_t count)
{
    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;

    try {
        // 打开文件和数据集
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) throw std::runtime_error("Cannot open file");

        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dset_id < 0) throw std::runtime_error("Cannot open dataset");

        // 获取数据空间
        filespace = H5Dget_space(dset_id);
        if (filespace < 0) throw std::runtime_error("Cannot get dataspace");

        hsize_t dims[3];
        H5Sget_simple_extent_dims(filespace, dims, NULL);

        // 设置读取区域
        hsize_t offset_[3] = {offset, 0, 0};
        hsize_t count_[3] = {count, dims[1], dims[2]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);

        // 创建内存空间
        memspace = H5Screate_simple(3, count_, NULL);
        if (memspace < 0) throw std::runtime_error("Cannot create memory space");

        // 调整数据数组大小并读取数据
        data.resize(count * dims[1] * dims[2]);
        if (H5Dread(dset_id, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, data.data()) < 0) {
            throw std::runtime_error("Cannot read dataset");
        }

    } catch(const std::exception &error) {
        std::cerr << "Error reading dataset: " << error.what() << std::endl;
        return false;
    }

    // Clean up resources
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);

    return true;
}

bool IOUtils::read4DimData(const std::string &filename, const std::string &datasetName,
                           U16Array &data, hsize_t offset, hsize_t count)
{
    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;

    try {
        // 打开文件和数据集
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) throw std::runtime_error("Cannot open file");

        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dset_id < 0) throw std::runtime_error("Cannot open dataset");

        // 获取数据空间
        filespace = H5Dget_space(dset_id);
        if (filespace < 0) throw std::runtime_error("Cannot get dataspace");

        hsize_t dims[4];
        H5Sget_simple_extent_dims(filespace, dims, NULL);

        // 设置读取区域
        hsize_t offset_[4] = {offset, 0, 0, 0};
        hsize_t count_[4] = {count, dims[1], dims[2], dims[3]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);

        // 创建内存空间
        memspace = H5Screate_simple(4, count_, NULL);
        if (memspace < 0) throw std::runtime_error("Cannot create memory space");

        // 调整数据数组大小并读取数据
        data.resize(count * dims[1] * dims[2] * dims[3]);
        if (H5Dread(dset_id, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, data.data()) < 0) {
            throw std::runtime_error("Cannot read dataset");
        }

    } catch(const std::exception &error) {
        std::cerr << "Error reading dataset: " << error.what() << std::endl;
        return false;
    }

    // Clean up resources
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);

    return true;
}

bool IOUtils::read4DimData(const std::string &filename, const std::string &datasetName,
                           FArray &data, hsize_t offset, hsize_t count, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;
    hid_t xfer_plist = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
        
        // Open file and dataset
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        
        // Get dataspace
        filespace = H5Dget_space(dset_id);
        hsize_t dims[4];
        H5Sget_simple_extent_dims(filespace, dims, NULL);
        
        // Set read region
        hsize_t offset_[4] = {offset, 0, 0, 0};
        hsize_t count_[4] = {count, dims[1], dims[2], dims[3]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);
        
        // Create memory space
        memspace = H5Screate_simple(4, count_, NULL);
        
        // Set collective data transfer properties
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        // Adjust data array size and read data
        if (H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, xfer_plist, data.data()) < 0) {
            throw std::runtime_error("Cannot read dataset");
        }
        
    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error reading dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (xfer_plist >= 0) H5Pclose(xfer_plist);
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}

bool IOUtils::read4DimData(const std::string &filename, const std::string &datasetName,
                           U16Array &data, hsize_t offset, hsize_t count, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;
    hid_t xfer_plist = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
        
        // Open file and dataset
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        
        // Get dataspace
        filespace = H5Dget_space(dset_id);
        hsize_t dims[4];
        H5Sget_simple_extent_dims(filespace, dims, NULL);
        
        // Set read region
        hsize_t offset_[4] = {offset, 0, 0, 0};
        hsize_t count_[4] = {count, dims[1], dims[2], dims[3]};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);
        
        // Create memory space
        memspace = H5Screate_simple(4, count_, NULL);
        
        // Set collective data transfer properties
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        // Adjust data array size and read data
        if (H5Dread(dset_id, H5T_NATIVE_UINT16, memspace, filespace, xfer_plist, data.data()) < 0) {
            throw std::runtime_error("Cannot read dataset");
        }
        
    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error reading dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (xfer_plist >= 0) H5Pclose(xfer_plist);
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}

bool IOUtils::createFileDataset(const std::string &filename, const std::string &datasetName, const std::vector<hsize_t> &dims, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

        // All processes participate in file creation
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        if (file_id < 0) throw std::runtime_error("Cannot create file");

        // Create file space
        filespace = H5Screate_simple(dims.size(), dims.data(), NULL);
        if (filespace < 0) throw std::runtime_error("Cannot create file space");

        // Create dataset
        dset_id = H5Dcreate2(file_id, datasetName.c_str(), H5T_NATIVE_FLOAT, filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dset_id < 0) throw std::runtime_error("Cannot create dataset");

    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error creating dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}

bool IOUtils::write3DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;
    hid_t xfer_plist = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

        // Open existing file
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
        if (file_id < 0) throw std::runtime_error("Cannot open file");
        
        // Open existing dataset
        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dset_id < 0) throw std::runtime_error("Cannot open dataset");
        
        // Set write region
        filespace = H5Dget_space(dset_id);
        if (filespace < 0) throw std::runtime_error("Cannot get file space");
        
        hsize_t count_[3] = {data.size() / (dims[1] * dims[2]), dims[1], dims[2]};
        hsize_t offset_[3] = {offset, 0, 0};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);
        
        // Create memory space
        memspace = H5Screate_simple(3, count_, NULL);
        if (memspace < 0) throw std::runtime_error("Cannot create memory space");
        
        // Set collective data transfer properties
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        // Write data
        if (H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, xfer_plist, data.data()) < 0) {
            throw std::runtime_error("Cannot write dataset");
        }
        
    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error writing dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (xfer_plist >= 0) H5Pclose(xfer_plist);
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}

bool IOUtils::write4DimData(const std::string &filename, const std::string &datasetName, const FArray &data, 
                            const std::vector<hsize_t> &dims, hsize_t offset, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    hid_t file_id = H5I_INVALID_HID;
    hid_t dset_id = H5I_INVALID_HID;
    hid_t filespace = H5I_INVALID_HID;
    hid_t memspace = H5I_INVALID_HID;
    hid_t plist_id = H5I_INVALID_HID;
    hid_t xfer_plist = H5I_INVALID_HID;

    try {
        // Create and set parallel access properties
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

        // Open existing file
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id);
        if (file_id < 0) throw std::runtime_error("Cannot open file");
        
        // Open existing dataset
        dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dset_id < 0) throw std::runtime_error("Cannot open dataset");
        
        // Set write region
        filespace = H5Dget_space(dset_id);
        if (filespace < 0) throw std::runtime_error("Cannot get file space");
        
        hsize_t count_[4] = {data.size() / (dims[1] * dims[2] * dims[3]), dims[1], dims[2], dims[3]};
        hsize_t offset_[4] = {offset, 0, 0, 0};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_, NULL, count_, NULL);
        
        // Create memory space
        memspace = H5Screate_simple(4, count_, NULL);
        if (memspace < 0) throw std::runtime_error("Cannot create memory space");
        
        // Set collective data transfer properties
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        
        // Write data
        if (H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, xfer_plist, data.data()) < 0) {
            throw std::runtime_error("Cannot write dataset");
        }
        
    } catch(const std::exception &error) {
        std::cerr << "Process " << rank << " Error writing dataset: " << error.what() << std::endl;
        MPI_Abort(comm, 1);
    }

    // Clean up resources
    if (xfer_plist >= 0) H5Pclose(xfer_plist);
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset_id >= 0) H5Dclose(dset_id);
    if (file_id >= 0) H5Fclose(file_id);
    if (plist_id >= 0) H5Pclose(plist_id);

    return true;
}