#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>

std::vector<float> readPrincipalComponents(const H5::Group& group, const std::string& datasetName) {
    H5::DataSet dataset = group.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims, nullptr);

    // Ensure dataset has two dimensions
    if (dims[0] <= 0 || dims[1] <= 0) {
        std::cerr << "Invalid dimensions for dataset '" << datasetName << "'" << std::endl;
        throw std::runtime_error("Invalid dataset dimensions");
    }
    std::cout << "Dataset '" << datasetName << "' dimensions: " << dims[0] << " x " << dims[1] << std::endl;

    // Allocate memory for PCA components (the second dimension)
    std::vector<float> principalComponents(dims[1]);

    // Define hyperslab to read only the PCA components (first row)
    hsize_t offset[2] = {0, 0};
    hsize_t count[2] = {1, dims[1]};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // Define memory space for 1x(dim[1])
    hsize_t memDims[2] = {1, dims[1]};
    H5::DataSpace memSpace(2, memDims);

    // Read data
    dataset.read(principalComponents.data(), H5::PredType::NATIVE_FLOAT, memSpace, dataspace);

    return principalComponents;
}

std::vector<float> extractParameters(const H5::H5File& file, const std::string& path){
    std::vector<float> parameters;
    try {
        H5::Group group = file.openGroup(path);
        parameters = readPrincipalComponents(group, "pcaBasis");
    } catch (H5::Exception& e) {
        std::cerr << "Error reading BFM parameters: " << e.getDetailMsg() << std::endl;
        return parameters;
    }
    std::cout << "Shape Parameters: " << parameters.size() << " values" << std::endl;
    return parameters;
}

std::vector<float> extractShapeParameters(const H5::H5File& file){
    return extractParameters(file, "/shape/model");
}

std::vector<float> extractExpressionParameters(const H5::H5File& file){
    return extractParameters(file, "/expression/model");
}

std::vector<float> extractColorParameters(const H5::H5File& file){
    return extractParameters(file, "/color/model");
}

void readH5File(const std::string& filename){
    H5::H5File file(filename, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    auto x = extractShapeParameters(file);
    extractExpressionParameters(file);
    extractColorParameters(file);
}



#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H