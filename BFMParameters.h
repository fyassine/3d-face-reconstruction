#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>
#include "Eigen.h"

//WIP
struct BfmProperties {

    //General Properties:
    int numberOfVertices{};
    int numberOfTriangles{};
    std::vector<Eigen::Vector3f> vertices;
    std::vector<int> triangles;
    //TODO: landmarks?!

    //Parameters:
    std::vector<float> colorMean;
    std::vector<float> colorPcaBasis;
    std::vector<float> colorPcaVariance;

    std::vector<float> shapeMean;
    std::vector<float> shapePcaBasis;
    std::vector<float> shapePcaVariance;

    std::vector<float> expressionMean;
    std::vector<float> expressionPcaBasis;
    std::vector<float> expressionPcaVariance;
};

static void readHDF5Data(const H5::H5File& file, const std::string& groupPath, const std::string& datasetPath, std::vector<float>& target) {
    try {
        H5::Group group = file.openGroup(groupPath);
        H5::DataSet dataset = group.openDataSet(datasetPath);

        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims, nullptr); // Assuming 1D data
        target.resize(dims[0]);

        dataset.read(target.data(), H5::PredType::NATIVE_FLOAT);
    } catch (H5::Exception& e) {
        std::cerr << "Error reading BFM parameters: " << e.getDetailMsg() << std::endl;
    }
}

//@param path -> path to .h5 file
//initializeMethod durch constructor ersetzen?!
static void initializeBFM(const std::string& path, BfmProperties& properties){

    //BfmProperties properties;
    H5::H5File file(path, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    //Read Shape
    readHDF5Data(file, "/shape/model", "mean", properties.shapeMean);
    //readData(file, "/shape/model", "pcaBasis", properties.shapePcaBasis); //TODO: PCABasis for Modifications?!
    readHDF5Data(file, "/shape/model", "pcaVariance", properties.shapePcaVariance);
    //Read Expression
    readHDF5Data(file, "/color/model", "mean", properties.colorMean);
    //readData(file, "/color/model", "pcaBasis", properties.colorPcaBasis);
    readHDF5Data(file, "/color/model", "pcaVariance", properties.colorPcaVariance);
    //Read Color
    readHDF5Data(file, "/expression/model", "mean", properties.expressionMean);
    //readData(file, "/expression/model", "pcaBasis", properties.expressionPcaBasis);
    readHDF5Data(file, "/expression/model", "pcaVariance", properties.expressionPcaVariance);


    properties.numberOfVertices = properties.shapeMean.size() / 3;
    std::cout << "Vertices: " << properties.numberOfVertices << std::endl;
    std::cout << "Color Mean: " << properties.colorMean.size() << " values" << std::endl;
    std::cout << "Shape Variance: " << properties.shapePcaVariance.size() << " values" << std::endl;

    //Faces
    const std::string inputFile = std::string("../../../Data/faces.txt");
    std::ifstream inFile(inputFile);
    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        int firstInt;
        int secondInt, thirdInt, fourthInt;
        if (iss >> firstInt >> secondInt >> thirdInt >> fourthInt) {
            properties.triangles.push_back(secondInt);
            properties.triangles.push_back(thirdInt);
            properties.triangles.push_back(fourthInt);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    properties.numberOfTriangles = properties.triangles.size() / 3;
    std::cout << "Faces: " << properties.numberOfTriangles << std::endl;
}

/*std::vector<float> readPrincipalComponents(const H5::Group& group, const std::string& datasetName) {
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
    extractShapeParameters(file);
    extractExpressionParameters(file);
    extractColorParameters(file);
}

void readModelPath(const std::string& filename){
    H5::H5File file(filename, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    try {
        H5::Group group = file.openGroup("catalog/MorphableModel");
        H5::DataSet dataset = group.openDataSet("modelPath");
        H5::DataSpace dataspace = dataset.getSpace();
        int size = dataspace.getSimpleExtentNpoints();
        std::vector<char> model_path(size);
        dataset.read(model_path.data(), H5::PredType::C_S1);
        std::cout << "ModelPath: " << std::string(model_path.begin(), model_path.end()) << std::endl;

    } catch (H5::Exception& e) {
        std::cerr << "Error opening Path" << e.getDetailMsg() << std::endl;
    }
}*/

/*static double* get_vertices(BFM bfm, Parameters params) {
    MatrixXd shape_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXd shape_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXd exp_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.exp_pca_var, 100, 1);
    MatrixXd exp_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.exp_pca_basis, 85764, 100);

    MatrixXd shape_result = shape_pca_basis * (shape_pca_var.cwiseSqrt().cwiseProduct(params.shape_weights));
    MatrixXd exp_result = exp_pca_basis * (exp_pca_var.cwiseSqrt().cwiseProduct(params.exp_weights));
    MatrixXd result = shape_result + exp_result;
    MatrixXd mapped_vertices = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_mean, 85764, 1);

    double* sum = new double[85764];
    for (int i = 0; i < 85764; i ++) {
        sum[i] = bfm.shape_mean[i] + bfm.exp_mean[i] + result(i);
    }
    return sum;
}*/

#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H