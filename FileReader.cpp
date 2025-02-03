#include <iostream>
#include <fstream>
#include "FileReader.h"

FileReader::FileReader() = default;
FileReader::~FileReader() = default;

std::vector<Vector3d> FileReader::read3DVerticesFromTxt(const std::string& path) {
    const std::string inputFile = std::string(dataFolderPath + path);
    std::ifstream inFile(inputFile);
    std::string line;
    std::vector<Eigen::Vector3d> vertices;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        float first, second, third;
        if (iss >> first >> second >> third) {
            Eigen::Vector3f newVertex;
            newVertex.x() = first;
            newVertex.y() = second;
            newVertex.z() = third;
            vertices.emplace_back(newVertex);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return vertices;
}

std::vector<int> FileReader::readIntFromTxt(const std::string &path) {
    std::vector<int> result;
    const std::string inputFile = std::string(dataFolderPath + path);
    std::ifstream inFile(inputFile);
    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        int value;
        if (iss >> value) {
            result.push_back(value);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return result;
}

std::vector<double> FileReader::readHDF5File(const std::string &filePath, const std::string &groupPath, const std::string &datasetPath) {

    H5::H5File file(dataFolderPath + filePath, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    std::vector<double> target;

    try {
        H5::Group group = file.openGroup(groupPath);
        H5::DataSet dataset = group.openDataSet(datasetPath);
        H5::DataSpace dataSpace = dataset.getSpace();
        hsize_t dims[1];
        dataSpace.getSimpleExtentDims(dims, nullptr);
        target.resize(dims[0]);
        dataset.read(target.data(), H5::PredType::NATIVE_FLOAT);
    } catch (H5::Exception& e) {
        std::cerr << "Error reading BFM parameters: " << e.getDetailMsg() << std::endl;
    }
    file.close();
    return target;
}

MatrixXd FileReader::readMatrixHDF5File(const std::string &filePath, const std::string &groupPath,
                                        const std::string &datasetPath) {

    H5::H5File file(dataFolderPath + filePath, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    MatrixXd matrix;

    try {
        H5::Group group = file.openGroup(groupPath);
        H5::DataSet dataset = group.openDataSet(datasetPath);
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 2) {
            throw std::runtime_error("Dataset is not 2-dimensional.");
        }
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, nullptr);
        matrix.resize(static_cast<int>(dims[0]), static_cast<int>(dims[1]));
        std::vector<double> buffer(static_cast<size_t>(dims[0] * dims[1]));
        dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
        matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                buffer.data(), static_cast<int>(dims[0]), static_cast<int>(dims[1]));
    } catch (H5::Exception& e) {
        std::cerr << "Error reading HDF5 dataset: " << e.getDetailMsg() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    file.close();
    return matrix;
}
