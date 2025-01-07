#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>
#include "Eigen.h"
#include "nlohmann/json.hpp"

//WIP
struct BfmProperties {

    //General Properties:
    int numberOfVertices{};
    int numberOfTriangles{};
    std::vector<Eigen::Vector3f> vertices;
    std::vector<int> triangles;
    std::vector<Eigen::Vector3f> landmarks;
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

static std::vector<Eigen::Vector3f> getVertices(BfmProperties properties){
    std::vector<Eigen::Vector3f> vertices;
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3f newVertex;
        newVertex.x() = properties.shapeMean[i] + properties.expressionMean[i];
        newVertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1];
        newVertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2];
        vertices.emplace_back(newVertex);
    }
    return vertices;
}

static std::vector<Eigen::Vector3i> getColorValues(BfmProperties properties){
    std::vector<Eigen::Vector3i> colorValues;
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3i newColorValue;
        newColorValue.x() = (int) (properties.colorMean[i] * 255);
        newColorValue.y() = (int) (properties.colorMean[i + 1] * 255);
        newColorValue.z() = (int) (properties.colorMean[i + 2] * 255);
        colorValues.emplace_back(newColorValue);
    }
    return colorValues;
}

static void readHDF5Data(const H5::H5File& file, const std::string& groupPath, const std::string& datasetPath, std::vector<float>& target) {
    try {
        H5::Group group = file.openGroup(groupPath);
        H5::DataSet dataset = group.openDataSet(datasetPath);

        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims, nullptr);
        target.resize(dims[0]);

        dataset.read(target.data(), H5::PredType::NATIVE_FLOAT);
    } catch (H5::Exception& e) {
        std::cerr << "Error reading BFM parameters: " << e.getDetailMsg() << std::endl;
    }
}

/*static void extractLandmarks(const H5::H5File& file, const BfmProperties& properties){
    try {
        H5::Group group = file.openGroup("/metadata/landmarks");
        H5::DataSet dataset = group.openDataSet("json");
        H5::StrType strType = dataset.getStrType();
        ssize_t size = strType.getSize();
        char* buffer = new char[size + 1];
        dataset.read(buffer, strType);
        buffer[size] = '\0';
        std::string jsonString(buffer);
        delete[] buffer;

        nlohmann::json json = nlohmann::json::parse(jsonString);

        std::vector<Eigen::Vector3f> coordinates;


        // Iterate through the JSON array and extract coordinates
        for (const auto& landmark : json) {

        }
        // For demonstration, print the coordinates
        for (const auto& vec : coordinates) {
            std::cout << "Coordinates: [" << vec.x() << ", " << vec.y() << ", " << vec.z() << "]" << std::endl;
        }
    } catch (H5::Exception& e) {
        std::cerr << "Error reading BFM parameters: " << e.getDetailMsg() << std::endl;
    }
    std::cout << properties.landmarks.size() << std::endl;
}*/

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

    std::vector<Eigen::Vector3f> landmarks = {
            {-73919.3, 30876.3, 19849.9},
            {-70737.7, 5242.63, 23751.0},
            {-67043.4, -16524.0, 30461.0},
            {-63378.7, -32740.5, 38981.6},
            {-55343.4, -50362.6, 48677.7},
            {-42926.8, -63089.2, 63733.5},
            {-27970.4, -71516.9, 81654.7},
            {-16291.4, -74698.6, 96297.7},
            {-48.2388, -76505.2, 99442.1},
            {16255.0, -74471.6, 96308.6},
            {27908.9, -71417.7, 81694.5},
            {42855.4, -63229.9, 63746.2},
            {56170.0, -49032.8, 48846.8},
            {63225.8, -33092.0, 38968.2},
            {66704.6, -16882.1, 30449.8},
            {70054.7, 4963.69, 23643.8},
            {73154.5, 30854.0, 20055.8},
            {-57346.8, 45510.3, 80483.7},
            {-49635.7, 51531.3, 92479.5},
            {-38958.6, 53447.6, 100454.0},
            {-29608.2, 52236.6, 104277.0},
            {-18702.9, 50264.2, 106598.0},
            {17860.7, 50466.9, 106508.0},
            {28544.3, 51538.8, 103733.0},
            {38062.0, 52791.3, 100413.0},
            {49630.6, 50820.8, 91989.9},
            {56468.8, 46294.8, 81118.2},
            {-146.369, 34128.1, 110560.0},
            {-116.763, 24000.5, 118205.0},
            {-20.8564, 14347.0, 125517.0},
            {13.2323, 4430.29, 131560.0},
            {-11574.5, -7575.11, 110567.0},
            {-5116.18, -7991.94, 114465.0},
            {-164.637, -9165.1, 117358.0},
            {4602.21, -8039.56, 114443.0},
            {11130.6, -7555.56, 110556.0},
            {-44627.6, 33016.0, 85710.8},
            {-37033.8, 36953.1, 92643.3},
            {-27261.5, 37833.3, 94881.4},
            {-18330.4, 33041.4, 91429.2},
            {-29058.9, 29625.8, 94046.1},
            {-37779.7, 29124.5, 91864.5},
            {18462.4, 33379.7, 92001.2},
            {27972.3, 38007.0, 94475.5},
            {37036.2, 36535.4, 91911.6},
            {45188.7, 33094.9, 85347.9},
            {38052.3, 29349.8, 91259.8},
            {29458.6, 29444.7, 93770.8},
            {-25539.6, -33627.8, 98388.9},
            {-15805.3, -27611.8, 110100.0},
            {-6628.37, -23768.2, 115463.0},
            {-192.025, -24960.5, 116208.0},
            {7616.04, -23944.8, 115018.0},
            {17230.8, -27470.0, 108906.0},
            {25143.8, -33333.3, 98094.7},
            {18575.5, -36133.4, 104735.0},
            {10856.5, -38386.0, 111107.0},
            {-175.34, -39321.0, 113471.0},
            {-11149.0, -38397.3, 111204.0},
            {-18695.9, -35550.9, 104488.0},
            {-22495.2, -32862.2, 99654.2},
            {-8644.35, -30469.4, 109984.0},
            {-279.868, -30878.0, 111425.0},
            {8088.77, -30539.5, 109996.0},
            {23331.7, -32803.8, 98654.9},
            {8140.39, -31052.9, 109408.0},
            {-257.674, -31426.9, 110736.0},
            {-9651.11, -30987.6, 108781.0},
    };

    for (int i = 0; i < landmarks.size(); ++i) {
        landmarks[i] /= 1000;
    }
    properties.landmarks = landmarks;
}



#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H