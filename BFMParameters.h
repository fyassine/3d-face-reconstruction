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
            {-73919.3f, 30876.3f, 19849.9f},
            {-70737.7f, 5242.63f, 23751.0f},
            {-67043.4f, -16524.0f, 30461.0f},
            {-63378.7f, -32740.5f, 38981.6f},
            {-55343.4f, -50362.6f, 48677.7f},
            {-42926.8f, -63089.2f, 63733.5f},
            {-27970.4f, -71516.9f, 81654.7f},
            {-16291.4f, -74698.6f, 96297.7f},
            {-48.2388f, -76505.2f, 99442.1f},
            {16255.0f, -74471.6f, 96308.6f},
            {27908.9f, -71417.7f, 81694.5f},
            {42855.4f, -63229.9f, 63746.2f},
            {56170.0f, -49032.8f, 48846.8f},
            {63225.8f, -33092.0f, 38968.2f},
            {66704.6f, -16882.1f, 30449.8f},
            {70054.7f, 4963.69f, 23643.8f},
            {73154.5f, 30854.0f, 20055.8f},
            {-57346.8f, 45510.3f, 80483.7f},
            {-49635.7f, 51531.3f, 92479.5f},
            {-38958.6f, 53447.6f, 100454.0f},
            {-29608.2f, 52236.6f, 104277.0f},
            {-18702.9f, 50264.2f, 106598.0f},
            {17860.7f, 50466.9f, 106508.0f},
            {28544.3f, 51538.8f, 103733.0f},
            {38062.0f, 52791.3f, 100413.0f},
            {49630.6f, 50820.8f, 91989.9f},
            {56468.8f, 46294.8f, 81118.2f},
            {-146.369f, 34128.1f, 110560.0f},
            {-116.763f, 24000.5f, 118205.0f},
            {-20.8564f, 14347.0f, 125517.0f},
            {13.2323f, 4430.29f, 131560.0f},
            {-11574.5f, -7575.11f, 110567.0f},
            {-5116.18f, -7991.94f, 114465.0f},
            {-164.637f, -9165.1f, 117358.0f},
            {4602.21f, -8039.56f, 114443.0f},
            {11130.6f, -7555.56f, 110556.0f},
            {-44627.6f, 33016.0f, 85710.8f},
            {-37033.8f, 36953.1f, 92643.3f},
            {-27261.5f, 37833.3f, 94881.4f},
            {-18330.4f, 33041.4f, 91429.2f},
            {-29058.9f, 29625.8f, 94046.1f},
            {-37779.7f, 29124.5f, 91864.5f},
            {18462.4f, 33379.7f, 92001.2f},
            {27972.3f, 38007.0f, 94475.5f},
            {37036.2f, 36535.4f, 91911.6f},
            {45188.7f, 33094.9f, 85347.9f},
            {38052.3f, 29349.8f, 91259.8f},
            {29458.6f, 29444.7f, 93770.8f},
            {-25539.6f, -33627.8f, 98388.9f},
            {-15805.3f, -27611.8f, 110100.0f},
            {-6628.37f, -23768.2f, 115463.0f},
            {-192.025f, -24960.5f, 116208.0f},
            {7616.04f, -23944.8f, 115018.0f},
            {17230.8f, -27470.0f, 108906.0f},
            {25143.8f, -33333.3f, 98094.7f},
            {18575.5f, -36133.4f, 104735.0f},
            {10856.5f, -38386.0f, 111107.0f},
            {-175.34f, -39321.0f, 113471.0f},
            {-11149.0f, -38397.3f, 111204.0f},
            {-18695.9f, -35550.9f, 104488.0f},
            {-22495.2f, -32862.2f, 99654.2f},
            {-8644.35f, -30469.4f, 109984.0f},
            {-279.868f, -30878.0f, 111425.0f},
            {8088.77f, -30539.5f, 109996.0f},
            {23331.7f, -32803.8f, 98654.9f},
            {8140.39f, -31052.9f, 109408.0f},
            {-257.674f, -31426.9f, 110736.0f},
            {-9651.11f, -30987.6f, 108781.0f},
    };


    for (int i = 0; i < landmarks.size(); ++i) {
        landmarks[i] /= 1000;
    }
    properties.landmarks = landmarks;
}



#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H