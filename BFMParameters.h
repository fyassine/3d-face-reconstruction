#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>
#include "Eigen.h"
#include "nlohmann/json.hpp"
#include "ProcrustesAligner.h"

//WIP
struct BfmProperties {

    //General Properties:
    int numberOfVertices{};
    int numberOfTriangles{};
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> initialOffsets;
    Eigen::Vector3f initialOffset;
    std::vector<int> triangles;
    std::vector<Eigen::Vector3f> landmarks;
    //TODO: landmarks?!

    Matrix4f transformation;

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

    std::vector<float> colorWeight;
    std::vector<float> shapeWeight;
    std::vector<float> expressionWeight;
};

/*static void setInitialOffset(Eigen::Vector3f initialOffset, BfmProperties properties){
    for (int i = 0; i < properties.shapeMean.size() / 3; ++i) {
        Eigen::Vector3f offsetVector;
        properties.initialOffsets[i].x() += initialOffset.x();
        properties.initialOffsets[i].y() += initialOffset.y();
        properties.initialOffsets[i].z() += initialOffset.z();
    }
}*/

static void setInitialOffset(Eigen::Vector3f initialOffset, BfmProperties& properties) {
    properties.initialOffset.x() = initialOffset.x();
    std::cout << "InitialOffset:" << properties.initialOffset.x() << std::endl;
    properties.initialOffset.y() = initialOffset.y();
    properties.initialOffset.z() = initialOffset.z();
}

static std::vector<Eigen::Vector3f> getVertices(BfmProperties properties){
    std::vector<Eigen::Vector3f> vertices;
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3f newVertex;
        newVertex.x() = properties.shapeMean[i] + properties.expressionMean[i] + properties.initialOffset.x();
        if(i==0){
            std::cout << "With InitialOffset:" << newVertex.x() << std::endl;
            std::cout << "Without InitialOffset:" << properties.shapeMean[i] + properties.expressionMean[i] << std::endl;
            std::cout << "InitialOffset:" << properties.initialOffset.x() << std::endl;
        }
        newVertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1] + properties.initialOffset.y();
        newVertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2] + properties.initialOffset.z();
        Eigen::Vector4f transformationVector;
        if(i == 0){
            std::cout << "Old Vertex: " << newVertex.x() << ", " << newVertex.y() << ", " << newVertex.z() << " VS. ";
        }
        transformationVector.x() = newVertex.x();
        transformationVector.y() = newVertex.y();
        transformationVector.z() = newVertex.z();
        transformationVector.w() = 1.0f;
        transformationVector = properties.transformation * transformationVector;
        //newVertex.x() = transformationVector.x();
        //newVertex.y() = transformationVector.y();
        //newVertex.z() = transformationVector.z();
        if(i == 0){
            std::cout << "New Vertex: " << newVertex.x() << ", " << newVertex.y() << ", " << newVertex.z() << ";" << std::endl;
        }
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

Eigen::Vector3f convert2Dto3D(const Eigen::Vector2f& pt_2d, float depth, const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T, const Eigen::Matrix4f& extrinsics) {
    /*// Step 1: Invert the Intrinsic Matrix K
    Eigen::Matrix3f K_inv = K.inverse();

    // Step 2: Create a homogeneous 2D point (x, y, 1)
    Eigen::Vector3f pt_2d_homogeneous(pt_2d.x(), pt_2d.y(), 1.0f);

    // Step 3: Perform scalar multiplication with depth first, then matrix multiplication
    Eigen::Vector3f pt_3d_camera = depth * pt_2d_homogeneous; // Depth multiplied with the homogeneous 2D point
    pt_3d_camera = K_inv * pt_3d_camera;  // Now multiply with the inverse of K

    // Step 4: Apply the extrinsic transformation (rotation R and translation T)
    Eigen::Vector3f pt_3d_world = R * pt_3d_camera + T;

    return pt_3d_world;*/

    float Xcoord = (pt_2d.x() - K(0,2)) / K(0,0); // Xc
    float Ycoord = (pt_2d.y() - K(1,2)) / K(1,1); // Yc
    float Zcoord = depth; // Zc

    // Step 2: Create camera coordinates
    Eigen::Vector3f cameraCoords(Xcoord * Zcoord, Ycoord * Zcoord, Zcoord);

    // Step 3: Apply extrinsic transformation (inverse of extrinsics already includes rotation and translation)
    Eigen::Vector4f pt_3d_world_4f = extrinsics.inverse() * Eigen::Vector4f(cameraCoords.x(), cameraCoords.y(), cameraCoords.z(), 1);

    // Step 4: Convert to 3D world coordinates
    Eigen::Vector3f pt_3d_world(pt_3d_world_4f.x(), pt_3d_world_4f.y(), pt_3d_world_4f.z());
    std::cout << "Target: " << pt_3d_world << std::endl;

    return pt_3d_world;
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

    std::vector<Eigen::Vector3f> landmarks;

    landmarks.push_back({-73919.3f, 30876.3f, 19849.9f});
    landmarks.push_back({-70737.7f, 5242.63f, 23751.0f});
    landmarks.push_back({-67043.4f, -16524.0f, 30461.0f});
    landmarks.push_back({-63378.7f, -32740.5f, 38981.6f});
    landmarks.push_back({-55343.4f, -50362.6f, 48677.7f});
    landmarks.push_back({-42926.8f, -63089.2f, 63733.5f});
    landmarks.push_back({-27970.4f, -71516.9f, 81654.7f});
    landmarks.push_back({-16291.4f, -74698.6f, 96297.7f});
    landmarks.push_back({-48.2388f, -76505.2f, 99442.1f});
    landmarks.push_back({16255.0f, -74471.6f, 96308.6f});
    landmarks.push_back({27908.9f, -71417.7f, 81694.5f});
    landmarks.push_back({42855.4f, -63229.9f, 63746.2f});
    landmarks.push_back({56170.0f, -49032.8f, 48846.8f});
    landmarks.push_back({63225.8f, -33092.0f, 38968.2f});
    landmarks.push_back({66704.6f, -16882.1f, 30449.8f});
    landmarks.push_back({70054.7f, 4963.69f, 23643.8f});
    landmarks.push_back({73154.5f, 30854.0f, 20055.8f});
    landmarks.push_back({-57346.8f, 45510.3f, 80483.7f});
    landmarks.push_back({-49635.7f, 51531.3f, 92479.5f});
    landmarks.push_back({-38958.6f, 53447.6f, 100454.0f});
    landmarks.push_back({-29608.2f, 52236.6f, 104277.0f});
    landmarks.push_back({-18702.9f, 50264.2f, 106598.0f});
    landmarks.push_back({17860.7f, 50466.9f, 106508.0f});
    landmarks.push_back({28544.3f, 51538.8f, 103733.0f});
    landmarks.push_back({38062.0f, 52791.3f, 100413.0f});
    landmarks.push_back({49630.6f, 50820.8f, 91989.9f});
    landmarks.push_back({56468.8f, 46294.8f, 81118.2f});
    landmarks.push_back({-146.369f, 34128.1f, 110560.0f});
    landmarks.push_back({-116.763f, 24000.5f, 118205.0f});
    landmarks.push_back({-20.8564f, 14347.0f, 125517.0f});
    landmarks.push_back({13.2323f, 4430.29f, 131560.0f});
    landmarks.push_back({-11574.5f, -7575.11f, 110567.0f});
    landmarks.push_back({-5116.18f, -7991.94f, 114465.0f});
    landmarks.push_back({-164.637f, -9165.1f, 117358.0f});
    landmarks.push_back({4602.21f, -8039.56f, 114443.0f});
    landmarks.push_back({11130.6f, -7555.56f, 110556.0f});
    landmarks.push_back({-44627.6f, 33016.0f, 85710.8f});
    landmarks.push_back({-37033.8f, 36953.1f, 92643.3f});
    landmarks.push_back({-27261.5f, 37833.3f, 94881.4f});
    landmarks.push_back({-18330.4f, 33041.4f, 91429.2f});
    landmarks.push_back({-29058.9f, 29625.8f, 94046.1f});
    landmarks.push_back({-37779.7f, 29124.5f, 91864.5f});
    landmarks.push_back({18462.4f, 33379.7f, 92001.2f});
    landmarks.push_back({27972.3f, 38007.0f, 94475.5f});
    landmarks.push_back({37036.2f, 36535.4f, 91911.6f});
    landmarks.push_back({45188.7f, 33094.9f, 85347.9f});
    landmarks.push_back({38052.3f, 29349.8f, 91259.8f});
    landmarks.push_back({29458.6f, 29444.7f, 93770.8f});
    landmarks.push_back({-25539.6f, -33627.8f, 98388.9f});
    landmarks.push_back({-15805.3f, -27611.8f, 110100.0f});
    landmarks.push_back({-6628.37f, -23768.2f, 115463.0f});
    landmarks.push_back({-192.025f, -24960.5f, 116208.0f});
    landmarks.push_back({7616.04f, -23944.8f, 115018.0f});
    landmarks.push_back({17230.8f, -27470.0f, 108906.0f});
    landmarks.push_back({25143.8f, -33333.3f, 98094.7f});
    landmarks.push_back({18575.5f, -36133.4f, 104735.0f});
    landmarks.push_back({10856.5f, -38386.0f, 111107.0f});
    landmarks.push_back({-175.34f, -39321.0f, 113471.0f});
    landmarks.push_back({-11149.0f, -38397.3f, 111204.0f});
    landmarks.push_back({-18695.9f, -35550.9f, 104488.0f});
    landmarks.push_back({-22495.2f, -32862.2f, 99654.2f});
    landmarks.push_back({-8644.35f, -30469.4f, 109984.0f});
    landmarks.push_back({-279.868f, -30878.0f, 111425.0f});
    landmarks.push_back({8088.77f, -30539.5f, 109996.0f});
    landmarks.push_back({23331.7f, -32803.8f, 98654.9f});
    landmarks.push_back({8140.39f, -31052.9f, 109408.0f});
    landmarks.push_back({-257.674f, -31426.9f, 110736.0f});
    landmarks.push_back({-9651.11f, -30987.6f, 108781.0f});
    for (int i = 0; i < landmarks.size(); ++i) {
        landmarks[i] /= 1000;
    }
    properties.landmarks = landmarks;


    properties.initialOffset.x() = 0;
    properties.initialOffset.y() = 0;
    properties.initialOffset.z() = 0;
    ProcrustesAligner aligner;
    std::vector<Vector3f> sourcePoints;
    std::vector<Vector3f> targetPoints;
    std::vector<Vector2f> landmarksImage;

    landmarksImage.emplace_back(118.63, 207.90);
    landmarksImage.emplace_back(121.40, 238.45);
    landmarksImage.emplace_back(129.03, 266.74);
    landmarksImage.emplace_back(135.65, 290.42);
    landmarksImage.emplace_back(144.91, 315.50);
    landmarksImage.emplace_back(159.29, 338.10);
    landmarksImage.emplace_back(176.18, 354.60);
    landmarksImage.emplace_back(197.80, 368.85);
    landmarksImage.emplace_back(231.05, 376.89);
    landmarksImage.emplace_back(263.81, 368.39);
    landmarksImage.emplace_back(284.42, 354.98);
    landmarksImage.emplace_back(299.53, 338.77);
    landmarksImage.emplace_back(312.64, 316.09);
    landmarksImage.emplace_back(321.79, 291.46);
    landmarksImage.emplace_back(327.73, 267.79);
    landmarksImage.emplace_back(334.75, 239.87);
    landmarksImage.emplace_back(338.17, 209.36);
    landmarksImage.emplace_back(140.88, 189.96);
    landmarksImage.emplace_back(152.32, 183.01);
    landmarksImage.emplace_back(167.81, 181.91);
    landmarksImage.emplace_back(183.30, 185.09);
    landmarksImage.emplace_back(197.17, 189.60);
    landmarksImage.emplace_back(258.65, 188.47);
    landmarksImage.emplace_back(273.00, 184.76);
    landmarksImage.emplace_back(288.51, 181.96);
    landmarksImage.emplace_back(304.41, 183.74);
    landmarksImage.emplace_back(315.77, 190.90);
    landmarksImage.emplace_back(227.64, 218.48);
    landmarksImage.emplace_back(227.83, 240.47);
    landmarksImage.emplace_back(228.08, 262.71);
    landmarksImage.emplace_back(228.03, 279.00);
    landmarksImage.emplace_back(209.92, 283.81);
    landmarksImage.emplace_back(217.74, 286.87);
    landmarksImage.emplace_back(228.39, 289.29);
    landmarksImage.emplace_back(238.74, 286.91);
    landmarksImage.emplace_back(246.27, 283.88);
    landmarksImage.emplace_back(160.67, 212.50);
    landmarksImage.emplace_back(169.76, 206.67);
    landmarksImage.emplace_back(183.94, 206.30);
    landmarksImage.emplace_back(197.27, 214.27);
    landmarksImage.emplace_back(185.12, 218.79);
    landmarksImage.emplace_back(170.66, 218.54);
    landmarksImage.emplace_back(257.29, 215.81);
    landmarksImage.emplace_back(271.76, 208.94);
    landmarksImage.emplace_back(286.64, 209.49);
    landmarksImage.emplace_back(295.74, 214.61);
    landmarksImage.emplace_back(285.68, 220.59);
    landmarksImage.emplace_back(270.20, 219.98);
    landmarksImage.emplace_back(189.80, 311.63);
    landmarksImage.emplace_back(202.43, 308.06);
    landmarksImage.emplace_back(219.75, 304.47);
    landmarksImage.emplace_back(228.73, 305.60);
    landmarksImage.emplace_back(237.76, 304.15);
    landmarksImage.emplace_back(254.45, 306.87);
    landmarksImage.emplace_back(267.31, 310.00);
    landmarksImage.emplace_back(254.50, 320.58);
    landmarksImage.emplace_back(242.90, 327.55);
    landmarksImage.emplace_back(230.52, 329.15);
    landmarksImage.emplace_back(218.24, 328.10);
    landmarksImage.emplace_back(205.96, 322.16);
    landmarksImage.emplace_back(194.03, 311.23);
    landmarksImage.emplace_back(217.34, 312.62);
    landmarksImage.emplace_back(228.73, 312.21);
    landmarksImage.emplace_back(240.18, 312.04);
    landmarksImage.emplace_back(264.51, 309.28);
    landmarksImage.emplace_back(241.31, 315.35);
    landmarksImage.emplace_back(229.96, 316.80);
    landmarksImage.emplace_back(218.95, 315.88);

    std::vector<float> depthValues;

    depthValues.emplace_back(-81.9494f);
    depthValues.emplace_back(-84.0708f);
    depthValues.emplace_back(-86.1261f);
    depthValues.emplace_back(-84.3486f);
    depthValues.emplace_back(-75.0432f);
    depthValues.emplace_back(-54.3234f);
    depthValues.emplace_back(-26.7269f);
    depthValues.emplace_back(-3.1504f);
    depthValues.emplace_back(5.1858f);
    depthValues.emplace_back(-3.5604f);
    depthValues.emplace_back(-27.7868f);
    depthValues.emplace_back(-55.2481f);
    depthValues.emplace_back(-76.0480f);
    depthValues.emplace_back(-85.3210f);
    depthValues.emplace_back(-86.0624f);
    depthValues.emplace_back(-83.5348f);
    depthValues.emplace_back(-80.9916f);
    depthValues.emplace_back(17.3624f);
    depthValues.emplace_back(37.2231f);
    depthValues.emplace_back(50.5675f);
    depthValues.emplace_back(58.4185f);
    depthValues.emplace_back(61.5788f);
    depthValues.emplace_back(61.7214f);
    depthValues.emplace_back(58.3998f);
    depthValues.emplace_back(50.4918f);
    depthValues.emplace_back(37.2096f);
    depthValues.emplace_back(16.9451f);
    depthValues.emplace_back(61.6455f);
    depthValues.emplace_back(72.1690f);
    depthValues.emplace_back(81.8564f);
    depthValues.emplace_back(81.3163f);
    depthValues.emplace_back(47.0413f);
    depthValues.emplace_back(52.8052f);
    depthValues.emplace_back(55.7183f);
    depthValues.emplace_back(52.7529f);
    depthValues.emplace_back(47.1556f);
    depthValues.emplace_back(23.9380f);
    depthValues.emplace_back(38.4398f);
    depthValues.emplace_back(39.0451f);
    depthValues.emplace_back(33.2425f);
    depthValues.emplace_back(36.0291f);
    depthValues.emplace_back(32.2369f);
    depthValues.emplace_back(33.3726f);
    depthValues.emplace_back(39.1702f);
    depthValues.emplace_back(37.9189f);
    depthValues.emplace_back(23.0803f);
    depthValues.emplace_back(31.8327f);
    depthValues.emplace_back(36.2982f);
    depthValues.emplace_back(24.8443f);
    depthValues.emplace_back(43.2507f);
    depthValues.emplace_back(53.4487f);
    depthValues.emplace_back(54.1908f);
    depthValues.emplace_back(53.1766f);
    depthValues.emplace_back(43.3501f);
    depthValues.emplace_back(26.1465f);
    depthValues.emplace_back(40.6935f);
    depthValues.emplace_back(45.9826f);
    depthValues.emplace_back(47.4366f);
    depthValues.emplace_back(46.3287f);
    depthValues.emplace_back(40.4743f);
    depthValues.emplace_back(27.2977f);
    depthValues.emplace_back(45.8984f);
    depthValues.emplace_back(48.3717f);
    depthValues.emplace_back(46.1922f);
    depthValues.emplace_back(28.1654f);
    depthValues.emplace_back(49.3157f);
    depthValues.emplace_back(50.5824f);
    depthValues.emplace_back(48.8912f);

    Matrix3f K;
    K << 0.005971, -0.001186, -0.002454,
            0.006100, -0.001137, -0.002745,
            0.000035, -0.000007, -0.000016;

    Matrix3f R;
    R << 0.403446, 0.169169, -0.899229,
    0.573882, -0.812221, 0.104676,
    0.712665, 0.558283, 0.424770;

    Matrix4f extrinsics;
    extrinsics << 0.403446, 0.169169, -0.899229, -212.836227,
    0.573882, -0.812221, 0.104676, 294.292480,
    0.712665, 0.558283, 0.424770, 107.552269,
    0.000000, 0.000000, 0.000000, 1.000000;

    Vector3f T(-212.836227, 294.292480, 107.552269);

    //GetTargetLandmarks
    for (int i = 0; i < landmarksImage.size(); ++i) {
        targetPoints.emplace_back(convert2Dto3D(landmarksImage[i], depthValues[i], K, R, T, extrinsics));
    }
    //End GetTargetLandmarks
    std::cout << targetPoints.size() << std::endl;

    Matrix4f estimatedPose = aligner.estimatePose(landmarks, targetPoints);
    properties.transformation = estimatedPose;
}

#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H
