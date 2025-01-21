#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include "fstream"
#include <H5Cpp.h>
#include "Eigen.h"
#include "ProcrustesAligner.h"
#include "ImageExtraction.h"

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
    Eigen::MatrixXf colorPcaBasis;
    std::vector<float> colorPcaVariance;

    std::vector<float> shapeMean;
    Eigen::MatrixXf shapePcaBasis;
    std::vector<float> shapePcaVariance;

    std::vector<float> expressionMean;
    Eigen::MatrixXf expressionPcaBasis;
    std::vector<float> expressionPcaVariance;

    Eigen::VectorXf colorParams;
    Eigen::VectorXf shapeParams;
    Eigen::VectorXf expressionParams;
};

static void setInitialOffset(Eigen::Vector3f initialOffset, BfmProperties& properties) {
    properties.initialOffset.x() = initialOffset.x();
    std::cout << "InitialOffset:" << properties.initialOffset.x() << std::endl;
    properties.initialOffset.y() = initialOffset.y();
    properties.initialOffset.z() = initialOffset.z();
}

static std::vector<Eigen::Vector3f> getVertices(BfmProperties properties){
    std::vector<Eigen::Vector3f> vertices;

    std::cout << "Start" << std::endl;
    Eigen::VectorXf shapeVar = Eigen::Map<Eigen::VectorXf>(properties.shapePcaVariance.data(), properties.shapePcaVariance.size());
    Eigen::VectorXf modifiedShape = properties.shapePcaBasis * (shapeVar.cwiseSqrt().cwiseProduct(properties.shapeParams));
    Eigen::VectorXf expressionVar = Eigen::Map<Eigen::VectorXf>(properties.expressionPcaVariance.data(), properties.expressionPcaVariance.size());
    Eigen::VectorXf modifiedExpression = properties.expressionPcaBasis * (expressionVar.cwiseSqrt().cwiseProduct(properties.expressionParams));
    std::cout << "End" << std::endl;

    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3f newVertex;

        newVertex.x() = properties.shapeMean[i] + properties.expressionMean[i] + modifiedShape[i] + modifiedExpression[i];
        newVertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1] + modifiedShape[i + 1] + modifiedExpression[i + 1];
        newVertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2] + modifiedShape[i + 2] + modifiedExpression[i + 2];
        Eigen::Vector4f transformationVector;

        transformationVector.x() = newVertex.x();
        transformationVector.y() = newVertex.y();
        transformationVector.z() = newVertex.z();
        transformationVector.w() = 1.0f;
        transformationVector = properties.transformation * transformationVector;
        newVertex.x() = transformationVector.x();
        newVertex.y() = transformationVector.y();
        newVertex.z() = transformationVector.z();
        if(i == 0){
            std::cout << "New Vertex: " << newVertex.x() << ", " << newVertex.y() << ", " << newVertex.z() << ";" << std::endl;
        }
        vertices.emplace_back(newVertex);
    }
    return vertices;
}

static std::vector<Eigen::Vector3f> getNormals(BfmProperties properties){
    auto bfmVertices = getVertices(properties);
    std::vector<Vector3f> normals = std::vector<Vector3f>(bfmVertices.size(), Vector3f::Zero());

    for (size_t i = 0; i < properties.triangles.size(); i+=3) {
        auto triangle0 = properties.triangles[i];
        auto triangle1 = properties.triangles[i+1];
        auto triangle2 = properties.triangles[i+2];
        Vector3f faceNormal = (bfmVertices[triangle1] - bfmVertices[triangle0]).cross(bfmVertices[triangle2] - bfmVertices[triangle0]);
        normals[triangle0] += faceNormal;
        normals[triangle1] += faceNormal;
        normals[triangle2] += faceNormal;
    }
    // Normalize normals
    int zeroNormals = 0;
    for (size_t i = 0; i < bfmVertices.size(); i++) {
        if (normals[i].norm() < 1e-10) {
            zeroNormals++;
        }
        normals[i].normalize();
    }
    return normals;
}

static std::vector<Eigen::Vector3i> getColorValues(BfmProperties properties){
    Eigen::VectorXf colorVar = Eigen::Map<Eigen::VectorXf>(properties.colorPcaVariance.data(), properties.colorPcaVariance.size());
    Eigen::VectorXf modifiedColor = properties.colorPcaBasis * (colorVar.cwiseSqrt().cwiseProduct(properties.colorParams));

    std::vector<Eigen::Vector3i> colorValues;
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3i newColorValue;
        newColorValue.x() = (int) ((properties.colorMean[i] + modifiedColor[i]) * 255);
        newColorValue.y() = (int) ((properties.colorMean[i + 1] + modifiedColor[i + 1]) * 255);
        newColorValue.z() = (int) ((properties.colorMean[i + 2] + modifiedColor[i + 1]) * 255);
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

static void readHDF5DataMatrix(const H5::H5File& file, const std::string& groupPath, const std::string& datasetPath, Eigen::MatrixXf& target) {
    try {
        // Open the specified group and dataset
        H5::Group group = file.openGroup(groupPath);
        H5::DataSet dataset = group.openDataSet(datasetPath);

        // Get the dataspace of the dataset and determine its dimensions
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        if (rank != 2) {
            throw std::runtime_error("Dataset is not 2-dimensional.");
        }

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, nullptr);

        // Resize the target matrix based on the dataset dimensions
        target.resize(static_cast<int>(dims[0]), static_cast<int>(dims[1]));

        // Read the data into a temporary vector and map it to the matrix
        std::vector<float> buffer(static_cast<size_t>(dims[0] * dims[1]));
        dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

        // Map the vector into the Eigen matrix
        target = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                buffer.data(), static_cast<int>(dims[0]), static_cast<int>(dims[1]));
    } catch (H5::Exception& e) {
        std::cerr << "Error reading HDF5 dataset: " << e.getDetailMsg() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

static void readFaces(){
//TODO: Not done
    /*const std::string inputFile = std::string("../../../Data/faces.txt");
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
    std::cout << "Faces: " << properties.numberOfTriangles << std::endl;*/
}

static std::vector<Eigen::Vector3f> readLandmarksBFM(const std::string& path){
    const std::string inputFile = std::string(path);
    std::ifstream inFile(inputFile);
    std::string line;
    std::vector<Eigen::Vector3f> landmarks;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        float first, second, third;
        if (iss >> first >> second >> third) {
            Eigen::Vector3f newLandmark;
            newLandmark.x() = first;
            newLandmark.y() = second;
            newLandmark.z() = third;
            landmarks.emplace_back(newLandmark);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return landmarks;
}

static std::vector<Vector2f> readLandmarksInputImage(const std::string& path){
    const std::string inputFile = std::string(path);
    std::ifstream inFile(inputFile);
    std::string line;
    std::vector<Vector2f> landmarksImage;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        float first, second;
        if (iss >> first >> second) {
            Eigen::Vector2f newLandmark;
            newLandmark.x() = first;
            newLandmark.y() = second;
            landmarksImage.emplace_back(newLandmark);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return landmarksImage;
}

static std::vector<float> readLandmarksDepthInputImage(const std::string& path){
    const std::string inputFile = std::string(path);
    std::ifstream inFile(inputFile);
    std::string line;
    std::vector<float> depthValues;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        float first;
        if (iss >> first) {
            depthValues.emplace_back(first);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return depthValues;
}

static Eigen::Vector3f convert2Dto3D(const Eigen::Vector2f& point, float depth, const Eigen::Matrix3f& depthIntrinsics, const Eigen::Matrix4f& extrinsics) {
    float fX = depthIntrinsics(0, 0);
    float fY = depthIntrinsics(1, 1);
    float cX = depthIntrinsics(0, 2);
    float cY = depthIntrinsics(1, 2);

    float x = (point.x() - cX) * depth / fX;
    float y = (point.y() - cY) * depth / fY;
    float z = depth;

    Matrix4f depthExtrinsicsInv = extrinsics.inverse();

    Vector4f cameraCoord = Vector4f(x, y, z, 1.0f);
    Vector4f worldCoords = depthExtrinsicsInv * cameraCoord;

    return Eigen::Vector3f(worldCoords.x(), worldCoords.y(), worldCoords.z());
}

static Eigen::Vector2f convert3Dto2D(const Eigen::Vector3f& point, const Eigen::Matrix3f& depthIntrinsics, const Eigen::Matrix4f& extrinsics) {
    Eigen::Matrix4f depthExtrinsicsInv = extrinsics.inverse();
    Eigen::Vector4f worldCoord(point.x(), point.y(), point.z(), 1.0f);
    Eigen::Vector4f cameraCoord = depthExtrinsicsInv * worldCoord;

    float fX = depthIntrinsics(0, 0); // focal length in x direction
    float fY = depthIntrinsics(1, 1); // focal length in y direction
    float cX = depthIntrinsics(0, 2); // optical center in x direction
    float cY = depthIntrinsics(1, 2); // optical center in y direction

    float x = cameraCoord.x() / cameraCoord.z();
    float y = cameraCoord.y() / cameraCoord.z();

    float u = fX * x + cX;
    float v = fY * y + cY;

    return Eigen::Vector2f(u, v);
}

static float getDepthValueFromInputImage(const Eigen::Vector3f& point, std::vector<float> depthValues, int width, int height, const Eigen::Matrix3f& depthIntrinsics, const Eigen::Matrix4f& extrinsics){
    auto pixelCoordinates = convert3Dto2D(point, depthIntrinsics, extrinsics);
    return depthValues[(int) pixelCoordinates.x() + (int) pixelCoordinates.y() * width];
}

//@param path -> path to .h5 file
//initializeMethod durch constructor ersetzen?!
static void initializeBFM(const std::string& path, BfmProperties& properties, const InputImage& inputImage){

    //BfmProperties properties;
    H5::H5File file(path, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
    }

    //Read Shape
    readHDF5Data(file, "/shape/model", "mean", properties.shapeMean);
    readHDF5DataMatrix(file, "/shape/model", "pcaBasis", properties.shapePcaBasis); //TODO: PCABasis for Modifications?!
    readHDF5Data(file, "/shape/model", "pcaVariance", properties.shapePcaVariance);
    //Read Expression
    readHDF5Data(file, "/color/model", "mean", properties.colorMean);
    readHDF5DataMatrix(file, "/color/model", "pcaBasis", properties.colorPcaBasis);
    readHDF5Data(file, "/color/model", "pcaVariance", properties.colorPcaVariance);
    //Read Color
    readHDF5Data(file, "/expression/model", "mean", properties.expressionMean);
    readHDF5DataMatrix(file, "/expression/model", "pcaBasis", properties.expressionPcaBasis);
    readHDF5Data(file, "/expression/model", "pcaVariance", properties.expressionPcaVariance);

    properties.shapeParams = Eigen::VectorXf(199);
    properties.expressionParams = Eigen::VectorXf(100);
    properties.colorParams = Eigen::VectorXf(199);

    for (int i = 0; i < 199; ++i) {
        properties.shapeParams[i] = 0.02f;
        properties.colorParams[i] = 0.02f;
    }

    for (int i = 0; i < 100; ++i) {
        properties.expressionParams[i] = 0.02f;
    }

    properties.numberOfVertices = properties.shapeMean.size() / 3;
    std::cout << "Vertices: " << properties.numberOfVertices << std::endl;
    std::cout << "Color Mean: " << properties.colorMean.size() << " values" << std::endl;
    std::cout << "Shape Variance: " << properties.shapePcaVariance.size() << " values" << std::endl;
    std::cout << "PCA Basis " << properties.shapePcaBasis(0, 0) << std::endl;
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
    ProcrustesAligner aligner;

    std::vector<Eigen::Vector3f> targetPoints;
    //GetTargetLandmarks
    for (int i = 0; i < inputImage.depthValuesLandmarks.size(); ++i) {
        targetPoints.emplace_back(convert2Dto3D(inputImage.landmarks[i], inputImage.depthValuesLandmarks[i], inputImage.intrinsics, inputImage.extrinsics));
    }
    //End GetTargetLandmarks
    //std::cout << targetPoints.size() << std::endl;
    //Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    //rotationMatrix(0, 0) = -1; // cos(180°) = -1
    //rotationMatrix(1, 1) = -1; // cos(180°) = -1

    Matrix4f estimatedPose = aligner.estimatePose(landmarks, targetPoints);
    //+ translation: Halbe width und halbe height abziehen:

    properties.transformation = estimatedPose;// * rotationMatrix;
}

#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H
