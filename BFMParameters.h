#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include "fstream"
#include <H5Cpp.h>
#include "Eigen.h"
#include "ProcrustesAligner.h"
#include "ImageExtraction.h"


const std::string dataFolderPath = DATA_FOLDER_PATH;
const std::string resultFolderPath = RESULT_FOLDER_PATH;

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
    std::vector<int> landmark_indices;
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

static std::vector<Eigen::Vector3f> getVerticesWithoutProcrustes(BfmProperties properties){
    std::vector<Eigen::Vector3f> vertices;

    Eigen::VectorXf shapeVar = Eigen::Map<Eigen::VectorXf>(properties.shapePcaVariance.data(), properties.shapePcaVariance.size());
    Eigen::VectorXf modifiedShape = properties.shapePcaBasis * (shapeVar.cwiseSqrt().cwiseProduct(properties.shapeParams));
    Eigen::VectorXf expressionVar = Eigen::Map<Eigen::VectorXf>(properties.expressionPcaVariance.data(), properties.expressionPcaVariance.size());
    Eigen::VectorXf modifiedExpression = properties.expressionPcaBasis * (expressionVar.cwiseSqrt().cwiseProduct(properties.expressionParams));

    //std::cout << "Modified Geometry: " << std::endl;
    //std::cout << modifiedExpression + modifiedShape << std::endl;

    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) { //Why is number of vertices * 3 not the same as properties.shapeMean.size()
        Eigen::Vector3f newVertex;

        //newVertex.x() = properties.shapeMean[i] + properties.expressionMean[i];
        //newVertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1];
        //newVertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2];

        newVertex.x() = properties.shapeMean[i] + properties.expressionMean[i] + modifiedShape[i] + modifiedExpression[i];
        newVertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1] + modifiedShape[i + 1] + modifiedExpression[i + 1];
        newVertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2] + modifiedShape[i + 2] + modifiedExpression[i + 2];

        vertices.emplace_back(newVertex);
    }
    return vertices;
}

static std::vector<Eigen::Vector3f> getVertices(BfmProperties properties) {
    std::vector<Eigen::Vector3f> vertices;

    Eigen::VectorXf shapeVar = Eigen::Map<Eigen::VectorXf>(properties.shapePcaVariance.data(), properties.shapePcaVariance.size());
    Eigen::VectorXf modifiedShape = properties.shapePcaBasis * (shapeVar.cwiseSqrt().cwiseProduct(properties.shapeParams));
    Eigen::VectorXf expressionVar = Eigen::Map<Eigen::VectorXf>(properties.expressionPcaVariance.data(), properties.expressionPcaVariance.size());
    Eigen::VectorXf modifiedExpression = properties.expressionPcaBasis * (expressionVar.cwiseSqrt().cwiseProduct(properties.expressionParams));

    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector4f vertex;
        vertex.x() = properties.shapeMean[i] + properties.expressionMean[i];
        vertex.y() = properties.shapeMean[i + 1] + properties.expressionMean[i + 1];
        vertex.z() = properties.shapeMean[i + 2] + properties.expressionMean[i + 2];
        vertex.w() = 1.0f;

        vertex.x() += modifiedShape[i] + modifiedExpression[i];
        vertex.y() += modifiedShape[i + 1] + modifiedExpression[i + 1];
        vertex.z() += modifiedShape[i + 2] + modifiedExpression[i + 2];

        vertex = properties.transformation * vertex;

        vertices.emplace_back(Eigen::Vector3f(vertex.x(), vertex.y(), vertex.z()));
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

static std::vector<Eigen::Vector3f> getColorValuesF(BfmProperties properties){
    Eigen::VectorXf colorVar = Eigen::Map<Eigen::VectorXf>(properties.colorPcaVariance.data(), properties.colorPcaVariance.size());
    Eigen::VectorXf modifiedColor = properties.colorPcaBasis * (colorVar.cwiseSqrt().cwiseProduct(properties.colorParams));

    std::vector<Eigen::Vector3f> colorValues;
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        Eigen::Vector3f newColorValue;
        newColorValue.x() = ((properties.colorMean[i] + modifiedColor[i]));
        newColorValue.y() = ((properties.colorMean[i + 1] + modifiedColor[i + 1]));
        newColorValue.z() = ((properties.colorMean[i + 2] + modifiedColor[i + 1]));
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

template <typename T>
static Eigen::Matrix<T, 2, 1> convert3Dto2DTemplate(
        const Eigen::Matrix<T, 3, 1>& point,
        const Eigen::Matrix<T, 3, 3>& depthIntrinsics,
        const Eigen::Matrix<T, 4, 4>& extrinsics) {

    Eigen::Matrix<T, 4, 4> depthExtrinsicsInv = extrinsics.inverse();
    Eigen::Matrix<T, 4, 1> worldCoord(point.x(), point.y(), point.z(), T(1));
    Eigen::Matrix<T, 4, 1> cameraCoord = depthExtrinsicsInv * worldCoord;

    T fX = depthIntrinsics(0, 0); // focal length in x direction
    T fY = depthIntrinsics(1, 1); // focal length in y direction
    T cX = depthIntrinsics(0, 2); // optical center in x direction
    T cY = depthIntrinsics(1, 2); // optical center in y direction

    T x = cameraCoord.x() / cameraCoord.z();
    T y = cameraCoord.y() / cameraCoord.z();

    T u = fX * x + cX;
    T v = fY * y + cY;

    return Eigen::Matrix<T, 2, 1>(u, v);
}

static float getDepthValueFromInputImage(const Eigen::Vector3f& point, std::vector<float> depthValues, int width, int height, const Eigen::Matrix3f& depthIntrinsics, const Eigen::Matrix4f& extrinsics){
    auto pixelCoordinates = convert3Dto2D(point, depthIntrinsics, extrinsics);
    return depthValues[(int) pixelCoordinates.x() + (int) pixelCoordinates.y() * width];
}

static Eigen::Vector3f getColorValueFromInputImage(const Eigen::Vector3f& point, std::vector<Eigen::Vector3f> colorValues, int width, int height, const Eigen::Matrix3f& depthIntrinsics, const Eigen::Matrix4f& extrinsics){
    auto pixelCoordinates = convert3Dto2D(point, depthIntrinsics, extrinsics);
    return colorValues[(int) pixelCoordinates.x() + (int) pixelCoordinates.y() * width];
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
        properties.shapeParams[i] = 0.0f;//0.02f;
        properties.colorParams[i] = 0.0f;//0.02f;
    }

    for (int i = 0; i < 100; ++i) {
        properties.expressionParams[i] = 0.0f;//0.02f;
    }

    properties.numberOfVertices = properties.shapeMean.size() / 3;
    //Faces
    const std::string inputFile = std::string(dataFolderPath + "faces.txt");
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

    ProcrustesAligner aligner;

    std::vector<Eigen::Vector3f> targetPoints;
    //GetTargetLandmarks
    for (int i = 0; i < inputImage.depthValuesLandmarks.size(); ++i) {
        targetPoints.emplace_back(convert2Dto3D(inputImage.landmarks[i], inputImage.depthValuesLandmarks[i], inputImage.intrinsics, inputImage.extrinsics));
    }
    std::vector<int> indices;
    //TODO: If you want to use jaw, comment code in
    indices.emplace_back(18463);
    indices.emplace_back(21941);
    indices.emplace_back(21354);
    indices.emplace_back(13904);
    indices.emplace_back(16415);
    indices.emplace_back(14167);
    indices.emplace_back(25399);
    indices.emplace_back(18193);
    indices.emplace_back(15867);
    indices.emplace_back(12269);
    indices.emplace_back(11447);
    indices.emplace_back(10004);
    indices.emplace_back(7724);
    indices.emplace_back(200);
    indices.emplace_back(7402);
    indices.emplace_back(1888);
    indices.emplace_back(4545);
    indices.emplace_back(22050);
    indices.emplace_back(20167);
    indices.emplace_back(25269);
    indices.emplace_back(24554);
    indices.emplace_back(20687);
    indices.emplace_back(6735);
    indices.emplace_back(12889);
    indices.emplace_back(6242);
    indices.emplace_back(7173);
    indices.emplace_back(72);
    indices.emplace_back(15934);
    indices.emplace_back(18713);
    indices.emplace_back(15845);
    indices.emplace_back(18086);
    indices.emplace_back(25821);
    indices.emplace_back(18740);
    indices.emplace_back(19201);
    indices.emplace_back(4806);
    indices.emplace_back(6746);
    indices.emplace_back(20818);
    indices.emplace_back(17528);
    indices.emplace_back(16910);
    indices.emplace_back(14469);
    indices.emplace_back(14337);
    indices.emplace_back(18680);
    indices.emplace_back(3321);
    indices.emplace_back(10654);
    indices.emplace_back(3681);
    indices.emplace_back(3126);
    indices.emplace_back(738);
    indices.emplace_back(12881);
    indices.emplace_back(16152);
    indices.emplace_back(27501);
    indices.emplace_back(18363);
    indices.emplace_back(18353);
    indices.emplace_back(4453);
    indices.emplace_back(1663);
    indices.emplace_back(1353);
    indices.emplace_back(2251);
    indices.emplace_back(4158);
    indices.emplace_back(15821);
    indices.emplace_back(15440);
    indices.emplace_back(21063);
    indices.emplace_back(15443);
    indices.emplace_back(17507);
    indices.emplace_back(15825);
    indices.emplace_back(3664);
    indices.emplace_back(1739);
    indices.emplace_back(7753);
    indices.emplace_back(15819);
    indices.emplace_back(21705);
    properties.landmark_indices = indices;//getLandmarkIndices(properties);
    std::vector<Eigen::Vector3f> landmarks;
    auto vertices = getVerticesWithoutProcrustes(properties);
    for (int i = 0; i < indices.size(); ++i) {
        landmarks.emplace_back(vertices[indices[i]]);
    }

    Matrix4f estimatedPose = aligner.estimatePose(landmarks, targetPoints);
    //+ translation: Halbe width und halbe height abziehen:

    properties.transformation = estimatedPose;// * rotationMatrix;
}

#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H
