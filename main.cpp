#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "PointCloud.h"

#include "FacialLandmarks.h"
#include "Optimization.h"
#include "Rendering.h"
#include "BFMParameters.h"
#include "ProcrustesAligner.h"
#include "ImageExtraction.h"

using namespace Eigen;
using namespace std;

int main() {
    InputImage inputImage = readVideoData(dataFolderPath + "20250127_200932.bag");
    const std::string imagePath = std::string(resultFolderPath + "color_frame_corrected.png");
    //const std::string imagePath = std::string(dataFolderPath + "testmyface.png");
    const std::string shapePredictorPath = std::string(dataFolderPath + "shape_predictor_68_face_landmarks.dat");
    const std::string outputPath = std::string(resultFolderPath + "output_corrected.png");
    //const char* imagePath, const char* shapePredictorPath, bool saveResult=false, const char* resultPath=""
    DrawLandmarksOnImage(imagePath, outputPath, shapePredictorPath);
    auto landmarks2D = GetLandmarkVector(imagePath, shapePredictorPath);
    inputImage.landmarks = landmarks2D;
    std::vector<Eigen::Vector2f> landmarksImage;
    for (int i = 18; i < 68; ++i) {
        landmarksImage.emplace_back(landmarks2D[i]);
    }
    inputImage.landmarks = landmarksImage;
    //printInputImage(inputImage);

    calculateDepthValuesLandmarks(inputImage);

    std::cout << "TEST" << std::endl;
    const std::string outputPlyPath = std::string(resultFolderPath + "outputModel.ply");
    const std::string outputLandmarkPlyPath = std::string(resultFolderPath + "landmarks.ply");
    const std::string h5TestFile = std::string(dataFolderPath + "model2019_face12.h5");
    std::cout << "0" << std::endl;
    BfmProperties properties;
    properties = getProperties(h5TestFile, inputImage);
    std::cout << "1" << std::endl;
    convertParametersToPlyWithoutProcrustes(properties, resultFolderPath + "InitialBfmModel.ply");
    std::cout << "2: " << inputImage.depthValuesLandmarks.size() << std::endl;
    std::vector<Eigen::Vector3f> targetPoints;
    //GetTargetLandmarks
    for (int i = 0; i < inputImage.depthValuesLandmarks.size(); ++i) {
        targetPoints.emplace_back(convert2Dto3D(inputImage.landmarks[i], inputImage.depthValuesLandmarks[i], inputImage.intrinsics, inputImage.extrinsics));
    }
    std::cout << "3" << std::endl;
    std::vector<Eigen::Vector2f> pointCloudVertices;
    for (int i = 0; i < inputImage.height; ++i) {
        for (int j = 0; j < inputImage.width; ++j) {
            pointCloudVertices.emplace_back(Eigen::Vector2f(j, i));
        }
    }

    std::vector<Eigen::Vector3i> color255;
    for (int i = 0; i < inputImage.color.size(); ++i) {
        color255.emplace_back(Eigen::Vector3i(inputImage.color[i].x() * 255, inputImage.color[i].y() * 255, inputImage.color[i].z() * 255));
    }

    std::vector<Eigen::Vector3f> landmarksFromIndices;
    for (int i = 0; i < properties.landmark_indices.size(); ++i) {
        landmarksFromIndices.emplace_back(getVertices(properties)[properties.landmark_indices[i]]);
    }
    convertVerticesTest(landmarksFromIndices, resultFolderPath + "landmarksFromIndices.ply");

    convertParametersToPly(properties, resultFolderPath + "BfmAfterProcrustes.ply");
    Optimization optimizer;
    optimizer.optimize(properties, inputImage);

    std::vector<float> parsedVertices;
    //auto originalVertices = getVertices(properties);
    auto originalVertices = getVerticesWithoutProcrustes(properties); //Sollte das nicht die normale get vertices methode sein?
    for (int i = 0; i < originalVertices.size(); ++i) {
        parsedVertices.push_back(originalVertices[i].x());
        parsedVertices.push_back(originalVertices[i].y());
        parsedVertices.push_back(originalVertices[i].z());
        if(i == 0){
            std::cout << "TestVertex: " << originalVertices[i].x() << ", " << originalVertices[i].y() << ", " << originalVertices[i].z() << ";" << std::endl;
        }
    }

    std::vector<int> parsedColor;
    auto originalColorValues = getColorValues(properties);
    for (int i = 0; i < originalColorValues.size(); ++i) {
        parsedColor.push_back(originalColorValues[i].x());
        parsedColor.push_back(originalColorValues[i].y());
        parsedColor.push_back(originalColorValues[i].z());
        if(i == 0){
            std::cout << originalColorValues[i].x() / 255.0f << ", " << originalColorValues[i].y() / 255.0f<< ", " << originalColorValues[i].z() / 255.0f << ";" << std::endl;
        }
    }

    getPointCloud(pointCloudVertices, inputImage.depthValues, color255, resultFolderPath +"pls.ply", inputImage.intrinsics, inputImage.extrinsics);
    convertVerticesTest(targetPoints, resultFolderPath + "warumklapptdasnicht.ply");

    std::vector<Eigen::Vector3f> landmarksFromIndicesAfterOpt;
    for (int i = 0; i < properties.landmark_indices.size(); ++i) {
        landmarksFromIndicesAfterOpt.emplace_back(getVertices(properties)[properties.landmark_indices[i]]);
    }
    convertVerticesTest(landmarksFromIndicesAfterOpt, resultFolderPath + "landmarksFromIndicesAfterOptimization.ply");

    std::vector<Eigen::Vector3f> landmarksss = readLandmarksBFM(dataFolderPath + "InitialLandmarkCoordinates.txt");
    convertVerticesTest(landmarksss, resultFolderPath + "InitialLandmarks.ply");

    convertParametersToPly(properties, resultFolderPath + "BfmModel.ply");
    renderFaceOnTopOfImage(1280, 720, parsedVertices, properties.triangles, parsedColor, (resultFolderPath + "color_frame_corrected.png").c_str(), inputImage, properties.transformation);
}
