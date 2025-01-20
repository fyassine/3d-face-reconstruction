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

// TODO define marker for data folder

int runPipeline() {
    //BfmProperties bfmProperties;
    //initializeBFM("model2019_face12.h5", bfmProperties);
    // TODO: Use our own data -> get landmarks with dlib -> get respective depth values -> convert to 3d
    // TODO decouple Procrutes from bfm initialization
    // getVertices() returns the transformed bfm vertices
    //Optimization::optimizeSparseTerms();
    // TODO implement sparse optimization (use formulas from the lecture)
    //Optimization::optimizeDenseTerms();
    // TODO include code from regularization inside optimizeDenseTerms
    // TODO save end result
    return -1;
}

/*void test1(){
    // NELI MARK: - Read input.txt file from Data folder
    const std::string inputFile = std::string("../../Data/input.txt");
    std::ifstream inFile(inputFile);

    // write output file to output.txt in Result folder
    std::ofstream outFile("../../Result/output.txt");
    if (outFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            outFile << line << std::endl;
        }
    }
    // Close output file.
    outFile.close();
    const std::string imagePath = std::string("../../../Data/testface.jpg");
    const std::string shapePredictorPath = std::string("../../../Data/shape_predictor_68_face_landmarks.dat");
    const std::string outputPath = std::string("../../../Result/output.png");
    //const char* imagePath, const char* shapePredictorPath, bool saveResult=false, const char* resultPath=""

    //DrawLandmarksOnImage(imagePath, outputPath, shapePredictorPath);

    const std::string outputPlyPath = std::string("../../../Result/outputModel.ply");
    const std::string outputLandmarkPlyPath = std::string("../../../Result/landmarks.ply");

    const std::string h5TestFile = std::string("../../../Data/model2019_face12.h5");
    convertParametersToPly(getProperties(h5TestFile), outputPlyPath);
    //convertLandmarksToPly(getProperties(h5TestFile), outputLandmarkPlyPath);
    convertVerticesTest(getProperties(h5TestFile).landmarks, outputLandmarkPlyPath);

    std::vector<Vector3f> landmarksImage3D;

    std::vector<float> x_values = {118.633575, 121.40035, 129.03134, 135.65099, 144.90646, 159.28775,
                                   176.17978, 197.80319, 231.05284, 263.81207, 284.41855, 299.53244,
                                   312.6431, 321.78937, 327.72693, 334.74503, 338.16705, 140.88348,
                                   152.31776, 167.80539, 183.29922, 197.17053, 258.6491, 273.00055,
                                   288.51187, 304.41492, 315.76587, 227.64235, 227.82626, 228.08022,
                                   228.03061, 209.92389, 217.74379, 228.39432, 238.74161, 246.2651,
                                   160.67302, 169.76, 183.93503, 197.26814, 185.1238, 170.65523,
                                   257.28702, 271.75507, 286.64493, 295.73615, 285.6814, 270.19943,
                                   189.80122, 202.42989, 219.74544, 228.72658, 237.7608, 254.45328,
                                   267.30786, 254.50342, 242.89995, 230.52003, 218.24008, 205.95857,
                                   194.02777, 217.33896, 228.73225, 240.18138, 264.51337, 241.31241,
                                   229.95511, 218.95059};

    std::vector<float> y_values = {207.89679, 238.44565, 266.74094, 290.42352, 315.50366, 338.1003,
                                   354.59723, 368.85245, 376.88562, 368.3873, 354.9839, 338.77045,
                                   316.0865, 291.46332, 267.78693, 239.8689, 209.36215, 189.96228,
                                   183.01056, 181.90622, 185.08984, 189.59637, 188.46643, 184.75644,
                                   181.96451, 183.7356, 190.89777, 218.48036, 240.47267, 262.70953,
                                   278.99994, 283.8117, 286.87433, 289.287, 286.91333, 283.88385,
                                   212.50165, 206.66745, 206.30171, 214.27065, 218.79277, 218.53519,
                                   215.80704, 208.94217, 209.48972, 214.60966, 220.58864, 219.9838,
                                   311.62946, 308.06317, 304.47278, 305.59863, 304.14722, 306.86954,
                                   309.99823, 320.577, 327.54572, 329.1473, 328.09967, 322.15704,
                                   311.22595, 312.6209, 312.2082, 312.0429, 309.28287, 315.35104,
                                   316.79926, 315.88007};

    std::vector<float> z_values = {-81.94941, -84.07083, -86.126144, -84.3486, -75.04321, -54.323433,
                                   -26.726929, -3.1503677, 5.185814, -3.5604172, -27.786797, -55.24808,
                                   -76.047966, -85.321014, -86.06242, -83.53477, -80.991585, 17.362434,
                                   37.223076, 50.567467, 58.418495, 61.578804, 61.721413, 58.39985,
                                   50.4918, 37.209602, 16.945091, 61.645515, 72.16903, 81.85641,
                                   81.31628, 47.041298, 52.8052, 55.71827, 52.75286, 47.155617,
                                   23.93798, 38.43978, 39.045067, 33.24253, 36.029137, 32.236855,
                                   33.372612, 39.170235, 37.91886, 23.080315, 31.83274, 36.29818,
                                   24.844307, 43.250725, 53.44866, 54.190773, 53.176598, 43.35009,
                                   26.146507, 40.69346, 45.982567, 47.436638, 46.328682, 40.47431,
                                   27.29766, 45.89837, 48.37171, 46.19216, 28.165382, 49.315742,
                                   50.58239, 48.891167};

    for (size_t i = 0; i < 68; ++i) {
        landmarksImage3D.emplace_back(x_values[i], y_values[i], z_values[i]);
    }
    convertVerticesTest(landmarksImage3D, "../../../Result/testLandmarks.ply");
    //initializeBFM(h5TestFile);
    //readH5File(h5TestFile);
    //readModelPath(h5TestFile);
    BfmProperties properties;
    Eigen::Vector3f initialTest;
    initialTest.x() = 0.1;
    initialTest.y() = 0.1;
    initialTest.z() = 0.1;
    properties.initialOffset = initialTest;
    properties = getProperties(h5TestFile);

    //setupGLFW(800, 800);
    std::vector<float> parsedVertices;
    auto originalVertices = getVertices(properties);
    for (int i = 0; i < originalVertices.size(); ++i) {
        parsedVertices.push_back(originalVertices[i].x() / 250); // TODO: We have to change the division. We might be able to do this by setting up a projection matrix and camera view matrix and enabling the depth test
        parsedVertices.push_back(originalVertices[i].y() / 250);
        parsedVertices.push_back(originalVertices[i].z() / 250);
        if(i == 0){
            std::cout << originalVertices[i].x() << ", " << originalVertices[i].y() << ", " << originalVertices[i].z() << ";" << std::endl;
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

    //renderLoop(800, 800, parsedVertices, properties.triangles);
    //renderWithShaders(800, 800, parsedVertices, properties.triangles, parsedColor, "../../../Data/face.png");
    //renderWithShaders(800, 800, parsedVertices, properties.triangles, parsedColor, "../../../Data/Einstein.jpg");
    renderFaceOnTopOfImage(450, 450, parsedVertices, properties.triangles, parsedColor, "../../../Data/testface.jpg");
    //renderBackground();
}*/

void test2(){

}

int main() {
    InputImage inputImage = readVideoData("../../../Data/20250116_182138.bag");
    const std::string imagePath = std::string("../../../Result/color_frame_corrected.png");
    //const std::string imagePath = std::string("../../../Data/testmyface.png");
    const std::string shapePredictorPath = std::string("../../../Data/shape_predictor_68_face_landmarks.dat");
    const std::string outputPath = std::string("../../../Result/output_corrected.png");
    //const char* imagePath, const char* shapePredictorPath, bool saveResult=false, const char* resultPath=""
    DrawLandmarksOnImage(imagePath, outputPath, shapePredictorPath);
    auto landmarks2D = GetLandmarkVector(imagePath, shapePredictorPath);
    inputImage.landmarks = landmarks2D;
    //printInputImage(inputImage);

    calculateDepthValuesLandmarks(inputImage);

    std::cout << "TEST" << std::endl;
    const std::string outputPlyPath = std::string("../../../Result/outputModel.ply");
    const std::string outputLandmarkPlyPath = std::string("../../../Result/landmarks.ply");
    const std::string h5TestFile = std::string("../../../Data/model2019_face12.h5");

    BfmProperties properties;
    properties = getProperties(h5TestFile, inputImage);

    //setupGLFW(800, 800);
    std::vector<float> parsedVertices;
    auto originalVertices = getVertices(properties);
    for (int i = 0; i < originalVertices.size(); ++i) {
        parsedVertices.push_back(originalVertices[i].x()); // TODO: We have to change the division. We might be able to do this by setting up a projection matrix and camera view matrix and enabling the depth test
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

    std::vector<Eigen::Vector3f> targetPoints;
    //GetTargetLandmarks
    for (int i = 0; i < inputImage.depthValuesLandmarks.size(); ++i) {
        targetPoints.emplace_back(convert2Dto3D(inputImage.landmarks[i], inputImage.depthValuesLandmarks[i], inputImage.intrinsics, inputImage.extrinsics));
    }

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


    Optimization optimizer;
    optimizer.optimizeDenseTerms(properties, inputImage);
    getPointCloud(pointCloudVertices, inputImage.depthValues, color255, "../../../Result/pls.ply", inputImage.intrinsics, inputImage.extrinsics);
    convertVerticesTest(targetPoints, "../../../Result/warumklapptdasnicht.ply");
    convertLandmarksToPly(properties, "../../../Result/BfmTranslationTest.ply");
    convertParametersToPly(properties, "../../../Result/BfmModel.ply");
    renderFaceOnTopOfImage(1280, 720, parsedVertices, properties.triangles, parsedColor, "../../../Result/color_frame_corrected.png", inputImage, properties.transformation);
}
