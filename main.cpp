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

using namespace Eigen;
using namespace std;

int main() {
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
    const std::string imagePath = std::string("../../../Data/face.png");
    const std::string shapePredictorPath = std::string("../../../Data/shape_predictor_68_face_landmarks.dat");
    const std::string outputPath = std::string("../../../Result/output.png");
    //const char* imagePath, const char* shapePredictorPath, bool saveResult=false, const char* resultPath=""

    //DrawLandmarksOnImage(imagePath, outputPath, shapePredictorPath);

    const std::string outputPlyPath = std::string("../../../Result/outputModel.ply");
    const std::string outputLandmarkPlyPath = std::string("../../../Result/landmarks.ply");

    const std::string h5TestFile = std::string("../../../Data/model2019_face12.h5");
    convertParametersToPly(getProperties(h5TestFile), outputPlyPath);
    convertLandmarksToPly(getProperties(h5TestFile), outputLandmarkPlyPath);
    //initializeBFM(h5TestFile);
    //readH5File(h5TestFile);
    //readModelPath(h5TestFile);
    BfmProperties properties;
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
    renderFaceOnTopOfImage(800, 800, parsedVertices, properties.triangles, parsedColor, "../../../Data/face.png");

    //renderBackground();
}
