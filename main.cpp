#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <Eigen/Dense>
#include <vector>

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
}
