#include <iostream>
#include <fstream>

#include "Eigen.h"

#include "FacialLandmarks.h"
#include "ProcrustesAligner.h"
#include "BaselFaceModel.h"

//#include "Renderer.h"
//#include "InputDataExtractor.h"
#include "InputData.h"
#include "Optimizer.h"
#include "ModelConverter.h"
#include "FaceReconstructor.h"

using namespace Eigen;
using namespace std;

#define LEO_LOOKING_NORMAL "20250127_200932.bag"
#define NELI_LOOKING_SERIOUS "20250116_183206.bag"
#define LEO_CRAZY "20250201_195224.bag"
#define LEO_LONG "20250205_172132.bag"
#define LEO_NEUTRAL_BACKGROUND "20250207_115228.bag"
#define LEO_EXPRESSIONS "20250207_115412.bag"
#define LEO_VID "20250205_172132.bag"

int main(){
    InputData inputSource = InputDataExtractor::extractInputData(LEO_LOOKING_NORMAL);
    BaselFaceModel sourceBaselFaceModel;
    FaceReconstructor::reconstructFace(&sourceBaselFaceModel, &inputSource);
    return 0;
}