#ifndef FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H
#define FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include "Optimizer.h"
#include "Renderer.h"

class FaceReconstructor {
public:
    FaceReconstructor() = default;
    ~FaceReconstructor() = default;
    static void reconstructFace(BaselFaceModel *baselFaceModel, InputData *inputData, const std::string& path);
};


#endif //FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H
