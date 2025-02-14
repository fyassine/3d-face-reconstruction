#ifndef FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H
#define FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include "Optimizer.h"
#include "Renderer.h"
#include "ModelConverter.h"

class FaceReconstructor {
public:
    FaceReconstructor() = default;
    ~FaceReconstructor() = default;
    static void reconstructFace(BaselFaceModel *baselFaceModel, InputData *inputData);
    static void expressionTransfer(BaselFaceModel *sourceFaceModel, BaselFaceModel *targetFaceModel, InputData *sourceData, InputData *targetData);
};


#endif //FACE_RECONSTRUCTION_FACERECONSTRUCTOR_H
