#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"

class Optimizer {
public:
    Optimizer(BaselFaceModel* baselFaceModel);
    ~Optimizer();
    void optimizeSparseTerms();
    void optimizeDenseGeometryTerm();
    void optimizeDenseColorTerm();
private:
    BaselFaceModel* baselFaceModel;
    //TODO InputData
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
