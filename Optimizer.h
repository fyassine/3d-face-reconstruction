#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"
#include "SingleInputFrame.h"

class Optimizer {
public:
    Optimizer(BaselFaceModel* baselFaceModel, SingleInputFrame* singleInputFrame);
    ~Optimizer();
    void optimizeSparseTerms();
    void optimizeDenseGeometryTerm();
    void optimizeDenseColorTerm();
    void optimize();
private:
    BaselFaceModel* m_baselFaceModel;
    SingleInputFrame* m_singleInputFrame;
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
