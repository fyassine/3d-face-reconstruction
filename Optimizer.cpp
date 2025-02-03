#include "Optimizer.h"

Optimizer::Optimizer(BaselFaceModel *baselFaceModel, SingleInputFrame *singleInputFrame) {
    m_baselFaceModel = baselFaceModel;
    m_singleInputFrame = singleInputFrame;
}

Optimizer::~Optimizer() = default;

void Optimizer::optimizeSparseTerms() {

}

void Optimizer::optimizeDenseGeometryTerm() {

}

void Optimizer::optimizeDenseColorTerm() {

}

void Optimizer::optimize() {
    optimizeSparseTerms();
    optimizeDenseGeometryTerm();
    optimizeDenseColorTerm();
}
