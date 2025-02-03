#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include <ceres/ceres.h>

#define NUM_SHAPE_PARAMETERS 199
#define NUM_EXPRESSION_PARAMETERS 100
#define NUM_COLOR_PARAMETERS 199

class Optimizer {
public:
    Optimizer(BaselFaceModel* baselFaceModel, InputData* inputData);
    ~Optimizer();
    void optimizeSparseTerms();
    void optimizeDenseGeometryTerm();
    void optimizeDenseColorTerm();
    void optimize();
    void configureSolver();
private:
    BaselFaceModel* m_baselFaceModel;
    InputData* m_inputData;
    ceres::Solver::Options options;
};

struct SparseOptimizationCost {
public:
    SparseOptimizationCost(BaselFaceModel* baselFaceModel, Vector3d landmark_image, int landmark_bfm_index)
            : m_baselFaceModel{baselFaceModel}, m_landmark_image{landmark_image}, m_landmark_bfm_index{landmark_bfm_index} {}

    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        auto shapeMean = m_baselFaceModel->getShapeMean();
        auto expressionMean = m_baselFaceModel->getExpressionMean();

        auto shapePcaBasis = m_baselFaceModel->getShapePcaBasis();
        auto expressionPcaBasis = m_baselFaceModel->getExpressionPcaBasis();

        auto shapeVariance = m_baselFaceModel->getShapePcaVariance();
        auto expressionVariance = m_baselFaceModel->getExpressionPcaVariance();

        auto transformationMatrix = m_baselFaceModel->getTransformation();

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                T(shapeMean[m_landmark_bfm_index * 3]) + T(expressionMean[m_landmark_bfm_index * 3]),
                T(shapeMean[m_landmark_bfm_index * 3 + 1]) + T(expressionMean[m_landmark_bfm_index * 3 + 1]),
                T(shapeMean[m_landmark_bfm_index * 3 + 2]) + T(expressionMean[m_landmark_bfm_index * 3 + 2]),
                T(1)
        );

        for (int i = 0; i < NUM_SHAPE_PARAMETERS; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            T param = T(sqrt(shapeVariance[i])) * shape[i];
            offset += Eigen::Matrix<T, 4, 1>(
                    param * T(shapePcaBasis(vertex_idx, i)),
                    param * T(shapePcaBasis(vertex_idx + 1, i)),
                    param * T(shapePcaBasis(vertex_idx + 2, i)),
                    T(0)
            );
        }

        for (int i = 0; i < NUM_EXPRESSION_PARAMETERS; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            T param = T(sqrt(expressionVariance[i])) * expression[i];
            offset += Eigen::Matrix<T, 4, 1>(
                    param * T(expressionPcaBasis(vertex_idx, i)),
                    param * T(expressionPcaBasis(vertex_idx + 1, i)),
                    param * T(expressionPcaBasis(vertex_idx + 2, i)),
                    T(0)
            );
        }

        Eigen::Matrix<T, 4, 1> transformedVertex = transformationMatrix.cast<T>() * offset;

        residuals[0] = transformedVertex.x() - T(m_landmark_image.x());
        residuals[1] = transformedVertex.y() - T(m_landmark_image.y());
        residuals[2] = transformedVertex.z() - T(m_landmark_image.z());

        return true;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    Vector3d m_landmark_image;
    int m_landmark_bfm_index;
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
