#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include <ceres/ceres.h>

#define NUM_SHAPE_PARAMETERS 199
#define NUM_EXPRESSION_PARAMETERS 100
#define NUM_COLOR_PARAMETERS 199

//#define SHAPE_REG_WEIGHT_SPARSE 0.002
//#define EXPRESSION_REG_WEIGHT_SPARSE 0.001

#define SHAPE_REG_WEIGHT_SPARSE 0.01
#define EXPRESSION_REG_WEIGHT_SPARSE 0.01

#define SHAPE_REG_WEIGHT_DENSE 0.03
#define EXPRESSION_REG_WEIGHT_DENSE 0.005
#define COLOR_REG_WEIGHT_DENSE 0.07

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

//TODO: Set weights so that all are on same scale
struct DenseOptimizationCost {
public:
    DenseOptimizationCost(BaselFaceModel* baselFaceModel, Vector3d point_image, int landmark_bfm_index)
            : m_baselFaceModel{baselFaceModel}, m_point_image{point_image}, m_landmark_bfm_index{landmark_bfm_index} {}

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

        residuals[0] = transformedVertex.x() - T(m_point_image.x()); //TODO: l2 norm
        residuals[1] = transformedVertex.y() - T(m_point_image.y()); //compute cos between two normals; 1- cos
        residuals[2] = transformedVertex.z() - T(m_point_image.z());

        return true;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    Vector3d m_point_image;
    int m_landmark_bfm_index;
};

struct ColorOptimizationCost {
public:
    ColorOptimizationCost(BaselFaceModel* baselFaceModel, Vector3d colorImage, int vertexIndex)
            : m_baselFaceModel{baselFaceModel}, m_color_image{colorImage}, m_vertex_index{vertexIndex} {}

    template <typename T>
    bool operator()(const T* const color,
                    T* residuals) const {

        auto colorMean = m_baselFaceModel->getColorMean();
        auto colorPcaBasis = m_baselFaceModel->getColorPcaBasis();
        auto colorVariance = m_baselFaceModel->getColorPcaVariance();

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                T(colorMean[m_vertex_index * 3]),
                T(colorMean[m_vertex_index * 3 + 1]),
                T(colorMean[m_vertex_index * 3 + 2]),
                T(1)
        );

        for (int i = 0; i < NUM_COLOR_PARAMETERS; ++i) {
            int vertex_idx = m_vertex_index * 3;
            T param = T(sqrt(colorVariance[i])) * color[i];
            offset += Eigen::Matrix<T, 4, 1>(
                    param * T(colorPcaBasis(vertex_idx, i)),
                    param * T(colorPcaBasis(vertex_idx + 1, i)),
                    param * T(colorPcaBasis(vertex_idx + 2, i)),
                    T(0)
            );
        }

        residuals[0] = offset.x() - T(m_color_image.x()); //l2 norm
        residuals[1] = offset.y() - T(m_color_image.y()); //compute cos between two normals; 1 - cos
        residuals[2] = offset.z() - T(m_color_image.z());

        return true;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    Vector3d m_color_image;
    int m_vertex_index;
};

struct GeometryRegularizationCost {
    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residual) const {
        T reg_energy_geometry = T(0);
        T reg_energy_expression = T(0);

        for (int i = 0; i < num_identity_params; ++i) {
            reg_energy_geometry += pow(shape[i] / T(identity_std_dev[i]), 2);
        }
        for (int i = 0; i < num_expression_params; ++i) {
            reg_energy_expression += pow(expression[i] / T(expression_std_dev[i]), 2);
        }

        residual[0] = reg_energy_geometry;
        residual[1] = reg_energy_expression;
        return true;
    }

    GeometryRegularizationCost(const std::vector<double>& id_std,
                               const std::vector<double>& exp_std)
            : identity_std_dev(id_std)
            , expression_std_dev(exp_std) {}

    const std::vector<double> identity_std_dev;
    const std::vector<double> expression_std_dev;

    static constexpr int num_identity_params = 199;
    static constexpr int num_expression_params = 100;
};

struct ShapeRegularizerCost
{
    ShapeRegularizerCost(double shapeWeight, std::vector<double> variance) : m_shape_weight(shapeWeight), m_variance(variance) {}

    template<typename T>
    bool operator()(T const* shape, T* residuals) const
    {
        for (int i = 0; i < NUM_SHAPE_PARAMETERS; i++) {
            //residuals[i] = shape[i] * T(m_shape_weight); //squared values -> l2 norm
            residuals[i] = (shape[i] / sqrt(m_variance[i])) * m_shape_weight;
        }
        return true;
    }

private:
    double m_shape_weight;
    std::vector<double> m_variance;
};

struct ExpressionRegularizerCost
{
    ExpressionRegularizerCost(double expressionWeight, std::vector<double> variance) : m_expression_weight(expressionWeight), m_variance(variance) {}

    template<typename T>
    bool operator()(T const* expression, T* residuals) const
    {
        for (int i = 0; i < NUM_EXPRESSION_PARAMETERS; i++) {
            //residuals[i] = expression[i] * T(m_expression_weight); //TODO: Try to divide by eigenvalue -> var
            residuals[i] = (expression[i] / sqrt(m_variance[i])) * m_expression_weight;
        }
        return true;
    }

private:
    double m_expression_weight;
    std::vector<double> m_variance;
};

struct ColorRegularizerCost
{
    ColorRegularizerCost(double colorWeight, std::vector<double> variance) : m_color_weight(colorWeight), m_variance(variance) {}

    template<typename T>
    bool operator()(T const* color, T* residuals) const
    {
        for (int i = 0; i < NUM_COLOR_PARAMETERS; i++) {
            //residuals[i] = color[i] * T(m_color_weight);
            residuals[i] = color[i] / m_variance[i];
        }
        return true;
    }

private:
    double m_color_weight;
    std::vector<double> m_variance;
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
