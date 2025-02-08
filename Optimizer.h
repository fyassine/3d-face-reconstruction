#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include <ceres/ceres.h>

#include <utility>

#define NUM_SHAPE_PARAMETERS 199
#define NUM_EXPRESSION_PARAMETERS 100
#define NUM_COLOR_PARAMETERS 199

#define SHAPE_REG_WEIGHT_SPARSE 1
#define EXPRESSION_REG_WEIGHT_SPARSE 1

#define SHAPE_REG_WEIGHT_DENSE 1
#define EXPRESSION_REG_WEIGHT_DENSE 1
#define COLOR_REG_WEIGHT_DENSE 1

#define OUTLIER_THRESHOLD 0.004

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
            : m_baselFaceModel{baselFaceModel}, m_landmark_image{std::move(landmark_image)}, m_landmark_bfm_index{landmark_bfm_index} {}

    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        auto& shapeMean = m_baselFaceModel->getShapeMean();
        auto& expressionMean = m_baselFaceModel->getExpressionMean();

        auto& shapePcaBasis = m_baselFaceModel->getShapePcaBasis();
        auto& expressionPcaBasis = m_baselFaceModel->getExpressionPcaBasis();

        auto& shapeVariance = m_baselFaceModel->getShapePcaVariance();
        auto& expressionVariance = m_baselFaceModel->getExpressionPcaVariance();

        auto transformationMatrix = m_baselFaceModel->getTransformation();

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                T(shapeMean[m_landmark_bfm_index * 3]) + T(expressionMean[m_landmark_bfm_index * 3]),
                T(shapeMean[m_landmark_bfm_index * 3 + 1]) + T(expressionMean[m_landmark_bfm_index * 3 + 1]),
                T(shapeMean[m_landmark_bfm_index * 3 + 2]) + T(expressionMean[m_landmark_bfm_index * 3 + 2]),
                T(1)
        );

        int vertex_idx = m_landmark_bfm_index * 3;

        for (int i = 0; i < NUM_SHAPE_PARAMETERS; ++i) {
            T param = T(sqrt(shapeVariance[i])) * shape[i];
            offset.x() += param * T(shapePcaBasis(vertex_idx, i));
            offset.y() += param * T(shapePcaBasis(vertex_idx + 1, i));
            offset.z() += param * T(shapePcaBasis(vertex_idx + 2, i));
        }

        for (int i = 0; i < NUM_EXPRESSION_PARAMETERS; ++i) {
            T param = T(sqrt(expressionVariance[i])) * expression[i];
            offset.x() += param * T(expressionPcaBasis(vertex_idx, i));
            offset.y() += param * T(expressionPcaBasis(vertex_idx + 1, i));
            offset.z() += param * T(expressionPcaBasis(vertex_idx + 2, i));
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
            : m_baselFaceModel{baselFaceModel}, m_point_image{std::move(point_image)}, m_landmark_bfm_index{landmark_bfm_index} {}

    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        auto& shapeMean = m_baselFaceModel->getShapeMean();
        auto& expressionMean = m_baselFaceModel->getExpressionMean();
        auto& shapePcaBasis = m_baselFaceModel->getShapePcaBasis();
        auto& expressionPcaBasis = m_baselFaceModel->getExpressionPcaBasis();
        auto& shapeVariance = m_baselFaceModel->getShapePcaVariance();
        auto& expressionVariance = m_baselFaceModel->getExpressionPcaVariance();
        auto& transformationMatrix = m_baselFaceModel->getTransformation();

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                T(shapeMean[m_landmark_bfm_index * 3]) + T(expressionMean[m_landmark_bfm_index * 3]),
                T(shapeMean[m_landmark_bfm_index * 3 + 1]) + T(expressionMean[m_landmark_bfm_index * 3 + 1]),
                T(shapeMean[m_landmark_bfm_index * 3 + 2]) + T(expressionMean[m_landmark_bfm_index * 3 + 2]),
                T(1)
        );

        int vertex_idx = m_landmark_bfm_index * 3;

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> shapeParams(shape, NUM_SHAPE_PARAMETERS);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> expressionParams(expression, NUM_EXPRESSION_PARAMETERS);
        Eigen::Map<const Eigen::VectorXd> shapeVarEigen(shapeVariance.data(), NUM_SHAPE_PARAMETERS);
        Eigen::Map<const Eigen::VectorXd> exprVarEigen(expressionVariance.data(), NUM_EXPRESSION_PARAMETERS);
        Eigen::Matrix<T, Eigen::Dynamic, 1> sqrt_shape_var = shapeParams.cwiseProduct(shapeVarEigen.array().sqrt().matrix().template cast<T>());
        Eigen::Matrix<T, Eigen::Dynamic, 1> sqrt_expr_var = expressionParams.cwiseProduct(exprVarEigen.array().sqrt().matrix().template cast<T>());
        Eigen::Matrix<T, 3, 1> shape_offset = shapePcaBasis.block(vertex_idx, 0, 3, NUM_SHAPE_PARAMETERS).template cast<T>() * sqrt_shape_var;
        Eigen::Matrix<T, 3, 1> expr_offset = expressionPcaBasis.block(vertex_idx, 0, 3, NUM_EXPRESSION_PARAMETERS).template cast<T>() * sqrt_expr_var;

        offset.x() += shape_offset.x() + expr_offset.x();
        offset.y() += shape_offset.y() + expr_offset.y();
        offset.z() += shape_offset.z() + expr_offset.z();

        /*for (int i = 0; i < NUM_SHAPE_PARAMETERS; ++i) {
            T param = T(sqrt(shapeVariance[i])) * shape[i];
            offset.x() += param * T(shapePcaBasis(vertex_idx, i));
            offset.y() += param * T(shapePcaBasis(vertex_idx + 1, i));
            offset.z() += param * T(shapePcaBasis(vertex_idx + 2, i));
        }

        for (int i = 0; i < NUM_EXPRESSION_PARAMETERS; ++i) {
            T param = T(sqrt(expressionVariance[i])) * expression[i];
            offset.x() += param * T(expressionPcaBasis(vertex_idx, i));
            offset.y() += param * T(expressionPcaBasis(vertex_idx + 1, i));
            offset.z() += param * T(expressionPcaBasis(vertex_idx + 2, i));
        }*/

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
            residuals[i] = pow((shape[i] / sqrt(m_variance[i])), 2) * m_shape_weight;
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
            residuals[i] = pow((expression[i] / sqrt(m_variance[i])), 2) * m_expression_weight;
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
            residuals[i] = pow((color[i] / sqrt(m_variance[i])), 2) * m_color_weight; //color[i] / m_variance[i];
        }
        return true;
    }

private:
    double m_color_weight;
    std::vector<double> m_variance;
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
