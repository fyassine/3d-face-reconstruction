#ifndef FACE_RECONSTRUCTION_OPTIMIZER_H
#define FACE_RECONSTRUCTION_OPTIMIZER_H

#include "BaselFaceModel.h"
#include "InputData.h"
#include <ceres/ceres.h>

#include <utility>

#define NUM_SHAPE_PARAMETERS 199
#define NUM_EXPRESSION_PARAMETERS 100
#define NUM_COLOR_PARAMETERS 199

#define SHAPE_REG_WEIGHT_SPARSE 0.05
#define EXPRESSION_REG_WEIGHT_SPARSE 0.05

#define SHAPE_REG_WEIGHT_DENSE 0.05
#define EXPRESSION_REG_WEIGHT_DENSE 0.05
#define COLOR_REG_WEIGHT_DENSE 0.05

#define OUTLIER_THRESHOLD 0.015 //0.01//0.004

class Optimizer {
public:
    Optimizer(BaselFaceModel* baselFaceModel, InputData* inputData);
    ~Optimizer();
    void optimizeSparseTerms();
    void optimizeDenseTerms();
    void configureSolver();

    // NEW: A method to set the weights at runtime.
    void setWeights(double shapeSparse,
                    double expressionSparse,
                    double shapeDense,
                    double expressionDense,
                    double colorDense)
    {
        m_shapeRegWeightSparse = shapeSparse;
        m_expressionRegWeightSparse = expressionSparse;
        m_shapeRegWeightDense = shapeDense;
        m_expressionRegWeightDense = expressionDense;
        m_colorRegWeightDense = colorDense;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    InputData* m_inputData;
    ceres::Solver::Options options;

    double m_shapeRegWeightSparse = DEFAULT_SHAPE_REG_WEIGHT_SPARSE;
    double m_expressionRegWeightSparse = DEFAULT_EXPRESSION_REG_WEIGHT_SPARSE;
    double m_shapeRegWeightDense = DEFAULT_SHAPE_REG_WEIGHT_DENSE;
    double m_expressionRegWeightDense = DEFAULT_EXPRESSION_REG_WEIGHT_DENSE;
    double m_colorRegWeightDense = DEFAULT_COLOR_REG_WEIGHT_DENSE;
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

        auto& transformationMatrix = m_baselFaceModel->getTransformation();

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
        auto normal = m_baselFaceModel->getNormals()[m_landmark_bfm_index];

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                T(shapeMean[m_landmark_bfm_index * 3]) + T(expressionMean[m_landmark_bfm_index * 3]),
                T(shapeMean[m_landmark_bfm_index * 3 + 1]) + T(expressionMean[m_landmark_bfm_index * 3 + 1]),
                T(shapeMean[m_landmark_bfm_index * 3 + 2]) + T(expressionMean[m_landmark_bfm_index * 3 + 2]),
                T(1)
        );

        int vertex_idx = m_landmark_bfm_index * 3;

        Map<const Matrix<T, Dynamic, 1>> shapeParams(shape, NUM_SHAPE_PARAMETERS);
        Map<const Matrix<T, Dynamic, 1>> expressionParams(expression, NUM_EXPRESSION_PARAMETERS);
        Map<const VectorXd> shapeVarEigen(shapeVariance.data(), NUM_SHAPE_PARAMETERS);
        Map<const VectorXd> exprVarEigen(expressionVariance.data(), NUM_EXPRESSION_PARAMETERS);
        Matrix<T, Dynamic, 1> sqrt_shape_var = shapeParams.cwiseProduct(shapeVarEigen.array().sqrt().matrix().template cast<T>());
        Matrix<T, Dynamic, 1> sqrt_expr_var = expressionParams.cwiseProduct(exprVarEigen.array().sqrt().matrix().template cast<T>());
        Matrix<T, 3, 1> shape_offset = shapePcaBasis.block(vertex_idx, 0, 3, NUM_SHAPE_PARAMETERS).template cast<T>() * sqrt_shape_var;
        Matrix<T, 3, 1> expr_offset = expressionPcaBasis.block(vertex_idx, 0, 3, NUM_EXPRESSION_PARAMETERS).template cast<T>() * sqrt_expr_var;

        offset.x() += shape_offset.x() + expr_offset.x();
        offset.y() += shape_offset.y() + expr_offset.y();
        offset.z() += shape_offset.z() + expr_offset.z();

        Matrix<T, 4, 1> transformedVertex = transformationMatrix.cast<T>() * offset;

        residuals[0] = transformedVertex.x() - T(m_point_image.x());
        residuals[1] = transformedVertex.y() - T(m_point_image.y());
        residuals[2] = transformedVertex.z() - T(m_point_image.z());

        //residuals[3] = (transformedVertex.x() - T(m_point_image.x())) * T(normal.x());
        //residuals[4] = (transformedVertex.y() - T(m_point_image.y())) * T(normal.y());
        //residuals[5] = (transformedVertex.z() - T(m_point_image.z())) * T(normal.z());

        residuals[3] = Matrix<T, 3, 1>(transformedVertex.x() - T(m_point_image.x()),
                                      transformedVertex.y() - T(m_point_image.y()),
                                      transformedVertex.z() - T(m_point_image.z())).dot(normal.cast<T>());

        //TODO: point-to-point, point-to-plane

        return true;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    Vector3d m_point_image;
    int m_landmark_bfm_index;
};

struct ColorOptimizationCost {
public:
    ColorOptimizationCost(BaselFaceModel* baselFaceModel,
                          Vector3d colorImage,
                          int vertexIndex)
    : m_baselFaceModel{baselFaceModel}, m_color_image{std::move(colorImage)}, m_vertex_index{vertexIndex} {}

    template <typename T>
    bool operator()(const T* const color,
                    const T* const illumination,
                    T* residuals) const {

        auto& colorMean = m_baselFaceModel->getColorMean();
        auto& colorPcaBasis = m_baselFaceModel->getColorPcaBasis();
        auto& colorVariance = m_baselFaceModel->getColorPcaVariance();
        auto& normals = m_baselFaceModel->getNormals();
        int vertex_idx = m_vertex_index * 3;

        Eigen::Matrix<T, 4, 1> offset = Eigen::Matrix<T, 4, 1>(
                                                               T(colorMean[m_vertex_index * 3]),
                                                               T(colorMean[m_vertex_index * 3 + 1]),
                                                               T(colorMean[m_vertex_index * 3 + 2]),
                                                               T(1)
                                                               );

        for (int i = 0; i < NUM_COLOR_PARAMETERS; ++i) {
            T param = T(sqrt(colorVariance[i])) * color[i];
            offset.x() += param * T(colorPcaBasis(vertex_idx, i));
            offset.y() += param * T(colorPcaBasis(vertex_idx + 1, i));
            offset.z() += param * T(colorPcaBasis(vertex_idx + 2, i));
        }


       // Get normal for this vertex
       //Eigen::Matrix<T, 3, 1> normal = m_baselFaceModel->getNormals()[m_vertex_index].template cast<T>();
       Eigen::Matrix<T, 3, 1> normal = normals[m_vertex_index].template cast<T>();
        // Compute SH basis
       T shBasis[9];
       // Compute SH basis functions up to the second order (9 terms)
       shBasis[0] = T(0.28209479);
       shBasis[1] = T(0.48860251) * normal.y();
       shBasis[2] = T(0.48860251) * normal.z();
       shBasis[3] = T(0.48860251) * normal.x();
       shBasis[4] = T(1.09254843) * normal.x() * normal.y();
       shBasis[5] = T(1.09254843) * normal.y() * normal.z();
       shBasis[6] = T(0.31539157) * (T(3) * normal.z() * normal.z() - T(1));
       shBasis[7] = T(1.09254843) * normal.x() * normal.z();
       shBasis[8] = T(0.54627421) * (normal.x() * normal.x() - normal.y() * normal.y());

       // Apply SH lighting
       Eigen::Matrix<T, 3, 1> shLighting = Eigen::Matrix<T, 3, 1>::Zero();
       for (int i = 0; i < 27; i+=3) {
           shLighting(0) += shBasis[i / 3] * T(illumination[i]);  // Red
           shLighting(1) += shBasis[i / 3] * T(illumination[i + 1]);  // Green
           shLighting(2) += shBasis[i / 3] * T(illumination[i + 2]);  // Blue
       }

       // Apply SH illumination to color
       Eigen::Matrix<T, 3, 1> shadedColor = offset.template head<3>().cwiseProduct(shLighting);

        residuals[0] = offset.x() - T(m_color_image.x());
        residuals[1] = offset.y() - T(m_color_image.y());
        residuals[2] = offset.z() - T(m_color_image.z());

        return true;
    }

private:
    BaselFaceModel* m_baselFaceModel;
    Vector3d m_color_image;
    int m_vertex_index;
};

struct ShapeRegularizerCost
{
    ShapeRegularizerCost(double shapeWeight, std::vector<double> variance) : m_shape_weight(shapeWeight), m_variance(variance) {}

    template<typename T>
    bool operator()(T const* shape, T* residuals) const
    {
        for (int i = 0; i < NUM_SHAPE_PARAMETERS; i++) {
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
    ColorRegularizerCost(double colorWeight, std::vector<double> variance) : m_color_weight(colorWeight), m_variance(std::move(variance)) {}

    template<typename T>
    bool operator()(T const* color, T* residuals) const
    {
        for (int i = 0; i < NUM_COLOR_PARAMETERS; i++) {
            residuals[i] = (color[i] / sqrt(m_variance[i])) * m_color_weight;
        }
        return true;
    }

private:
    double m_color_weight;
    std::vector<double> m_variance;
};

struct WeightSearch
{
public:
    static void runSparseWeightTrials(const std::string &bagPath);
    static void runDenseWeightTrials(const std::string &bagPath);
    static void runSparseWeightTrial(const std::string &bagPath,
                      double shapeWeight,
                      double expressionWeight);
    static void runDenseWeightTrial(const std::string &bagPath,
                             double shapeWeight,
                             double expressionWeight,
                             double colorWeight);
};


#endif //FACE_RECONSTRUCTION_OPTIMIZER_H
