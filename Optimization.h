#ifndef FACE_RECONSTRUCTION_OPTIMIZATION_H
#define FACE_RECONSTRUCTION_OPTIMIZATION_H
#include <ceres/ceres.h>
#include "Eigen.h"
#include "BFMParameters.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

// TODO: Put illumination in seperate header file

using json = nlohmann::json;

struct Illumination {
    
    static Eigen::Vector3f computeIllumination(const Eigen::Vector3f& normal,
                                               const Eigen::Matrix3f& gammaMatrix) {
        // Compute SH basis functions (up to order 2)
        double x = normal.x();
        double y = normal.y();
        double z = normal.z();
        std::vector<double> B(9);
        B[0] = 1.0;              // l=0
        B[1] = y;                // l=1
        B[2] = z;                // l=1
        B[3] = x;                // l=1
        B[4] = x * y;            // l=2
        B[5] = y * z;            // l=2
        B[6] = 2 * z * z - x * x - y * y; // l=2
        B[7] = x * z;            // l=2
        B[8] = x * x - y * y;    // l=2
        
        // Compute illumination for each color channel using the gammaMatrix
        float L_R = 0.0f, L_G = 0.0f, L_B = 0.0f; // Use float for consistent types
        for (int i = 0; i < 9; ++i) {
              L_R += gammaMatrix(0, i % 3) * B[i];  // Red: coefficients in row 0
              L_G += gammaMatrix(1, i % 3) * B[i];  // Green: coefficients in row 1
              L_B += gammaMatrix(2, i % 3) * B[i];  // Blue: coefficients in row 2
          }
        
        // Return the RGB illumination as an Eigen::Vector3f
        return Eigen::Vector3f(L_R, L_G, L_B);
    }
};

//TODO: THIS SCRIPT IS SUBJECT TO CHANGE!!! Don't look!!! It's ugly!!!

struct GeometryOptimization {
public:
    GeometryOptimization(const Eigen::Vector3f& vertex,
                         const float& depth,
                         const Eigen::Vector3f& normal,
                         const Eigen::MatrixXf& shapePcaBasis,
                         const Eigen::MatrixXf& expressionPcaBasis,
                         const std::vector<float>& shapeMean,
                         const std::vector<float>& expressionMean,
                         const std::vector<float>& shapeVariance,
                         const std::vector<float>& expressionVariance,
                         int vertex_id) :
            m_vertex(vertex), m_depth(depth), m_normal(normal),
            m_shapePcaBasis(shapePcaBasis), m_expressionBasis{expressionPcaBasis}, m_shapeMean{shapeMean},
            m_expressionMean{expressionMean}, m_shapeVariance{shapeVariance},
            m_expressionVariance{expressionVariance}, m_vertex_id(vertex_id) {}

    template<typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();

        shape_offset.x() = T(m_shapeMean[m_vertex_id * 3]);
        shape_offset.y() = T(m_shapeMean[m_vertex_id * 3 + 1]);
        shape_offset.z() = T(m_shapeMean[m_vertex_id * 3 + 2]);

        expression_offset.x() = T(m_expressionMean[m_vertex_id * 3]);
        expression_offset.y() = T(m_expressionMean[m_vertex_id * 3 + 1]);
        expression_offset.z() = T(m_expressionMean[m_vertex_id * 3 + 2]);

        // Each parameter influences a single vertex coordinate
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            shape_offset += Eigen::Matrix<T, 3, 1>(
                    //T value = T(sqrt(bfm.shape_pca_var[i])) * shape_weights[i];
                    //face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx * 3, i))),        //vllt. column und row vertauschen?!
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx * 3 + 1, i))), //maybe rows +1 wrong? Instead rows +0
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx * 3 + 2, i)))     //maybe rows +1 wrong? Instead rows +0, wenn 0 dann ganzes model standard,
                                                                                 //aber wenn != 0, dann wären das ja einfach random die nachbar werte
            );
        }

        for (int i = 0; i < num_expression_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            expression_offset += Eigen::Matrix<T, 3, 1>(
                    T(expression[i] * T(m_expressionBasis(vertex_idx * 3, i))),
                    T(expression[i] * T(m_expressionBasis(vertex_idx * 3 + 1, i))), //maybe rows +1 wrong? Instead rows +0
                    T(expression[i] * T(m_expressionBasis(vertex_idx * 3 + 2, i))) //maybe rows +1 wrong? Instead rows +0
            );
        }
        Eigen::Matrix<T, 3, 1> transformedVertex = m_vertex.cast<T>() + shape_offset + expression_offset;

        //transformedVertex = m_vertex.cast<T>(); //TODO: Remove the line and think of a solution for the pcaBasis

        T point_to_point = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).norm();
        T point_to_plane = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).dot(m_normal.cast<T>());

        residuals[0] = point_to_point;
        residuals[1] = point_to_plane;

        return true;
    }

private:
    const Eigen::Vector3f m_vertex;
    const float m_depth;
    const Eigen::Vector3f m_normal;

    const std::vector<float>& m_shapeMean;
    const std::vector<float>& m_expressionMean;

    const std::vector<float>& m_shapeVariance;
    const std::vector<float>& m_expressionVariance;

    static const int num_shape_params = 199;
    static const int num_expression_params = 100;

    const Eigen::MatrixXf& m_shapePcaBasis;
    const Eigen::MatrixXf& m_expressionBasis;
    const int m_vertex_id;
};

struct ColorOptimization {
public:
    ColorOptimization(const Eigen::Vector3f& albedo, const Eigen::Vector3f& image_color, const Eigen::Vector3f& illumination, const Eigen::MatrixXf& colorPcaBasis, int color_id)
        : m_albedo(albedo), m_image_color(image_color), m_illumination(illumination), m_colorPcaBasis(colorPcaBasis), m_color_id(color_id) {}

    template <typename T>
    bool operator()(const T* const color, T* residuals) const {

        // Normalize illumination
        /*T illumination_normalized[3];
        T illum_sum = m_illumination.cast<T>().sum();


        if (illum_sum == T(0)) {
            return false;
        }

        illumination_normalized[0] = T(m_illumination.x()) / illum_sum;
        illumination_normalized[1] = T(m_illumination.y()) / illum_sum;
        illumination_normalized[2] = T(m_illumination.z()) / illum_sum;

        // Compute adjusted albedo with normalized illumination
        Eigen::Matrix<T, 3, 1> adjusted_albedo = m_albedo.cast<T>().cwiseProduct(
        Eigen::Matrix<T, 3, 1>(illumination_normalized[0], illumination_normalized[1], illumination_normalized[2]));*/

        // Compute color offset
        Eigen::Matrix<T, 3, 1> color_offset = Eigen::Matrix<T, 3, 1>::Zero();
        for (int i = 0; i < num_color_params; ++i) {
            int color_idx = m_color_id * 3;
            color_offset += Eigen::Matrix<T, 3, 1>(
                    T(color[i] * T(m_colorPcaBasis(color_idx, i))),
                    T(color[i] * T(m_colorPcaBasis(color_idx + 1, i))), //maybe rows +1 wrong? Instead rows +0
                    T(color[i] * T(m_colorPcaBasis(color_idx + 2, i)))  //maybe rows +1 wrong? Instead rows +0
            );
        }
        
        // Apply illumination scaling directly
        Eigen::Matrix<T, 3, 1> albedo = m_albedo.cast<T>() + color_offset;

        // Illumination is already computed and passed as an argument
        Eigen::Matrix<T, 3, 1> illumination = m_illumination.cast<T>(); // Converted to T for optimization

        // Apply illumination to color (this step assumes illumination modifies the albedo in RGB channels)
        Eigen::Matrix<T, 3, 1> modifiedColor = albedo.cwiseProduct(illumination);

        // Compute residuals (compare modified color to image color)
        T resulting_color = Eigen::Matrix<T, 3, 1>(modifiedColor.x() - T(m_image_color.x()),
                                                   modifiedColor.y() - T(m_image_color.y()),
                                                   modifiedColor.z() - T(m_image_color.z())).norm();
        
//        // Uncomment everything below together (if needed)
//        Eigen::Matrix<T, 3, 1> modifiedColor = m_albedo.cast<T>() + color_offset;
//
//        T resulting_color = Eigen::Matrix<T, 3, 1>(modifiedColor.x() - T(m_image_color.x()),
//                                                   modifiedColor.y() - T(m_image_color.y()),
//                                                   modifiedColor.z() - T(m_image_color.z())).norm();
//        // Compute per-channel residuals
//        //residuals[0] = adjusted_albedo.x() - color_offset.x();
//        //residuals[1] = adjusted_albedo.y() - color_offset.y();
//        //residuals[2] = adjusted_albedo.z() - color_offset.z();
//        /*residuals[0] = modifiedColor.x() - T(m_image_color.x());
//        residuals[1] = modifiedColor.y() - T(m_image_color.y());
//        residuals[2] = modifiedColor.z() - T(m_image_color.z());*/
        residuals[0] = resulting_color;
        return true;
    }

private:
    const Eigen::Vector3f m_albedo;
    const Eigen::Vector3f m_image_color;
    const Eigen::Vector3f m_illumination;
    const Eigen::MatrixXf& m_colorPcaBasis;
    const int m_color_id;
    static const int num_color_params = 199;
};

/*struct SparseOptimization{
public:
    SparseOptimization(const Eigen::Vector2f& landmark_position_input, const Eigen::Vector2f& landmark_bfm)
            : m_landmark_positions_input(landmark_position_input) {}

    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();

        // Each parameter influences a single vertex coordinate
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            shape_offset += Eigen::Matrix<T, 3, 1>(
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx, i))),        //vllt. column und row vertauschen?!
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 1, i))), //maybe rows +1 wrong? Instead rows +0
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 2, i)))     //maybe rows +1 wrong? Instead rows +0, wenn 0 dann ganzes model standard,
                    //aber wenn != 0, dann wären das ja einfach random die nachbar werte
            );
        }

        for (int i = 0; i < num_expression_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            expression_offset += Eigen::Matrix<T, 3, 1>(
                    T(expression[i] * T(m_expressionBasis(vertex_idx, i))),
                    T(expression[i] * T(m_expressionBasis(vertex_idx + 1, i))), //maybe rows +1 wrong? Instead rows +0
                    T(expression[i] * T(m_expressionBasis(vertex_idx + 2, i))) //maybe rows +1 wrong? Instead rows +0
            );
        }
        Eigen::Matrix<T, 3, 1> transformedVertex = m_vertex.cast<T>() + shape_offset + expression_offset;

        //transformedVertex = m_vertex.cast<T>(); //TODO: Remove the line and think of a solution for the pcaBasis

        T point_to_point = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).norm();
        T point_to_plane = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).dot(m_normal.cast<T>());

        residuals[0] = point_to_point;
        residuals[1] = point_to_plane;

        return true;


        T result = Eigen::Matrix<T, 2, 1>(landmark_position_bfm[0] - T(m_landmark_positions_input.x()),
                                          landmark_position_bfm[1] - T(m_landmark_positions_input.y())
                                          ).norm();

        residuals[0] = result;
        return true;
    }

private:
    const Eigen::Vector2f m_landmark_positions_input;

    static const int num_shape_params = 199;
    static const int num_expression_params = 100;

    const Eigen::MatrixXf& m_shapePcaBasis;
    const Eigen::MatrixXf& m_expressionBasis;
};*/

class Optimization {
public:
    static void optimize(BfmProperties&, InputImage&);
private:
    static void configureSolver(ceres::Solver::Options& options);
    static void optimizeSparseTerms();
    static void optimizeDenseTerms(BfmProperties&, InputImage&);
    static void optimizeColor();
    static void regularize(BfmProperties&);
};

struct RegularizationFunction
{
    RegularizationFunction(double weight, int number)
            : m_weight { weight }, m_number {number}
    {}

    template<typename T>
    bool operator()(T const* params, T* residuals) const
    {
        for (int i = 0; i < m_number; ++i) {
            residuals[i] = params[i] * T(sqrt(m_weight));
        }
        return true;
    }

private:
    const double m_weight;
    const int m_number;
};

struct RegularizationTerm {
    template <typename T>
    bool operator()(const T* const identity_params,
                    const T* const albedo_params,
                    const T* const expression_params,
                    T* residual) const {
        T reg_energy = T(0);

        // Identity parameters regularization
        for (int i = 0; i < num_identity_params; ++i) {
            reg_energy += pow(identity_params[i] / T(identity_std_dev[i]), 2);
        }

        // Albedo parameters regularization
        for (int i = 0; i < num_albedo_params; ++i) {
            reg_energy += pow(albedo_params[i] / T(albedo_std_dev[i]), 2);
        }

        // Expression parameters regularization
        for (int i = 0; i < num_expression_params; ++i) {
            reg_energy += pow(expression_params[i] / T(expression_std_dev[i]), 2);
        }

        residual[0] = reg_energy;
        return true;
    }

    // Constructor to pass standard deviations if they're not constant
    RegularizationTerm(const std::vector<double>& id_std,
                       const std::vector<double>& alb_std,
                       const std::vector<double>& exp_std)
            : identity_std_dev(id_std)
            , albedo_std_dev(alb_std)
            , expression_std_dev(exp_std) {}

    const std::vector<double> identity_std_dev;
    const std::vector<double> albedo_std_dev;
    const std::vector<double> expression_std_dev;

    static constexpr int num_identity_params = 199;
    static constexpr int num_albedo_params = 199;
    static constexpr int num_expression_params = 100;
};

struct GeometryRegularizationTerm {
    template <typename T>
    bool operator()(const T* const identity_params,
                    const T* const expression_params,
                    T* residual) const {
        T reg_energy = T(0);

        // Identity parameters regularization
        for (int i = 0; i < num_identity_params; ++i) {
            reg_energy += pow(identity_params[i] / T(identity_std_dev[i]), 2);
        }
        // Expression parameters regularization
        for (int i = 0; i < num_expression_params; ++i) {
            reg_energy += pow(expression_params[i] / T(expression_std_dev[i]), 2);
        }

        residual[0] = reg_energy;
        return true;
    }

    // Constructor to pass standard deviations if they're not constant
    GeometryRegularizationTerm(const std::vector<double>& id_std,
                               const std::vector<double>& exp_std)
            : identity_std_dev(id_std)
            , expression_std_dev(exp_std) {}

    const std::vector<double> identity_std_dev;
    const std::vector<double> expression_std_dev;

    static constexpr int num_identity_params = 199;
    static constexpr int num_expression_params = 100;
};

struct ColorRegularizationTerm {
    template <typename T>
    bool operator()(const T* const albedo_params, T* residual) const {
        T reg_energy = T(0);

        // Albedo parameters regularization
        for (int i = 0; i < num_albedo_params; ++i) {
            reg_energy += pow(albedo_params[i] / T(albedo_std_dev[i]), 2);
        }

        residual[0] = reg_energy;
        return true;
    }

    // Constructor to pass standard deviations if they're not constant
    ColorRegularizationTerm(const std::vector<double>& alb_std)
            : albedo_std_dev(alb_std) {}

    const std::vector<double> albedo_std_dev;
    static constexpr int num_albedo_params = 199;
};


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
