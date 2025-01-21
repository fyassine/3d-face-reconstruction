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

struct GeometryOptimization{
public:
    GeometryOptimization(const Eigen::Vector3f& vertex,
                         const float& depth,
                         const Eigen::Vector3f& normal,
                         const Eigen::MatrixXf& shapePcaBasis,
                         int vertex_id) :
            m_vertex(vertex), m_depth(depth), m_normal(normal),
            m_shapePcaBasis(shapePcaBasis), m_vertex_id(vertex_id) {}

    template<typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();

        // Each parameter influences a single vertex coordinate
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            shape_offset += Eigen::Matrix<T, 3, 1>(
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx, i))),
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 1, i))),
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 2, i)))
            );
        }

        for (int i = 0; i < num_expression_params; ++i) {
            expression_offset += Eigen::Matrix<T, 3, 1>(
                    T(expression[i]),
                    T(expression[i]),
                    T(expression[i])
            );
        }
        Eigen::Matrix<T, 3, 1> transformedVertex = m_vertex.cast<T>() + shape_offset + expression_offset;

        transformedVertex = m_vertex.cast<T>(); //TODO: Remove the line and think of a solution for the pcaBasis

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

    static const int num_shape_params = 199;
    static const int num_expression_params = 100;

    const Eigen::MatrixXf& m_shapePcaBasis;
    const int m_vertex_id;
};

struct ColorOptimization {
public:
    ColorOptimization(const Eigen::Vector3d& albedo, const Eigen::Vector3d& illumination)
        : m_albedo(albedo), m_illumination(illumination) {}

    template <typename T>
    bool operator()(const T* const color, T* residuals) const {
        // Normalize illumination
        T illumination_normalized[3];
        T illum_sum = m_illumination.cast<T>().sum();
        
        if (illum_sum == T(0)) {
            return false;
        }

        illumination_normalized[0] = T(m_illumination.x()) / illum_sum;
        illumination_normalized[1] = T(m_illumination.y()) / illum_sum;
        illumination_normalized[2] = T(m_illumination.z()) / illum_sum;

        // Compute adjusted albedo with normalized illumination
        Eigen::Matrix<T, 3, 1> adjusted_albedo = m_albedo.cast<T>().cwiseProduct(
        Eigen::Matrix<T, 3, 1>(illumination_normalized[0], illumination_normalized[1], illumination_normalized[2]));

        // Compute color offset
        Eigen::Matrix<T, 3, 1> color_offset(color[0], color[1], color[2]);

        // Compute per-channel residuals
        residuals[0] = adjusted_albedo.x() - color_offset.x();
        residuals[1] = adjusted_albedo.y() - color_offset.y();
        residuals[2] = adjusted_albedo.z() - color_offset.z();

        return true;
    }

private:
    const Eigen::Vector3d m_albedo;
    const Eigen::Vector3d m_illumination;
};

class Optimization {
public:
    static void optimizeDenseTerms(BfmProperties&, InputImage&);
    static void optimizeSparseTerms();

private:
    static void configureSolver(ceres::Solver::Options& options);
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


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
