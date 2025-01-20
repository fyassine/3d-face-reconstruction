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
                         const Eigen::Vector3f& normal):
            m_vertex(vertex), m_depth(depth), m_normal(normal)
    {}

    template<typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();

        for (int i = 0; i < num_shape_params; ++i) {
            shape_offset.x() += shape[i * 3];
            shape_offset.y() += shape[i * 3 + 1];
            shape_offset.z() += shape[i * 3 + 2];
        }

        for (int i = 0; i < num_expression_params; ++i) {
            expression_offset.x() += expression[i * 3];
            expression_offset.y() += expression[i * 3 + 1];
            expression_offset.z() += expression[i * 3 + 2];
        }
                
        Eigen::Matrix<T, 3, 1> transformedVertex = m_vertex.cast<T>() + shape_offset + expression_offset;
        T point_to_point = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).norm();
        T point_to_plane = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth)).dot(m_normal.cast<T>());

        // NOTE: Point-to-point & point-to-plane shouldn't be combined:
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


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
