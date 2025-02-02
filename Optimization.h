#ifndef FACE_RECONSTRUCTION_OPTIMIZATION_H
#define FACE_RECONSTRUCTION_OPTIMIZATION_H
#include <ceres/ceres.h>
#include "Eigen.h"
#include "BFMParameters.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>
#include "Rendering.h"

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

struct GeometryOptimization {
public:
    GeometryOptimization(Eigen::Vector3f vertex,
                         const float& depth,
                         Eigen::Vector3f normal,
                         const BfmProperties& bfmProperties,
                         int vertex_id) :
            m_vertex(std::move(vertex)), m_depth(depth), m_normal(std::move(normal)), m_bfm_properties(bfmProperties), m_vertex_id(vertex_id) {}

    template<typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 4, 1> shape_offset = Eigen::Matrix<T, 4, 1>::Zero();
        Eigen::Matrix<T, 4, 1> expression_offset = Eigen::Matrix<T, 4, 1>::Zero();
        shape_offset.w() = T(1.0);
        expression_offset.w() = T(1.0);

        auto& m_shapePcaBasis = m_bfm_properties.shapePcaBasis.cast<double>();
        auto& m_expressionBasis = m_bfm_properties.expressionPcaBasis.cast<double>();
        //int vertex_idx = m_vertex_id * 3;

        /*shape_offset = m_bfm_properties.shapePcaBasis.block<4, num_shape_params>(vertex_idx, 0)
                               .template cast<T>()
                       * Eigen::Map<const Eigen::Matrix<T, num_shape_params, 1>>(shape);
        expression_offset = m_bfm_properties.expressionPcaBasis.block<4, num_expression_params>(vertex_idx, 0)
                                    .template cast<T>()
                            * Eigen::Map<const Eigen::Matrix<T, num_expression_params, 1>>(expression);*/
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            shape_offset += Eigen::Matrix<T, 4, 1>(
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx, i))),
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 1, i))),
                    T(shape[i] * T(m_shapePcaBasis(vertex_idx + 2, i))),
                    T(0)
            );
        }

        for (int i = 0; i < num_expression_params; ++i) {
            int vertex_idx = m_vertex_id * 3;
            expression_offset += Eigen::Matrix<T, 4, 1>(
                    T(expression[i] * T(m_expressionBasis(vertex_idx, i))),
                    T(expression[i] * T(m_expressionBasis(vertex_idx + 1, i))),
                    T(expression[i] * T(m_expressionBasis(vertex_idx + 2, i))),
                    T(0)
            );
        }

        Eigen::Matrix<T, 4, 1> vertex4d = Eigen::Matrix<T, 4, 1>(T(m_vertex.x()), T(m_vertex.y()), T(m_vertex.z()), T(1));
        Eigen::Matrix<T, 4, 1> transformedVertex = (m_bfm_properties.transformation.cast<T>() * vertex4d) + shape_offset + expression_offset;


        auto point_to_point = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                  transformedVertex.y(),
                                                  transformedVertex.z() - T(m_depth));

        T point_to_plane = Eigen::Matrix<T, 3, 1>(transformedVertex.x(),
                                                              transformedVertex.y(),
                                                              transformedVertex.z() - T(m_depth)).dot(m_normal.cast<T>());

        residuals[0] = point_to_point.squaredNorm();
        residuals[1] = point_to_plane;

        return true;
    }

private:
    const Eigen::Vector3f m_vertex;
    const float m_depth;
    const Eigen::Vector3f m_normal;

    const BfmProperties& m_bfm_properties;

    static const int num_shape_params = 199;
    static const int num_expression_params = 100;

    const int m_vertex_id;
};

struct ColorOptimization {
public:
    ColorOptimization(const Eigen::Vector3f& albedo, const Eigen::Vector3f& image_color, const BfmProperties& bfmProperties, int color_id)
        : m_albedo(albedo), m_image_color(image_color), m_bfm_properties(bfmProperties), m_color_id(color_id) {}

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
        auto& m_colorPcaBasis = m_bfm_properties.colorPcaBasis;

        // Compute color offset
        Eigen::Matrix<T, 3, 1> color_offset = Eigen::Matrix<T, 3, 1>::Zero();
        for (int i = 0; i < num_color_params; ++i) {
            int color_idx = m_color_id * 3;
            color_offset += Eigen::Matrix<T, 3, 1>(
                    T(color[i] * T(m_colorPcaBasis(color_idx, i))),
                    T(color[i] * T(m_colorPcaBasis(color_idx + 1, i))),
                    T(color[i] * T(m_colorPcaBasis(color_idx + 2, i)))
            );
        }
        
        // Apply illumination scaling directly
        Eigen::Matrix<T, 3, 1> albedo = m_albedo.cast<T>() + color_offset;

        // Illumination is already computed and passed as an argument
        //Eigen::Matrix<T, 3, 1> illumination = m_illumination.cast<T>(); // Converted to T for optimization

        // Apply illumination to color (this step assumes illumination modifies the albedo in RGB channels)
        //Eigen::Matrix<T, 3, 1> modifiedColor = albedo.cwiseProduct(illumination);

        // Compute residuals (compare modified color to image color)
        T resulting_color = Eigen::Matrix<T, 3, 1>(albedo.x() - T(m_image_color.x()),
                                                   albedo.y() - T(m_image_color.y()),
                                                   albedo.z() - T(m_image_color.z())).norm();
        
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
    const BfmProperties& m_bfm_properties;
    const int m_color_id;
    static const int num_color_params = 199;
};

struct SparseOptimization{
public:
    SparseOptimization(const Eigen::Vector3d& landmark_position_input, const Eigen::Vector3d& landmark_bfm, const int landmark_bfm_index, const BfmProperties& bfmProperties, const int index)
            : m_landmark_positions_input(landmark_position_input), m_bfm_properties(bfmProperties), m_landmark_bfm(landmark_bfm), m_landmark_bfm_index(landmark_bfm_index), m_current_index(index) {}

    template <typename T>
    bool operator()(const T* const shape,
                    const T* const expression,
                    T* residuals) const {

        Eigen::Matrix<T, 4, 1> baseLandmark = Eigen::Matrix<T, 4, 1>(
                T(m_landmark_bfm.x()),
                T(m_landmark_bfm.y()),
                T(m_landmark_bfm.z()),
                T(1)
        );

        Eigen::Matrix<T, 4, 1> shapeOffset = Eigen::Matrix<T, 4, 1>::Zero();
        auto m_shapePcaBasis = m_bfm_properties.shapePcaBasis.cast<double>();
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            shapeOffset += Eigen::Matrix<T, 4, 1>(
                    shape[i] * T(m_shapePcaBasis(vertex_idx, i)),
                    shape[i] * T(m_shapePcaBasis(vertex_idx + 1, i)),
                    shape[i] * T(m_shapePcaBasis(vertex_idx + 2, i)),
                    T(0)
            );
        }

        Eigen::Matrix<T, 4, 1> expressionOffset = Eigen::Matrix<T, 4, 1>::Zero();
        auto m_expressionBasis = m_bfm_properties.expressionPcaBasis.cast<double>();
        for (int i = 0; i < num_expression_params; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            expressionOffset += Eigen::Matrix<T, 4, 1>(
                    expression[i] * T(m_expressionBasis(vertex_idx, i)),
                    expression[i] * T(m_expressionBasis(vertex_idx + 1, i)),
                    expression[i] * T(m_expressionBasis(vertex_idx + 2, i)),
                    T(0)
            );
        }

        //TODO: to get good looking result, add offset after transformation?!
        // Apply modifications to base position before transformation
        Eigen::Matrix<T, 4, 1> modifiedVertex = baseLandmark; //TODO: eigentlich müsste hier das offset geaddet werden oder nicht?!

        // Apply transformation
        Eigen::Matrix<T, 4, 1> transformedVertex = m_bfm_properties.transformation.cast<T>() * modifiedVertex + shapeOffset + expressionOffset;

        /*if(m_current_index == 0){
            std::cout << "BFM Landmark: " << baseLandmark << std::endl;
            std::cout << "Shape Offset: " << shapeOffset << std::endl;
            std::cout << "Expression Offset: " << expressionOffset << std::endl;
            std::cout << "Modified Vertex: " << modifiedVertex << std::endl;
            std::cout << "Transformed Vertex: " << transformedVertex << std::endl;
            std::cout << "Image Landmark: " << m_landmark_positions_input << std::endl;
        }*/

        // Calculate residuals
        residuals[0] = transformedVertex.x() - T(m_landmark_positions_input.x());
        residuals[1] = transformedVertex.y() - T(m_landmark_positions_input.y());
        residuals[2] = transformedVertex.z() - T(m_landmark_positions_input.z());

        return true;
    }

    void evaluateWithDoubles(const double* shape, const double* expression) const {
        Eigen::Vector4f baseLandmark(
                m_landmark_bfm.x(),
                m_landmark_bfm.y(),
                m_landmark_bfm.z(),
                1.0
        );

        Eigen::Vector4f shapeOffset = Eigen::Vector4f::Zero();
        for (int i = 0; i < num_shape_params; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            shapeOffset += Eigen::Vector4f(
                    shape[i] * m_bfm_properties.shapePcaBasis(vertex_idx, i),
                    shape[i] * m_bfm_properties.shapePcaBasis(vertex_idx + 1, i),
                    shape[i] * m_bfm_properties.shapePcaBasis(vertex_idx + 2, i),
                    0.0
            );
        }

        Eigen::Vector4f expressionOffset = Eigen::Vector4f::Zero();
        for (int i = 0; i < num_expression_params; ++i) {
            int vertex_idx = m_landmark_bfm_index * 3;
            expressionOffset += Eigen::Vector4f(
                    expression[i] * m_bfm_properties.expressionPcaBasis(vertex_idx, i),
                    expression[i] * m_bfm_properties.expressionPcaBasis(vertex_idx + 1, i),
                    expression[i] * m_bfm_properties.expressionPcaBasis(vertex_idx + 2, i),
                    0.0
            );
        }

        // Print intermediate values
        std::cout << "\nDebug Values for landmark " << m_current_index << ":\n";
        std::cout << "Base Landmark:\n" << baseLandmark << "\n";
        std::cout << "Shape Offset:\n" << shapeOffset << "\n"; //Das ist 0 weil shape und expression offset bei 0 beginnen
        std::cout << "Expression Offset:\n" << expressionOffset << "\n"; //Das ist 0 weil shape und expression offset bei 0 beginnen

        // Try both ways and compare
        std::cout << "\nMethod 1 (offset before transform):\n";
        Eigen::Vector4f modifiedVertex1 = baseLandmark + shapeOffset + expressionOffset;
        Eigen::Vector4f transformedVertex1 = m_bfm_properties.transformation * modifiedVertex1;
        std::cout << "Modified Vertex:\n" << modifiedVertex1 << "\n";
        std::cout << "Transformed Vertex:\n" << transformedVertex1 << "\n";

        std::cout << "\nMethod 2 (offset after transform):\n";
        Eigen::Vector4f modifiedVertex2 = baseLandmark;
        Eigen::Vector4f transformedVertex2 = m_bfm_properties.transformation * modifiedVertex2 + shapeOffset + expressionOffset;
        std::cout << "Modified Vertex:\n" << modifiedVertex2 << "\n";
        std::cout << "Transformed Vertex:\n" << transformedVertex2 << "\n";

        std::cout << "\nTarget Position:\n" << m_landmark_positions_input << "\n";

        // Also print some stats about the transformation matrix
        std::cout << "\nTransformation Matrix:\n" << m_bfm_properties.transformation << "\n";
        std::cout << "Transformation determinant: " << m_bfm_properties.transformation.determinant() << "\n";
    }

private:
    const Eigen::Vector3d m_landmark_positions_input;
    const Eigen::Vector3d m_landmark_bfm;
    static const int num_shape_params = 199;
    static const int num_expression_params = 100;
    const BfmProperties m_bfm_properties;
    const int m_landmark_bfm_index;
    const int m_current_index;
};

class Optimization {
public:
    static void optimize(BfmProperties&, InputImage&);
private:
    static void configureSolver(ceres::Solver::Options&);
    static void optimizeSparseTerms(BfmProperties&, InputImage&, Eigen::VectorXd& shapeParams, Eigen::VectorXd& expressionParams);
    static void optimizeDenseTerms(BfmProperties&, InputImage&, ceres::Problem&);
    static void optimizeColor();
    static void regularize(BfmProperties&, ceres::Problem&);
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
        T reg_energy_geometry = T(0);
        T reg_energy_expression = T(0);

        // Identity parameters regularization
        for (int i = 0; i < num_identity_params; ++i) {
            reg_energy_geometry += pow(identity_params[i] / T(identity_std_dev[i]), 2);
            //residual[i] = identity_params[i] * T(sqrt(10));
        }
        // Expression parameters regularization
        for (int i = 0; i < num_expression_params; ++i) {
            reg_energy_expression += pow(expression_params[i] / T(expression_std_dev[i]), 2);
            //residual[num_identity_params + i] = expression_params[i] * T(sqrt(8));
        }

        residual[0] = reg_energy_geometry * T(sqrt(100000));
        residual[1] = reg_energy_expression * T(sqrt(1));
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
