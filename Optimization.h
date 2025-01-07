#ifndef FACE_RECONSTRUCTION_OPTIMIZATION_H
#define FACE_RECONSTRUCTION_OPTIMIZATION_H
#include <ceres/ceres.h>
#include "Eigen.h"

//TODO: THIS SCRIPT IS SUBJECT TO CHANGE!!! Don't look!!! It's ugly!!!

struct GeometryOptimization{
public:
    GeometryOptimization(const Eigen::Vector3d& vertex, const float& depth, const Eigen::Vector3d& normal):
            m_vertex(vertex), m_depth(depth), m_normal(normal)
    {}

    template<typename T>
    bool operator()(const T* const shape, const T* const expression, T* residual) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();

        for (int i = 0; i < num_shape_params; ++i) {

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

        residual[0] = point_to_point + point_to_plane;
        return true;
    }

private:
    const Eigen::Vector3d m_vertex;
    const float m_depth;
    const Eigen::Vector3d m_normal;

    static const int num_shape_params = 6; //0, 1, 2: translation; 3, 4, 5: rotation
    static const int num_expression_params = 50; //TODO: Change!!!
};

struct ColorOptimization{
public:
    //datatype illumination
    ColorOptimization(const Eigen::Vector3d& albedo):
            m_albedo(albedo)
    {}

    template<typename T>
    bool operator()(const T* const color, T* residual) const {
        Eigen::Matrix<T, 3, 1> color_offset = Eigen::Matrix<T, 3, 1>::Zero();
        color_offset.x() = color[0];
        color_offset.y() = color[1];
        color_offset.z() = color[2];
        residual[0] = (m_albedo.cast<T>() - color_offset).norm();
        return true;
    }
    // TODO - convert h5 into usable parameters -> fix the library
    // TODO - finish color residual block

private:
    const Eigen::Vector3d m_albedo;
    //TODO: Illumination
};


class Optimization {
public:
    static void optimizeDenseTerms();
    static void optimizeSparseTerms();

private:
    static void configureSolver(ceres::Solver::Options& options);
};


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
