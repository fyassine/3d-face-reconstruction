#ifndef FACE_RECONSTRUCTION_OPTIMIZATION_H
#define FACE_RECONSTRUCTION_OPTIMIZATION_H
#include <ceres/ceres.h>
#include "Eigen.h"

//TODO: THIS SCRIPT IS SUBJECT TO CHANGE!!! Don't look!!! It's ugly!!!

struct GeometryOptimization{
public:
    GeometryOptimization(const Eigen::Vector3f& vertex, const float& depth, const Eigen::Vector3f& normal):
            m_vertex(vertex), m_depth(depth), m_normal(normal)
    {}

    template<typename T>
    bool operator()(const T* const shape, const T* const expression, T* residual) const {

        Eigen::Matrix<T, 3, 1> shape_offset = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> expression_offset = Eigen::Matrix<T, 3, 1>::Zero();
        auto transformedVertex = m_vertex.cast<T>() + shape_offset + expression_offset;

        //Rotation not yet included. Maybe better to take two params instead of 6 for shape, replacing the floats with vectors?!
        //because we ignore rotation for now, until we can test it
        shape_offset.x() += shape[0];
        shape_offset.y() += shape[1];
        shape_offset.z() += shape[2];

        for (int i = 0; i < num_expression_params; ++i) {
            expression_offset.x() += expression[i * 3];
            expression_offset.y() += expression[i * 3 + 1];
            expression_offset.z() += expression[i * 3 + 2];
        }
        
        T depth = T(m_depth);
        T point_to_point_residual = (transformedVertex.z() - depth) * (transformedVertex.z() - depth); //is this square necessary? -> ceres should take care of that?!
        Eigen::Matrix<T, 3, 1> normal_T = m_normal.cast<T>();

        T point_to_plane_residual = normal_T.dot(transformedVertex - Eigen::Matrix<T, 3, 1>(T(0), T(0), depth));

        residual[0] = point_to_point_residual + point_to_plane_residual;
        return true;
    }

private:
    const Eigen::Vector3f m_vertex;
    const float m_depth;
    const Eigen::Vector3f m_normal;

    static const int num_shape_params = 6; //0, 1, 2: translation; 3, 4, 5: rotation
    static const int num_expression_params = 50; //TODO: Change!!!
};

class Optimization {
public:
    static void optimizeDenseTerms();
    static void optimizeSparseTerms();

private:
    static void configureSolver(ceres::Solver::Options& options);
};


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
