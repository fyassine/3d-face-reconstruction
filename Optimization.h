//
// Created by Leo on 21.12.2024.
//


#ifndef FACE_RECONSTRUCTION_OPTIMIZATION_H
#define FACE_RECONSTRUCTION_OPTIMIZATION_H
#include <ceres/ceres.h>
#include "Eigen.h"

struct GeometryOptimization{
public:
    GeometryOptimization(const Eigen::Vector3f& vertex, const Eigen::Vector3f& depth, const Eigen::Vector3f& normal):
            m_vertex{vertex},
            m_depth{depth},
            m_normal{normal}
    {}

    template<typename T>
    bool operator()(const T* const P, T* residuals) const {
        residuals[0] = 0;
        return true;
    }
private:
    Eigen::Vector3f m_vertex;
    Eigen::Vector3f m_depth;
    Eigen::Vector3f m_normal;
};

class Optimization {

public:


    void optimizeDenseTerms(){
        ceres::Problem problem;
        ceres::Solver::Options options;
        configureSolver(options);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        //ceres::Solver::options options;
        //problem.AddResidualBlock(GeometryOptimization)
    }
private:

    void configureSolver(ceres::Solver::Options& options) {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 8;
    }

};


#endif //FACE_RECONSTRUCTION_OPTIMIZATION_H
