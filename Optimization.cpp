#include "Optimization.h"
#include <vector>
#include <iostream>


//Not sure about params and return type yet
void Optimization::optimizeDenseTerms() {
    ceres::Problem problem;

    // Mock data
    std::vector<Eigen::Vector3d> vertices; // Fill with actual data
    std::vector<Eigen::Vector3d> rgbData;
    std::vector<double> depths;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> shapeParams(6, 0.0f);        // 6 elements, initialized to 0.0
    std::vector<double> expressionParams(150, 0.0f);  // prob not floats, but okay for testing purposes
    std::vector<double> colorParams(3, 0.0f);
    //End Mock data

    for (size_t i = 0; i < vertices.size(); ++i) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 1, 6, 50>(
                        new GeometryOptimization(vertices[i], depths[i], normals[i])),
                nullptr,
                shapeParams.data(),
                expressionParams.data()
                );
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ColorOptimization, 1, 3>(
                new ColorOptimization(rgbData[i])),
                nullptr,
                colorParams.data()
                );
    }

    ceres::Solver::Options options;
    configureSolver(options);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
}

void Optimization::optimizeSparseTerms() {
    ceres::Problem problem;
}

void Optimization::configureSolver(ceres::Solver::Options &options) {
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = 1;
    options.max_num_iterations = 1; //maybe make it 100
    options.num_threads = 8;
}


