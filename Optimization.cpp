#include "Optimization.h"
#include <vector>
#include <iostream>

double GetDepthForVertex(Eigen::Vector3d vertex){
    return 0;
}

//Not sure about params and return type yet
void Optimization::optimizeDenseTerms(BfmProperties& properties, InputImage& inputImage) {
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

    auto bfmVertices = getVertices(properties);
    for (size_t i = 0; i < bfmVertices.size(); ++i) {
        Eigen::Vector3f vertexBfm = bfmVertices[i];
        float depthInputImage = getDepthValueFromInputImage()
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<GeometryOptimization, 2, 199, 100>(
                        new GeometryOptimization(bfmVertices[i], 10.0f, bfmVertices[i])),
                nullptr,
                properties.shapeWeight,
                properties.expressionWeight
                );
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ColorOptimization, 1, 3>(
                new ColorOptimization(rgbData[i], rgbData[i])), // replace second rgbData[i] with illumination
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


